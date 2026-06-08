# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RoboTwin reward model with T5 text encoder and cross-attention fusion.

This module implements a text-conditioned reward model that:
  - Extracts visual features from images with a ResNet18 backbone.
  - Encodes task descriptions with a frozen T5 encoder.
  - Fuses the two modalities via cross-attention (visual queries, text
    keys/values).
  - Predicts a binary scalar reward via a lightweight MLP head.

Architecture overview::

    Image (B, C, H, W)
        └─ ResNet18 (layers 1-4)
               └─ (B, 512, 7, 7) → reshape → visual tokens (B, N_v, 512)
                                                           │
    Instruction (list[str])                                │
        └─ T5Tokenizer + T5EncoderModel (frozen)           │
               └─ (B, seq_len, d_model)                    │
                       └─ Linear projection                │
                              └─ text tokens (B, N_t, 512) ┘
                                                ▼
                                   MultiheadAttention
                              (query=visual, key/value=text)
                                                ▼
                              (B, N_v, 512) → mean-pool → (B, 512)
                                                ▼
                                        MLP → sigmoid → reward (B,)
"""

import contextlib
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from omegaconf import DictConfig

from rlinf.models.embodiment.reward.base_image_reward_model import BaseImageRewardModel


class RoboTwinT5CrossAttnRewardModel(BaseImageRewardModel):
    """Text-conditioned reward model using T5 encoder and cross-attention.

    Args:
        cfg: DictConfig with the following optional fields:
            - t5_model_name (str): HuggingFace model id or local path for T5
              encoder.  Defaults to ``"t5-base"``.
            - freeze_t5 (bool): Whether to freeze T5 weights during training.
              Defaults to ``True``.
            - num_attn_heads (int): Number of attention heads in cross-attn.
              Defaults to ``8``.
            - attn_dropout (float): Dropout probability in attention.
              Defaults to ``0.0``.
            - hidden_dim (int): MLP hidden dimension.  Defaults to ``256``.
            - head_dropout (float): Dropout probability in the MLP head.
              Defaults to ``0.1``.
            - max_text_length (int): Maximum token length for text inputs.
              Defaults to ``64``.
            - image_size (list): Expected image size ``[C, H, W]``.
              Defaults to ``[3, 224, 224]``.
            - normalize (bool): Apply ImageNet normalisation.  Defaults to
              ``True``.
            - model_path (str | None): Optional checkpoint path to load.
    """

    _VISUAL_DIM = 512  # ResNet18 layer4 output channels

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.t5_model_name: str = cfg.get("t5_model_name", "t5-base")
        self.freeze_t5: bool = cfg.get("freeze_t5", True)
        self.num_attn_heads: int = cfg.get("num_attn_heads", 8)
        self.attn_dropout: float = cfg.get("attn_dropout", 0.0)
        self.hidden_dim: int = cfg.get("hidden_dim", 256)
        self.head_dropout: float = cfg.get("head_dropout", 0.1)
        self.max_text_length: int = cfg.get("max_text_length", 64)

        self._build_visual_encoder()
        self._build_text_encoder()
        self._build_cross_attn_head()
        self._load_model()

    # ------------------------------------------------------------------
    # Builder helpers
    # ------------------------------------------------------------------

    def _build_visual_encoder(self) -> None:
        """Build ResNet18 feature extractor (up to layer4, no pooling/fc)."""
        backbone = tv_models.resnet18(weights="IMAGENET1K_V1")
        self.visual_encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        # Output shape for 224x224 input: (B, 512, 7, 7)
        # → spatial tokens: (B, 49, 512)

    def _build_text_encoder(self) -> None:
        """Build T5 tokeniser and encoder (optionally frozen)."""
        from transformers import AutoTokenizer, T5EncoderModel

        self.t5_tokenizer = AutoTokenizer.from_pretrained(self.t5_model_name)
        self.t5_encoder = T5EncoderModel.from_pretrained(self.t5_model_name)

        if self.freeze_t5:
            for param in self.t5_encoder.parameters():
                param.requires_grad = False

        # Project T5 hidden states → visual dim so both spaces match.
        t5_hidden_size: int = self.t5_encoder.config.d_model
        self.text_proj = nn.Linear(t5_hidden_size, self._VISUAL_DIM)

    def _build_cross_attn_head(self) -> None:
        """Build cross-attention layer and reward head MLP."""
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self._VISUAL_DIM,
            num_heads=self.num_attn_heads,
            dropout=self.attn_dropout,
            batch_first=True,
        )
        self.ln_attn = nn.LayerNorm(self._VISUAL_DIM)
        self.reward_head = nn.Sequential(
            nn.Linear(self._VISUAL_DIM, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.head_dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def _load_model(self) -> None:
        """Load checkpoint weights if ``cfg.model_path`` is set."""
        model_path = self.cfg.get("model_path", None)
        if model_path is None:
            return

        if model_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(model_path)
        else:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

        # Strip common prefixes added by FSDP / DDP.
        cleaned: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            for prefix in ("module.", "_orig_mod.", "model."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
            cleaned[k] = v

        self.load_state_dict(cleaned, strict=True)

    # ------------------------------------------------------------------
    # Internal forward utilities
    # ------------------------------------------------------------------

    def _encode_visual(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into spatial visual tokens.

        Args:
            images: Raw image tensor ``(B, C, H, W)`` or ``(B, H, W, C)``.
                Byte or float; range [0, 255] or [0, 1].

        Returns:
            Visual tokens of shape ``(B, N_v, 512)``.
        """
        images = self.preprocess_images(images)
        feat = self.visual_encoder(images)  # (B, 512, H', W')
        B, C, Hf, Wf = feat.shape
        # (B, 512, H', W') → (B, H'*W', 512)
        return feat.permute(0, 2, 3, 1).reshape(B, Hf * Wf, C)

    def _encode_text(
        self, instructions: list[str], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text instructions via T5 and project to visual dim.

        Args:
            instructions: List of ``B`` instruction strings.
            device: Target device for tokenizer outputs.

        Returns:
            Tuple of:
                - text_tokens: ``(B, seq_len, 512)``
                - attention_mask: ``(B, seq_len)``  (1 = real token, 0 = pad)
        """
        encoding = self.t5_tokenizer(
            instructions,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        ctx = torch.no_grad() if self.freeze_t5 else contextlib.nullcontext()
        with ctx:
            text_out = self.t5_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
        # (B, seq_len, d_model) → (B, seq_len, 512)
        text_tokens = self.text_proj(text_out.last_hidden_state)
        return text_tokens, attention_mask

    def _fuse(
        self,
        visual_tokens: torch.Tensor,
        instructions: Optional[list[str]],
    ) -> torch.Tensor:
        """Fuse visual tokens with text tokens via cross-attention.

        If ``instructions`` is ``None`` the visual tokens are pooled directly
        (useful for ablation / inference without text).

        Args:
            visual_tokens: ``(B, N_v, 512)``
            instructions: List of ``B`` instruction strings or ``None``.

        Returns:
            Pooled feature vector ``(B, 512)``.
        """
        if instructions is not None:
            text_tokens, attention_mask = self._encode_text(
                instructions, device=visual_tokens.device
            )
            # key_padding_mask: True means *ignore* that position.
            key_padding_mask = attention_mask == 0  # (B, seq_len)
            attn_out, _ = self.cross_attn(
                query=visual_tokens,
                key=text_tokens,
                value=text_tokens,
                key_padding_mask=key_padding_mask,
            )
            visual_tokens = self.ln_attn(visual_tokens + attn_out)

        # Mean-pool spatial tokens → (B, 512)
        return visual_tokens.mean(dim=1)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        input_data: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        instructions: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Training forward pass.

        Args:
            input_data: Image tensor ``(B, C, H, W)``.
            labels: Binary labels ``(B,)`` – 1 for success, 0 for fail.
                When ``None`` only probabilities / logits are returned.
            instructions: Task-description strings, length ``B``.  When
                ``None`` cross-attention is skipped.

        Returns:
            Dict with keys:
                - ``"loss"``: BCE loss (zero tensor when ``labels`` is ``None``).
                - ``"accuracy"``: Fraction of correct predictions.
                - ``"logits"``: Raw logits ``(B,)``.
                - ``"probabilities"``: Sigmoid probabilities ``(B,)``.
        """
        visual_tokens = self._encode_visual(input_data)
        pooled = self._fuse(visual_tokens, instructions)
        logits = self.reward_head(pooled).squeeze(-1)  # (B,)
        probabilities = torch.sigmoid(logits)

        if labels is not None:
            labels_f = labels.float().to(logits.device)
            loss = F.binary_cross_entropy_with_logits(logits, labels_f)
            predictions = (probabilities > 0.5).float()
            accuracy = (predictions == labels_f).float().mean()
        else:
            loss = torch.tensor(0.0, device=logits.device)
            accuracy = torch.tensor(0.0, device=logits.device)

        return {
            "loss": loss,
            "accuracy": accuracy,
            "logits": logits,
            "probabilities": probabilities,
        }

    def compute_reward(
        self,
        images: torch.Tensor,
        task_descriptions: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Inference-time reward computation (no gradient).

        Args:
            images: Image tensor ``(B, C, H, W)`` or ``(B, H, W, C)``.
                Automatically moved to the model's device if necessary.
            task_descriptions: Optional list of ``B`` instruction strings.

        Returns:
            Reward probabilities of shape ``(B,)``.
        """
        # Ensure input is on the same device as model parameters, regardless
        # of whether the caller (e.g. EnvWorker) is on CPU or GPU.
        model_device = next(self.parameters()).device
        images = images.to(model_device)
        with torch.no_grad():
            visual_tokens = self._encode_visual(images)
            pooled = self._fuse(visual_tokens, task_descriptions)
            logits = self.reward_head(pooled).squeeze(-1)
            return torch.sigmoid(logits)

    @torch.no_grad()
    def predict_rew(
        self,
        obs: torch.Tensor,
        instructions: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """WanEnv兼容的reward预测接口。
        
        这个方法与WanEnv的调用方式完全兼容:
        - WanEnv传入: obs [B, 3, H, W] (值域可能是[-1, 1]或[0, 1])
        - WanEnv期望返回: rewards [B]
        
        Args:
            obs: 图像观测 tensor ``(B, 3, H, W)``。
                 值域可能是 [-1, 1] (来自WanEnv的current_obs)
                 或 [0, 1] (标准的归一化图像)。
            instructions: 任务描述字符串列表，长度 ``B``。
        
        Returns:
            rewards: 奖励概率 tensor ``(B,)``，范围 [0, 1]。
        """
        # 1. 处理输入值域
        # WanEnv的current_obs是[-1, 1]，需要转换到[0, 1]
        if obs.min() < 0:
            # 从[-1, 1]转换到[0, 1]
            obs = (obs + 1.0) / 2.0
        
        # 2. 使用BaseImageRewardModel的预处理 (resize + ImageNet normalize)
        obs = self.preprocess_images(obs)
        
        # 3. 调用compute_reward (内部会处理device转移和no_grad)
        return self.compute_reward(obs, task_descriptions=instructions)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        config: Optional[dict] = None,
    ) -> "RoboTwinT5CrossAttnRewardModel":
        """从checkpoint加载预训练模型。
        
        Args:
            checkpoint_path: .pth文件路径
            config: 可选的配置dict，如果不提供则使用默认配置
        
        Returns:
            加载了权重的RoboTwinT5CrossAttnRewardModel实例
        """
        from omegaconf import DictConfig
        
        # 默认配置 (与训练时一致)
        default_config = {
            "model_type": "robotwin_t5_crossattn",
            "t5_model_name": "t5-base",
            "freeze_t5": True,
            "max_text_length": 64,
            "num_attn_heads": 8,
            "attn_dropout": 0.0,
            "hidden_dim": 256,
            "head_dropout": 0.1,
            "image_size": [3, 224, 224],
            "normalize": True,
            "precision": "fp32",
            "add_value_head": False,
            "is_lora": False,
            "freeze_vit": False,
            "use_flash_attention": False,
        }
        
        # 合并用户配置
        if config is not None:
            default_config.update(config)
        
        cfg = DictConfig(default_config)
        
        # 创建模型实例
        model = cls(cfg)
        
        # 加载checkpoint
        print(f"Loading reward model checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # checkpoint是直接的state_dict格式
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        # 加载权重
        model.load_state_dict(state_dict, strict=True)
        print(f"✓ Reward model loaded successfully")
        
        return model
