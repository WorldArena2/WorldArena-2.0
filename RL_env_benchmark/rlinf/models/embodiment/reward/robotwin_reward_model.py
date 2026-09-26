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

"""RoboTwin reward model with T5 text conditioning."""

from __future__ import annotations

import contextlib
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from omegaconf import DictConfig

from rlinf.models.embodiment.reward.base_image_reward_model import BaseImageRewardModel


class RoboTwinT5CrossAttnRewardModel(BaseImageRewardModel):
    """Text-conditioned image reward model for RoboTwin world-model rollouts."""

    _VISUAL_DIM = 512

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

    def _build_visual_encoder(self) -> None:
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

    def _build_text_encoder(self) -> None:
        from transformers import AutoTokenizer, T5EncoderModel

        self.t5_tokenizer = AutoTokenizer.from_pretrained(self.t5_model_name)
        self.t5_encoder = T5EncoderModel.from_pretrained(self.t5_model_name)

        if self.freeze_t5:
            for param in self.t5_encoder.parameters():
                param.requires_grad = False

        self.text_proj = nn.Linear(self.t5_encoder.config.d_model, self._VISUAL_DIM)

    def _build_cross_attn_head(self) -> None:
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
        model_path = self.cfg.get("model_path", None)
        if model_path is None:
            return

        if str(model_path).endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(model_path)
        else:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

        cleaned: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            for prefix in ("module.", "_orig_mod.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            cleaned[key] = value

        self.load_state_dict(cleaned, strict=True)

    def _encode_visual(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess_images(images)
        feat = self.visual_encoder(images)
        batch, channels, height, width = feat.shape
        return feat.permute(0, 2, 3, 1).reshape(batch, height * width, channels)

    def _encode_text(
        self, instructions: list[str], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return self.text_proj(text_out.last_hidden_state), attention_mask

    def _fuse(
        self,
        visual_tokens: torch.Tensor,
        instructions: Optional[list[str]],
    ) -> torch.Tensor:
        if instructions is not None:
            text_tokens, attention_mask = self._encode_text(
                instructions, device=visual_tokens.device
            )
            key_padding_mask = attention_mask == 0
            attn_out, _ = self.cross_attn(
                query=visual_tokens,
                key=text_tokens,
                value=text_tokens,
                key_padding_mask=key_padding_mask,
            )
            visual_tokens = self.ln_attn(visual_tokens + attn_out)

        return visual_tokens.mean(dim=1)

    def forward(
        self,
        input_data: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        instructions: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        visual_tokens = self._encode_visual(input_data)
        pooled = self._fuse(visual_tokens, instructions)
        logits = self.reward_head(pooled).squeeze(-1)
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

    @torch.no_grad()
    def compute_reward(
        self,
        images: torch.Tensor,
        task_descriptions: Optional[list[str]] = None,
    ) -> torch.Tensor:
        model_device = next(self.parameters()).device
        images = images.to(model_device)
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
        if obs.min() < 0:
            obs = (obs + 1.0) / 2.0
        return self.compute_reward(obs, task_descriptions=instructions)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        config: Optional[dict] = None,
    ) -> "RoboTwinT5CrossAttnRewardModel":
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
        if config is not None:
            default_config.update(config)

        model = cls(DictConfig(default_config))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=True)
        return model
