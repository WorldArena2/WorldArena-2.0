import argparse
import importlib
import json
import math
import os
import random
import copy
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from data.tactile_dataset import TactileHDF5Dataset, worker_init_fn
from wan.configs import WAN_CONFIGS
from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae2_2 import Wan2_2_VAE


@dataclass
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    distributed: bool


class PlaceholderTactileVAE(nn.Module):
    """Reserved tactile-VAE interface.

    Current default mode returns shape-correct placeholder latent so the full
    visual+tactile training pipeline can run before tactile VAE is ready.
    """

    def __init__(self, mode: str = "zeros"):
        super().__init__()
        if mode not in {"zeros", "random"}:
            raise ValueError(f"Unsupported placeholder mode: {mode}")
        self.mode = mode

    @torch.no_grad()
    def encode(
        self, tactile: torch.Tensor, target_shape: Sequence[int]
    ) -> torch.Tensor:
        """Args:
        tactile: [B, C, V_tac, T, H, W]
        target_shape: (C_lat, F_lat, H_lat, W_lat)
        Returns:
        latent: [B, C_lat, F_lat, H_lat, W_lat]
        """
        b = tactile.shape[0]
        c_lat, f_lat, h_lat, w_lat = [int(x) for x in target_shape]
        if self.mode == "zeros":
            return tactile.new_zeros((b, c_lat, f_lat, h_lat, w_lat))
        return torch.randn(
            b,
            c_lat,
            f_lat,
            h_lat,
            w_lat,
            dtype=tactile.dtype,
            device=tactile.device,
        )


class Trainer:
    def __init__(self, args: argparse.Namespace, dist_info: DistInfo):
        self.args = args
        self.dist = dist_info
        self.loss_history: List[Dict[str, float]] = []

        self.device = torch.device(f"cuda:{self.dist.local_rank}")
        torch.cuda.set_device(self.device)

        self._set_seed(args.seed + self.dist.rank)
        self._ensure_dir(args.output_dir)

        cfg = WAN_CONFIGS[args.config]
        self.num_train_timesteps = int(cfg.num_train_timesteps)

        self.text_encoder = self._build_text_encoder(cfg)
        self.vae = self._build_visual_vae(cfg)
        # Use the visual Wan VAE for tactile video encoding/decoding
        self.tactile_vae = self.vae

        self.dataset = TactileHDF5Dataset(
            data_roots=[args.data_root],
            valid_cam=["head,wrist"],
            task_names=args.task_names,
            samples_per_episode=args.samples_per_episode,
            sample_size=(args.sample_h, args.sample_w),
            sample_n_frames=args.sample_n_frames,
            chunk=args.chunk,
            action_chunk=args.action_chunk,
            dataset_info_cache_path=args.dataset_info_cache_path,
            preload_to_memory=args.preload_to_memory,
            use_unified_prompt=args.use_unified_prompt,
            unified_prompt=args.unified_prompt,
            max_episodes_per_task=args.max_episodes_per_task,
            max_total_episodes=args.max_total_episodes,
            fast_index=args.fast_index,
        )

        self.action_modalities_dims = None
        if getattr(args, "enable_action_expert", False):
            sample_item = self.dataset[0]
            self.action_modalities_dims = {
                "action": int(sample_item["actions"].shape[-1]),
                "state": int(sample_item["state"].shape[-1]),
                "virtual_force": int(sample_item["virtual_force"].shape[-1]),
            }
            args.action_in_channels = sum(self.action_modalities_dims.values())
            args.action_out_channels = sum(self.action_modalities_dims.values())

        latent_dim = getattr(getattr(self.vae, "model", None), "z_dim", None)
        self.model = self._build_multiview_model(cfg, latent_dim=latent_dim)
        if not args.skip_pretrained:
            self._load_pretrained_weights(
                source=args.pretrained_source,
                checkpoint_path=args.pretrained_checkpoint_path,
                checkpoint_dir=args.checkpoint_dir,
            )

        if getattr(self.args, "enable_action_expert", False):
            for name, param in self.model.named_parameters():
                param.requires_grad = name.startswith("action_")

        self.model.train()
        if not getattr(self.args, "enable_action_expert", False):
            self.model.requires_grad_(True)
        self.model.to(self.device)

        model_params = [p for p in self.model.parameters() if p.requires_grad]

        self.use_deepspeed = bool(args.use_deepspeed)
        self.ds_engine = None

        if self.use_deepspeed:
            deepspeed = importlib.import_module("deepspeed")

            ds_cfg = self._load_deepspeed_config(args.deepspeed_config)
            zero_cfg = ds_cfg.setdefault("zero_optimization", {})
            if int(zero_cfg.get("stage", 0)) == 3:
                zero_cfg["stage3_gather_16bit_weights_on_model_save"] = True
            ds_cfg["train_micro_batch_size_per_gpu"] = args.batch_size
            ds_cfg["gradient_accumulation_steps"] = args.grad_accum_steps
            ds_cfg.setdefault("optimizer", {})
            ds_cfg["optimizer"]["type"] = "AdamW"
            ds_cfg["optimizer"]["params"] = {
                "lr": args.lr,
                "betas": [args.beta1, args.beta2],
                "eps": args.adam_eps,
                "weight_decay": args.weight_decay,
            }
            self.ds_engine, self.optimizer, _, _ = deepspeed.initialize(
                model=self.model,
                model_parameters=model_params,
                config=ds_cfg,
            )
            self.model = self.ds_engine
        else:
            self.optimizer = AdamW(
                model_params,
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
            )
            if self.dist.distributed:
                self.model = DDP(
                    self.model,
                    device_ids=[self.dist.local_rank],
                    output_device=self.dist.local_rank,
                    find_unused_parameters=args.find_unused_parameters,
                )

        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(not self.use_deepspeed and args.use_fp16)
        )
        self.global_step = 0

        self.sampler = None
        if self.dist.distributed:
            self.sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.dist.world_size,
                rank=self.dist.rank,
                shuffle=True,
                drop_last=True,
            )

        self.loader = DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=(self.sampler is None),
            sampler=self.sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn,
            persistent_workers=(args.num_workers > 0),
        )

    def _base_model(self) -> WanModel:
        if isinstance(self.model, DDP):
            return self.model.module
        if hasattr(self.model, "module"):
            return self.model.module
        return self.model

    @staticmethod
    def _ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _build_text_encoder(self, cfg):
        if self.args.disable_text_condition:
            return None

        t5_device = torch.device("cpu") if self.args.t5_cpu else self.device
        t5_dtype = torch.float32 if self.args.t5_cpu else cfg.t5_dtype

        text_encoder = T5EncoderModel(
            text_len=cfg.text_len,
            dtype=t5_dtype,
            device=t5_device,
            checkpoint_path=os.path.join(self.args.checkpoint_dir, cfg.t5_checkpoint),
            tokenizer_path=os.path.join(self.args.checkpoint_dir, cfg.t5_tokenizer),
            shard_fn=None,
        )
        text_encoder.model.eval().requires_grad_(False)
        return text_encoder

    def _build_visual_vae(self, cfg):
        vae = Wan2_2_VAE(
            vae_pth=os.path.join(self.args.checkpoint_dir, cfg.vae_checkpoint),
            device=self.device,
        )
        vae.model.eval().requires_grad_(False)
        return vae

    def _build_multiview_model(self, cfg, latent_dim: Optional[int] = None):
        model_cfg = copy.deepcopy(cfg)
        model_cfg.model_type = getattr(model_cfg, "model_type", "ti2v")
        if latent_dim is None:
            latent_dim = getattr(model_cfg, "in_dim", None)
        num_layers = (
            int(self.args.debug_num_layers)
            if self.args.debug_num_layers is not None
            else model_cfg.num_layers
        )
        model_kwargs = dict(
            model_type=model_cfg.model_type,
            patch_size=model_cfg.patch_size,
            text_len=model_cfg.text_len,
            in_dim=getattr(model_cfg, "in_dim", latent_dim or 16),
            dim=model_cfg.dim,
            ffn_dim=model_cfg.ffn_dim,
            freq_dim=model_cfg.freq_dim,
            text_dim=getattr(model_cfg, "text_dim", 4096),
            out_dim=getattr(model_cfg, "out_dim", latent_dim or 16),
            num_heads=model_cfg.num_heads,
            num_layers=num_layers,
            window_size=model_cfg.window_size,
            qk_norm=model_cfg.qk_norm,
            cross_attn_norm=model_cfg.cross_attn_norm,
            enable_multiview_attn=True,
            enable_tactile_intra_attn=True,
            max_num_views=self.args.max_num_views,
            use_view_pos_emb=True,
            tactile_dim_ratio=self.args.tactile_dim_ratio,
            joint_dim_ratio=self.args.joint_dim_ratio,
            eps=model_cfg.eps,
            init_weights=not self.args.skip_init_weights,
            use_activation_checkpoint=True,
        )

        if getattr(self.args, "enable_action_expert", False):
            action_num_attention_heads = int(
                getattr(self.args, "action_num_attention_heads", 16)
            )
            action_attention_head_dim = int(
                getattr(self.args, "action_attention_head_dim", 64)
            )
            action_num_layers = int(getattr(self.args, "action_num_layers", 12))
            model_kwargs.update(
                dict(
                    action_expert=True,
                    action_in_channels=getattr(self.args, "action_in_channels", None),
                    action_out_channels=getattr(self.args, "action_out_channels", None),
                    action_num_attention_heads=action_num_attention_heads,
                    action_attention_head_dim=action_attention_head_dim,
                    action_rope_dim=action_attention_head_dim,
                    action_num_layers=action_num_layers,
                    action_final_embeddings=not getattr(
                        self.args, "disable_action_final_embeddings", False
                    ),
                    learnable_action_state=not getattr(
                        self.args, "disable_learnable_action_state", False
                    ),
                    action_norm_elementwise_affine=False,
                    action_output_modalities=getattr(
                        self, "action_modalities_dims", None
                    ),
                )
            )
        model = WanModel(**model_kwargs)
        return model

    @staticmethod
    def _normalize_state_dict(
        state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        normalized = state_dict
        for prefix in ("model.", "module."):
            if normalized and all(key.startswith(prefix) for key in normalized.keys()):
                normalized = {
                    key[len(prefix) :]: value for key, value in normalized.items()
                }
        return normalized

    def _load_pretrained_weights(
        self,
        source: str,
        checkpoint_path: Optional[str],
        checkpoint_dir: str,
    ):
        if source == "wan":
            print("Loading pretrained weights from wan...")
            pretrained = WanModel.from_pretrained(checkpoint_dir)
            pretrained_state = pretrained.state_dict()
            del pretrained
            source_desc = checkpoint_dir
        elif source == "vidar":
            print("Loading pretrained weights from vidar...")
            if checkpoint_path is None:
                raise ValueError(
                    "--pretrained_checkpoint_path is required when --pretrained_source vidar is used"
                )
            raw_state = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(raw_state, dict):
                for candidate_key in (
                    "state_dict",
                    "model",
                    "model_state_dict",
                    "ema",
                    "module",
                ):
                    candidate = raw_state.get(candidate_key)
                    if isinstance(candidate, dict):
                        raw_state = candidate
                        break
            if not isinstance(raw_state, dict):
                raise TypeError(
                    f"Unsupported vidar checkpoint format: {type(raw_state)!r}"
                )
            pretrained_state = self._normalize_state_dict(raw_state)
            source_desc = checkpoint_path
        else:
            raise ValueError(f"Unsupported pretrained source: {source}")

        model_state = self.model.state_dict()
        loaded_keys = []
        skipped_keys = []

        for key, value in pretrained_state.items():
            if key in model_state and model_state[key].shape == value.shape:
                model_state[key] = value
                loaded_keys.append(key)
            else:
                skipped_keys.append(key)

        self.model.load_state_dict(model_state, strict=True)
        self._log(
            f"Loaded {len(loaded_keys)} pretrained tensors from {source_desc}; "
            f"kept {len(skipped_keys)} tensors random-initialized for new multiview/tactile modules."
        )

    @staticmethod
    def _load_deepspeed_config(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _log(self, msg: str):
        if self.dist.rank == 0:
            print(msg, flush=True)

    @staticmethod
    def _count_module_params(module: Optional[nn.Module]) -> int:
        if module is None:
            return 0
        return sum(getattr(p, "ds_numel", p.numel()) for p in module.parameters())

    @staticmethod
    def _count_trainable_params(module: nn.Module) -> int:
        return sum(
            getattr(p, "ds_numel", p.numel())
            for p in module.parameters()
            if getattr(p, "requires_grad", False)
        )

    def _log_attention_param_stats(self):
        model = self._base_model()

        self_attn_params = 0
        self_attn_tactile_params = 0
        joint_attn_params = 0

        for block in model.blocks:
            self_attn_params += self._count_module_params(
                getattr(block, "self_attn", None)
            )

            # tactile-related: norm + down/up projections + tactile self-attention
            self_attn_tactile_params += self._count_module_params(
                getattr(block, "norm1_tactile", None)
            )
            self_attn_tactile_params += self._count_module_params(
                getattr(block, "tactile_down", None)
            )
            self_attn_tactile_params += self._count_module_params(
                getattr(block, "self_attn_tactile", None)
            )
            self_attn_tactile_params += self._count_module_params(
                getattr(block, "tactile_up", None)
            )

            # joint-related: norm + down/up projections + joint attention
            joint_attn_params += self._count_module_params(
                getattr(block, "norm_joint", None)
            )
            joint_attn_params += self._count_module_params(
                getattr(block, "joint_down", None)
            )
            joint_attn_params += self._count_module_params(
                getattr(block, "joint_attn", None)
            )
            joint_attn_params += self._count_module_params(
                getattr(block, "joint_up", None)
            )

        ratio_tactile_vs_self = (
            (self_attn_tactile_params / self_attn_params)
            if self_attn_params > 0
            else 0.0
        )
        ratio_joint_vs_self = (
            (joint_attn_params / self_attn_params) if self_attn_params > 0 else 0.0
        )
        # total trainable params in the model
        total_trainable_params = self._count_trainable_params(model)

        ratio_self_vs_total = (
            (self_attn_params / total_trainable_params)
            if total_trainable_params > 0
            else 0.0
        )
        ratio_tactile_vs_total = (
            (self_attn_tactile_params / total_trainable_params)
            if total_trainable_params > 0
            else 0.0
        )
        ratio_joint_vs_total = (
            (joint_attn_params / total_trainable_params)
            if total_trainable_params > 0
            else 0.0
        )

        self._log(
            "[ParamStats] "
            f"self_attn={self_attn_params:,}, "
            f"self_attn_tactile_related={self_attn_tactile_params:,}, "
            f"joint_attn_related={joint_attn_params:,}, "
            f"total_trainable={total_trainable_params:,}"
        )
        self._log(
            "[ParamStats][Ratio] "
            f"self_attn:self_attn_tactile_related:joint_attn_related = 1:{ratio_tactile_vs_self:.4f}:{ratio_joint_vs_self:.4f}; "
            f"vs_total (self:tactile:joint:total) = {ratio_self_vs_total:.4f}:{ratio_tactile_vs_total:.4f}:{ratio_joint_vs_total:.4f}:1.0000"
        )

    def _log_action_expert_param_stats(self):
        if not getattr(self.args, "enable_action_expert", False):
            return

        model = self._base_model()

        total_params = 0
        trainable_params = 0
        action_params = 0
        action_trainable_params = 0

        for name, param in model.named_parameters():
            numel = getattr(param, "ds_numel", param.numel())
            total_params += numel
            if param.requires_grad:
                trainable_params += numel

            if name.startswith("action_"):
                action_params += numel
                if param.requires_grad:
                    action_trainable_params += numel

        ratio_action_vs_total = (action_params / total_params) if total_params > 0 else 0.0
        ratio_action_trainable_vs_trainable = (
            (action_trainable_params / trainable_params) if trainable_params > 0 else 0.0
        )

        self._log(
            "[ActionExpert][ParamStats] "
            f"action_total={action_params:,}, "
            f"action_trainable={action_trainable_params:,}, "
            f"model_total={total_params:,}, "
            f"model_trainable={trainable_params:,}"
        )
        self._log(
            "[ActionExpert][Ratio] "
            f"action/total={ratio_action_vs_total:.6f}, "
            f"action_trainable/model_trainable={ratio_action_trainable_vs_trainable:.6f}"
        )

    def _encode_text(
        self, captions: List[str], repeat_per_scene: int
    ) -> List[torch.Tensor]:
        if self.text_encoder is None:
            text_dim = int(self._base_model().text_dim)
            dummy = torch.zeros(1, text_dim, device=self.device)
            outs = [dummy for _ in captions]
        else:
            with torch.no_grad():
                device = torch.device("cpu") if self.args.t5_cpu else self.device
                outs = self.text_encoder(captions, device)
                outs = [x.to(self.device) for x in outs]

        flat: List[torch.Tensor] = []
        for item in outs:
            flat.extend([item] * repeat_per_scene)
        return flat

    @staticmethod
    def _latent_seq_len(latent: torch.Tensor, patch_size: Sequence[int]) -> int:
        _, f, h, w = latent.shape
        return int(f * (h // patch_size[1]) * (w // patch_size[2]))

    def _build_timestep_tokens(
        self, t_scalar: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        # t_scalar: [B_total], output: [B_total, seq_len]
        return t_scalar.float().unsqueeze(1).repeat(1, seq_len)

    @staticmethod
    def _align_sequence_to_length(
        sequence: torch.Tensor,
        target_length: int,
        pad_value: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence.ndim == 2:
            sequence = sequence.unsqueeze(1)

        batch_size, current_length, feature_dim = sequence.shape
        device = sequence.device
        dtype = sequence.dtype

        if current_length == target_length:
            mask = torch.ones(batch_size, target_length, 1, device=device, dtype=dtype)
            return sequence, mask

        if current_length > target_length:
            trimmed = sequence[:, current_length - target_length :]
            mask = torch.ones(batch_size, target_length, 1, device=device, dtype=dtype)
            return trimmed, mask

        pad_length = target_length - current_length
        pad = torch.full(
            (batch_size, pad_length, feature_dim), pad_value, device=device, dtype=dtype
        )
        sequence = torch.cat([pad, sequence], dim=1)
        mask = torch.cat(
            [
                torch.zeros(batch_size, pad_length, 1, device=device, dtype=dtype),
                torch.ones(batch_size, current_length, 1, device=device, dtype=dtype),
            ],
            dim=1,
        )
        return sequence, mask

    def _build_joint_action_batch(
        self,
        batch: Dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actions = batch["actions"].to(device=device, dtype=dtype)
        state = batch["state"].to(device=device, dtype=dtype)
        virtual_force = batch["virtual_force"].to(device=device, dtype=dtype)

        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        if state.ndim == 2:
            state = state.unsqueeze(1)
        if virtual_force.ndim == 2:
            virtual_force = virtual_force.unsqueeze(0)

        target_length = actions.shape[1]
        virtual_force, virtual_force_mask = self._align_sequence_to_length(
            virtual_force, target_length
        )

        # Align full state sequence to target length and supervise all timesteps
        state_target, state_mask = self._align_sequence_to_length(
            state, target_length
        )
        state_mask = state_mask.expand(-1, -1, state_target.shape[-1]).contiguous()

        action_mask = torch.ones_like(actions)
        force_mask = virtual_force_mask.expand(
            -1, -1, virtual_force.shape[-1]
        ).contiguous()

        joint_targets = torch.cat([actions, state_target, virtual_force], dim=-1)
        joint_masks = torch.cat([action_mask, state_mask, force_mask], dim=-1)
        action_timestep = (
            torch.arange(target_length, device=device, dtype=torch.long)
            .unsqueeze(0)
            .repeat(actions.shape[0], 1)
        )
        action_states = torch.zeros_like(joint_targets)
        action_states = action_states.to(
            dtype=self._base_model().action_proj_in.weight.dtype
        )
        return action_states, action_timestep, joint_targets, joint_masks

    @staticmethod
    def _masked_weighted_mse(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        loss = (prediction.float() - target.float()).pow(2) * mask.float()
        denom = mask.float().sum().clamp(min=1.0)
        return loss.sum() / denom

    def _forward_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        video = batch["video"].to(
            self.device, non_blocking=True
        )  # [B, C, V_vis, T, H, W]
        tactile = batch["tactile"].to(
            self.device, non_blocking=True
        )  # [B, C, 2, T, H, W]
        captions = list(batch["caption"])

        bs = video.shape[0]
        n_vis_views = video.shape[2]  # 动态获取视觉视角数（1 或 2）

        with torch.no_grad():
            # 动态编码所有视觉视角
            visual_latents_list = []
            for v in range(n_vis_views):
                view_videos = [video[i, :, v].contiguous() for i in range(bs)]
                visual_latents_list.append(self.vae.encode(view_videos))

            # tactile: [B, C, 2, T, H, W]
            # 左右拼接：先取左右视角然后在宽度维度上拼接（沿最后一维），得到 [B, C, T, H, W*2]
            left = tactile[:, :, 0].contiguous()  # [B, C, T, H, W]
            right = tactile[:, :, 1].contiguous()  # [B, C, T, H, W]
            tactile_concat = torch.cat([left, right], dim=-1)  # [B, C, T, H, W*2]
            tactile_videos_list = [tactile_concat[i].contiguous() for i in range(bs)]
            tactile_latents = self.vae.encode(tactile_videos_list)

        clean_latents: List[torch.Tensor] = []
        tactile_view_mask: List[bool] = []
        view_batch_sizes: List[int] = []

        for i in range(bs):
            # 追加所有视觉视角
            for v in range(n_vis_views):
                clean_latents.append(visual_latents_list[v][i])
                tactile_view_mask.append(False)
            # 追加触觉
            clean_latents.append(tactile_latents[i])
            tactile_view_mask.append(True)
            view_batch_sizes.append(n_vis_views + 1)

        b_total = len(clean_latents)

        # Flow-matching style corruption: x_t = (1-sigma) * x0 + sigma * eps
        # target flow = eps - x0
        t_int = torch.randint(
            low=0,
            high=self.num_train_timesteps,
            size=(b_total,),
            device=self.device,
            dtype=torch.long,
        )
        sigma = (t_int.float() / float(self.num_train_timesteps)).clamp(0.0, 1.0)

        noisy_latents: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        for i, x0 in enumerate(clean_latents):
            eps = torch.randn_like(x0)
            s = sigma[i].view(1, 1, 1, 1)
            xt = (1.0 - s) * x0 + s * eps
            flow = eps - x0
            noisy_latents.append(xt)
            targets.append(flow)

        patch_size = self._base_model().patch_size
        seq_len = max(self._latent_seq_len(x, patch_size) for x in noisy_latents)
        t_tokens = self._build_timestep_tokens(t_int, seq_len)
        context = self._encode_text(captions, repeat_per_scene=n_vis_views + 1)

        if getattr(self.args, "enable_action_expert", False):
            action_states, action_timestep, joint_targets, joint_masks = (
                self._build_joint_action_batch(
                    batch,
                    device=self.device,
                    dtype=video.dtype,
                )
            )

            model_output = self.model(
                x=noisy_latents,
                t=t_tokens,
                context=context,
                seq_len=seq_len,
                y=None,
                view_batch_sizes=view_batch_sizes,
                tactile_view_mask=tactile_view_mask,
                action_states=action_states,
                action_timestep=action_timestep,
                return_video=False,
                return_action=True,
            )

            action_pred = (
                model_output[0] if isinstance(model_output, tuple) else model_output
            )
            return self._masked_weighted_mse(action_pred, joint_targets, joint_masks)

        pred = self.model(
            x=noisy_latents,
            t=t_tokens,
            context=context,
            seq_len=seq_len,
            y=None,
            view_batch_sizes=view_batch_sizes,
            tactile_view_mask=tactile_view_mask,
        )

        losses = [
            F.mse_loss(p.float(), y.float(), reduction="mean")
            for p, y in zip(pred, targets)
        ]
        return torch.stack(losses).mean()

    def _save_ckpt(self, epoch: int):
        if self.use_deepspeed:
            tag = f"epoch_{epoch:04d}_step_{self.global_step:08d}"
            save_filename = f"ckpt_step_{self.global_step:08d}.pt"
            saved = self.model.save_16bit_model(
                self.args.output_dir, save_filename=save_filename
            )
            if self.dist.distributed:
                dist.barrier()
            if self.dist.rank == 0:
                if saved:
                    self._log(
                        f"Saved consolidated torch checkpoint to {os.path.join(self.args.output_dir, save_filename)}"
                    )
                else:
                    self._log(
                        "DeepSpeed skipped saving 16bit model for this checkpoint."
                    )
            if self.dist.distributed:
                dist.barrier()
            self._save_loss_curve()
            return

        if self.dist.rank != 0:
            return

        model_to_save = self._base_model()
        ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model": model_to_save.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "args": vars(self.args),
        }
        path = os.path.join(
            self.args.output_dir, f"ckpt_step_{self.global_step:08d}.pt"
        )
        torch.save(ckpt, path)
        self._save_loss_curve()

    def _append_loss_history(self, loss_value: float):
        self.loss_history.append(
            {"step": float(self.global_step), "loss": float(loss_value)}
        )

    def _save_loss_curve(self):
        if self.dist.rank != 0:
            return

        if not self.loss_history:
            return

        raw_json_path = os.path.join(self.args.output_dir, "loss_curve_raw.json")
        raw_csv_path = os.path.join(self.args.output_dir, "loss_curve_raw.csv")
        json_path = os.path.join(self.args.output_dir, "loss_curve.json")
        csv_path = os.path.join(self.args.output_dir, "loss_curve.csv")
        png_path = os.path.join(self.args.output_dir, "loss_curve.png")

        raw_payload = [
            {"step": int(item["step"]), "loss": float(item["loss"])}
            for item in self.loss_history
        ]
        with open(raw_json_path, "w", encoding="utf-8") as f:
            json.dump(raw_payload, f, ensure_ascii=False, indent=2)

        with open(raw_csv_path, "w", encoding="utf-8") as f:
            f.write("step,loss\n")
            for item in raw_payload:
                f.write(f"{item['step']},{item['loss']}\n")

        window = max(1, int(self.args.log_every))
        window = min(window, len(raw_payload))
        if window <= 1 or len(raw_payload) <= 1:
            smoothed_payload = raw_payload
        else:
            smoothed_payload = []
            losses = np.asarray(
                [item["loss"] for item in raw_payload], dtype=np.float64
            )
            steps = np.asarray([item["step"] for item in raw_payload], dtype=np.int64)
            kernel = np.ones(window, dtype=np.float64) / float(window)
            smoothed_losses = np.convolve(losses, kernel, mode="valid")
            smoothed_steps = steps[window - 1 :]
            for step, loss_value in zip(
                smoothed_steps.tolist(), smoothed_losses.tolist()
            ):
                smoothed_payload.append({"step": int(step), "loss": float(loss_value)})

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(smoothed_payload, f, ensure_ascii=False, indent=2)

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("step,loss\n")
            for item in smoothed_payload:
                f.write(f"{item['step']},{item['loss']}\n")

        try:
            import matplotlib.pyplot as plt

            steps = [item["step"] for item in smoothed_payload]
            losses = [item["loss"] for item in smoothed_payload]
            plt.figure(figsize=(8, 4))
            plt.plot(steps, losses, linewidth=1.5)
            plt.xlabel("step")
            plt.ylabel("loss")
            plt.title("Training Loss Curve")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(png_path, dpi=150)
            plt.close()
        except Exception as exc:
            self._log(
                f"[LossCurve] Saved smoothed history to {json_path} and {csv_path} (raw: {raw_json_path}, {raw_csv_path}), but skipped PNG: {exc}"
            )

    def train(self):
        if self.dist.rank == 0:
            self._log_action_expert_param_stats()
            self._log_attention_param_stats()

        for epoch in range(self.args.epochs):
            for pass_idx in range(getattr(self.args, "passes_per_epoch", 1)):
                if self.sampler is not None:
                    # ensure different seed/shuffle each pass
                    sampler_epoch = (
                        epoch * getattr(self.args, "passes_per_epoch", 1) + pass_idx
                    )
                    self.sampler.set_epoch(sampler_epoch)

                iterator = tqdm(
                    self.loader,
                    disable=(self.dist.rank != 0),
                    desc=f"epoch {epoch} pass {pass_idx}",
                )

                if not self.use_deepspeed:
                    self.optimizer.zero_grad(set_to_none=True)

                for step, batch in enumerate(iterator):
                    amp_enabled = (not self.use_deepspeed) and (
                        self.args.use_fp16 or self.args.use_bf16
                    )
                    amp_dtype = torch.float16 if self.args.use_fp16 else torch.bfloat16

                    with torch.autocast(
                        device_type="cuda", enabled=amp_enabled, dtype=amp_dtype
                    ):
                        loss = self._forward_loss(batch) / self.args.grad_accum_steps

                    if self.use_deepspeed:
                        self.model.backward(loss)
                        self.model.step()
                    else:
                        if self.scaler.is_enabled():
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()

                        if (step + 1) % self.args.grad_accum_steps == 0:
                            if self.scaler.is_enabled():
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)

                    if self.dist.rank == 0:
                        self._append_loss_history(
                            float(loss.item() * self.args.grad_accum_steps)
                        )

                    self.global_step += 1

                    if (
                        self.dist.rank == 0
                        and self.global_step % self.args.log_every == 0
                    ):
                        iterator.set_postfix(
                            loss=float(loss.item() * self.args.grad_accum_steps)
                        )

                    if self.global_step % self.args.save_every == 0:
                        self._save_ckpt(epoch)

            # after completing all passes for this epoch, save checkpoint
            self._save_ckpt(epoch)

        if self.dist.rank == 0:
            self._save_loss_curve()


def init_dist(args: argparse.Namespace) -> DistInfo:
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if distributed:
        if args.use_deepspeed:
            deepspeed = importlib.import_module("deepspeed")
            deepspeed.init_distributed(timeout=timedelta(minutes=60))
        else:
            dist.init_process_group(
                backend=args.dist_backend,
                timeout=timedelta(minutes=60),
            )

    return DistInfo(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        distributed=distributed,
    )


def cleanup_dist():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Wan multi-view tactile training")

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Wan2.2 checkpoint root",
    )
    parser.add_argument(
        "--pretrained_source",
        type=str,
        default="wan",
        choices=["wan", "vidar"],
        help="Source of pretrained DiT weights to load",
    )
    parser.add_argument(
        "--pretrained_checkpoint_path",
        type=str,
        default=None,
        help="Path to vidar .pt checkpoint when --pretrained_source vidar is used",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root dir containing UniVTAC hdf5 files",
    )
    parser.add_argument("--dataset_info_cache_path", type=str, default=None)

    parser.add_argument(
        "--config", type=str, default="ti2v-5B", choices=list(WAN_CONFIGS.keys())
    )

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)

    parser.add_argument("--sample_h", type=int, default=192)
    parser.add_argument("--sample_w", type=int, default=256)
    parser.add_argument("--sample_n_frames", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=9)
    parser.add_argument("--action_chunk", type=int, default=9)
    parser.add_argument(
        "--previous_pick_mode",
        type=str,
        default="random",
        choices=["random", "uniform"],
    )
    parser.add_argument("--samples_per_episode", type=int, default=10)
    parser.add_argument("--preload_to_memory", default=True)
    parser.add_argument("--task_names", type=str, nargs="+", default=None)
    parser.add_argument("--max_episodes_per_task", type=int, default=None)
    parser.add_argument("--max_total_episodes", type=int, default=None)
    parser.add_argument("--fast_index", action="store_true")

    parser.add_argument("--use_unified_prompt", action="store_true")
    parser.add_argument(
        "--unified_prompt",
        type=str,
        default="The robotic arm performs a precise insertion task with stable contact.",
    )

    parser.add_argument("--disable_text_condition", action="store_true")
    parser.add_argument("--t5_cpu", action="store_true")

    parser.add_argument(
        "--tactile_placeholder_mode",
        type=str,
        default="zeros",
        choices=["zeros", "random"],
    )
    parser.add_argument("--max_num_views", type=int, default=8)

    parser.add_argument(
        "--output_dir", type=str, default="results/train_multiview_tactile"
    )
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=500)

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dist_backend", type=str, default="nccl")
    parser.add_argument("--find_unused_parameters", action="store_true")

    parser.add_argument("--use_deepspeed", action="store_true")
    parser.add_argument(
        "--deepspeed_config", type=str, default="train/deepspeed_zero2.json"
    )

    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--skip_pretrained", action="store_true")
    parser.add_argument("--skip_init_weights", action="store_true")
    parser.add_argument(
        "--tactile_dim_ratio",
        type=float,
        default=0.25,
        help="Internal width ratio for tactile self-attention modules",
    )
    parser.add_argument(
        "--joint_dim_ratio",
        type=float,
        default=0.5,
        help="Internal width ratio for joint attention modules",
    )
    parser.add_argument(
        "--debug_num_layers",
        type=int,
        default=None,
        help="Override WanModel num_layers for debug runs",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable quick debug preset for fast smoke run",
    )
    parser.add_argument(
        "--enable_action_expert",
        action="store_true",
        help="Enable the second-stage action/state/force expert",
    )
    parser.add_argument(
        "--action_num_layers",
        type=int,
        default=12,
        help="Number of action expert blocks (smaller than backbone layers is allowed)",
    )
    parser.add_argument(
        "--action_num_attention_heads",
        type=int,
        default=16,
        help="Number of attention heads used by action expert",
    )
    parser.add_argument(
        "--action_attention_head_dim",
        type=int,
        default=64,
        help="Per-head dimension of action expert attention",
    )
    parser.add_argument(
        "--disable_action_final_embeddings",
        action="store_true",
        help="Disable FiLM-like final embedding modulation in action expert",
    )
    parser.add_argument(
        "--disable_learnable_action_state",
        action="store_true",
        help="Disable learnable action state token initialization",
    )

    parser.add_argument(
        "--passes_per_epoch",
        type=int,
        default=20,
        help="Number of full passes over the dataset to count as one epoch (default: 10)",
    )

    return parser


def apply_debug_preset(args: argparse.Namespace) -> argparse.Namespace:
    if not args.debug:
        return args

    # Keep it minimal and robust: fast indexing + tiny model + tiny data slice.
    args.batch_size = 1
    args.num_workers = 0
    args.log_every = 1
    args.samples_per_episode = 1

    args.sample_n_frames = 8
    args.sample_h = 96
    args.sample_w = 128
    args.chunk = 5
    args.action_chunk = 5

    args.skip_pretrained = True
    args.skip_init_weights = True
    if args.debug_num_layers is None:
        args.debug_num_layers = 2

    if args.max_episodes_per_task is None:
        args.max_episodes_per_task = 2
    if args.max_total_episodes is None:
        args.max_total_episodes = 16
    args.fast_index = True

    if not args.use_fp16:
        args.use_bf16 = True

    return args


def main():
    parser = build_parser()
    args = parser.parse_args()
    args = apply_debug_preset(args)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")
    if args.use_fp16 and args.use_bf16:
        raise ValueError("Choose at most one of --use_fp16 / --use_bf16.")
    if not os.path.exists(args.data_root):
        raise FileNotFoundError(args.data_root)
    if not os.path.exists(args.checkpoint_dir):
        raise FileNotFoundError(args.checkpoint_dir)
    if args.pretrained_source == "vidar" and not os.path.exists(
        args.pretrained_checkpoint_path
    ):
        raise FileNotFoundError(args.pretrained_checkpoint_path)

    dist_info = init_dist(args)
    trainer = Trainer(args, dist_info)
    trainer.train()
    cleanup_dist()


if __name__ == "__main__":
    main()
