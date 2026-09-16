# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from wan.configs import WAN_CONFIGS
from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae2_2 import Wan2_2_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanTactile:
    """
    Standalone tactile+visual inference wrapper that directly owns:
      - T5 text encoder & tokenizer
      - Wan2.2 VAE
      - WanModel (base DiT with multiview/tactile branches)

    Provides `infer()` for first-frame-conditioned generation of
    visual-view videos + tactile-view video via explicit denoising loop.
    """

    def __init__(
        self,
        config: str,
        checkpoint_dir: str,
        device_id: int = 0,
        t5_cpu: bool = False,
        init_on_cpu: bool = False,
        param_dtype: torch.dtype = torch.bfloat16,
        model_ckpt_path: Optional[str] = None,
        enable_action_expert: bool = False,
        action_in_channels: Optional[int] = None,
        action_out_channels: Optional[int] = None,
        action_modalities_dims: Optional[Dict[str, int]] = None,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device(f"cuda:{device_id}")
        self.param_dtype = param_dtype
        self.t5_cpu = t5_cpu

        cfg = WAN_CONFIGS[config]
        self.cfg = cfg
        self.num_train_timesteps = int(cfg.num_train_timesteps)
        self.vae_stride = cfg.vae_stride
        self.patch_size = cfg.patch_size

        # ------------------------------------------------------------------
        # 1. Text Encoder (T5)
        # ------------------------------------------------------------------
        t5_device = torch.device("cpu") if t5_cpu else self.device
        t5_dtype = torch.float32 if t5_cpu else cfg.t5_dtype
        self.text_encoder = T5EncoderModel(
            text_len=cfg.text_len,
            dtype=t5_dtype,
            device=t5_device,
            checkpoint_path=os.path.join(checkpoint_dir, cfg.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, cfg.t5_tokenizer),
            shard_fn=None,
        )
        self.text_encoder.model.eval().requires_grad_(False)

        # ------------------------------------------------------------------
        # 2. VAE
        # ------------------------------------------------------------------
        # If init_on_cpu, create VAE on CPU to avoid init-time GPU peak;
        # infer() will move it to GPU when needed.
        vae_device = "cpu" if init_on_cpu else self.device
        self.vae = Wan2_2_VAE(
            vae_pth=os.path.join(checkpoint_dir, cfg.vae_checkpoint),
            device=vae_device,
        )
        self.vae.model.eval().requires_grad_(False)
        latent_dim = getattr(getattr(self.vae, "model", None), "z_dim", None)
        in_dim = getattr(cfg, "in_dim", latent_dim or 16)
        out_dim = getattr(cfg, "out_dim", latent_dim or 16)

        # ------------------------------------------------------------------
        # 3. Build action kwargs (explicitly controlled by enable_action_expert)
        # ------------------------------------------------------------------
        action_kwargs: Dict[str, Any] = {}
        if enable_action_expert:
            # Resolve in/out channels from modalities or explicit args
            if action_modalities_dims is not None:
                inferred_ch = sum(int(v) for v in action_modalities_dims.values())
                action_in_channels = action_in_channels if action_in_channels is not None else inferred_ch
                action_out_channels = action_out_channels if action_out_channels is not None else inferred_ch

            # Fallback to checkpoint args if still missing
            if (action_in_channels is None or action_out_channels is None) and (
                model_ckpt_path is not None and os.path.exists(model_ckpt_path)
            ):
                ckpt_preview = torch.load(model_ckpt_path, map_location="cpu")
                saved_args = ckpt_preview.get("args", {}) if isinstance(ckpt_preview, dict) else {}
                if saved_args:
                    action_in_channels = action_in_channels or saved_args.get("action_in_channels")
                    action_out_channels = action_out_channels or saved_args.get("action_out_channels")
                    action_modalities_dims = action_modalities_dims or saved_args.get("action_modalities_dims")

            if action_in_channels is None or action_out_channels is None:
                raise ValueError(
                    "enable_action_expert=True requires action_in_channels/action_out_channels "
                    "or action_modalities_dims, and they were not found in checkpoint args."
                )

            # Load other hyperparams from checkpoint args or use defaults
            saved_args = {}
            if model_ckpt_path is not None and os.path.exists(model_ckpt_path):
                ckpt_preview = torch.load(model_ckpt_path, map_location="cpu")
                saved_args = ckpt_preview.get("args", {}) if isinstance(ckpt_preview, dict) else {}

            action_attention_head_dim = int(saved_args.get("action_attention_head_dim", 64)) if saved_args else 64
            action_kwargs = dict(
                action_expert=True,
                action_in_channels=action_in_channels,
                action_out_channels=action_out_channels,
                action_num_attention_heads=int(saved_args.get("action_num_attention_heads", 16)) if saved_args else 16,
                action_attention_head_dim=action_attention_head_dim,
                action_rope_dim=action_attention_head_dim,
                action_num_layers=int(saved_args.get("action_num_layers", 28)) if saved_args else 28,
                action_final_embeddings=not saved_args.get("disable_action_final_embeddings", False) if saved_args else True,
                learnable_action_state=not saved_args.get("disable_learnable_action_state", False) if saved_args else True,
                action_norm_elementwise_affine=False,
            )
            if action_modalities_dims is not None:
                action_kwargs["action_output_modalities"] = action_modalities_dims

        # ------------------------------------------------------------------
        # 4. Base Model (WanModel) with multiview/tactile structure
        # ------------------------------------------------------------------
        model_kwargs = dict(
            model_type=getattr(cfg, "model_type", "ti2v"),
            patch_size=cfg.patch_size,
            text_len=cfg.text_len,
            in_dim=in_dim,
            dim=cfg.dim,
            ffn_dim=cfg.ffn_dim,
            freq_dim=cfg.freq_dim,
            text_dim=getattr(cfg, "text_dim", 4096),
            out_dim=out_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            window_size=cfg.window_size,
            qk_norm=cfg.qk_norm,
            cross_attn_norm=cfg.cross_attn_norm,
            enable_multiview_attn=True,
            enable_tactile_intra_attn=True,
            max_num_views=8,
            use_view_pos_emb=True,
            tactile_dim_ratio=0.25,
            joint_dim_ratio=0.5,
            eps=cfg.eps,
            init_weights=False,
            use_activation_checkpoint=False,
        )
        if action_kwargs:
            model_kwargs.update(action_kwargs)

        self.model = WanModel(**model_kwargs)

        # Load weights: priority -> explicit ckpt > diffusers dir
        if model_ckpt_path is not None and os.path.exists(model_ckpt_path):
            print(f"[WanTactile] Loading checkpoint from {model_ckpt_path}")
            ckpt = torch.load(model_ckpt_path, map_location="cpu")
            if isinstance(ckpt, dict):
                state_dict = ckpt.get("model")
                if state_dict is None:
                    for key in ("state_dict", "module", "ema"):
                        if key in ckpt and isinstance(ckpt[key], dict):
                            state_dict = ckpt[key]
                            break
                if state_dict is None:
                    state_dict = ckpt
            else:
                state_dict = ckpt

            if isinstance(state_dict, dict):
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                if missing:
                    print(f"[WanTactile] Missing keys: {len(missing)}")
                if unexpected:
                    print(f"[WanTactile] Unexpected keys: {unexpected}")
            else:
                print("[WanTactile] Warning: checkpoint format not recognized")
        else:
            # Try diffusers-format pretrained directory
            try:
                pretrained = WanModel.from_pretrained(checkpoint_dir)
                self.model.load_state_dict(pretrained.state_dict(), strict=False)
                del pretrained
                print(f"[WanTactile] Loaded diffusers weights from {checkpoint_dir}")
            except Exception as e:
                print(f"[WanTactile] Warning: failed to auto-load weights: {e}")

        self.model.eval().requires_grad_(False)
        if init_on_cpu:
            # Keep weights on CPU; infer(offload_model=True) will move to GPU.
            self.model.to(dtype=self.param_dtype)
        else:
            self.model.to(self.device)
            self.model.to(dtype=self.param_dtype)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _preprocess_image(
        self,
        img: Union[Image.Image, torch.Tensor],
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Convert input to [C, 1, H, W] float32 tensor on target device."""
        if isinstance(img, Image.Image):
            img = img.convert("RGB")
            img = img.resize((width, height), Image.LANCZOS)
            img = TF.to_tensor(img).sub_(0.5).div_(0.5)
        elif isinstance(img, torch.Tensor):
            if img.dim() == 3 and img.shape[-1] in (1, 3):
                # [H, W, C] -> [C, H, W]
                img = img.permute(2, 0, 1)
            if img.dim() == 3:
                img = img.unsqueeze(1)  # [C, 1, H, W]
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        img = img.to(device=device, dtype=torch.float32)
        if img.dim() == 3:
            img = img.unsqueeze(1)
        return img  # [C, 1, H, W]

    @staticmethod
    def _match_latent_shape(latent: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        if tuple(latent.shape) == target_shape:
            return latent
        if latent.dim() != 4:
            raise ValueError(f"Expected 4D latent, got shape={tuple(latent.shape)}")
        resized = F.interpolate(
            latent.unsqueeze(0),
            size=target_shape[1:],
            mode="trilinear",
            align_corners=False,
        )
        return resized.squeeze(0)

    @staticmethod
    def _make_first_frame_mask(shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Mask with first temporal frame = 0 (condition), others = 1 (denoise)."""
        mask = torch.ones(shape, device=device)
        mask[:, 0] = 0.0
        return mask

    def encode_text(
        self, prompt: str, negative_prompt: str = ""
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Encode prompt & negative prompt. Returns lists of [L, C] tensors."""
        device = self.device
        if not self.t5_cpu:
            self.text_encoder.model.to(device)
            ctx = self.text_encoder([prompt], device)
            ctx_null = self.text_encoder([negative_prompt], device)
        else:
            ctx = self.text_encoder([prompt], torch.device("cpu"))
            ctx_null = self.text_encoder([negative_prompt], torch.device("cpu"))
            ctx = [c.to(device) for c in ctx]
            ctx_null = [c.to(device) for c in ctx_null]
        return ctx, ctx_null

    # ------------------------------------------------------------------
    # Main inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def infer(
        self,
        prompt: str,
        first_frame_head: Union[Image.Image, torch.Tensor],
        first_frame_wrist: Union[Image.Image, torch.Tensor],
        first_frame_tactile: Union[Image.Image, torch.Tensor],
        num_frames: int = 81,
        size: Tuple[int, int] = (1280, 704),  # (W, H)
        visual_views: int = 2,
        num_inference_steps: int = 50,
        shift: float = 5.0,
        seed: int = -1,
        sample_solver: str = "unipc",
        offload_model: bool = False,
        enable_action_expert: bool = False,
        action_states: Optional[torch.Tensor] = None,
        action_timestep: Optional[torch.Tensor] = None,
        action_chunk: int = 9,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        First-frame-conditioned generation for visual + tactile videos.
        Per-view spatial sizes are preserved exactly as in training.
        """
        assert num_frames % 4 == 1, "num_frames must satisfy 4n+1 for Wan VAE"

        device = self.device
        dtype = self.param_dtype
        ow, oh = size

        # ------------------------------------------------------------------
        # 1. Latent shape bookkeeping — per-view shapes (training-consistent)
        # ------------------------------------------------------------------
        C_lat = self.vae.model.z_dim
        F_lat = (num_frames - 1) // self.vae_stride[0] + 1
        H_lat = oh // self.vae_stride[1]
        W_lat = ow // self.vae_stride[2]
        visual_shape = (C_lat, F_lat, H_lat, W_lat)
        ow_tac = ow * 2
        W_lat_tac = ow_tac // self.vae_stride[2]
        tactile_shape = (C_lat, F_lat, H_lat, W_lat_tac)

        # Collect shapes in view order: visual first, then tactile
        target_shapes = [visual_shape] * visual_views + [tactile_shape]
        total_views = visual_views + 1

        # seq_len = max patch-token length across views (same as training)
        seq_lens = []
        for shp in target_shapes:
            f, h, w = shp[1], shp[2], shp[3]
            sl = f * (h // self.patch_size[1]) * (w // self.patch_size[2])
            seq_lens.append(sl)
        sp_size = 1  # inference without sequence parallelism
        seq_len = int(math.ceil(max(seq_lens) / sp_size)) * sp_size

        # ------------------------------------------------------------------
        # 2. Text encoding
        # ------------------------------------------------------------------
        context, _ = self.encode_text(prompt, "")
        context = context * total_views

        # ------------------------------------------------------------------
        # 3. Encode first-frame conditions via VAE
        # ------------------------------------------------------------------
        if offload_model:
            self.model.cpu()
            self.vae.model.to(device)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        z_list = []
        # visual views — map provided first-frame args to each visual view
        visual_first_frames = [first_frame_head, first_frame_wrist]
        for v_idx in range(visual_views):
            frame = visual_first_frames[min(v_idx, len(visual_first_frames) - 1)]
            img_v = self._preprocess_image(frame, oh, ow, device)
            z_v = self.vae.encode([img_v])[0]
            z_v = self._match_latent_shape(z_v, visual_shape)
            z_list.append(z_v)

        # tactile view
        img_t = self._preprocess_image(first_frame_tactile, oh, ow_tac, device)
        z_t = self.vae.encode([img_t])[0]
        z_t = self._match_latent_shape(z_t, tactile_shape)
        z_list.append(z_t)

        if offload_model:
            self.vae.model.cpu()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # 4. Initialize noise & apply first-frame mask (per-view shapes)
        # ------------------------------------------------------------------
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        g = torch.Generator(device=device).manual_seed(seed)

        noise = [
            torch.randn(*shp, dtype=torch.float32, generator=g, device=device)
            for shp in target_shapes
        ]

        mask2_list = [self._make_first_frame_mask(shp, device) for shp in target_shapes]
        latent = [(1.0 - mask2_list[i]) * z_list[i] + mask2_list[i] * noise[i]
                  for i in range(total_views)]

        # ------------------------------------------------------------------
        # 4.5 Action expert setup (training-consistent zero initialization)
        # ------------------------------------------------------------------
        pred_action = None
        if enable_action_expert:
            if not getattr(self.model, "action_expert", False):
                raise RuntimeError("enable_action_expert=True but model has no action_expert.")
            if action_states is None:
                action_dim = getattr(self.model, "action_in_channels", None)
                if action_dim is None:
                    raise RuntimeError("Cannot infer action_in_channels from model.")
                action_states = torch.zeros(
                    1, action_chunk, action_dim, device=device, dtype=torch.float32
                )
            if action_timestep is None:
                action_timestep = torch.arange(
                    action_chunk, device=device, dtype=torch.long
                ).unsqueeze(0)
            # match action_proj_in dtype
            action_dtype = self.model.action_proj_in.weight.dtype
            action_states = action_states.to(dtype=action_dtype)

        # ------------------------------------------------------------------
        # 5. Build schedulers — visual views share one scheduler (batched);
        #    tactile uses its own because spatial shape differs.
        # ------------------------------------------------------------------
        def _make_scheduler():
            if sample_solver == "unipc":
                s = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                s.set_timesteps(num_inference_steps, device=device, shift=shift)
                return s
            elif sample_solver == "dpm++":
                s = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sigmas = get_sampling_sigmas(num_inference_steps, shift)
                timesteps, _ = retrieve_timesteps(s, device=device, sigmas=sigmas)
                s.timesteps = timesteps
                return s
            else:
                raise ValueError(f"Unsupported solver: {sample_solver}")

        visual_scheduler = _make_scheduler() if visual_views > 0 else None
        tactile_scheduler = _make_scheduler()
        timesteps = visual_scheduler.timesteps if visual_scheduler is not None else tactile_scheduler.timesteps

        # ------------------------------------------------------------------
        # 6. Denoising loop
        # ------------------------------------------------------------------
        if offload_model:
            self.vae.model.cpu()
            self.model.to(device)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        tactile_view_mask = [False] * visual_views + [True]
        view_batch_sizes = [total_views]

        for t in tqdm(timesteps, desc="Denoising"):
            latent_input = [x.to(device) for x in latent]

            # Per-view timestep tokens (shapes differ)
            t_scalar = torch.full(
                (total_views, seq_len),
                t.item(),
                dtype=torch.float32,
                device=device,
            )

            with torch.amp.autocast("cuda", dtype=dtype):
                if enable_action_expert:
                    # Action-expert branch: keep CFG-capable structure for future use
                    out = self.model(
                        x=latent_input,
                        t=t_scalar,
                        context=context,
                        seq_len=seq_len,
                        y=None,
                        view_batch_sizes=view_batch_sizes,
                        tactile_view_mask=tactile_view_mask,
                        action_states=action_states,
                        action_timestep=action_timestep,
                        return_video=True,
                        return_action=True,
                    )
                    
                    if isinstance(out, tuple):
                        # out[0]: video noise pred [total_views, ...]
                        # out[1]: action pred [B, action_chunk, D]
                        noise_pred = [out[0][i] for i in range(total_views)]
                        pred_action = out[1]
                    else:
                        noise_pred = [out[i] for i in range(total_views)]
                        pred_action = None

                else:
                    # No-CFG path (matches training)
                    noise_pred_cond = self.model(
                        x=latent_input,
                        t=t_scalar,
                        context=context,
                        seq_len=seq_len,
                        y=None,
                        view_batch_sizes=view_batch_sizes,
                        tactile_view_mask=tactile_view_mask,
                        return_video=True,
                        return_action=False,
                    )
                    noise_pred = [noise_pred_cond[i] for i in range(total_views)]
                    pred_action = None

            # Scheduler step — visual views batched, tactile separate
            if visual_views > 0 and visual_scheduler is not None:
                visual_pred = torch.stack(noise_pred[:visual_views], dim=0)
                visual_latent = torch.stack(latent[:visual_views], dim=0)
                visual_x0 = visual_scheduler.step(
                    visual_pred, t, visual_latent, return_dict=False)[0]
                for i in range(visual_views):
                    latent[i] = visual_x0[i]

            tactile_pred = noise_pred[visual_views].unsqueeze(0)
            tactile_latent = latent[visual_views].unsqueeze(0)
            tactile_x0 = tactile_scheduler.step(
                tactile_pred, t, tactile_latent, return_dict=False)[0]
            latent[visual_views] = tactile_x0.squeeze(0)

            # Re-apply first-frame condition
            for i in range(total_views):
                latent[i] = (1.0 - mask2_list[i]) * z_list[i] + mask2_list[i] * latent[i]

        if offload_model:
            self.model.cpu()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # 7. VAE decode
        # ------------------------------------------------------------------
        if offload_model:
            self.model.cpu()
            self.vae.model.to(device)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        videos = self.vae.decode(latent)

        if offload_model:
            self.vae.model.cpu()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        return videos[:visual_views], videos[visual_views], pred_action