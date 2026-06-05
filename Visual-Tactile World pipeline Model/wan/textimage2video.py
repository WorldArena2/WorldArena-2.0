# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_2 import Wan2_2_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.utils import best_output_size, masks_like


class WanTI2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_2_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model = self._configure_model(
            model=self.model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

        # tactile branch now reuses the same Wan VAE directly.
        self.tactile_vae = self.vae

    @staticmethod
    def _to_wan_vae_video_tensor(video_tensor: torch.Tensor) -> torch.Tensor:
        if video_tensor.dim() != 4:
            raise ValueError(f'Expected a 4D video tensor, got shape={tuple(video_tensor.shape)}')
        if video_tensor.shape[0] in (1, 3) and video_tensor.shape[1] not in (1, 3):
            return video_tensor.contiguous()
        if video_tensor.shape[1] in (1, 3):
            return video_tensor.permute(1, 0, 2, 3).contiguous()
        return video_tensor.permute(1, 0, 2, 3).contiguous()

    def _encode_tactile_tensor(self, tactile_tensor: torch.Tensor) -> torch.Tensor:
        tactile_tensor = tactile_tensor.to(self.device)
        if tactile_tensor.dim() == 5:
            tactile_batch = [self._to_wan_vae_video_tensor(sample) for sample in tactile_tensor]
        elif tactile_tensor.dim() == 4:
            tactile_batch = [self._to_wan_vae_video_tensor(tactile_tensor)]
        else:
            raise ValueError(f'Unsupported tactile tensor rank={tactile_tensor.dim()}')

        with torch.no_grad():
            tactile_latents = self.vae.encode(tactile_batch)
        if not tactile_latents:
            raise ValueError('Wan VAE returned an empty tactile latent list.')
        return tactile_latents[0].float()

    @staticmethod
    def _match_latent_shape(latent: torch.Tensor, target_shape):
        target_shape = tuple(target_shape)
        if tuple(latent.shape) == target_shape:
            return latent
        if latent.dim() != 4:
            raise ValueError(f'Expected a 4D latent tensor, got shape={tuple(latent.shape)}')
        resized = F.interpolate(
            latent.unsqueeze(0),
            size=target_shape[1:],
            mode='trilinear',
            align_corners=False,
        )
        return resized.squeeze(0)

    def _load_tactile_tensor(self, tactile_path):
        if tactile_path is None:
            return None
        if not os.path.exists(tactile_path):
            raise FileNotFoundError(f'Tactile input not found: {tactile_path}')

        obj = torch.load(tactile_path, map_location='cpu')
        if isinstance(obj, dict):
            tensor = None
            for key in ['tactile', 'data', 'x', 'input', 'tensor']:
                if key in obj and torch.is_tensor(obj[key]):
                    tensor = obj[key]
                    break
            if tensor is None:
                for v in obj.values():
                    if torch.is_tensor(v):
                        tensor = v
                        break
            if tensor is None:
                raise ValueError(
                    f'No tensor found in tactile input dict: {tactile_path}')
        elif torch.is_tensor(obj):
            tensor = obj
        else:
            raise ValueError(
                f'Unsupported tactile input type from {tactile_path}: {type(obj)}'
            )

        tensor = tensor.float()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0

        # normalize shape to [B, T, C, H, W] or [B, C, H, W]
        if tensor.dim() == 3:  # [C, H, W]
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 4:
            # [T, C, H, W] -> [1, T, C, H, W]
            # [B, C, H, W] stays 4D
            if tensor.shape[1] in (1, 3) and tensor.shape[0] > 1:
                tensor = tensor.unsqueeze(0)
            elif tensor.shape[1] not in (1, 3):
                tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 5:
            pass
        else:
            raise ValueError(
                f'Unsupported tactile tensor rank={tensor.dim()} from {tactile_path}'
            )
        return tensor

    def _build_tactile_view(self, tactile_path, target_shape):
        tactile_tensor = self._load_tactile_tensor(tactile_path)
        if tactile_tensor is None:
            return None

        tactile_latent = self._encode_tactile_tensor(tactile_tensor)
        tactile_latent = self._match_latent_shape(tactile_latent, target_shape)
        return tactile_latent.float()

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def generate(self,
                 input_prompt,
                 img=None,
                 size=(1280, 704),
                 max_area=704 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 view_batch_sizes=None,
                 tactile_path=None,
                 return_all_views=False):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            size (`tuple[int]`, *optional*, defaults to (1280,704)):
                Controls video resolution, (width,height).
            max_area (`int`, *optional*, defaults to 704*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            view_batch_sizes (List[int] or Tensor, *optional*):
                Multi-view grouping for flattened batch, used only when
                `enable_multiview_attn=True` in Wan model config.

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # i2v
        if img is not None:
            return self.i2v(
                input_prompt=input_prompt,
                img=img,
                max_area=max_area,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                n_prompt=n_prompt,
                seed=seed,
                offload_model=offload_model,
                view_batch_sizes=view_batch_sizes,
                tactile_path=tactile_path)
        # t2v
        return self.t2v(
            input_prompt=input_prompt,
            size=size,
            frame_num=frame_num,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            n_prompt=n_prompt,
            seed=seed,
            offload_model=offload_model,
            view_batch_sizes=view_batch_sizes,
            tactile_path=tactile_path)

    def t2v(self,
            input_prompt,
            size=(1280, 704),
            frame_num=121,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=50,
            guide_scale=5.0,
            n_prompt="",
            seed=-1,
            offload_model=True,
            view_batch_sizes=None,
                 tactile_path=None,
                 return_all_views=False):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (`tuple[int]`, *optional*, defaults to (1280,704)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 121):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            view_batch_sizes (List[int] or Tensor, *optional*):
                Multi-view grouping for flattened batch, used only when
                `enable_multiview_attn=True` in Wan model config.

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        visual_views = 1
        if view_batch_sizes is not None:
            if torch.is_tensor(view_batch_sizes):
                view_batch_sizes = view_batch_sizes.tolist()
            view_batch_sizes = [int(v) for v in view_batch_sizes]
            assert len(view_batch_sizes) == 1, (
                'TI2V t2v currently supports a single scene with multiple views.'
            )
            visual_views = view_batch_sizes[0]
            assert visual_views > 0

        tactile_view = self._build_tactile_view(tactile_path, target_shape)
        use_tactile = tactile_view is not None
        total_views = visual_views + (1 if use_tactile else 0)
        multiview_mode = total_views > 1
        tactile_view_mask = ([False] * visual_views + [True]
                     ) if use_tactile else None

        context = context * total_views
        context_null = context_null * total_views
        effective_view_batch_sizes = [total_views] if multiview_mode else None

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g) for _ in range(visual_views)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync(),
        ):

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise
            mask1, mask2 = masks_like(noise, zero=False)

            arg_c = {
                'context': context,
                'seq_len': seq_len,
                'view_batch_sizes': effective_view_batch_sizes,
                'tactile_view_mask': tactile_view_mask,
            }
            arg_null = {
                'context': context_null,
                'seq_len': seq_len,
                'view_batch_sizes': effective_view_batch_sizes,
                'tactile_view_mask': tactile_view_mask,
            }

            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()

            for _, t in enumerate(tqdm(timesteps)):
                if multiview_mode:
                    latent_model_input = latents + ([tactile_view
                                                    ] if use_tactile else [])
                    timestep = torch.stack([t] * total_views).to(self.device)

                    temp_ts = (mask2[0][0][:, ::2, ::2] * t).flatten()
                    temp_ts = torch.cat([
                        temp_ts,
                        temp_ts.new_ones(seq_len - temp_ts.size(0)) * t
                    ])
                    timestep = temp_ts.unsqueeze(0).repeat(total_views, 1)

                    noise_pred_cond_all = self.model(
                        latent_model_input, t=timestep, **arg_c)
                    noise_pred_uncond_all = self.model(
                        latent_model_input, t=timestep, **arg_null)
                    noise_pred_cond = torch.stack(
                        noise_pred_cond_all[:visual_views], dim=0)
                    noise_pred_uncond = torch.stack(
                        noise_pred_uncond_all[:visual_views], dim=0)

                    noise_pred = noise_pred_uncond + guide_scale * (
                        noise_pred_cond - noise_pred_uncond)

                    latent_batch = torch.stack(latents[:visual_views], dim=0)
                    temp_x0 = sample_scheduler.step(
                        noise_pred,
                        t,
                        latent_batch,
                        return_dict=False,
                        generator=seed_g)[0]
                    latents = [u for u in temp_x0][:visual_views]
                else:
                    latent_model_input = latents
                    timestep = [t]

                    timestep = torch.stack(timestep)

                    temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                    temp_ts = torch.cat([
                        temp_ts,
                        temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                    ])
                    timestep = temp_ts.unsqueeze(0)

                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **arg_c)[0]
                    noise_pred_uncond = self.model(
                        latent_model_input, t=timestep, **arg_null)[0]

                    noise_pred = noise_pred_uncond + guide_scale * (
                        noise_pred_cond - noise_pred_uncond)

                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents[0].unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)[0]
                    latents = [temp_x0.squeeze(0)]
            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        if self.rank != 0:
            return None
        return videos if return_all_views else videos[0]

    def i2v(self, input_prompt, img, max_area=704 * 1280, frame_num=121, shift=5.0, sample_solver='unipc', sampling_steps=40, guide_scale=5.0, n_prompt="", seed=-1, offload_model=True, view_batch_sizes=None, tactile_path=None, return_all_views=False):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 704*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 121):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            view_batch_sizes (List[int] or Tensor, *optional*):
                Multi-view grouping for flattened batch, used only when
                `enable_multiview_attn=True` in Wan model config.

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (121)
                - H: Frame height (from max_area)
                - W: Frame width (from max_area)
        """
        # preprocess
        ih, iw = img.height, img.width
        dh, dw = self.patch_size[1] * self.vae_stride[1], self.patch_size[
            2] * self.vae_stride[2]
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)

        scale = max(ow / iw, oh / ih)
        img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)

        # center-crop
        x1 = (img.width - ow) // 2
        y1 = (img.height - oh) // 2
        img = img.crop((x1, y1, x1 + ow, y1 + oh))
        assert img.width == ow and img.height == oh

        # to tensor
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)
        #[C,F,H,W]

        F = frame_num
        seq_len = ((F - 1) // self.vae_stride[0] + 1) * (
            oh // self.vae_stride[1]) * (ow // self.vae_stride[2]) // (
                self.patch_size[1] * self.patch_size[2])
        seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
            oh // self.vae_stride[1],
            ow // self.vae_stride[2],
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu: 
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        visual_views = 1
        if view_batch_sizes is not None:
            if torch.is_tensor(view_batch_sizes):
                view_batch_sizes = view_batch_sizes.tolist()
            view_batch_sizes = [int(v) for v in view_batch_sizes]
            assert len(view_batch_sizes) == 1, (
                'TI2V i2v currently supports a single scene with multiple views.'
            )
            visual_views = view_batch_sizes[0]
            assert visual_views > 0

        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        oh // self.vae_stride[1], ow // self.vae_stride[2])
        tactile_view = self._build_tactile_view(tactile_path, target_shape)
        use_tactile = tactile_view is not None
        total_views = visual_views + (1 if use_tactile else 0)
        multiview_mode = total_views > 1
        effective_view_batch_sizes = [total_views] if multiview_mode else None
        tactile_view_mask = ([False] * visual_views + [True]
                        ) if use_tactile else None

        context = context * total_views
        context_null = context_null * total_views
        z = self.vae.encode([img])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync(),
        ):

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            if multiview_mode:
                noise = [
                    torch.randn(
                        self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        oh // self.vae_stride[1],
                        ow // self.vae_stride[2],
                        dtype=torch.float32,
                        generator=seed_g,
                        device=self.device) for _ in range(visual_views)
                ]
                latent = noise
                z = z * visual_views
                mask1, mask2 = masks_like(noise, zero=True)
                latent = [
                    (1. - mask2[i]) * z[i] + mask2[i] * latent[i]
                    for i in range(visual_views)
                ]
            else:
                latent = noise
                mask1, mask2 = masks_like([noise], zero=True)
                latent = (1. - mask2[0]) * z[0] + mask2[0] * latent

            arg_c = {
                'context': context,
                'seq_len': seq_len,
                'view_batch_sizes': effective_view_batch_sizes,
                'tactile_view_mask': tactile_view_mask,
            }

            arg_null = {
                'context': context_null,
                'seq_len': seq_len,
                'view_batch_sizes': effective_view_batch_sizes,
                'tactile_view_mask': tactile_view_mask,
            }

            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()

            for _, t in enumerate(tqdm(timesteps)):
                if multiview_mode:
                    latent_model_input = [u.to(self.device) for u in latent]
                    if use_tactile:
                        latent_model_input.append(tactile_view)
                    timestep = torch.stack([t] * total_views).to(self.device)

                    temp_ts = (mask2[0][0][:, ::2, ::2] * t).flatten()
                    temp_ts = torch.cat([
                        temp_ts,
                        temp_ts.new_ones(seq_len - temp_ts.size(0)) * t
                    ])
                    timestep = temp_ts.unsqueeze(0).repeat(total_views, 1)

                    noise_pred_cond_all = self.model(
                        latent_model_input, t=timestep, **arg_c)
                    if offload_model:
                        torch.cuda.empty_cache()
                    noise_pred_uncond_all = self.model(
                        latent_model_input, t=timestep, **arg_null)
                    if offload_model:
                        torch.cuda.empty_cache()
                    noise_pred_cond = torch.stack(
                        noise_pred_cond_all[:visual_views], dim=0)
                    noise_pred_uncond = torch.stack(
                        noise_pred_uncond_all[:visual_views], dim=0)
                    noise_pred = noise_pred_uncond + guide_scale * (
                        noise_pred_cond - noise_pred_uncond)

                    latent_batch = torch.stack(latent[:visual_views], dim=0)
                    temp_x0 = sample_scheduler.step(
                        noise_pred,
                        t,
                        latent_batch,
                        return_dict=False,
                        generator=seed_g)[0]
                    latent = [u for u in temp_x0][:visual_views]
                    latent = [
                        (1. - mask2[i]) * z[i] + mask2[i] * latent[i]
                        for i in range(visual_views)
                    ]
                    x0 = latent
                else:
                    latent_model_input = [latent.to(self.device)]
                    timestep = [t]

                    timestep = torch.stack(timestep).to(self.device)

                    temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                    temp_ts = torch.cat([
                        temp_ts,
                        temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                    ])
                    timestep = temp_ts.unsqueeze(0)

                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **arg_c)[0]
                    if offload_model:
                        torch.cuda.empty_cache()
                    noise_pred_uncond = self.model(
                        latent_model_input, t=timestep, **arg_null)[0]
                    if offload_model:
                        torch.cuda.empty_cache()
                    noise_pred = noise_pred_uncond + guide_scale * (
                        noise_pred_cond - noise_pred_uncond)

                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latent.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)[0]
                    latent = temp_x0.squeeze(0)
                    latent = (1. - mask2[0]) * z[0] + mask2[0] * latent

                    x0 = [latent]
                    del latent_model_input, timestep

            if offload_model:
                self.model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latent, x0
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        if self.rank != 0:
            return None
        return videos if return_all_views else videos[0]
