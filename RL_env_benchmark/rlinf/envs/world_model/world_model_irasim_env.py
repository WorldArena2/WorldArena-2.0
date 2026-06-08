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
"""IRASim world-model environment for RLinf."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from rlinf.envs.utils import recursive_to_device
from rlinf.envs.world_model.base_world_env import BaseWorldEnv

__all__ = ["IRASIMEnv"]


class IRASIMEnv(BaseWorldEnv):
    """World-model environment backed by IRASim action-conditioned diffusion."""

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        record_metrics=True,
        worker_info=None,
    ):
        self.chunk = int(cfg.chunk)
        self.condition_frame_length = int(cfg.condition_frame_length)
        self.num_frames = int(cfg.num_frames)
        if self.num_frames != self.chunk + self.condition_frame_length:
            raise ValueError(
                "num_frames must equal condition_frame_length + chunk, "
                f"got num_frames={self.num_frames}, "
                f"condition_frame_length={self.condition_frame_length}, chunk={self.chunk}"
            )

        self.action_dim = int(cfg.action_dim)
        self.action_key = str(cfg.get("action_key", "abs_action"))
        self.image_size = tuple(int(x) for x in cfg.image_size)
        self.num_inference_steps = int(cfg.get("num_inference_steps", 50))
        self.guidance_scale = float(cfg.get("guidance_scale", 1.0))
        self.vae_encode_batch_size = int(cfg.get("vae_encode_batch_size", 8))
        self.default_task_description = str(
            cfg.get("default_task_description", cfg.get("task_name", ""))
        )
        self.enable_latent_cache = bool(cfg.get("enable_latent_cache", True))
        self.wm_env_type = str(cfg.get("wm_env_type", "robotwin"))

        if self.condition_frame_length != 1:
            raise ValueError(
                "IRASim RobotWin checkpoints currently require condition_frame_length=1, "
                f"got {self.condition_frame_length}"
            )

        self._annotation_cache: dict[str, dict[str, Any]] = {}
        self._is_offloaded = False
        self.current_obs = None
        self.current_latent = None
        self.current_state = None
        self.gt_last_frames = None
        self.task_descriptions = [self.default_task_description] * num_envs
        self._decord = None
        self._decord_cpu = None
        self._OmegaConf = None
        self._AutoencoderKL = None
        self._PNDMScheduler = None
        self._get_models = None
        self._Trajectory2VideoGenPipeline = None

        super().__init__(
            cfg, num_envs, seed_offset, total_num_processes, worker_info, record_metrics
        )

        self.use_fixed_reset_state_ids = bool(cfg.use_fixed_reset_state_ids)
        self.group_size = int(cfg.group_size)
        self.num_group = self.num_envs // self.group_size

        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

        self._setup_irasim_import_path()
        self._lazy_import_irasim_modules()
        self._load_world_model_components()
        self.reward_model = self._load_reward_model().eval().to(self.device)
        self.action_stats = self._load_action_stats(cfg.action_stat_path)

    def _setup_irasim_import_path(self):
        irasim_root = str(self.cfg.get("irasim_root", "")).strip()
        if not irasim_root:
            raise ValueError("env config must provide irasim_root")
        if irasim_root not in sys.path:
            sys.path.insert(0, irasim_root)
        self.irasim_root = Path(irasim_root)

    def _lazy_import_irasim_modules(self):
        if self._AutoencoderKL is not None:
            return

        from decord import VideoReader, cpu
        from diffusers.models import AutoencoderKL
        from diffusers.schedulers import PNDMScheduler
        from omegaconf import OmegaConf

        from models import get_models
        from sample.pipeline_trajectory2videogen import Trajectory2VideoGenPipeline

        self._decord = VideoReader
        self._decord_cpu = cpu
        self._OmegaConf = OmegaConf
        self._AutoencoderKL = AutoencoderKL
        self._PNDMScheduler = PNDMScheduler
        self._get_models = get_models
        self._Trajectory2VideoGenPipeline = Trajectory2VideoGenPipeline

    def _build_dataset(self, cfg):
        data_mode = str(cfg.get("data_mode", "val")).lower()
        meta_path = cfg.val_data_meta if data_mode == "val" else cfg.train_data_meta
        data_dir = Path(str(cfg.train_data_dir))
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if not isinstance(metadata, list) or len(metadata) == 0:
            raise ValueError(f"No episode metadata found in {meta_path}")
        self.data_dir = data_dir
        self.meta_path = Path(meta_path)
        self.data_mode = data_mode
        return metadata

    def _build_model_args(self, wm_cfg):
        return argparse.Namespace(
            model=wm_cfg.get("model", "IRASim-XL/2"),
            latent_size=[self.image_size[0] // 8, self.image_size[1] // 8],
            num_frames=self.num_frames,
            learn_sigma=wm_cfg.get("learn_sigma", False),
            extras=wm_cfg.get("extras", 3),
            attention_mode=wm_cfg.get("attention_mode", "math"),
            dataset=wm_cfg.get("dataset", "robotwin2"),
            final_frame_ada=wm_cfg.get("final_frame_ada", True),
        )

    def _load_checkpoint_state_dict(self, ckpt_path: str):
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "ema" in checkpoint:
            return checkpoint["ema"]
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            return checkpoint["model"]
        return checkpoint

    def _load_world_model_components(self):
        wm_cfg = self._OmegaConf.load(str(self.cfg.wm_config_path))
        if int(wm_cfg.get("num_frames", self.num_frames)) != self.num_frames:
            raise ValueError(
                "env num_frames must match IRASim training config, "
                f"got env={self.num_frames}, wm_config={wm_cfg.get('num_frames')}"
            )
        model_args = self._build_model_args(wm_cfg)
        self.irasim_model = self._get_models(model_args)

        state_dict = self._load_checkpoint_state_dict(str(self.cfg.wm_ckpt_path))
        model_state = self.irasim_model.state_dict()
        filtered_state = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
        }
        self.irasim_model.load_state_dict(filtered_state, strict=False)
        self.irasim_model = self.irasim_model.to(self.device).eval()

        self.vae = self._AutoencoderKL.from_pretrained(
            str(self.cfg.vae_model_path),
            subfolder="vae",
            use_safetensors=True,
        ).to(self.device)
        self.vae.eval()

        self.scheduler = self._PNDMScheduler(
            beta_start=float(wm_cfg.get("beta_start", 0.0001)),
            beta_end=float(wm_cfg.get("beta_end", 0.02)),
            beta_schedule=str(wm_cfg.get("beta_schedule", "linear")),
            num_train_timesteps=1000,
        )

        self.pipeline = self._Trajectory2VideoGenPipeline(
            vae=self.vae,
            scheduler=self.scheduler,
            transformer=self.irasim_model,
        )

    def _load_reward_model(self):
        reward_type = self.cfg.reward_model.type
        if reward_type == "ResnetRewModel":
            from diffsynth.models.reward_model import ResnetRewModel

            rew_model = ResnetRewModel(self.cfg.reward_model.from_pretrained)
        elif reward_type == "TaskEmbedResnetRewModel":
            from diffsynth.models.reward_model import TaskEmbedResnetRewModel

            rew_model = TaskEmbedResnetRewModel(
                checkpoint_path=self.cfg.reward_model.from_pretrained,
                task_suite_name=self.cfg.task_suite_name,
            )
        elif reward_type == "RoboTwinT5CrossAttn":
            from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel

            t5_model_name = self.cfg.reward_model.get("t5_model_name", "t5-base")
            rew_model = RoboTwinT5CrossAttnRewardModel.from_pretrained(
                self.cfg.reward_model.from_pretrained,
                config={"t5_model_name": t5_model_name},
            )
        elif reward_type == "QwenVLMProgressRewardModel":
            from omegaconf import DictConfig
            from rlinf.models.embodiment.reward import QwenVLMProgressRewardModel

            rew_model = QwenVLMProgressRewardModel(DictConfig(self.cfg.reward_model))
        elif reward_type == "LPIPSLastFrameRewardModel":
            from omegaconf import DictConfig
            from rlinf.models.embodiment.reward import LPIPSLastFrameRewardModel

            rew_model = LPIPSLastFrameRewardModel(DictConfig(self.cfg.reward_model))
        else:
            raise ValueError(f"Unknown reward model type: {reward_type}")
        return rew_model

    def _load_action_stats(self, stats_path: str) -> dict[str, np.ndarray]:
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Action stats path does not exist: {stats_path}")
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

        if "state_01" in stats and "state_99" in stats:
            p01 = np.asarray(stats["state_01"], dtype=np.float32)
            p99 = np.asarray(stats["state_99"], dtype=np.float32)
        elif "state" in stats:
            p01 = np.asarray(stats["state"]["p01"], dtype=np.float32)
            p99 = np.asarray(stats["state"]["p99"], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported stat.json format in {stats_path}")

        if p01.shape[0] != self.action_dim or p99.shape[0] != self.action_dim:
            raise ValueError(
                f"Action stats dim mismatch: expected {self.action_dim}, got {p01.shape[0]} and {p99.shape[0]}"
            )
        return {"p01": p01, "p99": p99}

    def _normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        p01 = self.action_stats["p01"][None, None, :]
        p99 = self.action_stats["p99"][None, None, :]
        actions = (actions - p01) / (p99 - p01 + 1e-8)
        actions = np.clip(actions, 0.0, 1.0)
        return (actions * 2.0 - 1.0).astype(np.float32)

    def _resolve_path(self, path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path
        return self.data_dir / path

    def _load_annotation(self, ann_file: str) -> dict[str, Any]:
        ann_path = str(self._resolve_path(ann_file))
        if ann_path not in self._annotation_cache:
            with open(ann_path, "r", encoding="utf-8") as f:
                self._annotation_cache[ann_path] = json.load(f)
        return self._annotation_cache[ann_path]

    def _extract_states(self, ann: dict[str, Any]) -> np.ndarray:
        if "state" in ann:
            return np.asarray(ann["state"], dtype=np.float32)
        if (
            "observation.state.joint_position" in ann
            and "observation.state.gripper_position" in ann
        ):
            joint_pos = np.asarray(ann["observation.state.joint_position"], dtype=np.float32)
            gripper_pos = np.asarray(ann["observation.state.gripper_position"], dtype=np.float32)
            return np.concatenate(
                [
                    joint_pos[:, :6],
                    gripper_pos[:, 0:1],
                    joint_pos[:, 6:],
                    gripper_pos[:, 1:2],
                ],
                axis=1,
            ).astype(np.float32)
        raise KeyError("No state information found in IRASim annotation")

    def _extract_task_description(self, episode_meta: dict[str, Any], ann: dict[str, Any]) -> str:
        for key in ("text", "task_description", "task", "instruction"):
            value = episode_meta.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for key in ("task_name", "task_description", "text"):
            value = ann.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return self.default_task_description

    def _load_video_frame(self, video_path: Path, frame_idx: int) -> np.ndarray:
        reader = self._decord(str(video_path), ctx=self._decord_cpu(0), num_threads=1)
        if len(reader) == 0:
            raise ValueError(f"Empty video: {video_path}")
        if frame_idx < 0:
            frame_idx = len(reader) + frame_idx
        frame_idx = max(0, min(frame_idx, len(reader) - 1))
        return reader.get_batch([frame_idx]).asnumpy()[0]

    def _load_first_frame(self, video_path: Path) -> np.ndarray:
        return self._load_video_frame(video_path, 0)

    def _load_last_frame(self, video_path: Path) -> np.ndarray:
        return self._load_video_frame(video_path, -1)

    def _encode_frames_to_latent(self, frames_uint8: np.ndarray) -> torch.Tensor:
        if frames_uint8.ndim == 3:
            frames_uint8 = frames_uint8[None, ...]
        frames = torch.from_numpy(frames_uint8).float().to(self.device)
        frames = frames.permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0

        latent_batches = []
        with torch.no_grad():
            for start_idx in range(0, frames.shape[0], self.vae_encode_batch_size):
                batch = frames[start_idx : start_idx + self.vae_encode_batch_size]
                latents = self.vae.encode(batch).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                latent_batches.append(latents)

        return torch.cat(latent_batches, dim=0)

    def _init_metrics(self):
        self.success_once = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.returns = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            if self.record_metrics:
                self.success_once[mask] = False
                self.returns[mask] = 0.0
        else:
            self.prev_step_reward[:] = 0.0
            if self.record_metrics:
                self.success_once[:] = False
                self.returns[:] = 0.0
            self._elapsed_steps = 0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward
        if isinstance(terminations, torch.Tensor):
            self.success_once = self.success_once | terminations
        else:
            terminations_tensor = torch.tensor(terminations, device=self.device, dtype=torch.bool)
            self.success_once = self.success_once | terminations_tensor
        episode_info["success_once"] = self.success_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = torch.full(
            (self.num_envs,),
            self.elapsed_steps,
            dtype=torch.float32,
            device=self.device,
        )
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"].clamp(min=1.0)
        infos["episode"] = episode_info
        return infos

    def _calc_step_reward(self, chunk_rewards):
        reward_diffs = torch.zeros((self.num_envs, self.chunk), dtype=torch.float32, device=self.device)
        for i in range(self.chunk):
            reward_diffs[:, i] = self.cfg.reward_coef * chunk_rewards[:, i] - self.prev_step_reward
            self.prev_step_reward = self.cfg.reward_coef * chunk_rewards[:, i]
        return reward_diffs if self.use_rel_reward else chunk_rewards

    def _estimate_success_from_rewards(self, chunk_rewards):
        success_threshold = float(getattr(self.cfg, "success_reward_threshold", 0.9))
        max_reward_in_chunk = chunk_rewards.max(dim=1)[0]
        return (max_reward_in_chunk >= success_threshold).to(self.device)

    def update_reset_state_ids(self):
        total_num_episodes = len(self.dataset)
        reset_state_ids = torch.randint(
            low=0,
            high=total_num_episodes,
            size=(self.num_group,),
            generator=self._generator,
        )
        self.reset_state_ids = reset_state_ids.repeat_interleave(repeats=self.group_size).to(self.device)

    @torch.no_grad()
    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = {},
        episode_indices: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        self.onload()
        self.elapsed_steps = 0

        if self.is_start:
            if self.use_fixed_reset_state_ids:
                episode_indices = self.reset_state_ids
            self._is_start = False

        if episode_indices is None:
            if seed is not None:
                np.random.seed(seed[0] if isinstance(seed, list) else seed)
            replace = len(self.dataset) < self.num_envs
            episode_indices = np.random.choice(len(self.dataset), size=self.num_envs, replace=replace)
        elif isinstance(episode_indices, torch.Tensor):
            episode_indices = episode_indices.cpu().numpy()

        frame_batch = []
        gt_last_frame_batch = []
        state_batch = []
        task_descriptions = []

        for episode_idx in episode_indices:
            episode_meta = self.dataset[int(episode_idx)]
            ann = self._load_annotation(episode_meta["ann_file"])
            video_path = self._resolve_path(episode_meta["file_path"])
            first_frame_uint8 = self._load_first_frame(video_path)
            gt_last_frame_uint8 = self._load_last_frame(video_path)
            episode_states = self._extract_states(ann)
            state_batch.append(episode_states[0])

            frame = torch.from_numpy(first_frame_uint8).float().permute(2, 0, 1) / 255.0
            gt_last_frame = torch.from_numpy(gt_last_frame_uint8).float().permute(2, 0, 1) / 255.0
            if tuple(frame.shape[1:]) != self.image_size:
                frame = F.interpolate(
                    frame.unsqueeze(0),
                    size=self.image_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            if tuple(gt_last_frame.shape[1:]) != self.image_size:
                gt_last_frame = F.interpolate(
                    gt_last_frame.unsqueeze(0),
                    size=self.image_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            frame = frame * 2.0 - 1.0
            gt_last_frame = gt_last_frame * 2.0 - 1.0
            frame_batch.append(frame)
            gt_last_frame_batch.append(gt_last_frame)
            task_descriptions.append(self._extract_task_description(episode_meta, ann))

        frame_tensor = torch.stack(frame_batch, dim=0).to(self.device)
        self.current_obs = frame_tensor.unsqueeze(2).unsqueeze(3)
        self.current_state = torch.from_numpy(np.stack(state_batch, axis=0)).to(
            self.device, dtype=torch.float32
        )
        self.gt_last_frames = torch.stack(gt_last_frame_batch, dim=0).to(self.device)
        self.task_descriptions = task_descriptions

        frame_uint8 = ((frame_tensor.clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0)
        frame_uint8 = frame_uint8.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
        self.current_latent = self._encode_frames_to_latent(frame_uint8).unsqueeze(1)

        self._reset_metrics()
        return self._wrap_obs(), {}

    @torch.no_grad()
    def step(self, actions=None, auto_reset=True):
        raise NotImplementedError("step in IRASIMEnv is not implemented, use chunk_step")

    def _infer_next_chunk_rewards(self):
        if self.reward_model is None:
            raise ValueError("Reward model is not loaded")

        extract_chunk_obs = self.current_obs.permute(0, 3, 1, 2, 4, 5)
        extract_chunk_obs = extract_chunk_obs[:, -self.chunk :, :, :, :, :]
        extract_chunk_obs = extract_chunk_obs.reshape(self.num_envs * self.chunk, 3, 1, *self.image_size)
        extract_chunk_obs = extract_chunk_obs.squeeze(2).to(self.device)

        reward_type = self.cfg.reward_model.type
        if reward_type == "ResnetRewModel":
            rewards = self.reward_model.predict_rew(extract_chunk_obs).reshape(self.num_envs, self.chunk)
        elif reward_type == "TaskEmbedResnetRewModel":
            instructions = []
            for env_idx in range(self.num_envs):
                instructions.extend([self.task_descriptions[env_idx]] * self.chunk)
            rewards = self.reward_model.predict_rew(extract_chunk_obs, instructions).reshape(self.num_envs, self.chunk)
        elif reward_type == "RoboTwinT5CrossAttn":
            extract_chunk_obs_float = ((extract_chunk_obs + 1.0) / 2.0).float()
            instructions = []
            for env_idx in range(self.num_envs):
                instructions.extend([self.task_descriptions[env_idx]] * self.chunk)
            rewards = self.reward_model.compute_reward(
                extract_chunk_obs_float,
                task_descriptions=instructions,
            ).reshape(self.num_envs, self.chunk)
        elif reward_type == "QwenVLMProgressRewardModel":
            instructions = []
            for env_idx in range(self.num_envs):
                instructions.extend([self.task_descriptions[env_idx]] * self.chunk)
            rewards = self.reward_model.compute_reward(
                extract_chunk_obs,
                task_descriptions=instructions,
            ).reshape(self.num_envs, self.chunk)
        elif reward_type == "LPIPSLastFrameRewardModel":
            if self.gt_last_frames is None:
                raise ValueError(
                    "GT last frames are not initialized; call reset() first."
                )
            reference_frames = (
                self.gt_last_frames.unsqueeze(1)
                .repeat(1, self.chunk, 1, 1, 1)
                .reshape(self.num_envs * self.chunk, 3, *self.image_size)
                .float()
            )
            rewards = self.reward_model.compute_reward(
                extract_chunk_obs.float(),
                references=reference_frames,
            ).reshape(self.num_envs, self.chunk)
        else:
            raise ValueError(f"Unknown reward model type: {reward_type}")
        return rewards

    @torch.no_grad()
    def _infer_next_chunk_frames(self, actions):
        actions_np = actions.cpu().numpy() if isinstance(actions, torch.Tensor) else np.asarray(actions)
        if actions_np.shape != (self.num_envs, self.chunk, self.action_dim):
            raise ValueError(
                f"Unexpected actions shape {actions_np.shape}, expected {(self.num_envs, self.chunk, self.action_dim)}"
            )

        if self.action_key == "abs_action":
            self.current_state = torch.from_numpy(actions_np[:, -1, :]).to(
                self.device, dtype=torch.float32
            )
        elif self.action_key == "delta_action" and self.current_state is not None:
            self.current_state = self.current_state + torch.from_numpy(
                actions_np[:, -1, :]
            ).to(self.device, dtype=torch.float32)

        actions_norm = self._normalize_actions(actions_np)
        actions_tensor = torch.from_numpy(actions_norm).to(self.device).float()

        if self.current_latent is None:
            last_frame = self.current_obs[:, :, 0, -1, :, :]
            last_frame_uint8 = ((last_frame.clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0)
            last_frame_uint8 = last_frame_uint8.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
            self.current_latent = self._encode_frames_to_latent(last_frame_uint8).unsqueeze(1)

        pred_videos, pred_latents = self.pipeline(
            actions_tensor,
            mask_x=self.current_latent,
            video_length=self.num_frames,
            height=self.image_size[0],
            width=self.image_size[1],
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            device=self.device,
            return_dict=False,
            output_type="both",
        )

        new_frames = pred_videos[:, self.condition_frame_length :, :, :, :]
        new_frames = new_frames.permute(0, 2, 1, 3, 4).contiguous()
        new_frames = new_frames.unsqueeze(2)

        self.current_obs = torch.cat([self.current_obs, new_frames], dim=3)
        max_frames = self.condition_frame_length + self.chunk
        if self.current_obs.shape[3] > max_frames:
            self.current_obs = self.current_obs[:, :, :, -max_frames:, :, :]

        if self.enable_latent_cache:
            self.current_latent = pred_latents[:, -1:, :, :, :].contiguous()
        else:
            last_frame = self.current_obs[:, :, 0, -1, :, :]
            last_frame_uint8 = ((last_frame.clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0)
            last_frame_uint8 = last_frame_uint8.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
            self.current_latent = self._encode_frames_to_latent(last_frame_uint8).unsqueeze(1)

    def _build_obs_from_frame(self, frame_tensor: torch.Tensor):
        full_image = frame_tensor.permute(0, 2, 3, 1)
        full_image = (full_image + 1.0) / 2.0 * 255.0
        full_image = torch.clamp(full_image, 0, 255).to(torch.uint8)

        states = self.current_state
        if states is None:
            states = torch.zeros(
                (self.num_envs, self.action_dim), device=self.device, dtype=torch.float32
            )
        return {
            "main_images": full_image,
            "wrist_images": None,
            "states": states,
            "task_descriptions": self.task_descriptions,
        }

    def _wrap_obs(self):
        last_frame = self.current_obs[:, :, 0, -1, :, :]
        return self._build_obs_from_frame(last_frame)

    def _wrap_chunk_obs_list(self):
        chunk_frames = self.current_obs[:, :, 0, -self.chunk :, :, :]
        obs_list = []
        for frame_idx in range(chunk_frames.shape[2]):
            obs_list.append(self._build_obs_from_frame(chunk_frames[:, :, frame_idx, :, :]))
        return obs_list

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        final_obs = extracted_obs
        final_info = infos
        extracted_obs, infos = self.reset()
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    @torch.no_grad()
    def chunk_step(self, policy_output_action):
        self.onload()
        self._infer_next_chunk_frames(policy_output_action)
        self.elapsed_steps += self.chunk

        chunk_obs_list = self._wrap_chunk_obs_list()
        extracted_obs = chunk_obs_list[-1]
        chunk_rewards = self._infer_next_chunk_rewards()
        chunk_rewards_tensors = self._calc_step_reward(chunk_rewards)

        if getattr(self.cfg, "print_chunk_rewards", False):
            raw = chunk_rewards.detach().cpu().numpy()
            diff = chunk_rewards_tensors.detach().cpu().numpy()
            for env_idx in range(self.num_envs):
                raw_str = ", ".join([f"{v:.4f}" for v in raw[env_idx]])
                diff_str = ", ".join([f"{v:.4f}" for v in diff[env_idx]])
                print(
                    f"[chunk={self.elapsed_steps // self.chunk}] env{env_idx} raw=[{raw_str}] diff=[{diff_str}]"
                )

        estimated_success = self._estimate_success_from_rewards(chunk_rewards)
        raw_chunk_terminations = torch.zeros(self.num_envs, self.chunk, dtype=torch.bool, device=self.device)
        raw_chunk_terminations[:, -1] = estimated_success

        raw_chunk_truncations = torch.zeros(self.num_envs, self.chunk, dtype=torch.bool, device=self.device)
        if self.elapsed_steps >= int(self.cfg.max_episode_steps):
            raw_chunk_truncations[:, -1] = True

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            reset_obs, infos = self._handle_auto_reset(past_dones, extracted_obs, {})
            returned_obs_list = list(chunk_obs_list)
            returned_obs_list[-1] = reset_obs
            extracted_obs = reset_obs
        else:
            infos = {}
            returned_obs_list = chunk_obs_list

        if self.record_metrics:
            infos = self._record_metrics(chunk_rewards_tensors.sum(dim=1), past_terminations, infos)

        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations
        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations

        infos_list = [{} for _ in range(max(0, self.chunk - 1))] + [infos]

        return (
            returned_obs_list,
            chunk_rewards_tensors,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def offload(self):
        if self._is_offloaded:
            return
        self.irasim_model = self.irasim_model.to("cpu")
        self.vae = self.vae.to("cpu")
        self.reward_model = self.reward_model.to("cpu")
        self.current_obs = recursive_to_device(self.current_obs, "cpu")
        self.current_latent = recursive_to_device(self.current_latent, "cpu")
        self.current_state = recursive_to_device(self.current_state, "cpu")
        self.gt_last_frames = recursive_to_device(self.gt_last_frames, "cpu")
        self.prev_step_reward = self.prev_step_reward.cpu()
        self.reset_state_ids = self.reset_state_ids.cpu()
        if self.record_metrics:
            self.success_once = self.success_once.cpu()
            self.returns = self.returns.cpu()
        torch.cuda.empty_cache()
        self._is_offloaded = True

    def onload(self):
        if not self._is_offloaded:
            return
        self.irasim_model = self.irasim_model.to(self.device)
        self.vae = self.vae.to(self.device)
        self.reward_model = self.reward_model.to(self.device)
        self.current_obs = recursive_to_device(self.current_obs, self.device)
        self.current_latent = recursive_to_device(self.current_latent, self.device)
        self.current_state = recursive_to_device(self.current_state, self.device)
        self.gt_last_frames = recursive_to_device(self.gt_last_frames, self.device)
        self.prev_step_reward = self.prev_step_reward.to(self.device)
        self.reset_state_ids = self.reset_state_ids.to(self.device)
        if self.record_metrics:
            self.success_once = self.success_once.to(self.device)
            self.returns = self.returns.to(self.device)
        torch.cuda.empty_cache()
        self._is_offloaded = False
