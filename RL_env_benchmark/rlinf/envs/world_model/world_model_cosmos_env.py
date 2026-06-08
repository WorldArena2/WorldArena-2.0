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

"""Cosmos Predict2.5 world-model environment for RLinf."""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from omegaconf import DictConfig

from rlinf.envs.utils import recursive_to_device
from rlinf.envs.world_model.base_world_env import BaseWorldEnv

__all__ = ["CosmosEnv"]


class CosmosEnv(BaseWorldEnv):
    """World-model environment backed by Cosmos Predict2.5 action-conditioned model."""

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
                "num_frames must be equal to condition_frame_length + chunk, "
                f"got {self.num_frames}, {self.condition_frame_length}, {self.chunk}"
            )
        self.action_dim = int(cfg.action_dim)
        self.image_size = tuple(cfg.image_size)
        self.wm_env_type = cfg.get("wm_env_type", "robotwin")
        self.inference_backend = cfg.get("inference_backend", "local")
        if self.inference_backend not in {"local", "rpc"}:
            raise ValueError(
                f"Unsupported inference_backend={self.inference_backend}, expected local|rpc"
            )
        self._annotation_cache: dict[str, dict[str, Any]] = {}
        rpc_cfg = cfg.get("rpc", {})
        self._rpc_url = (
            str(rpc_cfg.get("url"))
            if rpc_cfg.get("url", None) is not None
            else cfg.get("rpc_url", None)
        )
        self._rpc_timeout_s = float(
            rpc_cfg.get("timeout_s", cfg.get("rpc_timeout_s", 120.0))
        )
        self._NUM_CONDITIONAL_FRAMES_KEY = None

        super().__init__(
            cfg, num_envs, seed_offset, total_num_processes, worker_info, record_metrics
        )

        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.group_size = cfg.group_size
        self.num_group = self.num_envs // self.group_size

        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

        self.model = None
        if self.inference_backend == "local":
            self._setup_cosmos_import_path()
            self._load_cosmos_interfaces()
            self.model = self._load_world_model().eval()
        else:
            if self._rpc_url is None:
                raise ValueError("rpc.url must be set when inference_backend=rpc")
            self._rpc_url = self._rpc_url.rstrip("/")
        self.reward_model = self._load_reward_model().eval().to(self.device)

        self.action_stats = (
            self._load_action_stats(cfg.stat_path)
            if self.inference_backend == "local"
            else None
        )

        self.current_obs = None
        self.task_descriptions = [""] * self.num_envs
        self.annotation_files = [""] * self.num_envs

        self.trans_norm = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                )
            ]
        )
        self._is_offloaded = False

    def _setup_cosmos_import_path(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        cosmos_root = self.cfg.get("cosmos_root", str(repo_root / "cosmos-predict2.5"))
        if cosmos_root not in sys.path:
            sys.path.insert(0, cosmos_root)

    def _load_cosmos_interfaces(self) -> None:
        try:
            from cosmos_predict2._src.predict2.action.datasets.dataset_local import (
                Dataset_3D,
            )
            from cosmos_predict2._src.predict2.action.models.action_conditioned_video2world_rectified_flow_model import (
                NUM_CONDITIONAL_FRAMES_KEY,
            )
            from cosmos_predict2._src.predict2.utils.model_loader import (
                load_model_from_checkpoint,
            )
        except ImportError as exc:
            raise ImportError(
                "Failed to import Cosmos modules. Ensure cosmos-predict2.5 exists and "
                "dependencies are installed."
            ) from exc

        self._Dataset3D = Dataset_3D
        self._NUM_CONDITIONAL_FRAMES_KEY = NUM_CONDITIONAL_FRAMES_KEY
        self._load_model_from_checkpoint = load_model_from_checkpoint

    def _build_dataset(self, cfg):
        if self.inference_backend == "rpc":
            dataset_size = self._rpc_get_dataset_size()
            return list(range(dataset_size))

        action_scaler = cfg.get("action_scaler", None)
        if action_scaler is None:
            action_scaler = [20.0] * self.chunk

        dataset = self._Dataset3D(
            train_annotation_path=cfg.train_annotation_path,
            val_annotation_path=cfg.val_annotation_path,
            test_annotation_path=cfg.get("test_annotation_path", cfg.val_annotation_path),
            video_path=cfg.video_base_path,
            fps_downsample_ratio=cfg.get("fps_downsample_ratio", 1),
            num_action_per_chunk=cfg.get("num_action_per_chunk", self.chunk),
            cam_ids=cfg.get("cam_ids", [0]),
            accumulate_action=cfg.get("accumulate_action", False),
            video_size=cfg.get("video_size", [240, 320]),
            val_start_frame_interval=cfg.get("val_start_frame_interval", 1),
            mode=cfg.get("data_mode", "val"),
            control_mode=cfg.get("control_mode", "joint"),
            action_dim=self.action_dim,
            gripper_dim=cfg.get("gripper_dim", 2),
            action_scaler=action_scaler,
            action_mode=cfg.get("action_mode", "absolute"),
            action_norm_type=cfg.get("action_norm_type", "stat"),
            action_stat_path=cfg.stat_path,
            gripper_rescale_factor=cfg.get("gripper_rescale_factor", 1.0),
            state_key=cfg.get("state_key", "state"),
            gripper_key=cfg.get("gripper_key", None),
            device=str(self.device),
        )
        return dataset

    def _load_world_model(self):
        model, _config = self._load_model_from_checkpoint(
            experiment_name=self.cfg.world_model.experiment,
            s3_checkpoint_dir=self.cfg.world_model.ckpt_path,
            config_file=self.cfg.world_model.config_file,
            load_ema_to_reg=self.cfg.world_model.get("load_ema_to_reg", True),
            to_device=str(self.device),
        )
        return model

    def _encode_array_b64(self, arr: np.ndarray) -> str:
        buffer = io.BytesIO()
        np.save(buffer, arr, allow_pickle=False)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _decode_array_b64(self, payload: str) -> np.ndarray:
        raw = base64.b64decode(payload.encode("ascii"))
        buffer = io.BytesIO(raw)
        return np.load(buffer, allow_pickle=False)

    def _rpc_request(self, method: str, endpoint: str, payload: Optional[dict] = None):
        if self._rpc_url is None:
            raise ValueError("RPC url is not configured for cosmos rpc backend")
        url = f"{self._rpc_url}{endpoint}"
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url=url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self._rpc_timeout_s) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"Cosmos RPC request failed {method} {endpoint}: HTTP {exc.code} {error_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cosmos RPC request failed {method} {endpoint}: {exc}"
            ) from exc

    def _rpc_get_dataset_size(self) -> int:
        try:
            resp = self._rpc_request("GET", "/dataset_size")
            return int(resp["dataset_size"])
        except Exception:
            fallback = int(self.cfg.rpc.get("dataset_size_fallback", 10000))
            return fallback

    def _rpc_reset_samples(
        self, episode_indices: Optional[np.ndarray]
    ) -> tuple[np.ndarray, list[str], list[str]]:
        payload = {"num_envs": self.num_envs}
        if episode_indices is not None:
            payload["episode_indices"] = [int(x) for x in episode_indices.tolist()]
        resp = self._rpc_request("POST", "/reset_samples", payload)
        frames = self._decode_array_b64(resp["initial_frames_b64"])
        tasks = [str(x) for x in resp["task_descriptions"]]
        ann_files = [str(x) for x in resp["annotation_files"]]
        return frames, tasks, ann_files

    def _rpc_infer_chunk(self, frames_uint8: np.ndarray, actions: np.ndarray) -> np.ndarray:
        payload = {
            "frames_b64": self._encode_array_b64(frames_uint8),
            "actions_b64": self._encode_array_b64(actions.astype(np.float32)),
            "annotation_files": self.annotation_files,
        }
        resp = self._rpc_request("POST", "/infer_chunk", payload)
        return self._decode_array_b64(resp["pred_frames_b64"])

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
            from rlinf.models.embodiment.reward import QwenVLMProgressRewardModel

            rew_model = QwenVLMProgressRewardModel(DictConfig(self.cfg.reward_model))
        else:
            raise ValueError(f"Unknown reward model type: {reward_type}")
        return rew_model

    def _load_action_stats(self, stats_path: str) -> dict[str, np.ndarray]:
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Action stats path does not exist: {stats_path}")
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

        if "action" in stats:
            action_stats = stats["action"]
            q01 = np.asarray(action_stats["q01"], dtype=np.float32)
            q99 = np.asarray(action_stats["q99"], dtype=np.float32)
        elif "state_01" in stats and "state_99" in stats:
            q01 = np.asarray(stats["state_01"], dtype=np.float32)
            q99 = np.asarray(stats["state_99"], dtype=np.float32)
        else:
            first_key = next(iter(stats))
            action_stats = stats[first_key]["action"]
            q01 = np.asarray(action_stats["q01"], dtype=np.float32)
            q99 = np.asarray(action_stats["q99"], dtype=np.float32)

        if q01.shape[0] != self.action_dim or q99.shape[0] != self.action_dim:
            raise ValueError(
                f"Action stats dim mismatch: expected {self.action_dim}, got "
                f"{q01.shape[0]} and {q99.shape[0]}"
            )
        return {"q01": q01, "q99": q99}

    def _normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        if self.action_stats is None:
            return actions.astype(np.float32)
        q01 = self.action_stats["q01"][None, None, :]
        q99 = self.action_stats["q99"][None, None, :]
        actions_norm = 2.0 * ((actions - q01) / (q99 - q01 + 1e-6)) - 1.0
        return np.clip(actions_norm, -1.0, 1.0)

    def _load_label_from_sample(self, sample_idx: int) -> dict[str, Any]:
        sample_meta = self.dataset.samples[sample_idx]
        ann_file = sample_meta["ann_file"]
        if ann_file not in self._annotation_cache:
            with open(ann_file, "r", encoding="utf-8") as f:
                self._annotation_cache[ann_file] = json.load(f)
        return self._annotation_cache[ann_file]

    def _extract_task_description(self, sample_idx: int) -> str:
        label = self._load_label_from_sample(sample_idx)
        if isinstance(label.get("task_name"), str):
            return label["task_name"]
        texts = label.get("texts")
        if isinstance(texts, list) and texts and isinstance(texts[0], str):
            return texts[0]
        return "click the bell"

    def _get_annotation_file(self, sample_idx: int) -> str:
        return self.dataset.samples[sample_idx]["ann_file"]

    def _batchify(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if self._NUM_CONDITIONAL_FRAMES_KEY is None:
            raise ValueError("NUM_CONDITIONAL_FRAMES_KEY is not available in rpc mode")
        batch: dict[str, Any] = {}
        for key in samples[0]:
            values = [sample[key] for sample in samples]
            if isinstance(values[0], torch.Tensor):
                batch[key] = torch.stack(values, dim=0)
            elif isinstance(values[0], (int, float)):
                batch[key] = torch.tensor(values)
            else:
                batch[key] = values
        batch[self._NUM_CONDITIONAL_FRAMES_KEY] = int(
            self.cfg.world_model.get("num_conditional_frames", 1)
        )
        return batch

    def _move_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        if self.model is None:
            return batch
        for key, value in list(batch.items()):
            if not isinstance(value, torch.Tensor):
                continue
            if key == "video":
                batch[key] = value.to(device=self.device)
            else:
                batch[key] = value.to(**self.model.tensor_kwargs)
        return batch

    def _maybe_compute_text_embeddings(self, batch: dict[str, Any]) -> None:
        if self.model is None:
            return
        text_encoder_config = getattr(self.model.config, "text_encoder_config", None)
        if text_encoder_config is None:
            return
        if not getattr(text_encoder_config, "compute_online", False):
            return
        text_embeddings = self.model.text_encoder.compute_text_embeddings_online(
            batch, self.model.input_caption_key
        )
        batch["t5_text_embeddings"] = text_embeddings
        batch["t5_text_mask"] = torch.ones(
            text_embeddings.shape[0],
            text_embeddings.shape[1],
            device=text_embeddings.device,
        )

    def _build_video_from_frame(self, frame_uint8: torch.Tensor) -> torch.Tensor:
        c, h, w = frame_uint8.shape
        video = torch.zeros((c, self.num_frames, h, w), dtype=torch.uint8)
        video[:, 0] = frame_uint8
        return video

    def _build_data_from_frame_actions(
        self,
        frame_uint8: torch.Tensor,
        actions_norm: torch.Tensor,
        annotation_file: str,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "action": actions_norm.float(),
            "video": self._build_video_from_frame(frame_uint8),
            "annotation_file": annotation_file,
            "t5_text_embeddings": torch.zeros(512, 1024, dtype=torch.bfloat16),
            "ai_caption": "",
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 4,
            "image_size": 256 * torch.ones(4),
            "num_frames": self.num_frames,
            "padding_mask": torch.zeros(1, 256, 256),
        }
        return data

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            if self.record_metrics:
                self.success_once[mask] = False
                self.returns[mask] = 0
        else:
            self.prev_step_reward[:] = 0
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
            terminations_tensor = torch.tensor(
                terminations, device=self.device, dtype=torch.bool
            )
            self.success_once = self.success_once | terminations_tensor
        episode_info["success_once"] = self.success_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = torch.full(
            (self.num_envs,),
            self.elapsed_steps,
            dtype=torch.float32,
            device=self.device,
        )
        episode_info["reward"] = episode_info["return"] / episode_info[
            "episode_len"
        ].clamp(min=1.0)
        infos["episode"] = episode_info
        return infos

    def _calc_step_reward(self, chunk_rewards):
        reward_diffs = torch.zeros(
            (self.num_envs, self.chunk), dtype=torch.float32, device=self.device
        )
        for i in range(self.chunk):
            reward_diffs[:, i] = (
                self.cfg.reward_coef * chunk_rewards[:, i] - self.prev_step_reward
            )
            self.prev_step_reward = self.cfg.reward_coef * chunk_rewards[:, i]
        if self.use_rel_reward:
            return reward_diffs
        return chunk_rewards

    def _estimate_success_from_rewards(self, chunk_rewards):
        success_threshold = getattr(self.cfg, "success_reward_threshold", 0.9)
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
        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        ).to(self.device)

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
            episode_indices = np.random.choice(
                len(self.dataset), size=self.num_envs, replace=replace
            )
        elif isinstance(episode_indices, torch.Tensor):
            episode_indices = episode_indices.cpu().numpy()

        if self.inference_backend == "rpc":
            frames_uint8, task_descriptions, annotation_files = self._rpc_reset_samples(
                episode_indices
            )
            if frames_uint8.shape[0] != self.num_envs:
                raise ValueError(
                    f"RPC reset returned {frames_uint8.shape[0]} envs, expected {self.num_envs}"
                )
            frame = torch.from_numpy(frames_uint8).to(self.device).float() / 255.0
            frame = frame.permute(0, 3, 1, 2)
            if frame.shape[-2:] != self.image_size:
                frame = F.interpolate(
                    frame,
                    size=self.image_size,
                    mode="bilinear",
                    align_corners=False,
                )
            frame = self.trans_norm(frame)
            self.current_obs = frame.unsqueeze(2).unsqueeze(3)
            self.task_descriptions = task_descriptions
            self.annotation_files = annotation_files
            self._reset_metrics()
            return self._wrap_obs(), {}

        img_tensors = []
        task_descriptions = []
        annotation_files = []
        for env_idx, episode_idx in enumerate(episode_indices):
            del env_idx
            sample_idx = int(episode_idx)
            sample = self.dataset[sample_idx]
            first_frame = sample["video"][:, 0, :, :].float()
            if first_frame.max() > 1.0:
                first_frame = first_frame / 255.0
            if first_frame.shape[1:] != self.image_size:
                first_frame = F.interpolate(
                    first_frame.unsqueeze(0),
                    size=self.image_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            first_frame = self.trans_norm(first_frame)
            img_tensors.append(first_frame)
            task_descriptions.append(self._extract_task_description(sample_idx))
            annotation_files.append(self._get_annotation_file(sample_idx))

        stacked = torch.stack(img_tensors, dim=0).to(self.device)
        self.current_obs = stacked.unsqueeze(2).unsqueeze(3)
        self.task_descriptions = task_descriptions
        self.annotation_files = annotation_files
        self._reset_metrics()

        return self._wrap_obs(), {}

    @torch.no_grad()
    def step(self, actions=None, auto_reset=True):
        raise NotImplementedError("step in CosmosEnv is not implemented, use chunk_step")

    @torch.no_grad()
    def _infer_next_chunk_frames(self, actions: np.ndarray | torch.Tensor) -> None:
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = actions
        if actions_np.shape != (self.num_envs, self.chunk, self.action_dim):
            raise ValueError(
                f"Expected actions shape ({self.num_envs}, {self.chunk}, {self.action_dim}), "
                f"got {actions_np.shape}"
            )
        current_frame = self.current_obs[:, :, 0, -1, :, :]
        frame_uint8 = ((current_frame + 1.0) / 2.0 * 255.0).clamp(0, 255).to(torch.uint8)

        if self.inference_backend == "rpc":
            pred_uint8 = self._rpc_infer_chunk(
                frame_uint8.detach().cpu().numpy(),
                actions_np,
            )
            if pred_uint8.shape[1] < self.chunk:
                raise ValueError(
                    f"RPC predicted temporal length {pred_uint8.shape[1]} < chunk {self.chunk}"
                )
            pred = torch.from_numpy(pred_uint8).to(self.device).float() / 255.0
            pred = pred.permute(0, 4, 1, 2, 3)  # [B,3,T,H,W]
            pred = pred * 2.0 - 1.0
            new_frames = pred[:, :, -self.chunk :, :, :].unsqueeze(2)
            self.current_obs = torch.cat([self.current_obs, new_frames], dim=3)
            max_keep = self.condition_frame_length + self.chunk
            if self.current_obs.shape[3] > max_keep:
                self.current_obs = self.current_obs[:, :, :, -max_keep:, :, :]
            return

        actions_norm_np = self._normalize_actions(actions_np).astype(np.float32)
        actions_norm = torch.from_numpy(actions_norm_np)

        samples = []
        for env_idx in range(self.num_envs):
            sample = self._build_data_from_frame_actions(
                frame_uint8[env_idx],
                actions_norm[env_idx],
                self.annotation_files[env_idx],
            )
            samples.append(sample)

        batch = self._batchify(samples)
        batch = self._move_to_device(batch)
        self._maybe_compute_text_embeddings(batch)

        sample_latents = self.model.generate_samples_from_batch(
            batch,
            guidance=self.cfg.world_model.get("guidance", 0.0),
            seed=self.seed + self.elapsed_steps,
            num_steps=self.cfg.world_model.get("num_steps", 35),
            is_negative_prompt=False,
        )
        video = (
            self.model.decode(sample_latents)
            if hasattr(self.model, "decode")
            else sample_latents
        )
        video = video.float()
        if video.max() <= 1.0 and video.min() >= 0.0:
            video = video * 2.0 - 1.0
        else:
            video = video.clamp(-1.0, 1.0)

        if video.shape[2] < self.chunk:
            raise ValueError(
                f"Decoded video temporal length {video.shape[2]} is smaller than chunk {self.chunk}"
            )
        new_frames = video[:, :, -self.chunk :, :, :]
        new_frames = new_frames.unsqueeze(2)
        self.current_obs = torch.cat([self.current_obs, new_frames], dim=3)

        max_keep = self.condition_frame_length + self.chunk
        if self.current_obs.shape[3] > max_keep:
            self.current_obs = self.current_obs[:, :, :, -max_keep:, :, :]

    @torch.no_grad()
    def _infer_next_chunk_rewards(self):
        if self.reward_model is None:
            raise ValueError("Reward model is not loaded")

        frames = self.current_obs[:, :, 0, -self.chunk :, :, :]
        bsz, c, chunk, h, w = frames.shape
        frames_flat = frames.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        frames_flat = frames_flat.to(self.device)

        reward_type = self.cfg.reward_model.type
        if reward_type == "ResnetRewModel":
            rewards = self.reward_model.predict_rew(frames_flat)
        elif reward_type == "TaskEmbedResnetRewModel":
            instructions = []
            for env_idx in range(self.num_envs):
                instructions.extend([self.task_descriptions[env_idx]] * self.chunk)
            rewards = self.reward_model.predict_rew(frames_flat, instructions)
        elif reward_type == "RoboTwinT5CrossAttn":
            frames_01 = ((frames_flat + 1.0) / 2.0).float()
            instructions = []
            for env_idx in range(self.num_envs):
                instructions.extend([self.task_descriptions[env_idx]] * self.chunk)
            rewards = self.reward_model.compute_reward(
                frames_01, task_descriptions=instructions
            )
        elif reward_type == "QwenVLMProgressRewardModel":
            frames_flat = ((frames_flat + 1.0) / 2.0).clamp(0.0, 1.0)
            instructions = []
            for env_idx in range(self.num_envs):
                instructions.extend([self.task_descriptions[env_idx]] * self.chunk)
            rewards = self.reward_model.compute_reward(
                frames_flat, task_descriptions=instructions
            )
        else:
            raise ValueError(f"Unknown reward model type: {reward_type}")

        rewards = rewards.reshape(bsz, chunk)
        return rewards

    def _wrap_obs(self):
        last_frame = self.current_obs[:, :, 0, -1, :, :]
        full_image = last_frame.permute(0, 2, 3, 1)
        full_image = (full_image + 1.0) / 2.0 * 255.0
        full_image = torch.clamp(full_image, 0, 255).to(torch.uint8)
        states = torch.zeros(
            (self.num_envs, self.action_dim), device=self.device, dtype=torch.float32
        )
        return {
            "main_images": full_image,
            "wrist_images": None,
            "states": states,
            "task_descriptions": self.task_descriptions,
        }

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

        extracted_obs = self._wrap_obs()
        chunk_rewards = self._infer_next_chunk_rewards()
        chunk_rewards_tensors = self._calc_step_reward(chunk_rewards)

        # Debug: print per-chunk, per-step rewards (aligned with other world model envs)
        if getattr(self.cfg, "print_chunk_rewards", False):
            with torch.no_grad():
                # chunk_rewards: [num_envs, chunk], chunk_rewards_tensors: [num_envs, chunk]
                raw = chunk_rewards.cpu().numpy()  # raw reward model output
                diff = chunk_rewards_tensors.cpu().numpy()  # step / relative reward
                for env_idx in range(self.num_envs):
                    raw_str = ", ".join([f"{v:.4f}" for v in raw[env_idx]])
                    diff_str = ", ".join([f"{v:.4f}" for v in diff[env_idx]])
                    print(
                        f"[chunk={self.elapsed_steps // self.chunk}] env{env_idx} "
                        f"raw=[{raw_str}] diff=[{diff_str}]"
                    )

        estimated_success = self._estimate_success_from_rewards(chunk_rewards)
        raw_chunk_terminations = torch.zeros(
            self.num_envs, self.chunk, dtype=torch.bool, device=self.device
        )
        raw_chunk_terminations[:, -1] = estimated_success

        raw_chunk_truncations = torch.zeros(
            self.num_envs, self.chunk, dtype=torch.bool, device=self.device
        )
        is_truncated = self.elapsed_steps >= self.cfg.max_episode_steps
        if is_truncated:
            raw_chunk_truncations[:, -1] = True

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(
                past_dones, extracted_obs, {}
            )
        else:
            infos = {}

        infos = self._record_metrics(
            chunk_rewards_tensors.sum(dim=1), past_terminations, infos
        )
        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations
        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations

        return (
            [extracted_obs],
            chunk_rewards_tensors,
            chunk_terminations,
            chunk_truncations,
            [infos],
        )

    def offload(self):
        if self._is_offloaded:
            return
        if self.model is not None:
            self.model = self.model.to("cpu")
        self.reward_model = self.reward_model.to("cpu")
        self.current_obs = recursive_to_device(self.current_obs, "cpu")
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
        if self.model is not None:
            self.model = self.model.to(self.device)
        self.reward_model = self.reward_model.to(self.device)
        self.current_obs = recursive_to_device(self.current_obs, self.device)
        self.prev_step_reward = self.prev_step_reward.to(self.device)
        self.reset_state_ids = self.reset_state_ids.to(self.device)
        if self.record_metrics:
            self.success_once = self.success_once.to(self.device)
            self.returns = self.returns.to(self.device)
        self._is_offloaded = False


if __name__ == "__main__":
    from hydra import compose
    from hydra.core.global_hydra import GlobalHydra
    from hydra.initialize import initialize_config_dir

    os.environ.setdefault("EMBODIED_PATH", "examples/embodiment")
    repo_root = Path(__file__).resolve().parents[3]
    config_dir = Path(
        os.environ.get("EMBODIED_CONFIG_DIR", repo_root / "examples/embodiment/config")
    ).resolve()
    config_name = os.environ.get(
        "EMBODIED_CONFIG_NAME", "cosmos_robotwin_click_bell_grpo_openpi_pi05"
    )

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg_ = compose(config_name=config_name)
        cfg = cfg_["env"]["train"]

    print(
        f"[CosmosEnv smoke-test] config={config_name}, "
        f"backend={cfg.get('inference_backend', 'local')}"
    )
    env = CosmosEnv(
        cfg,
        num_envs=cfg.total_num_envs,
        seed_offset=0,
        total_num_processes=1,
    )
    obs, info = env.reset()
    print("[reset] obs keys:", list(obs.keys()), "info keys:", list(info.keys()))
    print(
        "[reset] main_images:",
        tuple(obs["main_images"].shape),
        obs["main_images"].dtype,
    )

    zeros_actions = np.zeros(
        (cfg.total_num_envs, cfg.chunk, cfg.action_dim), dtype=np.float32
    )
    out = env.chunk_step(zeros_actions)
    obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos = out
    print("[chunk_step] obs_list_len:", len(obs_list), "info_list_len:", len(infos))
    print(
        "[chunk_step] rewards:",
        tuple(chunk_rewards.shape),
        chunk_rewards.dtype,
    )
    print(
        "[chunk_step] terminations:",
        tuple(chunk_terminations.shape),
        "truncations:",
        tuple(chunk_truncations.shape),
    )
