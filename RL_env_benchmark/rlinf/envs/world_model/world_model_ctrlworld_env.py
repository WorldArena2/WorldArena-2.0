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

"""Ctrl-World (SVD-based) world model environment for RLinf."""

from __future__ import annotations

import collections
import json
import os
import sys
import types
from typing import Optional, Union

import einops
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from rlinf.envs.world_model.base_world_env import BaseWorldEnv

__all__ = ["CtrlWorldEnv"]


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------


class CtrlWorldDataset:
    """Dataset wrapper for the annotation format produced by extract_latent_robotwin.py.

    Directory structure::

        dataset_root/
          annotation/{data_type}/{traj_id}.json
          latent_videos/{data_type}/{traj_id}/0.pt   # SVD-VAE latent (T, 4, 24, 40)
          videos/{data_type}/{traj_id}/0.mp4

    Each annotation JSON contains::

        {
            "texts": ["task description"],
            "video_length": 50,
            "videos": [{"video_path": "..."}],
            "latent_videos": [{"latent_video_path": "..."}],
            "states": [[...], ...]   # 14-dim reordered actions
        }
    """

    def __init__(self, dataset_root: str, data_type: str = "val"):
        import glob

        self.dataset_root = dataset_root
        self.data_type = data_type
        ann_dir = os.path.join(dataset_root, "annotation", data_type)
        self.ann_files = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
        if len(self.ann_files) == 0:
            raise FileNotFoundError(
                f"No annotation JSON files found in {ann_dir}. "
                "Please run extract_latent_robotwin.py first."
            )

    def __len__(self) -> int:
        return len(self.ann_files)

    def __getitem__(self, idx: int) -> dict:
        with open(self.ann_files[idx]) as f:
            ann = json.load(f)
        return ann


# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------


class CtrlWorldEnv(BaseWorldEnv):
    """World-model environment backed by Ctrl-World (SVD + Action_encoder2).

    Key differences from WanEnv / OpenSoraEnv:

    * ``image_queue`` stores VAE **latents** of shape ``(1, 4, 72, 40)``
      (72 = 24×3, three camera views stacked along H; for RobotWin the three
      views are identical head-cam latents).
    * ``chunk = 5`` (``num_frames`` in Ctrl-World), NOT 8 like Wan/OpenSora.
    * History sampling uses ``history_idx = [0, 0, -8, -6, -4, -2]``
      (non-uniform), so a deque long enough to hold at least 9 frames is kept.
    * Actions are normalized with p01/p99 bound normalization → [-1, 1] before
      being fed to the Action_encoder2 MLP.
    * The model is loaded via ``sys.path.insert`` from ``cfg.ctrlworld_repo_path``.
    """

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        record_metrics: bool = True,
        worker_info=None,
    ):
        # ---------- hyper-parameters needed by _build_dataset (called in super()) ----------
        self.chunk: int = cfg.chunk                          # = 5 (num_frames)
        self.num_history: int = cfg.num_history              # = 6
        self.image_size: tuple = tuple(cfg.image_size)       # = (256, 256)
        self.action_dim: int = cfg.get("action_dim", 14)
        self.wm_env_type: str = cfg.get("wm_env_type", "robotwin")
        # OpenPI/Aloha commonly outputs 14-dim action in:
        #   [L_joint(6), L_gripper(1), R_joint(6), R_gripper(1)].
        # Ctrl-World robotwin training annotations/statistics use reordered layout:
        #   [L_joint(6), R_joint(6), L_gripper(1), R_gripper(1)].
        # Keep this configurable so existing checkpoints can opt out if needed.
        self.reorder_robotwin_action_for_ctrlworld: bool = cfg.get(
            "reorder_robotwin_action_for_ctrlworld", True
        )
        # history_idx: non-uniform sampling indices into the image_queue deque
        self.history_idx: list = list(cfg.get("history_idx", [0, 0, -8, -6, -4, -2]))

        super().__init__(
            cfg, num_envs, seed_offset, total_num_processes, worker_info, record_metrics
        )

        # ---------- reset-state management (mirrors WanEnv / GenieEnv) ----------
        self.use_fixed_reset_state_ids: bool = cfg.use_fixed_reset_state_ids
        self.group_size: int = cfg.group_size
        self.num_group: int = self.num_envs // self.group_size
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

        # ---------- load Ctrl-World model ----------
        ctrlworld_repo = cfg.ctrlworld_repo_path
        if ctrlworld_repo not in sys.path:
            sys.path.insert(0, ctrlworld_repo)

        from models.ctrl_world import CrtlWorld  # noqa: PLC0415

        args = types.SimpleNamespace(
            svd_model_path=cfg.svd_model_path,
            clip_model_path=cfg.clip_model_path,
            action_dim=self.action_dim,
            num_history=self.num_history,
            num_frames=self.chunk,          # Ctrl-World: num_frames = predicted chunk size
            text_cond=cfg.get("text_cond", True),
            frame_level_cond=cfg.get("frame_level_cond", True),
            his_cond_zero=cfg.get("his_cond_zero", False),
            motion_bucket_id=cfg.get("motion_bucket_id", 127),
            fps=cfg.get("fps", 7),
        )
        self.ctrlworld_model = CrtlWorld(args)
        state_dict = torch.load(cfg.ctrlworld_ckpt_path, map_location="cpu")
        self.ctrlworld_model.load_state_dict(state_dict, strict=True)
        self.ctrlworld_model.to(self.device).eval()

        # Cache pipeline class for __call__ invocation
        from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline  # noqa: PLC0415

        self.pipeline_cls = CtrlWorldDiffusionPipeline

        # Inference hyper-parameters
        self.num_inference_steps: int = cfg.get("num_inference_steps", 50)
        self.guidance_scale: float = cfg.get("guidance_scale", 1.0)
        self.decode_chunk_size: int = cfg.get("decode_chunk_size", 7)
        self.svd_width: int = cfg.get("width", 320)     # SVD native width
        self.svd_height: int = cfg.get("height", 192)   # SVD native height

        # ---------- action normalisation statistics ----------
        # stat.json format: {"state_01": [...], "state_99": [...]}
        with open(cfg.data_stat_path) as f:
            stat = json.load(f)
        self.action_p01 = np.array(stat["state_01"], dtype=np.float32)  # (14,)
        self.action_p99 = np.array(stat["state_99"], dtype=np.float32)  # (14,)

        # ---------- reward model ----------
        self.reward_model = self._load_reward_model().eval().to(self.device)

        # ---------- runtime state ----------
        # current_obs: float32 [-1, 1], shape [B, 3, 1, T, H, W]
        self.current_obs: Optional[torch.Tensor] = None
        self.task_descriptions: list = [""] * self.num_envs

        # image_queue[env_idx]: deque of (1, 4, 72, 40) latent tensors
        # maxlen must accommodate the most negative history_idx (e.g. -8 → need ≥9 frames)
        queue_maxlen = max(16, abs(min(self.history_idx)) + 2)
        self.image_queue: list = [
            collections.deque(maxlen=queue_maxlen) for _ in range(self.num_envs)
        ]

        # action_queue[env_idx]: deque of (1, action_dim) float32 tensors
        self.action_queue: list = [
            collections.deque(maxlen=queue_maxlen) for _ in range(self.num_envs)
        ]

        self._is_offloaded: bool = False

    # ------------------------------------------------------------------
    # BaseWorldEnv abstract methods
    # ------------------------------------------------------------------

    def _build_dataset(self, cfg):
        """Return a CtrlWorldDataset for reset."""
        return CtrlWorldDataset(
            dataset_root=cfg.initial_image_path,
            data_type=cfg.get("data_type", "val"),
        )

    def step(self, actions):
        """Single-frame step – not used for world-model envs; delegates to chunk_step."""
        return self.chunk_step(actions)

    # ------------------------------------------------------------------
    # Reward model loader (mirrors GenieEnv / WanEnv)
    # ------------------------------------------------------------------

    def _load_reward_model(self):
        """Instantiate the reward model from cfg.reward_model."""
        rm_type = self.cfg.reward_model.type
        if rm_type == "ResnetRewModel":
            from diffsynth.models.reward_model import ResnetRewModel

            return ResnetRewModel(self.cfg.reward_model.from_pretrained)
        elif rm_type == "TaskEmbedResnetRewModel":
            from diffsynth.models.reward_model import TaskEmbedResnetRewModel

            return TaskEmbedResnetRewModel(
                checkpoint_path=self.cfg.reward_model.from_pretrained,
                task_suite_name=self.cfg.task_suite_name,
            )
        elif rm_type == "RoboTwinT5CrossAttn":
            from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel

            t5_model_name = self.cfg.reward_model.get("t5_model_name", "t5-base")
            return RoboTwinT5CrossAttnRewardModel.from_pretrained(
                self.cfg.reward_model.from_pretrained,
                config={"t5_model_name": t5_model_name},
            )
        elif rm_type == "QwenVLMProgressRewardModel":
            from rlinf.models.embodiment.reward import QwenVLMProgressRewardModel

            return QwenVLMProgressRewardModel(DictConfig(self.cfg.reward_model))
        else:
            raise ValueError(f"Unknown reward model type: {rm_type}")

    # ------------------------------------------------------------------
    # Metric helpers (mirrors GenieEnv)
    # ------------------------------------------------------------------

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
            self.success_once = self.success_once | torch.tensor(
                terminations, device=self.device, dtype=torch.bool
            )
        episode_info["success_once"] = self.success_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = torch.full(
            (self.num_envs,),
            self.elapsed_steps,
            dtype=torch.float32,
            device=self.device,
        )
        episode_info["reward"] = episode_info["return"] / (
            episode_info["episode_len"].clamp(min=1)
        )
        infos["episode"] = episode_info
        return infos

    def _calc_step_reward(self, chunk_rewards: torch.Tensor) -> torch.Tensor:
        """Compute relative-or-absolute per-step rewards (mirrors GenieEnv)."""
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
        else:
            return chunk_rewards

    def update_reset_state_ids(self):
        """Randomly assign dataset episodes to env groups."""
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

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reset(
        self,
        *,
        seed=None,
        options=None,
        episode_indices=None,
    ):
        """Reset all environments and return initial observations."""
        self.onload()
        self.elapsed_steps = 0

        if self.is_start:
            if self.use_fixed_reset_state_ids:
                episode_indices = self.reset_state_ids
            self._is_start = False

        if episode_indices is None:
            episode_indices = np.random.choice(
                len(self.dataset), size=self.num_envs, replace=True
            )
        elif isinstance(episode_indices, torch.Tensor):
            episode_indices = episode_indices.cpu().numpy()

        task_descs: list = []

        for env_idx, ep_idx in enumerate(episode_indices):
            ann = self.dataset[int(ep_idx)]

            # 1. Task description
            task_descs.append(ann["texts"][0] if ann.get("texts") else "")

            # 2. Load first-frame latent: (T, 4, 24, 40) → take frame 0
            latent_rel_path = ann["latent_videos"][0]["latent_video_path"]
            latent_path = os.path.join(self.dataset.dataset_root, latent_rel_path)
            latent_single = torch.load(latent_path, map_location="cpu")  # (T, 4, 24, 40)
            first_latent = latent_single[0:1]  # (1, 4, 24, 40)

            # 3. Replicate to 3 views by stacking along H (72 = 24×3)
            #    RobotWin only has head_camera; 3 views are identical.
            first_latent_3view = torch.cat([first_latent] * 3, dim=2)  # (1, 4, 72, 40)
            first_latent_3view = first_latent_3view.to(self.device)

            # 4. Fill image_queue with the initial latent
            self.image_queue[env_idx].clear()
            for _ in range(self.image_queue[env_idx].maxlen):
                self.image_queue[env_idx].append(first_latent_3view.clone())

            # 5. Fill action_queue with the initial action (or zeros)
            if ann.get("states") and len(ann["states"]) > 0:
                init_action = torch.tensor(
                    ann["states"][0], dtype=torch.float32
                ).unsqueeze(0)  # (1, 14)
            else:
                init_action = torch.zeros(1, self.action_dim, dtype=torch.float32)
            self.action_queue[env_idx].clear()
            for _ in range(self.action_queue[env_idx].maxlen):
                self.action_queue[env_idx].append(init_action.clone().to(self.device))

        self.task_descriptions = task_descs

        # 6. Decode the initial latent → pixel frame → current_obs
        #    Stack latest latent from each env: [B, 4, 72, 40]
        all_first_latents = torch.cat(
            [list(self.image_queue[i])[-1] for i in range(self.num_envs)], dim=0
        )  # [B, 4, 72, 40]

        # Take view 0: H[0:24]
        view0_latent = all_first_latents[:, :, 0:24, :]  # [B, 4, 24, 40]
        pixel_frames = self._decode_latent(view0_latent)  # [B, 3, 192, 320]
        pixel_frames = F.interpolate(
            pixel_frames, size=self.image_size, mode="bilinear", align_corners=False
        )  # [B, 3, H, W]

        # [B, 3, H, W] → [B, 3, 1, 1, H, W]
        self.current_obs = pixel_frames.unsqueeze(2).unsqueeze(3)

        self._reset_metrics()
        return self._wrap_obs(), {}

    # ------------------------------------------------------------------
    # Core inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _infer_next_chunk_frames(
        self, actions: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """Run CtrlWorldDiffusionPipeline and update current_obs / queues.

        Args:
            actions: ``[B, chunk=5, action_dim=14]`` float32, raw policy output.

        Side effects:
            Updates ``self.current_obs``, ``self.image_queue``, ``self.action_queue``.
        """
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        actions = actions.float()  # [B, 5, 14]
        if (
            self.wm_env_type == "robotwin"
            and self.action_dim == 14
            and self.reorder_robotwin_action_for_ctrlworld
        ):
            # Convert Aloha order -> Ctrl-World robotwin order:
            # [L0..L5, Lg, R0..R5, Rg] -> [L0..L5, R0..R5, Lg, Rg]
            actions = torch.cat(
                [actions[..., 0:6], actions[..., 7:13], actions[..., 6:7], actions[..., 13:14]],
                dim=-1,
            )
        B = self.num_envs

        # 1. p01/p99 bound normalisation → [-1, 1]
        p01 = self.action_p01[None, None, :]  # (1, 1, 14)
        p99 = self.action_p99[None, None, :]  # (1, 1, 14)
        actions_np = actions.cpu().numpy()    # [B, 5, 14]
        actions_norm = 2.0 * (actions_np - p01) / (p99 - p01 + 1e-8) - 1.0
        actions_norm = np.clip(actions_norm, -1.0, 1.0)  # [B, 5, 14]

        model_dtype = next(self.ctrlworld_model.parameters()).dtype
        actions_tensor = (
            torch.from_numpy(actions_norm).to(self.device).to(model_dtype)
        )  # [B, 5, 14]

        # 2. Build action_cond: history actions + new chunk actions → [B, 11, 14]
        all_action_cond = []
        for env_idx in range(B):
            q = list(self.action_queue[env_idx])  # list of (1, 14) tensors
            his_actions = torch.cat(
                [q[idx] for idx in self.history_idx], dim=0
            ).to(model_dtype)  # (6, 14)
            action_cond = torch.cat(
                [his_actions, actions_tensor[env_idx]], dim=0
            )  # (11, 14)
            all_action_cond.append(action_cond)
        action_cond_batch = torch.stack(all_action_cond, dim=0)  # [B, 11, 14]

        # 3. Encode actions → text_token [B, 11, 1024]
        text_token = self.ctrlworld_model.action_encoder(
            action_cond_batch,
            texts=self.task_descriptions,
            text_tokinizer=self.ctrlworld_model.tokenizer,
            text_encoder=self.ctrlworld_model.text_encoder,
            frame_level_cond=True,
        )  # [B, 11, 1024]

        # 4. Build image_cond [B, 4, 72, 40] and history [B, 6, 4, 72, 40]
        batch_image_cond = []
        batch_history = []
        for env_idx in range(B):
            q = list(self.image_queue[env_idx])  # list of (1, 4, 72, 40) tensors
            batch_image_cond.append(q[-1])  # (1, 4, 72, 40) – current frame
            his_frames = torch.cat(
                [q[idx] for idx in self.history_idx], dim=0
            )  # (6, 4, 72, 40)
            batch_history.append(his_frames)

        image_cond_batch = torch.cat(batch_image_cond, dim=0).to(
            self.device, model_dtype
        )  # [B, 4, 72, 40]
        history_batch = torch.stack(batch_history, dim=0).to(
            self.device, model_dtype
        )  # [B, 6, 4, 72, 40]

        # 5. Diffusion inference
        pipeline = self.ctrlworld_model.pipeline
        _, pred_latents = self.pipeline_cls.__call__(
            pipeline,
            image=image_cond_batch,          # [B, 4, 72, 40]
            text=text_token,                 # [B, 11, 1024]
            height=int(self.svd_height * 3), # 576 = 192×3
            width=self.svd_width,            # 320
            num_frames=self.chunk,           # 5
            history=history_batch,           # [B, 6, 4, 72, 40]
            num_inference_steps=self.num_inference_steps,
            decode_chunk_size=self.decode_chunk_size,
            max_guidance_scale=self.guidance_scale,
            fps=7,
            motion_bucket_id=127,
            mask=None,
            output_type="latent",
            return_dict=False,
            frame_level_cond=True,
        )
        # pred_latents: [B, chunk, 4, 72, 40]

        # 6. Split 3 views, take view 0, decode to pixels
        #    [B, chunk, 4, 72, 40] → [B*3, chunk, 4, 24, 40]
        pred_views = einops.rearrange(
            pred_latents, "b f c (m h) w -> (b m) f c h w", m=3
        )
        view0 = pred_views[::3]  # [B, chunk, 4, 24, 40] (every 3rd batch = view 0)

        B_v, T_pred, C_lat, Hl, Wl = view0.shape
        flat_latents = view0.reshape(B_v * T_pred, C_lat, Hl, Wl)  # [B*chunk, 4, 24, 40]
        pixel_frames = self._decode_latent(flat_latents)             # [B*chunk, 3, 192, 320]
        pixel_frames = F.interpolate(
            pixel_frames, size=self.image_size, mode="bilinear", align_corners=False
        )  # [B*chunk, 3, 256, 256]
        pixel_frames = pixel_frames.reshape(B_v, T_pred, 3, *self.image_size)

        # [B, chunk, 3, H, W] → [B, 3, chunk, H, W] → [B, 3, 1, chunk, H, W]
        new_frames = pixel_frames.permute(0, 2, 1, 3, 4).unsqueeze(2)

        # 7. Update current_obs
        self.current_obs = torch.cat([self.current_obs, new_frames], dim=3)
        max_keep = self.num_history + self.chunk
        if self.current_obs.shape[3] > max_keep:
            self.current_obs = self.current_obs[:, :, :, -max_keep:, :, :]

        # 8. Update image_queue: append the last predicted latent
        for env_idx in range(B):
            last_latent = pred_latents[env_idx, -1:, :, :, :]  # (1, 4, 72, 40)
            self.image_queue[env_idx].append(last_latent.detach())

        # 9. Update action_queue: append the last normalised action
        for env_idx in range(B):
            last_action = actions_tensor[env_idx, -1:, :]  # (1, 14)
            self.action_queue[env_idx].append(last_action.detach())

    @torch.no_grad()
    def _infer_next_chunk_rewards(self) -> torch.Tensor:
        """Score the latest chunk of generated frames with the reward model.

        Returns:
            ``[B, chunk]`` float32 reward tensor.
        """
        B, _c, _v, t, h, w = self.current_obs.shape
        chunk_obs = self.current_obs[:, :, 0, -self.chunk:, :, :]  # [B, 3, chunk, H, W]
        frames_flat = (
            chunk_obs.permute(0, 2, 1, 3, 4)
            .reshape(B * self.chunk, 3, h, w)
            .float()
        )  # [B*chunk, 3, H, W]  values in [-1, 1]

        rm_type = self.cfg.reward_model.type

        if rm_type == "ResnetRewModel":
            rewards = self.reward_model.predict_rew(frames_flat)
        elif rm_type == "TaskEmbedResnetRewModel":
            instructions = [
                self.task_descriptions[env_idx]
                for env_idx in range(B)
                for _ in range(self.chunk)
            ]
            rewards = self.reward_model.predict_rew(frames_flat, instructions)
        elif rm_type == "RoboTwinT5CrossAttn":
            # Reward model expects [0, 1]; current_obs is [-1, 1]
            frames_01 = ((frames_flat + 1.0) / 2.0).clamp(0.0, 1.0)
            instructions = [
                self.task_descriptions[env_idx]
                for env_idx in range(B)
                for _ in range(self.chunk)
            ]
            rewards = self.reward_model.compute_reward(
                frames_01, task_descriptions=instructions
            )
        elif rm_type == "QwenVLMProgressRewardModel":
            frames_flat = ((frames_flat + 1.0) / 2.0).clamp(0.0, 1.0)
            instructions = [
                self.task_descriptions[env_idx]
                for env_idx in range(B)
                for _ in range(self.chunk)
            ]
            rewards = self.reward_model.compute_reward(
                frames_flat, task_descriptions=instructions
            )
        else:
            raise ValueError(f"Unknown reward model type: {rm_type}")

        return rewards.reshape(B, self.chunk)

    # ------------------------------------------------------------------
    # _wrap_obs
    # ------------------------------------------------------------------

    def _wrap_obs(self) -> dict:
        """Extract the last frame and convert to policy-expected format.

        Returns:
            dict with keys:

            * ``main_images``:    ``[B, H, W, 3]`` uint8 [0, 255]
            * ``wrist_images``:   ``None``
            * ``states``:         ``[B, action_dim]`` float32 zeros (placeholder)
            * ``task_descriptions``: list of length B
        """
        # current_obs: [B, 3, 1, T, H, W], values in [-1, 1]
        last_frame = self.current_obs[:, :, 0, -1, :, :]   # [B, 3, H, W]
        img = last_frame.permute(0, 2, 3, 1)                # [B, H, W, 3]
        img = (img + 1.0) / 2.0 * 255.0
        img = img.clamp(0, 255).to(torch.uint8)
        return {
            "main_images": img,
            "wrist_images": None,
            "states": torch.zeros(
                self.num_envs, self.action_dim, dtype=torch.float32, device=self.device
            ),
            "task_descriptions": self.task_descriptions,
        }

    # ------------------------------------------------------------------
    # chunk_step
    # ------------------------------------------------------------------

    def chunk_step(self, policy_output_action):
        """Advance the environment by one chunk.

        Args:
            policy_output_action: ``[B, chunk, action_dim]`` raw policy actions.

        Returns:
            ``(obs_list, chunk_rewards, chunk_terminations, chunk_truncations, info_list)``
        """
        self.onload()

        actions = (
            policy_output_action.cpu().numpy()
            if isinstance(policy_output_action, torch.Tensor)
            else policy_output_action
        )

        # Generate next chunk of frames
        self._infer_next_chunk_frames(actions)

        # Compute rewards
        raw_rewards = self._infer_next_chunk_rewards()     # [B, chunk]
        chunk_rewards = self._calc_step_reward(raw_rewards)

        # Debug: print per-chunk, per-step rewards
        if getattr(self.cfg, "print_chunk_rewards", False):
            with torch.no_grad():
                raw = raw_rewards.cpu().numpy()   # 原始 reward model 输出
                diff = chunk_rewards.cpu().numpy()  # 差分 reward
                for env_idx in range(self.num_envs):
                    raw_str = ", ".join([f"{v:.4f}" for v in raw[env_idx]])
                    diff_str = ", ".join([f"{v:.4f}" for v in diff[env_idx]])
                    print(f"[chunk={self.elapsed_steps}] env{env_idx} "
                          f"raw=[{raw_str}] diff=[{diff_str}]")

        # Success: any frame in chunk exceeds threshold
        success_threshold = self.cfg.get("success_reward_threshold", 0.9)
        estimated_success = raw_rewards.max(dim=1).values >= success_threshold  # [B]

        # Termination / truncation (set at last chunk frame only)
        self.elapsed_steps += 1
        chunk_terminations = torch.zeros(
            self.num_envs, self.chunk, dtype=torch.bool, device=self.device
        )
        chunk_truncations = torch.zeros(
            self.num_envs, self.chunk, dtype=torch.bool, device=self.device
        )
        chunk_terminations[:, -1] = estimated_success
        chunk_truncations[:, -1] = self.elapsed_steps >= self.cfg.max_episode_steps

        # Logging
        step_reward = chunk_rewards[:, -1]
        infos: dict = {}
        self._record_metrics(step_reward, estimated_success, infos)

        obs = self._wrap_obs()

        if self.cfg.get("enable_offload", False):
            self.offload()

        return [obs], chunk_rewards, chunk_terminations, chunk_truncations, [infos]

    # ------------------------------------------------------------------
    # offload / onload
    # ------------------------------------------------------------------

    def offload(self):
        """Move Ctrl-World model and reward model to CPU to free GPU memory."""
        if not self._is_offloaded:
            self.ctrlworld_model.to("cpu")
            self.reward_model.to("cpu")
            torch.cuda.empty_cache()
            self._is_offloaded = True

    def onload(self):
        """Move models back to GPU."""
        if self._is_offloaded:
            self.ctrlworld_model.to(self.device)
            self.reward_model.to(self.device)
            self._is_offloaded = False

    # ------------------------------------------------------------------
    # VAE decode helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _decode_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode SVD-VAE latents to pixel tensor in [-1, 1].

        Args:
            latents: ``[N, 4, H', W']`` latent tensors (already scaled by scaling_factor).

        Returns:
            ``[N, 3, H, W]`` float32 pixels in [-1, 1].
        """
        vae = self.ctrlworld_model.vae
        vae_dtype = next(vae.parameters()).dtype
        latents = latents.to(self.device, vae_dtype)
        decoded_chunks = []
        for i in range(0, latents.shape[0], self.decode_chunk_size):
            chunk = latents[i : i + self.decode_chunk_size]
            chunk = chunk / vae.config.scaling_factor
            decoded_chunks.append(
                vae.decode(chunk, num_frames=chunk.shape[0]).sample
            )
        pixels = torch.cat(decoded_chunks, dim=0)  # [N, 3, H, W], values in [-1, 1]
        return pixels.float()
