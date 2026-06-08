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

"""GENIE (STMaskGIT + MagViT2) world model environment for RLinf."""

from __future__ import annotations

import collections
import io
import json
import os
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig

from rlinf.envs.utils import recursive_to_device
from rlinf.envs.world_model.base_world_env import BaseWorldEnv

__all__ = ["GenieEnv"]

# ---------------------------------------------------------------------------
# Helper: make roboscape/genie importable regardless of PYTHONPATH
# ---------------------------------------------------------------------------
_ROBOSCAPE_ROOT = Path(__file__).resolve().parents[3] / "roboscape" / "genie"
if str(_ROBOSCAPE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROBOSCAPE_ROOT))


class GenieEnv(BaseWorldEnv):
    """World-model environment backed by GENIE (STMaskGIT + MagViT2).

    Compared with WanEnv:
    * Inference runs in *discrete token space* – every generated frame is a
      (H_tok × W_tok) int64 tensor that must be decoded via MagViT2.
    * Each chunk_step requires ``chunk`` sequential forward passes through the
      Transformer (one per frame), unlike Wan which generates the whole chunk
      in one diffusion call.
    * Three models are managed: STMaskGIT (world model), VQModel (tokenizer /
      decoder) and the reward model.
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
        # BaseWorldEnv.__init__ calls self._build_dataset(cfg) so all
        # hyper-parameters needed by that method must be set first.
        self.chunk: int = cfg.chunk
        self.num_prompt_frames: int = cfg.num_prompt_frames
        self.window_size: int = cfg.window_size
        assert self.window_size == self.num_prompt_frames + self.chunk, (
            f"window_size ({self.window_size}) must equal "
            f"num_prompt_frames ({self.num_prompt_frames}) + chunk ({self.chunk})"
        )
        self.image_size: tuple[int, int] = tuple(cfg.image_size)
        self.action_dim: int = cfg.get("action_dim", 14)
        self.latent_side_len: int = cfg.get("latent_side_len", 16)
        self.wm_env_type: str = cfg.get("wm_env_type", "robotwin")

        super().__init__(
            cfg, num_envs, seed_offset, total_num_processes, worker_info, record_metrics
        )

        # ------------------------------------------------------------------
        # Reset-state management (mirrors WanEnv)
        # ------------------------------------------------------------------
        self.use_fixed_reset_state_ids: bool = cfg.use_fixed_reset_state_ids
        self.group_size: int = cfg.group_size
        self.num_group: int = self.num_envs // self.group_size
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

        # ------------------------------------------------------------------
        # Optional action normalisation statistics
        # (only needed when encode_vla_robotwin.sh was run with --stat_file)
        # ------------------------------------------------------------------
        self.action_stat = None
        self.action_p01: Optional[np.ndarray] = None
        self.action_p99: Optional[np.ndarray] = None
        stat_path = cfg.get("action_stat_file", None)
        if stat_path and os.path.exists(stat_path):
            with open(stat_path) as fh:
                self.action_stat = json.load(fh)
            self.action_p01 = np.array(
                self.action_stat["state_01"], dtype=np.float32
            )
            self.action_p99 = np.array(
                self.action_stat["state_99"], dtype=np.float32
            )

        # ------------------------------------------------------------------
        # World model (STMaskGIT)
        # ------------------------------------------------------------------
        from genie.config import GenieConfig
        from genie.st_mask_git import STMaskGIT

        genie_config = GenieConfig.from_pretrained(cfg.genie_config_path)
        genie_config.T = self.window_size
        genie_config.S = self.latent_side_len ** 2
        genie_config.use_mup = False
        self.world_model = STMaskGIT.from_pretrained(cfg.genie_ckpt_path).to(
            self.device
        )
        # Patch the config in case the checkpoint was saved with different T/S
        self.world_model.config.T = genie_config.T
        self.world_model.config.S = genie_config.S
        self.world_model.eval()

        # ------------------------------------------------------------------
        # Video tokenizer / decoder (MagViT2 VQModel)
        # ------------------------------------------------------------------
        from magvit2.config import VQConfig
        from magvit2.models.lfqgan import VQModel

        vq_config = VQConfig()
        self.tokenizer = VQModel(
            vq_config, ckpt_path=cfg.magvit2_ckpt_path
        ).to(self.device)
        self.tokenizer.eval()

        # ------------------------------------------------------------------
        # Reward model
        # ------------------------------------------------------------------
        self.reward_model = self._load_reward_model().eval().to(self.device)

        # ------------------------------------------------------------------
        # Runtime state
        # ------------------------------------------------------------------
        # current_obs: float32 [-1, 1], shape [B, 3, 1, T, H, W]
        self.current_obs: Optional[torch.Tensor] = None
        # Per-env GT trajectory last frame in pixel space: [B, 3, H, W], [-1, 1].
        self.gt_last_frames: Optional[torch.Tensor] = None
        self.task_descriptions: list[str] = [""] * self.num_envs
        # segment_id -> final frame index in token dataset.
        self._segment_last_index: dict[int, int] = self._build_segment_last_index()

        # image_queue[env_idx]: deque of (H_tok, W_tok) int64 tensors
        self.image_queue: list[collections.deque] = [
            collections.deque(maxlen=self.num_prompt_frames)
            for _ in range(self.num_envs)
        ]
        # action_queue[env_idx]: deque of (action_dim,) float32 tensors
        self.action_queue: list[collections.deque] = [
            collections.deque(maxlen=self.num_prompt_frames)
            for _ in range(self.num_envs)
        ]

        self._is_offloaded: bool = False

    # ------------------------------------------------------------------
    # BaseWorldEnv abstract methods
    # ------------------------------------------------------------------

    def _build_dataset(self, cfg):
        """Build the RawTokenDataset used to sample initial frames on reset."""
        from genie.config import GenieConfig
        from data_worldarena import RawTokenDataset

        genie_config = GenieConfig.from_pretrained(cfg.genie_config_path)
        dataset = RawTokenDataset(
            data_dir=cfg.initial_image_path,
            window_size=self.window_size,
            config=genie_config,
            stride=1,
            split="val",
            use_action=True,
            use_text=False,
            rollout=False,
        )
        return dataset

    def _load_reward_model(self):
        """Instantiate the reward model from cfg.reward_model (mirrors WanEnv)."""
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
        elif rm_type == "LPIPSLastFrameRewardModel":
            from rlinf.models.embodiment.reward import LPIPSLastFrameRewardModel

            return LPIPSLastFrameRewardModel(DictConfig(self.cfg.reward_model))
        else:
            raise ValueError(f"Unknown reward model type: {rm_type}")

    def _build_segment_last_index(self) -> dict[int, int]:
        """Build mapping from segment id to its last frame index."""
        seg = getattr(self.dataset, "segment_ids", None)
        if seg is None:
            return {}
        seg_flat = np.asarray(seg).reshape(-1)
        segment_last_index: dict[int, int] = {}
        for idx, seg_id in enumerate(seg_flat):
            segment_last_index[int(seg_id)] = idx
        return segment_last_index

    def _get_gt_last_token_for_episode(
        self, episode_idx: int, fallback_token: torch.Tensor
    ) -> torch.Tensor:
        """Get GT trajectory last-frame token map for one episode."""
        if not self._segment_last_index:
            return fallback_token
        start_ind = int(self.dataset.valid_start_inds[episode_idx])
        seg_raw = self.dataset.segment_ids[start_ind]
        seg_id = int(np.asarray(seg_raw).reshape(-1)[0])
        last_idx = self._segment_last_index.get(seg_id, start_ind)
        gt_last_np = np.asarray(self.dataset.data[last_idx]).astype(np.int64)
        return torch.from_numpy(gt_last_np)

    # ------------------------------------------------------------------
    # Metric helpers (mirrors WanEnv)
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
        """Compute relative-or-absolute per-step rewards (mirrors WanEnv)."""
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

    def _estimate_success_from_rewards(
        self, chunk_rewards: torch.Tensor
    ) -> torch.Tensor:
        success_threshold = getattr(self.cfg, "success_reward_threshold", 0.9)
        max_reward = chunk_rewards.max(dim=1)[0]
        return (max_reward >= success_threshold).to(self.device)

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

    # ------------------------------------------------------------------
    # Token ↔ pixel conversion
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _decode_tokens(self, tokens: torch.LongTensor) -> torch.Tensor:
        """Decode MagViT2 discrete tokens to pixel tensor.

        Args:
            tokens: ``(N, H_tok, W_tok)`` int64 tensor on self.device.

        Returns:
            ``(N, 3, H, W)`` float32 tensor in [-1, 1] range.
        """
        tokens = tokens.to(self.device)
        codebook_dim = self.tokenizer.quantize.codebook_dim

        def _decode(tok):
            flat = rearrange(tok, "b h w -> b (h w)")
            quant = self.tokenizer.quantize.get_codebook_entry(
                flat,
                bhwc=tok.shape + (codebook_dim,),
            ).flip(1)
            return self.tokenizer.decode(quant.to(device=self.device, dtype=torch.bfloat16))

        if self.tokenizer.use_ema:
            with self.tokenizer.ema_scope():
                pixel = _decode(tokens)
        else:
            pixel = _decode(tokens)

        return pixel.float()  # (N, 3, H, W) in [-1, 1]

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = None,
        episode_indices: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        """Reset all environments and return initial observations."""
        self.onload()
        self.elapsed_steps = 0

        if self.is_start:
            if self.use_fixed_reset_state_ids:
                episode_indices = self.reset_state_ids
            self._is_start = False

        if len(self.dataset) < self.num_envs:
            raise ValueError(
                f"Dataset too small: {len(self.dataset)} episodes, need {self.num_envs}"
            )

        if episode_indices is None:
            if seed is not None:
                np.random.seed(seed[0] if isinstance(seed, list) else seed)
            episode_indices = np.random.choice(
                len(self.dataset), size=self.num_envs, replace=False
            )
        elif isinstance(episode_indices, torch.Tensor):
            episode_indices = episode_indices.cpu().numpy()

        # ----------------------------------------------------------------
        # Load initial token windows from dataset
        # ----------------------------------------------------------------
        task_descriptions: list[str] = []
        gt_last_tokens: list[torch.Tensor] = []
        for env_idx, ep_idx in enumerate(episode_indices):
            sample = self.dataset[int(ep_idx)]

            # input_ids: (window_size * H_tok * W_tok,) int64
            tokens = sample["input_ids"].reshape(
                self.window_size, self.latent_side_len, self.latent_side_len
            )
            # actions: (window_size * action_dim,) float32
            actions_flat = sample["actions"]
            if actions_flat is not None:
                actions = actions_flat.reshape(self.window_size, -1)
                # Trim / pad to action_dim
                if actions.shape[1] > self.action_dim:
                    actions = actions[:, : self.action_dim]
                elif actions.shape[1] < self.action_dim:
                    pad = torch.zeros(
                        self.window_size,
                        self.action_dim - actions.shape[1],
                        dtype=torch.float32,
                    )
                    actions = torch.cat([actions, pad], dim=1)
            else:
                actions = torch.zeros(
                    self.window_size, self.action_dim, dtype=torch.float32
                )

            # Fill queues with the first num_prompt_frames
            self.image_queue[env_idx].clear()
            self.action_queue[env_idx].clear()
            for t in range(self.num_prompt_frames):
                self.image_queue[env_idx].append(
                    tokens[t].to(torch.int64).to(self.device)
                )
                self.action_queue[env_idx].append(
                    actions[t].to(torch.float32).to(self.device)
                )

            gt_last_tok = self._get_gt_last_token_for_episode(
                int(ep_idx), fallback_token=tokens[-1]
            )
            gt_last_tokens.append(gt_last_tok.to(torch.int64).to(self.device))

            task_descriptions.append("")

        self.task_descriptions = task_descriptions

        # ----------------------------------------------------------------
        # Decode initial frames token → pixel to build current_obs
        # ----------------------------------------------------------------
        # tokens_batch: [B, num_prompt_frames, H_tok, W_tok]
        tokens_batch = torch.stack(
            [torch.stack(list(self.image_queue[i])) for i in range(self.num_envs)]
        ).to(self.device)  # [B, T_p, Ht, Wt]

        B, T_p, Ht, Wt = tokens_batch.shape
        flat_tokens = tokens_batch.reshape(B * T_p, Ht, Wt)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            pixel_frames = self._decode_tokens(flat_tokens)  # [B*T_p, 3, H, W]
        pixel_frames = pixel_frames.reshape(
            B, T_p, 3, pixel_frames.shape[2], pixel_frames.shape[3]
        )  # [B, T_p, 3, H, W]

        # Resize to configured image_size if necessary
        if pixel_frames.shape[-2:] != self.image_size:
            pixel_frames = F.interpolate(
                pixel_frames.reshape(B * T_p, 3, *pixel_frames.shape[-2:]),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            ).reshape(B, T_p, 3, *self.image_size)

        # Decode GT trajectory last frame (one per env) for LPIPS reward.
        gt_tokens_batch = torch.stack(gt_last_tokens, dim=0).to(self.device)  # [B,Ht,Wt]
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            gt_last_frames = self._decode_tokens(gt_tokens_batch)  # [B,3,H,W]
        if gt_last_frames.shape[-2:] != self.image_size:
            gt_last_frames = F.interpolate(
                gt_last_frames,
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            )
        self.gt_last_frames = gt_last_frames

        # Conform to current_obs layout: [B, 3, 1, T, H, W]
        # [B, T_p, 3, H, W] -> [B, 3, T_p, H, W] -> [B, 3, 1, T_p, H, W]
        self.current_obs = pixel_frames.permute(0, 2, 1, 3, 4).unsqueeze(2)

        self._reset_metrics()
        return self._wrap_obs(), {}

    # ------------------------------------------------------------------
    # World-model inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _infer_next_chunk_frames(
        self, actions: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """Auto-regressively generate ``self.chunk`` new frames.

        Args:
            actions: ``[B, chunk, action_dim]`` float32.  Raw policy outputs –
                will be normalised internally only if ``cfg.action_stat_file``
                was provided and the matching statistics were loaded.
        """
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        actions = actions.float().to(self.device)  # [B, chunk, action_dim]

        # Optional p01/p99 normalisation (mirrors encode_vla_robotwin.py)
        if self.action_stat is not None:
            p01 = torch.from_numpy(self.action_p01).to(self.device)
            p99 = torch.from_numpy(self.action_p99).to(self.device)
            actions_proc = 2.0 * (actions - p01) / (p99 - p01 + 1e-8) - 1.0
            actions_proc = torch.clamp(actions_proc, -1.0, 1.0)
        else:
            # encode_vla_robotwin.sh did not pass --stat_file; raw values used
            actions_proc = actions

        all_new_pixel: list[torch.Tensor] = []

        for step in range(self.chunk):
            # ---- Build prompt_THW [B, window_size, Ht, Wt] ----
            prompt_list: list[torch.Tensor] = []
            for env_idx in range(self.num_envs):
                hist_tokens = torch.stack(
                    list(self.image_queue[env_idx])
                )  # [num_prompt_frames, Ht, Wt]
                n_mask = self.window_size - self.num_prompt_frames
                mask_frames = torch.full(
                    (n_mask, self.latent_side_len, self.latent_side_len),
                    self.world_model.mask_token_id,
                    dtype=torch.long,
                    device=self.device,
                )
                prompt_list.append(torch.cat([hist_tokens, mask_frames], dim=0))
            prompt_THW = torch.stack(prompt_list, dim=0)  # [B, T, Ht, Wt]

            # ---- Build prompt_action [B, window_size, action_dim] ----
            action_list: list[torch.Tensor] = []
            for env_idx in range(self.num_envs):
                hist_acts = torch.stack(
                    list(self.action_queue[env_idx])
                )  # [num_prompt_frames, action_dim]
                new_act = actions_proc[env_idx, step].unsqueeze(0)  # [1, action_dim]
                pad_len = self.window_size - self.num_prompt_frames - 1
                pad_acts = torch.zeros(
                    pad_len, self.action_dim, device=self.device
                )
                action_list.append(
                    torch.cat([hist_acts, new_act, pad_acts], dim=0)
                )
            prompt_action = torch.stack(action_list, dim=0)  # [B, T, action_dim]

            # ---- Call STMaskGIT ----
            samples_HW, _ = self.world_model.maskgit_generate(
                prompt_THW,
                prompt_action,
                out_t=self.num_prompt_frames,
                maskgit_steps=2,
                temperature=0.0,
            )
            # samples_HW: [B, Ht, Wt] int64

            # ---- Decode to pixel ----
            new_frame = self._decode_tokens(samples_HW)  # [B, 3, H, W] float32 [-1,1]
            # Resize if needed
            if new_frame.shape[-2:] != self.image_size:
                new_frame = F.interpolate(
                    new_frame, size=self.image_size, mode="bilinear", align_corners=False
                )
            all_new_pixel.append(new_frame)

            # ---- Update queues ----
            for env_idx in range(self.num_envs):
                self.image_queue[env_idx].append(samples_HW[env_idx])
                self.action_queue[env_idx].append(actions_proc[env_idx, step])

        # ---- Append new frames to current_obs ----
        # all_new_pixel: list[chunk × (B, 3, H, W)]
        new_frames = torch.stack(all_new_pixel, dim=1)  # [B, chunk, 3, H, W]
        new_frames = new_frames.permute(0, 2, 1, 3, 4).unsqueeze(2)  # [B,3,1,chunk,H,W]
        self.current_obs = torch.cat([self.current_obs, new_frames], dim=3)

        # Sliding window: keep at most num_prompt_frames + chunk frames
        max_frames = self.num_prompt_frames + self.chunk
        if self.current_obs.shape[3] > max_frames:
            self.current_obs = self.current_obs[:, :, :, -max_frames:, :, :]

    # ------------------------------------------------------------------
    # Reward inference
    # ------------------------------------------------------------------

    def _infer_next_chunk_rewards(self) -> torch.Tensor:
        """Compute per-frame rewards for the last ``chunk`` frames."""
        if self.reward_model is None:
            raise ValueError("Reward model is not loaded")

        _B, _c, _v, _t, h, w = self.current_obs.shape
        # Extract the newly generated chunk frames
        chunk_obs = self.current_obs[:, :, 0, -self.chunk :, :, :]  # [B, 3, chunk, H, W]
        # Flatten to [B*chunk, 3, H, W]
        frames_flat = (
            chunk_obs.permute(0, 2, 1, 3, 4)
            .reshape(self.num_envs * self.chunk, 3, h, w)
            .float()
        )

        rm_type = self.cfg.reward_model.type

        if rm_type == "ResnetRewModel":
            rewards = self.reward_model.predict_rew(frames_flat)
        elif rm_type == "TaskEmbedResnetRewModel":
            instructions = [
                self.task_descriptions[env_idx]
                for env_idx in range(self.num_envs)
                for _ in range(self.chunk)
            ]
            rewards = self.reward_model.predict_rew(frames_flat, instructions)
        elif rm_type == "RoboTwinT5CrossAttn":
            # Convert [-1, 1] → [0, 1] before passing to compute_reward
            # (which applies ImageNet normalisation internally once)
            frames_01 = ((frames_flat + 1.0) / 2.0).clamp(0.0, 1.0)
            instructions = [
                self.task_descriptions[env_idx]
                for env_idx in range(self.num_envs)
                for _ in range(self.chunk)
            ]
            rewards = self.reward_model.compute_reward(
                frames_01, task_descriptions=instructions
            )
        elif rm_type == "QwenVLMProgressRewardModel":
            frames_flat = ((frames_flat + 1.0) / 2.0).clamp(0.0, 1.0)
            instructions = [
                self.task_descriptions[env_idx]
                for env_idx in range(self.num_envs)
                for _ in range(self.chunk)
            ]
            rewards = self.reward_model.compute_reward(
                frames_flat, task_descriptions=instructions
            )
        elif rm_type == "LPIPSLastFrameRewardModel":
            if self.gt_last_frames is None:
                raise ValueError(
                    "GT last frames are not initialized; call reset() first."
                )
            gt_flat = (
                self.gt_last_frames.unsqueeze(1)
                .repeat(1, self.chunk, 1, 1, 1)
                .reshape(self.num_envs * self.chunk, 3, h, w)
                .float()
            )
            rewards = self.reward_model.compute_reward(frames_flat, references=gt_flat)
        else:
            raise ValueError(f"Unknown reward model type: {rm_type}")

        return rewards.reshape(self.num_envs, self.chunk)

    # ------------------------------------------------------------------
    # Observation wrapping
    # ------------------------------------------------------------------

    def _wrap_obs(self) -> dict:
        """Wrap the latest frame into the standard observation dict."""
        b, c, v, t, h, w = self.current_obs.shape
        last_frame = self.current_obs[:, :, 0, -1, :, :]  # [B, 3, H, W]

        full_image = last_frame.permute(0, 2, 3, 1)  # [B, H, W, 3]
        full_image = (full_image + 1.0) / 2.0 * 255.0
        full_image = torch.clamp(full_image, 0, 255)

        if full_image.shape[1:3] != tuple(self.image_size):
            full_image = F.interpolate(
                full_image.permute(0, 3, 1, 2).float(),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)

        full_image = full_image.to(torch.uint8)
        states = torch.zeros(
            (b, self.action_dim), device=self.device, dtype=torch.float32
        )

        return {
            "main_images": full_image,             # [B, H, W, 3] uint8 [0,255]
            "wrist_images": None,
            "states": states,                      # [B, action_dim] float32 zeros
            "task_descriptions": self.task_descriptions,
        }

    # ------------------------------------------------------------------
    # chunk_step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, actions=None, auto_reset=True):
        raise NotImplementedError(
            "GenieEnv does not implement step(); use chunk_step() instead."
        )

    @torch.no_grad()
    def chunk_step(self, policy_output_action):
        """Advance the environment by one chunk of actions.

        Args:
            policy_output_action: ``[B, chunk, action_dim]`` array or tensor.

        Returns:
            Tuple ``([obs], rewards, terminations, truncations, [infos])``.
        """
        self.onload()

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            self._infer_next_chunk_frames(policy_output_action)

        self.elapsed_steps += self.chunk
        extracted_obs = self._wrap_obs()
        chunk_rewards = self._infer_next_chunk_rewards()
        chunk_rewards_tensors = self._calc_step_reward(chunk_rewards)

        # Debug: print per-chunk, per-step rewards
        if getattr(self.cfg, "print_chunk_rewards", False):
            with torch.no_grad():
                # chunk_rewards: [num_envs, chunk], chunk_rewards_tensors: [num_envs, chunk]
                raw = chunk_rewards.cpu().numpy()  # 原始 reward model 输出
                diff = chunk_rewards_tensors.cpu().numpy()  # 差分 reward
                for env_idx in range(self.num_envs):
                    raw_str = ", ".join([f"{v:.4f}" for v in raw[env_idx]])
                    diff_str = ", ".join([f"{v:.4f}" for v in diff[env_idx]])
                    print(f"[chunk={self.elapsed_steps//self.chunk}] env{env_idx} "
                          f"raw=[{raw_str}] diff=[{diff_str}]")

        estimated_success = self._estimate_success_from_rewards(chunk_rewards)

        raw_chunk_terminations = torch.zeros(
            self.num_envs, self.chunk, dtype=torch.bool, device=self.device
        )
        raw_chunk_terminations[:, -1] = estimated_success

        raw_chunk_truncations = torch.zeros_like(raw_chunk_terminations)
        truncated = torch.tensor(
            self.elapsed_steps >= self.cfg.max_episode_steps
        ).to(self.device)
        if truncated.any():
            raw_chunk_truncations[:, -1] = truncated

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

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        final_obs = extracted_obs
        final_info = infos
        extracted_obs, infos = self.reset()
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        return extracted_obs, infos

    # ------------------------------------------------------------------
    # offload / onload
    # ------------------------------------------------------------------

    def offload(self):
        """Move all models and runtime tensors to CPU to free GPU memory."""
        if self._is_offloaded:
            return
        self.world_model = self.world_model.to("cpu")
        self.tokenizer = self.tokenizer.to("cpu")
        self.reward_model = self.reward_model.to("cpu")
        if self.current_obs is not None:
            self.current_obs = self.current_obs.to("cpu")
        self.prev_step_reward = self.prev_step_reward.cpu()
        self.reset_state_ids = self.reset_state_ids.cpu()
        if self.record_metrics:
            self.success_once = self.success_once.cpu()
            self.returns = self.returns.cpu()
        torch.cuda.empty_cache()
        self._is_offloaded = True

    def onload(self):
        """Move all models and runtime tensors back to the execution device."""
        if not self._is_offloaded:
            return
        self.world_model = self.world_model.to(self.device)
        self.tokenizer = self.tokenizer.to(self.device)
        self.reward_model = self.reward_model.to(self.device)
        if self.current_obs is not None:
            self.current_obs = self.current_obs.to(self.device)
        self.prev_step_reward = self.prev_step_reward.to(self.device)
        self.reset_state_ids = self.reset_state_ids.to(self.device)
        if self.record_metrics:
            self.success_once = self.success_once.to(self.device)
            self.returns = self.returns.to(self.device)
        self._is_offloaded = False

    # ------------------------------------------------------------------
    # State serialisation (for async runner compatibility)
    # ------------------------------------------------------------------

    def get_state(self) -> bytes:
        """Serialise runtime state to CPU bytes buffer."""
        env_state = {
            "current_obs": recursive_to_device(self.current_obs, "cpu")
            if self.current_obs is not None
            else None,
            "task_descriptions": self.task_descriptions,
            "elapsed_steps": self.elapsed_steps,
            "prev_step_reward": self.prev_step_reward.cpu(),
            "_is_start": self._is_start,
            "reset_state_ids": self.reset_state_ids.cpu(),
            "generator_state": self._generator.get_state(),
        }
        if self.record_metrics:
            env_state.update(
                {
                    "success_once": self.success_once.cpu(),
                    "returns": self.returns.cpu(),
                }
            )
        # Serialise image_queue and action_queue (move each tensor to CPU)
        env_state["image_queue"] = [
            [tok.cpu() for tok in q] for q in self.image_queue
        ]
        env_state["action_queue"] = [
            [act.cpu() for act in q] for q in self.action_queue
        ]
        buf = io.BytesIO()
        torch.save(env_state, buf)
        return buf.getvalue()
