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

"""iVideoGPT Transformer world-model environment for RLinf."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from rlinf.data.datasets.world_model import NpyTrajectoryDatasetWrapper
from rlinf.envs.utils import recursive_to_device
from rlinf.envs.world_model.base_world_env import BaseWorldEnv

__all__ = ["IVideoGPTEnv"]


class IVideoGPTEnv(BaseWorldEnv):
    """World-model environment backed by iVideoGPT action-conditioned transformer."""

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
        self.segment_length = int(cfg.get("segment_length", self.chunk + self.condition_frame_length))
        if self.segment_length != self.chunk + self.condition_frame_length:
            raise ValueError(
                "segment_length must equal condition_frame_length + chunk for online RL rollout, "
                f"got segment_length={self.segment_length}, "
                f"condition_frame_length={self.condition_frame_length}, chunk={self.chunk}"
            )

        self.action_dim = int(cfg.action_dim)
        image_size = cfg.image_size
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = tuple(image_size)
        self.wm_env_type = cfg.get("wm_env_type", "robotwin")

        super().__init__(
            cfg, num_envs, seed_offset, total_num_processes, worker_info, record_metrics
        )

        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.group_size = cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

        self._setup_ivideogpt_import_path()
        self._load_ivideogpt_models()
        self.reward_model = self._load_reward_model().eval().to(self.device)

        self.current_obs = None
        self.task_descriptions = [""] * self.num_envs

        self.do_sample = bool(cfg.get("do_sample", True))
        self.temperature = float(cfg.get("temperature", 1.0))
        self.top_k = int(cfg.get("top_k", 100))
        self.pad_token_id = int(cfg.get("pad_token_id", 50256))

        self._is_offloaded = False

    def _setup_ivideogpt_import_path(self):
        repo_root = Path(__file__).resolve().parents[3]
        ivideogpt_root = self.cfg.get("ivideogpt_root", str(repo_root / "iVideoGPT"))
        if ivideogpt_root not in sys.path:
            sys.path.insert(0, ivideogpt_root)

    def _build_dataset(self, cfg):
        data_dir = Path(cfg.initial_image_path)
        npz_files = sorted(str(p) for p in data_dir.glob("*.npz"))
        if len(npz_files) > 0:
            self._dataset_mode = "npz"
            return npz_files

        self._dataset_mode = "npy"
        action_key = cfg.get("action_key", "abs_action")
        return NpyTrajectoryDatasetWrapper(
            cfg.initial_image_path,
            action_key=action_key,
        )

    def _load_ivideogpt_models(self):
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModelForCausalLM

        from ivideogpt.transformer import HeadModelWithAction
        from ivideogpt.vq_model import CompressiveVQModel

        model_bundle_path = self.cfg.model_bundle_path
        self.tokenizer = CompressiveVQModel.from_pretrained(
            model_bundle_path,
            subfolder="tokenizer",
            low_cpu_mem_usage=False,
        ).to(self.device)
        self.tokenizer.eval()

        if int(self.tokenizer.context_length) != self.condition_frame_length:
            raise ValueError(
                "condition_frame_length must match tokenizer.context_length, "
                f"got {self.condition_frame_length} vs {self.tokenizer.context_length}"
            )

        config = AutoConfig.from_pretrained(model_bundle_path, subfolder="transformer")
        config.vocab_size = (
            self.tokenizer.num_vq_embeddings + self.tokenizer.num_dyn_embeddings + 2
        )
        llm = AutoModelForCausalLM.from_config(config).to(self.device)

        self.context_tokens_per_frame = int(self.cfg.get("context_tokens_per_frame", 257))
        self.tokens_num_per_dyna = int(self.cfg.get("tokens_per_dyna", 16))
        self.prelude_tokens_num = (
            self.context_tokens_per_frame * self.condition_frame_length - 1
        )

        self.model = HeadModelWithAction(
            llm,
            action_dim=self.action_dim,
            prelude_tokens_num=self.prelude_tokens_num,
            tokens_num_per_dyna=self.tokens_num_per_dyna,
            context=self.condition_frame_length,
            segment_length=self.segment_length,
        ).to(self.device)

        state_dict = load_file(
            os.path.join(model_bundle_path, "transformer", "model.safetensors")
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

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
        else:
            raise ValueError(f"Unknown reward model type: {reward_type}")
        return rew_model

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

    def _load_from_npz(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=False)
        image_key = self.cfg.get("image_key", None)
        if image_key is not None and image_key in data:
            images = data[image_key]
        else:
            images = None
            for key in data.files:
                arr = data[key]
                if isinstance(arr, np.ndarray) and arr.ndim >= 3:
                    if arr.shape[-1] == 3:
                        images = arr
                        break
            if images is None:
                raise ValueError(f"No image array found in npz file: {npz_path}")

        if images.ndim == 5 and images.shape[1] == 1:
            images = images[:, 0]
        if images.ndim == 3:
            images = images[None, ...]
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Unexpected image shape in {npz_path}: {images.shape}")

        first_frame = images[0]
        img_tensor = torch.from_numpy(first_frame).permute(2, 0, 1).float()
        if img_tensor.max() > 1.5:
            img_tensor = img_tensor / 255.0

        task_desc = str(self.cfg.get("default_task_description", "click the bell"))
        return img_tensor, task_desc

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

        if len(self.dataset) < self.num_envs:
            raise ValueError(
                f"Not enough episodes in dataset. Found {len(self.dataset)}, need {self.num_envs}"
            )

        if episode_indices is None:
            if seed is not None:
                np.random.seed(seed[0] if isinstance(seed, list) else seed)
            episode_indices = np.random.choice(
                len(self.dataset), size=self.num_envs, replace=False
            )
        elif isinstance(episode_indices, torch.Tensor):
            episode_indices = episode_indices.cpu().numpy()

        img_tensors = []
        task_descriptions = []

        for episode_idx in episode_indices:
            if self._dataset_mode == "npz":
                img_tensor, task_desc = self._load_from_npz(self.dataset[int(episode_idx)])
            else:
                episode_data = self.dataset[int(episode_idx)]
                first_frame = episode_data["start_items"][0]
                img_tensor = first_frame["image"]
                task_desc = str(episode_data.get("task", ""))

            if img_tensor.shape[1:] != self.image_size:
                img_tensor = F.interpolate(
                    img_tensor.unsqueeze(0),
                    size=self.image_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            img_tensor = img_tensor * 2.0 - 1.0
            env_img_tensor = img_tensor.unsqueeze(1).repeat(
                1, self.condition_frame_length, 1, 1
            )
            img_tensors.append(env_img_tensor)
            task_descriptions.append(task_desc)

        stacked_imgs = torch.stack(img_tensors, dim=0).to(self.device)
        self.current_obs = stacked_imgs.unsqueeze(2).to(self.device)
        self.task_descriptions = task_descriptions
        self._reset_metrics()

        return self._wrap_obs(), {}

    @torch.no_grad()
    def step(self, actions=None, auto_reset=True):
        raise NotImplementedError("step in IVideoGPTEnv is not implemented, use chunk_step")

    def _infer_next_chunk_rewards(self):
        if self.reward_model is None:
            raise ValueError("Reward model is not loaded")

        extract_chunk_obs = self.current_obs.permute(
            0, 3, 1, 2, 4, 5
        )  # [B, T, 3, 1, H, W]
        extract_chunk_obs = extract_chunk_obs[:, -self.chunk :, :, :, :, :]
        extract_chunk_obs = extract_chunk_obs.reshape(self.num_envs * self.chunk, 3, 1, *self.image_size)
        extract_chunk_obs = extract_chunk_obs.squeeze(2).to(self.device)  # [B*chunk, 3, H, W]

        if self.cfg.reward_model.type == "ResnetRewModel":
            rewards = self.reward_model.predict_rew(extract_chunk_obs).reshape(
                self.num_envs, self.chunk
            )
        elif self.cfg.reward_model.type == "TaskEmbedResnetRewModel":
            instructions = []
            for env_idx in range(self.num_envs):
                instructions.extend([self.task_descriptions[env_idx]] * self.chunk)
            rewards = self.reward_model.predict_rew(
                extract_chunk_obs, instructions
            ).reshape(self.num_envs, self.chunk)
        elif self.cfg.reward_model.type == "RoboTwinT5CrossAttn":
            extract_chunk_obs_float = ((extract_chunk_obs + 1.0) / 2.0).float()
            instructions = []
            for env_idx in range(self.num_envs):
                instructions.extend([self.task_descriptions[env_idx]] * self.chunk)
            rewards = self.reward_model.compute_reward(
                extract_chunk_obs_float, task_descriptions=instructions
            ).reshape(self.num_envs, self.chunk)
        elif self.cfg.reward_model.type == "QwenVLMProgressRewardModel":
            extract_chunk_obs = ((extract_chunk_obs + 1.0) / 2.0).clamp(0.0, 1.0)
            instructions = []
            for env_idx in range(self.num_envs):
                instructions.extend([self.task_descriptions[env_idx]] * self.chunk)
            rewards = self.reward_model.compute_reward(
                extract_chunk_obs, task_descriptions=instructions
            ).reshape(self.num_envs, self.chunk)
        else:
            raise ValueError(f"Unknown reward model type: {self.cfg.reward_model.type}")

        return rewards

    def _infer_next_chunk_frames(self, actions):
        actions_tensor = (
            torch.from_numpy(actions).to(self.device)
            if isinstance(actions, np.ndarray)
            else actions.to(self.device)
        )
        actions_tensor = actions_tensor.float()
        if actions_tensor.shape != (self.num_envs, self.chunk, self.action_dim):
            raise ValueError(
                "Unexpected actions shape for IVideoGPTEnv, got "
                f"{tuple(actions_tensor.shape)}, expected "
                f"{(self.num_envs, self.chunk, self.action_dim)}"
            )

        cond_frames = self.current_obs[:, :, 0, -self.condition_frame_length :, :, :]
        cond_frames = cond_frames.permute(0, 2, 1, 3, 4)  # [B, T, 3, H, W]
        cond_frames = ((cond_frames + 1.0) / 2.0).clamp(0.0, 1.0)

        # NOTE: CompressiveVQModel.tokenize expects at least one future frame
        # (T > context_length). For online rollout we only have context frames
        # here, so append one duplicate frame just for tokenization.
        tokenize_input = torch.cat([cond_frames, cond_frames[:, -1:, ...]], dim=1)
        tokens, _ = self.tokenizer.tokenize(
            tokenize_input.to(self.device), self.condition_frame_length
        )
        # Keep one trailing SDF token in prelude, matching iVideoGPT inference path.
        gen_input = tokens[:, : self.prelude_tokens_num + 1]

        max_new_tokens = (1 + self.tokens_num_per_dyna) * self.chunk - 1
        generated_tokens = self.model.generate(
            gen_input,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_k=self.top_k,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.pad_token_id,
            action=actions_tensor,
        )

        recon_output = self.tokenizer.detokenize(
            generated_tokens, self.condition_frame_length
        ).clamp(0.0, 1.0)
        pred_frames = recon_output[:, self.condition_frame_length :, :, :, :]  # [B, chunk, 3, H, W]
        pred_frames = pred_frames.permute(0, 2, 1, 3, 4)  # [B, 3, chunk, H, W]
        pred_frames = pred_frames * 2.0 - 1.0
        pred_frames = pred_frames.unsqueeze(2)  # [B, 3, 1, chunk, H, W]

        self.current_obs = torch.cat([self.current_obs, pred_frames], dim=3)
        max_frames = self.condition_frame_length + self.chunk
        if self.current_obs.shape[3] > max_frames:
            self.current_obs = self.current_obs[:, :, :, -max_frames:, :, :]

    def _wrap_obs(self):
        last_frame = self.current_obs[:, :, 0, -1, :, :]  # [B, 3, H, W]
        full_image = last_frame.permute(0, 2, 3, 1)
        full_image = (full_image + 1.0) / 2.0 * 255.0
        full_image = torch.clamp(full_image, 0, 255).to(torch.uint8)

        states = torch.zeros(
            (self.num_envs, self.action_dim), device=self.device, dtype=torch.float32
        )
        obs = {
            "main_images": full_image,
            "wrist_images": None,
            "states": states,
            "task_descriptions": self.task_descriptions,
        }
        return obs

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

        if getattr(self.cfg, "print_chunk_rewards", False):
            with torch.no_grad():
                raw = chunk_rewards.cpu().numpy()
                diff = chunk_rewards_tensors.cpu().numpy()
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
        if self.elapsed_steps >= self.cfg.max_episode_steps:
            raw_chunk_truncations[:, -1] = True

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(past_dones, extracted_obs, {})
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
        self.tokenizer = self.tokenizer.to("cpu")
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
        self.tokenizer = self.tokenizer.to(self.device)
        self.model = self.model.to(self.device)
        self.reward_model = self.reward_model.to(self.device)
        self.current_obs = recursive_to_device(self.current_obs, self.device)
        self.prev_step_reward = self.prev_step_reward.to(self.device)
        self.reset_state_ids = self.reset_state_ids.to(self.device)
        if self.record_metrics:
            self.success_once = self.success_once.to(self.device)
            self.returns = self.returns.to(self.device)
        self._is_offloaded = False
