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

"""LPIPS-based reward model for world-model environments."""

from __future__ import annotations

from typing import Any, Optional

import torch
from omegaconf import DictConfig

from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel


class LPIPSLastFrameRewardModel(BaseRewardModel):
    """Reward generated frames by LPIPS distance to the GT last frame."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        try:
            import lpips
        except ImportError as exc:
            raise ImportError(
                "LPIPSLastFrameRewardModel requires `lpips`. "
                "Please install dependency `lpips` first."
            ) from exc

        self.lpips_net = str(cfg.get("lpips_net", "vgg"))
        self.reward_transform = str(cfg.get("reward_transform", "one_minus")).lower()
        if self.reward_transform not in {"one_minus", "negative", "raw"}:
            raise ValueError(
                f"Unsupported reward_transform: {self.reward_transform}. "
                "Expected one of {'one_minus', 'negative', 'raw'}."
            )

        self.lpips_model = lpips.LPIPS(net=self.lpips_net).eval()

    def _to_nchw_minus1_1(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 4:
            raise ValueError(f"Expected rank-4 tensor, got shape {tuple(images.shape)}")
        if images.shape[-1] in [1, 3, 4]:
            images = images.permute(0, 3, 1, 2)
        images = images.float()
        if images.dtype == torch.uint8 or images.max() > 1.5:
            images = images / 255.0
        if images.min() >= 0.0:
            images = images * 2.0 - 1.0
        return images.clamp(-1.0, 1.0)

    def forward(
        self,
        input_data: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        rewards = self.compute_reward(input_data, references=labels)
        device = rewards.device
        return {
            "loss": torch.tensor(0.0, device=device),
            "accuracy": torch.tensor(0.0, device=device),
            "logits": rewards,
            "probabilities": rewards,
        }

    @torch.no_grad()
    def compute_reward(
        self,
        observations: torch.Tensor,
        task_descriptions: Optional[list[str]] = None,
        references: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if references is None:
            raise ValueError(
                "LPIPSLastFrameRewardModel.compute_reward requires `references` "
                "for GT last frames."
            )
        if observations.shape[0] != references.shape[0]:
            raise ValueError(
                "Batch size mismatch between observations and references: "
                f"{observations.shape[0]} vs {references.shape[0]}"
            )

        obs = self._to_nchw_minus1_1(observations)
        ref = self._to_nchw_minus1_1(references).to(obs.device)
        dist = self.lpips_model(obs, ref).view(-1)

        if self.reward_transform == "raw":
            reward = dist
        elif self.reward_transform == "negative":
            reward = -dist
        else:
            reward = 1.0 - dist.clamp(0.0, 1.0)
        return reward.float()
