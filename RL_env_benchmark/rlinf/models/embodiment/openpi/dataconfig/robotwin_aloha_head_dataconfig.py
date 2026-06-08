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

"""Single-view RoboTwin / Aloha data config for pi0 / pi05.

Unlike :class:`LeRobotAlohaDataConfig`, this config only feeds the head
camera to the model. The pi0/pi05 backbone always has three image slots;
:class:`AlohaInputs` handles a missing wrist view by filling it with a
zero image and setting the corresponding ``image_mask`` to False, so we
simply never emit the wrist keys from the dataset.
"""

import dataclasses
import pathlib

import numpy as np
import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import aloha_policy


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaHeadOnlyDataConfig(DataConfigFactory):
    """Aloha/RoboTwin data config that uses a single (head) camera view."""

    # Default prompt to use if the dataset does not contain prompt information.
    default_prompt: str | None = None

    # If True, converts absolute joint actions to delta actions (relative to
    # the current state). Kept aligned with the multi-view RoboTwin config.
    extra_delta_transform: bool = True

    # If True, remaps data into the internal Pi0 space (joint flip, gripper
    # scaling). Leave False for standard RoboTwin data.
    adapt_to_pi: bool = False

    # Only the head camera is repacked. cam_left_wrist / cam_right_wrist are
    # intentionally omitted; AlohaInputs will substitute zeros and mask them.
    repack_transforms: _transforms.Group = dataclasses.field(
        default_factory=lambda: _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.cam_high",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
    )

    def generate_observations(
        self, image: np.ndarray, state: np.ndarray, prompt: str
    ) -> dict:
        return {
            "observation/image": image,
            "observation/state": state,
            "prompt": prompt,
        }

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )

        # Same 14-dim mask as the multi-view RoboTwin config:
        # [Left arm 6 joints (delta), left gripper (abs),
        #  Right arm 6 joints (delta), right gripper (abs)].
        if self.extra_delta_transform:
            delta_action_mask = np.array(
                [True] * 6 + [False] + [True] * 6 + [False],
                dtype=bool,
            )
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )
