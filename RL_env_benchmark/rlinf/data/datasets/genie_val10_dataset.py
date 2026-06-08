# Copyright 2025 The RLinf Authors.
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

"""Dataset for the val10 (RoboTwin-format) dataset used by Genie world model.

Directory layout expected under ``val10_root``::

    <val10_root>/
      <task_name>/
        aloha-agilex_clean_50/aloha-agilex_clean_50/
          video/        episode0000.mp4, episode0001.mp4, ...
          actions/      episode0000.npy, ...          shape (T, 14)
          states/       episode0000.npy, ...          shape (T, 14)  [optional]
          instructions/ episode0000.json              {"instruction": "..."}

Each ``__getitem__`` returns a dict::

    {
        "image"      : np.ndarray  (H, W, 3) uint8  – first frame of the episode
        "actions"    : np.ndarray  (T, 14)   float32 – all actions
        "state"      : np.ndarray  (14,)     float32 – first-frame robot state
                                                       (zeros if states dir absent)
        "task"       : str                           – instruction string
        "task_name"  : str                           – task directory name
        "episode"    : str                           – episode file stem
    }
"""

from __future__ import annotations

import glob
import json
import os
from typing import Optional

import numpy as np
from torch.utils.data import Dataset


def _find_all_episodes(
    val10_root: str,
    task_filter: Optional[list[str]] = None,
) -> list[dict]:
    """Scan val10_root and return a list of episode metadata dicts.

    Mirrors ``find_all_episodes`` in rollout_policy_genie_closedloop.py.
    """
    episodes: list[dict] = []
    for task_dir in sorted(glob.glob(os.path.join(val10_root, "*"))):
        if not os.path.isdir(task_dir):
            continue
        task_name = os.path.basename(task_dir)
        if task_filter and task_name not in task_filter:
            continue

        data_dir = os.path.join(task_dir, "aloha-agilex_clean_50", "aloha-agilex_clean_50")
        if not os.path.isdir(data_dir):
            continue

        video_dir = os.path.join(data_dir, "video")
        actions_dir = os.path.join(data_dir, "actions")
        states_dir = os.path.join(data_dir, "states")
        instructions_dir = os.path.join(data_dir, "instructions")

        if not all(os.path.isdir(d) for d in [video_dir, actions_dir, instructions_dir]):
            continue

        for video_file in sorted(glob.glob(os.path.join(video_dir, "episode*.mp4"))):
            episode_name = os.path.splitext(os.path.basename(video_file))[0]
            action_file = os.path.join(actions_dir, f"{episode_name}.npy")
            state_file = os.path.join(states_dir, f"{episode_name}.npy")
            instruction_file = os.path.join(instructions_dir, f"{episode_name}.json")

            if not (os.path.exists(action_file) and os.path.exists(instruction_file)):
                continue

            episodes.append(
                {
                    "task_name": task_name,
                    "episode": episode_name,
                    "video_path": video_file,
                    "action_path": action_file,
                    "state_path": state_file if os.path.exists(state_file) else None,
                    "instruction_path": instruction_file,
                }
            )
    return episodes


class GenieVal10Dataset(Dataset):
    """PyTorch Dataset wrapping a val10-format directory.

    On first access the video is opened with ``decord`` and only the first
    frame is decoded, so construction is cheap (no full-video decode at init).

    Args:
        val10_root: Root directory of the val10 dataset.
        task_filter: Optional list of task names to include.  ``None`` = all.
        resize: If given, resize the first frame to ``(resize, resize)``
                before returning.  ``None`` = no resize.
    """

    def __init__(
        self,
        val10_root: str,
        task_filter: Optional[list[str]] = None,
        resize: Optional[int] = None,
    ) -> None:
        self.val10_root = val10_root
        self.resize = resize
        self.episodes = _find_all_episodes(val10_root, task_filter=task_filter)
        if not self.episodes:
            raise ValueError(
                f"No episodes found in val10_root={val10_root!r}. "
                "Check the directory layout."
            )

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> dict:
        ep = self.episodes[index]

        # ------------------------------------------------------------------
        # 1. Read first frame from video (cheap: decode only frame 0)
        # ------------------------------------------------------------------
        try:
            from decord import VideoReader, cpu  # noqa: PLC0415

            vr = VideoReader(ep["video_path"], ctx=cpu(0), num_threads=1)
            first_frame: np.ndarray = vr[0].asnumpy()  # (H, W, 3) uint8
        except ImportError as exc:
            raise ImportError(
                "decord is required to load val10 video files. "
                "Install it with: pip install decord"
            ) from exc

        if self.resize is not None:
            from PIL import Image  # noqa: PLC0415

            img_pil = Image.fromarray(first_frame).resize(
                (self.resize, self.resize), Image.BILINEAR
            )
            first_frame = np.array(img_pil)  # (resize, resize, 3) uint8

        # ------------------------------------------------------------------
        # 2. Load actions  shape: (T, action_dim)
        # ------------------------------------------------------------------
        actions: np.ndarray = np.load(ep["action_path"]).astype(np.float32)

        # ------------------------------------------------------------------
        # 3. Load robot state (first frame)  shape: (state_dim,)
        # ------------------------------------------------------------------
        if ep["state_path"] is not None and os.path.exists(ep["state_path"]):
            states = np.load(ep["state_path"]).astype(np.float32)
            state: np.ndarray = states[0] if states.ndim == 2 else states
        else:
            action_dim = actions.shape[1] if actions.ndim == 2 else actions.shape[0]
            state = np.zeros(action_dim, dtype=np.float32)

        # ------------------------------------------------------------------
        # 4. Load instruction
        # ------------------------------------------------------------------
        with open(ep["instruction_path"], "r") as f:
            instruction_data = json.load(f)
        instruction: str = instruction_data.get("instruction", "")

        return {
            "image": first_frame,          # (H, W, 3) uint8
            "actions": actions,            # (T, action_dim) float32
            "state": state,               # (action_dim,) float32
            "task": instruction,
            "task_name": ep["task_name"],
            "episode": ep["episode"],
        }
