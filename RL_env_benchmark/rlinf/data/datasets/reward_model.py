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

import os
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


@dataclass
class RewardDatasetPayload:
    """Canonical payload schema for processed reward dataset files."""

    images: list[torch.Tensor]
    labels: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.images) != len(self.labels):
            raise ValueError("Images and labels must have same length")
        self.labels = [int(v) for v in self.labels]

    def to_dict(self) -> dict[str, Any]:
        return {
            "images": self.images,
            "labels": self.labels,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], source: str = "<memory>"):
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid processed dataset payload from {source}")
        return cls(
            images=payload.get("images", []),
            labels=payload.get("labels", []),
            metadata=payload.get("metadata", {}),
        )

    def save(self, path: str) -> None:
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path: str):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls.from_dict(payload, source=path)


class RewardBinaryDataset(Dataset):
    """Dataset for binary classification reward model training.

    Uses per-frame 'is_obj_placed' field from infos to determine success/fail labels.
    This is more accurate than using episode-level labels from filenames.
    """

    def __init__(
        self,
        data_path: str,
    ):
        """Initialize dataset from a preprocessed .pt file.

        Args:
            data_path: Path to preprocessed dataset .pt file.

        Required payload schema is defined by `RewardDatasetPayload`.
        """
        payload = RewardDatasetPayload.load(data_path)
        self.images = payload.images
        self.labels = payload.labels
        self.metadata = payload.metadata

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get (image, label) pair.

        Returns:
            Tuple of (image tensor (C, H, W), label (0 or 1))
        """
        return self.images[idx], torch.tensor(self.labels[idx], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Text-conditioned reward dataset (images + instructions + labels)
# ---------------------------------------------------------------------------


@dataclass
class TextCondRewardDatasetPayload:
    """Payload schema for text-conditioned reward dataset files.

    Each sample associates an image tensor, a task-description string, and a
    binary success/fail label.
    """

    images: list[torch.Tensor]
    instructions: list[str]
    labels: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (len(self.images) == len(self.instructions) == len(self.labels)):
            raise ValueError(
                "images, instructions and labels must all have the same length."
            )
        self.labels = [int(v) for v in self.labels]

    def to_dict(self) -> dict[str, Any]:
        return {
            "images": self.images,
            "instructions": self.instructions,
            "labels": self.labels,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], source: str = "<memory>"):
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid text-cond reward dataset payload from {source}")
        return cls(
            images=payload.get("images", []),
            instructions=payload.get("instructions", []),
            labels=payload.get("labels", []),
            metadata=payload.get("metadata", {}),
        )

    def save(self, path: str) -> None:
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path: str):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls.from_dict(payload, source=path)


def text_cond_reward_collate_fn(
    batch: list[tuple[torch.Tensor, str, torch.Tensor]],
) -> tuple[torch.Tensor, list[str], torch.Tensor]:
    """Collate function for TextCondRewardBinaryDataset.

    Handles the mixed-type batch ``(tensor, str, tensor)`` that the default
    PyTorch collate cannot process.

    Args:
        batch: List of ``(image, instruction, label)`` tuples.

    Returns:
        Tuple of stacked images ``(B, C, H, W)``, list of instruction strings
        of length ``B``, and stacked labels ``(B,)``.
    """
    images = default_collate([item[0] for item in batch])
    instructions = [item[1] for item in batch]
    labels = default_collate([item[2] for item in batch])
    return images, instructions, labels


class TextCondRewardBinaryDataset(Dataset):
    """Dataset for text-conditioned binary reward model training.

    Each item is a triple ``(image_tensor, instruction_str, label_tensor)``.
    Use ``text_cond_reward_collate_fn`` as the DataLoader collate function.
    """

    def __init__(self, data_path: str) -> None:
        """Load a preprocessed ``.pt`` file produced by
        ``preprocess_robotwin_reward_dataset.py``.

        Args:
            data_path: Path to a file saved by
                ``TextCondRewardDatasetPayload.save``.
        """
        payload = TextCondRewardDatasetPayload.load(data_path)
        self.images: list[torch.Tensor] = payload.images
        self.instructions: list[str] = payload.instructions
        self.labels: list[int] = payload.labels
        self.metadata: dict[str, Any] = payload.metadata

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, torch.Tensor]:
        """Return ``(image, instruction, label)``."""
        return (
            self.images[idx],
            self.instructions[idx],
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )
