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

"""Preprocess RoboTwin HDF5 demo data for text-conditioned reward model training.

This script reads the episode HDF5 files produced by the RoboTwin data
collection pipeline, extracts RGB frames together with task-description
instructions and per-episode success labels, and saves train/val ``.pt``
split files in the :class:`TextCondRewardDatasetPayload` format.

Data directory layout expected (per data root)::

    <data_root>/
        data/
            episode0.hdf5
            episode1.hdf5
            ...
        instructions/
            episode0.json      # {"seen": ["..."], "unseen": ["..."]}
            episode1.json
            ...
        scene_info.json        # per-episode metadata including `success` flag

Each HDF5 file contains:
    observation/head_camera/rgb – (T,) array of JPEG-encoded bytes strings
    joint_action/vector         – (T, 14) float64 joint actions

Labelling strategy
------------------
* **Successful episodes** – the *last 5 %* of frames are labelled ``1``
  (task completed); frames sampled from the *first 95 %* of the episode are
  labelled ``0`` (in-progress, not yet done).
* **Failed episodes** – frames sampled from the episode are labelled ``0``.

This produces a balanced binary classification dataset: does this frame show
the completed task state?

Usage example::

    # Edit DATA_ROOTS and OUTPUT_DIR in this file, then run:
    python examples/reward/preprocess_robotwin_reward_dataset.py
"""

import io
import json
import os
import random
from glob import glob
from typing import Optional

import torch
from PIL import Image
from tqdm import tqdm
from rlinf.data.datasets.reward_model import TextCondRewardDatasetPayload
from rlinf.utils.logging import get_logger

logger = get_logger()

# ============================================================================
# Hard-coded configuration – edit these before running
# ============================================================================

# List of data-root directories, each containing data/, instructions/, scene_info.json
DATA_ROOTS: list[str] = [
    "/manifold-obs/wzl/vla_robotwin_4k_320/10radiodata_10000/adjust_bottle/demo_clean",
    "/manifold-obs/wzl/vla_robotwin_4k_320/fulldata_40000/adjust_bottle/demo_clean"
]

# Output directory for processed train.pt / val.pt
OUTPUT_DIR: str = "/manifold-obs/tangyinzhou/RLinf/logs/robotwin_reward_data_adjust_bottle"

# Fraction of samples reserved for validation
VAL_SPLIT: float = 0.2

# Frames to sample from each failed episode as negatives (0 = use all frames)
NUM_FAIL_FRAMES: int = 40

# In-progress frames to sample from each successful episode as negatives
NUM_SUCCESS_NEG_FRAMES: int = 20

# Random seed for deterministic shuffling
SEED: int = 42

# Debug mode: if True, only process DEBUG_EPISODES_PER_ROOT episodes per data root
DEBUG_MODE: bool = False
DEBUG_EPISODES_PER_ROOT: int = 10

# ============================================================================

# Target spatial resolution fed to the reward model.
_TARGET_H: int = 224
_TARGET_W: int = 224


# ---------------------------------------------------------------------------
# Image decoding helpers
# ---------------------------------------------------------------------------


def _decode_jpeg(raw: bytes) -> torch.Tensor:
    """Decode JPEG bytes → float32 RGB tensor ``(3, H, W)`` in [0, 1].

    Args:
        raw: Raw JPEG bytes (as read from HDF5 ``|S…`` dataset entry).

    Returns:
        Float32 tensor of shape ``(3, H, W)`` with values in ``[0, 1]``.
    """
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    # Resize to target resolution.
    if img.size != (_TARGET_W, _TARGET_H):
        img = img.resize((_TARGET_W, _TARGET_H), Image.BILINEAR)
    tensor = torch.from_numpy(
        # PIL gives HWC uint8 numpy array.
        __import__("numpy").asarray(img, dtype="float32") / 255.0
    )  # (H, W, 3)
    return tensor.permute(2, 0, 1).contiguous()  # (3, H, W)


# ---------------------------------------------------------------------------
# Episode loading
# ---------------------------------------------------------------------------


def _load_scene_info(data_root: str) -> dict:
    """Load and return the scene_info.json mapping.

    Returns a dict keyed by ``"episode_N"`` whose values contain at least
    ``success`` (bool) and ``instruction`` (str).
    """
    scene_info_path = os.path.join(data_root, "scene_info.json")
    if not os.path.exists(scene_info_path):
        raise FileNotFoundError(f"scene_info.json not found in {data_root}")
    with open(scene_info_path) as fh:
        return json.load(fh)


def _load_instruction(data_root: str, episode_idx: int) -> str:
    """Load instruction for the given episode from the instructions/ dir.

    Falls back to an empty string when the file is missing.
    """
    instr_path = os.path.join(
        data_root, "instructions", f"episode{episode_idx}.json"
    )
    if not os.path.exists(instr_path):
        return ""
    with open(instr_path) as fh:
        data = json.load(fh)
    # Use the "seen" instruction variant when available, else fall back.
    seen = data.get("seen", [])
    return seen[0] if seen else data.get("unseen", [""])[0]


def _sample_indices(n: int, k: int, keep_last: bool = False) -> list[int]:
    """Return at most ``k`` evenly-spaced indices from ``[0, n)``.

    When ``keep_last`` is ``True`` the last index (``n-1``) is always included
    regardless of spacing.
    """
    if k <= 0 or k >= n:
        return list(range(n))
    if keep_last:
        extra = [n - 1]
        remainder = k - 1
        non_last_n = n - 1
        if remainder <= 0:
            return extra
        step = max(1, non_last_n // remainder)
        indices = sorted(set(list(range(0, non_last_n, step))[:remainder] + extra))
    else:
        step = max(1, n // k)
        indices = sorted(set(range(0, n, step)))[:k]
    return indices


def load_robotwin_episodes(
    data_root: str,
    num_fail_frames_per_episode: int = 5,
    num_success_neg_frames: int = 3,
) -> list[dict]:
    """Load all episodes from one RoboTwin demo directory.

    For each episode the function extracts:
    * The *last 5 %* of frames as positive samples (``label=1``) for
      successful episodes.
    * A handful of in-progress frames from the *first 95 %* as negative
      samples (``label=0``).

    Args:
        data_root: Root directory of the demo split (contains ``data/``,
            ``instructions/``, and ``scene_info.json``).
        num_fail_frames_per_episode: How many frames to sample from failed
            episodes as negatives.  Set to ``0`` to use all frames.
        num_success_neg_frames: How many *early* frames from successful
            episodes to include as negatives (in-progress state).

    Returns:
        List of dicts, each with keys
        ``{"image": tensor, "instruction": str, "label": int}``.
    """
    try:
        import h5py  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "h5py is required for preprocessing RoboTwin HDF5 data.  "
            "Install it with: pip install h5py"
        ) from exc

    import h5py

    scene_info = _load_scene_info(data_root)
    data_dir = os.path.join(data_root, "data")
    hdf5_files = sorted(glob(os.path.join(data_dir, "episode*.hdf5")))

    if not hdf5_files:
        raise ValueError(f"No episode HDF5 files found in {data_dir}")

    # Debug mode: limit episodes per data root
    if DEBUG_MODE:
        hdf5_files = hdf5_files[:DEBUG_EPISODES_PER_ROOT]
        logger.info(f"[DEBUG MODE] Limited to {len(hdf5_files)} episodes from {data_dir}")
    else:
        logger.info(f"Found {len(hdf5_files)} episode files in {data_dir}")

    samples: list[dict] = []
    loaded = 0
    skipped = 0

    for hdf5_path in tqdm(hdf5_files):
        basename = os.path.splitext(os.path.basename(hdf5_path))[0]  # "episodeN"
        # Map "episodeN" → "episode_N" used in scene_info keys.
        ep_idx_str = basename.replace("episode", "")
        scene_key = f"episode_{ep_idx_str}"
        ep_idx = int(ep_idx_str)

        meta = scene_info.get(scene_key)
        if meta is None:
            logger.debug(f"No scene_info entry for {scene_key}, skipping.")
            skipped += 1
            continue

        success: bool = bool(meta.get("success", False))

        # Prefer instruction from JSON file; fall back to scene_info field.
        instruction: str = _load_instruction(data_root, ep_idx)
        if not instruction:
            instruction = meta.get("instruction", "")

        try:
            with h5py.File(hdf5_path, "r") as hf:
                rgb_dataset = hf["observation/head_camera/rgb"]
                T = len(rgb_dataset)

                if T == 0:
                    skipped += 1
                    continue

                if success:
                    # --- Positive samples: last 5% of episode ---
                    pos_start = max(0, int(T * 0.95))
                    for i in range(pos_start, T):
                        raw = bytes(rgb_dataset[i])
                        samples.append(
                            {
                                "image": _decode_jpeg(raw),
                                "instruction": instruction,
                                "label": 1,
                            }
                        )

                    # --- Negative samples: first 95% of episode ---
                    neg_end = max(1, pos_start)  # first 95 %
                    neg_indices = _sample_indices(
                        neg_end,
                        num_success_neg_frames,
                        keep_last=False,
                    )
                    for i in neg_indices:
                        raw = bytes(rgb_dataset[i])
                        samples.append(
                            {
                                "image": _decode_jpeg(raw),
                                "instruction": instruction,
                                "label": 0,
                            }
                        )
                else:
                    # --- Failed episode: all sampled frames are negatives ---
                    fail_indices = _sample_indices(
                        T,
                        num_fail_frames_per_episode,
                        keep_last=True,
                    )
                    for i in fail_indices:
                        raw = bytes(rgb_dataset[i])
                        samples.append(
                            {
                                "image": _decode_jpeg(raw),
                                "instruction": instruction,
                                "label": 0,
                            }
                        )

            loaded += 1

        except Exception as exc:
            logger.warning(f"Failed to load {hdf5_path}: {exc}")
            skipped += 1

    total = len(samples)
    pos = sum(s["label"] for s in samples)
    neg = total - pos
    logger.info(
        f"Loaded {loaded} episodes ({skipped} skipped). "
        f"Samples: {total} total – {pos} success (label=1), {neg} fail (label=0)"
    )
    return samples


def load_all_robotwin_episodes(
    data_roots: list[str],
    num_fail_frames_per_episode: int = 5,
    num_success_neg_frames: int = 3,
) -> list[dict]:
    """Load all episodes from multiple RoboTwin demo directories.

    Args:
        data_roots: List of root directories.
        num_fail_frames_per_episode: Frames per failed episode.
        num_success_neg_frames: In-progress frames per successful episode.

    Returns:
        Combined list of samples from all data roots.
    """
    all_samples: list[dict] = []

    for data_root in data_roots:
        logger.info(f"Processing data root: {data_root}")
        samples = load_robotwin_episodes(
            data_root=data_root,
            num_fail_frames_per_episode=num_fail_frames_per_episode,
            num_success_neg_frames=num_success_neg_frames,
        )
        all_samples.extend(samples)

    total = len(all_samples)
    pos = sum(s["label"] for s in all_samples)
    neg = total - pos
    logger.info(
        f"All data roots combined – "
        f"{total} total samples: {pos} success (label=1), {neg} fail (label=0)"
    )
    return all_samples


# ---------------------------------------------------------------------------
# Split and save
# ---------------------------------------------------------------------------


def split_and_save(
    samples: list[dict],
    train_output_path: str,
    val_output_path: str,
    val_split: float = 0.2,
    random_seed: Optional[int] = 42,
) -> dict:
    """Shuffle, split and save train/val ``.pt`` files.

    Args:
        samples: List of ``{"image", "instruction", "label"}`` dicts.
        train_output_path: Destination path for the train split ``.pt`` file.
        val_output_path: Destination path for the val split ``.pt`` file.
        val_split: Fraction of samples reserved for validation.
        random_seed: Seed for deterministic shuffling.

    Returns:
        Metadata dict summarising the split.
    """
    rng = random.Random(random_seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_split))
    val_samples = shuffled[:n_val]
    train_samples = shuffled[n_val:]

    def _save(split_samples: list[dict], path: str, split_name: str) -> None:
        imgs = [s["image"] for s in split_samples]
        instrs = [s["instruction"] for s in split_samples]
        lbls = [s["label"] for s in split_samples]
        pos = sum(lbls)
        payload = TextCondRewardDatasetPayload(
            images=imgs,
            instructions=instrs,
            labels=lbls,
            metadata={
                "split": split_name,
                "num_samples": len(split_samples),
                "num_positive": pos,
                "num_negative": len(split_samples) - pos,
                "target_image_size": [_TARGET_H, _TARGET_W],
            },
        )
        payload.save(path)
        logger.info(
            f"Saved {split_name} split → {path}  "
            f"({len(split_samples)} samples: {pos} pos / {len(split_samples)-pos} neg)"
        )

    _save(train_samples, train_output_path, "train")
    _save(val_samples, val_output_path, "val")

    return {
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "train_pos": sum(s["label"] for s in train_samples),
        "val_pos": sum(s["label"] for s in val_samples),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_path = os.path.join(OUTPUT_DIR, "train.pt")
    val_path = os.path.join(OUTPUT_DIR, "val.pt")

    samples = load_all_robotwin_episodes(
        data_roots=DATA_ROOTS,
        num_fail_frames_per_episode=NUM_FAIL_FRAMES,
        num_success_neg_frames=NUM_SUCCESS_NEG_FRAMES,
    )

    if not samples:
        raise RuntimeError(f"No samples extracted from data roots: {DATA_ROOTS}")

    meta = split_and_save(
        samples=samples,
        train_output_path=train_path,
        val_output_path=val_path,
        val_split=VAL_SPLIT,
        random_seed=SEED,
    )

    print("=" * 72)
    print("RoboTwin reward dataset preprocessing complete")
    print(f"  Data roots: {len(DATA_ROOTS)} directories")
    for dr in DATA_ROOTS:
        print(f"    - {dr}")
    print(f"  Train: {train_path}  ({meta['num_train']} samples, {meta['train_pos']} pos)")
    print(f"  Val  : {val_path}  ({meta['num_val']} samples, {meta['val_pos']} pos)")
    print("=" * 72)


if __name__ == "__main__":
    main()
