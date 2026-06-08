# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Convert RoboTwin-style HDF5 episodes into LeRobot v2.x datasets.

Source layout (per task):
    <src_root>/<task>/demo_clean/
        data/episode*.hdf5
        instructions/episode*.json

Each HDF5 episode contains joint_action/vector (T,14) and
observation/{head,left,right,front}_camera/rgb (T,) of JPEG bytes.

For single-view pi0/pi05 SFT we pair this converter with
``pi05_aloha_robotwin_head`` (see
rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_head_dataconfig.py):
only the head camera is emitted as ``observation.images.cam_high``;
AlohaInputs fills the remaining two wrist slots with zeros and masks
them out at train time.

Output (per task) is a LeRobot v2 dataset at
    <out_root>/<repo_id>/
which is how HF_LEROBOT_HOME + repo_id resolve at training time.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def _set_hf_home(out_root: str) -> None:
    # Must be set BEFORE importing lerobot, since HF_LEROBOT_HOME is captured
    # at module import time.
    os.environ["HF_LEROBOT_HOME"] = out_root


def _decode_jpeg(buf: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(bytes(buf))).convert("RGB")
    return np.asarray(img, dtype=np.uint8)  # (H, W, 3)


def _pick_prompt(inst_path: Path, rng: random.Random) -> str:
    with open(inst_path, "r") as f:
        d = json.load(f)
    pool = list(d.get("seen", []))
    if not pool:
        pool = list(d.get("unseen", []))
    if not pool:
        raise RuntimeError(f"No prompt found in {inst_path}")
    return rng.choice(pool)


def build_features(height: int, width: int) -> dict:
    img_feat = {
        "dtype": "image",
        "shape": (height, width, 3),
        "names": ["height", "width", "channels"],
    }
    return {
        "observation.images.cam_high": img_feat,
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": [f"s{i}" for i in range(14)],
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": [f"a{i}" for i in range(14)],
        },
    }


def convert_task(
    src_task_dir: Path,
    out_root: Path,
    repo_id: str,
    fps: int,
    max_episodes: int | None,
    seed: int,
) -> None:
    _set_hf_home(str(out_root))

    # Import AFTER setting HF_LEROBOT_HOME.
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    data_dir = src_task_dir / "demo_clean" / "data"
    inst_dir = src_task_dir / "demo_clean" / "instructions"

    ep_files = sorted(
        data_dir.glob("episode*.hdf5"),
        key=lambda p: int(p.stem.replace("episode", "")),
    )
    if max_episodes is not None:
        ep_files = ep_files[:max_episodes]
    if not ep_files:
        raise RuntimeError(f"No episodes found under {data_dir}")

    # Peek the first frame to infer image size.
    with h5py.File(ep_files[0], "r") as f:
        sample = _decode_jpeg(f["observation/head_camera/rgb"][0])
    H, W, _ = sample.shape

    # The target dataset directory is <HF_LEROBOT_HOME>/<repo_id>/.
    dataset_dir = out_root / repo_id
    if dataset_dir.exists():
        raise RuntimeError(
            f"{dataset_dir} already exists; delete it or pick a fresh repo_id."
        )

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=build_features(H, W),
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4,
    )

    rng = random.Random(seed)

    for ep_path in ep_files:
        inst_path = inst_dir / (ep_path.stem + ".json")
        prompt = _pick_prompt(inst_path, rng)

        with h5py.File(ep_path, "r") as f:
            actions = np.asarray(f["joint_action/vector"][:], dtype=np.float32)  # (T,14)
            head_rgb = f["observation/head_camera/rgb"][:]  # (T,) bytes
            T = actions.shape[0]
            assert head_rgb.shape[0] == T, f"length mismatch in {ep_path}"

            for t in range(T):
                img = _decode_jpeg(head_rgb[t])
                frame = {
                    "observation.images.cam_high": img,
                    # No explicit proprio in the source, so feed zeros of the
                    # same dimension as the action. pi05_aloha_robotwin with
                    # discrete_state_input=True does not consume this anyway.
                    "observation.state": np.zeros(14, dtype=np.float32),
                    "action": actions[t],
                    "task": prompt,
                }
                ds.add_frame(frame)

        ds.save_episode()
        print(f"  saved {ep_path.name}  T={T}")

    print(f"Done. Dataset at {dataset_dir}  episodes={ds.num_episodes}  frames={ds.num_frames}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src_root",
        default="/manifold-obs/wzl/vla_robotwin_4k_320/ref",
        help="Directory containing <task>/demo_clean/...",
    )
    ap.add_argument(
        "--out_root",
        default="/your/path//RLinf/datasets/lerobot",
        help="HF_LEROBOT_HOME for the produced dataset(s).",
    )
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=["adjust_bottle", "click_bell"],
        help="Task subdirs to convert.",
    )
    ap.add_argument(
        "--repo_id_prefix",
        default="rlinf/robotwin_headcam",
        help="repo_id is <prefix>_<task> (must be <org>/<name>).",
    )
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--max_episodes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        repo_id = f"{args.repo_id_prefix}_{task}"
        print(f"\n=== converting {task} -> {out_root / repo_id} ===")
        convert_task(
            src_task_dir=src_root / task,
            out_root=out_root,
            repo_id=repo_id,
            fps=args.fps,
            max_episodes=args.max_episodes,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
