#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Visualize action distribution per dimension for two RoboTwin datasets.

Example:
    python toolkits/replay_buffer/visualize_robotwin_action_distribution.py \
        --dataset_a /your/path//Robotwin2.0Data/dataset/adjust_bottle/aloha-agilex_clean_50 \
        --dataset_b /manifold-obs/wzl/vla_robotwin_4k_320/ref/adjust_bottle/demo_clean \
        --output_dir /tmp/robotwin_action_dist
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATASET_A = (
    "/your/path//Robotwin2.0Data/dataset/"
    "adjust_bottle/aloha-agilex_clean_50"
)
DEFAULT_DATASET_B = (
    "/manifold-obs/wzl/vla_robotwin_4k_320/ref/adjust_bottle/demo_clean"
)

ACTION_KEY_CANDIDATES = (
    "joint_action/vector",
    "action/vector",
    "action",
    "actions",
)


def _resolve_data_dir(dataset_root: Path) -> Path:
    data_dir = dataset_root / "data"
    if data_dir.exists():
        return data_dir
    return dataset_root


def _find_episode_files(dataset_root: Path) -> list[Path]:
    data_dir = _resolve_data_dir(dataset_root)
    episode_files = sorted(
        data_dir.glob("episode*.hdf5"),
        key=lambda p: int(p.stem.replace("episode", "")) if p.stem.replace("episode", "").isdigit() else p.stem,
    )
    if not episode_files:
        raise FileNotFoundError(f"No episode*.hdf5 found in {data_dir}")
    return episode_files


def _discover_action_key(h5_file: h5py.File) -> str:
    for key in ACTION_KEY_CANDIDATES:
        if key in h5_file:
            return key

    discovered: list[str] = []

    def _collector(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if isinstance(obj, h5py.Dataset) and "action" in name.lower():
            discovered.append(name)

    h5_file.visititems(_collector)
    if discovered:
        return discovered[0]

    raise KeyError(
        "Cannot find action dataset. Tried keys: "
        f"{ACTION_KEY_CANDIDATES} and auto-discovery."
    )


def _load_action_from_episode(episode_path: Path) -> np.ndarray:
    with h5py.File(episode_path, "r") as f:
        action_key = _discover_action_key(f)
        actions = np.asarray(f[action_key][:], dtype=np.float32)

    if actions.ndim == 1:
        actions = actions[:, None]
    elif actions.ndim > 2:
        actions = actions.reshape(actions.shape[0], -1)
    return actions


def load_all_actions(
    dataset_root: Path,
    max_episodes: int | None,
) -> np.ndarray:
    episode_files = _find_episode_files(dataset_root)
    if max_episodes is not None:
        episode_files = episode_files[:max_episodes]

    actions: list[np.ndarray] = []
    for episode_file in tqdm(
        episode_files,
        desc=f"Loading actions from {dataset_root.name}",
        unit="episode",
    ):
        episode_actions = _load_action_from_episode(episode_file)
        actions.append(episode_actions)

    if not actions:
        raise RuntimeError(f"No action loaded from {dataset_root}")
    return np.concatenate(actions, axis=0)


def _compute_stats(actions: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": np.mean(actions, axis=0).tolist(),
        "std": np.std(actions, axis=0).tolist(),
        "min": np.min(actions, axis=0).tolist(),
        "max": np.max(actions, axis=0).tolist(),
        "q01": np.percentile(actions, 1, axis=0).tolist(),
        "q99": np.percentile(actions, 99, axis=0).tolist(),
    }


def _plot_per_dimension_distribution(
    actions_a: np.ndarray,
    actions_b: np.ndarray,
    name_a: str,
    name_b: str,
    bins: int,
    output_png: Path,
) -> None:
    num_dims = min(actions_a.shape[1], actions_b.shape[1])
    cols = 4
    rows = math.ceil(num_dims / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.8, rows * 3.6))
    if isinstance(axes, np.ndarray):
        axes_flat = axes.reshape(-1)
    else:
        axes_flat = np.array([axes])

    for dim in range(num_dims):
        ax = axes_flat[dim]
        ax.hist(
            actions_a[:, dim],
            bins=bins,
            density=True,
            alpha=0.45,
            label=name_a,
            color="#1f77b4",
        )
        ax.hist(
            actions_b[:, dim],
            bins=bins,
            density=True,
            alpha=0.45,
            label=name_b,
            color="#ff7f0e",
        )
        ax.set_title(f"action[{dim}]")
        ax.grid(alpha=0.25, linestyle="--")

    for idx in range(num_dims, len(axes_flat)):
        axes_flat[idx].axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Action Distribution Per Dimension", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize action distribution per dimension for two datasets."
    )
    parser.add_argument("--dataset_a", type=Path, default=Path(DEFAULT_DATASET_A))
    parser.add_argument("--dataset_b", type=Path, default=Path(DEFAULT_DATASET_B))
    parser.add_argument("--name_a", type=str, default="aloha-agilex_clean_50")
    parser.add_argument("--name_b", type=str, default="demo_clean")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/action_distribution/adjust_bottle"),
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Optionally use first N episodes from each dataset.",
    )
    parser.add_argument("--bins", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    actions_a = load_all_actions(args.dataset_a, args.max_episodes)
    actions_b = load_all_actions(args.dataset_b, args.max_episodes)

    if actions_a.shape[1] != actions_b.shape[1]:
        print(
            "Warning: action dims mismatch "
            f"{actions_a.shape[1]} vs {actions_b.shape[1]}; "
            "only plotting common dimensions."
        )

    plot_path = args.output_dir / "action_distribution_compare.png"
    stats_path = args.output_dir / "action_stats_compare.json"

    _plot_per_dimension_distribution(
        actions_a=actions_a,
        actions_b=actions_b,
        name_a=args.name_a,
        name_b=args.name_b,
        bins=args.bins,
        output_png=plot_path,
    )

    payload = {
        "dataset_a": str(args.dataset_a),
        "dataset_b": str(args.dataset_b),
        "name_a": args.name_a,
        "name_b": args.name_b,
        "shape_a": list(actions_a.shape),
        "shape_b": list(actions_b.shape),
        "stats_a": _compute_stats(actions_a),
        "stats_b": _compute_stats(actions_b),
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Loaded actions: A={actions_a.shape}, B={actions_b.shape}")
    print(f"Saved figure: {plot_path}")
    print(f"Saved stats : {stats_path}")


if __name__ == "__main__":
    main()
