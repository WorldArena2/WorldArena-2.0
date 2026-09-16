"""
Offline action evaluation for rollout predictions.

Metrics:
  1. Action MSE: per-frame MSE averaged over trajectory.
  2. NATSR (Normalized Action Trajectory Success Rate):
     Inspired by MANIPTRANS (2025) trajectory-level thresholding.
     - Normalize per-dimension error by GT range.
     - A trajectory (or half-trajectory) is "successful" only if ALL frames
       have ALL dimension errors below the threshold.
     - Each episode is split into first/second half as two independent samples.

Usage:
    python metric/eval_action_offline.py \
        --pred_dir /path/to/pred_actions \
        --gt_dir /path/to/gt_actions \
        --output_json action_metrics.json \
        --natsr_threshold 0.1
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm


def compute_action_mse(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute mean squared error between pred and gt actions."""
    return float(np.mean((pred - gt) ** 2))


def compute_natsr(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.15) -> float:
    """
    Compute Normalized Action Trajectory Success Rate for a single trajectory segment.

    We use absolute error directly (no range normalization) because GT action ranges
    can be extremely small for some dimensions (e.g. gripper state), which would
    over-inflate normalized errors. The threshold is an absolute bound on per-dim error.

    A segment is successful only if every frame's every dimension is below threshold.

    Args:
        pred: [T, D] predicted action
        gt:   [T, D] ground-truth action
        threshold: absolute error threshold (default 0.15)

    Returns:
        1.0 if successful, 0.0 otherwise
    """
    error = np.abs(pred - gt)  # [T, D]
    per_frame_success = np.all(error < threshold, axis=1)  # [T]
    return float(np.all(per_frame_success))


def evaluate_episode(
    pred_path: Path,
    gt_path: Path,
    threshold: float,
) -> Dict:
    """Evaluate a single episode."""
    pred = np.load(pred_path).astype(np.float32)  # [T, 22]
    gt = np.load(gt_path).astype(np.float32)       # [T, 7]

    # Take first 7 dims of pred to align with GT
    pred = pred[:, :7]

    # Align length (truncate to min length)
    min_len = min(len(pred), len(gt))
    pred = pred[:min_len]
    gt = gt[:min_len]

    # Action MSE
    mse = compute_action_mse(pred, gt)

    # NATSR: split into first and second half
    mid = min_len // 2
    natsr_first = compute_natsr(pred[:mid], gt[:mid], threshold)
    natsr_second = compute_natsr(pred[mid:], gt[mid:], threshold)

    return {
        "length": min_len,
        "mse": mse,
        "natsr_first_half": natsr_first,
        "natsr_second_half": natsr_second,
    }


def main():
    parser = argparse.ArgumentParser(description="Offline action evaluation")
    parser.add_argument("--dataroot", type=str, required=True,
                        help="Root directory containing pred_actions/ and gt_actions/ subdirectories")
    parser.add_argument("--output_json", type=str, default="action_metrics.json")
    parser.add_argument("--natsr_threshold", type=float, default=0.1,
                        help="Normalized error threshold for NATSR (default: 0.1)")
    args = parser.parse_args()

    dataroot = Path(args.dataroot)
    pred_dir = dataroot / "pred_actions"
    gt_dir = dataroot / "gt_actions"

    if not pred_dir.exists():
        print(f"Error: pred_dir not found: {pred_dir}")
        return
    if not gt_dir.exists():
        print(f"Error: gt_dir not found: {gt_dir}")
        return

    pred_files = sorted(pred_dir.glob("*_pred_action.npy"))
    if len(pred_files) == 0:
        print(f"No pred files found in {pred_dir}")
        return

    results: List[Dict] = []
    all_mse: List[float] = []
    all_natsr_samples: List[float] = []

    for pred_path in tqdm(pred_files, desc="Evaluating episodes"):
        episode_name = pred_path.name.replace("_pred_action.npy", "")
        gt_path = gt_dir / f"{episode_name}_gt_actions.npy"

        if not gt_path.exists():
            print(f"  Warning: GT not found for {episode_name}, skipping")
            continue

        metrics = evaluate_episode(pred_path, gt_path, args.natsr_threshold)

        # Split into two independent samples (first half & second half)
        for half_name, natsr_val in [("first_half", metrics["natsr_first_half"]),
                                      ("second_half", metrics["natsr_second_half"])]:
            results.append({
                "episode": episode_name,
                "segment": half_name,
                "length": metrics["length"] // 2,
                "mse": metrics["mse"],
                "natsr": natsr_val,
            })
            all_natsr_samples.append(natsr_val)
        all_mse.append(metrics["mse"])

    summary = {
        "num_episodes": len(pred_files),
        "num_samples": len(results),
        "natsr_threshold": args.natsr_threshold,
        "avg_mse": float(np.mean(all_mse)) if all_mse else 0.0,
        "avg_natsr": float(np.mean(all_natsr_samples)) if all_natsr_samples else 0.0,
        "per_sample": results,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nEvaluation complete!")
    print(f"Episodes evaluated: {summary['num_episodes']}")
    print(f"Samples (after split): {summary['num_samples']}")
    print(f"NATSR threshold: {summary['natsr_threshold']}")
    print(f"  Avg Action MSE: {summary['avg_mse']:.6f}")
    print(f"  Avg NATSR: {summary['avg_natsr']:.4f}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
