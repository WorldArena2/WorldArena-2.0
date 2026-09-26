import argparse
import json
from pathlib import Path
import numpy as np
import cv2
import torch
from tqdm import tqdm

try:
    from pytorch_msssim import SSIM
    HAS_PYTORCH_MSSSIM = True
except ImportError:
    HAS_PYTORCH_MSSSIM = False
    from scipy.ndimage import gaussian_filter


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        raise ValueError(f"No frames read from {path}")
    return np.stack(frames, axis=0).astype(np.float32)  # [T, H, W, C]


def compute_psnr(img1, img2, max_val=255.0):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def compute_ssim_numpy(img1, img2, max_val=255.0):
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    ssim_channels = []
    for c in range(img1.shape[-1]):
        x = img1[..., c]
        y = img2[..., c]

        mu_x = gaussian_filter(x, sigma=1.5)
        mu_y = gaussian_filter(y, sigma=1.5)

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x_sq = gaussian_filter(x ** 2, sigma=1.5) - mu_x_sq
        sigma_y_sq = gaussian_filter(y ** 2, sigma=1.5) - mu_y_sq
        sigma_xy = gaussian_filter(x * y, sigma=1.5) - mu_xy

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        ssim_channels.append(ssim_map.mean())

    return np.mean(ssim_channels)


def main():
    parser = argparse.ArgumentParser(description="Evaluate PSNR and SSIM for generated videos")
    parser.add_argument("--dataroot", type=str, required=True, help="Root directory containing generate_videos/ and gt_videos/ subdirectories")
    parser.add_argument("--output_json", type=str, default="psnr_ssim_results.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataroot = Path(args.dataroot)
    pred_dir = dataroot / "generate_videos"
    gt_dir = dataroot / "gt_videos"

    if not pred_dir.exists():
        print(f"Error: pred_dir not found: {pred_dir}")
        return
    if not gt_dir.exists():
        print(f"Error: gt_dir not found: {gt_dir}")
        return

    pred_files = sorted(pred_dir.glob("*.mp4"))
    if len(pred_files) == 0:
        print(f"No mp4 files found in {pred_dir}")
        return

    if HAS_PYTORCH_MSSSIM:
        ssim_module = SSIM(data_range=255.0, size_average=True, channel=3).to(args.device)
        print("Using pytorch_msssim for SSIM computation")
    else:
        print("pytorch_msssim not available, using numpy/scipy fallback for SSIM computation")

    results = []
    all_psnr = []
    all_ssim = []

    for pred_path in tqdm(pred_files, desc="Evaluating videos"):
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            print(f"Warning: GT not found for {pred_path.name}, skipping")
            continue

        try:
            pred_video = read_video(pred_path)  # [T, H, W, C], float32, [0, 255]
            gt_video = read_video(gt_path)
        except Exception as e:
            print(f"Error reading {pred_path.name}: {e}, skipping")
            continue

        # Check resolution and resize pred if needed
        if pred_video.shape[1:3] != gt_video.shape[1:3]:
            print(f"Warning: Resolution mismatch for {pred_path.name}, resizing pred from {pred_video.shape[1:3]} to {gt_video.shape[1:3]}")
            pred_video_resized = []
            for t in range(len(pred_video)):
                resized = cv2.resize(pred_video[t].astype(np.uint8), (gt_video.shape[2], gt_video.shape[1]), interpolation=cv2.INTER_LINEAR)
                pred_video_resized.append(resized)
            pred_video = np.stack(pred_video_resized, axis=0).astype(np.float32)

        min_len = min(len(pred_video), len(gt_video))
        if len(pred_video) != len(gt_video):
            print(f"Warning: Length mismatch for {pred_path.name}, using first {min_len} frames")

        pred_video = pred_video[:min_len]
        gt_video = gt_video[:min_len]

        # Compute PSNR per frame
        psnr_list = [compute_psnr(pred_video[t], gt_video[t], max_val=255.0) for t in range(min_len)]
        avg_psnr = np.mean(psnr_list)

        # Compute SSIM
        if HAS_PYTORCH_MSSSIM:
            pred_tensor = torch.from_numpy(pred_video).permute(0, 3, 1, 2).to(args.device)  # [T, C, H, W]
            gt_tensor = torch.from_numpy(gt_video).permute(0, 3, 1, 2).to(args.device)

            # Process in batches to avoid OOM
            batch_size = 64
            ssim_vals = []
            for i in range(0, min_len, batch_size):
                batch_pred = pred_tensor[i:i+batch_size]
                batch_gt = gt_tensor[i:i+batch_size]
                with torch.no_grad():
                    ssim_val = ssim_module(batch_pred, batch_gt)
                ssim_vals.append(ssim_val.item() * batch_pred.shape[0])
            avg_ssim = sum(ssim_vals) / min_len
        else:
            ssim_list = [compute_ssim_numpy(pred_video[t], gt_video[t], max_val=255.0) for t in range(min_len)]
            avg_ssim = np.mean(ssim_list)

        result = {
            "video_name": pred_path.name,
            "num_frames": min_len,
            "psnr": float(avg_psnr),
            "ssim": float(avg_ssim),
        }
        results.append(result)
        all_psnr.append(avg_psnr)
        all_ssim.append(avg_ssim)

    summary = {
        "num_videos": len(results),
        "average_psnr": float(np.mean(all_psnr)) if all_psnr else 0.0,
        "average_ssim": float(np.mean(all_ssim)) if all_ssim else 0.0,
        "per_video": results,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nEvaluation complete!")
    print(f"Number of videos: {summary['num_videos']}")
    print(f"Average PSNR: {summary['average_psnr']:.4f}")
    print(f"Average SSIM: {summary['average_ssim']:.4f}")
    print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
