import argparse
import os

import torch

from data.tactile_dataset import TactileHDF5Dataset
from wan.configs import WAN_CONFIGS
from wan.tactile_model import WanTactile
from wan.utils.utils import save_video
import numpy as np
import json
from pathlib import Path
#from train_tactile_multiview import _build_joint_action_batch

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Wan tactile multiview inference (sequential)")

    # Model & checkpoint
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Wan2.2 pretrained checkpoint root",
    )
    parser.add_argument(
        "--model_ckpt_path",
        type=str,
        required=True,
        help="Path to your trained .pt checkpoint (contains 'model' state_dict)",
    )

    # Data
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root dir containing UniVTAC / realvideo hdf5 files",
    )
    parser.add_argument("--config", type=str, default="ti2v-5B", choices=list(WAN_CONFIGS.keys()))
    parser.add_argument("--task_names", type=str, nargs="+", default=None)

    # Dataset params (must match training)
    parser.add_argument("--sample_h", type=int, default=192)
    parser.add_argument("--sample_w", type=int, default=256)
    parser.add_argument("--sample_n_frames", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=9)
    parser.add_argument("--use_unified_prompt", action="store_true")
    parser.add_argument(
        "--unified_prompt",
        type=str,
        default="The robotic arm performs a precise insertion task with stable contact.",
    )

    # Inference settings
    parser.add_argument("--num_samples", type=int, default=5, help="Sequential samples to infer")
    parser.add_argument("--visual_views", type=int, default=2, help="How many visual views to generate")
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device_id", type=int, default=0)
    # Action expert
    parser.add_argument("--enable_action_expert", action="store_true", help="Enable action expert inference")
    parser.add_argument("--action_chunk", type=int, default=9, help="Chunk size for action expert (must divide total frames)")
    # Output
    parser.add_argument("--output_dir", type=str, default="./results")
    return parser

def main():
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.device_id}")
    torch.cuda.set_device(device)
    cfg = WAN_CONFIGS[args.config]

    # ------------------------------------------------------------------
    # 1. Dataset — sequential (no shuffle)
    # ------------------------------------------------------------------
    with open(Path(args.data_root) / "metadata_val.json", "r", encoding="utf-8") as f:
        meta_list = json.load(f)

    # 转为绝对路径字符串列表，符合数据集读取要求
    hdf5_paths = [str(Path(args.data_root) / item["hdf5_path"]) for item in meta_list]
    hdf5_paths = [str(path) for path in hdf5_paths]  # 确保是纯字符串列表

    print(f"Loaded {len(hdf5_paths)} hdf5 files")

    dataset = TactileHDF5Dataset(
        hdf5_paths=hdf5_paths,
        data_roots=[args.data_root],
        valid_cam=["head", "wrist"],
        task_names=args.task_names,
        samples_per_episode=1,
        sample_size=(args.sample_h, args.sample_w),
        sample_n_frames=args.sample_n_frames,
        chunk=args.chunk,
        use_unified_prompt=args.use_unified_prompt,
        unified_prompt=args.unified_prompt,
    )

    # ------------------------------------------------------------------
    # 2. Model — WanTactile owns VAE + T5 + DiT
    # ------------------------------------------------------------------
    print("load model...")
    model = WanTactile(
        config=args.config,
        t5_cpu=True,
        init_on_cpu=True,
        checkpoint_dir=args.checkpoint_dir,
        device_id=args.device_id,
        model_ckpt_path=args.model_ckpt_path,
        param_dtype=torch.bfloat16,
    )

    # ------------------------------------------------------------------
    # 3. Sequential inference
    # ------------------------------------------------------------------
    for idx in range(args.num_samples):
        if idx >= len(dataset):
            print(f"Dataset exhausted at index {idx}")
            break

        sample = dataset[idx]
        caption = sample["caption"]
        video = sample["video"]       # [C, V, T, H, W]
        tactile = sample["tactile"]   # [C, V_tac, T, H, W]

        # --- Build first-frame conditions ---------------------------------
        # Visual condition: head camera (view 0) first frame
        # --- Build first-frame conditions ---------------------------------
        # 严格复用训练时的视角拆分：head (view 0) + wrist (view 1)
        first_frame_head  = video[:, 0, 0]  # [C, H, W]
        first_frame_wrist = video[:, 1, 0]  # [C, H, W]

        # Tactile condition: concat left & right at t=0 (same as training preprocessing)
        if tactile.dim() == 5 and tactile.shape[1] == 2:
            left = tactile[:, 0, 0]   # [C, H, W]
            right = tactile[:, 1, 0]  # [C, H, W]
            first_frame_tactile = torch.cat([left, right], dim=-1)  # [C, H, W*2]
        else:
            first_frame_tactile = tactile[:, 0, 0] if tactile.dim() == 5 else tactile[:, 0]

        # --- Inference ----------------------------------------------------
        print(f"\n[{idx+1}/{args.num_samples}] Prompt: {caption}")
        action_chunk = args.action_chunk if args.action_chunk is not None else args.chunk
        pred_visuals, pred_tactile, pred_action = model.infer(
            prompt=caption,
            first_frame_head=first_frame_head,
            first_frame_wrist=first_frame_wrist,
            first_frame_tactile=first_frame_tactile,
            num_frames=args.sample_n_frames,
            size=(args.sample_w, args.sample_h),   # (W, H)
            visual_views=args.visual_views,
            num_inference_steps=args.sampling_steps,
            shift=args.shift,
            seed=args.seed + idx,
            sample_solver=args.sample_solver,
            offload_model=True,
            enable_action_expert=args.enable_action_expert,
            action_chunk=action_chunk,
        )

        # --- Save predictions ---------------------------------------------
        prefix = f"sample_{idx:03d}"
        cam_names = ["head", "wrist"]

        # Predicted visual views
        for v_idx, pred in enumerate(pred_visuals):
            cam_name = cam_names[v_idx] if v_idx < len(cam_names) else f"view{v_idx}"
            save_video(
                tensor=pred.unsqueeze(0),
                save_file=os.path.join(args.output_dir, f"{prefix}_pred_{cam_name}.mp4"),
                fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1),
            )
        save_video(
            tensor=pred_tactile.unsqueeze(0),
            save_file=os.path.join(args.output_dir, f"{prefix}_pred_tactile.mp4"),
            fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1),
        )
        print(f"  Saved pred: {len(pred_visuals)} visual views + tactile")
        if pred_action is not None:
            np.save(os.path.join(args.output_dir, f"{prefix}_pred_action.npy"), pred_action.cpu().numpy())
            print(" + action", end="")
        print()

        # --- Save GT ------------------------------------------------------
        for v_idx in range(min(args.visual_views, video.shape[1])):
            cam_name = cam_names[v_idx] if v_idx < len(cam_names) else f"view{v_idx}"
            save_video(
                tensor=video[:, v_idx].unsqueeze(0),
                save_file=os.path.join(args.output_dir, f"{prefix}_gt_{cam_name}.mp4"),
                fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1),
            )

        if tactile.dim() == 5 and tactile.shape[1] == 2:
            gt_tactile = torch.cat([tactile[:, 0], tactile[:, 1]], dim=-1)
            save_video(
                tensor=gt_tactile.unsqueeze(0),
                save_file=os.path.join(args.output_dir, f"{prefix}_gt_tactile.mp4"),
                fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1),
            )

        # Save GT action/state/force if available
        if args.enable_action_expert:
            for key in ["actions", "state", "virtual_force"]:
                val = sample.get(key)
                if val is not None:
                    np.save(
                        os.path.join(args.output_dir, f"{prefix}_gt_{key}.npy"),
                        val.cpu().numpy() if torch.is_tensor(val) else val,
                    )

    print(f"\n🏁 Done. All results saved to {args.output_dir}")


if __name__ == "__main__":
    main()