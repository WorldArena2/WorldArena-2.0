# Tactile Multiview Video Generation

This repository provides training and validation code for tactile multiview video generation based on a diffusion transformer backbone with multiview joint attention and tactile intra-attention layers.

## 1. Environment Setup

### 1.1 Create Python Environment

```bash
conda create -n tactile python=3.12.12 -c conda-forge
conda activate tactile
```

### 1.2 Install PyTorch

```bash
pip install torch==2.9.1 torchvision==0.24.1
```


### 1.3 Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Dataset Configuration

We use the **UniVTAC** dataset for training and validation. The dataset contains multiview RGB episodes, tactile marker images, actions, joint states, and virtual forces.

**Dataset link:** [https://modelscope.cn/datasets/byml2024/UniVTAC](https://modelscope.cn/datasets/byml2024/UniVTAC)

### Download the dataset

```bash
pip install modelscope
modelscope download --dataset byml2024/UniVTAC --local_dir ./data/UniVTAC
```

After downloading, set the dataset path in the launch scripts (replace `/path/to/your/data` in the `.sh` files with your actual dataset path).

## 3. Training and Validation

We provide example training and validation scripts based on **Wan 2.2 TI2V-5B**. You can download the pretrained weights from [HuggingFace](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) or [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B).

Before running any script, update the placeholder paths in the corresponding `.sh` files:
- `/path/to/your/code` — path to this repository
- `/path/to/your/conda/env/bin` — path to your conda environment `bin` directory
- `/path/to/your/checkpoint_dir` — directory containing the base model checkpoints
- `/path/to/your/output_dir` — directory where checkpoints and logs will be saved
- `/path/to/your/data` — path to the downloaded UniVTAC dataset
- `/path/to/your/pretrained_checkpoint.pt` — path to your pretrained checkpoint (if applicable)

### 3.1 Training

**Eight-GPU training (from base pretrained weights):**
```bash
bash train/run_train_tactile_multiview_8gpu.sh
```

**Stage-2 training with action expert (eight GPUs):**
```bash
bash train/run_train_tactile_multiview_stage2_8gpu.sh
```

Key configurable flags inside the scripts:
- `--tactile_dim_ratio`: shrink tactile self-attention inner dimension (default 0.25)
- `--joint_dim_ratio`: shrink multiview joint-attention inner dimension (default 0.5)
- `--enable_action_expert`: enable the action-expert branch and freeze the diffusion backbone
- `--use_deepspeed --deepspeed_config train/deepspeed_zero2.json`: enable DeepSpeed

### 3.2 Validation

**Standard single-GPU validation:**
```bash
bash train/val_tactile_multiview.sh
```

**Action-expert head validation (single GPU):**
```bash
bash train/val_tactile_multiview_head.sh
```

## 4. Evaluation Metrics

We provide evaluation scripts under the `metric/` directory for both visual quality and action prediction:

- **Visual validation (PSNR / SSIM):**
  ```bash
  python metric/val_psnr_ssim.py --dataroot <path_to_results> --output_json results.json
  ```
  Computes PSNR and SSIM between generated videos and ground-truth videos.

- **Action offline reference validation (MSE / NATSR):**
  ```bash
  python metric/eval_action_offline.py --dataroot <path_to_results> --output_json action_metrics.json --natsr_threshold 0.1
  ```
  Evaluates end-effector action prediction accuracy using Mean Squared Error (MSE) and Normalized Action Success Rate (NATSR).

Please refer to the individual scripts for additional command-line options.
