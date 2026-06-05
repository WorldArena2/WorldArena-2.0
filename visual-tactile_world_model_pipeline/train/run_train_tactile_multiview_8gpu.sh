#!/usr/bin/env bash
set -euo pipefail

cd /path/to/your/code

# Eight-GPU training launcher for tactile multiview fine-tuning.
# Override env vars before running if needed.
export PATH="/path/to/your/conda/env/bin:$PATH"
export PYTHONPATH="/path/to/your/code:${PYTHONPATH:-}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/path/to/your/checkpoint_dir}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/your/output_dir}"
DATA_ROOT="${DATA_ROOT:-/path/to/your/data}"
PRETRAINED_SOURCE="${PRETRAINED_SOURCE:-wan}"
PRETRAINED_CHECKPOINT_PATH="${PRETRAINED_CHECKPOINT_PATH:-/path/to/your/pretrained_checkpoint.pt}"

# Optional: choose which GPUs to use.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Optional: override the torchrun port.
MASTER_PORT="${MASTER_PORT:-29512}"

torchrun \
  --nproc_per_node=8 \
  --master_port="${MASTER_PORT}" \
  train/train_tactile_multiview.py \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --pretrained_source "${PRETRAINED_SOURCE}" \
  --pretrained_checkpoint_path "${PRETRAINED_CHECKPOINT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --data_root "${DATA_ROOT}" \
  --batch_size 16 \
  --num_workers 4 \
  --log_every 10 \
  --save_every 50000 \
  --tactile_dim_ratio 0.25 \
  --joint_dim_ratio 0.5 \
  --use_bf16 \
  --use_deepspeed \
  --deepspeed_config train/deepspeed_zero2.json \
  --chunk 9 \
  --action_chunk 45 \
  "$@"
