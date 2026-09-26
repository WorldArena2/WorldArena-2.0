#!/usr/bin/env bash
set -euo pipefail

cd /path/to/your/code

# Eight-GPU training launcher for stage-2 (action_expert) training.
# Based on train/run_train_tactile_multiview_8gpu.sh but enables the action expert.
export PATH="/path/to/your/conda/env/bin:$PATH"
export PYTHONPATH="/path/to/your/code:${PYTHONPATH:-}"

# Defaults (override via env vars)
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/path/to/your/checkpoint_dir}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/your/output_dir}"
DATA_ROOT="${DATA_ROOT:-/path/to/your/data}"
PRETRAINED_SOURCE="${PRETRAINED_SOURCE:-vidar}"
PRETRAINED_CHECKPOINT_PATH="${PRETRAINED_CHECKPOINT_PATH:-/path/to/your/stage1_checkpoint.pt}"
# If you want stage-2 to continue from a stage-1 trainer checkpoint (saved by Trainer), insert_HDMI  lift_bottle
# set the `STAGE1_CKPT` env var to that checkpoint path. The launcher will then
# use `--pretrained_source vidar --pretrained_checkpoint_path $STAGE1_CKPT` so the
# Trainer loads matching tensors from the checkpoint's `model` field.
# Optional: choose which GPUs to use.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Optional: override the torchrun port.
MASTER_PORT="${MASTER_PORT:-29512}"

# Action expert size knobs (target ~300M by default; override by env vars)
ACTION_NUM_LAYERS="${ACTION_NUM_LAYERS:-28}"
ACTION_NUM_ATTENTION_HEADS="${ACTION_NUM_ATTENTION_HEADS:-16}"
ACTION_ATTENTION_HEAD_DIM="${ACTION_ATTENTION_HEAD_DIM:-64}"

torchrun \
  --nproc_per_node=8 \
  --master_port="${MASTER_PORT}" \
  train/train_tactile_multiview.py \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --pretrained_source "${PRETRAINED_SOURCE}" \
  --pretrained_checkpoint_path "${PRETRAINED_CHECKPOINT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --data_root "${DATA_ROOT}" \
  --batch_size 64 \
  --num_workers 8 \
  --log_every 50 \
  --save_every 50000 \
  --tactile_dim_ratio 0.25 \
  --joint_dim_ratio 0.5 \
  --use_bf16 \
  --use_deepspeed \
  --deepspeed_config train/deepspeed_zero2.json \
  --enable_action_expert \
  --action_num_layers "${ACTION_NUM_LAYERS}" \
  --action_num_attention_heads "${ACTION_NUM_ATTENTION_HEADS}" \
  --action_attention_head_dim "${ACTION_ATTENTION_HEAD_DIM}" \
  --task_names lift_bottle \
  --chunk 9 \
  --action_chunk 45 \
  "$@"
