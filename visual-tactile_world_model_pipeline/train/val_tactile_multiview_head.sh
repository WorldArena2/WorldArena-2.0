#!/usr/bin/env bash
set -euo pipefail

cd /path/to/your/code

# Conda environment (same as training)
export PATH="/path/to/your/conda/env/bin:$PATH"
export PYTHONPATH="/path/to/your/code:${PYTHONPATH:-}"

# ------------------------------------------------------------------
# Paths — override via env vars if needed
# ------------------------------------------------------------------
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/path/to/your/checkpoint_dir}"
MODEL_CKPT_PATH="${MODEL_CKPT_PATH:-/path/to/your/model_checkpoint.pt}"
DATA_ROOT="${DATA_ROOT:-/path/to/your/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/your/output_dir}"

# Optional: choose GPU
export CUDA_VISIBLE_DEVICES=1

# ------------------------------------------------------------------
# Run validation / inference (single-GPU, no torchrun needed)
# ------------------------------------------------------------------
#chunk 是给数据集的 sample_n_frames 是给模型的， action_chunk 是给动作head的
python \
  train/validate_tactile_multiview_head.py \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --model_ckpt_path "${MODEL_CKPT_PATH}" \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --config "ti2v-5B" \
  --num_samples 3 \
  --sample_h 192 \
  --sample_w 256 \
  --sample_n_frames 9 \
  --chunk 9 \
  --sampling_steps 10 \
  --guide_scale 5.0 \
  --shift 5.0 \
  --visual_views 2 \
  --sample_solver "unipc" \
  --seed 2026 \
  --device_id 0 \
  --action_chunk 9 \
  --enable_action_expert \
  "$@"