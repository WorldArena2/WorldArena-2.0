#!/bin/bash
set -euo pipefail
# Usage: run_evaluation_JEPA.sh <GEN_VIDEO_DIR> 

GEN_VIDEO_DIR=${1:-}

ROOT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUTPUT_ROOT="$ROOT_DIR/output_JEDi"

cd ./video_quality/JEDi
source $(conda info --base)/etc/profile.d/conda.sh
conda activate WorldArena_JEPA
export PATH="your absolute path/WorldArena_JEPA/bin:$PATH"

python batch.py \
	--real_dir REAL_DIR_TO_GT \
	--gen_dir "$GEN_VIDEO_DIR" \
    --output_root "$OUTPUT_ROOT" 

