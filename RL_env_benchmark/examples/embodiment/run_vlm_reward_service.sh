#! /bin/bash

set -euo pipefail

BACKEND=${1:-sglang}
MODEL_PATH=${2:-"/your/path//RLinf/Qwen3.5-9B"}
SERVED_MODEL_NAME=${3:-"Qwen3.5-9B"}
HOST=${4:-"0.0.0.0"}
PORT=${5:-"8000"}

echo "Starting VLM reward service with backend=${BACKEND}"
echo "Model path: ${MODEL_PATH}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Endpoint: http://${HOST}:${PORT}/v1"

if [ "${BACKEND}" = "sglang" ]; then
    # Tip: run this script in a dedicated conda env where sglang is installed.
    exec python -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --host "${HOST}" \
        --port "${PORT}"
elif [ "${BACKEND}" = "vllm" ]; then
    # Tip: run this script in a dedicated conda env where vllm is installed.
    exec vllm serve "${MODEL_PATH}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --host "${HOST}" \
        --port "${PORT}"
else
    echo "Unsupported backend: ${BACKEND}"
    echo "Usage: bash examples/embodiment/run_vlm_reward_service.sh [sglang|vllm] [model_path] [served_model_name] [host] [port]"
    exit 1
fi
