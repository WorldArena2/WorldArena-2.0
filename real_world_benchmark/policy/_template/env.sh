# shellcheck shell=bash
# Template policy environment defaults (A-side).

export POLICY_MODULE="${POLICY_MODULE:-real_world_benchmark.policy._template.policy}"
export POLICY_ID="${POLICY_ID:-TemplatePolicy}"
export POLICY_ACTION_FORMAT="${POLICY_ACTION_FORMAT:-joint}"
export POLICY_CONTROL_ARM="${POLICY_CONTROL_ARM:-right}"
export POLICY_ACTION_DIM="${POLICY_ACTION_DIM:-14}"
export POLICY_CHUNK_SIZE="${POLICY_CHUNK_SIZE:-20}"
export POLICY_CKPT_DIR="${POLICY_CKPT_DIR:-}"
export POLICY_CONFIG="${POLICY_CONFIG:-}"

export POLICY_HOST="${POLICY_HOST:-0.0.0.0}"
export POLICY_PORT="${POLICY_PORT:-8000}"

# Hub mode (leave empty for WebSocket-only):
# export HUB_POLICY_URL="https://<hub-host>/policy"
# export HUB_TOKEN=""
