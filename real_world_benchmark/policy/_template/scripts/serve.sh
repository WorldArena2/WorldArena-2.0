#!/usr/bin/env bash
# Serve template / custom policy over wa-policy-v1 or wa-hub-v1.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "${POLICY_DIR}/env.sh"

HOST="${POLICY_HOST}"
PORT="${POLICY_PORT}"
WORKER_KEY="${POLICY_ID}"

cd "${REPO_ROOT}"

HUB_ARGS=()
if [[ -n "${HUB_POLICY_URL:-}" ]]; then
  echo "[serve] Hub worker → ${HUB_POLICY_URL}  worker_key=${WORKER_KEY}"
  HUB_ARGS=(--hub-url "${HUB_POLICY_URL}" --worker-key "${WORKER_KEY}")
  if [[ -n "${HUB_TOKEN:-}" ]]; then
    HUB_ARGS+=(--hub-token "${HUB_TOKEN}")
  fi
else
  echo "[serve] WebSocket ws://${HOST}:${PORT} (wa-policy-v1)"
  HUB_ARGS=(--host "${HOST}" --port "${PORT}")
fi

exec python -m real_world_benchmark.serve_policy_worldarena \
  "${POLICY_MODULE}" \
  "${HUB_ARGS[@]}"
