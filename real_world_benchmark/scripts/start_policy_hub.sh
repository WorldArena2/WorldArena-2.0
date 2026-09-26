#!/usr/bin/env bash
# Start the smoke Policy as a Hub long-poll worker (wa-hub-v1).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

HUB_URL="${HUB_POLICY_URL:?Set HUB_POLICY_URL, e.g. https://<hub-host>/policy}"
WORKER_KEY="${POLICY_ID:-SmokePolicy}"

ARGS=(--hub-url "${HUB_URL}" --worker-key "${WORKER_KEY}")
if [[ -n "${HUB_TOKEN:-}" ]]; then
  ARGS+=(--hub-token "${HUB_TOKEN}")
fi

exec python -m real_world_benchmark.serve_policy_worldarena \
  real_world_benchmark.examples.policy_template.policy \
  "${ARGS[@]}" \
  "$@"
