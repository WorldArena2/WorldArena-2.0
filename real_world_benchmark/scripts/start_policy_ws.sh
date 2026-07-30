#!/usr/bin/env bash
# Start the smoke Policy over WebSocket (wa-policy-v1).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
exec python -m real_world_benchmark.serve_policy_worldarena \
  real_world_benchmark.examples.policy_template.policy \
  --host "${POLICY_HOST:-0.0.0.0}" \
  --port "${POLICY_PORT:-8000}" \
  "$@"
