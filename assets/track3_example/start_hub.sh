#!/usr/bin/env bash
# 启动本地 WorldArena Hub（wa-hub-v1，双端口：policy / robot）。
#
# Usage:
#   cd track3_example
#   bash start_hub.sh
#
# 默认监听：
#   - policy 端口：18000
#   - robot 端口：19000

set -euo pipefail

# 切换到本脚本所在目录（track3_example/），使用相对路径。
cd "$(dirname "${BASH_SOURCE[0]}")"

# 使用当前目录作为 PYTHONPATH，以便导入本地 worldarena/ 包。
export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}$(pwd)"

HUB_HOST="${HUB_HOST:-127.0.0.1}"
HUB_POLICY_PORT="${HUB_POLICY_PORT:-18000}"
HUB_ROBOT_PORT="${HUB_ROBOT_PORT:-19000}"

python serve_hub.py \
  --host "${HUB_HOST}" \
  --policy-port "${HUB_POLICY_PORT}" \
  --robot-port "${HUB_ROBOT_PORT}"
