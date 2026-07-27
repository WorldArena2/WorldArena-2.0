#!/usr/bin/env bash
# 启动参赛策略 Worker，连接到 WorldArena Hub（机器 B）。
#
# Usage:
#   cd track3_example
#   bash start_policy_worker.sh
#
# 运行前请将下面的占位符替换为正式参赛前提供的真实值：
#   - <PENDING_HUB_GATEWAY_URL>
#   - <PENDING_POLICY_ID>

set -euo pipefail

# 切换到本脚本所在目录（track3_example/），使用相对路径。
cd "$(dirname "${BASH_SOURCE[0]}")"

# 使用当前目录作为 PYTHONPATH，以便导入本地 worldarena/ 包。
export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}$(pwd)"

# 可通过环境变量覆盖默认值；正式参赛前替换为真实值。
HUB_GATEWAY_URL="${HUB_GATEWAY_URL:-<PENDING_HUB_GATEWAY_URL>}"
POLICY_ID="${POLICY_ID:-<PENDING_POLICY_ID>}"

python serve_policy_worldarena.py \
  ./dummy_policy.py \
  --hub-url "${HUB_GATEWAY_URL}/policy" \
  --worker-key "${POLICY_ID}"
