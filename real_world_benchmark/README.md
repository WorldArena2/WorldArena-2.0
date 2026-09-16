# WorldArena A-Side Policy Worker Example

本分支是面向模型侧 A 端的最小接入示例，用于实现 `wa-policy-v1` / `wa-hub-v1` policy worker，并将本地 `Policy.infer(new_obs)` 接入 WorldArena 调度。

不包含：
- C 端机器人本体实现、私有机器人通信包
- B 端 `benchmark_runner` / Hub server 调度实现、task suite 评测代码

## 目录结构

```text
.
├── serve_policy_worldarena.py   # A 端启动入口（WebSocket / Hub worker）
├── policy_loader.py             # load_policy：按文件路径或模块路径加载 Policy
├── worldarena/                  # 协议、schema、序列化、legacy bridge、Hub client
│   ├── policy_remote.py         # wa-policy-v1 WebSocket server
│   ├── hub_policy_worker.py     # wa-hub-v1 policy worker loop
│   ├── hub_worker.py            # Hub HTTP long-poll client
│   ├── bridges/legacy_policy.py # ObservationPacket ↔ new_obs / actions ↔ ActionPacket
│   ├── schema.py / protocol.py / serde.py
│   └── hub_json.py / hub_codec.py / hub_protocol.py
├── examples/policy_template/    # 最小可运行 smoke Policy（零权重）
├── policy/_template/            # 可复制的自定义 Policy 模板
├── docs/
│   ├── policy_a_standard_protocol.md
│   ├── policy_a_standard_protocol_vision_only.md
│   ├── new_policy_integration.md
│   └── wa-hub-v1-api.md
├── scripts/start_policy_ws.sh
├── scripts/start_policy_hub.sh
├── requirements-a.txt
└── pyproject.toml
```

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements-a.txt
```

可选：若观测包中含 JPEG 图像字段且需要本地解码，安装 `opencv-python-headless`（已列在 requirements 中）。

## 实现 Policy 类

策略模块需提供名为 `Policy` 的类：

```python
from typing import Any, Dict, Optional
import numpy as np

class Policy:
    def __init__(self, config_path: Optional[str] = None):
        ...

    def reset(self, reset_info: Optional[Dict[str, Any]] = None) -> None:
        ...

    def infer(self, new_obs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "actions": np.zeros((chunk, action_dim), dtype=np.float32),
            "policy_metadata": {
                "policy_id": "MyPolicy",
                "action_format": "joint",  # or eef6d_single / eef6d
                "action_dim": action_dim,
                "chunk_size": chunk,
            },
            "policy_timing": {"infer_ms": 0.0},
        }
```

`new_obs` 由框架从 canonical `ObservationPacket` 经 legacy bridge 转换得到。  
**任务自然语言指令来自 B 端 task suite**，A 端应从以下位置读取（二者等价）：

- `new_obs["prompt"]`
- `ObservationPacket.context.task_instruction`

可直接参考：

- 可运行 smoke 示例：`examples/policy_template/policy.py`
- 自定义模板：`policy/_template/policy.py`

## WebSocket 模式启动（wa-policy-v1）

```bash
python -m real_world_benchmark.serve_policy_worldarena \
  real_world_benchmark.examples.policy_template.policy \
  --host 0.0.0.0 \
  --port 8000
```

或：

```bash
bash scripts/start_policy_ws.sh
```

B 端以 `ws://<A-host>:8000` 连接。

## Hub worker 模式启动（wa-hub-v1）

A 端主动出站长轮询，无需公网入站端口：

```bash
export HUB_POLICY_URL="https://<hub-host>/policy"
export POLICY_ID="MyPolicy_task_v1"   # 即 worker-key
# export HUB_TOKEN="..."              # 如网关需要

python -m real_world_benchmark.serve_policy_worldarena \
  real_world_benchmark.examples.policy_template.policy \
  --hub-url "${HUB_POLICY_URL}" \
  --worker-key "${POLICY_ID}"
```

或：

```bash
export HUB_POLICY_URL="https://<hub-host>/policy"
export POLICY_ID="MyPolicy_task_v1"
bash scripts/start_policy_hub.sh
```

### worker-key 与 B 端配置一致

Hub 用 `worker_key` 把评测任务路由到对应 policy worker。

- A 端：`--worker-key` / 环境变量 `POLICY_ID`
- B 端：task suite / runner 配置中的 policy worker key **必须与上述字符串完全一致**

不一致时，Hub 无法把 infer 任务投递到你的 worker。

## actions 维度与 action_format 约定

| `action_format` | 典型 `actions` shape | 说明 |
|---|---|---|
| `joint` | `(chunk, 14)` | 双臂关节绝对位置，前 7 维左臂、后 7 维右臂 |
| `joint`（单臂） | `(chunk, 7)` 或 `(chunk, 8)` | 需在 metadata 中声明 `control_arm` |
| `eef6d_single` | `(chunk, 10)` | 单臂相机系 eef6d + gripper |
| `eef6d` | `(chunk, ≥20)` | 双臂 eef6d |

在 `policy_metadata` 中返回：

- `action_format`
- `action_dim` / `chunk_size`
- 单臂时的 `control_arm`（`left` / `right`）

框架会把 `{"actions": ndarray}` 转为 canonical `ActionPacket` 回传 B 端。

## Legacy bridge（A 端必需路径）

默认开启 legacy bridge：

1. `ObservationPacket` → `new_obs`（含 `images` / `state` / `prompt` / 可选 `tactile`）
2. `Policy.infer(new_obs)` → `{"actions": ...}`
3. actions + metadata/timing → `ActionPacket`（并透传 `policy_metadata` / `policy_timing`）

仅接受 canonical packet、不走 `new_obs` 时，可加 `--no-legacy-bridge`（需自行处理 packet）。

### 图像颜色（RGB / JPEG）

- `new_obs["images"]` 与触觉 `rectify`：官方 bridge 解出后按 **RGB** 使用
- 线协议 JPEG 由 C 端对 RGB 数组直接 `cv2.imencode` 生成；bridge 用 `cv2.imdecode` 解回后颜色正确
- 若自行用 PIL 等普通解码器解 `frame_bytes` / 触觉 JPEG，会得到 **R/B 对调**（像 BGR）。详见 [协议 §6.2.1](docs/policy_a_standard_protocol.md)

### 可选触觉字段（`new_obs["tactile"]`）

触觉任务时，bridge 会额外写入：

- `tactile[role]`：每路 pad 的字段字典（常见 key：`rectify` / `force` / `wrench_6d`，扩展还有 `marker2d` / `mesh3dflow`）
- `tactile_profile`：`tactile_raw`（保证 `rectify`）或 `tactile_derived`（保证 `force` + `wrench_6d`）等
- `tactile_history[role]`：可选历史栈，时间维在前

线协议 `field_type`（如 `rectify_bgr`、`force_xyz`）与 legacy key 的完整对照、role 命名约定见 [A 端标准协议 §6.5](docs/policy_a_standard_protocol.md)。注意：`rectify_bgr` 是 wire 名；bridge 解出后的 `rectify` 按 RGB 用。

## 文档

- [A 端标准协议](docs/policy_a_standard_protocol.md)（含触觉字段说明）
- [A 端标准协议（纯视觉）](docs/policy_a_standard_protocol_vision_only.md)
- [新 Policy 接入指南](docs/new_policy_integration.md)
- [Hub API](docs/wa-hub-v1-api.md)

## 本地冒烟

```bash
# 加载并 infer 一次
python -c "
from real_world_benchmark.policy_loader import load_policy
import numpy as np
mod = load_policy('real_world_benchmark.examples.policy_template.policy')
p = mod.Policy()
out = p.infer({'images': {'cam_high': np.zeros((64,64,3), dtype=np.uint8)}, 'prompt': 'demo', 'state': np.zeros(14, dtype=np.float32)})
assert out['actions'].ndim == 2
print('ok', out['actions'].shape)
"

python -m real_world_benchmark.serve_policy_worldarena --help
```
