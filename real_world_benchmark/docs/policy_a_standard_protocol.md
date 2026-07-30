# 模型侧 A 端标准协议

本文档规定模型侧 A 端接入 `real_world_benchmark` 测试框架时需要遵循的标准协议。

当前代码基线如下：

- 策略协议版本：`wa-policy-v1`
- Hub 协议版本：`wa-hub-v1`
- Canonical 数据 Schema：`worldarena.v1`
- Legacy 策略兼容协议：`rwb-policy-v1`

## 1. 适用范围与接入目标

模型侧 A 端接入 live 评测时，必须具备以下能力：

1. 接收评测侧下发的观测输入
2. 返回符合标准的策略动作输出
3. 支持 episode 级 `reset`
4. 支持健康检查 `health`

模型侧支持以下两种接入方式：

1. `wa-policy-v1` WebSocket 直连
2. `wa-hub-v1` HTTP Hub worker

接入方式选型要求如下：

- 在局域网环境或可直连 WebSocket 的网络环境中，应实现 `wa-policy-v1`
- 在跨公网、跨 NAT 或 API 网关不支持 WebSocket 的网络环境中，应实现 `wa-hub-v1` policy worker

策略输入输出支持以下两种工作模式：

1. Canonical 模式：输入 `ObservationPacket`，输出 `ActionPacket`
2. Legacy 兼容模式：输入 `new_obs`，输出 `{"actions": ...}`

规范要求如下：

- 新接入策略应优先兼容 Canonical 模式
- 现有仅实现 `Policy.infer(new_obs)` 的策略，可通过 legacy bridge 接入

## 2. 策略实现接口

策略实现必须提供 Python 类 `Policy`。

最小接口定义如下：

```python
from typing import Any, Dict, Optional

class Policy:
    def __init__(self, config_path: Optional[str] = None):
        ...

    def reset(self, reset_info: Optional[Dict[str, Any]] = None) -> None:
        ...

    def infer(self, new_obs: Dict[str, Any]) -> Dict[str, Any]:
        ...
```

规范要求：

- `Policy.infer()` 为必选接口
- `Policy.reset()` 为推荐接口；如实现，框架会在每个 episode 开始时调用
- 当前仓库默认通过 `Policy.infer(new_obs)` 承接实际推理逻辑，即使外层传输协议为 Canonical 模式，服务端也会先执行 `ObservationPacket -> new_obs` bridge，再调用 `Policy.infer(new_obs)`

## 3. 整体调用时序

### 3.1 WebSocket 直连模式

```text
benchmark_runner
  -> connect ws://policy-host:8000
  <- server metadata

for each episode:
  -> reset
  -> infer
  <- ActionPacket / infer result
```

### 3.2 Hub 模式

```text
policy worker -> POST /workers/register
policy worker -> GET  /workers/poll
policy worker -> POST /workers/result
policy worker -> POST /workers/heartbeat
```

Hub 下发给 policy worker 的任务 `endpoint` 包括：

- `health`
- `reset`
- `infer`

如内部已实现统一的 `dispatch(endpoint, payload)` 分发逻辑，则 WebSocket 与 Hub 两种接入方式通常仅在传输层存在差异。

## 4. WebSocket 协议要求

### 4.1 连接建立

策略服务应监听一个 WebSocket 地址，例如：

```text
ws://<policy-host>:8000
```

连接建立后，服务端第一帧必须主动发送 metadata，编码为 `msgpack`。字段示例如下：

```json
{
  "protocol": "wa-policy-v1",
  "schema_version": "worldarena.v1",
  "policy_source": "policy/YourPolicy/policy.py",
  "supports_reset": true,
  "supports_legacy_new_obs": true
}
```

其中以下字段为必选字段：

- `protocol`
- `schema_version`

其中以下字段为推荐字段：

- `policy_source`
- `supports_reset`
- `supports_legacy_new_obs`

### 4.2 编码

- WebSocket 帧必须采用 `msgpack`
- 结构字段语义必须与 JSON 表达保持一致
- `ObservationPacket`、`ActionPacket` 中的二进制字段必须按 schema 原样传输

代码实现参考：

- [worldarena/policy_remote.py](/workspace/Johnny/real_world_benchmark/worldarena/policy_remote.py)
- [worldarena/serde.py](/workspace/Johnny/real_world_benchmark/worldarena/serde.py)

## 5. Policy 端 RPC

所有请求均为对象类型，且至少包含：

```json
{
  "endpoint": "infer"
}
```

### 5.1 `health`

请求：

```json
{
  "endpoint": "health"
}
```

响应：

```json
{
  "status": "ok",
  "protocol": "wa-policy-v1",
  "schema_version": "worldarena.v1"
}
```

### 5.2 `reset`

请求：

```json
{
  "endpoint": "reset",
  "reset_info": {}
}
```

响应：

```json
{
  "status": "reset successful",
  "protocol": "wa-policy-v1"
}
```

规范要求：

- 如实现 `Policy.reset()`，服务端必须在收到 `reset` 时调用该接口
- `reset_info` 为可扩展字段，策略不得假设其结构固定

### 5.3 `infer`

`infer` 支持两种请求载荷。

#### 5.3.1 Canonical 请求

请求：

```json
{
  "endpoint": "infer",
  "observation_packet": {
    "schema_version": "worldarena.v1",
    "session_id": "run_001",
    "episode_id": "ep_003",
    "task_id": "pick_place",
    "task_instruction": "Pick up the object and place it in the tray",
    "embodiment_id": "robot_site_01",
    "embodiment_type": "single_arm",
    "policy_id": "team_a",
    "policy_interface_version": "wa-policy-v1",
    "adapter_version": "1.0.0",
    "observation_timestamp_ns": 1719123456789000000,
    "step_index": 12,
    "robot_state": {"arms": []},
    "camera_observations": [],
    "tactile_observations": [],
    "safety_state": {
      "emergency_stop_active": false,
      "velocity_limit_active": false,
      "workspace_limit_active": false,
      "action_clipped": false,
      "safety_status": "ok",
      "active_constraints": []
    },
    "network_state": {
      "uplink_latency_ms": 0.0,
      "downlink_latency_ms": 0.0,
      "round_trip_time_ms": 0.0,
      "jitter_ms": 0.0,
      "packet_drop_detected": false
    }
  }
}
```

规范要求：

- 当请求中包含 `observation_packet` 时，服务端必须接受该请求
- 当前仓库默认行为是先将 `ObservationPacket` 转换为 legacy `new_obs`，再调用 `Policy.infer(new_obs)`

#### 5.3.2 Legacy 请求

请求：

```json
{
  "endpoint": "infer",
  "new_obs": {
    "images": {},
    "state": [],
    "prompt": "Pick up the object and place it in the tray"
  }
}
```

规范要求：

- 仅当 `supports_legacy_new_obs=true` 时，服务端才应接受 `new_obs` 请求
- 当未启用 legacy bridge 时，服务端应拒绝仅含 `new_obs` 的请求

## 6. Legacy `new_obs` 输入规范

当前评测框架在策略侧默认调用 `Policy.infer(new_obs)`。`new_obs` 为 Python `dict`，核心字段如下。

### 6.1 顶层字段

常见字段包括：

- `images`
- `state`
- `prompt`
- `task_id`
- `first_frame`
- `joint_qpos`
- `joint_qpos_left`
- `joint_qpos_right`
- `left_arm_joint_state`
- `right_arm_joint_state`
- `left_end_pose`
- `right_end_pose`
- `tactile`
- `tactile_profile`

规范要求：

- 策略实现不得假设所有字段恒定存在
- 策略实现应按字段存在性进行兼容处理

### 6.2 `images`

`new_obs["images"]` 中的标准映射如下：

- `cam_high`：由 canonical `camera_role=global` 映射得到
- `cam_left_wrist`：由 `camera_role=left_wrist` 映射得到
- `cam_right_wrist`：由 `camera_role=right_wrist` 映射得到
- `cam_wrist`：兼容别名；优先等于 `cam_right_wrist`，其次等于 `cam_left_wrist`
- `cam_high_memory`：当 `global` 相机包含历史帧时写入，shape 为 `(T, H, W, 3)`

规范要求：

- 图像数据通常为 `np.ndarray`
- 图像排列必须按 `HWC`
- 图像颜色顺序必须按 RGB
- `cam_high_memory` 为可选字段，仅在请求历史帧且返回值中存在历史帧时出现

### 6.3 `state`

`state` 的典型形式包括：

1. 双臂 eef6d 风格向量
2. 双臂 joint 向量
3. 单臂或信息不足情况下的兜底向量

当前 bridge 行为如下：

- 若左右臂末端位姿均存在且为 `base` 坐标系，则构造双臂 eef6d 状态向量
- 若仅具备双臂关节状态，则构造双臂 joint 拼接向量
- 若均不满足，则返回 `np.zeros((32,), dtype=np.float32)`

规范要求：

- 策略实现不得仅依据 `state.shape` 推断全部语义
- 如需精确读取本体状态，应优先结合 `joint_qpos_*`、`left_end_pose`、`right_end_pose` 等辅助字段

### 6.4 `prompt` 与 `task_id`

- `prompt` 对应 `SessionContext.task_instruction`
- `task_id` 对应 `SessionContext.task_id`

规范要求：

- 策略如依赖自然语言任务描述，应读取 `prompt`
- 策略如依赖任务枚举标识，应读取 `task_id`

### 6.5 `tactile`

当观测中包含触觉信息时，bridge 会写入：

```python
new_obs["tactile"] = {
    "<role>": {
        "rectify": np.ndarray,     # uint8 (700, 400, 3), BGR
        "force": np.ndarray,       # float32 (35, 20, 3), per-cell xyz force in N
        "wrench_6d": np.ndarray,   # float32 (6,), [Fx, Fy, Fz, Tx, Ty, Tz]
        "marker2d": np.ndarray,    # float32 (26, 14, 2), tangential marker displacement
        "mesh3dflow": np.ndarray,  # float32 (35, 20, 3), 3D mesh deformation vectors
        "contact_state": bool,
        "contact_confidence": float,
    }
}
```

规范要求：

- `tactile` 为可选字段
- `tactile_profile` 为可选字段
- 策略如不处理触觉，应在缺失 `tactile` 字段时正常工作
- 图像字段在 B/C/A canonical 传输中可能使用 `encoding=jpeg`，bridge 会在进入 `new_obs` 前解码；策略侧仍读取 `np.ndarray`
- float tactile 字段在 canonical 传输中可能使用 `encoding=raw+zstd`，bridge 会在进入 `new_obs` 前解压；策略侧仍读取 `np.ndarray`
- 如策略读取 `force`，不要将其误解为单个 `(3,)` 力向量；当前约定是整张 `(35,20,3)` 的三维力分布

## 7. `Policy.infer()` 输出规范

`Policy.infer()` 必须返回 `dict`。

### 7.1 必选字段

必选字段如下：

- `actions`

规范要求：

- `actions` 必须可转换为 `np.ndarray`
- `actions` 支持以下形状：
  - `(T, D)`
  - `(1, D)`
  - `(D,)`

### 7.2 推荐字段

推荐字段如下：

- `policy_timing`
- `policy_metadata`

推荐字段示例：

```python
{
    "actions": np.ndarray,
    "policy_timing": {
        "infer_ms": 12.3
    },
    "policy_metadata": {
        "action_format": "joint",
        "control_arm": "right"
    }
}
```

### 7.3 可选附加字段

可选附加字段如下：

- `tactile_force`
- `auxiliary`

当前 `wa-policy-v1` 服务端会在 infer 响应中透传这些字段，但评测主链路不依赖这些字段。

## 8. 动作编码规范

策略输出 `actions` 后，框架会将其转换为 canonical `ActionPacket`。转换行为取决于 `actions` 维度和 `policy_metadata.action_format`。

### 8.1 支持的动作格式

当前支持以下动作格式：

- `joint`
- `eef6d`
- `eef6d_single`
- `end_pose_base`
- `auto`

规范要求：

- 如未显式提供 `policy_metadata.action_format`，默认按 `auto` 处理
- 新策略应明确返回 `policy_metadata.action_format`

### 8.2 `joint`

`joint` 模式下：

- 当 `D >= 14` 且为双臂动作时，按左右臂拆分 joint 目标
- 当 `D in (7, 8)` 时，视为单臂 joint 动作，需结合 `policy_metadata.control_arm`

规范要求：

- 单臂 joint 动作必须同时提供 `policy_metadata.control_arm`
- `control_arm` 取值应为 `left` 或 `right`

### 8.3 `eef6d`

`eef6d` 模式面向双臂 task-space absolute 动作，典型维度为 20：

- `0:3`：左臂位置
- `3:9`：左臂旋转 6D
- `9`：左夹爪
- `10:13`：右臂位置
- `13:19`：右臂旋转 6D
- `19`：右夹爪

转换结果为：

- `ActionPacket.action_mode = task_space_absolute`

### 8.4 `eef6d_single`

`eef6d_single` 模式面向单臂 task-space absolute 动作，典型维度为 10。

规范要求：

- 当 `actions.shape[-1] == 10` 时，建议显式返回：
  - `policy_metadata.action_format = "eef6d_single"`
  - `policy_metadata.control_arm = "left" | "right"`

### 8.5 `end_pose_base`

`end_pose_base` 模式要求每步输出 8 维：

- `pose7`：`[x, y, z, qx, qy, qz, qw]`
- `gripper`：1 维

转换结果为：

- `ActionPacket.action_mode = task_space_absolute`

## 9. Canonical infer 响应规范

在 `wa-policy-v1` infer 响应中，服务端可返回两类合法结果：

1. 直接返回 canonical `ActionPacket` 结构
2. 返回 legacy 结果 `{"actions": ...}`，由框架重建 `ActionPacket`

### 9.1 推荐响应形式

推荐响应形式如下：

```json
{
  "schema_version": "worldarena.v1",
  "session_id": "run_001",
  "episode_id": "ep_003",
  "task_id": "pick_place",
  "observation_timestamp_ns": 1719123456789000000,
  "inference_timestamp_ns": 1719123456799000000,
  "action_apply_timestamp_ns": 1719123456799000000,
  "action_mode": "joint_absolute",
  "action_chunk": [
    {
      "relative_step": 0,
      "arm_actions": [
        {
          "arm_id": "right",
          "joint_position_rad": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0],
          "gripper_target_open_ratio": 0.0
        }
      ]
    }
  ],
  "expected_horizon_steps": 1,
  "policy_latency_ms": 12.3,
  "status": "ok",
  "actions": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0]],
  "policy_metadata": {
    "action_format": "joint",
    "control_arm": "right"
  },
  "policy_timing": {
    "infer_ms": 12.3
  }
}
```

规范要求：

- 如返回 canonical `ActionPacket`，应同时保留 `actions` 原始数组，便于调试与容错修复
- 如返回 `policy_metadata`，其中的 `action_format` 与 `control_arm` 应与 `actions` 实际语义一致

### 9.2 最小兼容响应形式

最小兼容响应形式如下：

```python
{
    "actions": np.zeros((1, 14), dtype=np.float32)
}
```

规范要求：

- 该返回形式仍为合法返回
- 框架会基于 `actions` 自动推断并重建 `ActionPacket`

## 10. Hub 模式协议要求

当策略服务无法被 `benchmark_runner` 直接通过 WebSocket 访问时，应实现 `wa-hub-v1` policy worker 模式。

完整 Hub API 参考：

- [docs/wa-hub-v1-api.md](/workspace/Johnny/real_world_benchmark/docs/wa-hub-v1-api.md)

本节仅列出策略侧必须实现的最小接口集合。

### 10.1 注册

`POST /workers/register`

请求：

```json
{
  "role": "policy",
  "worker_key": "team_a_policy",
  "metadata": {
    "protocol": "wa-policy-v1",
    "schema_version": "worldarena.v1",
    "policy_source": "policy/YourPolicy/policy.py",
    "supports_reset": true,
    "supports_legacy_new_obs": true
  }
}
```

响应：

```json
{
  "worker_id": "policy-xxx",
  "role": "policy",
  "worker_key": "team_a_policy",
  "poll_url": "/workers/poll?worker_id=policy-xxx",
  "heartbeat_interval_s": 15
}
```

### 10.2 心跳

`POST /workers/heartbeat`

```json
{
  "worker_id": "policy-xxx"
}
```

### 10.3 拉任务

`GET /workers/poll?worker_id=policy-xxx&timeout_s=25`

有任务时返回：

```json
{
  "task": {
    "request_id": "uuid",
    "session_id": "run_001",
    "role": "policy",
    "endpoint": "infer",
    "deadline_ms": 120000,
    "payload": {
      "observation_packet": {
        "schema_version": "worldarena.v1"
      }
    }
  }
}
```

### 10.4 回结果

`POST /workers/result`

成功：

```json
{
  "request_id": "uuid",
  "worker_id": "policy-xxx",
  "status": "ok",
  "result": {
    "status": "ok",
    "protocol": "wa-policy-v1"
  },
  "error": null
}
```

失败：

```json
{
  "request_id": "uuid",
  "worker_id": "policy-xxx",
  "status": "error",
  "result": null,
  "error": {
    "code": "WORKER_EXECUTION_ERROR",
    "message": "infer failed",
    "traceback": "..."
  }
}
```

规范要求：

- Hub 的 JSON 传输层不得直接放置二进制
- 所有 `bytes` 字段必须编码为 `{"$b64$": "..."}`

编码逻辑参考：

- [worldarena/hub_json.py](/workspace/Johnny/real_world_benchmark/worldarena/hub_json.py)

## 11. 启动方式

### 11.1 WebSocket 模式

```bash
python -m real_world_benchmark.serve_policy_worldarena /path/to/policy.py \
  --host 0.0.0.0 \
  --port 8000
```

### 11.2 Hub 模式

```bash
python -m real_world_benchmark.serve_policy_worldarena /path/to/policy.py \
  --hub-url https://<gateway>/policy \
  --worker-key <policy_id>
```

### 11.3 禁用 legacy bridge

如策略服务仅接受 canonical `ObservationPacket`，可禁用 legacy bridge：

```bash
python -m real_world_benchmark.serve_policy_worldarena /path/to/policy.py \
  --no-legacy-bridge
```

规范要求：

- 默认启动方式会启用 legacy bridge
- 禁用后，服务端不得再接受仅包含 `new_obs` 的 infer 请求

## 12. 实现要求

如不复用现有 `serve_policy_worldarena.py`，而是自行实现模型侧服务，则至少需要满足以下要求。

### 12.1 必选要求

1. 正确响应 `health`
2. 正确处理 `reset`
3. 正确处理 `infer`
4. 返回合法的 `actions` 或合法的 `ActionPacket`
5. 在 WebSocket 或 Hub 模式下完成 metadata / register / poll / result 的协议流程

### 12.2 推荐要求

1. 实现 `Policy.reset()`
2. 显式返回 `policy_metadata.action_format`
3. 单臂动作场景显式返回 `policy_metadata.control_arm`
4. 返回 `policy_timing.infer_ms`
5. 在 infer 响应中保留原始 `actions`

## 13. 仓库内相关实现位置

- 协议常量：[worldarena/protocol.py](/workspace/Johnny/real_world_benchmark/worldarena/protocol.py)
- 策略 WebSocket 协议：[worldarena/policy_remote.py](/workspace/Johnny/real_world_benchmark/worldarena/policy_remote.py)
- Legacy bridge：[worldarena/bridges/legacy_policy.py](/workspace/Johnny/real_world_benchmark/worldarena/bridges/legacy_policy.py)
- 启动入口：[serve_policy_worldarena.py](/workspace/Johnny/real_world_benchmark/serve_policy_worldarena.py)
- Hub worker：[worldarena/hub_policy_worker.py](/workspace/Johnny/real_world_benchmark/worldarena/hub_policy_worker.py)
- Canonical schema：[worldarena/schema.py](/workspace/Johnny/real_world_benchmark/worldarena/schema.py)
- Hub API 文档：[docs/wa-hub-v1-api.md](/workspace/Johnny/real_world_benchmark/docs/wa-hub-v1-api.md)

## 14. 接入结论

模型侧 A 端完成 `wa-policy-v1` 服务实现，并满足“标准接收观测输入、标准返回动作输出”两项核心要求后，即可接入现有评测框架；如网络环境不支持 WebSocket，则应在保持相同 endpoint 语义的前提下实现 `wa-hub-v1` policy worker。
