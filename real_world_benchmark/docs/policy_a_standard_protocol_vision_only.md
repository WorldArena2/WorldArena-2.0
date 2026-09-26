# 模型侧 A 端标准协议（纯视觉版）

本文档规定模型侧 A 端以视觉观测与机器人状态为输入接入 `real_world_benchmark` 测试框架时需要遵循的标准协议。

当前代码基线如下：

- 策略协议版本：`wa-policy-v1`
- Hub 协议版本：`wa-hub-v1`
- Canonical 数据 Schema：`worldarena.v1`
- Legacy 策略兼容协议：`rwb-policy-v1`

本版本适用于以下场景：

- 策略模型仅使用视觉与机器人本体状态

## 1. 适用范围与接入目标

模型侧 A 端接入 live 评测时，必须具备以下能力：

1. 接收评测侧下发的视觉观测与机器人状态
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
- `Policy.reset()` 为推荐接口
- 当前仓库默认通过 `Policy.infer(new_obs)` 承接实际推理逻辑，即使外层传输协议为 Canonical 模式，服务端也会先执行 `ObservationPacket -> new_obs` bridge，再调用 `Policy.infer(new_obs)`

## 3. WebSocket 协议要求

### 3.1 连接建立

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

### 3.2 编码

- WebSocket 帧必须采用 `msgpack`
- 结构字段语义必须与 JSON 表达保持一致

## 4. Policy 端 RPC

所有请求均为对象类型，且至少包含：

```json
{
  "endpoint": "infer"
}
```

### 4.1 `health`

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

### 4.2 `reset`

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

### 4.3 `infer`

`infer` 支持两种请求载荷。

#### 4.3.1 Canonical 请求

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

- 策略实现应仅依赖视觉观测与机器人状态相关字段

#### 4.3.2 Legacy 请求

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

- `new_obs` 的有效字段范围以视觉观测与机器人状态相关字段为主

## 5. Legacy `new_obs` 输入规范

当前评测框架在策略侧默认调用 `Policy.infer(new_obs)`。`new_obs` 为 Python `dict`，核心字段如下：

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

规范要求：

- 策略实现不得假设所有字段恒定存在
- 策略实现应按字段存在性进行兼容处理

### 5.1 `images`

`new_obs["images"]` 中的标准映射如下：

- `cam_high`：由 canonical `camera_role=global` 映射得到
- `cam_left_wrist`：由 `camera_role=left_wrist` 映射得到
- `cam_right_wrist`：由 `camera_role=right_wrist` 映射得到
- `cam_wrist`：兼容别名；优先等于 `cam_right_wrist`，其次等于 `cam_left_wrist`
- `cam_high_memory`：当 `global` 相机包含历史帧时写入，shape 为 `(T, H, W, 3)`

规范要求：

- 图像为 `HWC`，官方 legacy bridge 解出后颜色顺序为 **RGB**
- 线协议 JPEG 由 C 端对 RGB 数组直接 `cv2.imencode`；bridge 用 `cv2.imdecode` 解回后按 RGB 使用即可
- 若自行用 PIL 等普通解码器解 `frame_bytes`，会得到 R/B 对调（像 BGR）。完整说明见完整协议文档 §6.2.1（`policy_a_standard_protocol.md`）

### 5.2 `state`

`state` 的典型形式包括：

1. 双臂 eef6d 风格向量
2. 双臂 joint 向量
3. 单臂或信息不足情况下的兜底向量

### 5.3 `prompt` 与 `task_id`

- `prompt` 对应 `SessionContext.task_instruction`
- `task_id` 对应 `SessionContext.task_id`

## 6. `Policy.infer()` 输出规范

`Policy.infer()` 必须返回 `dict`。

### 6.1 必选字段

- `actions`

规范要求：

- `actions` 必须可转换为 `np.ndarray`
- `actions` 支持以下形状：
  - `(T, D)`
  - `(1, D)`
  - `(D,)`

### 6.2 推荐字段

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

## 7. 动作编码规范

当前支持以下动作格式：

- `joint`
- `eef6d`
- `eef6d_single`
- `end_pose_base`
- `auto`

规范要求：

- 如未显式提供 `policy_metadata.action_format`，默认按 `auto` 处理
- 新策略应明确返回 `policy_metadata.action_format`

## 8. Hub 模式协议要求

当策略服务无法被 `benchmark_runner` 直接通过 WebSocket 访问时，应实现 `wa-hub-v1` policy worker 模式。

### 8.1 注册

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

### 8.2 规范要求

- Hub 模式下策略输入语义与 WebSocket 模式保持一致

## 9. 实现要求

### 9.1 必选要求

1. 正确响应 `health`
2. 正确处理 `reset`
3. 正确处理 `infer`
4. 返回合法的 `actions` 或合法的 `ActionPacket`
5. 仅依赖视觉观测与机器人状态相关字段

### 9.2 推荐要求

1. 实现 `Policy.reset()`
2. 显式返回 `policy_metadata.action_format`
3. 单臂动作场景显式返回 `policy_metadata.control_arm`
4. 返回 `policy_timing.infer_ms`

## 10. 仓库内相关实现位置

- 策略 WebSocket 协议：[worldarena/policy_remote.py](/workspace/Johnny/real_world_benchmark/worldarena/policy_remote.py)
- Legacy bridge：[worldarena/bridges/legacy_policy.py](/workspace/Johnny/real_world_benchmark/worldarena/bridges/legacy_policy.py)
- 启动入口：[serve_policy_worldarena.py](/workspace/Johnny/real_world_benchmark/serve_policy_worldarena.py)
- Hub worker：[worldarena/hub_policy_worker.py](/workspace/Johnny/real_world_benchmark/worldarena/hub_policy_worker.py)

## 11. 接入结论

模型侧 A 端完成 `wa-policy-v1` 服务实现，并满足“标准接收视觉观测与机器人状态、标准返回动作输出”两项核心要求后，即可作为纯视觉策略接入现有评测框架。
