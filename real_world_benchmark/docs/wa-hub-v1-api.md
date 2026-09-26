# wa-hub-v1 HTTP Hub 最小 API 草案

> **协议版本**：`wa-hub-v1` · **Schema** `worldarena.v1`（不变）  
> **适用场景**：中央调度（Hub）部署在火山 MLP API 网关等 **仅支持 HTTP/HTTPS、不支持 WebSocket** 的公网入口；策略 Worker（A）与本体 Worker（C）**主动出站**连接 Hub（方案 A）。  
> **Endpoint 字段**：Worker 任务与 Orchestrator RPC 中的 `endpoint` 值与 `worldarena/protocol.py` 中 `POLICY_ENDPOINT_*` / `ROBOT_ENDPOINT_*` **完全一致**。

---

## 1. 架构

```text
机器 A (policy worker)  ──HTTPS──►  Hub policy  :8000  (公网 https://<gateway>/policy/...)
机器 C (robot worker)   ──HTTPS──►  Hub robot   :9000  (公网 https://<gateway>/robot/...)
benchmark_runner (B)      ──HTTP──►  127.0.0.1:8000/orchestrator/policy/*  (localhost only)
                          ──HTTP──►  127.0.0.1:9000/orchestrator/robot/*   (localhost only)
```

- **双端口**：策略 Worker 与 Orchestrator policy RPC 共用 **8000**；本体 Worker 与 Orchestrator robot RPC 共用 **9000**。
- **API 网关**：同域名路径分流 `/policy` → 8000、`/robot` → 9000；网关**剥前缀**（容器内路由为 `/workers/...`，无前缀）。
- **Schema 层**：仍使用 `ObservationPacket` / `ActionPacket` / `SessionContext` / `EpisodeEvent`（`worldarena.v1`）。
- **传输层**：WebSocket `wa-policy-v1` / `wa-robot-v1` 在跨公网场景下由 **HTTP Hub 长轮询** 替代；同 LAN 评测可继续使用 WebSocket 直连。
- **任务字段 `endpoint`**：与现有 WS 协议同名，便于 Worker 复用同一套 dispatch 逻辑。
- **编码方式**：重 payload 推荐 `application/msgpack`，二进制字段（如 JPEG `frame_bytes` 或 `raw+zstd` tactile `data_bytes`）按 bytes 原样传输，避免 JSON/base64 膨胀；`application/json` 仍保留兼容，二进制字段经 `$b64$` 包装。

---

## 2. 公共约定

### 2.1 Base URL

**公网（A / C Worker 使用）**

```text
https://<apigateway-host>/policy    → 容器 :8000
https://<apigateway-host>/robot     → 容器 :9000
```

示例（火山 MLP）：

```text
https://sd927rs5hj10d8m3a03lg.apigateway-cn-beijing.volceapi.com/policy
https://sd927rs5hj10d8m3a03lg.apigateway-cn-beijing.volceapi.com/robot
```

**本机 Orchestrator（B 上 benchmark_runner，仅 localhost）**

```text
http://127.0.0.1:8000/orchestrator/policy/{endpoint}
http://127.0.0.1:9000/orchestrator/robot/{endpoint}
http://127.0.0.1:8000/sessions/bind
```

下文 §3–§5 中的路径为**容器内路由**（网关剥 `/policy`、`/robot` 前缀后）。若网关不剥前缀，启动 Hub 时设 `--policy-url-prefix /policy --robot-url-prefix /robot`。

### 2.2 请求头

| Header | 必填 | 说明 |
|--------|------|------|
| `Content-Type` | POST | 推荐 `application/msgpack`；兼容 `application/json` |
| `Accept` | 可选 | 设置为 `application/msgpack` 时，Hub 对重 payload 响应返回 msgpack |
| `Authorization` | 若 Hub 开启 Token | `Bearer <token>` |
| `X-Hub-Protocol` | 推荐 | `wa-hub-v1` |
| `X-Request-Id` | 推荐 | UUID；Orchestrator 与 Worker 幂等、去重 |

当前实现中，`HubOrchestratorClient` 的 `orchestrator/*` RPC、`HubWorkerClient` 的 `workers/poll` 响应和 `workers/result` 提交默认使用 msgpack。`workers/register`、`workers/heartbeat`、`sessions/bind`、`health` 仍使用 JSON，因为 payload 很小且便于调试。

### 2.3 任务信封（Hub → Worker，`poll` 响应）

Hub 向 Worker 下发任务时，`endpoint` 使用现有常量：

| role | `endpoint` 取值（`protocol.py`） |
|------|----------------------------------|
| policy | `health` · `reset` · `infer` |
| robot | `health` · `reset` · `get_observation` · `apply_action` · `report_event` |

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "session_id": "sess_ep_001",
  "role": "policy",
  "endpoint": "infer",
  "deadline_ms": 120000,
  "payload": {}
}
```

### 2.4 结果信封（Worker → Hub，`workers/result`）

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "worker_id": "policy-vital-sh-01",
  "status": "ok",
  "result": {},
  "error": null
}
```

失败时：

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "worker_id": "robot-agilex-01",
  "status": "error",
  "result": null,
  "error": {
    "code": "WORKER_EXECUTION_ERROR",
    "message": "apply_action failed: safety stop engaged",
    "traceback": "..."
  }
}
```

### 2.5 幂等

- `apply_action`：`payload.idempotency_key`（或 `request_id`）在本体 Worker 侧去重，防止 HTTP 重试重复执行。
- `infer`：相同 `request_id` 重放应返回相同 `ActionPacket`（或明确 `409`）。

---

## 3. Hub 管理接口

### 3.1 `GET /health`

**响应 200：**

```json
{
  "status": "ok",
  "protocol": "wa-hub-v1",
  "schema_version": "worldarena.v1",
  "transport": "http",
  "registered_workers": {
    "policy": 1,
    "robot": 1
  }
}
```

---

## 4. Worker 接口（A / C 主动调用）

### 4.1 `POST /workers/register`

Worker 启动后注册，获得 `worker_id`。

**请求：**

```json
{
  "role": "policy",
  "worker_key": "vital_act",
  "metadata": {
    "protocol": "wa-policy-v1",
    "schema_version": "worldarena.v1",
    "policy_source": "real_world_benchmark/policy/ViTAL/policy.py",
    "supports_reset": true,
    "supports_legacy_new_obs": true
  }
}
```

本体 Worker 示例：

```json
{
  "role": "robot",
  "worker_key": "agilex_lab_01",
  "metadata": {
    "protocol": "wa-robot-v1",
    "schema_version": "worldarena.v1",
    "adapter_id": "robot_private_adapter",
    "adapter_version": "adapter.robot_private.1.2.0",
    "embodiment_id": "agilex_lab_01",
    "embodiment_type": "single_arm",
    "supported_tactile_roles": ["left_gripper", "right_gripper"],
    "supported_tactile_profiles": ["tactile_raw"],
    "default_tactile_profile": "tactile_raw"
  }
}
```

**响应 200：**

```json
{
  "worker_id": "policy-vital-sh-01-a1b2c3",
  "role": "policy",
  "worker_key": "vital_act",
  "poll_url": "/workers/poll?worker_id=policy-vital-sh-01-a1b2c3",
  "heartbeat_interval_s": 15
}
```

### 4.2 `POST /workers/heartbeat`

**请求：**

```json
{
  "worker_id": "policy-vital-sh-01-a1b2c3"
}
```

**响应 200：** `{"status": "ok"}`

### 4.3 `GET /workers/poll`

长轮询等待任务。`timeout_s` 建议 **25**（低于 API 网关读超时，默认 60s）。

**Query：** `worker_id=policy-vital-sh-01-a1b2c3&timeout_s=25`

**响应 200（有任务）— `endpoint: infer`（对齐 `POLICY_ENDPOINT_INFER`）：**

```json
{
  "task": {
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "session_id": "sess_ep_001",
    "role": "policy",
    "endpoint": "infer",
    "deadline_ms": 120000,
    "payload": {
      "observation_packet": {
        "context": {
          "schema_version": "worldarena.v1",
          "session_id": "sess_ep_001",
          "episode_id": "ep_001",
          "task_id": "peg_insertion_vital",
          "task_instruction": "Insert the peg into the hole",
          "embodiment_id": "agilex_lab_01",
          "embodiment_type": "single_arm",
          "policy_id": "vital_act",
          "policy_interface_version": "wa-policy-v1",
          "adapter_version": "adapter.robot_private.1.2.0"
        },
        "observation_timestamp_ns": 1719123456789000000,
        "step_index": 12,
        "robot_state": { "arms": [] },
        "camera_observations": [],
        "tactile_observations": []
      }
    }
  }
}
```

**响应 200（有任务）— `endpoint: reset`（对齐 `POLICY_ENDPOINT_RESET`）：**

```json
{
  "task": {
    "request_id": "660e8400-e29b-41d4-a716-446655440001",
    "session_id": "sess_ep_002",
    "role": "policy",
    "endpoint": "reset",
    "deadline_ms": 30000,
    "payload": {
      "reset_info": {},
      "context": {
        "schema_version": "worldarena.v1",
        "session_id": "sess_ep_002",
        "episode_id": "ep_002",
        "task_id": "peg_insertion_vital",
        "task_instruction": "Insert the peg into the hole",
        "embodiment_id": "agilex_lab_01",
        "policy_id": "vital_act"
      }
    }
  }
}
```

**响应 200（有任务）— `endpoint: get_observation`（对齐 `ROBOT_ENDPOINT_GET_OBSERVATION`）：**

```json
{
  "task": {
    "request_id": "770e8400-e29b-41d4-a716-446655440002",
    "session_id": "sess_ep_001",
    "role": "robot",
    "endpoint": "get_observation",
    "deadline_ms": 60000,
    "payload": {
      "context": {
        "schema_version": "worldarena.v1",
        "session_id": "sess_ep_001",
        "episode_id": "ep_001",
        "task_id": "peg_insertion_vital",
        "embodiment_id": "agilex_lab_01",
        "policy_id": "vital_act"
      },
      "step_index": 12,
      "observation_history": {
        "camera_roles": { "global": 5 }
      }
    }
  }
}
```

**响应 200（有任务）— `endpoint: apply_action`（对齐 `ROBOT_ENDPOINT_APPLY_ACTION`）：**

```json
{
  "task": {
    "request_id": "880e8400-e29b-41d4-a716-446655440003",
    "session_id": "sess_ep_001",
    "role": "robot",
    "endpoint": "apply_action",
    "deadline_ms": 30000,
    "payload": {
      "action_packet": {
        "context": { "schema_version": "worldarena.v1", "session_id": "sess_ep_001", "step_index": 12 },
        "action_mode": "joint_absolute",
        "arm_actions": [],
        "step_index": 12
      },
      "idempotency_key": "sess_ep_001:12:apply_action"
    }
  }
}
```

**响应 200（有任务）— `endpoint: report_event`（对齐 `ROBOT_ENDPOINT_REPORT_EVENT`）：**

```json
{
  "task": {
    "request_id": "990e8400-e29b-41d4-a716-446655440004",
    "session_id": "sess_ep_001",
    "role": "robot",
    "endpoint": "report_event",
    "deadline_ms": 10000,
    "payload": {
      "event": {
        "event_type": "success",
        "session_id": "sess_ep_001",
        "episode_id": "ep_001",
        "step_index": 45,
        "timestamp_ns": 1719123499999000000,
        "message": "human marked success"
      }
    }
  }
}
```

**响应 204：** 轮询超时，无任务（Worker 应立即再次 `poll`）。

### 4.4 `POST /hub/v1/workers/result`

**`infer` 成功结果（`ActionPacket`）：**

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "worker_id": "policy-vital-sh-01-a1b2c3",
  "status": "ok",
  "result": {
    "context": {
      "schema_version": "worldarena.v1",
      "session_id": "sess_ep_001",
      "episode_id": "ep_001",
      "policy_id": "vital_act"
    },
    "action_mode": "joint_absolute",
    "arm_actions": [
      {
        "arm_id": "left",
        "joint_positions": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "gripper_position": 0.0
      }
    ],
    "step_index": 12
  },
  "error": null
}
```

**`reset` / `health` / `report_event` ack：**

```json
{
  "request_id": "660e8400-e29b-41d4-a716-446655440001",
  "worker_id": "policy-vital-sh-01-a1b2c3",
  "status": "ok",
  "result": {
    "status": "reset successful",
    "protocol": "wa-policy-v1"
  },
  "error": null
}
```

**`get_observation` 成功结果（`ObservationPacket`）：** 结构与 WS `wa-robot-v1` 响应相同，省略大字段 `frame_bytes` 示例。

**`apply_action` 成功结果：**

```json
{
  "request_id": "880e8400-e29b-41d4-a716-446655440003",
  "worker_id": "robot-agilex-01-d4e5f6",
  "status": "ok",
  "result": {
    "status": "applied",
    "step_index": 12
  },
  "error": null
}
```

---

## 5. Orchestrator 接口（benchmark_runner @ B）

Hub 与 `benchmark_runner` 同机部署时，Orchestrator 通过同步 RPC 调用已注册 Worker（Hub 内部经 poll/result 队列转发）。

### 5.1 `POST /sessions/bind`

绑定一次 live eval 使用的 policy / robot worker。

**请求：**

```json
{
  "session_id": "sess_ep_001",
  "policy_worker_key": "vital_act",
  "robot_worker_key": "agilex_lab_01"
}
```

**响应 200：**

```json
{
  "session_id": "sess_ep_001",
  "policy_worker_id": "policy-vital-sh-01-a1b2c3",
  "robot_worker_id": "robot-agilex-01-d4e5f6",
  "status": "bound"
}
```

### 5.2 `POST /orchestrator/policy/{endpoint}`

`{endpoint}` ∈ `health` | `reset` | `infer`（即 `POLICY_ENDPOINT_*`）。

**示例：`POST .../orchestrator/policy/infer`**

```json
{
  "session_id": "sess_ep_001",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "observation_packet": { "...": "同 WS wa-policy-v1 infer 请求体" }
}
```

**响应 200：** 直接返回 `ActionPacket` JSON（与 WS 响应同构）。

### 5.3 `POST /orchestrator/robot/{endpoint}`

`{endpoint}` ∈ `health` | `reset` | `get_observation` | `apply_action` | `report_event`（即 `ROBOT_ENDPOINT_*`）。

**示例：`POST .../orchestrator/robot/get_observation`**

```json
{
  "session_id": "sess_ep_001",
  "request_id": "770e8400-e29b-41d4-a716-446655440002",
  "context": { "...": "SessionContext" },
  "step_index": 12,
  "observation_history": { "camera_roles": { "global": 5 } }
}
```

**响应 200：** 直接返回 `ObservationPacket` JSON。

---

## 6. 单步 Live Eval 时序

```text
1. A: register(role=policy)     C: register(role=robot)
2. A/C: 循环 poll + heartbeat
3. runner: sessions/bind
4. runner: orchestrator/robot/get_observation  → Hub 排队 → C poll 收到 → C result
5. runner: orchestrator/policy/infer           → Hub 排队 → A poll 收到 → A result
6. runner: orchestrator/robot/apply_action     → Hub 排队 → C poll 收到 → C result
7. 重复 4–6 直至 episode 结束
```

---

## 7. 与 WS 协议对照

| WS（wa-policy-v1 / wa-robot-v1） | HTTP Hub（wa-hub-v1） |
|----------------------------------|------------------------|
| 持久 WebSocket 连接 | Worker `poll` 长轮询 + `result` POST |
| 首帧 metadata | `workers/register` 响应 + `metadata` 字段 |
| `{ "endpoint": "infer", ... }` | `task.endpoint == "infer"` + `task.payload` |
| 同构 JSON body | 不变（`worldarena.v1`） |

---

## 8. 部署提示（火山 MLP API 网关）

- 容器双端口：**`0.0.0.0:8000`**（policy）、**`0.0.0.0:9000`**（robot）；启动：`python -m real_world_benchmark.serve_hub` 或 `scripts/start_hub_volcano.sh`。
- 网关路径 **`/policy` → 8000**、**`/robot` → 9000**；默认**剥前缀**（容器内 `/health`，非 `/policy/health`）。
- Orchestrator 与 `sessions/bind` **仅 localhost**；不暴露第三个公网端口。
- 网关 **不支持 WebSocket** 时仅使用本草案（已实测 `server rejected WebSocket connection`）。
- 调大网关 **读超时**（建议 ≥ 300s）以覆盖大模型 `infer`。
- 确认 **请求体上限**（默认约 60MB）满足多相机 + 触觉观测。
- Worker `poll` 的 `timeout_s` 保持在网关读超时以下（推荐 25s）。

**B — Hub**

```bash
python -m real_world_benchmark.serve_hub --policy-port 8000 --robot-port 9000 --host 0.0.0.0
```

**A — Policy Worker**

```bash
python -m real_world_benchmark.serve_policy_worldarena /path/to/policy.py \
  --hub-url https://<gateway>/policy --worker-key vital_act
```

**C — Robot Worker**

```bash
python -m real_world_benchmark.serve_robot \
  --hub-url https://<gateway>/robot --worker-key agilex_lab_01 \
  --embodiment-id agilex_lab_01
```

**B — benchmark_runner**

```bash
python -m real_world_benchmark.benchmark_runner --hub-mode \
  --hub-policy-key vital_act --hub-robot-key agilex_lab_01 \
  --policy-protocol wa-policy-v1 --mode live --send-action \
  --task-suite real_world_benchmark/example_task_suite.json \
  --policy-id vital_act --site-id agilex_lab_01
```

---

## 9. 代码映射

| 模块 | 文件 |
|------|------|
| Hub 协议常量 | `worldarena/hub_protocol.py` |
| WS 协议常量 | `worldarena/protocol.py` |
| Hub 核心 / 服务 | `worldarena/hub_core.py` · `worldarena/hub_server.py` · `serve_hub.py` |
| JSON 编解码 | `worldarena/hub_json.py` |
| Worker 客户端 | `worldarena/hub_worker.py` · `hub_policy_worker.py` · `hub_robot_worker.py` |
| Orchestrator 客户端 | `worldarena/hub_orchestrator.py` |
| 调度集成 | `benchmark_runner.py` · `worldarena/orchestration.py` |
| 测试 | `test_worldarena_hub.py` |
| 火山启动脚本 | `scripts/start_hub_volcano.sh` · `scripts/vital_fake_machine_b_hub.sh` |
