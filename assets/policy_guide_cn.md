# WorldArena 外部策略接入指南

> 本文档面向需要在自有机器（机器 A）上部署策略 Worker，并接入我方中央调度（机器 B）与真机本体（机器 C）的参赛团队。

Track 3 支持**两套平台**。动作维度与观测字段取决于分配到的机器人：

| 平台 | 动作格式 | `action_dim` | 说明 |
|---|---|---|---|
| **AgileX** 双臂 | `joint_absolute`（qpos） | **14** | 左臂 7 + 右臂 7 |
| **Franka** 单臂 | `end_pose_base` | **8** | `[x, y, z, qw, qx, qy, qz, gripper]` |

---

## 一、架构与数据流

```text
机器 A（参赛）              机器 B（我方）              机器 C（我方）
策略 Worker  ──HTTPS───→  中央 Hub  ←──HTTPS───  机器人本体 Worker
   ↑                          ↓                          ↑
Policy.infer()               调度                       真机 
```

- **机器 A（策略侧）**：由参赛团队提供，只需要能访问公网，**不需要固定公网 IP**。
- **机器 B（调度侧）**：提供模型中转与调度。
- **机器 C（真机侧）**：真机动作执行与观测反馈。

机器 A 通过**主动出站** HTTPS 长轮询连接到我方 Hub，接收观测、返回动作。
机器 A 可以进行长时间挂载策略。

---

## 二、策略接口定义

外部策略只需实现一个 Python 类，文件路径和类名任意，但需满足以下接口：

```python
from typing import Any, Dict
import numpy as np

class Policy:
    def __init__(self, config_path: str | None = None):
        """初始化模型、加载配置等。"""
        ...

    def reset(self, reset_info: Dict[str, Any] | None = None) -> None:
        """每个 episode 开始时调用一次。"""
        ...

    def infer(self, new_obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        每步推理接口。

        Args:
            new_obs: 我方转换后的观测字典，字段定义见第三章。

        Returns:
            dict，必须包含:
                - "actions": np.ndarray, shape (chunk, action_dim)
            可选包含:
                - "policy_metadata": dict
                - "policy_timing": dict
                - "tactile_force": np.ndarray
        """
        ...
```

### 2.1 返回字段说明

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `actions` | `np.ndarray` | 是 | 动作块，`shape=(chunk, action_dim)`。AgileX 上 `action_dim=14`，Franka 上 `action_dim=8`。具体 `chunk` 由策略自行决定（常见 20、25） |

### 2.2 AgileX 动作语义（`joint_absolute` / qpos）

```text
actions[t] = [左臂 j1..j6, 左夹爪, 右臂 j1..j6, 右夹爪]  # 14 维
```

每一行是一个绝对关节位置目标。我方按顺序执行返回的关节目标。数据集中用于控制的演示标签位于 `observations/qpos`。

### 2.3 Franka 动作语义（`end_pose_base`）

```text
actions[t] = [x, y, z, qw, qx, qy, qz, gripper]  # 8 维
```

- 位置 `(x, y, z)` 位于**机器人基坐标系**（单位：米）
- 姿态 `(qw, qx, qy, qz)` 为 **wxyz** 四元数
- `gripper` 为夹爪开合指令

数据集中用于控制的演示标签位于 `observations/end_pose[:, 0:8]`。

---

## 三、我方发送给策略的观测 `new_obs`

### 3.1 AgileX 顶层字段

```python
{
    "images": { ... },                   # 视觉图像
    "state": np.ndarray,                 # 14D 双臂关节（前 7 维左臂，后 7 维右臂）
    "joint_qpos": np.ndarray,            # 14D 双臂关节（前 7 维左臂，后 7 维右臂）
    "right_arm_joint_state": np.ndarray, # 7D 右臂关节
    "left_arm_joint_state": np.ndarray,  # 7D 左臂关节
    "tactile": { ... },                  # 仅视觉-触觉任务存在
    "prompt": str,                       # 任务自然语言描述
    "tactile_profile": str,              # 触觉标签，如 "tactile_raw"
    "task_id": str,                      # 当前任务 ID
}
```

### 3.2 AgileX `images` 字段

```python
new_obs["images"] = {
    "cam_high": np.ndarray,          # uint8 HWC，俯视 / 第三人称 RGB
    "cam_wrist_left": np.ndarray,    # uint8 HWC，左手腕相机 RGB
    "cam_wrist_right": np.ndarray,   # uint8 HWC，右手腕相机 RGB
}
```

分别对应数据集中的 `cam_high.mp4`、`cam_left_wrist.mp4`、`cam_right_wrist.mp4`。

### 3.3 AgileX `tactile` 字段

**纯视觉任务**与**视觉-触觉任务**共用同一套观测管线。区别仅在于：视觉-触觉任务会在 `new_obs["tactile"]` 中携带触觉/力觉数据，而**纯视觉任务不会返回任何触觉信息**（`new_obs["tactile"]` 不存在）。

对于视觉-触觉任务：

```python
new_obs["tactile"] = {
    # 触觉图像（来自 Xense 传感器）
    "left_gripper": {
        "rectify": np.ndarray,   # uint8 HWC (BGR)，左夹爪片触觉图
    },
    "right_gripper": {
        "rectify": np.ndarray,   # uint8 HWC (BGR)，右夹爪片触觉图
    },

    # 腕部力觉（来自力/力矩传感器）
    "left_wrist_force": {
        "wrench_6d": np.ndarray,  # float32 shape (6,)，[Fx, Fy, Fz, Tx, Ty, Tz]
    },
    "right_wrist_force": {
        "wrench_6d": np.ndarray,  # float32 shape (6,)
    },
}
```

**注意**：

- 触觉图像访问方式：`new_obs["tactile"]["left_gripper"]["rectify"]` / `["right_gripper"]["rectify"]`
- 力/力矩通过 `wrench_6d` 提供，例如：

  ```python
  left_force = new_obs["tactile"]["left_wrist_force"]["wrench_6d"]   # float32 (6,)
  right_force = new_obs["tactile"]["right_wrist_force"]["wrench_6d"] # float32 (6,)
  tactile_force = np.concatenate([left_force, right_force], axis=0)  # float32 (12,)
  ```

- 力觉数据为**原始传感器输出**。
- 真机测试时，我方返回的触觉信息与数据集中提供的触觉信息保持一致（包括 `tactile_information.hdf5` 中的 Marker2D / Mesh3D 等相关字段）。

### 3.4 Franka 顶层字段

已在 Franka endpose 远程推理链路中验证：

```python
{
    "images": { ... },                   # 视觉图像
    "joint_qpos": np.ndarray,            # float32 (8,)  7 关节 + 夹爪
    "joint_qpos_left": np.ndarray,       # float32 (8,)  有效臂别名
    "left_arm_joint_state": np.ndarray,  # float32 (8,)  与 joint_qpos 相同
    "left_end_pose": np.ndarray,         # float32 (7,)  [x, y, z, qw, qx, qy, qz]
    "state": np.ndarray,                 # float32 (32,) padding 状态缓冲
    "first_frame": np.ndarray,           # uint8 HWC，cam_high 的拷贝
    "prompt": str,                       # 任务自然语言描述
    "task_id": str,                      # 当前任务 ID
}
```

说明：

- `joint_qpos_left` / `left_arm_joint_state` / `left_end_pose` 指向有效 Franka 臂（沿用双臂接口命名）。
- 本体感知请优先使用 `joint_qpos` 与 `left_end_pose`；`state` 为固定长度 padding 缓冲。
- 当前 Franka 纯视觉任务**不包含**触觉字段。

### 3.5 Franka `images` 字段

```python
new_obs["images"] = {
    "cam_high": np.ndarray,        # uint8 (480, 640, 3)，第三人称 / 头部 RGB
    "cam_left_wrist": np.ndarray,  # uint8 (480, 640, 3)，腕部 RGB
    "cam_wrist": np.ndarray,       # uint8 (480, 640, 3)，腕部 RGB（与 cam_left_wrist 同源）
}
```

分别对应数据集中的 `third_person.mp4` 与 `wrist.mp4`。在当前 Franka 部署中，`cam_left_wrist` 与 `cam_wrist` 为同一路腕部相机流（为接口兼容而复制）。

---

## 四、示例策略

### 4.1 AgileX 虚拟策略（14 维关节）

```python
# dummy_policy_agilex.py
from typing import Any, Dict, Optional
import numpy as np


class Policy:
    def __init__(self, config_path: Optional[str] = None):
        self.chunk = 25
        self.action_dim = 14  # 双臂 qpos

    def reset(self, reset_info: Optional[Dict[str, Any]] = None) -> None:
        pass

    def infer(self, new_obs: Dict[str, Any]) -> Dict[str, Any]:
        # 示例：保持当前关节（真实参赛时请替换为模型推理）
        qpos = np.asarray(new_obs["joint_qpos"], dtype=np.float32).reshape(14)
        actions = np.repeat(qpos[None, :], self.chunk, axis=0)
        return {
            "actions": actions,
            "policy_metadata": {
                "policy_id": "dummy_agilex",
                "action_format": "joint_absolute",
            },
            "policy_timing": {"infer_ms": 0.0},
        }
```

### 4.2 Franka 虚拟策略（8 维 endpose）

```python
# dummy_policy_franka.py
from typing import Any, Dict, Optional
import numpy as np


class Policy:
    def __init__(self, config_path: Optional[str] = None):
        self.chunk = 25
        self.action_dim = 8  # end_pose_base

    def reset(self, reset_info: Optional[Dict[str, Any]] = None) -> None:
        pass

    def infer(self, new_obs: Dict[str, Any]) -> Dict[str, Any]:
        # 示例：保持当前末端位姿（真实参赛时请替换为模型推理）
        pose7 = np.asarray(new_obs["left_end_pose"], dtype=np.float32).reshape(7)
        gripper = np.asarray(new_obs["joint_qpos"], dtype=np.float32).reshape(-1)[-1:]
        end_pose_8d = np.concatenate([pose7, gripper], axis=0)
        actions = np.repeat(end_pose_8d[None, :], self.chunk, axis=0)
        return {
            "actions": actions.astype(np.float32),
            "policy_metadata": {
                "policy_id": "dummy_franka",
                "action_format": "end_pose_base",
            },
            "policy_timing": {"infer_ms": 0.0},
        }
```

### 4.3 启动示例策略

`track3_example/` 目录下已提供启动脚本 `start_policy_worker.sh`：

```bash
cd track3_example
bash start_policy_worker.sh
```

或手动执行：

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python serve_policy_worldarena.py \
  ./dummy_policy.py \
  --hub-url <PENDING_HUB_GATEWAY_URL>/policy \
  --worker-key <PENDING_POLICY_ID>
```

其中 `<PENDING_POLICY_ID>` 必须与我方机器 B 的 `benchmark_runner` 配置一致。请使用与分配平台动作格式匹配的示例策略。

---

## 五、Hub 网关地址

正式参赛时，我方会向参赛队伍提供具体的 Hub 网关地址：

```text
<PENDING_HUB_GATEWAY_URL>/policy
```

---

## 六、Worker Key 与身份匹配

机器 A 注册到 Hub 时使用的 `worker-key`，必须与我方向参赛团队提供的 `--worker-key` 一致。

| 机器 | 命令/配置 | key |
|---|---|---|
| A | `--worker-key <PENDING_POLICY_ID>` | 参赛团队与我方协商的 ID |

---

## 七、需要保持运行的服务

外部团队机器 A 上只需挂起一个进程。在 `track3_example/` 目录下直接运行：

```bash
bash start_policy_worker.sh
```

或手动执行：

```bash
python serve_policy_worldarena.py \
  ./dummy_policy.py \
  --hub-url <PENDING_HUB_GATEWAY_URL>/policy \
  --worker-key <PENDING_POLICY_ID>
```

该进程需要：
- 向 Hub 注册
- 心跳保活（约 15 秒一次）
- 长轮询接收 `infer` / `reset` 任务
- 推理后回传 `ActionPacket`

**不需要**开放 A 机器的入站端口，所有通信都是 A 主动出站到 Hub。

### 7.1 本地自查：启动虚拟 Hub（机器 B）

如需在本地模拟机器 B 的 Hub 进行自查或调试，可在 `track3_example/` 目录下启动本地虚拟 Hub：

```bash
bash start_hub.sh
```

默认监听：
- policy 端口：`127.0.0.1:18000`
- robot 端口：`127.0.0.1:19000`

然后在另一个终端让策略 Worker 接入本地 Hub：

```bash
export HUB_GATEWAY_URL=http://127.0.0.1:18000
export POLICY_ID=dummy_local
bash start_policy_worker.sh
```

---

## 八、网络要求

- 机器 A 能访问公网 HTTPS（具体域名正式参赛提供）。
- 火山网关**不支持 WebSocket Upgrade**，因此必须使用 HTTP Hub 长轮询模式。

---

## 九、环境说明

外部团队可在我们提供的通信环境上加入模型依赖环境。

### 9.1 创建 Python 环境

```bash
conda create -n real_eval python=3.10 -y
conda activate real_eval
```

### 9.2 安装 PyTorch

PyTorch 请根据参赛机器 CUDA 版本单独安装。以下仅作示例（CUDA 12.4）：

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

### 9.3 安装其他依赖

`track3_example/` 目录下提供了从评测环境导出的 `requirements.txt`。在 `track3_example/` 目录下执行：

```bash
pip install -r requirements.txt
```

---

## 十、常见问题

### Q1：动作坐标系是什么？

取决于平台：

- **AgileX：** 14 维绝对关节位置（`joint_absolute` / qpos），左臂在前、右臂在后。
- **Franka：** 8 维绝对末端位姿（`end_pose_base`）：`[x, y, z, qw, qx, qy, qz, gripper]`，位置在机器人基坐标系，四元数为 **wxyz**。

chunk 长度由策略自行决定，或赛前协商。

### Q2：如果策略崩溃或断开怎么办？

Hub 会检测到心跳丢失，我方调度侧会中止当前 episode。参赛团队重启 worker 后会自动重新注册。

### Q3：数据集文件与推理观测如何对应？

| 平台 | 数据集相机 | 推理图像键 | 控制标签 |
|---|---|---|---|
| AgileX | `cam_high` / `cam_left_wrist` / `cam_right_wrist` | `cam_high` / `cam_wrist_left` / `cam_wrist_right` | `observations/qpos` `(T, 14)` |
| Franka | `third_person` / `wrist` | `cam_high` / `cam_left_wrist` & `cam_wrist` | `observations/end_pose[:, 0:8]` |

---

## 十一、最小接入检查清单

- [ ] 已确认 `POLICY_ID`（正式参赛前提供）
- [ ] 已确认 Hub 网关地址（正式参赛前提供）
- [ ] 机器 A 可访问公网 HTTPS
- [ ] 已确认分配平台（**AgileX 14 维 qpos** 或 **Franka 8 维 endpose**）
- [ ] 策略实现 `Policy.infer(new_obs)` 并返回正确 `action_dim` 的 `{"actions": (chunk, action_dim) ndarray}`
- [ ] 已与我方约定测试时间窗口

---
