# WorldArena 外部策略接入指南

> 本文档面向需要在自有机器（机器 A）上部署策略 Worker，并接入我方中央调度（机器 B）与真机本体（机器 C）的参赛团队。

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
机器 A 可以进行长时间挂载策略

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
| `actions` | `np.ndarray` | 是 | 动作块，`shape=(chunk, action_dim)`。示例：`chunk=25`，`action_dim=14`（双臂关节，维度拼接顺序左臂-右臂）。具体 `chunk` 由策略自行决定 |

**动作语义**：`actions[t]` 表示第 `t` 步的目标关节位置（`joint_absolute`），我方会按顺序执行。返回的动作块长度 `chunk` 由策略自行决定（常见为 20、25 等），`action_dim` 为 14（双臂关节，前 7 维为左臂，后 7 维为右臂）。

---

## 三、我方发送给策略的观测 `new_obs`

### 3.1 顶层字段

```python
{
    "images": { ... },                  # 视觉图像
    "state": np.ndarray,                # 14D 双臂关节（前 7 维为左臂，后 7 维为右臂）
    "joint_qpos": np.ndarray,           # 14D 双臂关节（前 7 维为左臂，后 7 维为右臂）
    "right_arm_joint_state": np.ndarray,# 7D 右臂关节
    "left_arm_joint_state": np.ndarray, # 7D 左臂关节
    "tactile": { ... },                 # 触觉 / 力觉
    "prompt": str,                      # 任务自然语言描述
    "tactile_profile": str,             # 触觉标签，如 "tactile_raw"
    "task_id": str,                     # 当前任务 ID
}
```

### 3.2 `images` 字段

```python
new_obs["images"] = {
    "cam_high": np.ndarray,     # uint8 HWC，俯视/第三人称 RGB
    "cam_wrist_right": np.ndarray,    # uint8 HWC，右手腕相机 RGB
    "cam_wrist_left": np.ndarray,    # uint8 HWC，左手腕相机 RGB
}
```

### 3.3 `tactile` 字段

**纯视觉任务**与**视觉-触觉任务**共用同一套观测管线。区别仅在于：视觉-触觉任务会在 `new_obs["tactile"]` 中携带触觉/力觉数据，而**纯视觉任务不会返回任何触觉信息**（`new_obs["tactile"]` 不存在）。

对于视觉-触觉任务，`tactile` 字段包含两类 role, 注意**触觉只在右手夹爪上提供**：

```python
new_obs["tactile"] = {
    # 触觉图像（来自 Xense 传感器）
    "left_gripper": {
        "rectify": np.ndarray,   # uint8 HWC，左夹爪片触觉图
    },
    "right_gripper": {
        "rectify": np.ndarray,   # uint8 HWC，右夹爪片触觉图
    },

    # 腕部力觉（来自力/力矩传感器）
    "left_wrist_force": {
        "wrench_6d": np.ndarray,  # float32 shape (6,)，[Fx, Fy, Fz, Tx, Ty, Tz] 左贴片
    },
    "right_wrist_force": {
        "wrench_6d": np.ndarray,  # float32 shape (6,) 右贴片
    },
}
```

**注意**：
- `left_gripper` / `right_gripper` 提供的是**触觉图像**（`rectify`），访问方式例如：
  ```python
  left_tactile_image = new_obs["tactile"]["left_gripper"]["rectify"]   # uint8 HWC
  right_tactile_image = new_obs["tactile"]["right_gripper"]["rectify"] # uint8 HWC
  ```
- `left_wrist_force` / `right_wrist_force` 提供的是**合力/力矩** `wrench_6d`。访问方式例如：
  ```python
  left_force = new_obs["tactile"]["left_wrist_force"]["wrench_6d"]   # float32 (6,)
  right_force = new_obs["tactile"]["right_wrist_force"]["wrench_6d"] # float32 (6,)
  tactile_force = np.concatenate([left_force, right_force], axis=0)  # float32 (12,)
  ```
- 力觉数据为**原始传感器输出**。

---

## 四、示例：输出固定真值的虚拟策略

下面是一个最小可运行的示例策略，它忽略观测输入，直接输出一组固定动作。外部团队可在此基础上替换为自己的模型。

```python
# dummy_policy.py
from typing import Any, Dict, Optional
import numpy as np


class Policy:
    def __init__(self, config_path: Optional[str] = None):
        # 动作块长度与维度（示例，参赛队伍可自行设定）
        self.chunk = 25
        self.action_dim = 14  # 双臂关节：前 7 维为左臂，后 7 维为右臂

    def reset(self, reset_info: Optional[Dict[str, Any]] = None) -> None:
        pass

    def infer(self, new_obs: Dict[str, Any]) -> Dict[str, Any]:
        # 示例：输出固定动作序列（真实参赛时请替换为模型推理）
        actions = np.zeros((self.chunk, self.action_dim), dtype=np.float32)
        actions[:, 0] = 0.1   # 左臂关节 0
        actions[:, 7] = 0.1   # 右臂关节 0

        return {
            "actions": actions,
            "policy_metadata": {"policy_id": "dummy"},
            "policy_timing": {"infer_ms": 0.0},
        }
```

### 4.1 启动该示例策略

`challenge_pre/` 目录下已提供启动脚本 `start_policy_worker.sh`，使用相对路径即可直接启动策略 Worker：

```bash
cd challenge_pre
bash start_policy_worker.sh
```

脚本内容如下（也可手动执行）：

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python serve_policy_worldarena.py \
  ./dummy_policy.py \
  --hub-url <PENDING_HUB_GATEWAY_URL>/policy \
  --worker-key <PENDING_POLICY_ID>
```

其中 `<PENDING_POLICY_ID>` 必须与我方机器 B 的 `benchmark_runner` 配置一致。

---

## 五、Hub 网关地址

正式参赛前，我方会向参赛队伍提供具体的 Hub 网关地址：

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

外部团队机器 A 上只需挂起一个进程。在 `challenge_pre/` 目录下直接运行：

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

该进程会：
- 向 Hub 注册
- 心跳保活（约 15 秒一次）
- 长轮询接收 `infer` / `reset` 任务
- 推理后回传 `ActionPacket`

**不需要**开放 A 机器的入站端口，所有通信都是 A 主动出站到 Hub。

### 7.1 本地自查：启动虚拟 Hub（机器 B）

如需在本地模拟机器 B 的 Hub 进行自查或调试，可在 `challenge_pre/` 目录下启动本地虚拟 Hub：

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

`challenge_pre/` 目录下提供了从 ViTAL 环境导出的 `requirements.txt`。在 `challenge_pre/` 目录下执行：

```bash
pip install -r requirements.txt
```
---

## 十、常见问题


### Q1：动作坐标系是什么？

当前任务输出为**双臂 14D 关节绝对位置**（`joint_absolute`），前 7 维为左臂，后 7 维为右臂。动作维度为 14，chunk 长度由策略自行决定（示例为 25）。

### Q2：如果策略崩溃或断开怎么办？

Hub 会检测到心跳丢失，我方调度侧会中止当前 episode。参赛团队重启 worker 后会自动重新注册。

---

## 十一、最小接入检查清单

- [ ] 已确认 `POLICY_ID`（正式参赛前提供）
- [ ] 已确认 Hub 网关地址（正式参赛前提供）
- [ ] 机器 A 可访问公网 HTTPS
- [ ] 策略实现 `Policy.infer(new_obs)` 返回 `{"actions": (chunk, 14) ndarray}`（chunk 可自行决定或赛前协商）
- [ ] 已与我方约定测试时间窗口

---
