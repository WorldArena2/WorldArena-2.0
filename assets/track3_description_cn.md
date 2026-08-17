# Track 3：真机操作任务赛道
**[中文文档](https://github.com/WorldArena2/WorldArena-2.0/blob/main/assets/track3_description_cn.md)**

**[English Document](https://github.com/WorldArena2/WorldArena-2.0/blob/main/assets/track3_description.md)**
## 概述

Track 3 是 WorldArena benchmark 的**真机操作任务赛道**。参赛队伍需在真实机器人上部署策略，通过**真机执行成功率**进行评判。

Track 3 当前覆盖**两套机器人本体**：

| 平台 | 形态 | 控制方式 | 传感器 |
|---|---|---|---|
| **AgileX** | 双臂协作机器人 | 绝对**关节 / qpos**（`joint_absolute`，14 维） | 头部 + 左右腕部相机；可选 Xense 触觉 |
| **Franka** | Franka Emika Panda 单臂 | 绝对**末端位姿 / endpose**（`end_pose_base`，8 维） | 第三人称 + 腕部相机 |

我们鼓励参赛的世界模型（WAM）同时尝试 AgileX 上的**纯视觉任务**与**视觉-触觉任务**，以及 Franka 上的纯视觉任务，以展示在视觉与接触反馈下的真实世界泛化操作能力。

相关任务数据请参见 Hugging Face：[WorldArena/WorldArena2.0](https://huggingface.co/datasets/WorldArena/WorldArena2.0)

---

## 任务类别

### 1. AgileX 纯视觉任务

这些任务主要依赖视觉感知（头部相机与左右腕部相机），不提供触觉反馈。

| 任务 | 描述 |
|---|---|
| 擦桌子 | 抓取毛巾并擦拭桌面。 |
| 倒水 | 拿起容器并将水倒入目标杯子。 |
| 清理桌面 | 将桌面上的物品移走、分类并放到指定位置。 |
| 指令跟随清理桌面 | 按照自然语言指令清理桌面（例如：“把红色方块放到篮子里”）。 |
| 手冲咖啡 | 完成手冲咖啡流程，包括放置滤纸和倒水。 |
| 叠衣服 | 将桌上的一件衣物折叠好。 |
| 叠纸盒 | 将扁平的纸盒折叠成组装好的形状。 |

### 2. AgileX 视觉-触觉任务

这些任务需要进行接触丰富的操作。提供触觉反馈（力/力矩与触觉图像）以实现稳健执行。

| 任务 | 描述 |
|---|---|
| 夹薯片 | 拿起一片易碎的薯片而不压碎它，放到盘子里。 |
| 削皮黄瓜 | 固定黄瓜并用削皮器削去外皮。 |
| 插两孔插头 | 对准并将两孔插头插入插座。 |

### 3. Franka 纯视觉任务

这些任务在 Franka Emika Panda 单臂平台上采集与评测，配备第三人称相机与腕部相机，不提供触觉反馈。

| 任务 | 描述 |
|---|---|
| 擦桌子 | 抓取毛巾/抹布并擦拭桌面。 |
| 倒水 | 拿起容器并将水倒入目标杯子。 |
| 清理桌面 | 将桌面上的物品移走、分类并放到指定位置。 |

---

## 数据集

Track 3 的真机演示数据托管于 Hugging Face：[WorldArena/WorldArena2.0](https://huggingface.co/datasets/WorldArena/WorldArena2.0)。仓库按任务组织，每个任务包含若干条真人演示的真机 episode。

### 任务目录

| 目录名 | 任务名称 | 平台 / 类别 |
|---|---|---|
| `clean_table` | 清理桌面 | AgileX 纯视觉（含 Franka 变体） |
| `clean_table_instruction_follow` | 指令跟随清理桌面 | AgileX 纯视觉 |
| `fold_box` | 叠纸盒 | AgileX 纯视觉 |
| `fold_shirt` | 叠衣服 | AgileX 纯视觉 |
| `pour_over_coffee` | 手冲咖啡 | AgileX 纯视觉 |
| `pour_water` | 倒水 | AgileX 纯视觉（含 Franka 变体） |
| `wipe_table` | 擦桌子 | AgileX 纯视觉（含 Franka 变体） |
| `insert` | 插两孔插头 | AgileX 视觉-触觉 |
| `peel_cucumber` | 削皮黄瓜 | AgileX 视觉-触觉 |
| `pick_potato_chip` | 夹薯片 | AgileX 视觉-触觉 |

### 数据规模

AgileX 上每个任务约包含 **100 条**演示数据，其中 `pour_over_coffee`（手冲咖啡）目前共有 **83 条**有效演示。Franka 则为上述三个纯视觉任务提供演示数据。

---

## 平台 A：AgileX（关节 / qpos 控制）

### 单条 Episode 文件（纯视觉）

- `cam_high.mp4`：头部 / 第三人称相机视频。
- `cam_left_wrist.mp4`：左臂腕部相机视频。
- `cam_right_wrist.mp4`：右臂腕部相机视频。
- `episode.hdf5`：运行时机器人状态与动作；动作 / qpos 为 **14 维关节位置**。
- `meta.json`：该 episode 的元信息（任务标签、设备、采集时间等）。

### 单条 Episode 文件（视觉-触觉）

在纯视觉文件基础上额外提供：

- `tactile_gripper_left_rectify.mp4`：左夹爪触觉贴片 rectify 后的视频。
- `tactile_gripper_right_rectify.mp4`：右夹爪触觉贴片 rectify 后的视频。
- `tactile_information.hdf5`：触觉力学信息（力、Marker2D、Mesh3D 等）。

触觉 episode 的 `episode.hdf5` 还会保存触觉时间戳，以及 `tactile_enabled=True` 等属性。

### HDF5 要点

| 数据集 | 形状（示例） | 说明 |
|---|---|---|
| `observations/qpos` | `(T, 14)` | 双臂关节位置：左 7 + 右 7（各 6 关节 + 夹爪）。 |
| `action` | `(T, 14)` | 记录的关节动作流（布局与 qpos 相同）。 |
| `observations/end_pose` | `(T, 14)` | 双臂末端位姿（仅供参考；**不是**控制指令）。 |
| `observations/qvel` / `observations/effort` | `(T, 14)` | 关节速度 / 力矩。 |

典型采集帧率：**30 FPS**。

### 动作格式（AgileX）

AgileX 真机测试采用**绝对关节位置**（`joint_absolute` / qpos）。动作向量为 **14 维**：

```text
[左臂 j1..j6, 左夹爪, 右臂 j1..j6, 右夹爪]
```

策略输出需为关节空间动作，真机系统接收后直接执行。控制用演示标签位于 `observations/qpos`（以及对齐的 `action` 流）。

---

## 平台 B：Franka（末端位姿 / endpose 控制）

### 单条 Episode 文件

- `third_person.mp4`：第三人称 / 头部相机视频（推理时对应 `cam_high`）。
- `wrist.mp4`：腕部相机视频（推理时对应 `cam_wrist` / `cam_left_wrist`）。
- `episode.hdf5`：运行时机器人状态与 endpose / 关节轨迹。
- `robot_state.json`：逐帧机器人状态日志（末端位姿、关节状态、夹爪开合等）。
- `metadata.json`：该 episode 的元信息（任务标签、FPS、帧数、时长等）。
- `camera_timestamps.json`：各相机时间戳与帧对齐信息。

### HDF5 要点

| 数据集 | 形状（示例） | 说明 |
|---|---|---|
| `observations/end_pose` | `(T, 16)` | 末端位姿轨迹。有效 Franka 臂占用前 **8** 维，其余为双臂风格 padding。 |
| `observations/qpos` | `(T, 14)` | 关节位置（带 padding；仅供参考）。 |
| `observations/qvel` / `observations/effort` | `(T, 14)` | 关节速度 / 力矩（带 padding）。 |
| `action` | `(T, 14)` | 记录流（带 padding）。 |
| `observation/camera/head` / `left` | `(T,)` | 编码后的相机帧。 |

典型采集帧率：**15 FPS**。

### 动作格式（Franka）

Franka 真机测试采用**绝对末端位姿**控制（`end_pose_base`）。有效动作向量为 **8 维**：

```text
[x, y, z, qx, qy, qz, qw, gripper]
```

- `x, y, z`：末端在**机器人基坐标系**下的位置（米）
- `qx, qy, qz, qw`：姿态四元数（**xyzw**）
- `gripper`：夹爪开合指令

在 HDF5 中，请使用 `observations/end_pose[:, 0:8]` 作为有效 Franka endpose 标签。

---

## 评价指标

每个任务通过其**真机执行成功率**进行评价。如果机器人在允许的步数内完成整个任务且未触发安全干预，则该次尝试视为成功。最终排名综合所有任务的成功率，并可能按任务难度进行归一化。

---

## 任务难度评分

我们根据所需精度、接触推理、可变形物体处理和多步规划复杂度，对每个任务进行 **1（最简单）到 10（最难）** 的评分。

### AgileX 纯视觉任务

| 任务 | 难度（1-10） | 评分理由 |
|---|---|---|
| 擦桌子 | 4 | 重复运动，目标区域大，精度要求低。 |
| 倒水 | 6 | 需要控制倾斜角度并视觉跟踪液面。 |
| 清理桌面 | 5 | 抓取、分类与搬运物体，中等精度要求。 |
| 指令跟随清理桌面 | 6 | 在桌面清理基础上增加语言理解。 |
| 手冲咖啡 | 9 | 多步骤精细操作，涉及易碎物体和液体处理。 |
| 叠衣服 | 9 | 可变形物体操作，需要处理褶皱和精确折叠。 |
| 叠纸盒 | 9 | 刚性部件装配，需要精确折痕和边角对齐。 |

### AgileX 视觉-触觉任务

| 任务 | 难度（1-10） | 评分理由 |
|---|---|---|
| 夹薯片 | 8 | 易碎物体，小物体，需要轻柔夹持力和滑移检测。 |
| 削皮黄瓜 | 9 | 持续接触、稳定力度和协调的手臂运动。 |
| 插两孔插头 | 8 | 精确对准加上接触丰富的插入过程。 |

### Franka 纯视觉任务

| 任务 | 难度（1-10） | 评分理由 |
|---|---|---|
| 擦桌子 | 4 | 重复运动，目标区域大，精度要求低。 |
| 倒水 | 6 | 需要控制倾斜角度并视觉跟踪液面。 |
| 清理桌面 | 5 | 抓取、分类与搬运物体，中等精度要求。 |

---

## 参赛建议

- 参赛队伍可提交任意任务 / 平台子集。
- 我们鼓励 WAM 同时参加 AgileX 的**纯视觉类别与视觉-触觉类别**，因为后者测试的是具备接触感知能力的世界模型。
- **控制方式必须与平台匹配**：
  - AgileX → **14 维关节 / qpos**（`joint_absolute`）
  - Franka → **8 维末端位姿**（`end_pose_base`）
- 策略需对真实世界的传感噪声、延迟和部分可观测性具备鲁棒性。
- 接口、分平台观测字段与部署说明请参见同文件夹下的 [policy_guide_cn.md](https://github.com/WorldArena2/WorldArena-2.0/blob/main/assets/policy_guide_cn.md)。
