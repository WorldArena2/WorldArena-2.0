# Cosmos-Predict2.5 作为 RL 环境的集成指南（对照版）

本文档是 `docs/world_model_rl_integration.md` 的 **Cosmos-Predict2.5（RobotWin click_bell）专项落地版本**。  
目标是：在你已经有 Cosmos 训练脚本与推理脚本的前提下，将 Cosmos 接入 RLinf，作为世界模型环境参与 GRPO/PPO 训练。

---

## 目录

1. [整体架构（Cosmos 对照）](#1-整体架构cosmos-对照)
2. [核心数据流（Cosmos 对照）](#2-核心数据流cosmos-对照)
3. [Step 1：注册新环境类型](#step-1注册新环境类型)
4. [Step 2：实现 BaseWorldEnv 子类](#step-2实现-baseworldenv-子类)
   - [4.1 `__init__` 构造函数](#41-__init__-构造函数)
   - [4.2 `_build_dataset`：数据集加载](#42-_build_dataset数据集加载)
   - [4.3 `reset`：环境重置](#43-reset环境重置)
   - [4.4 `_infer_next_chunk_frames`：世界模型推理](#44-_infer_next_chunk_frames世界模型推理)
   - [4.5 `_infer_next_chunk_rewards`：奖励计算](#45-_infer_next_chunk_rewards奖励计算)
   - [4.6 `_wrap_obs`：观测打包](#46-_wrap_obs观测打包)
   - [4.7 `chunk_step`：驱动单步推进](#47-chunk_step驱动单步推进)
   - [4.8 `offload` / `onload`：显存管理](#48-offload--onload显存管理)
5. [Step 3：动作适配（prepare_actions）](#step-3动作适配prepare_actions)
6. [Step 4：环境配置 YAML（Cosmos 版）](#step-4环境配置-yamlcosmos-版)
7. [Step 5：训练配置 YAML（Cosmos 版）](#step-5训练配置-yamlcosmos-版)
8. [关键一致性检查清单（Cosmos 版）](#8-关键一致性检查清单cosmos-版)
9. [可直接 import 的 Cosmos 接口清单](#9-可直接-import-的-cosmos-接口清单)
10. [最小可运行落地步骤](#10-最小可运行落地步骤)
11. [双环境部署（推荐 A：RPC 解耦）](#11-双环境部署推荐-arpc-解耦)

---

## 1. 整体架构（Cosmos 对照）

Cosmos 接入 RLinf 的方式与 Wan / OpenSora 一致：  
世界模型只在 `EnvWorker` 内部实现，`Runner`、`Actor`、`Rollout` 主流程不需要改。

```text
run_embodiment.sh
  -> Runner
    -> RolloutWorker (Policy 输出 chunk actions)
      -> prepare_actions()
      -> EnvWorker(CosmosEnv).chunk_step()
         -> _infer_next_chunk_frames(actions)
         -> _infer_next_chunk_rewards()
         -> _wrap_obs()
    -> ActorWorker (GRPO/PPO 更新)
```

---

## 2. 核心数据流（Cosmos 对照）

### 2.1 Cosmos 推理脚本给出的关键约束

在 `cosmos-predict2.5/scripts/infer_vla_click_bell.py` 中：

- 仅支持 `num_conditional_frames=1`
- 自回归 rollout 的 chunk 大小为 `dataset.sequence_length - 1`

这意味着 Cosmos 环境通常采用：

- `condition_frame_length = 1`
- `chunk = 12`（当 `sequence_length=13` 时）
- `num_frames = condition_frame_length + chunk = 13`

### 2.2 训练时数据链路

```text
reset() 读入首帧 (uint8 [0,255] 或 float [0,1])
  -> 内部缓存 current_obs: float32 [-1,1], [B,3,1,T,H,W]
  -> _wrap_obs(): uint8 [0,255], [B,H,W,3] (给 Policy)
  -> policy 输出动作: [B,chunk,action_dim]
  -> Cosmos 归一化动作后推理，生成新帧
  -> 奖励模型使用最新 chunk 帧打分 [B,chunk]
```

---

## Step 1：注册新环境类型

**文件：** `rlinf/envs/__init__.py`

### 1.1 增加枚举

```python
class SupportedEnvType(Enum):
    ...
    COSMOSWM = "cosmos_wm"
```

### 1.2 增加工厂分支

```python
elif env_type == SupportedEnvType.COSMOSWM:
    from rlinf.envs.world_model.world_model_cosmos_env import CosmosEnv
    return CosmosEnv
```

---

## Step 2：实现 BaseWorldEnv 子类

**建议文件：** `rlinf/envs/world_model/world_model_cosmos_env.py`

可直接参考：

- `rlinf/envs/world_model/world_model_wan_env.py`
- `rlinf/envs/world_model/world_model_opensora_env.py`
- `rlinf/envs/world_model/world_model_ctrlworld_env.py`

### 4.1 `__init__` 构造函数

Cosmos 版本建议初始化以下核心字段：

| 字段 | 含义 | 推荐值（RobotWin click_bell） |
|------|------|-------------------------------|
| `self.chunk` | 每次环境步预测帧数 | `12` |
| `self.condition_frame_length` | 条件帧长度 | `1` |
| `self.num_frames` | `condition + chunk` | `13` |
| `self.action_dim` | 动作维度 | `14` |
| `self.image_size` | 观测分辨率 | `[256, 256]` |
| `self.current_obs` | 内部时序缓存 | `None`（reset 后初始化） |
| `self.task_descriptions` | 任务文本 | 长度为 `num_envs` |

模型加载建议：

1. 世界模型：使用 `load_model_from_checkpoint(...)`
2. 奖励模型：优先复用 `RoboTwinT5CrossAttn`（与 Wan / OpenSora RobotWin 一致）

### 4.2 `_build_dataset`：数据集加载

建议两种实现路径：

1. **快速对齐**：直接使用 Cosmos 的 `Dataset_3D`
2. **工程化**：包装成 RLinf 世界模型统一数据集接口（只暴露 reset 必需字段）

reset 需要至少拿到：

- 首帧图像
- task description
- （可选）state/init pose

### 4.3 `reset`：环境重置

典型流程：

1. 按 `episode_indices` 选择轨迹
2. 取首帧并 resize 到 `image_size`
3. 归一化到 `[-1,1]`
4. 初始化 `self.current_obs`，形状 `[B,3,1,1,H,W]`
5. 记录 `task_descriptions`
6. 返回 `_wrap_obs()`

### 4.4 `_infer_next_chunk_frames`：世界模型推理

该方法是 Cosmos 集成核心。建议迁移自 `infer_vla_click_bell.py` 的 autoregressive 逻辑：

1. 输入动作 `actions: [B,chunk,action_dim]`
2. 做动作归一化（q01/q99 到 `[-1,1]`）
3. 构造 Cosmos batch（含首帧 + 动作序列）
4. 调用：
   - `model.generate_samples_from_batch(...)`
   - `model.decode(...)`（若存在）
5. 取预测段并统一到 `[-1,1]`
6. 追加到 `self.current_obs`

建议方法签名：

```python
def _infer_next_chunk_frames(self, actions: np.ndarray | torch.Tensor) -> None:
    ...
```

### 4.5 `_infer_next_chunk_rewards`：奖励计算

与已有世界模型环境保持一致：

1. 取 `self.current_obs` 中最后 `chunk` 帧
2. `[-1,1] -> [0,1]`
3. reshape 为 `[B*chunk,3,H,W]`
4. 奖励模型逐帧打分并 reshape 回 `[B,chunk]`

### 4.6 `_wrap_obs`：观测打包

返回字段必须与现有 world model env 完全一致：

```python
obs = {
    "main_images": uint8 [B,H,W,3],
    "wrist_images": None,
    "states": float32 zeros [B,action_dim],
    "task_descriptions": list[str],
}
```

### 4.7 `chunk_step`：驱动单步推进

流程建议：

1. `_infer_next_chunk_frames(policy_output_action)`
2. `chunk_rewards = _infer_next_chunk_rewards()`
3. `chunk_rewards_tensors = _calc_step_reward(chunk_rewards)`
4. 按阈值在最后一个 step 标记 termination
5. 超出 `max_episode_steps` 标记 truncation
6. 返回 5-tuple（与 Wan/OpenSora 对齐）

### 4.8 `offload` / `onload`：显存管理

建议最小实现：

- `offload`: world model + reward model -> CPU，`torch.cuda.empty_cache()`
- `onload`: 迁回 `self.device`

---

## Step 3：动作适配（prepare_actions）

**文件：** `rlinf/envs/action_utils.py`

新增 `cosmos_wm` 分支，RobotWin 可先透传：

```python
elif env_type == SupportedEnvType.COSMOSWM:
    if wm_env_type == "robotwin":
        chunk_actions = raw_chunk_actions
    else:
        raise NotImplementedError(...)
```

建议实践：

- `prepare_actions()` 只做任务语义适配（例如 gripper 维符号）
- Cosmos 的 q01/q99 归一化放在 `CosmosEnv._infer_next_chunk_frames()` 内

---

## Step 4：环境配置 YAML（Cosmos 版）

**建议文件：** `examples/embodiment/config/env/cosmos_robotwin_click_bell.yaml`

示例模板：

```yaml
env_type: cosmos_wm
task_suite_name: robotwin_click_bell
wm_env_type: robotwin

total_num_envs: null
auto_reset: False
ignore_terminations: False
max_steps_per_rollout_epoch: 240
max_episode_steps: 240

use_rel_reward: True
reward_coef: 5.0

seed: 0
group_size: 1
use_fixed_reset_state_ids: True
use_ordered_reset_state_ids: False
specific_reset_id: null

video_cfg:
  save_video: True
  info_on_video: True
  video_base_dir: ${runner.logger.log_path}/video/train

enable_offload: False

# ===== Cosmos world model =====
chunk: 12
condition_frame_length: 1
num_frames: 13
image_size: [256, 256]
action_dim: 14
action_key: abs_action

# 数据/统计文件
initial_image_path: /path/to/cosmos_dataset_root
stat_path: /path/to/dataset_statistics.json

# Cosmos checkpoint 加载参数
world_model:
  experiment: vla_click_bell_action_conditioned_rectified_flow_2b_240x320
  ckpt_path: /path/to/checkpoints/iter_0000xxxx
  config_file: cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py
  guidance: 0.0
  num_steps: 35
  num_conditional_frames: 1

reward_model:
  type: RoboTwinT5CrossAttn
  from_pretrained: /path/to/reward_model.pth
  t5_model_name: /path/to/t5-base
```

---

## Step 5：训练配置 YAML（Cosmos 版）

**建议文件：** `examples/embodiment/config/cosmos_robotwin_click_bell_grpo_openpi_pi05.yaml`

关键是三处一致：

1. `env.train.chunk`
2. `actor.model.num_action_chunks`
3. `actor.model.openpi.action_chunk`

示例（仅示意关键项）：

```yaml
algorithm:
  adv_type: grpo
  loss_type: actor
  group_size: 4
  reward_coef: 5.0

env:
  train:
    total_num_envs: 4
    group_size: ${algorithm.group_size}
    chunk: 12
    action_dim: 14
    reward_coef: ${algorithm.reward_coef}

actor:
  model:
    num_action_chunks: 12
    action_dim: 14
    openpi:
      action_chunk: 12
      action_env_dim: 14
```

---

## 8. 关键一致性检查清单（Cosmos 版）

提交前逐项确认：

- `env.chunk == actor.model.num_action_chunks == actor.model.openpi.action_chunk == 12`
- `env.action_dim == actor.model.action_dim == actor.model.openpi.action_env_dim == 14`
- `num_frames == condition_frame_length + chunk == 13`
- `num_conditional_frames == 1`（与 Cosmos 推理逻辑一致）
- 动作归一化统计文件存在，且维度与 action_dim 一致
- 图像链路一致：`uint8 -> [-1,1] -> uint8(policy) / [0,1](reward)`
- `total_num_envs % group_size == 0`
- `total_num_envs % env_world_size == 0`

---

## 9. 可直接 import 的 Cosmos 接口清单

本节用于回答一个常见问题：在实现 `CosmosEnv` 时，哪些能力可以直接从 `cosmos-predict2.5` 引入，哪些更建议在 RLinf 侧自维护。

### 9.1 推荐直接 import（稳定模块 API）

以下接口建议直接使用：

```python
from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint
from cosmos_predict2._src.predict2.action.datasets.dataset_local import Dataset_3D
from cosmos_predict2._src.predict2.action.models.action_conditioned_video2world_rectified_flow_model import NUM_CONDITIONAL_FRAMES_KEY
from cosmos_predict2._src.predict2.inference.utils import read_video
```

说明：

- `load_model_from_checkpoint`：用于加载 Cosmos 世界模型（核心入口）
- `Dataset_3D`：可用于快速复用 click_bell 数据读取逻辑
- `NUM_CONDITIONAL_FRAMES_KEY`：batch 字段常量，避免手写字符串
- `read_video`：仅在需要 GT 对比/可视化时使用

### 9.2 可选 import（配置常量）

在快速接入阶段，也可以直接引用配置模块中的常量，例如：

- `vla_click_bell_action_scaler`
- `vla_click_bell_base_path`
- `vla_click_bell_stat_path`
- `vla_click_bell_train_annotation_path`
- `vla_click_bell_val_annotation_path`

这些常量位于：

`cosmos_predict2._src.predict2.action.configs.action_conditioned.data`

更工程化的做法是把路径和 scaler 写入 RLinf 的 env yaml，而不是硬编码 import。

### 9.3 不是模块函数，而是模型实例方法

下面两个调用不是独立函数，不能直接 `from ... import`：

- `model.generate_samples_from_batch(...)`
- `model.decode(...)`

正确用法是先通过 `load_model_from_checkpoint(...)` 得到 `model`，再调用模型实例方法。

### 9.4 不建议直接 import 的脚本级 helper

`scripts/infer_vla_click_bell.py` 中下列函数是脚本 glue code，建议在 `CosmosEnv` 内部实现同等私有方法，而不是跨脚本引用：

- `batchify`
- `move_to_device`
- `maybe_compute_text_embeddings`
- `normalize_actions`
- `build_data_from_frame_actions`

建议在 `CosmosEnv` 内部对应实现为：

- `_batchify(...)`
- `_move_to_device(...)`
- `_maybe_compute_text_embeddings(...)`
- `_normalize_actions(...)`
- `_build_data_from_frame_actions(...)`

这样可以减少对脚本文件结构变更的耦合风险。

---

## 10. 最小可运行落地步骤

建议按以下最小顺序推进：

1. 先加 `cosmos_wm` 枚举和 `CosmosEnv` 空壳，确保配置可加载
2. 实现 `reset()` + `_wrap_obs()`，验证观测 shape/range
3. 实现 `_infer_next_chunk_frames()`，单 env 验证能生成并更新 `current_obs`
4. 实现 `_infer_next_chunk_rewards()` + `chunk_step()`
5. `total_num_envs=1` 先做 smoke test
6. 再放开到 GRPO（`group_size>1`）

---

## 11. 双环境部署（推荐 A：RPC 解耦）

当 `cosmos-predict2.5` 与 RLinf 训练环境（如 openpi+robotwin）在同一 venv 发生依赖冲突时，建议使用 RPC 解耦：

- **Cosmos 环境（Python 3.10）**：只启动推理服务
- **RLinf 环境（现有训练 venv）**：只跑训练，`CosmosEnv` 通过 HTTP 调用服务

### 11.1 启动 Cosmos RPC 服务

在 cosmos 环境中执行：

```bash
cd cosmos-predict2.5
python scripts/serve_vla_click_bell_rpc.py \
  --host 127.0.0.1 \
  --port 18080 \
  --device cuda \
  --ckpt_path /your/path//tmp/imaginaire4-output/cosmos_predict2_action_conditioned/vla_robotwin/vla_click_bell_/checkpoints/iter_000002000
```

### 11.2 RLinf 侧配置

`examples/embodiment/config/env/cosmos_robotwin_click_bell.yaml` 中开启：

```yaml
inference_backend: rpc
rpc:
  url: http://127.0.0.1:18080
```

训练时 `CosmosEnv` 会：

- `reset` 调 `/reset_samples`
- `chunk_step` 调 `/infer_chunk`

无需在 RLinf 训练 venv 内安装 cosmos 依赖。

### 11.3 健康检查

RPC 服务启动后可先检查：

```bash
curl http://127.0.0.1:18080/health
curl http://127.0.0.1:18080/dataset_size
```

---

## 附：建议的 CosmosEnv 方法骨架

```python
class CosmosEnv(BaseWorldEnv):
    def __init__(...):
        super().__init__(...)
        ...

    def _build_dataset(self, cfg):
        ...

    @torch.no_grad()
    def reset(self, *, seed=None, options=None, episode_indices=None):
        ...
        return self._wrap_obs(), {}

    @torch.no_grad()
    def _infer_next_chunk_frames(self, actions):
        ...

    @torch.no_grad()
    def _infer_next_chunk_rewards(self):
        ...
        return chunk_rewards

    def _wrap_obs(self):
        ...
        return obs

    @torch.no_grad()
    def chunk_step(self, policy_output_action):
        ...
        return [obs], chunk_rewards_tensors, chunk_terminations, chunk_truncations, [infos]

    def offload(self):
        ...

    def onload(self):
        ...
```

---

如果后续你希望，我可以基于这份文档再给一版“代码侧逐文件修改清单”（精确到每个文件要新增/改动的函数和字段），用于直接开工提 PR。
