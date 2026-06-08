# iVideoGPT Transformer 作为 RL 环境的集成指南（对照版）

本文档是 `docs/world_model_rl_integration.md` 的 **iVideoGPT（RobotWin click_bell）专项落地版本**。  
目标是：在你已经有

- 数据编码与模型训练脚本：`iVideoGPT/scripts/finetune/robotwin-click-bell-64-act-cond.sh`
- 推理与可视化脚本：`iVideoGPT/scripts/inference/infer-click-bell-transformer.sh`

的前提下，把 iVideoGPT 接入 RLinf，作为世界模型环境参与 GRPO/PPO 训练。

---

## 目录

1. [整体架构（iVideoGPT 对照）](#1-整体架构ivideogpt-对照)
2. [核心数据流（iVideoGPT 对照）](#2-核心数据流ivideogpt-对照)
3. [Step 0：先决条件（编码、训练、离线推理）](#step-0先决条件编码训练离线推理)
4. [Step 1：注册新环境类型](#step-1注册新环境类型)
5. [Step 2：实现 BaseWorldEnv 子类](#step-2实现-baseworldenv-子类)
   - [4.1 `__init__` 构造函数](#41-__init__-构造函数)
   - [4.2 `_build_dataset`：数据集加载](#42-_build_dataset数据集加载)
   - [4.3 `reset`：环境重置](#43-reset环境重置)
   - [4.4 `_infer_next_chunk_frames`：世界模型推理](#44-_infer_next_chunk_frames世界模型推理)
   - [4.5 `_infer_next_chunk_rewards`：奖励计算](#45-_infer_next_chunk_rewards奖励计算)
   - [4.6 `_wrap_obs`：观测打包](#46-_wrap_obs观测打包)
   - [4.7 `chunk_step`：驱动单步推进](#47-chunk_step驱动单步推进)
   - [4.8 `offload` / `onload`：显存管理](#48-offload--onload显存管理)
6. [Step 3：动作适配（prepare_actions）](#step-3动作适配prepare_actions)
7. [Step 4：环境配置 YAML（iVideoGPT 版）](#step-4环境配置-yamlivideogpt-版)
8. [Step 5：训练配置 YAML（iVideoGPT 版）](#step-5训练配置-yamlivideogpt-版)
9. [关键一致性检查清单（iVideoGPT 版）](#关键一致性检查清单ivideogpt-版)
10. [与 Wan / OpenSora 的差异点](#与-wan--opensora-的差异点)

---

## 1. 整体架构（iVideoGPT 对照）

iVideoGPT 接入 RLinf 的方式与 Wan / OpenSora 一致：  
世界模型实现放在 `EnvWorker` 内部，`Runner` / `Actor` / `Rollout` 主流程不改。

```text
run_embodiment.sh
  -> Runner
    -> RolloutWorker (Policy 输出 chunk actions)
      -> prepare_actions()
      -> EnvWorker(iVideoGPTEnv).chunk_step()
         -> _infer_next_chunk_frames(actions)
         -> _infer_next_chunk_rewards()
         -> _wrap_obs()
    -> ActorWorker (GRPO/PPO 更新)
```

---

## 2. 核心数据流（iVideoGPT 对照）

```text
reset() 取初始图像 uint8 [0,255]
  -> 内部缓存 current_obs: float32 [-1,1], [B,3,1,T,H,W]
  -> _wrap_obs(): uint8 [0,255], [B,H,W,3] (给 Policy)
  -> policy 输出动作: [B,chunk,action_dim]
  -> iVideoGPT: tokenize(context frame) + action-conditioned generate + detokenize
  -> 生成新帧: float32 [0,1] -> 转为 [-1,1]
  -> 奖励模型输入最新 chunk 帧: float32 [0,1]
  -> chunk_rewards: [B,chunk]
```

关键点：  
iVideoGPT 的推理核心不是 diffusion scheduler，而是

- `CompressiveVQModel.tokenize / detokenize`
- `HeadModelWithAction.generate(...)`

这部分逻辑直接来自 `iVideoGPT/inference/predict.py`。

---

## Step 0：先决条件（编码、训练、离线推理）

这一节对应你当前已有 pipeline，可作为接入前检查：

1. **数据编码完成**：`dataset_path` 指向编码后的 `.npz` 数据目录
2. **模型训练完成**：`robotwin-click-bell-64-act-cond.sh` 产出 transformer checkpoint（`model.safetensors`）
3. **离线推理可运行**：`infer-click-bell-transformer.sh` 能正确出图（gif）

只要这三步已打通，RL 集成的核心就是把 `predict.py` 的单次推理流程迁移到环境的 `_infer_next_chunk_frames()`。

---

## Step 1：注册新环境类型

**文件：** `rlinf/envs/__init__.py`

### 1.1 新增枚举

```python
class SupportedEnvType(str, Enum):
    ...
    IVIDEOGPTWM = "ivideogpt_wm"
```

### 1.2 新增工厂分支

```python
def get_env_cls(env_type, env_cfg=None, ...):
    ...
    elif env_type == SupportedEnvType.IVIDEOGPTWM:
        from rlinf.envs.world_model.world_model_ivideogpt_env import IVideoGPTEnv
        return IVideoGPTEnv
```

---

## Step 2：实现 BaseWorldEnv 子类

**建议文件：** `rlinf/envs/world_model/world_model_ivideogpt_env.py`

参考实现：

- `rlinf/envs/world_model/world_model_wan_env.py`
- `rlinf/envs/world_model/world_model_opensora_env.py`

### 4.1 `__init__` 构造函数

iVideoGPT 版建议初始化以下字段：

| 字段 | 含义 | RobotWin click_bell 推荐值 |
|------|------|----------------------------|
| `self.chunk` | 每次环境步生成帧数 | `8`（与 Pi05 `action_chunk` 对齐） |
| `self.condition_frame_length` | 条件帧长度 | `1`（与你的推理脚本一致） |
| `self.segment_length` | 推理总帧长度 | `condition_frame_length + chunk` |
| `self.action_dim` | 动作维度 | `14` |
| `self.image_size` | 输出分辨率 | `256` |
| `self.current_obs` | 内部缓存 | reset 后为 `[B,3,1,T,H,W]` |

模型加载建议：

1. `CompressiveVQModel.from_pretrained(model_bundle, subfolder="tokenizer")`
2. `AutoConfig.from_pretrained(model_bundle, subfolder="transformer")`
3. `AutoModelForCausalLM.from_config(config)`
4. `HeadModelWithAction(...)`
5. `load_file(.../transformer/model.safetensors)` 后 `load_state_dict`
6. 奖励模型（复用现有 `RoboTwinT5CrossAttn` 或 `QwenVLMProgressRewardModel`）

> `model_bundle` 目录结构应与推理脚本一致：`tokenizer/` + `transformer/`。

### 4.2 `_build_dataset`：数据集加载

需要能在 `reset()` 时拿到：

- 初始图像（`uint8 [0,255]`，`[T,H,W,3]`）
- 动作序列（`float32 [T,14]`，可选）
- 任务描述（可选）

可直接参考 `iVideoGPT/inference/utils.py` 的 `NPZParser` 约定：

- 图像 key 默认 `image`（或 `DISPLAY_KEY` 中定义的 key）
- 动作 key 为 `action`

### 4.3 `reset`：环境重置

推荐流程：

1. 根据 `episode_indices` 选轨迹
2. 取首帧或首 `condition_frame_length` 帧
3. 转成 float32 并做 `[-1,1]` 归一化
4. 初始化 `self.current_obs` 为 `[B,3,1,T,H,W]`
5. 填充 `task_descriptions`
6. 返回 `_wrap_obs()`

### 4.4 `_infer_next_chunk_frames`：世界模型推理

这是 iVideoGPT 接入核心，对应迁移自 `inference/predict.py`。

**输入动作：**

- 形状：`[B, chunk, action_dim]`
- dtype：`float32`
- RobotWin：通常不做额外符号翻转，直接透传

**核心实现思路（单次 chunk）：**

1. 从 `self.current_obs` 取最近 `condition_frame_length` 帧，转换成 `[0,1]` 的 `pixel_values`
2. 调 `tokenizer.tokenize(pixel_values, context_length=self.condition_frame_length)`
3. 构造 `gen_input`（与 `predict.py` 同逻辑）
4. `model.generate(..., action=actions_tensor)` 生成未来 token
5. `tokenizer.detokenize(...)` 得到未来帧（`[0,1]`）
6. 取最后 `chunk` 帧，转为 `[-1,1]`
7. 追加到 `self.current_obs`

建议骨架：

```python
@torch.no_grad()
def _infer_next_chunk_frames(self, actions):
    # actions: [B, chunk, action_dim]
    # 1) current_obs[-1] -> [0,1] pixel_values
    # 2) tokenize + generate(action=...)
    # 3) detokenize -> pred_frames [B, chunk, 3, H, W] in [0,1]
    # 4) to [-1,1] and append to self.current_obs
    ...
```

> 你离线推理脚本默认 `segment_length=16` 是为了可视化长序列；RL 环境里建议设 `segment_length = condition_frame_length + chunk`，保证与策略 action chunk 一致。

### 4.5 `_infer_next_chunk_rewards`：奖励计算

保持与现有世界模型环境一致：

1. 取 `self.current_obs` 最新 `chunk` 帧（`[-1,1]`）
2. 映射到 `[0,1]`
3. reshape 成 `[B*chunk, 3, H, W]`
4. 调奖励模型打分
5. reshape 回 `[B, chunk]`

### 4.6 `_wrap_obs`：观测打包

返回格式必须与 Wan/OpenSora 完全一致：

```python
obs = {
    "main_images": uint8 [B,H,W,3],  # 由 current_obs 最新帧 [-1,1] 转回 [0,255]
    "wrist_images": None,
    "states": float32 zeros [B, action_dim],
    "task_descriptions": list[str],
}
```

### 4.7 `chunk_step`：驱动单步推进

标准流程：

1. `_infer_next_chunk_frames(policy_output_action)`
2. `chunk_rewards = _infer_next_chunk_rewards()`
3. `chunk_rewards = _calc_step_reward(chunk_rewards)`（若用相对奖励）
4. 仅在最后一帧标记 `termination/truncation`
5. 返回标准 5-tuple

### 4.8 `offload` / `onload`：显存管理

最小实现建议：

- `offload`: tokenizer / transformer / reward model -> CPU，再 `torch.cuda.empty_cache()`
- `onload`: 全部迁回 `self.device`

对 iVideoGPT 来说，`tokenizer + transformer + reward` 并存显存压力较大，建议默认支持 `enable_offload`。

---

## Step 3：动作适配（prepare_actions）

**文件：** `rlinf/envs/action_utils.py`

新增 `ivideogpt_wm` 分支，RobotWin 先透传即可：

```python
elif env_type == SupportedEnvType.IVIDEOGPTWM:
    if wm_env_type == "robotwin":
        chunk_actions = raw_chunk_actions
    else:
        raise NotImplementedError(...)
```

建议原则：

- `prepare_actions()` 只做任务语义适配（例如 gripper 符号）
- iVideoGPT 特定的数据整理（context/chunk 对齐）放在 `IVideoGPTEnv._infer_next_chunk_frames()`

---

## Step 4：环境配置 YAML（iVideoGPT 版）

**建议文件：** `examples/embodiment/config/env/ivideogpt_robotwin_click_bell.yaml`

示例模板（可直接改路径）：

```yaml
env_type: ivideogpt_wm
task_suite_name: robotwin_click_bell
wm_env_type: robotwin

total_num_envs: null
auto_reset: False
ignore_terminations: False
max_steps_per_rollout_epoch: 200
max_episode_steps: 200

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

# ===== iVideoGPT world model =====
chunk: 8
condition_frame_length: 1
segment_length: 9        # = condition_frame_length + chunk
image_size: 256
action_dim: 14
action_key: action

# reset 所需数据（编码后的 npz 目录）
initial_image_path: /path/to/encoded_data/click_bell
dataset_name: click_bell

# 推理模型 bundle（目录下需有 tokenizer/ 与 transformer/）
model_bundle_path: /path/to/ivideogpt_click_bell_bundle

# 奖励模型（示例）
reward_model:
  type: RoboTwinT5CrossAttn
  from_pretrained: /path/to/reward_model.pth
  t5_model_name: /path/to/t5-base
```

---

## Step 5：训练配置 YAML（iVideoGPT 版）

**建议文件：** `examples/embodiment/config/ivideogpt_robotwin_click_bell_grpo_openpi_pi05.yaml`

关键一致性（最重要）：

1. `env.train.chunk`
2. `actor.model.num_action_chunks`
3. `actor.model.openpi.action_chunk`

三者必须一致；`action_dim` 同理三处一致。

示例（只保留关键项）：

```yaml
defaults:
  - env/ivideogpt_robotwin_click_bell@env.train
  - env/robotwin_click_bell@env.eval
  - model/pi0_5@actor.model

algorithm:
  adv_type: grpo
  loss_type: actor
  group_size: 4
  reward_coef: 5.0

env:
  train:
    total_num_envs: 4
    group_size: ${algorithm.group_size}
    chunk: 8
    action_dim: 14
    reward_coef: ${algorithm.reward_coef}
  eval:
    total_num_envs: 4
    group_size: 1

actor:
  model:
    num_action_chunks: 8
    action_dim: 14
    openpi:
      action_chunk: 8
      action_env_dim: 14
```

---

## 关键一致性检查清单（iVideoGPT 版）

提交前逐项确认：

- `env.chunk == actor.model.num_action_chunks == actor.model.openpi.action_chunk`
- `env.action_dim == actor.model.action_dim == actor.model.openpi.action_env_dim == 14`
- `segment_length == condition_frame_length + chunk`
- `condition_frame_length == tokenizer.context_length`（推荐保持一致）
- `model_bundle_path/tokenizer` 与 `model_bundle_path/transformer/model.safetensors` 存在
- 图像链路一致：`uint8 -> [-1,1] -> uint8(policy) / [0,1](reward)`
- `total_num_envs % group_size == 0`
- `total_num_envs % env_world_size == 0`

---

## 与 Wan / OpenSora 的差异点

| 维度 | Wan / OpenSora | iVideoGPT |
|------|----------------|-----------|
| 生成范式 | diffusion 采样 | token 自回归生成 |
| 条件输入 | 条件图像/latent + action | context frame token + action token conditioning |
| 关键接口 | `pipe(...)` / `scheduler.sample(...)` | `tokenize -> model.generate(action=...) -> detokenize` |
| 默认 context | Wan=5 / OpenSora=4 | 常用 `1` |
| chunk 建议 | 多为 `8` | 建议仍用 `8`（与 Pi05 对齐） |
| 离线脚本 `segment_length` | 常固定 | 可为可视化设大值；RL 建议 `cond + chunk` |

---

## 最小落地顺序（建议）

1. 先加 `ivideogpt_wm` 枚举 + `IVideoGPTEnv` 空壳（确保配置可加载）
2. 实现 `reset()` + `_wrap_obs()`，验证观测 shape/range
3. 实现 `_infer_next_chunk_frames()`，单 env 验证能更新 `current_obs`
4. 实现 `_infer_next_chunk_rewards()` + `chunk_step()`
5. `total_num_envs=1` 做 smoke test
6. 再放开到 `group_size>1` 的 GRPO

---

如果你希望，我可以下一步继续给你一份“**逐文件修改清单 + 代码骨架补丁**”（直接对应 `rlinf/envs/__init__.py`、`world_model_ivideogpt_env.py`、`action_utils.py`、两份 yaml），你可以直接按清单开工或我直接帮你改好。
