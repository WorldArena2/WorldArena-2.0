# IRASim 作为 RL 环境的集成指南（对照版）

本文档是 `docs/world_model_rl_integration.md` 的 **IRASim（RobotWin adjust_bottle / click_bell）专项落地版本**。

目标是：在你已经完成 IRASim 的 **数据编码 / SFT 训练 / 离线推理与可视化** 的前提下，把 IRASim 接入 RLinf，作为世界模型环境参与 GRPO / PPO 训练。

你当前已经可用的两个 checkpoint 为：

- `/your/path/lizhuohang/IRASim/results/vla_robotwin_4k_320_from_scratch/adjust_bottle/checkpoints/0102000.pt`
- `/your/path/lizhuohang/IRASim/results/vla_robotwin_4k_320_from_scratch/click_bell/checkpoints/0102000.pt`

IRASim 对应的关键事实如下：

- 训练配置来自 `configs/train/robotwin2/frame_ada_adjust_bottle_4k320.yaml` 和 `configs/train/robotwin2/frame_ada_click_bell_4k320.yaml`
- 推理入口是 `scripts/infer_robotwin2_val.py`
- 任务级推理脚本是 `scripts/infer_vla4k320_adjust_bottle_8gpu.sh` 和 `scripts/infer_vla4k320_click_bell_8gpu.sh`
- 数据集加载器是 `dataset/dataset_robotwin2.py`
- 世界模型采样接口是 `sample/pipeline_trajectory2videogen.py`

---

## 目录

1. [整体架构（IRASim 对照）](#1-整体架构irasim-对照)
2. [核心数据流（IRASim 对照）](#2-核心数据流irasim-对照)
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
7. [Step 4：环境配置 YAML（IRASim 版）](#step-4环境配置-yamlirasim-版)
8. [Step 5：训练配置 YAML（IRASim 版）](#step-5训练配置-yamlirasim-版)
9. [关键一致性检查清单（IRASim 版）](#9-关键一致性检查清单irasim-版)
10. [可直接 import 的 IRASim 接口清单](#10-可直接-import-的-irasim-接口清单)
11. [最小可运行落地步骤](#11-最小可运行落地步骤)
12. [执行命令示例](#12-执行命令示例)

---

## 1. 整体架构（IRASim 对照）

IRASim 接入 RLinf 的方式与 Wan / OpenSora / Cosmos / iVideoGPT 一致：世界模型只在 `EnvWorker` 内部实现，`Runner` / `RolloutWorker` / `ActorWorker` 主流程不需要改。

```text
run_embodiment.sh
  -> Runner
    -> RolloutWorker (Policy 输出 chunk actions)
      -> prepare_actions()
      -> EnvWorker(IRASimEnv).chunk_step()
         -> _infer_next_chunk_frames(actions)
         -> _infer_next_chunk_rewards()
         -> _wrap_obs()
    -> ActorWorker (GRPO/PPO 更新)
```

但 IRASim 与 Wan / OpenSora 最大的不同是：

1. **IRASim 只用 1 帧条件帧**
   - `mask_frame_num = 1`
2. **IRASim 单次 rollout 总长度是 16 帧**
   - `num_frames = 16`
3. **因此 RL 环境每一步真正新增的是 15 帧**
   - `chunk = num_frames - condition_frame_length = 15`

也就是说，IRASim 接入 RL 后最关键的配置对齐关系是：

```text
condition_frame_length = 1
chunk = 15
num_frames = 16
policy.action_chunk = 15
```

如果 Pi05 仍然保持 Wan / OpenSora 的 `action_chunk = 8`，IRASim 环境就无法直接无损对齐当前 checkpoint 的训练方式。

---

## 2. 核心数据流（IRASim 对照）

IRASim 的实际数据流应理解为：

```text
reset() 读入初始图像 uint8 [0,255]
  -> current_obs: float32 [-1,1], [B,3,1,T,H,W]
  -> current_latent: float32, [B,1,4,H/8,W/8]

policy 输出动作: [B,15,14]
  -> RobotWin 绝对动作 / 绝对状态空间
  -> 环境内按 stat.json 做 p01/p99 -> [-1,1] 归一化

IRASim pipeline(
    actions_norm,
    mask_x=current_latent,
    video_length=16,
    height=240,
    width=320,
)
  -> pred_videos:  [B,16,3,H,W]，值域约为 [-1,1]
  -> pred_latents: [B,16,4,H/8,W/8]

丢掉第 0 帧条件帧
  -> new_frames = pred_videos[:, 1:]   # [B,15,3,H,W]

更新内部缓存
  -> self.current_obs 追加 15 帧
  -> self.current_latent = pred_latents[:, -1:]

奖励模型
  -> latest chunk frames: [B,15,3,H,W]
  -> [0,1] 后 reshape 为 [B*15,3,H,W]
  -> 输出 chunk_rewards: [B,15]
```

和离线推理脚本 `scripts/infer_robotwin2_val.py` 相比，RL 环境里建议做一个工程优化：

- 离线脚本每轮会把最新一帧重新 VAE encode 回 latent
- RL 环境里更推荐直接缓存 `pred_latents[:, -1:]`

这样可以减少高频 rollout 时的 VAE 编码开销。

---

## Step 0：先决条件（编码、训练、离线推理）

### 0.1 数据编码 / 数据整理

你现在的 RobotWin 4k320 数据准备链路，对应仓库中的：

- `scripts/prepare_vla_robotwin_subset.py`
- 输出目录：`robotdata/vla_robotwin_4k_320_irasim/<task>`
- 元数据：`data/vla_robotwin_4k_320_<task>_{train,val}_meta.json`
- 统计信息：`robotdata/vla_robotwin_4k_320_irasim/<task>/stat.json`

这一步对 RL 集成的意义是：

- `reset()` 需要依赖这些 `meta.json` 找到 episode
- `_infer_next_chunk_frames()` 需要依赖 `stat.json` 做动作归一化

### 0.2 模型训练 / SFT

你当前的 task-specific 训练配置是：

- `configs/train/robotwin2/frame_ada_click_bell_4k320.yaml`
- `configs/train/robotwin2/frame_ada_adjust_bottle_4k320.yaml`

这两个配置都说明了当前 checkpoint 的关键约束：

- `dataset: robotwin2`
- `num_frames: 16`
- `mask_frame_num: 1`
- `video_size: [240, 320]`
- `normalize: True`
- `pre_encode: False`
- `action_dim = 14`

### 0.3 离线推理 / 可视化

在接入 RL 前，必须先验证离线推理已跑通：

- `scripts/infer_robotwin2_val.py`
- `scripts/infer_vla4k320_click_bell_8gpu.sh`
- `scripts/infer_vla4k320_adjust_bottle_8gpu.sh`

只要离线推理能正确生成 `*_gen.mp4` / `*_gtgen.mp4`，RL 集成的核心工作就只是把 `infer_robotwin2_val.py` 的单次 autoregressive 推理逻辑迁移到环境的 `_infer_next_chunk_frames()`。

---

## Step 1：注册新环境类型

**文件：** `rlinf/envs/__init__.py`

### 1.1 当前状态

你当前的 RLinf 里：

- `SupportedEnvType.IRASIMWM = "irasim_wm"` **已经存在**
- 但 `get_env_cls()` **还没有** `IRASIMWM` 的分支
- 也还没有 `world_model_irasim_env.py`

也就是说，第一步不是再加枚举，而是把工厂分支补完整。

### 1.2 增加工厂分支

```python
elif env_type == SupportedEnvType.IRASIMWM:
    from rlinf.envs.world_model.world_model_irasim_env import IRASimEnv
    return IRASimEnv
```

### 1.3 为什么这里必须 lazy import

原因与 Wan / OpenSora 相同：

- IRASim 依赖 `diffusers`
- 依赖 VAE / scheduler / transformer
- 初始化较重，不应在顶层强制导入

---

## Step 2：实现 BaseWorldEnv 子类

**建议文件：** `rlinf/envs/world_model/world_model_irasim_env.py`

参考实现：

- `rlinf/envs/world_model/world_model_wan_env.py`
- `rlinf/envs/world_model/world_model_opensora_env.py`
- `rlinf/envs/world_model/world_model_cosmos_env.py`
- `rlinf/envs/world_model/world_model_ivideogpt_env.py`

IRASim 版环境的核心职责是：

1. `reset()` 读数据集首帧
2. 将首帧编码为 `condition latent`
3. 接收策略输出的 `15` 步动作
4. 调用 IRASim 生成未来 `16` 帧
5. 丢掉条件帧，仅保留新增 `15` 帧
6. 奖励模型打分并返回标准 RLinf 观测

### 4.1 `__init__` 构造函数

IRASim 版建议初始化以下字段：

| 字段 | 含义 | RobotWin 4k320 推荐值 |
|------|------|-----------------------|
| `self.chunk` | 每次环境步新增帧数 | `15` |
| `self.condition_frame_length` | 条件帧长度 | `1` |
| `self.num_frames` | 单次采样总帧数 | `16` |
| `self.image_size` | 生成分辨率 | `[240, 320]` |
| `self.action_dim` | RobotWin 动作维度 | `14` |
| `self.num_inference_steps` | diffusion 采样步数 | `50` |
| `self.guidance_scale` | 推理 guidance | `1.0` |
| `self.current_obs` | 内部帧缓存 | `None` |
| `self.current_latent` | 当前条件 latent | `None` |
| `self.task_descriptions` | 任务文本 | 长度为 `num_envs` 的 list |
| `self.action_p01/p99` | 动作统计量 | 来自 task-specific `stat.json` |

#### 4.1.1 世界模型加载方式

这部分应直接迁移 `scripts/infer_robotwin2_val.py` 的加载逻辑：

1. `OmegaConf.load(wm_config_path)`
2. 构造 `model_args`
3. `get_models(model_args)` 创建 `IRASim-XL/2`
4. 加载 checkpoint：优先 `checkpoint["ema"]`，其次 `checkpoint["model"]`，否则直接当 state dict
5. `AutoencoderKL.from_pretrained(vae_model_path, subfolder="vae")`
6. `PNDMScheduler(...)`
7. `Trajectory2VideoGenPipeline(vae, scheduler, transformer)`

建议模型成员：

```python
self.irasim_model
self.vae
self.scheduler
self.pipeline
```

#### 4.1.2 奖励模型

IRASim 本身不带 reward head，因此这一层继续复用 RLinf 现有奖励模型：

- `RoboTwinT5CrossAttn`
- 或 `QwenVLMProgressRewardModel`

建议优先和 Wan / OpenSora RobotWin 保持一致，先复用现成 reward 配置。

#### 4.1.3 一个非常重要的对齐点

Pi05 训练配置里必须同步改成：

```text
num_action_chunks = 15
openpi.action_chunk = 15
```

因为当前 IRASim checkpoint 的 rollout 设计不是 `8 -> 8`，而是：

```text
1 个条件帧 + 15 个动作 = 16 帧总序列
```

### 4.2 `_build_dataset`：数据集加载

这里不建议直接复用训练态的 `Dataset_RoboTwin2` 滑窗输出作为 `reset()` 数据源。

原因是 RL `reset()` 的需求与训练 dataset 不完全一样：

- RL `reset()` 需要 **按 episode 取整条轨迹**
- 训练 dataset 则是 **按滑窗取长度 16 的 sample**

因此更推荐做一个轻量 wrapper，逻辑直接参考 `scripts/infer_robotwin2_val.py` 的 `load_episode_data()`。

需要从每个 episode 中读到：

```python
{
    "episode_id": int,
    "video": np.ndarray,   # [T,H,W,3], uint8
    "states": np.ndarray,  # [T,14], float32
    "task_description": str,
}
```

最小依赖字段是：

- `meta.json` 中的 `file_path`
- `meta.json` 中的 `ann_file`
- annotation 中的 `state`
- （可选）meta 中的 `text`

如果 `text` 不存在，可以在 env config 中提供：

- `default_task_description: click the bell`
- `default_task_description: adjust the bottle`

### 4.3 `reset`：环境重置

IRASim 版 `reset()` 建议流程：

1. 根据 `episode_indices` 从 meta 中取对应 episode
2. 读取视频第 `0` 帧作为 reset 图像
3. resize 到 `image_size=[240,320]`
4. 转成 `float32 [-1,1]`
5. VAE encode 为首帧 latent
6. 初始化：
   - `self.current_obs`
   - `self.current_latent`
   - `self.task_descriptions`
   - `self.elapsed_steps`
7. 返回 `_wrap_obs()`

reset 后内部状态建议为：

```python
self.current_obs.shape == [B, 3, 1, 1, H, W]
self.current_latent.shape == [B, 1, 4, H/8, W/8]
```

推荐伪代码：

```python
frame0_uint8 -> resize -> tensor [B,3,H,W] in [0,1]
frame0 = frame0 * 2 - 1
self.current_obs = frame0[:, :, None, None, :, :]
self.current_latent = self._encode_frames_to_latent(frame0_uint8)
return self._wrap_obs()
```

### 4.4 `_infer_next_chunk_frames`：世界模型推理

这是 IRASim 集成的核心，对应迁移自 `scripts/infer_robotwin2_val.py` 中：

- `normalize_actions()`
- `generate()`
- autoregressive rollout 的单 chunk 推理逻辑

#### 输入动作

- 形状：`[B, 15, 14]`
- dtype：`float32`
- 语义：RobotWin 的绝对 joint / gripper 状态空间

#### 建议实现流程

1. 将 policy 输出动作转成 `numpy` / `torch`
2. 用 task 对应的 `stat.json` 做：

```python
actions = (actions - p01) / (p99 - p01 + 1e-8)
actions = clip(actions, 0, 1)
actions = actions * 2 - 1
```

3. 若 `self.current_latent is None`，则从 `self.current_obs` 最新帧重新 encode（兜底逻辑）
4. 调用：

```python
pred_videos, pred_latents = self.pipeline(
    actions_norm,
    mask_x=self.current_latent,
    video_length=self.num_frames,
    height=self.image_size[0],
    width=self.image_size[1],
    num_inference_steps=self.num_inference_steps,
    guidance_scale=self.guidance_scale,
    device=self.device,
    return_dict=False,
    output_type="both",
)
```

5. 丢掉条件帧：

```python
new_frames = pred_videos[:, self.condition_frame_length:]
new_latent = pred_latents[:, -1:]
```

6. 将 `new_frames` 从 `[B,T,C,H,W]` 转成 `BaseWorldEnv` 使用的 `[B,C,1,T,H,W]`
7. 追加到 `self.current_obs`
8. 更新 `self.current_latent = new_latent`

#### 推荐优化

与离线推理脚本不同，RL 环境里推荐：

- **不要**每一步都重新 encode 最后一帧
- 直接缓存 `pred_latents[:, -1:]` 作为下一次 `mask_x`

这样可以显著降低 rollout 开销。

### 4.5 `_infer_next_chunk_rewards`：奖励计算

保持与现有世界模型环境一致：

1. 取 `self.current_obs` 最新 `chunk=15` 帧
2. `[-1,1] -> [0,1]`
3. reshape 为 `[B*15, 3, H, W]`
4. 送入 reward model 打分
5. reshape 回 `[B,15]`

建议保持 action-level reward，与 Wan / OpenSora 的 RobotWin 训练配置一致。

### 4.6 `_wrap_obs`：观测打包

返回格式必须与现有 world-model env 保持一致：

```python
obs = {
    "main_images": uint8 [B,H,W,3],
    "wrist_images": None,
    "states": float32 zeros [B,14],
    "task_descriptions": list[str],
}
```

其中：

- `main_images` 来自 `self.current_obs` 最新帧
- `states` 可以先返回零张量，和 Wan / OpenSora 做法一致

### 4.7 `chunk_step`：驱动单步推进

标准流程建议保持不变：

1. `_infer_next_chunk_frames(policy_output_action)`
2. `chunk_rewards = _infer_next_chunk_rewards()`
3. `chunk_rewards = _calc_step_reward(chunk_rewards)`（若使用相对奖励）
4. 仅在最后一个 step 标记 `termination` / `truncation`
5. 返回标准 5-tuple

成功判定可以继续沿用现有世界模型环境的方式，例如：

- `success_reward_threshold: 0.9`

### 4.8 `offload` / `onload`：显存管理

IRASim + VAE + Reward + Pi05 同卡时显存压力不小，建议默认支持：

- `offload`: `irasim_model / vae / reward_model -> CPU`
- `onload`: 全部迁回 `self.device`

最小实现即可：

```python
def offload(self):
    self.irasim_model.cpu()
    self.vae.cpu()
    self.reward_model.cpu()
    torch.cuda.empty_cache()

def onload(self):
    self.irasim_model.to(self.device)
    self.vae.to(self.device)
    self.reward_model.to(self.device)
```

---

## Step 3：动作适配（prepare_actions）

**文件：** `rlinf/envs/action_utils.py`

当前 `prepare_actions()` 已经对以下 world model 做了 RobotWin 透传：

- `opensora_wm`
- `wan_wm`
- `ivideogpt_wm`
- `cosmos_wm`

但 **还没有** `irasim_wm`。

建议最小改动是把 IRASim 加入与 Wan / OpenSora / iVideoGPT 相同的透传分支：

```python
elif (
    env_type == SupportedEnvType.OPENSORAWM
    or env_type == SupportedEnvType.WANWM
    or env_type == SupportedEnvType.IVIDEOGPTWM
    or env_type == SupportedEnvType.IRASIMWM
):
    if wm_env_type == "robotwin":
        chunk_actions = raw_chunk_actions
    else:
        raise NotImplementedError(...)
```

推荐做法是：

- `prepare_actions()` 只做任务语义层的适配
- `stat.json` 的 `p01/p99 -> [-1,1]` 归一化放在 `IRASimEnv._infer_next_chunk_frames()` 内

原因很简单：

- 归一化是 **IRASim 模型空间** 的要求
- 不是 policy 空间的要求

---

## Step 4：环境配置 YAML（IRASim 版）

**建议文件：** `examples/embodiment/config/env/irasim_robotwin_click_bell.yaml`

下面给出一个可直接照抄的模板：

```yaml
env_type: irasim_wm
task_suite_name: robotwin_click_bell
wm_env_type: robotwin

total_num_envs: null

auto_reset: False
ignore_terminations: False
max_steps_per_rollout_epoch: 225
max_episode_steps: 225

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
print_chunk_rewards: True
success_reward_threshold: 0.9

# ===== IRASim world model =====
chunk: 15
condition_frame_length: 1
num_frames: 16
image_size: [240, 320]
action_dim: 14
action_key: abs_action

irasim_root: /your/path/lizhuohang/IRASim
task_name: click_bell
default_task_description: click the bell

wm_ckpt_path: /your/path/lizhuohang/IRASim/results/vla_robotwin_4k_320_from_scratch/click_bell/checkpoints/0102000.pt
wm_config_path: /your/path/lizhuohang/IRASim/configs/train/robotwin2/frame_ada_click_bell_4k320.yaml
vae_model_path: /your/path/lizhuohang/IRASim/pretrained_models/stabilityai/stable-diffusion-xl-base-1.0

num_inference_steps: 50
guidance_scale: 1.0

train_data_dir: /your/path/lizhuohang/IRASim/robotdata/vla_robotwin_4k_320_irasim/click_bell
train_data_meta: /your/path/lizhuohang/IRASim/data/vla_robotwin_4k_320_click_bell_train_meta.json
val_data_meta: /your/path/lizhuohang/IRASim/data/vla_robotwin_4k_320_click_bell_val_meta.json
data_mode: val
action_stat_path: /your/path/lizhuohang/IRASim/robotdata/vla_robotwin_4k_320_irasim/click_bell/stat.json

enable_latent_cache: True

reward_model:
  type: RoboTwinT5CrossAttn
  from_pretrained: /manifold-obs/tangyinzhou/RLinf/diffsynth-studio/RLinf-Wan-RobotWin-ClickBell/resnet_rm.pth
  t5_model_name: /your/path//RLinf/pretrained_models/t5-base
```

### 4.1 `adjust_bottle` 版本怎么改

复制一份为：

- `examples/embodiment/config/env/irasim_robotwin_adjust_bottle.yaml`

只改这些 task-specific 字段：

- `task_suite_name: robotwin_adjust_bottle`
- `task_name: adjust_bottle`
- `default_task_description: adjust the bottle`
- `wm_ckpt_path`
- `wm_config_path`
- `train_data_dir`
- `train_data_meta`
- `val_data_meta`
- `action_stat_path`

其余字段原则上保持不变。

---

## Step 5：训练配置 YAML（IRASim 版）

**建议文件：** `examples/embodiment/config/irasim_robotwin_click_bell_grpo_openpi_pi05.yaml`

可以直接参考：

- `examples/embodiment/config/wan_robotwin_click_bell_grpo_openpi_pi05.yaml`
- `examples/embodiment/config/opensora_robotwin_click_bell_grpo_pi05.yaml`

IRASim 版最关键的改动只有两点：

1. `env.train` 默认改成 `env/irasim_robotwin_click_bell`
2. `Pi05 action_chunk` 改成 `15`

模板如下：

```yaml
defaults:
  - env/irasim_robotwin_click_bell@env.train
  - env/robotwin_click_bell@env.eval
  - model/pi0_5@actor.model
  - training_backend/fsdp@actor.fsdp_config
  - override hydra/job_logging: stdout

hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
    - file://${oc.env:EMBODIED_PATH}/config/

cluster:
  num_nodes: 1
  component_placement:
    actor,env,rollout: 0

runner:
  task_type: embodied
  logger:
    log_path: "../logs"
    project_name: rlinf
    experiment_name: "irasim_robotwin_click_bell_grpo_openpi_pi05"
    logger_backends: ["tensorboard"]

  max_epochs: 1000
  max_steps: -1
  only_eval: False
  val_check_interval: -1
  save_interval: 3

algorithm:
  normalize_advantages: True
  kl_penalty: kl
  group_size: 4
  rollout_epoch: 8
  eval_rollout_epoch: 1
  reward_type: action_level
  logprob_type: token_level
  entropy_type: token_level

  adv_type: grpo
  loss_type: actor
  loss_agg_func: token-mean
  kl_beta: 0.0
  entropy_bonus: 0.0

  gamma: 0.99
  gae_lambda: 0.95
  reward_coef: 5.0

  sampling_params:
    do_sample: True
    temperature_train: 1.6
    temperature_eval: 1.6
    top_k: -1
    top_p: 1.0
    repetition_penalty: 1.0

  length_params:
    max_new_token: null
    max_length: 1024
    min_length: 1

env:
  group_name: EnvGroup
  train:
    total_num_envs: 4
    group_size: ${algorithm.group_size}
    max_episode_steps: 225
    max_steps_per_rollout_epoch: 225
    reward_coef: ${algorithm.reward_coef}
    enable_offload: False
  eval:
    total_num_envs: 4
    auto_reset: True
    ignore_terminations: True
    max_episode_steps: 200
    max_steps_per_rollout_epoch: 200
    group_size: 1
    is_eval: True
    video_cfg:
      save_video: True
      video_base_dir: ${runner.logger.log_path}/video/eval

rollout:
  group_name: RolloutGroup
  generation_backend: huggingface
  enable_offload: False
  pipeline_stage_num: 1
  model:
    model_path: /manifold-obs/tangyinzhou/RLinf/logs/pi05-clickbell/pi05-clickbell/sft_openpi_click_bell_headcam/checkpoints/global_step_4000/actor
    precision: ${actor.model.precision}

actor:
  group_name: ActorGroup
  training_backend: fsdp
  micro_batch_size: 8
  global_batch_size: 800
  seed: 1234
  enable_offload: False

  model:
    model_path: /manifold-obs/tangyinzhou/RLinf/logs/pi05-clickbell/pi05-clickbell/sft_openpi_click_bell_headcam/checkpoints/global_step_4000/actor
    num_action_chunks: 15
    action_dim: 14
    add_value_head: True
    num_steps: 5
    openpi:
      config_name: pi05_aloha_robotwin_head
      num_images_in_input: 1
      noise_level: 0.3
      detach_critic_input: True
      action_chunk: 15
      action_env_dim: 14

  optim:
    lr: 2.0e-5
    value_lr: 3.0e-3
    adam_beta1: 0.9
    adam_beta2: 0.999
    adam_eps: 1.0e-05
    weight_decay: 0.01
    clip_grad: 1.0

  fsdp_config:
    strategy: fsdp
    gradient_checkpointing: True
    mixed_precision:
      param_dtype: ${actor.model.precision}
      reduce_dtype: ${actor.model.precision}
      buffer_dtype: ${actor.model.precision}

reward:
  use_reward_model: False

critic:
  use_critic_model: False
```

### 5.1 `adjust_bottle` 训练配置怎么改

复制为：

- `examples/embodiment/config/irasim_robotwin_adjust_bottle_grpo_openpi_pi05.yaml`

并替换：

- `defaults` 中的 env 文件
- `runner.logger.experiment_name`
- `rollout.model.model_path` / `actor.model.model_path`（若使用不同的 Pi05 SFT）
- `env.eval` 对应 task 的真实仿真器配置（如果你后续还要做 eval）

---

## 9. 关键一致性检查清单（IRASim 版）

接 IRASim 时，最容易出错的是下面这些对齐关系：

1. **策略 chunk 必须等于环境 chunk**
   - `actor.model.num_action_chunks == 15`
   - `actor.model.openpi.action_chunk == 15`
   - `env.train.chunk == 15`

2. **IRASim 序列长度必须满足**
   - `num_frames = condition_frame_length + chunk = 16`

3. **动作维度必须保持 RobotWin 14 维**
   - `action_dim = 14`

4. **动作归一化必须复用 task 对应的 `stat.json`**
   - `click_bell` 不能误用 `adjust_bottle` 的统计量

5. **reset 数据和 world model checkpoint 必须属于同一任务**
   - `click_bell` 模型配 `click_bell` meta / stat / task_description`

6. **离线推理要先验证通过，再启动 RL**
   - 优先确认 `scripts/infer_robotwin2_val.py` 对应 checkpoint 能正常出图

7. **当前 RLinf 只完成了枚举，没完成工厂与动作分支**
   - `SupportedEnvType.IRASIMWM` 已存在
   - `get_env_cls()` 还缺 `IRASIMWM`
   - `prepare_actions()` 还缺 `IRASIMWM`

8. **`run_embodiment.sh` 需要能 import IRASim**
   - 当前脚本显式加入了 `diffsynth-studio`
   - IRASim 也需要加入 `PYTHONPATH`

---

## 10. 可直接 import 的 IRASim 接口清单

建议在 `world_model_irasim_env.py` 中直接复用以下接口：

```python
from omegaconf import OmegaConf
from diffusers.models import AutoencoderKL
from diffusers.schedulers import PNDMScheduler

from models import get_models
from sample.pipeline_trajectory2videogen import Trajectory2VideoGenPipeline
```

如果你不想直接 import `scripts/infer_robotwin2_val.py`，建议把下列逻辑就地复制成 env 内部私有方法：

- checkpoint 加载逻辑
- `normalize_actions()`
- `encode_frames()`
- `load_episode_data()`

这样最稳，也最容易控制依赖边界。

---

## 11. 最小可运行落地步骤

建议按下面顺序接入：

1. **先验证离线推理**
   - 对 `click_bell` 和 `adjust_bottle` 的 `0102000.pt` 分别跑一次 `infer_robotwin2_val.py`
2. **让 RLinf 能 import IRASim**
   - 在 `examples/embodiment/run_embodiment.sh` 中补 `IRASIM_ROOT` 到 `PYTHONPATH`
3. **补环境注册**
   - `rlinf/envs/__init__.py` 加 `IRASIMWM -> IRASIMEnv`
4. **补动作分支**
   - `rlinf/envs/action_utils.py` 增加 `IRASIMWM`
5. **实现 `world_model_irasim_env.py`**
   - 重点完成 `_build_dataset`、`reset`、`_infer_next_chunk_frames`
6. **新增 env yaml**
   - `env/irasim_robotwin_click_bell.yaml`
   - `env/irasim_robotwin_adjust_bottle.yaml`
7. **新增训练 yaml**
   - `irasim_robotwin_click_bell_grpo_openpi_pi05.yaml`
   - `irasim_robotwin_adjust_bottle_grpo_openpi_pi05.yaml`
8. **启动 RL**
   - `bash examples/embodiment/run_embodiment.sh irasim_robotwin_click_bell_grpo_openpi_pi05`
   - `bash examples/embodiment/run_embodiment.sh irasim_robotwin_adjust_bottle_grpo_openpi_pi05`

---

## 12. 执行命令示例

### 12.1 先验证 click_bell 的离线推理

```bash
cd /your/path/lizhuohang/IRASim

python scripts/infer_robotwin2_val.py \
  --ckpt_path /your/path/lizhuohang/IRASim/results/vla_robotwin_4k_320_from_scratch/click_bell/checkpoints/0102000.pt \
  --config_path /your/path/lizhuohang/IRASim/configs/train/robotwin2/frame_ada_click_bell_4k320.yaml \
  --data_dir /your/path/lizhuohang/IRASim/robotdata/vla_robotwin_4k_320_irasim/click_bell \
  --val_meta /your/path/lizhuohang/IRASim/data/vla_robotwin_4k_320_click_bell_val_meta.json \
  --stat_path /your/path/lizhuohang/IRASim/robotdata/vla_robotwin_4k_320_irasim/click_bell/stat.json \
  --vae_path /your/path/lizhuohang/IRASim/pretrained_models/stabilityai/stable-diffusion-xl-base-1.0 \
  --output_dir /your/path/lizhuohang/IRASim/results/vla_robotwin_4k_320_from_scratch/click_bell_infer_debug \
  --num_inference_steps 50 \
  --num_frames 16 \
  --video_size 240 320 \
  --save_size 480 640 \
  --save_fps 24 \
  --target_total_frames 121 \
  --max_episodes 8 \
  --device cuda
```

### 12.2 再验证 adjust_bottle 的离线推理

```bash
cd /your/path/lizhuohang/IRASim

python scripts/infer_robotwin2_val.py \
  --ckpt_path /your/path/lizhuohang/IRASim/results/vla_robotwin_4k_320_from_scratch/adjust_bottle/checkpoints/0102000.pt \
  --config_path /your/path/lizhuohang/IRASim/configs/train/robotwin2/frame_ada_adjust_bottle_4k320.yaml \
  --data_dir /your/path/lizhuohang/IRASim/robotdata/vla_robotwin_4k_320_irasim/adjust_bottle \
  --val_meta /your/path/lizhuohang/IRASim/data/vla_robotwin_4k_320_adjust_bottle_val_meta.json \
  --stat_path /your/path/lizhuohang/IRASim/robotdata/vla_robotwin_4k_320_irasim/adjust_bottle/stat.json \
  --vae_path /your/path/lizhuohang/IRASim/pretrained_models/stabilityai/stable-diffusion-xl-base-1.0 \
  --output_dir /your/path/lizhuohang/IRASim/results/vla_robotwin_4k_320_from_scratch/adjust_bottle_infer_debug \
  --num_inference_steps 50 \
  --num_frames 16 \
  --video_size 240 320 \
  --save_size 480 640 \
  --save_fps 24 \
  --target_total_frames 121 \
  --max_episodes 8 \
  --device cuda
```

### 12.3 让 `run_embodiment.sh` 能 import IRASim

在 `examples/embodiment/run_embodiment.sh` 中，建议补上：

```bash
export IRASIM_ROOT=/your/path/lizhuohang/IRASim
export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${DIFFSYNTH_PATH}:${IRASIM_ROOT}:$PYTHONPATH"
```

### 12.4 启动 click_bell 的 RL 训练

```bash
cd /your/path//RLinf
bash examples/embodiment/run_embodiment.sh irasim_robotwin_click_bell_grpo_openpi_pi05
```

### 12.5 启动 adjust_bottle 的 RL 训练

```bash
cd /your/path//RLinf
bash examples/embodiment/run_embodiment.sh irasim_robotwin_adjust_bottle_grpo_openpi_pi05
```

---

## 最后总结

把 `world_model_rl_integration.md` 映射到 IRASim 后，最核心的结论只有三条：

1. **环境接口层面**：IRASim 仍然是标准 `BaseWorldEnv` 子类实现问题。
2. **模型接口层面**：核心就是迁移 `scripts/infer_robotwin2_val.py` 的单 chunk 推理逻辑。
3. **训练配置层面**：最关键不是路径，而是把 `Pi05 action_chunk` 从 `8` 改成 `15`，与 `IRASim num_frames=16, mask_frame_num=1` 对齐。

只要这三点对齐，IRASim 作为 RL 环境的接入方式，与 Wan / OpenSora / Cosmos / iVideoGPT 本质上是同一套范式。
