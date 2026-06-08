# 世界模型作为 RL 环境的集成指南

本文档从**通用视角**说明如何将一个已有推理脚本的视频生成世界模型（World Model）接入 RLinf，作为强化学习（RL）的仿真环境。文中以 **Wan（DiffSynth）** 和 **OpenSora（STDiT3）** 两个已实现的模型为参考，重点说明每个接口的数据格式与类型，以便未来接入新模型时参照执行。

---

## 目录

1. [整体架构](#1-整体架构)
2. [核心数据流](#2-核心数据流)
3. [Step 1：注册新环境类型](#step-1注册新环境类型)
4. [Step 2：实现 BaseWorldEnv 子类](#step-2实现-baseworldenv-子类)
   - [4.1 `__init__` 构造函数](#41-__init__-构造函数)
   - [4.2 `_build_dataset`：数据集加载](#42-_build_dataset数据集加载)
   - [4.3 `reset`：环境重置](#43-reset环境重置)
   - [4.4 `_infer_next_chunk_frames`：世界模型推理（核心接口）](#44-_infer_next_chunk_frames世界模型推理核心接口)
   - [4.5 `_infer_next_chunk_rewards`：奖励计算](#45-_infer_next_chunk_rewards奖励计算)
   - [4.6 `_wrap_obs`：观测打包（策略输入接口）](#46-_wrap_obs观测打包策略输入接口)
   - [4.7 `chunk_step`：驱动单步推进](#47-chunk_step驱动单步推进)
   - [4.8 `offload` / `onload`：显存管理](#48-offload--onload显存管理)
5. [Step 3：动作适配（prepare_actions）](#step-3动作适配prepare_actions)
6. [Step 4：编写环境配置 YAML](#step-4编写环境配置-yaml)
7. [Step 5：编写训练配置 YAML](#step-5编写训练配置-yaml)
8. [关键约束与一致性检查清单](#关键约束与一致性检查清单)
9. [图像归一化链路总览](#图像归一化链路总览)
10. [两个参考实现对比](#两个参考实现对比)

---

## 1. 整体架构

```
训练脚本（run_embodiment.sh）
    │
    ▼
Runner（embodied runner）
    │
    ├── RolloutWorker  ──────────────────────────────────────────┐
    │       │  predict_action_batch()                            │
    │       │  调用 Policy (e.g. Pi05) 生成 chunk actions        │
    │       │                                                    │
    │       ▼  prepare_actions()                                 │
    │   EnvWorker                                                │
    │       │                                                    │
    │       ├── env.reset(episode_indices)                       │
    │       │       └── 从数据集加载初始帧 → _wrap_obs() → obs   │
    │       │                                                    │
    │       └── env.chunk_step(chunk_actions)                    │
    │               ├── _infer_next_chunk_frames(actions)        │
    │               │       └── 调用世界模型生成视频帧            │
    │               ├── _infer_next_chunk_rewards()              │
    │               │       └── Reward 模型打分                  │
    │               └── _wrap_obs() → obs                       │
    │                                                            │
    └── ActorWorker  ◄───────────────────────────────────────────┘
            └── PPO/GRPO 更新策略参数
```

**关键原则：** 世界模型（视频生成模型）扮演"仿真器"角色，接收策略输出的动作序列，输出下一段视频帧作为新的观测；奖励模型对生成帧打分，作为 RL 信号。整个流程对 Policy 和 Runner 完全透明——新模型只需实现统一接口。

---

## 2. 核心数据流

```
数据集图像 (uint8 [0,255])
     │ 加载后归一化
     ▼
current_obs  (float32 [-1,1], shape: [B, C=3, V=1, T, H, W])
     │                          B=num_envs, V=num_views
     │ _wrap_obs() 反归一化
     ▼
Policy 输入 obs["main_images"]  (uint8 [0,255], shape: [B, H, W, 3])
     │ Policy 推理
     ▼
chunk_actions  (float32, 策略空间, shape: [B, chunk, action_dim])
     │ prepare_actions() 适配
     ▼
世界模型输入 actions  (float32, 模型空间, shape: [B, T_total, action_dim])
     │ 世界模型生成
     ▼
new_frames  (float32 [-1,1], shape: [B, C, T_chunk, H, W])
     │ 追加到 current_obs
     ▼
Reward 模型输入  (float32 [0,1], shape: [B*T, C, H, W])
     │ 奖励模型打分
     ▼
chunk_rewards  (float32, shape: [B, chunk])
```

---

## Step 1：注册新环境类型

**文件：** `rlinf/envs/__init__.py`

### 1.1 在枚举中添加新类型

```python
class SupportedEnvType(str, Enum):
    # ... 已有类型 ...
    WANWM      = "wan_wm"
    OPENSORAWM = "opensora_wm"
    MYNEWWM    = "my_new_wm"   # ← 新增，值为配置文件中使用的字符串
```

### 1.2 在工厂函数中注册

```python
def get_env_cls(env_type, env_cfg=None, ...):
    # ... 已有分支 ...
    elif env_type == SupportedEnvType.MYNEWWM:
        from rlinf.envs.world_model.world_model_mynew_env import MyNewEnv
        return MyNewEnv   # 返回类本身，不是实例
```

> **注意：** 使用延迟导入（lazy import），避免在顶层导入重量级依赖，防止其他用户不需要该模型时受到影响。

---

## Step 2：实现 BaseWorldEnv 子类

**文件位置：** `rlinf/envs/world_model/world_model_mynew_env.py`

**参考实现：**
- Wan：`rlinf/envs/world_model/world_model_wan_env.py`
- OpenSora：`rlinf/envs/world_model/world_model_opensora_env.py`

```python
from rlinf.envs.world_model.base_world_env import BaseWorldEnv

class MyNewEnv(BaseWorldEnv):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes,
                 record_metrics=True, worker_info=None):
        super().__init__(cfg, num_envs, seed_offset, total_num_processes,
                         worker_info, record_metrics)
        ...
```

### 4.1 `__init__` 构造函数

**必须从 cfg 读取并初始化以下属性：**

| 属性 | 类型 | 含义 | 典型值 |
|------|------|------|--------|
| `self.chunk` | `int` | 每次推理生成的帧数（= action chunk 长度）| `8` |
| `self.condition_frame_length` | `int` | 条件帧（历史帧）数量 | `5`（Wan）/ `4`（OpenSora）|
| `self.num_frames` | `int` | 模型单次处理总帧数 = condition + chunk | `13` |
| `self.image_size` | `tuple[int,int]` | `(H, W)`，输出图像尺寸 | `(256, 256)` |
| `self.action_dim` | `int` | 动作维度 | `14`（RobotWin）/ `7`（LIBERO）|
| `self.current_obs` | `Tensor \| None` | 内部帧缓冲，初始为 `None` | shape `[B,3,1,T,H,W]` |
| `self.task_descriptions` | `list[str]` | 每个环境的任务文字描述 | `["pick bell", ...]` |
| `self.image_queue` | `list` | 条件帧队列（每个 env 一个） | `deque` 或 `list` |

**必须加载的模型：**
- **世界模型主干**（e.g., pipeline / dit + vae + scheduler）
- **奖励模型**（reward model，用于对生成帧打分）

---

### 4.2 `_build_dataset`：数据集加载

```python
@abstractmethod
def _build_dataset(self, cfg):
    """返回用于 reset 的数据集包装器。"""
```

数据集需要提供初始帧供 `reset()` 使用。数据集中每个 episode 至少包含：
- `rgb` / `main_images`：图像帧，**uint8 [0,255]**，shape `[T, H, W, 3]` 或 `[T, 1, H, W, 3]`
- `actions`：动作序列，**float32**，shape `[T, action_dim]`
- `task_description`：任务描述字符串（可选）

---

### 4.3 `reset`：环境重置

```python
def reset(
    self,
    *,
    seed: Optional[Union[int, list[int]]] = None,
    options: Optional[dict] = {},
    episode_indices: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> dict:
```

**输入：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `episode_indices` | `np.ndarray \| torch.Tensor \| None` | 形状 `[B]`，指定每个环境使用数据集中的第几条轨迹；为 None 时随机采样 |

**内部处理流程：**
1. 根据 `episode_indices` 从数据集中加载对应轨迹的初始帧
2. 将 uint8 图像 `[0,255]` **归一化到 float32 `[-1,1]`** 存入 `self.current_obs`
3. 初始化 `self.image_queue`（每个 env 填充 `condition_frame_length` 帧）
4. 记录任务描述、初始 ee pose 等元信息
5. 调用 `_wrap_obs()` 返回标准观测字典

**返回值：** `dict`，格式见 [_wrap_obs](#46-_wrap_obs观测打包策略输入接口)

---

### 4.4 `_infer_next_chunk_frames`：世界模型推理（核心接口）

这是将推理脚本逻辑迁移到 RL 环境的核心方法。

```python
def _infer_next_chunk_frames(self, actions: np.ndarray | torch.Tensor) -> None:
    """
    调用世界模型，根据当前帧和动作序列预测下一段视频帧。
    结果写入 self.current_obs（in-place 更新）。
    """
```

**输入 `actions`：**

| 属性 | 值 |
|------|-----|
| 类型 | `np.ndarray` 或 `torch.Tensor`（函数内统一转换） |
| 形状 | `[B, chunk, action_dim]`，即 `[num_envs, 8, 14]` |
| 数值范围 | **策略输出的原始值**（通常为连续浮点数；OpenSora 会额外做 q01/q99 归一化） |
| dtype | `float32`（在函数内转为 `bfloat16` 传入模型） |

**Wan 模型调用示例（对应 `run_click_bell.sh` 中的 `pipe(...)` 调用）：**

```python
# 拼接历史动作（condition_action）与新动作
actions_tensor = torch.cat([self.condition_action, actions_tensor], dim=1)
# shape: [B, condition_frame_length + chunk, action_dim] = [B, 13, 14]

# 从 image_queue 取最后一帧作为 PIL 图像列表
input_image: list[PIL.Image]   # len=B，每帧 [H, W, 3] uint8

# 调用 DiffSynth WanVideoPipeline
output = self.pipe(
    input_image=input_image,         # 初始条件帧（PIL Image，uint8 [0,255]）
    input_image4=input_image4,       # 最近 4 帧（PIL Image 列表，len=4）
    action=actions_tensor,           # [B, T_total, action_dim]，bfloat16
    height=256, width=256,
    num_frames=self.num_frames + 4,  # 13 + 4
    num_inference_steps=self.num_inference_steps,
    bs_1=True,
)
# output shape: [B, T_chunk, H, W, 3]，像素值 [0,255] uint8
```

**OpenSora 模型调用示例（对应 `inference_libero.py` 中的 `scheduler.sample(...)` 调用）：**

```python
# 动作先做 q01/q99 归一化到 [-1, 1]
actions_normalized = 2 * ((actions - q01) / (q99 - q01)) - 1
y = torch.tensor(actions_normalized, device=device, dtype=dtype)
# shape: [B, chunk, action_dim] = [B, 8, 14]

# 条件帧（latent 空间）
mask_images = torch.concat(list(self.image_queue[env_idx]), dim=2)
# shape: [B, C_latent, T_cond_latent, H', W']

# 噪声
z = torch.randn(B, vae.out_channels, z_mask_frame_num, *latent_size[1:], ...)
z_full = torch.concat([mask_images, z], dim=2)   # [B, C, T_total_latent, H', W']
masks = [[0]*T_cond + [1]*T_mask] * B             # 0=已知, 1=待预测

samples = scheduler.sample(model, z=z_full, y=y, mask=masks, ...)
# samples shape: [B, C_latent, T_total_latent, H', W']

# 取预测部分并 VAE decode
pred_latents = samples[:, :, -z_mask_frame_num:, :, :]
pred_images = vae.decode(pred_latents)
# pred_images shape: [B, 3, T_chunk, H, W]，像素值 float32 [-1, 1]
```

**函数结束时必须更新 `self.current_obs`：**

```python
# 将新生成帧追加到历史观测
# self.current_obs shape: [B, 3, 1, T_old + T_chunk, H, W]
# 新帧统一归一化到 float32 [-1, 1] 后追加
new_frames_normalized = new_frames * 2.0 - 1.0   # 若输入为 [0,1]
# 或: new_frames_normalized = new_frames           # 若模型输出已是 [-1,1]
self.current_obs = torch.cat([self.current_obs, new_frames_normalized], dim=3)
```

---

### 4.5 `_infer_next_chunk_rewards`：奖励计算

```python
def _infer_next_chunk_rewards(self) -> torch.Tensor:
    """
    对最新生成的 chunk 帧逐帧调用奖励模型打分。
    
    Returns:
        chunk_rewards: float32 Tensor，shape [B, chunk]
                       每个值为该帧的奖励（通常在 [0, 1] 之间）
    """
```

**内部处理：**
1. 从 `self.current_obs` 中取最后 `chunk` 帧
2. **归一化变换**：`self.current_obs` 内部是 `[-1,1]`，奖励模型通常需要 `[0,1]` float32

```python
# 从 current_obs 取最新 chunk 帧
# self.current_obs: [B, 3, 1, T, H, W]
frames = self.current_obs[:, :, 0, -self.chunk:, :, :]   # [B, 3, chunk, H, W]

# 归一化到 [0,1]（针对 RoboTwinT5CrossAttn、ResnetRewModel 等）
frames_01 = (frames + 1.0) / 2.0    # [-1,1] → [0,1]

# 展平批次维度用于奖励模型
# [B, 3, chunk, H, W] → [B*chunk, 3, H, W]
frames_flat = frames_01.permute(0,2,1,3,4).reshape(-1, 3, H, W)

# 调用奖励模型
with torch.no_grad():
    rewards_flat = reward_model.compute_reward(
        frames_flat,                            # [B*chunk, 3, H, W]，float32 [0,1]
        task_descriptions=task_desc_repeated,   # list，len=B*chunk
    )   # → [B*chunk]

chunk_rewards = rewards_flat.reshape(B, self.chunk)   # [B, chunk]
```

**目前支持的奖励模型类型（`reward_model.type`）：**

| 类型 | 输入格式 | 说明 |
|------|----------|------|
| `ResnetRewModel` | `[N, 3, H, W]` float32 [0,1] | 纯视觉 ResNet 打分 |
| `TaskEmbedResnetRewModel` | `[N, 3, H, W]` float32 [0,1] + task embedding | 带任务嵌入的 ResNet |
| `RoboTwinT5CrossAttn` | `[N, 3, H, W]` float32 [0,1] + 文本描述 list | T5 编码任务描述后做 Cross-Attention |

---

### 4.6 `_wrap_obs`：观测打包（策略输入接口）

```python
def _wrap_obs(self) -> dict:
    """
    从内部 current_obs 提取最新帧，转换为 Policy 期望的观测格式。
    """
```

**返回值格式（所有世界模型环境必须严格遵守）：**

```python
obs = {
    "main_images": Tensor,          # shape: [B, H, W, 3]
                                    # dtype: torch.uint8
                                    # 数值范围: [0, 255]
                                    # 对应摄像头视角（head cam）

    "wrist_images": None,           # 世界模型无腕部摄像头，固定为 None
                                    # 若接入双视角模型可提供 [B, H, W, 3] uint8

    "states": Tensor,               # shape: [B, action_dim]
                                    # dtype: torch.float32
                                    # 世界模型无真实关节状态，用全零占位
                                    # action_dim 须与训练配置一致（用于 norm_stats 维度匹配）

    "task_descriptions": list[str], # len=B，每个 env 对应的任务自然语言描述
                                    # e.g., ["click the bell", "click the bell", ...]
}
```

**归一化转换（必须执行）：**

```python
# current_obs 内部维护为 float32 [-1, 1]
# _wrap_obs 输出给 Policy 时必须转换为 uint8 [0, 255]
last_frame = self.current_obs[:, :, 0, -1, :, :]   # [B, 3, H, W]
full_image = last_frame.permute(0, 2, 3, 1)          # [B, H, W, 3]
full_image = (full_image + 1.0) / 2.0 * 255.0        # [-1,1] → [0,255]
full_image = torch.clamp(full_image, 0, 255).to(torch.uint8)
```

> **重要：** Policy（如 Pi05）会用 `norm_stats` 对观测做二次归一化，所以这里输出 uint8 [0,255]，不要提前做策略归一化。

---

### 4.7 `chunk_step`：驱动单步推进

```python
def chunk_step(
    self,
    policy_output_action: np.ndarray | torch.Tensor
) -> tuple[list[dict], torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
```

**输入 `policy_output_action`：**

| 属性 | 值 |
|------|-----|
| 类型 | `np.ndarray` 或 `torch.Tensor` |
| 形状 | `[B, chunk, action_dim]`，即 `[num_envs, 8, 14]` |
| 含义 | 经 `prepare_actions()` 适配后的动作，仍为策略输出的原始值 |

**返回值（5-tuple）：**

```python
(
    obs_list,              # list[dict]，len=1，obs_list[0] 即 _wrap_obs() 的返回值
                           # 格式见 4.6 节

    chunk_rewards,         # torch.Tensor，shape [B, chunk]，dtype float32
                           # 每个时间步的奖励（已经过 _calc_step_reward 的差分处理）

    chunk_terminations,    # torch.Tensor，shape [B, chunk]，dtype bool
                           # True 表示该步终止（任务成功）
                           # 实现约定：只在最后一步 [:, -1] 置 True

    chunk_truncations,     # torch.Tensor，shape [B, chunk]，dtype bool
                           # True 表示该步截断（超时）
                           # 实现约定：只在最后一步 [:, -1] 置 True

    info_list,             # list[dict]，len=1，包含 episode 统计指标
                           # 常见 key: "episode_reward", "success_rate", "episode_length"
)
```

**成功/终止判定逻辑（推荐实现）：**

```python
# 基于奖励估计成功：任意帧奖励超过阈值（如 0.9）视为成功
success_threshold = 0.9   # 通常在 cfg 中配置
estimated_success = chunk_rewards.max(dim=1).values >= success_threshold   # [B], bool

# 在 chunk 最后一步标记终止/截断
chunk_terminations[:, -1] = estimated_success
chunk_truncations[:, -1] = (self.elapsed_steps >= self.cfg.max_episode_steps)
```

---

### 4.8 `offload` / `onload`：显存管理

在多 worker 并行训练时，Actor Worker 更新参数期间需要释放环境占用的 GPU 显存：

```python
def offload(self):
    """将世界模型和奖励模型移至 CPU，释放 GPU 显存。"""
    self.pipe.to("cpu")          # 或 self.model.to("cpu") 等
    self.reward_model.to("cpu")
    torch.cuda.empty_cache()
    self._is_offloaded = True

def onload(self):
    """将模型移回 GPU，准备推理。"""
    if self._is_offloaded:
        self.pipe.to(self.device)
        self.reward_model.to(self.device)
        self._is_offloaded = False
```

> `chunk_step` 和 `reset` 开头都应调用 `self.onload()`；`enable_offload: True` 时在步骤结束后调用 `self.offload()`。

---

## Step 3：动作适配（prepare_actions）

**文件：** `rlinf/envs/action_utils.py`

在 `prepare_actions()` 工厂函数中为新模型添加分支：

```python
def prepare_actions(raw_chunk_actions, env_type, model_type, num_action_chunks,
                    action_dim, action_scale=1.0, policy="widowx_bridge",
                    wm_env_type=None):
    ...
    elif env_type == SupportedEnvType.MYNEWWM:
        if wm_env_type == "my_task_type":
            chunk_actions = prepare_actions_for_my_task(raw_chunk_actions)
        else:
            raise NotImplementedError(...)
```

**各任务的动作处理规则：**

| 任务类型 | 动作维度 | 处理逻辑 |
|----------|----------|---------|
| LIBERO | 7 | gripper 维度做符号变换（第 6 维乘 -1） |
| RobotWin | 14 | 直接透传，不做任何变换 |
| 新任务 | 自定义 | 根据策略输出格式与环境输入格式的差异决定 |

---

## Step 4：编写环境配置 YAML

**文件位置：** `examples/embodiment/config/env/<model>_<task>.yaml`

**最小必填字段：**

```yaml
# 必填：对应 SupportedEnvType 枚举的字符串值
env_type: my_new_wm

# 必填：对应 wm_env_type 参数（用于 prepare_actions 分支）
wm_env_type: my_task_type

# 任务名称（用于数据集加载和视频记录）
task_suite_name: my_task_name

# 环境参数
total_num_envs: null            # 训练配置中覆盖
auto_reset: False
max_episode_steps: 200
max_steps_per_rollout_epoch: 200

# 奖励设置
use_rel_reward: True            # True=差分奖励（相邻帧奖励差），False=绝对奖励
reward_coef: 5.0                # 奖励缩放系数

# 分组（GRPO 需要 group_size > 1）
group_size: 1
use_fixed_reset_state_ids: True

# 动作维度（必须与训练数据和 Policy 一致）
action_dim: 14                  # RobotWin=14, LIBERO=7

# 世界模型参数（chunk 必须与 Policy 的 num_action_chunks 一致）
chunk: 8
condition_frame_length: 5       # 模型需要的历史帧数
num_frames: 13                  # = condition_frame_length + chunk
image_size: [256, 256]

# 数据集路径（用于 reset 时加载初始帧）
initial_image_path: /path/to/dataset/

# 模型权重路径（具体字段名由实现决定）
my_new_wm_ckpt_path: /path/to/checkpoint/

# 奖励模型
reward_model:
  type: RoboTwinT5CrossAttn
  from_pretrained: /path/to/reward_model.pth
  t5_model_name: /path/to/t5-base

# 视频录制
video_cfg:
  save_video: True
  video_base_dir: ${runner.logger.log_path}/video/train

enable_offload: False
```

---

## Step 5：编写训练配置 YAML

**文件位置：** `examples/embodiment/config/<model>_<task>_grpo_<policy>.yaml`

**核心约束（必须保持一致的参数）：**

```yaml
algorithm:
  adv_type: grpo
  loss_type: grpo_actor
  group_size: 4        # GRPO 每组样本数，必须整除 total_num_envs
  reward_coef: 5.0     # 须与 env.train.reward_coef 一致

env:
  train:
    total_num_envs: 4
    group_size: ${algorithm.group_size}   # ← 与 algorithm.group_size 绑定
    chunk: 8                              # ← 必须与 rollout/actor 的 num_action_chunks 一致
    action_dim: 14
    reward_coef: ${algorithm.reward_coef}

rollout:
  model:
    # Policy 模型路径
    model_path: /path/to/policy/checkpoint

actor:
  model:
    num_action_chunks: 8    # ← 必须与 env.chunk 一致
    action_dim: 14          # ← 必须与 env.action_dim 一致
    openpi:
      action_chunk: 8       # ← 必须与 env.chunk 一致
      action_env_dim: 14    # ← 必须与 env.action_dim 一致
```

> **整除约束（关键）：** `total_num_envs` 必须能被 `env_world_size`（通常为 GPU 数量）整除；同时 `total_num_envs` 必须是 `group_size` 的倍数。

---

## 关键约束与一致性检查清单

在提交 PR 前，请逐项确认：

| 项目 | 检查点 |
|------|--------|
| chunk 一致性 | `env.chunk` == `actor.num_action_chunks` == `actor.openpi.action_chunk` |
| action_dim 一致性 | `env.action_dim` == `actor.action_dim` == `actor.openpi.action_env_dim` == 模型训练时的 `action_dim` |
| num_frames 正确 | `num_frames` == `condition_frame_length + chunk` |
| GRPO 整除 | `total_num_envs` % `group_size` == 0 |
| GPU 整除 | `total_num_envs` % `env_world_size` == 0 |
| 图像归一化链路 | 数据集 uint8 → 内部 [-1,1] → _wrap_obs uint8 → 奖励模型 [0,1] |
| 奖励差分 | `use_rel_reward: True` 时使用差分奖励，防止奖励绝对值过高导致训练不稳 |
| reset 数据格式 | 数据集中 actions 的 key（`abs_action` 或 `delta_action`）与 `action_key` 配置一致 |

---

## 图像归一化链路总览

```
数据集存储                    数据集 uint8, [0, 255]
     │
     │ reset() 加载后立即做 [-1,1] 归一化
     ▼
current_obs (内部缓冲)        float32, [-1, 1]
     │                        shape: [B, 3, 1, T, H, W]
     │
     ├──→ 世界模型输入          依模型而异：
     │                          Wan：通过 image_queue 存 PIL Image [0,255]
     │                          OpenSora：通过 image_queue 存 latent 张量
     │
     ├──→ 奖励模型输入          float32, [0, 1]
     │                          (current_obs + 1) / 2
     │
     └──→ _wrap_obs() 输出     uint8, [0, 255]
               │                (current_obs + 1) / 2 * 255
               ▼
          Policy 输入 obs["main_images"]
               │
               │ Policy 内部 norm_stats 再次归一化（框架自动处理）
               ▼
          Policy 内部特征提取
```

---

## 两个参考实现对比

| 特性 | Wan（DiffSynth） | OpenSora（STDiT3） |
|------|-----------------|-------------------|
| **枚举值** | `wan_wm` | `opensora_wm` |
| **推理接口** | `WanVideoPipeline(...)` | `scheduler.sample(model, z, y, mask)` |
| **条件帧存储** | PIL Image 列表（pixel space）| VAE latent 张量（latent space，deque）|
| **条件帧数** | 5 帧（pixel space）| 4 帧（latent space，对应 1 latent frame per 4 images）|
| **动作归一化** | 无（原始值直传）| q01/q99 分位数归一化到 [-1,1] |
| **动作历史** | 保留 `condition_action`（5帧动作）拼接输入 | 不保留动作历史 |
| **KIR Trick** | 支持（`enable_kir: True`）| 不支持 |
| **VAE 类型** | 内置于 Pipeline | 独立 VAE（SDXL 或 OpenSoraVAE_V1_2）|
| **批处理方式** | 逐 env 串行推理 | 所有 env 一次批量推理 |
| **内部帧格式** | `[B, 3, 1, T, H, W]` float32 [-1,1] | `[B, 3, 1, T, H, W]` float32 [-1,1] |
| **环境配置文件** | `env/wan_robotwin_click_bell.yaml` | `env/opensora_robotwin_click_bell.yaml` |
| **训练配置文件** | `wan_robotwin_click_bell_grpo_openpi_pi05.yaml` | `opensora_robotwin_click_bell_grpo_pi05.yaml` |

---

## 快速接入新模型的最小实现步骤

1. **注册**：在 `rlinf/envs/__init__.py` 的 `SupportedEnvType` 和 `get_env_cls()` 中添加新项
2. **实现**：创建 `rlinf/envs/world_model/world_model_<name>_env.py`，继承 `BaseWorldEnv`，实现 `_build_dataset`、`reset`、`_infer_next_chunk_frames`、`_infer_next_chunk_rewards`、`_wrap_obs`、`chunk_step` 六个方法
3. **动作适配**：在 `rlinf/envs/action_utils.py` 的 `prepare_actions()` 中添加分支
4. **环境配置**：创建 `examples/embodiment/config/env/<name>_<task>.yaml`
5. **训练配置**：创建 `examples/embodiment/config/<name>_<task>_grpo_<policy>.yaml`，确保 `chunk` 和 `action_dim` 三处一致
6. **验证**：单 GPU 先跑 `total_num_envs=1` 的 smoke test，确认 `reset → chunk_step → obs` 的形状和数值范围符合上表
