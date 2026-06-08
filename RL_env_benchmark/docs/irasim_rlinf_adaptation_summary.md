# IRASim 接入 RLinf 的适配总结

## 当前结论

- 当前 click_bell 配置是双卡放置，因为 `component_placement` 配的是 `actor, env, rollout: 0,1`。
- 当前 `group_size = 4`、`total_num_envs = 8`、`pipeline_stage_num = 1` 这组参数在 RLinf 的整除规则下是合法的。
- 如果现在仍然启动失败，主问题更像是 Ray runtime，而不是之前的 `group_size` 数学问题。

## 我这次做了什么

### 1. 接入 IRASim world-model env

核心文件是 `rlinf/envs/world_model/world_model_irasim_env.py`。

这里完成了：
- 把 IRASim 包成 RLinf 可调用的环境。
- 对接 `reset()` 和 `chunk_step()`。
- 固化 RobotWin 约束：`condition_frame_length = 1`、`chunk = 15`、`num_frames = 16`。
- 加载 IRASim 模型、VAE、scheduler、reward model。
- 读取 `stat.json` 做动作归一化。

### 2. 注册 env 并接动作链路

- `rlinf/envs/__init__.py` 注册了 `irasim_wm`。
- `rlinf/envs/action_utils.py` 让 actor 输出动作后能正确送进 IRASim env。

### 3. 调整 IRASim 训练配置

重点文件：
- `examples/embodiment/config/irasim_robotwin_click_bell_grpo_openpi_pi05.yaml`
- `examples/embodiment/config/env/irasim_robotwin_click_bell.yaml`

主要围绕：
- 双卡 placement
- `group_size`
- `rollout_epoch`
- `total_num_envs`
- actor batch size
- video 保存
- WandB 日志

### 4. 修正 WandB 体验

相关文件是 `rlinf/utils/metric_logger.py`。

这里主要做了：
- 给 `train/*`、`env/*`、`rollout/*` 等指标定义更清楚的 step 关系。
- 关闭 WandB 的 console capture，减少和 `tee` 的冲突。

### 5. 修正 chunk 视频保存逻辑

这次新修的关键点：
- `rlinf/envs/wrappers/record_video.py`
- `examples/embodiment/config/env/irasim_robotwin_click_bell.yaml`

现在给 `RecordVideo` 增加了 `chunk_keep_mode`，并在 IRASim train 视频配置里设成了 `last`。

这意味着：
- 对 chunked env 来说，如果 observation 里带时间维，录像时只保留 chunk 的最后一帧。
- 不会再因为 chunk 内部时间维被展开，导致视频观感失真。

## IRASim 在 RL 里怎么工作

可以把这套系统理解成：
- OpenPI 是 actor。
- IRASim 是 env。

训练闭环是：
`observation -> actor(OpenPI) -> action -> IRASim rollout -> reward -> RL update`

更细一点：
- `reset()` 时读取起始观测并准备条件帧。
- actor 输出一段 `chunk=15` 的动作。
- IRASim 基于条件帧和动作生成未来帧。
- reward model 对生成结果打分。
- RLinf 用 reward 更新 actor。

## 为什么 `group_size` 会报错

关键约束是：
`env.train.total_num_envs // env_world_size // rollout.pipeline_stage_num`

这个结果必须能被 `algorithm.group_size` 整除。

所以：
- 单卡和双卡即使都写 `group_size: 4`，合法性也可能不同。
- 你把 `total_num_envs` 从 4 调到 8 后，旧的整除错误其实已经解决了。
- 后面再报错时，更像是 Ray runtime 问题。

## 为什么 WandB 一开始只有 system

因为 WandB 只能显示程序显式 `log()` 上去的标量。
在真正的 rollout 和训练 step 完成之前，通常只会先看到 `system`。
等 env metrics、training metrics、time metrics 开始上报后，loss 和 reward 曲线才会出现。

## 建议怎么看这套代码

建议按这个顺序看：
- `docs/irasim_world_model_rl_integration.md`
- `rlinf/envs/world_model/world_model_irasim_env.py`
- `examples/embodiment/config/env/irasim_robotwin_click_bell.yaml`
- `examples/embodiment/config/irasim_robotwin_click_bell_grpo_openpi_pi05.yaml`
- `rlinf/envs/wrappers/record_video.py`

## 一句话总结

这次工作的本质，是把 IRASim 从一个世界模型推理系统接成了 RLinf 可直接调度的环境，然后让 OpenPI 在这个环境里 rollout、拿 reward、再做 RL 更新。
