# Real-world Benchmark — 接口说明（中文）

概述
- 目的：统一对接真实机测试策略（Policy），以便把不同模型按同一输入/输出规范进行评估与比对。
- 位置：`real_world_benchmark/`

接口约定（核心）
- 测试者提供一个 Python 文件或模块，文件内定义类 `Policy`，并实现方法 `infer(self, new_obs)`。
- Benchmark 会导入该 `Policy`，构造或接收 `new_obs`，调用 `Policy.infer(new_obs)`，并解析返回的动作（`output`）。

new_obs（传入 `Policy.infer` 的观察）
- 类型：Python `dict`。
- 常见字段：
  - `images`: dict，按相机名映射到图像数组（numpy.ndarray，dtype uint8 或 float）
    - `cam_high`: ndarray(H, W, 3) — 俯视/主相机当前帧（示例也会复制到 `first_frame`）
    - `cam_left_wrist`, `cam_right_wrist`: ndarray(H, W, 3) — 腕部相机帧（可选）
    - `cam_high_memory`: ndarray(T, H, W, 3) — 历史帧序列（可选）
  - `first_frame`: ndarray(H, W, 3) — 与 `images['cam_high']` 相同（便于模型直接使用）
  - `state`: numpy.ndarray — 数值状态向量（robot state / eef / joints）
    - 常见格式：
      - eef6d 风格（示例代码中常见长度 20 或 32）：
        - 0:3 = left_pos (x,y,z)
        - 3:9 = left_rot6d (6-d continuous 表示)
        - 9:10 = left_gripper
        - 10:13 = right_pos
        - 13:19 = right_rot6d
        - 19:20 = right_gripper
        - 20:32 = padding（如存在）
      - joint 空间风格：例如两臂各 7 关节 → 长度 14
  - `prompt`: 可选字符串任务描述
  - 其它任意字段：策略可按需读取

注意：图像可能为 float（[0,1]）或 uint8（[0,255]），策略实现应健壮处理两种情况。

output（`Policy.infer` 的返回值）
- 类型：Python `dict`。
- 必含字段：
  - `actions`: numpy.ndarray 或 list，形状通常为 (T, D) 或 (1, D) 或 (D,)。
    - T = 时间步 horizon，D = action_dim（例如 32 或 14 等）
    - 当是 eef6d 风格（D >= 20 或 32）时，前 20 维含有效 eef 信息（参见 `state` 的映射），其余为 padding
- 可选字段：`policy_timing`、`video`、调试信息等
- 旋转格式：若输出为 6-d continuous rotation（6维），上层需按 cont6d->matrix->quat 进行转换（benchmark runner 会示例如何处理）

示例最小 new_obs

```py
new_obs = {
    'images': {'cam_high': np.zeros((240,320,3), dtype=np.uint8)},
    'first_frame': np.zeros((240,320,3), dtype=np.uint8),
    'state': np.zeros((32,), dtype=np.float32),
    'prompt': 'place the red block'
}
```

示例 output

```py
output = {
    'actions': np.zeros((1, 32), dtype=np.float32),
    'policy_timing': {'infer_ms': 12.3}
}
```

如何使用示例 runner
- 将你实现的策略文件（例如 `my_policy.py`，内部定义 `Policy`）放在任意位置。
- 直接运行示例 runner（离线 smoke test）：

```bash
python -m real_world_benchmark.benchmark_runner    # 默认加载 example_policy
python -m real_world_benchmark.benchmark_runner /path/to/my_policy.py
python -m real_world_benchmark.benchmark_runner my_module.path  # 作为模块导入
```

Training-data offline mode
- 该模式会从训练数据中按 `AgileXDataset` 读取样本，直接构造 `new_obs`，适合做“模型能力初筛”和离线接口验证。
- 默认数据目录与脚本里的 debug 路径一致，你可以通过 `--dataset-dir`、`--dataset-index`、`--dataset-step` 调整采样位置。

```bash
python -m real_world_benchmark.benchmark_runner /path/to/my_policy.py --mode dataset
python -m real_world_benchmark.benchmark_runner your.module.path --mode dataset --dataset-limit 20
python -m real_world_benchmark.benchmark_runner your.module.path --mode dataset --dataset-dir /path/to/train_data --dataset-index 1000 --dataset-step 30
```

Dataset mode args
- `--dataset-dir`：训练数据目录。
- `--dataset-index`：起始帧。
- `--dataset-step`：帧间隔，通常可设为 30，对齐真机滑窗节奏。
- `--dataset-limit`：评估多少个样本。
- `--dataset-action-horizon`：数据集动作长度，默认 50。
- `--dataset-action-type eef6d|joint_angle`：数据集动作类型。
- `--read-from-hdf5`：从 HDF5 读图像。

Live benchmark mode
- 真实机器人测试时使用 `--mode live`，runner 会像 `scripts/pi05_wma_server_eef_vpp.py` 一样调用 `wait_observation()`，把 `img_front/img_left/img_right` 组装成 `new_obs`，并读取 `left_end_pose/right_end_pose`、`left_arm_joint_state/right_arm_joint_state` 生成 `state`。
- 如果你的策略输出的是动作序列，并且希望回传到服务器，可加 `--send-action`。

```bash
python -m real_world_benchmark.benchmark_runner /path/to/my_policy.py --mode live --send-action
python -m real_world_benchmark.benchmark_runner your.module.path --mode live --max-steps 100
```

Live mode args
- `--use-history`：在 `new_obs['images']['cam_high_memory']` 中附加历史帧。
- `--action-format auto|eef6d|joint`：动作解析方式，默认自动判断。
- `--max-steps N`：live 迭代次数，`<=0` 表示持续运行。
- `--action-rate`：回传动作频率。

示例策略与 runner 位于：
- `real_world_benchmark/example_policy.py`
- `real_world_benchmark/benchmark_runner.py`

如果需要我可以：
- 增加更详细的动作坐标系说明（camera vs base）、时间序列约定。 
- 提供 `Policy` 抽象基类或类型提示文件以方便静态检查。
