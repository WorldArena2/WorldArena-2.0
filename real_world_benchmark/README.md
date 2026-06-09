# Real-world Benchmark — Interface Documentation (English)

## Overview
- **Purpose**: Provide a unified interface for evaluating real-robot test policies, enabling consistent input/output specifications across different models.
- **Location**: `real_world_benchmark/`

## Core Interface Contract
- The testee provides a Python file or module that defines a class `Policy` with a method `infer(self, new_obs)`.
- The benchmark imports the `Policy`, constructs or receives `new_obs`, calls `Policy.infer(new_obs)`, and parses the returned action (`output`).

## `new_obs` (Observation passed to `Policy.infer`)
- **Type**: Python `dict`
- **Common fields**:
  - `images`: `dict` mapping camera names to image arrays (`numpy.ndarray`, dtype `uint8` or `float`)
    - `cam_high`: ndarray(H, W, 3) — top/main camera current frame (also duplicated to `first_frame`)
    - `cam_left_wrist`, `cam_right_wrist`: ndarray(H, W, 3) — wrist camera frames (optional)
    - `cam_high_memory`: ndarray(T, H, W, 3) — history frame sequence (optional)
  - `first_frame`: ndarray(H, W, 3) — same as `images['cam_high']` (convenience for models)
  - `state`: `numpy.ndarray` — numeric state vector (robot state / eef / joints)
    - Typical formats:
      - **eef6d style** (common lengths 20 or 32 in example code):
        - 0:3 = left_pos (x,y,z)
        - 3:9 = left_rot6d (6-d continuous rotation)
        - 9:10 = left_gripper
        - 10:13 = right_pos
        - 13:19 = right_rot6d
        - 19:20 = right_gripper
        - 20:32 = padding (if present)
      - **Joint space style**: e.g. 7 joints per arm → length 14
  - `prompt`: optional string task description
  - Any other fields: policies may read as needed

**Note**: Images may be float ([0,1]) or uint8 ([0,255]); policy implementations should handle both robustly.

## `output` (Return value of `Policy.infer`)
- **Type**: Python `dict`
- **Mandatory field**:
  - `actions`: `numpy.ndarray` or list, shape typically `(T, D)`, `(1, D)`, or `(D,)`.
    - `T` = time steps horizon, `D` = action dimension (e.g. 32 or 14)
    - For eef6d style (D >= 20 or 32), first 20 dimensions contain valid eef info (see `state` mapping), the rest are padding
- **Optional fields**: `policy_timing`, `video`, debug info, etc.
- **Rotation format**: If output uses 6-d continuous rotation, the upper layer converts cont6d -> matrix -> quat (the benchmark runner provides an example).

### Example minimal `new_obs`
```python
new_obs = {
    'images': {'cam_high': np.zeros((240,320,3), dtype=np.uint8)},
    'first_frame': np.zeros((240,320,3), dtype=np.uint8),
    'state': np.zeros((32,), dtype=np.float32),
    'prompt': 'place the red block'
}
```

### Example `output`
```python
output = {
    'actions': np.zeros((1, 32), dtype=np.float32),
    'policy_timing': {'infer_ms': 12.3}
}
```

## How to use the example runner
- Place your policy file (e.g. `my_policy.py`, with a `Policy` class defined inside) anywhere.
- Run the example runner (offline smoke test) directly:

```bash
python -m real_world_benchmark.benchmark_runner    # loads example_policy by default
python -m real_world_benchmark.benchmark_runner /path/to/my_policy.py
python -m real_world_benchmark.benchmark_runner my_module.path  # import as module
```

## Training-data offline mode
- This mode reads samples from training data (`AgileXDataset`) to construct `new_obs`. Suitable for initial model capability screening and offline interface validation.
- Default data directory matches the debug path in the script; you can adjust sampling via `--dataset-dir`, `--dataset-index`, `--dataset-step`.

```bash
python -m real_world_benchmark.benchmark_runner /path/to/my_policy.py --mode dataset
python -m real_world_benchmark.benchmark_runner your.module.path --mode dataset --dataset-limit 20
python -m real_world_benchmark.benchmark_runner your.module.path --mode dataset --dataset-dir /path/to/train_data --dataset-index 1000 --dataset-step 30
```

### Dataset mode arguments
- `--dataset-dir` : Training data directory.
- `--dataset-index` : Starting frame index.
- `--dataset-step` : Frame interval (e.g., 30 to match real‑robot sliding window rhythm).
- `--dataset-limit` : Number of samples to evaluate.
- `--dataset-action-horizon` : Action horizon in dataset, default 50.
- `--dataset-action-type eef6d|joint_angle` : Action type in dataset.
- `--read-from-hdf5` : Read images from HDF5.

## Live benchmark mode
- For real‑robot testing, use `--mode live`. The runner assembles `new_obs` by calling `wait_observation()` (similar to `scripts/pi05_wma_server_eef_vpp.py`), using `img_front/img_left/img_right` for images, and reading `left_end_pose/right_end_pose`, `left_arm_joint_state/right_arm_joint_state` for the `state`.
- If your policy outputs action sequences and you want to send them back to the server, add `--send-action`.

```bash
python -m real_world_benchmark.benchmark_runner /path/to/my_policy.py --mode live --send-action
python -m real_world_benchmark.benchmark_runner your.module.path --mode live --max-steps 100
```

### Live mode arguments
- `--use-history` : Attach history frames to `new_obs['images']['cam_high_memory']`.
- `--action-format auto|eef6d|joint` : How to parse actions, auto‑detect by default.
- `--max-steps N` : Number of live iterations; `<=0` means run indefinitely.
- `--action-rate` : Frequency of sending actions back.

## Example policy and runner location
- `real_world_benchmark/example_policy.py`
- `real_world_benchmark/benchmark_runner.py`
