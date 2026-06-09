# World Model Environment Integration Guide

This guide describes how to train a world model and integrate it into RLinf as
an environment for policy optimization. The interface is model-agnostic: Wan is
used only as a concrete example.

## Path and Privacy Configuration

Example configs in this repository avoid private machine paths, usernames, mount points, and API keys. Local resources such as datasets, checkpoints, reward models, and simulator assets should be provided through environment variables or documented placeholders.

Common variables for world-model-based embodied training include:

```bash
export WAN_PATH=/path/to/diffsynth-studio
export WAN_ROBOTWIN_ADJUST_BOTTLE_CKPT=/path/to/RLinf-Wan-RobotWin-AdjustBottle
export WAN_ROBOTWIN_CLICK_BELL_CKPT=/path/to/RLinf-Wan-RobotWin-ClickBell
export OPENPI_CKPT_PATH=/path/to/openpi-checkpoint
export ROBOTWIN_REWARD_MODEL_PATH=/path/to/robotwin_reward_model.pth
export T5_MODEL_PATH=/path/to/t5-base
export ROBOTWIN_PATH=/path/to/RoboTwin
```

For the world-model training and integration workflow, see [`README_WORLD_MODEL_ENV.md`](README_WORLD_MODEL_ENV.md).

## 1. Goal

RLinf can use a learned world model in place of, or alongside, a simulator. The
world model receives an initial visual state and a chunk of policy actions, then
predicts the next visual observations. RLinf uses those observations and a
reward model to optimize the policy.

At a high level:

```text
raw robot trajectories
  -> world-model training dataset
  -> trained world-model checkpoint
  -> RLinf world-model environment
  -> policy rollout and RL update
```

The world model can run in either of two modes:

- **In-process env**: the world model is loaded directly inside the RLinf env worker.
- **HTTP env**: the world model runs as a separate server, and RLinf talks to it through a proxy env.

## 2. Components

A new world-model integration usually needs these pieces:

```text
World-model project
  - data encoder
  - training script
  - inference/checkpoint loader

RLinf
  - env implementation
  - optional HTTP server
  - optional HTTP proxy env
  - action conversion logic
  - env config
  - training config
```

For the existing Wan example, these files are relevant:

```text
diffsynth-studio/
  examples/wanvideo/model_training/encode_vla_to_rlinf.py
  examples/wanvideo/model_training/train_rlinf.py

rlinf/envs/world_model/
  world_model_wan_env.py
  wan_http_server.py
  world_model_wan_http_env.py

examples/embodiment/config/env/
  wan_robotwin_adjust_bottle_http.yaml

examples/embodiment/config/
  wan_robotwin_adjust_bottle_http_grpo_openpi_pi05.yaml
```

## 3. Paths and Privacy

Do not commit local machine paths, usernames, private mount points, API keys, or
organization-internal storage locations. Use environment variables or documented
placeholders instead.

Recommended variables:

```bash
export RLINF_ROOT=/path/to/RLinf
export WORLD_MODEL_ROOT=/path/to/world-model-project
export WAN_PATH=${RLINF_ROOT}/diffsynth-studio
export RAW_DATA_ROOT=/path/to/raw/robot_trajectories
export WM_DATA_ROOT=/path/to/encoded_world_model_data
export WAN_ROBOTWIN_ADJUST_BOTTLE_CKPT=/path/to/RLinf-Wan-RobotWin-AdjustBottle
export WAN_ROBOTWIN_CLICK_BELL_CKPT=/path/to/RLinf-Wan-RobotWin-ClickBell
export OPENPI_CKPT_PATH=/path/to/openpi-checkpoint
export ROBOTWIN_REWARD_MODEL_PATH=/path/to/robotwin_reward_model.pth
export T5_MODEL_PATH=/path/to/t5-base
export ROBOTWIN_PATH=/path/to/RoboTwin
```

Open-source configs should prefer Hydra environment interpolation:

```yaml
model_path: ${oc.env:OPENPI_CKPT_PATH,/path/to/openpi-checkpoint}
reward_model:
  from_pretrained: ${oc.env:ROBOTWIN_REWARD_MODEL_PATH,/path/to/reward_model.pth}
```

Before publishing, scan for:

```text
private absolute paths
usernames
API keys and tokens
internal hostnames or mount points
private dataset names that should not be public
```

## 4. Data Contract

The raw data format is world-model specific, but it should contain enough
information to train action-conditioned dynamics:

- visual observations
- robot states, if needed
- action vectors
- task descriptions or language instructions, if needed
- episode boundaries

A recommended raw layout is:

```text
${RAW_DATA_ROOT}/
  <data_source>/
    <task_name>/
      <split_or_collection_name>/
        data/
          episode*.hdf5
        instructions/
          episode*.json
        video/
          episode*.mp4
```

The data encoder should convert raw trajectories into the format expected by
the world-model trainer. A minimal action-conditioned video dataset usually has:

```text
${WM_DATA_ROOT}/<task_name>/
  train_data/
    <source>/<episode>/
      frames.npy
      actions.npy
  val_data/
    <source>/<episode>/
      frames.npy
      actions.npy
```

The exact filenames can differ, but document them clearly. The important part is
that the training script and env reset code agree on frame and action shapes.

### Wan Example

The Wan RoboTwin encoder reads raw HDF5 trajectories with:

```text
observation/head_camera/rgb   # JPEG bytes per frame
joint_action/vector           # [T, 14] action vector
```

It writes:

```text
rgb.npy       # [T, 1, H, W, 3] uint8
actions.npy   # [T, 1, action_dim] float32
```

Example:

```bash
cd ${WORLD_MODEL_ROOT}

python examples/wanvideo/model_training/encode_vla_to_rlinf.py \
  --source-root ${RAW_DATA_ROOT} \
  --output-root ${WM_DATA_ROOT} \
  --tasks adjust_bottle \
  --resize 256 256 \
  --workers 32
```

## 5. Training Contract

Every world-model trainer should clearly define:

- input dataset path
- frame window length
- action dimension
- image size
- base model or pretrained checkpoint path
- output checkpoint path

For chunk-based RL environments, the frame window should match:

```text
num_frames = condition_frame_length + action_chunk_length
```

For example, if the policy outputs 8 actions per chunk and the world model uses
5 condition frames:

```text
num_frames = 5 + 8 = 13
```

### Wan Example

Wan training uses `RLinfNpyDataset`. The trainer reads:

```text
${dataset_base_path}/train_data
${dataset_base_path}/val_data
```

Example:

```bash
cd ${WORLD_MODEL_ROOT}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml \
  examples/wanvideo/model_training/train_rlinf.py \
  --height 256 \
  --width 256 \
  --num_frames 13 \
  --dataset RLinfNpyDataset \
  --dataset_base_path ${WM_DATA_ROOT}/adjust_bottle \
  --learning_rate 1e-5 \
  --num_epochs 100000 \
  --trainable_models dit \
  --extra_inputs "input_image,action" \
  --output_path outputs/adjust_bottle
```

## 6. Runtime Checkpoint Layout

RLinf should not depend on training-time output conventions directly. Prepare a
runtime checkpoint directory with the files required for inference and reset:

```text
${WM_CKPT_DIR}/
  model checkpoint files
  tokenizer/config files, if needed
  reset_dataset/
```

The reset dataset is separate from the training dataset. It is used by RLinf to
choose initial states and task metadata for rollout.

Recommended reset sample contents:

- initial condition frames
- condition actions
- task description
- initial robot state, if needed
- reference or target frame, if the reward model needs it

### Wan Example

The current Wan env expects:

```text
${WM_CKPT_DIR}/
  dit_model.safetensors
  Wan2.2_VAE.pth
  dataset/
    episode*.npy
```

The RLinf config points to:

```yaml
wm_ckpt_path: ${oc.env:WM_CKPT_DIR}
model_path: ${.wm_ckpt_path}/dit_model.safetensors
vae_path: ${.wm_ckpt_path}/Wan2.2_VAE.pth
reset_dataset_path: ${.wm_ckpt_path}/dataset/
```

The actual Wan config names are:

```yaml
wan_wm_hf_ckpt_path: ${oc.env:WM_CKPT_DIR}
model_path: ${.wan_wm_hf_ckpt_path}/dit_model.safetensors
VAE_path: ${.wan_wm_hf_ckpt_path}/Wan2.2_VAE.pth
initial_image_path: ${.wan_wm_hf_ckpt_path}/dataset/
```

## 7. RLinf Env Interface

A world-model env should expose the same high-level interface expected by RLinf:

```python
reset() -> tuple[dict, dict]

chunk_step(actions) -> tuple[
    list[dict],      # obs_list
    Tensor,          # chunk_rewards, [num_envs, chunk]
    Tensor,          # terminations, [num_envs, chunk]
    Tensor,          # truncations, [num_envs, chunk]
    list[dict],      # infos_list
]
```

The observation dict must match the policy input adapter. For an image policy, a
typical observation is:

```python
{
    "main_images": ...,        # [B, H, W, C], uint8
    "wrist_images": ...,       # optional
    "states": ...,             # [B, state_dim]
    "task_descriptions": ...,  # list[str]
}
```

The env is responsible for:

- sampling or receiving reset states
- preparing model condition frames
- sending action chunks to the world model
- wrapping predicted frames into policy observations
- computing or delegating rewards
- producing termination and truncation flags

## 8. HTTP Env Interface

For a decoupled deployment, split the system into:

- **env-side server**: owns the heavy world model and GPU inference.
- **host-side proxy env**: owns RLinf-facing reset, reward, and training logic.

Minimal HTTP endpoints:

```text
GET  /health
POST /reset
POST /chunk_step
```

Minimal `/reset` payload:

```text
condition frames
condition actions
task descriptions
optional states or metadata
```

Minimal `/chunk_step` payload:

```text
actions  # [num_envs, action_chunk, action_dim]
```

Minimal `/chunk_step` response:

```text
predicted frames or current observation state
elapsed step count
optional model diagnostics
```

The transport format is replaceable. The current minimal implementation uses
pickle + base64 in JSON for simplicity. For production, use a safer and faster
format such as msgpack, Arrow, shared-memory tensors, object storage references,
or streaming.

### Wan HTTP Example

Start the world-model server:

```bash
cd ${RLINF_ROOT}

WORLD_MODEL_PATH=${RLINF_ROOT}/diffsynth-studio
WAN_PATH=${WORLD_MODEL_PATH} \
ROBOTWIN_PATH=${ROBOTWIN_PATH} \
  bash examples/embodiment/run_wan_http_server.sh \
  env/wan_robotwin_adjust_bottle_http
```

Start host-side RL training in another terminal:

```bash
cd ${RLINF_ROOT}

WAN_PATH=${RLINF_ROOT}/diffsynth-studio \
ROBOTWIN_PATH=${ROBOTWIN_PATH} \
  bash examples/embodiment/run_embodiment.sh \
  wan_robotwin_adjust_bottle_http_grpo_openpi_pi05 ALOHA
```

## 9. Reward Model

Reward computation can be placed in either the world-model process or the host
process. For RL training, keeping reward computation on the host side is often
easier because it avoids coupling the world-model server to task-specific
reward code.

A reward model should expose a simple method such as:

```python
compute_reward(observations, **kwargs) -> Tensor
```

Common reward inputs:

- predicted frames
- target or reference frames
- task descriptions
- robot states
- action chunks

### Wan Example

The current HTTP Wan path keeps reward on the host side. The server only
generates frames. The host-side proxy env receives generated frames, then calls
the configured reward model.

Example config:

```yaml
reward_model:
  type: RoboTwinT5CrossAttn
  from_pretrained: /path/to/reward_model.pth
  t5_model_name: /path/to/t5-base
```

## 10. Registering a New World Model

To add a new world model:

1. Add an env class under `rlinf/envs/world_model/`.
2. Register a new `env_type` in `rlinf/envs/__init__.py`.
3. Add action conversion logic in `rlinf/envs/action_utils.py` if needed.
4. Add an env config under `examples/embodiment/config/env/`.
5. Add a training config under `examples/embodiment/config/`.
6. Add an optional HTTP server and proxy env if the model should run out of process.

Example env type names:

```text
my_world_model
my_world_model_http
```

Example config skeleton:

```yaml
env_type: my_world_model_http
task_suite_name: my_task
wm_env_type: robotwin

total_num_envs: 2
group_size: 2
max_episode_steps: 8
max_steps_per_rollout_epoch: 8

model_path: /path/to/world_model_checkpoint
reset_dataset_path: /path/to/reset_dataset

condition_frame_length: 5
chunk: 8
num_frames: 13
image_size: [256, 256]
action_dim: 14

http:
  server_url: http://127.0.0.1:18080
  timeout: 600.0
```

## 11. Validation Checklist

Before running long RL jobs, verify:

- raw data can be encoded
- encoded train/val dataset can be loaded by the world-model trainer
- trained checkpoint can be loaded for inference
- reset dataset can be loaded by RLinf
- `num_frames == condition_frame_length + chunk`
- policy action dimension matches world-model action dimension
- `reset()` returns policy-compatible observations
- `chunk_step()` returns rewards and done flags with expected shapes
- HTTP `/health` returns OK, if using server mode
- HTTP `/reset` and `/chunk_step` return OK, if using server mode
- a smoke RL run reaches at least one training step

Expected smoke-test signals:

```text
POST /reset 200 OK
POST /chunk_step 200 OK
Global Step: 1/1
```

## 12. Open-Source Checklist

Before publishing a world-model integration:

- replace private absolute paths with environment variables
- remove API keys and credentials
- document raw data schema
- document encoded data schema
- document checkpoint layout
- provide a minimal config
- provide a smoke-test command
- isolate model-specific code behind a clear env or HTTP interface
- include a toy dataset or mock world model for CI when possible
# World Model Training and Environment Integration

This document describes how to train a world model and plug it into RLinf as an
RL environment. Wan is used as the concrete example, but the same contract can
be reused by another world model.

## 1. Overall Flow

```text
Raw robot trajectories
  -> encode into world-model training format
  -> train world model
  -> export checkpoint + reset dataset
  -> configure RLinf env
  -> run policy RL against the world-model env
```

For Wan + RoboTwin, the current split is:

- `diffsynth-studio/`: Wan training, inference, and checkpoint loading code.
- `rlinf/envs/world_model/world_model_wan_env.py`: in-process Wan env.
- `rlinf/envs/world_model/wan_http_server.py`: env-side HTTP Wan server.
- `rlinf/envs/world_model/world_model_wan_http_env.py`: host-side HTTP proxy env.
- `examples/embodiment/config/env/wan_robotwin_adjust_bottle_http.yaml`: minimal HTTP env config.
- `examples/embodiment/config/wan_robotwin_adjust_bottle_http_grpo_openpi_pi05.yaml`: minimal GRPO training config.

## 2. Recommended Directory Variables

Use environment variables instead of hard-coded local paths:

```bash
export RLINF_ROOT=/path/to/RLinf
export WAN_PATH=${RLINF_ROOT}/diffsynth-studio
export RAW_DATA_ROOT=/path/to/raw/robotwin_trajectories
export ENCODED_DATA_ROOT=${WAN_PATH}/encoded_data
export WM_CKPT_DIR=${WAN_PATH}/RLinf-Wan-RobotWin-AdjustBottle
export ROBOTWIN_PATH=/path/to/RoboTwin
```

`WAN_PATH` must be prepended to `PYTHONPATH` when running Wan training or RLinf
with Wan:

```bash
export PYTHONPATH=${WAN_PATH}:${RLINF_ROOT}:${ROBOTWIN_PATH}:${PYTHONPATH}
```

## 3. Raw Data Format

Wan training starts from raw robot trajectory files, not from encoded world-model
training tensors.

For the RoboTwin example, the encoder expects this layout:

```text
${RAW_DATA_ROOT}/
  <vla_source>/
    <task_name>/
      demo_clean/
        data/
          episode*.hdf5
        scene_info.json
        instructions/
          episode*.json
        video/
          episode*.mp4
```

Each `episode*.hdf5` should contain:

```text
observation/head_camera/rgb   # JPEG bytes per frame
joint_action/vector           # [T, 14] float action vector
```

The current Wan RoboTwin encoder supports sources such as:

```text
10radiodata_10000
fulldata_40000
ref
```

and tasks such as:

```text
adjust_bottle
click_bell
```

## 4. Encode Raw Data

The encoder converts raw HDF5 trajectories into `RLinfNpyDataset`, which is
consumed by Wan training.

```bash
cd ${WAN_PATH}

python examples/wanvideo/model_training/encode_vla_to_rlinf.py \
  --source-root ${RAW_DATA_ROOT} \
  --output-root ${ENCODED_DATA_ROOT} \
  --tasks adjust_bottle \
  --vlas 10radiodata_10000 fulldata_40000 ref \
  --resize 256 256 \
  --workers 32
```

The output layout is:

```text
${ENCODED_DATA_ROOT}/adjust_bottle/
  train_data/
    <vla_source>/<episode>/
      rgb.npy       # [T, 1, H, W, 3] uint8
      actions.npy   # [T, 1, 14] float32
  val_data/
    <vla_source>/<episode>/
      rgb.npy
      actions.npy
```

## 5. Train Wan

Wan training uses the encoded dataset above. A full training command is usually
wrapped by a script under:

```text
${WAN_PATH}/examples/wanvideo/model_training/full/
```

For example:

```bash
cd ${WAN_PATH}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml \
  examples/wanvideo/model_training/train_rlinf.py \
  --height 256 \
  --width 256 \
  --num_frames 13 \
  --dataset RLinfNpyDataset \
  --dataset_base_path ${ENCODED_DATA_ROOT}/adjust_bottle \
  --model_paths '[
    ["'${WAN_PATH}'/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors",
     "'${WAN_PATH}'/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors",
     "'${WAN_PATH}'/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"],
    "'${WAN_PATH}'/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
  ]' \
  --learning_rate 1e-5 \
  --num_epochs 100000 \
  --trainable_models dit \
  --extra_inputs "input_image,action" \
  --context_noise_sigma 0.0 \
  --static_video_prob 0.05 \
  --output_path outputs/adjust_bottle
```

The important dataset contract is:

- `--dataset_base_path` points to one task directory.
- The trainer reads `${dataset_base_path}/train_data` and `${dataset_base_path}/val_data`.
- `--num_frames` must match the world-model rollout window. In the current Wan env:

```text
num_frames = condition_frame_length + chunk = 5 + 8 = 13
```

## 6. Export Checkpoint for RLinf

After training, place the files needed by the RLinf world-model env under one
checkpoint directory:

```text
${WM_CKPT_DIR}/
  dit_model.safetensors
  Wan2.2_VAE.pth
  dataset/
    episode*.npy
```

The `dataset/` directory is used by RLinf env reset. It provides initial
condition frames, actions, task descriptions, and target frames for reward
calculation. It is different from the large encoded training dataset.

The RLinf env config points to these files:

```yaml
wan_wm_hf_ckpt_path: ${oc.env:WM_CKPT_DIR}
VAE_path: ${.wan_wm_hf_ckpt_path}/Wan2.2_VAE.pth
model_path: ${.wan_wm_hf_ckpt_path}/dit_model.safetensors
initial_image_path: ${.wan_wm_hf_ckpt_path}/dataset/
```

## 7. Use Wan as an In-Process RLinf Environment

Use `env_type: wan_wm` when the Wan model runs inside the RLinf env worker.

Minimal env config:

```yaml
env_type: wan_wm
task_suite_name: robotwin_adjust_bottle
wm_env_type: robotwin

total_num_envs: 32
group_size: 4
max_episode_steps: 200
max_steps_per_rollout_epoch: 200

wan_wm_hf_ckpt_path: ${oc.env:WM_CKPT_DIR}
VAE_path: ${.wan_wm_hf_ckpt_path}/Wan2.2_VAE.pth
model_path: ${.wan_wm_hf_ckpt_path}/dit_model.safetensors
initial_image_path: ${.wan_wm_hf_ckpt_path}/dataset/

num_inference_steps: 5
condition_frame_length: 5
chunk: 8
num_frames: 13
image_size: [256, 256]

action_dim: 14
action_key: abs_action

reward_model:
  type: RoboTwinT5CrossAttn
  from_pretrained: /path/to/reward_model.pth
  t5_model_name: /path/to/t5-base
```

Launch training:

```bash
cd ${RLINF_ROOT}
WAN_PATH=${WAN_PATH} ROBOTWIN_PATH=${ROBOTWIN_PATH} \
  bash examples/embodiment/run_embodiment.sh \
  wan_robotwin_adjust_bottle_grpo_openpi_pi05 ALOHA
```

## 8. Use Wan as an HTTP Environment

Use `env_type: wan_wm_http` when the Wan model should run in a separate env-side
process. This is useful when the policy/reward/training host should be decoupled
from the heavy world-model process.

Architecture:

```text
Host RLinf policy
  -> WanHttpProxyEnv
  -> HTTP /reset and /chunk_step
  -> Wan HTTP server
  -> generated frames
  -> host-side reward model
  -> RL update
```

Start the env-side Wan server:

```bash
cd ${RLINF_ROOT}
WAN_PATH=${WAN_PATH} ROBOTWIN_PATH=${ROBOTWIN_PATH} \
  bash examples/embodiment/run_wan_http_server.sh \
  env/wan_robotwin_adjust_bottle_http
```

Start host-side RL training in another terminal:

```bash
cd ${RLINF_ROOT}
WAN_PATH=${WAN_PATH} ROBOTWIN_PATH=${ROBOTWIN_PATH} \
  bash examples/embodiment/run_embodiment.sh \
  wan_robotwin_adjust_bottle_http_grpo_openpi_pi05 ALOHA
```

For GRPO, keep `group_size > 1`. The minimal verified HTTP config uses:

```yaml
total_num_envs: 2
group_size: 2
rollout_epoch: 1
max_steps_per_rollout_epoch: 8
pipeline_stage_num: 1
```

## 9. Runtime Responsibilities

### Env-Side Server

The HTTP server owns the heavy Wan pipeline:

- loads `dit_model.safetensors`
- loads `Wan2.2_VAE.pth`
- receives initial world-model state from host via `/reset`
- generates the next frame chunk via `/chunk_step`
- returns generated observation frames

The server does not choose episodes and does not compute rewards.

### Host-Side Proxy Env

The proxy env owns RL-facing environment logic:

- samples reset episodes from `initial_image_path`
- builds initial condition frames and condition actions
- sends reset state to the Wan server
- sends policy actions to `/chunk_step`
- receives generated frames
- computes reward locally
- returns `obs`, `reward`, `termination`, `truncation`, and `info` to RLinf

## 10. Reward Model

The reward model is loaded on the host side, not on the HTTP Wan server side.

The current Wan env supports reward model types such as:

```yaml
reward_model:
  type: RoboTwinT5CrossAttn
  from_pretrained: /path/to/reward_model.pth
  t5_model_name: /path/to/t5-base
```

Other examples include:

```yaml
reward_model:
  type: LPIPSLastFrameRewardModel
  lpips_net: vgg
  reward_transform: one_minus
```

A custom reward model should expose a `compute_reward(...)` method compatible
with the selected env implementation.

## 11. Replacing Wan with Another World Model

To integrate a new world model, keep the RLinf-facing contract stable and
replace the model-specific internals.

### Data Adapter

Provide an encoder from raw robot trajectories to your training dataset. It
should define:

- raw observation keys
- raw action keys
- output frame tensor format
- output action tensor format
- train/validation split layout

For the Wan example:

```text
raw:
  observation/head_camera/rgb
  joint_action/vector

encoded:
  rgb.npy       [T, 1, H, W, 3]
  actions.npy   [T, 1, action_dim]
```

### Training Script

Provide a training entry point that accepts:

- dataset path
- base model path
- output path
- frame window length
- action dimension
- image size

The output should include a reusable checkpoint directory.

### RLinf Env Implementation

Implement an RLinf env with the same high-level methods:

```python
reset() -> tuple[dict, dict]
chunk_step(actions) -> tuple[list[dict], rewards, terminations, truncations, list[dict]]
```

The observation dict should include fields expected by the policy. For OpenPI
RoboTwin head-camera training, the important fields are:

```python
{
    "main_images": ...,          # [B, H, W, 3], uint8
    "wrist_images": None,
    "states": ...,               # [B, action_dim]
    "task_descriptions": ...,
}
```

### HTTP Server Contract

If using the HTTP split, implement:

```text
GET  /health
POST /reset
POST /chunk_step
```

The minimal payload contract is:

```text
/reset input:
  current_obs
  condition_action
  task_descriptions
  elapsed_steps

/chunk_step input:
  actions  # [B, action_chunk, action_dim]

/chunk_step output:
  current_obs
  elapsed_steps
```

The payload format can be replaced. The current minimal implementation uses
pickle + base64 in JSON for simplicity. For production, prefer msgpack, Arrow,
shared storage, or streaming tensors.

### Config Contract

Add a new env type and config:

```yaml
env_type: my_world_model_http
wm_env_type: robotwin
total_num_envs: 2
group_size: 2
chunk: 8
condition_frame_length: 5
num_frames: 13
action_dim: 14
http:
  server_url: http://127.0.0.1:18080
```

Then register it in `rlinf/envs/__init__.py` and make sure action processing in
`rlinf/envs/action_utils.py` returns the action format expected by the world
model.

## 12. Validation Checklist

Before running long training, verify:

- raw data can be encoded into train/val directories
- world-model checkpoint loads
- `num_frames == condition_frame_length + chunk`
- policy action dimension matches world-model action dimension
- env reset returns policy-compatible observations
- HTTP server `/health` returns OK
- server receives `/reset` and `/chunk_step`
- host-side reward is non-NaN
- one rollout step reaches `Global Step: 1/1`

Minimal HTTP smoke test:

```bash
bash examples/embodiment/run_wan_http_server.sh env/wan_robotwin_adjust_bottle_http
```

In another terminal:

```bash
bash examples/embodiment/run_embodiment.sh \
  wan_robotwin_adjust_bottle_http_grpo_openpi_pi05 ALOHA
```

Expected result:

```text
POST /reset 200 OK
POST /chunk_step 200 OK
Global Step: 1/1
```

## 13. Open-Source Notes

Before publishing:

- replace local absolute paths with environment variables or documented placeholders
- do not commit API keys or private credentials
- document where to download or place pretrained base models
- document the raw data schema, not only encoded tensors
- include one small synthetic or toy dataset for CI if possible
- keep world-model-specific code isolated behind env/server interfaces
