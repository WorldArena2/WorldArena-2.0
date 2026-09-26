# WorldArena External Policy Integration Guide

> This document is intended for participating teams that need to deploy a policy Worker on their own machine (Machine A) and connect it to our central scheduler (Machine B) and physical robot system (Machine C).

Track 3 supports **two platforms**. Action dimension and observation fields depend on the assigned robot:

| Platform | Action format | `action_dim` | Notes |
|---|---|---|---|
| **AgileX** dual-arm | `joint_absolute` (qpos) | **14** | Left 7 + right 7 |
| **Franka** single-arm | `end_pose_base` | **8** | `[x, y, z, qx, qy, qz, qw, gripper]` |

---

## 1. Architecture and Data Flow

```text
Machine A (Participant)        Machine B (Organizer)        Machine C (Organizer)
Policy Worker  ──HTTPS───→     Central Hub     ←──HTTPS───  Robot Worker
     ↑                              ↓                              ↑
Policy.infer()                  Scheduling                    Physical Robot
```

- **Machine A (policy side):** Provided by the participating team. It only needs outbound Internet access and **does not require a fixed public IP address**.
- **Machine B (scheduling side):** Provides model relay and centralized scheduling.
- **Machine C (robot side):** Executes physical robot actions and returns observations.

Machine A actively establishes an outbound HTTPS long-polling connection to our Hub, receives observations, and returns actions.

The policy process on Machine A may remain running for an extended period.

---

## 2. Policy Interface Definition

The external policy only needs to implement a Python class. The file path and class name may be chosen freely, but the following interface must be supported:

```python
from typing import Any, Dict
import numpy as np

class Policy:
    def __init__(self, config_path: str | None = None):
        """Initialize the model, load configuration files, and perform other setup."""
        ...

    def reset(self, reset_info: Dict[str, Any] | None = None) -> None:
        """Called once at the beginning of each episode."""
        ...

    def infer(self, new_obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Per-step inference interface.

        Args:
            new_obs: Observation dictionary converted by our system.
                     See Section 3 for the field definitions.

        Returns:
            A dict that must contain:
                - "actions": np.ndarray, shape (chunk, action_dim)

            It may optionally contain:
                - "policy_metadata": dict
                - "policy_timing": dict
                - "tactile_force": np.ndarray
        """
        ...
```

### 2.1 Return Field Description

| Field | Type | Required | Description |
|---|---|---|---|
| `actions` | `np.ndarray` | Yes | Action chunk with `shape=(chunk, action_dim)`. `action_dim` is **14** on AgileX and **8** on Franka. The policy may choose its own `chunk` length (common values: 20, 25). |

### 2.2 AgileX Action Semantics (`joint_absolute` / qpos)

```text
actions[t] = [left_j1..left_j6, left_gripper, right_j1..right_j6, right_gripper]  # 14D
```

Each row is an absolute joint-position target. Our system executes the returned joint targets sequentially. Demonstration labels for control are stored in `observations/qpos` in the dataset HDF5.

### 2.3 Franka Action Semantics (`end_pose_base`)

```text
actions[t] = [x, y, z, qx, qy, qz, qw, gripper]  # 8D
```

- Position `(x, y, z)` is in the **robot base frame** (meters)
- Orientation `(qx, qy, qz, qw)` is an **xyzw** quaternion
- `gripper` is the gripper opening command

Demonstration labels for control are stored in `observations/end_pose[:, 0:8]` in the dataset HDF5.

---

## 3. Observation `new_obs` Sent to the Policy

### 3.1 AgileX Top-Level Fields

```python
{
    "images": { ... },                   # Visual observations
    "state": np.ndarray,                 # 14D dual-arm joints: first 7 left, last 7 right
    "joint_qpos": np.ndarray,            # 14D dual-arm joints: first 7 left, last 7 right
    "right_arm_joint_state": np.ndarray, # 7D right-arm joint state
    "left_arm_joint_state": np.ndarray,  # 7D left-arm joint state
    "tactile": { ... },                  # Present only for vision-tactile tasks
    "prompt": str,                       # Natural-language task description
    "tactile_profile": str,              # Tactile profile label, e.g. "tactile_raw"
    "task_id": str,                      # Current task ID
}
```

### 3.2 AgileX `images` Field

```python
new_obs["images"] = {
    "cam_high": np.ndarray,          # uint8 HWC, overhead / third-person RGB
    "cam_wrist_left": np.ndarray,    # uint8 HWC, left wrist-camera RGB
    "cam_wrist_right": np.ndarray,   # uint8 HWC, right wrist-camera RGB
}
```

These correspond to dataset videos `cam_high.mp4`, `cam_left_wrist.mp4`, and `cam_right_wrist.mp4`.

### 3.3 AgileX `tactile` Field

**Vision-only tasks** and **vision-tactile tasks** use the same observation pipeline. The only difference is that vision-tactile tasks include tactile and force data in `new_obs["tactile"]`, whereas **vision-only tasks do not return any tactile information** (`new_obs["tactile"]` is absent).

For vision-tactile tasks:

```python
new_obs["tactile"] = {
    # Tactile images from Xense sensors
    "left_gripper": {
        "rectify": np.ndarray,   # uint8 HWC (RGB), tactile image from the left gripper pad
    },
    "right_gripper": {
        "rectify": np.ndarray,   # uint8 HWC (RGB), tactile image from the right gripper pad
    },

    # Wrist force sensing from force/torque sensors
    "left_wrist_force": {
        "wrench_6d": np.ndarray,  # float32 shape (6,), [Fx, Fy, Fz, Tx, Ty, Tz]
    },
    "right_wrist_force": {
        "wrench_6d": np.ndarray,  # float32 shape (6,)
    },
}
```

**Notes:**

- Access tactile images via `new_obs["tactile"]["left_gripper"]["rectify"]` / `["right_gripper"]["rectify"]`.
- Access force/torque via `wrench_6d`. Example:

  ```python
  left_force = new_obs["tactile"]["left_wrist_force"]["wrench_6d"]   # float32 (6,)
  right_force = new_obs["tactile"]["right_wrist_force"]["wrench_6d"] # float32 (6,)
  tactile_force = np.concatenate([left_force, right_force], axis=0)  # float32 (12,)
  ```

- Force measurements are provided as **raw sensor outputs**.
- During real-robot testing, the tactile information returned by our system matches the tactile information provided in the dataset (including Marker2D / Mesh3D related fields in `tactile_information.hdf5`).

### 3.4 Franka Top-Level Fields

Verified on the live remote-inference path for the Franka endpose deployment:

```python
{
    "images": { ... },                      # Visual observations
    "joint_qpos": np.ndarray,               # float32 (8,)  7 joints + gripper
    "joint_qpos_left": np.ndarray,          # float32 (8,)  alias of the active arm
    "left_arm_joint_state": np.ndarray,     # float32 (8,)  same as joint_qpos
    "left_end_pose": np.ndarray,            # float32 (7,)  [x, y, z, qx, qy, qz, qw]
    "state": np.ndarray,                    # float32 (32,) padded state buffer
    "first_frame": np.ndarray,              # uint8 HWC, copy of cam_high
    "prompt": str,                          # Natural-language task description
    "task_id": str,                         # Current task ID
}
```

Notes:

- Fields such as `joint_qpos_left` / `left_arm_joint_state` / `left_end_pose` refer to the active Franka arm (naming retained from the dual-arm interface).
- Prefer `joint_qpos` and `left_end_pose` for proprioception; `state` is a fixed-length padded buffer.
- There is **no tactile** field on current Franka vision-only tasks.

### 3.5 Franka `images` Field

```python
new_obs["images"] = {
    "cam_high": np.ndarray,        # uint8 (480, 640, 3), third-person / head RGB
    "cam_left_wrist": np.ndarray,  # uint8 (480, 640, 3), wrist RGB
    "cam_wrist": np.ndarray,       # uint8 (480, 640, 3), wrist RGB (same source as cam_left_wrist)
}
```

These correspond to dataset videos `third_person.mp4` and `wrist.mp4`. On the current Franka setup, `cam_left_wrist` and `cam_wrist` are the same wrist camera stream (duplicated for interface compatibility).

---

## 4. Example Policies

### 4.1 AgileX Dummy Policy (14D Joint)

```python
# dummy_policy_agilex.py
from typing import Any, Dict, Optional
import numpy as np


class Policy:
    def __init__(self, config_path: Optional[str] = None):
        self.chunk = 25
        self.action_dim = 14  # dual-arm qpos

    def reset(self, reset_info: Optional[Dict[str, Any]] = None) -> None:
        pass

    def infer(self, new_obs: Dict[str, Any]) -> Dict[str, Any]:
        # Example: hold current joints (replace with model inference).
        qpos = np.asarray(new_obs["joint_qpos"], dtype=np.float32).reshape(14)
        actions = np.repeat(qpos[None, :], self.chunk, axis=0)
        return {
            "actions": actions,
            "policy_metadata": {
                "policy_id": "dummy_agilex",
                "action_format": "joint_absolute",
            },
            "policy_timing": {"infer_ms": 0.0},
        }
```

### 4.2 Franka Dummy Policy (8D Endpose)

```python
# dummy_policy_franka.py
from typing import Any, Dict, Optional
import numpy as np


class Policy:
    def __init__(self, config_path: Optional[str] = None):
        self.chunk = 25
        self.action_dim = 8  # end_pose_base

    def reset(self, reset_info: Optional[Dict[str, Any]] = None) -> None:
        pass

    def infer(self, new_obs: Dict[str, Any]) -> Dict[str, Any]:
        # Example: hold current endpose (replace with model inference).
        pose7 = np.asarray(new_obs["left_end_pose"], dtype=np.float32).reshape(7)
        gripper = np.asarray(new_obs["joint_qpos"], dtype=np.float32).reshape(-1)[-1:]
        end_pose_8d = np.concatenate([pose7, gripper], axis=0)
        actions = np.repeat(end_pose_8d[None, :], self.chunk, axis=0)
        return {
            "actions": actions.astype(np.float32),
            "policy_metadata": {
                "policy_id": "dummy_franka",
                "action_format": "end_pose_base",
            },
            "policy_timing": {"infer_ms": 0.0},
        }
```

### 4.3 Starting the Example Policy

A startup script named `start_policy_worker.sh` is provided in the `track3_example/` directory:

```bash
cd track3_example
bash start_policy_worker.sh
```

Or manually:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python serve_policy_worldarena.py \
  ./dummy_policy.py \
  --hub-url <PENDING_HUB_GATEWAY_URL>/policy \
  --worker-key <PENDING_POLICY_ID>
```

`<PENDING_POLICY_ID>` must match the corresponding `benchmark_runner` configuration on our Machine B. Use the dummy policy that matches the assigned platform action format.

---

## 5. Hub Gateway Address

Before the official evaluation, we will provide each participating team with the specific Hub gateway address:

```text
<PENDING_HUB_GATEWAY_URL>/policy
```

---

## 6. Worker Key and Identity Matching

The `worker-key` used when Machine A registers with the Hub must exactly match the `--worker-key` value provided to the participating team by the organizers.

| Machine | Command / Configuration | Key |
|---|---|---|
| A | `--worker-key <PENDING_POLICY_ID>` | ID agreed upon by the participating team and the organizers |

---

## 7. Services That Must Remain Running

The participating team only needs to keep one process running on Machine A. From the `track3_example/` directory, run:

```bash
bash start_policy_worker.sh
```

Alternatively, run the following command manually:

```bash
python serve_policy_worldarena.py \
  ./dummy_policy.py \
  --hub-url <PENDING_HUB_GATEWAY_URL>/policy \
  --worker-key <PENDING_POLICY_ID>
```

This process will:

- Register with the Hub
- Maintain a heartbeat, approximately once every 15 seconds
- Long-poll for `infer` and `reset` tasks
- Return an `ActionPacket` after inference

Machine A **does not need to expose any inbound ports**. All communication is initiated outbound from Machine A to the Hub.

### 7.1 Local Self-Check: Starting a Dummy Hub for Machine B

To simulate Machine B locally for testing or debugging, start the local dummy Hub from the `track3_example/` directory:

```bash
bash start_hub.sh
```

The default listening addresses are:

- Policy port: `127.0.0.1:18000`
- Robot port: `127.0.0.1:19000`

Then, in another terminal, connect the policy Worker to the local Hub:

```bash
export HUB_GATEWAY_URL=http://127.0.0.1:18000
export POLICY_ID=dummy_local
bash start_policy_worker.sh
```

---

## 8. Network Requirements

- Machine A must be able to access the public Internet through HTTPS. The specific domain name will be provided before the official evaluation.
- The Volcengine gateway **does not support WebSocket Upgrade**, so the HTTP Hub long-polling mode must be used.

---

## 9. Environment Setup

External teams may install their model dependencies on top of the communication environment provided by the organizers.

### 9.1 Creating the Python Environment

```bash
conda create -n real_eval python=3.10 -y
conda activate real_eval
```

### 9.2 Installing PyTorch

PyTorch should be installed according to the CUDA version available on the participating team's machine. The following command is only an example for CUDA 12.4:

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

### 9.3 Installing Other Dependencies

A `requirements.txt` file exported from the evaluation environment is provided in the `track3_example/` directory. Run the following command from that directory:

```bash
pip install -r requirements.txt
```

---

## 10. Frequently Asked Questions

### Q1: What action coordinate system is used?

It depends on the platform:

- **AgileX:** 14D absolute joint positions (`joint_absolute` / qpos), left arm then right arm.
- **Franka:** 8D absolute endpose (`end_pose_base`): `[x, y, z, qx, qy, qz, qw, gripper]` in the robot base frame (quaternion **xyzw**).

The chunk length may be chosen by the policy or agreed upon before evaluation.

### Q2: What happens if the policy crashes or disconnects?

The Hub detects a lost heartbeat, and the organizer-side scheduler terminates the current episode. After the participating team restarts the Worker, it automatically registers with the Hub again.

### Q3: How do dataset files map to inference observations?

| Platform | Dataset cameras | Inference image keys | Control labels |
|---|---|---|---|
| AgileX | `cam_high` / `cam_left_wrist` / `cam_right_wrist` | `cam_high` / `cam_wrist_left` / `cam_wrist_right` | `observations/qpos` `(T, 14)` |
| Franka | `third_person` / `wrist` | `cam_high` / `cam_left_wrist` & `cam_wrist` | `observations/end_pose[:, 0:8]` |

---

## 11. Minimal Integration Checklist

- [ ] `POLICY_ID` has been confirmed; it will be provided before the official evaluation
- [ ] The Hub gateway address has been confirmed; it will be provided before the official evaluation
- [ ] Machine A can access the public Internet through HTTPS
- [ ] The assigned platform is confirmed (**AgileX 14D qpos** or **Franka 8D endpose**)
- [ ] The policy implements `Policy.infer(new_obs)` and returns `{"actions": (chunk, action_dim) ndarray}` with the correct `action_dim`
- [ ] The evaluation time window has been agreed upon with the organizers

---
