# WorldArena External Policy Integration Guide

> This document is intended for participating teams that need to deploy a policy Worker on their own machine (Machine A) and connect it to our central scheduler (Machine B) and physical robot system (Machine C).

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
| `actions` | `np.ndarray` | Yes | Action chunk with `shape=(chunk, action_dim)`. Example: `chunk=25`, `action_dim=14` for a dual-arm robot, with the left-arm dimensions followed by the right-arm dimensions. The policy may choose its own `chunk` length. |

**Action semantics:** `actions[t]` represents the target joint position at step `t` using the `joint_absolute` action representation. Our system executes the returned actions sequentially. The action chunk length `chunk` may be chosen by the policy, with common values including 20 and 25. The action dimension `action_dim` is 14 for the dual-arm robot: the first 7 dimensions correspond to the left arm, and the last 7 dimensions correspond to the right arm.

---

## 3. Observation `new_obs` Sent to the Policy

### 3.1 Top-Level Fields

```python
{
    "images": { ... },                   # Visual observations
    "state": np.ndarray,                 # 14D dual-arm joints: first 7 left, last 7 right
    "joint_qpos": np.ndarray,            # 14D dual-arm joints: first 7 left, last 7 right
    "right_arm_joint_state": np.ndarray, # 7D right-arm joint state
    "left_arm_joint_state": np.ndarray,  # 7D left-arm joint state
    "tactile": { ... },                  # Tactile / force observations
    "prompt": str,                       # Natural-language task description
    "tactile_profile": str,              # Tactile profile label, e.g. "tactile_raw"
    "task_id": str,                      # Current task ID
}
```

### 3.2 `images` Field

```python
new_obs["images"] = {
    "cam_high": np.ndarray,          # uint8 HWC, overhead / third-person RGB image
    "cam_wrist_right": np.ndarray,   # uint8 HWC, right wrist-camera RGB image
    "cam_wrist_left": np.ndarray,    # uint8 HWC, left wrist-camera RGB image
}
```

### 3.3 `tactile` Field

**Vision-only tasks** and **vision-tactile tasks** use the same observation pipeline. The only difference is that vision-tactile tasks include tactile and force data in `new_obs["tactile"]`, whereas **vision-only tasks do not return any tactile information**, meaning that `new_obs["tactile"]` is absent.

For vision-tactile tasks, the `tactile` field contains two types of signals. Note that **tactile sensing is provided only on the right gripper**:

```python
new_obs["tactile"] = {
    # Tactile images from Xense sensors
    "left_gripper": {
        "rectify": np.ndarray,   # uint8 HWC, tactile image from the left gripper pad
    },
    "right_gripper": {
        "rectify": np.ndarray,   # uint8 HWC, tactile image from the right gripper pad
    },

    # Wrist force sensing from force/torque sensors
    "left_wrist_force": {
        "wrench_6d": np.ndarray,  # float32 shape (6,), [Fx, Fy, Fz, Tx, Ty, Tz], left sensor
    },
    "right_wrist_force": {
        "wrench_6d": np.ndarray,  # float32 shape (6,), right sensor
    },
}
```

**Notes:**

- `left_gripper` and `right_gripper` provide **tactile images** through the `rectify` field. For example:

  ```python
  left_tactile_image = new_obs["tactile"]["left_gripper"]["rectify"]   # uint8 HWC
  right_tactile_image = new_obs["tactile"]["right_gripper"]["rectify"] # uint8 HWC
  ```

- `left_wrist_force` and `right_wrist_force` provide combined force/torque measurements through `wrench_6d`. For example:

  ```python
  left_force = new_obs["tactile"]["left_wrist_force"]["wrench_6d"]   # float32 (6,)
  right_force = new_obs["tactile"]["right_wrist_force"]["wrench_6d"] # float32 (6,)
  tactile_force = np.concatenate([left_force, right_force], axis=0)  # float32 (12,)
  ```

- Force measurements are provided as **raw sensor outputs**.

---

## 4. Example: Dummy Policy with Fixed Actions

The following is a minimal runnable example policy. It ignores the observation input and directly returns a fixed action sequence. External teams may replace this logic with their own model inference implementation.

```python
# dummy_policy.py
from typing import Any, Dict, Optional
import numpy as np


class Policy:
    def __init__(self, config_path: Optional[str] = None):
        # Action chunk length and action dimension.
        # Participating teams may choose their own chunk length.
        self.chunk = 25
        self.action_dim = 14  # Dual-arm joints: first 7 left, last 7 right

    def reset(self, reset_info: Optional[Dict[str, Any]] = None) -> None:
        pass

    def infer(self, new_obs: Dict[str, Any]) -> Dict[str, Any]:
        # Example: return a fixed action sequence.
        # Replace this with model inference for an actual submission.
        actions = np.zeros((self.chunk, self.action_dim), dtype=np.float32)
        actions[:, 0] = 0.1   # Left-arm joint 0
        actions[:, 7] = 0.1   # Right-arm joint 0

        return {
            "actions": actions,
            "policy_metadata": {"policy_id": "dummy"},
            "policy_timing": {"infer_ms": 0.0},
        }
```

### 4.1 Starting the Example Policy

A startup script named `start_policy_worker.sh` is provided in the `challenge_pre/` directory. The policy Worker can be started directly using relative paths:

```bash
cd challenge_pre
bash start_policy_worker.sh
```

The script contains the following commands, which may also be executed manually:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python serve_policy_worldarena.py \
  ./dummy_policy.py \
  --hub-url <PENDING_HUB_GATEWAY_URL>/policy \
  --worker-key <PENDING_POLICY_ID>
```

`<PENDING_POLICY_ID>` must match the corresponding `benchmark_runner` configuration on our Machine B.

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

The participating team only needs to keep one process running on Machine A. From the `challenge_pre/` directory, run:

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

To simulate Machine B locally for testing or debugging, start the local dummy Hub from the `challenge_pre/` directory:

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

A `requirements.txt` file exported from the ViTAL environment is provided in the `challenge_pre/` directory. Run the following command from that directory:

```bash
pip install -r requirements.txt
```

---

## 10. Frequently Asked Questions

### Q1: What action coordinate system is used?

The current tasks use **14D dual-arm absolute joint positions** with the `joint_absolute` action representation. The first 7 dimensions correspond to the left arm, and the last 7 dimensions correspond to the right arm. The action dimension is 14, while the chunk length may be chosen by the policy or agreed upon before evaluation. The example uses a chunk length of 25.

### Q2: What happens if the policy crashes or disconnects?

The Hub detects a lost heartbeat, and the organizer-side scheduler terminates the current episode. After the participating team restarts the Worker, it automatically registers with the Hub again.

---

## 11. Minimal Integration Checklist

- [ ] `POLICY_ID` has been confirmed; it will be provided before the official evaluation
- [ ] The Hub gateway address has been confirmed; it will be provided before the official evaluation
- [ ] Machine A can access the public Internet through HTTPS
- [ ] The policy implements `Policy.infer(new_obs)` and returns `{"actions": (chunk, 14) ndarray}`; the chunk length may be chosen by the policy or agreed upon before evaluation
- [ ] The evaluation time window has been agreed upon with the organizers

---
