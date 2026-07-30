# Track 3: Real-Robot Manipulation
**[中文文档](https://github.com/WorldArena2/WorldArena-2.0/blob/main/assets/track3_description_cn.md)**

**[English Document](https://github.com/WorldArena2/WorldArena-2.0/blob/main/assets/track3_description.md)**
## Overview

Track 3 is the **real-robot task track** of the WorldArena benchmark. Participating teams deploy policies on physical robots and are evaluated by the **success rate** of completing the target manipulation.

Track 3 currently covers **two robot platforms**:

| Platform | Embodiment | Control | Sensors |
|---|---|---|---|
| **AgileX** | Dual-arm cobot | Absolute **joint / qpos** (`joint_absolute`, 14D) | Head + left/right wrist cameras; optional Xense tactile |
| **Franka** | Franka Emika Panda single-arm | Absolute **endpose** (`end_pose_base`, 8D) | Third-person + wrist cameras |

We encourage participating World Models (WAMs) to attempt both **vision-only** and **vision-tactile** tasks on AgileX, and the Franka vision-only tasks, demonstrating generalizable real-world manipulation under visual and contact feedback.

Task data is available on Hugging Face: [WorldArena/WorldArena2.0](https://huggingface.co/datasets/WorldArena/WorldArena2.0)

---

## Task Categories

### 1. AgileX Vision-Only Tasks

These tasks rely primarily on visual perception (head camera and left/right wrist cameras). No tactile feedback is provided.

| Task | Description |
|---|---|
| Wipe Table | Grab a towel and wipe the table surface. |
| Pour Water | Pick up a container and pour water into a target cup. |
| Clean Tabletop | Remove objects from the table, classify them, and place them in designated locations. |
| Instruction-Following Clean Tabletop | Clean the tabletop following a natural-language instruction (e.g., "move the red block to the basket"). |
| Hand-Drip Coffee | Perform a hand-drip coffee procedure, including filter placement and pouring. |
| Fold Clothes | Fold a piece of clothing on the table. |
| Fold Cardboard Box | Fold a flat cardboard box into its assembled shape. |

### 2. AgileX Vision-Tactile Tasks

These tasks require contact-rich manipulation. Tactile feedback (force/torque and tactile images) is provided to enable robust execution.

| Task | Description |
|---|---|
| Pick Potato Chip | Pick up a fragile potato chip without crushing it and place it on a plate. |
| Peel Cucumber | Hold a cucumber and peel its skin with a peeler. |
| Insert Two-Pin Plug | Align and insert a two-pin plug into a socket. |

### 3. Franka Vision-Only Tasks

These tasks are collected and evaluated on the Franka Emika Panda single-arm platform with third-person and wrist cameras. No tactile feedback is provided.

| Task | Description |
|---|---|
| Wipe Table | Grab a towel / cloth and wipe the table surface. |
| Pour Water | Pick up a container and pour water into a target cup. |
| Clean Tabletop | Remove objects from the table, classify them, and place them in designated locations. |

---

## Dataset

The real-robot demonstration data for Track 3 is hosted on Hugging Face: [WorldArena/WorldArena2.0](https://huggingface.co/datasets/WorldArena/WorldArena2.0). The repository is organized by task, and each task contains human-collected real-robot episodes.

### Task Directories

| Directory | Task | Platform / Category |
|---|---|---|
| `clean_table` | Clean Tabletop | AgileX Vision-Only (+ Franka variant) |
| `clean_table_instruction_follow` | Instruction-Following Clean Tabletop | AgileX Vision-Only |
| `fold_box` | Fold Cardboard Box | AgileX Vision-Only |
| `fold_shirt` | Fold Clothes | AgileX Vision-Only |
| `pour_over_coffee` | Hand-Drip Coffee | AgileX Vision-Only |
| `pour_water` | Pour Water | AgileX Vision-Only (+ Franka variant) |
| `wipe_table` | Wipe Table | AgileX Vision-Only (+ Franka variant) |
| `insert` | Insert Two-Pin Plug | AgileX Vision-Tactile |
| `peel_cucumber` | Peel Cucumber | AgileX Vision-Tactile |
| `pick_potato_chip` | Pick Potato Chip | AgileX Vision-Tactile |

### Data Volume

On AgileX, each task contains approximately **100** demonstration episodes, except `pour_over_coffee` (Hand-Drip Coffee), which currently has **83** valid demonstrations. Franka demonstrations are provided for the three vision-only tasks listed above.

---

## Platform A: AgileX (Joint / qpos Control)

### Per-Episode Files (Vision-Only)

- `cam_high.mp4`: Video from the head-mounted / third-person camera.
- `cam_left_wrist.mp4`: Video from the left wrist camera.
- `cam_right_wrist.mp4`: Video from the right wrist camera.
- `episode.hdf5`: Runtime robot states and actions. The action / qpos stream is **14D joint positions**.
- `meta.json`: Episode metadata (task label, device, collection timestamp, etc.).

### Per-Episode Files (Vision-Tactile)

Same as vision-only, plus:

- `tactile_gripper_left_rectify.mp4`: Rectified tactile patch video for the left gripper.
- `tactile_gripper_right_rectify.mp4`: Rectified tactile patch video for the right gripper.
- `tactile_information.hdf5`: Tactile mechanics information (force, Marker2D, Mesh3D, etc.).

`episode.hdf5` for tactile episodes additionally stores tactile timestamps and attributes such as `tactile_enabled=True`.

### HDF5 Highlights

| Dataset | Shape (example) | Description |
|---|---|---|
| `observations/qpos` | `(T, 14)` | Dual-arm joint positions: left 7 + right 7 (6 joints + gripper each). |
| `action` | `(T, 14)` | Recorded joint action stream (same layout as qpos). |
| `observations/end_pose` | `(T, 14)` | Dual-arm end-effector poses (for reference; **not** the control command). |
| `observations/qvel` / `observations/effort` | `(T, 14)` | Joint velocity / effort. |

Typical recording rate: **30 FPS**.

### Action Format (AgileX)

Real-robot testing on AgileX uses **absolute joint positions** (`joint_absolute` / qpos). The action vector is **14D**:

```text
[left_j1..left_j6, left_gripper, right_j1..right_j6, right_gripper]
```

Policies must output joint-space actions; the real-robot system receives and executes them directly. Demonstration labels for control are in `observations/qpos` (and the aligned `action` stream).

---

## Platform B: Franka (Endpose Control)

### Per-Episode Files

- `third_person.mp4`: Third-person / head camera video (maps to `cam_high` at inference).
- `wrist.mp4`: Wrist camera video (maps to `cam_wrist` / `cam_left_wrist` at inference).
- `episode.hdf5`: Runtime robot states and endpose / joint trajectories.
- `robot_state.json`: Per-frame robot state log (ee pose, joint state, gripper width, etc.).
- `metadata.json`: Episode metadata (task label, FPS, frame count, duration, etc.).
- `camera_timestamps.json`: Per-camera timestamps and frame alignment.

### HDF5 Highlights

| Dataset | Shape (example) | Description |
|---|---|---|
| `observations/end_pose` | `(T, 16)` | End-effector pose trajectory. Active Franka arm occupies the first **8** dims; remaining dims are dual-arm-style padding. |
| `observations/qpos` | `(T, 14)` | Joint positions (padded; provided for reference). |
| `observations/qvel` / `observations/effort` | `(T, 14)` | Joint velocity / effort (padded). |
| `action` | `(T, 14)` | Recorded stream (padded). |
| `observation/camera/head` / `left` | `(T,)` | Encoded camera frames. |

Typical recording rate: **15 FPS**.

### Action Format (Franka)

Real-robot testing on Franka uses **absolute endpose** control (`end_pose_base`). The effective action vector is **8D**:

```text
[x, y, z, qw, qx, qy, qz, gripper]
```

- `x, y, z`: end-effector position in the **robot base frame** (meters)
- `qw, qx, qy, qz`: orientation quaternion (**wxyz**)
- `gripper`: gripper opening command

In HDF5, use `observations/end_pose[:, 0:8]` as the active Franka endpose labels.

---

## Evaluation Metric

Each task is evaluated by its **real-robot success rate**. A trial is considered successful if the robot completes the full task within the allowed number of steps and without safety intervention. The final ranking aggregates success rates across tasks, with optional normalization by task difficulty.

---

## Task Difficulty Ratings

We rate each task on a scale of **1 (easiest) to 10 (hardest)** based on the required precision, contact reasoning, deformable-object handling, and multi-step planning.

### AgileX Vision-Only Tasks

| Task | Difficulty (1-10) | Rationale |
|---|---|---|
| Wipe Table | 4 | Repetitive motion, large target area, low precision requirement. |
| Pour Water | 6 | Requires controlled tilting and visual tracking of liquid level. |
| Clean Tabletop | 5 | Object picking, classification, and relocation; moderate precision. |
| Instruction-Following Clean Tabletop | 6 | Adds language grounding on top of tabletop cleaning. |
| Hand-Drip Coffee | 9 | Multi-step fine manipulation with fragile objects and liquid handling. |
| Fold Clothes | 9 | Deformable-object manipulation; wrinkle handling and precise folding. |
| Fold Cardboard Box | 9 | Rigid-part assembly with precise creasing and corner alignment. |

### AgileX Vision-Tactile Tasks

| Task | Difficulty (1-10) | Rationale |
|---|---|---|
| Pick Potato Chip | 8 | Fragile and small object; requires gentle grip force and slip detection. |
| Peel Cucumber | 9 | Sustained contact, consistent force, and coordinated arm motion. |
| Insert Two-Pin Plug | 8 | Precision alignment plus contact-rich insertion. |

### Franka Vision-Only Tasks

| Task | Difficulty (1-10) | Rationale |
|---|---|---|
| Wipe Table | 4 | Repetitive motion, large target area, low precision requirement. |
| Pour Water | 6 | Requires controlled tilting and visual tracking of liquid level. |
| Clean Tabletop | 5 | Object picking, classification, and relocation; moderate precision. |

---

## Participation Guidance

- Teams may submit policies for any subset of tasks / platforms.
- We encourage WAMs to compete in **both vision-only and vision-tactile categories** on AgileX, as the latter tests contact-aware world modeling.
- **Control must match the platform**:
  - AgileX → **14D joint / qpos** (`joint_absolute`)
  - Franka → **8D endpose** (`end_pose_base`)
- Policies should be robust to real-world sensory noise, latency, and partial observability.
- See [policy_guide.md](https://github.com/WorldArena2/WorldArena-2.0/blob/main/assets/policy_guide.md) for the policy Worker interface, per-platform observation schemas, and deployment instructions.
