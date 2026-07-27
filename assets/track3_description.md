# Track 3: Real-Robot Manipulation
**[中文文档](https://github.com/WorldArena2/WorldArena-2.0/blob/main/assets/track3_description_cn.md)**

**[English Document](https://github.com/WorldArena2/WorldArena-2.0/blob/main/assets/track3_description.md)**
## Overview

Track 3 is the **real-robot task track** of the WorldArena benchmark. Participating teams deploy their policies on a real AgileX dual-arm robot equipped with head/wrist cameras and optional Xense tactile sensors. Tasks are evaluated by the **success rate** of completing the target manipulation in real-world execution.

We encourage participating World Models (WAMs) to attempt both **vision-only** and **vision-tactile** tasks, demonstrating generalizable real-world manipulation under visual and contact feedback.

Task data is available on Hugging Face: [WorldArena/WorldArena2.0](https://huggingface.co/datasets/WorldArena/WorldArena2.0)

---

## Task Categories

### 1. Vision-Only Tasks

These tasks rely primarily on visual perception (head camera and wrist camera). No tactile feedback is provided; only head-camera and wrist-camera feedback is available.

| Task | Description |
|---|---|
| Wipe Table | Grab a towel and wipe the table surface. |
| Pour Water | Pick up a container and pour water into a target cup. |
| Clean Tabletop | Remove objects from the table, classify them, and place them in designated locations. |
| Instruction-Following Clean Tabletop | Clean the tabletop following a natural-language instruction (e.g., "move the red block to the basket"). |
| Hand-Drip Coffee | Perform a hand-drip coffee procedure, including filter placement and pouring. |
| Fold Clothes | Fold a piece of clothing on the table. |
| Fold Cardboard Box | Fold a flat cardboard box into its assembled shape. |

### 2. Vision-Tactile Tasks

These tasks require contact-rich manipulation. Tactile feedback (force/torque and tactile images) is provided to enable robust execution.

| Task | Description |
|---|---|
| Pick Potato Chip | Pick up a fragile potato chip without crushing it and place it on a plate. |
| Peel Cucumber | Hold a cucumber and peel its skin with a peeler. |
| Insert Two-Pin Plug | Align and insert a two-pin plug into a socket. |

---

## Dataset

The real-robot demonstration data for Track 3 is hosted on Hugging Face: [WorldArena/WorldArena2.0](https://huggingface.co/datasets/WorldArena/WorldArena2.0). The repository is organized by task, and each task contains human-collected real-robot episodes.

### Task Directories

| Directory | Task | Category |
|---|---|---|
| `clean_table` | Clean Tabletop | Vision-Only |
| `clean_table_instruction_follow` | Instruction-Following Clean Tabletop | Vision-Only |
| `fold_box` | Fold Cardboard Box | Vision-Only |
| `fold_shirt` | Fold Clothes | Vision-Only |
| `pour_over_coffee` | Hand-Drip Coffee | Vision-Only |
| `pour_water` | Pour Water | Vision-Only |
| `wipe_table` | Wipe Table | Vision-Only |
| `insert` | Insert Two-Pin Plug | Vision-Tactile |
| `peel_cucumber` | Peel Cucumber | Vision-Tactile |
| `pick_potato_chip` | Pick Potato Chip | Vision-Tactile |

### Data Volume

Each task contains approximately **100** demonstration episodes, except `pour_over_coffee` (Hand-Drip Coffee), which currently has **83** valid demonstrations.

### Per-Episode File Description

Each episode is stored in its own directory and contains the following files:

**Vision-Only Tasks**

- `cam_high.mp4`: Video from the head-mounted camera.
- `cam_left_wrist.mp4`: Video from the left wrist camera.
- `cam_right_wrist.mp4`: Video from the right wrist camera.
- `episode.hdf5`: Runtime robot states and actions, where the action component consists of **joint actions**; during replay and real-robot testing, the robot directly receives and executes these joint actions.
- `meta.json`: Episode metadata (e.g., task label, collection timestamp).

**Vision-Tactile Tasks**

Vision-tactile tasks provide the same files as vision-only tasks plus the following:

- `tactile_gripper_left_rectify.mp4`: Rectified tactile patch video for the left gripper.
- `tactile_gripper_right_rectify.mp4`: Rectified tactile patch video for the right gripper.
- `tactile_information.hdf5`: Tactile mechanics information, including force/torque and 3D deformation data.

### Action Format

All tasks use **joint actions** as the final control command during real-robot testing. Policy outputs must be in the robot's joint action space, which the real-robot system receives and executes directly.

---

## Evaluation Metric

Each task is evaluated by its **real-robot success rate**. A trial is considered successful if the robot completes the full task within the allowed number of steps and without safety intervention. The final ranking aggregates success rates across tasks, with optional normalization by task difficulty.

---

## Task Difficulty Ratings

We rate each task on a scale of **1 (easiest) to 10 (hardest)** based on the required precision, contact reasoning, deformable-object handling, and multi-step planning.

### Vision-Only Tasks

| Task | Difficulty (1-10) | Rationale |
|---|---|---|
| Wipe Table | 4 | Repetitive motion, large target area, low precision requirement. |
| Pour Water | 6 | Requires controlled tilting and visual tracking of liquid level. |
| Clean Tabletop | 5 | Object picking, classification, and relocation; moderate precision. |
| Instruction-Following Clean Tabletop | 6 | Adds language grounding on top of tabletop cleaning. |
| Hand-Drip Coffee | 9 | Multi-step fine manipulation with fragile objects and liquid handling. |
| Fold Clothes | 9 | Deformable-object manipulation; wrinkle handling and precise folding. |
| Fold Cardboard Box | 9 | Rigid-part assembly with precise creasing and corner alignment. |

### Vision-Tactile Tasks

| Task | Difficulty (1-10) | Rationale |
|---|---|---|
| Pick Potato Chip | 8 | Fragile and small object; requires gentle grip force and slip detection. |
| Peel Cucumber | 9 | Sustained contact, consistent force, and coordinated arm motion. |
| Insert Two-Pin Plug | 8 | Precision alignment plus contact-rich insertion. |

---

## Participation Guidance

- Teams may submit policies for any subset of tasks.
- We encourage WAMs to compete in **both vision-only and vision-tactile categories**, as the latter tests contact-aware world modeling.
- During real-robot testing, the robot receives and executes policy outputs as **joint actions**.
- Policies should be robust to real-world sensory noise, latency, and partial observability.
- See [policy_guide.md](https://github.com/WorldArena2/WorldArena-2.0/blob/main/assets/policy_guide.md) in this folder for the policy Worker interface and deployment instructions.
