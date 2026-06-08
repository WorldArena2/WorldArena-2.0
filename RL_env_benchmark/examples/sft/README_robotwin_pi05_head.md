# Single-View pi0.5 SFT on RoboTwin (head camera only)

End-to-end pipeline for fine-tuning pi0.5 separately on two RoboTwin
tasks — `adjust_bottle` and `click_bell` — with **14-dim dual-arm
actions** and a **single (head) camera** view.

The head view is the only visual input; pi0.5 keeps its 3-slot image
interface, and the two wrist slots are zeroed + masked out by
`AlohaInputs` at train time.

## Layout

Artifacts added for this workflow:

```
rlinf/models/embodiment/openpi/dataconfig/
    robotwin_aloha_head_dataconfig.py     # new LeRobotAlohaHeadOnlyDataConfig
    __init__.py                            # registers pi05_aloha_robotwin_head
examples/sft/
    scripts/convert_robotwin_hdf5_to_lerobot.py
    config/robotwin_sft_openpi_pi05_adjust_bottle.yaml
    config/robotwin_sft_openpi_pi05_click_bell.yaml
```

## 1. Convert RoboTwin HDF5 → LeRobot v2.1

Source format (per task):

```
/manifold-obs/wzl/vla_robotwin_4k_320/ref/<task>/demo_clean/
    data/episode*.hdf5        # joint_action/vector (T,14), observation/<cam>/rgb JPEG bytes
    instructions/episode*.json # {"seen": [...], "unseen": [...]}
```

Each episode's `observation/head_camera/rgb` is decoded into frames and
written as `observation.images.cam_high`; `joint_action/vector` becomes
`action`; `observation.state` is written as a 14-dim zero vector (the
source has no independent proprio, and `pi05_aloha_robotwin_head` uses
`discrete_state_input=True` so it does not consume the state). One
instruction is sampled from `seen` per episode and stored as the
LeRobot `task`.

Run (uses the project venv):

```bash
/your/path//RLinf/.venv_robotwin/bin/python \
  /your/path//RLinf/examples/sft/scripts/convert_robotwin_hdf5_to_lerobot.py \
    --src_root /manifold-obs/wzl/vla_robotwin_4k_320/ref \
    --out_root /your/path//RLinf/datasets/lerobot \
    --tasks adjust_bottle click_bell \
    --repo_id_prefix rlinf/robotwin_headcam \
    --fps 25
```

This produces two independent LeRobot datasets:

```
/your/path//RLinf/datasets/lerobot/
    rlinf/robotwin_headcam_adjust_bottle/
        meta/{info.json,episodes.jsonl,episodes_stats.jsonl,tasks.jsonl}
        data/chunk-000/episode_*.parquet
        videos/chunk-000/observation.images.cam_high/episode_*.mp4
    rlinf/robotwin_headcam_click_bell/
        ...
```

Useful flags:

- `--max_episodes N` — convert only the first N episodes (smoke test).
- `--seed` — controls which `seen` prompt is sampled per episode.
- `--tasks` — pass a subset, e.g. `--tasks click_bell` to redo one task.

The script refuses to write into an existing output dataset directory.
Delete (or pick a new `--repo_id_prefix`) to re-encode.

## 2. Fix norm stats and pi0.5 weights

`pi05_aloha_robotwin_head` inherits the same
`assets_dir="checkpoints/torch/pi05_aloha_robotwin/assets"` default.
At training time that path is overridden by `actor.model.model_path`
(see `_override_with_model_path` in
`rlinf/models/embodiment/openpi/dataconfig/__init__.py`), so make sure
the pi0.5 checkpoint directory you point at contains the
`norm_stats.json` file openpi expects.

Two options:

1. **Reuse the released RoboTwin norm stats** — point `model_path` at
   the checkpoint distributed with pi0.5 for RoboTwin. This is fine if
   your action/state distributions are close to the original dataset.
2. **Compute norm stats on the new dataset** — recommended. Run
   `openpi/scripts/compute_norm_stats.py --config-name pi05_aloha_robotwin_head`
   after pointing openpi at your converted dataset, and drop the
   resulting `norm_stats.json` next to the pi0.5 weights.

## 3. Train (per task)

Two yamls, one per task, already wired to the correct `repo_id`:

- `examples/sft/config/robotwin_sft_openpi_pi05_adjust_bottle.yaml`
- `examples/sft/config/robotwin_sft_openpi_pi05_click_bell.yaml`

Both reference:

```yaml
actor:
  model:
    model_path: "/path/to/pi05-model"      # TODO: replace
    action_dim: 14
    openpi:
      config_name: "pi05_aloha_robotwin_head"
      num_images_in_input: 1
  openpi_data:
    repo_id: "rlinf/robotwin_headcam_<task>"
data:
  train_data_paths: "/your/path//RLinf/datasets/lerobot"
```

`data.train_data_paths` is exported as `HF_LEROBOT_HOME` by
`examples/sft/train_vla_sft.py`, so the loader resolves the dataset at
`<train_data_paths>/<repo_id>/`.

Launch:

```bash
cd /your/path//RLinf
bash examples/sft/run_vla_sft.sh robotwin_sft_openpi_pi05_adjust_bottle
bash examples/sft/run_vla_sft.sh robotwin_sft_openpi_pi05_click_bell
```

Logs go to `logs/<timestamp>/run_embodiment.log`, tensorboard events to
the `log_path` set in the yaml.

## Knobs worth knowing

- **Single-view semantics.** The pi0.5 backbone always has 3 image
  slots (`256 tokens × 3`). `AlohaInputs` detects the missing
  `cam_left_wrist` / `cam_right_wrist` keys, fills them with zero
  images, and sets `image_mask=False`. Attention over the zero tokens
  is masked, so the model effectively trains on the head view only.
- **Delta actions.** `extra_delta_transform=True` in the head-only
  config applies the RoboTwin delta mask
  `[True]*6 + [False] + [True]*6 + [False]` — arms are trained as
  deltas, grippers as absolute. If your `joint_action/vector` is not
  in the Aloha layout (left arm 6 + left gripper + right arm 6 +
  right gripper), either change the mask in
  `robotwin_aloha_head_dataconfig.py` or set
  `actor.openpi_data.extra_delta_transform: false` in the yaml.
- **`adapt_to_pi`.** Default `False` for RoboTwin (joint/gripper
  convention already matches pi0). Set to `True` only if your data
  uses the Aloha real-robot convention.
- **`observation.state`.** Currently zero-filled. If you have real
  proprio, change the writer in
  `convert_robotwin_hdf5_to_lerobot.py:convert_task` and recompute
  norm stats.
- **`num_images_in_input`.** Only matters for the critic's value head;
  ignored in pure SFT (`add_value_head=False`, `use_critic_model=False`).
  Set to `1` for consistency.
