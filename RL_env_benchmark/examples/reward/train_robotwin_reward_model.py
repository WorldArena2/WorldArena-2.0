# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training script for the RoboTwin T5 + cross-attention reward model.

Pre-process the raw HDF5 data first::

    python examples/reward/preprocess_robotwin_reward_dataset.py \\
        --data-root /manifold-obs/wzl/vla_robotwin_4k_320/10radiodata_10000/adjust_bottle/demo_clean \\
        --output-dir logs/robotwin_reward_data

Then launch training::

    python examples/reward/train_robotwin_reward_model.py

Override dataset paths from the command line::

    python examples/reward/train_robotwin_reward_model.py \\
        data.train_data_paths=logs/robotwin_reward_data/train.pt \\
        data.val_data_paths=logs/robotwin_reward_data/val.pt

Override T5 model path (for offline environments)::

    python examples/reward/train_robotwin_reward_model.py \\
        actor.model.t5_model_name=/path/to/local/t5-base
"""

import json

import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.sft_runner import SFTRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.reward.reward_worker import FSDPTextCondRewardWorker

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="robotwin_reward_training",
)
def main(cfg) -> None:
    """Entry point – builds cluster, worker group, and runs the SFT loop."""
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create the text-conditioned reward worker group.
    actor_placement = component_placement.get_strategy("actor")
    actor_group = FSDPTextCondRewardWorker.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    # SFTRunner drives the training loop (train + periodic eval + checkpoint).
    runner = SFTRunner(
        cfg=cfg,
        actor=actor_group,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
