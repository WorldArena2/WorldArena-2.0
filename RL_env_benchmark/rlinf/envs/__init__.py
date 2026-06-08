# Copyright 2025 The RLinf Authors.
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

from enum import Enum


class SupportedEnvType(Enum):
    MANISKILL = "maniskill"
    LIBERO = "libero"
    ROBOTWIN = "robotwin"
    ISAACLAB = "isaaclab"
    METAWORLD = "metaworld"
    BEHAVIOR = "behavior"
    CALVIN = "calvin"
    ROBOCASA = "robocasa"
    REALWORLD = "realworld"
    FRANKASIM = "frankasim"
    HABITAT = "habitat"
    OPENSORAWM = "opensora_wm"
    WANWM = "wan_wm"
    NN_WORLD_MODEL = "nn_world_model"
    GENIE_WM = "genie_wm"
    CTRLWORLD_WM = "ctrlworld_wm"
    COSMOSWM = "cosmos_wm"
    IVIDEOGPTWM = "ivideogpt_wm"
    IRASIMWM = "irasim_wm"


def get_env_cls(env_type: str, env_cfg=None):
    """
    Get environment class based on environment type.

    Args:
        env_type: Type of environment (e.g., "maniskill", "libero", "isaaclab", etc.)
        env_cfg: Optional environment configuration. Required for "isaaclab" environment type.

    Returns:
        Environment class corresponding to the environment type.
    """

    env_type = SupportedEnvType(env_type)

    if env_type == SupportedEnvType.MANISKILL:
        if env_cfg.get("enable_offload", False):
            from rlinf.envs.maniskill.maniskill_offload_env import ManiskillOffloadEnv

            return ManiskillOffloadEnv
        else:
            from rlinf.envs.maniskill.maniskill_env import ManiskillEnv

            return ManiskillEnv
    elif env_type == SupportedEnvType.LIBERO:
        from rlinf.envs.libero.libero_env import LiberoEnv

        return LiberoEnv
    elif env_type == SupportedEnvType.ROBOTWIN:
        from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv

        return RoboTwinEnv
    elif env_type == SupportedEnvType.ISAACLAB:
        from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS

        if env_cfg is None:
            raise ValueError(
                "env_cfg is required for isaaclab environment type. "
                "Please provide env_cfg.init_params.id to select the task."
            )

        task_id = env_cfg.init_params.id
        assert task_id in REGISTER_ISAACLAB_ENVS, (
            f"Task type {task_id} has not been registered! "
            f"Available tasks: {list(REGISTER_ISAACLAB_ENVS.keys())}"
        )
        return REGISTER_ISAACLAB_ENVS[task_id]
    elif env_type == SupportedEnvType.METAWORLD:
        from rlinf.envs.metaworld.metaworld_env import MetaWorldEnv

        return MetaWorldEnv
    elif env_type == SupportedEnvType.BEHAVIOR:
        from rlinf.envs.behavior.behavior_env import BehaviorEnv

        return BehaviorEnv
    elif env_type == SupportedEnvType.CALVIN:
        from rlinf.envs.calvin.calvin_gym_env import CalvinEnv

        return CalvinEnv
    elif env_type == SupportedEnvType.ROBOCASA:
        from rlinf.envs.robocasa.robocasa_env import RobocasaEnv

        return RobocasaEnv
    elif env_type == SupportedEnvType.REALWORLD:
        from rlinf.envs.realworld.realworld_env import RealWorldEnv

        return RealWorldEnv
    elif env_type == SupportedEnvType.HABITAT:
        from rlinf.envs.habitat.habitat_env import HabitatEnv

        return HabitatEnv
    elif env_type == SupportedEnvType.FRANKASIM:
        from rlinf.envs.frankasim.frankasim_env import FrankaSimEnv

        return FrankaSimEnv
    elif env_type == SupportedEnvType.OPENSORAWM:
        from rlinf.envs.world_model.world_model_opensora_env import OpenSoraEnv

        return OpenSoraEnv
    elif env_type == SupportedEnvType.WANWM:
        from rlinf.envs.world_model.world_model_wan_env import WanEnv

        return WanEnv
    elif env_type == SupportedEnvType.GENIE_WM:
        from rlinf.envs.world_model.world_model_genie_env import GenieEnv

        return GenieEnv
    elif env_type == SupportedEnvType.CTRLWORLD_WM:
        from rlinf.envs.world_model.world_model_ctrlworld_env import CtrlWorldEnv

        return CtrlWorldEnv
    elif env_type == SupportedEnvType.COSMOSWM:
        from rlinf.envs.world_model.world_model_cosmos_env import CosmosEnv

        return CosmosEnv
    elif env_type == SupportedEnvType.IVIDEOGPTWM:
        from rlinf.envs.world_model.world_model_ivideogpt_env import IVideoGPTEnv

        return IVideoGPTEnv
    elif env_type == SupportedEnvType.IRASIMWM:
        from rlinf.envs.world_model.world_model_irasim_env import IRASIMEnv

        return IRASIMEnv
    else:
        raise NotImplementedError(f"Environment type {env_type} not implemented")
