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

"""HTTP server that hosts Wan world-model inference for one synchronous env."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from hydra import compose
from hydra.core.global_hydra import GlobalHydra
from hydra.initialize import initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from rlinf.envs.world_model.http_payload import decode_payload, encode_payload
from rlinf.envs.world_model.world_model_wan_env import WanEnv


class _NoRewardModel(torch.nn.Module):
    """Placeholder reward model so WanEnv can initialize on the server side."""

    def forward(self, *args, **kwargs):  # pragma: no cover - not used by server
        raise RuntimeError("Wan HTTP server does not compute rewards.")


class WanHttpServerCore(WanEnv):
    """WanEnv variant that only performs world-model frame generation."""

    def _load_reward_model(self):
        return _NoRewardModel()

    @torch.no_grad()
    def reset_from_host_state(self, state: dict[str, Any]) -> dict[str, Any]:
        current_obs = state["current_obs"].to(self.device)
        condition_action = state["condition_action"].to(self.device)

        self.current_obs = current_obs
        self.condition_action = condition_action
        self.task_descriptions = list(state.get("task_descriptions", [""] * self.num_envs))
        self.elapsed_steps = int(state.get("elapsed_steps", 0))
        self._is_start = False

        for env_idx in range(self.num_envs):
            self.image_queue[env_idx] = [
                self.current_obs[env_idx, :, 0, t_idx : t_idx + 1, :, :].detach().cpu()
                for t_idx in range(self.condition_frame_length)
            ]

        self._reset_metrics()
        return {"obs": self._wrap_obs()}

    @torch.no_grad()
    def chunk_step_without_reward(self, actions: Any) -> dict[str, Any]:
        self.onload()
        self._infer_next_chunk_frames(actions)
        self.elapsed_steps += self.chunk
        return {
            "current_obs": self.current_obs.detach().cpu(),
            "elapsed_steps": self.elapsed_steps,
        }


def _load_env_cfg(config_name: str, overrides: list[str]) -> DictConfig:
    repo_root = Path(__file__).resolve().parents[3]
    config_dir = Path(
        os.environ.get("EMBODIED_CONFIG_DIR", repo_root / "examples/embodiment/config")
    ).resolve()
    os.environ.setdefault("EMBODIED_PATH", str(repo_root / "examples/embodiment"))
    os.environ.setdefault("REPO_PATH", str(repo_root))

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = compose(config_name=config_name, overrides=overrides)

    # The server accepts either a plain env config or a top-level training config.
    if "env" in cfg and "train" in cfg.env:
        return cfg.env.train
    if "env" in cfg and "env_type" in cfg.env:
        return cfg.env
    return cfg


def build_app(env: WanHttpServerCore):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - dependency setup guard
        raise ImportError(
            "Wan HTTP server requires fastapi. Install requirements/embodied/models/wan.txt."
        ) from exc

    app = FastAPI(title="RLinf Wan HTTP Env")

    @app.get("/health")
    def health():
        return {"status": "ok", "num_envs": env.num_envs}

    @app.post("/reset")
    def reset(body: dict[str, str]):
        try:
            state = decode_payload(body["payload"])
            result = env.reset_from_host_state(state)
            return {"payload": encode_payload(result)}
        except Exception as exc:  # pragma: no cover - exercised in integration
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/chunk_step")
    def chunk_step(body: dict[str, str]):
        try:
            request = decode_payload(body["payload"])
            result = env.chunk_step_without_reward(request["actions"])
            return {"payload": encode_payload(result)}
        except Exception as exc:  # pragma: no cover - exercised in integration
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def main():
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - dependency setup guard
        raise ImportError(
            "Wan HTTP server requires uvicorn. Install requirements/embodied/models/wan.txt."
        ) from exc

    parser = argparse.ArgumentParser(description="Run a Wan world-model HTTP server.")
    parser.add_argument("--config-name", default="env/wan_robotwin_adjust_bottle_http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    env_cfg = _load_env_cfg(args.config_name, args.overrides)
    env_cfg = OmegaConf.create(OmegaConf.to_container(env_cfg, resolve=True))
    env_cfg.total_num_envs = int(env_cfg.get("total_num_envs", 1) or 1)
    env = WanHttpServerCore(
        cfg=env_cfg,
        num_envs=env_cfg.total_num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    uvicorn.run(build_app(env), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
