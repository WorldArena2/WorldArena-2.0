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

"""Host-side Wan HTTP proxy environment."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional, Union

import numpy as np
import torch

from rlinf.envs.world_model.http_payload import decode_payload, encode_payload
from rlinf.envs.world_model.world_model_wan_env import WanEnv


class _WanHttpClient:
    def __init__(self, server_url: str, timeout: float = 600.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def _post_payload(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps({"payload": encode_payload(payload)}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.server_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Wan HTTP server returned {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to connect to Wan HTTP server: {exc}") from exc
        return decode_payload(result["payload"])

    def reset(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._post_payload("/reset", state)

    def chunk_step(self, actions: Any) -> dict[str, Any]:
        return self._post_payload("/chunk_step", {"actions": actions})


class WanHttpProxyEnv(WanEnv):
    """WanEnv-compatible proxy that delegates frame generation to HTTP."""

    def _build_pipeline(self):
        # The host process keeps dataset/reward/training only; Wan inference runs
        # in the HTTP server process.
        return None

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info=None,
        record_metrics: bool = True,
    ):
        super().__init__(
            cfg=cfg,
            num_envs=num_envs,
            seed_offset=seed_offset,
            total_num_processes=total_num_processes,
            worker_info=worker_info,
            record_metrics=record_metrics,
        )
        http_cfg = getattr(cfg, "http", {})
        server_url = getattr(http_cfg, "server_url", "http://127.0.0.1:18080")
        timeout = float(getattr(http_cfg, "timeout", 600.0))
        self._http_client = _WanHttpClient(server_url=server_url, timeout=timeout)

    @torch.no_grad()
    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = {},
        episode_indices: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        obs, infos = super().reset(
            seed=seed, options=options, episode_indices=episode_indices
        )
        self._http_client.reset(
            {
                "current_obs": self.current_obs.detach().cpu(),
                "condition_action": self.condition_action.detach().cpu(),
                "task_descriptions": list(self.task_descriptions),
                "elapsed_steps": self.elapsed_steps,
            }
        )
        return obs, infos

    @torch.no_grad()
    def _infer_next_chunk_frames(self, actions):
        actions_to_send = (
            actions.detach().cpu() if isinstance(actions, torch.Tensor) else actions
        )
        result = self._http_client.chunk_step(actions_to_send)
        self.current_obs = result["current_obs"].to(self.device)

    def offload(self):
        if self._is_offloaded:
            return
        self.reward_model = self.reward_model.to("cpu")
        self.current_obs = self.current_obs.cpu() if self.current_obs is not None else None
        self.gt_last_frames = (
            self.gt_last_frames.cpu() if self.gt_last_frames is not None else None
        )
        self.prev_step_reward = self.prev_step_reward.cpu()
        self.reset_state_ids = self.reset_state_ids.cpu()
        if self.record_metrics:
            self.success_once = self.success_once.cpu()
            self.returns = self.returns.cpu()
        self._is_offloaded = True

    def onload(self):
        if not self._is_offloaded:
            return
        self.reward_model = self.reward_model.to(self.device)
        self.current_obs = self.current_obs.to(self.device)
        self.gt_last_frames = self.gt_last_frames.to(self.device)
        self.prev_step_reward = self.prev_step_reward.to(self.device)
        self.reset_state_ids = self.reset_state_ids.to(self.device)
        if self.record_metrics:
            self.success_once = self.success_once.to(self.device)
            self.returns = self.returns.to(self.device)
        self._is_offloaded = False
