"""Minimal runnable WorldArena Policy example (no model weights).

Implements the A-side ``Policy`` interface expected by
``serve_policy_worldarena``:

- ``infer(new_obs)`` returns ``{"actions": np.ndarray}``
- supports smoke / zero-action mode via env ``POLICY_SMOKE=1`` (default)
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import numpy as np


class Policy:
    """Zero-action / smoke Policy for A-side integration tests."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        # Dual-arm joint absolute: left 7 + right 7.
        self.action_dim = int(os.environ.get('POLICY_ACTION_DIM', '14'))
        self.chunk_size = int(os.environ.get('POLICY_CHUNK_SIZE', '20'))
        self.smoke = os.environ.get('POLICY_SMOKE', '1').strip().lower() not in ('0', 'false', 'no', 'off')
        self.action_format = os.environ.get('POLICY_ACTION_FORMAT', 'joint')
        print(
            f'[SmokePolicy] init action_dim={self.action_dim} chunk_size={self.chunk_size} '
            f'action_format={self.action_format!r} smoke={self.smoke}'
        )

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            'policy_id': 'SmokePolicy',
            'action_format': self.action_format,
            'action_dim': self.action_dim,
            'chunk_size': self.chunk_size,
            'smoke': self.smoke,
        }

    def reset(self, reset_info: Optional[Dict[str, Any]] = None) -> None:
        """Called by the framework at the start of each episode."""
        _ = reset_info

    def infer(self, new_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Run one inference step.

        ``new_obs`` is produced by the legacy bridge from ``ObservationPacket``.
        Task instruction from the B-side task suite is available as
        ``new_obs["prompt"]`` (same text as ``ObservationPacket.context.task_instruction``).
        """
        t0 = time.time()
        prompt = ''
        if isinstance(new_obs, dict):
            prompt = str(new_obs.get('prompt') or '')

        # Smoke / zero actions: no model weights required.
        actions = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)

        infer_ms = (time.time() - t0) * 1000.0
        meta = dict(self.metadata)
        if prompt:
            meta['last_prompt'] = prompt[:120]
        return {
            'actions': actions,
            'policy_timing': {'infer_ms': infer_ms},
            'policy_metadata': meta,
        }


if __name__ == '__main__':
    policy = Policy()
    sample = {
        'images': {'cam_high': np.zeros((240, 320, 3), dtype=np.uint8)},
        'state': np.zeros((14,), dtype=np.float32),
        'prompt': 'pick up the cup',
        'task_id': 'demo_task',
    }
    out = policy.infer(sample)
    print('actions.shape=', out['actions'].shape, 'prompt=', out['policy_metadata'].get('last_prompt'))
