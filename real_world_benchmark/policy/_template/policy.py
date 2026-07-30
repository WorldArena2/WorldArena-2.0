"""WorldArena Policy wrapper template.

Copy this directory to ``policy/<YourPolicyName>/`` and replace the TODO
sections with your own model loading / inference logic. The public interface
must remain compatible with ``serve_policy_worldarena``.

For a zero-weight runnable smoke example, see ``examples/policy_template``.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import numpy as np


class Policy:
    """Template policy matching the WorldArena A-side interface.

    Expected ``new_obs`` fields (typical dual-arm joint mode):
    - ``images['cam_high']`` / ``cam_left_wrist`` / ``cam_right_wrist``: uint8 HWC
    - ``state`` / ``joint_qpos`` / per-arm joint keys: float arrays
    - ``prompt``: task instruction from the B-side task suite
    - ``tactile``: optional, only for tactile tasks

    Environment variables:
    - ``POLICY_ACTION_DIM``: action vector width (default 14 = dual-arm joints)
    - ``POLICY_CHUNK_SIZE``: action chunk length (default 20)
    - ``POLICY_ACTION_FORMAT``: ``joint`` | ``eef6d_single`` | ``eef6d``
    - ``POLICY_CONTROL_ARM``: ``right`` | ``left`` (for single-arm formats)
    """

    def __init__(self, config_path: Optional[str] = None):
        self.action_format = os.environ.get('POLICY_ACTION_FORMAT', 'joint')
        self.control_arm = os.environ.get('POLICY_CONTROL_ARM', 'right')
        self.action_dim = int(os.environ.get('POLICY_ACTION_DIM', '14'))
        self.chunk_size = int(os.environ.get('POLICY_CHUNK_SIZE', '20'))
        self.ckpt_dir = os.environ.get('POLICY_CKPT_DIR', '')
        self.config_path = config_path or os.environ.get('POLICY_CONFIG', '')

        # TODO: load your model, tokenizer, norm stats, etc.
        print(
            f'[TemplatePolicy] init: action_format={self.action_format}, '
            f'control_arm={self.control_arm}, action_dim={self.action_dim}, '
            f'chunk_size={self.chunk_size}'
        )

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            'policy_id': 'TemplatePolicy',
            'action_format': self.action_format,
            'action_dim': self.action_dim,
            'control_arm': self.control_arm,
            'chunk_size': self.chunk_size,
        }

    def reset(self, reset_info: Optional[Dict[str, Any]] = None) -> None:
        """Called at the start of each episode. Optional."""
        _ = reset_info

    def infer(self, new_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Run one inference step.

        Returns:
            ``{'actions': np.ndarray (chunk_size, action_dim),
               'policy_timing': {'infer_ms': float},
               'policy_metadata': dict}``
        """
        t0 = time.time()
        # Task instruction from B-side task suite:
        #   new_obs['prompt']  ==  ObservationPacket.context.task_instruction
        _prompt = str((new_obs or {}).get('prompt') or '')

        # TODO: replace the placeholder below with your real inference.
        actions = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)

        infer_ms = (time.time() - t0) * 1000.0
        return {
            'actions': actions,
            'policy_timing': {'infer_ms': infer_ms},
            'policy_metadata': self.metadata,
        }
