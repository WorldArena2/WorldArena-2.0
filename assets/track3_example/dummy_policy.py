"""Dummy policy example for WorldArena Track 3.

This policy ignores the observation and outputs a fixed action sequence. It is
intended only as a minimal runnable template for external participants to
replace with their own model.
"""

from typing import Any, Dict, Optional

import numpy as np


class Policy:
    """Minimal WorldArena policy interface example."""

    def __init__(self, config_path: Optional[str] = None):
        # Action chunk length and dual-arm action dimension.
        # These are examples; participants may choose their own chunk length.
        self.chunk = 25
        self.action_dim = 14  # dual-arm joints: first 7 dims = left arm, last 7 dims = right arm

    def reset(self, reset_info: Optional[Dict[str, Any]] = None) -> None:
        """Called at the start of each episode."""
        pass

    def infer(self, new_obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run one inference step.

        Args:
            new_obs: Observation dict provided by WorldArena. See policy_guide.md
                for the full field specification.

        Returns:
            dict with key "actions": np.ndarray of shape (chunk, action_dim).
        """
        # Example: output a fixed action sequence.
        # Replace this with your model inference.
        actions = np.zeros((self.chunk, self.action_dim), dtype=np.float32)
        actions[:, 0] = 0.1  # placeholder motion on left arm joint 0
        actions[:, 7] = 0.1  # placeholder motion on right arm joint 0

        return {
            "actions": actions,
            "policy_metadata": {"policy_id": "dummy"},
            "policy_timing": {"infer_ms": 0.0},
        }
