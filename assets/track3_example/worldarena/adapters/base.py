"""Robot adapter base class for WorldArena 2.0."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from worldarena.schema import ActionPacket, ObservationPacket, SessionContext


class RobotAdapter(ABC):
    """Bridge between a robot private protocol and the WorldArena canonical schema."""

    adapter_id: str = ''
    adapter_version: str = ''
    embodiment_id: str = ''
    embodiment_type: str = ''

    @abstractmethod
    def private_observation_to_canonical(
        self,
        private_obs: Dict[str, Any],
        *,
        context: Optional[SessionContext] = None,
        step_index: int = 0,
    ) -> ObservationPacket:
        """Convert a vendor-specific observation dict to an ObservationPacket."""

    @abstractmethod
    def canonical_action_to_private(
        self,
        action_packet: ActionPacket,
        private_obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Convert an ActionPacket to vendor-specific action parameters."""

    @abstractmethod
    def apply_action(self, action_packet: ActionPacket, private_obs: Dict[str, Any]) -> None:
        """Apply a canonical action packet to the robot."""

    def reset(self, reset_info: Optional[Dict[str, Any]] = None) -> None:
        """Optional hook for episode reset on the robot side."""

    def metadata(self) -> Dict[str, Any]:
        meta = {
            'adapter_id': self.adapter_id,
            'adapter_version': self.adapter_version,
            'embodiment_id': self.embodiment_id,
            'embodiment_type': self.embodiment_type,
        }
        tactile_caps = self.tactile_capabilities()
        if tactile_caps:
            meta.update(tactile_caps)
        return meta
