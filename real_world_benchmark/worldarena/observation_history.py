"""Observation history helpers: role-based multi-frame requirements and validation."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

from real_world_benchmark.worldarena.protocol import CAMERA_ROLE_GLOBAL
from real_world_benchmark.worldarena.schema import CameraObservation, ObservationPacket

HISTORY_ROLE_ROBOT_STATE = 'robot_state'


@dataclasses.dataclass(frozen=True)
class ObservationHistoryConfig:
    """Benchmark / policy observation history requirements (Notion §5-B)."""

    camera_roles: Dict[str, int] = dataclasses.field(default_factory=dict)
    tactile_roles: Dict[str, int] = dataclasses.field(default_factory=dict)
    robot_state_len: int = 1
    history_interval_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            'robot_state_len': self.robot_state_len,
            'history_interval_s': self.history_interval_s,
        }
        if self.camera_roles:
            out['camera_roles'] = dict(self.camera_roles)
        if self.tactile_roles:
            out['tactile_roles'] = dict(self.tactile_roles)
        return out

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'ObservationHistoryConfig':
        if not data:
            return cls()
        camera_roles = {str(k): int(v) for k, v in (data.get('camera_roles') or {}).items()}
        tactile_roles = {str(k): int(v) for k, v in (data.get('tactile_roles') or {}).items()}
        return cls(
            camera_roles=camera_roles,
            tactile_roles=tactile_roles,
            robot_state_len=int(data.get('robot_state_len', 1)),
            history_interval_s=float(data.get('history_interval_s', 0.0)),
        )

    @classmethod
    def for_use_history(cls, *, global_depth: int = 5) -> 'ObservationHistoryConfig':
        """WMA-style default: history on ``global`` camera only."""
        if global_depth <= 1:
            return cls()
        return cls(camera_roles={CAMERA_ROLE_GLOBAL: global_depth})


@dataclasses.dataclass(frozen=True)
class ObservationHistoryCapabilities:
    """Embodiment / robot worker history capability declaration."""

    max_history_per_role: int = 1
    supported_camera_history_roles: List[str] = dataclasses.field(default_factory=list)
    supported_tactile_history_roles: List[str] = dataclasses.field(default_factory=list)
    supports_robot_state_history: bool = False
    history_interval_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            'max_history_per_role': self.max_history_per_role,
            'supports_robot_state_history': self.supports_robot_state_history,
            'history_interval_s': self.history_interval_s,
        }
        if self.supported_camera_history_roles:
            out['supported_camera_history_roles'] = list(self.supported_camera_history_roles)
        if self.supported_tactile_history_roles:
            out['supported_tactile_history_roles'] = list(self.supported_tactile_history_roles)
        return out

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'ObservationHistoryCapabilities':
        if not data:
            return cls()
        return cls(
            max_history_per_role=int(data.get('max_history_per_role', 1)),
            supported_camera_history_roles=[str(x) for x in data.get('supported_camera_history_roles', [])],
            supported_tactile_history_roles=[str(x) for x in data.get('supported_tactile_history_roles', [])],
            supports_robot_state_history=bool(data.get('supports_robot_state_history', False)),
            history_interval_s=float(data.get('history_interval_s', 0.0)),
        )


def camera_history_depth(config: ObservationHistoryConfig, camera_role: str) -> int:
    depth = int(config.camera_roles.get(camera_role, 1))
    return max(1, depth)


def tactile_history_depth(config: ObservationHistoryConfig, tactile_role: str) -> int:
    depth = int(config.tactile_roles.get(tactile_role, 1))
    return max(1, depth)


def validate_observation_history_capabilities(
    config: ObservationHistoryConfig,
    capabilities: ObservationHistoryCapabilities,
) -> None:
    """Ensure benchmark requirements fit within robot worker history capabilities."""
    for role, depth in config.camera_roles.items():
        if depth <= 1:
            continue
        if capabilities.max_history_per_role < depth:
            raise ValueError(
                f'camera_role={role!r} requires history depth {depth}, '
                f'robot max_history_per_role={capabilities.max_history_per_role}'
            )
        if capabilities.supported_camera_history_roles and role not in capabilities.supported_camera_history_roles:
            raise ValueError(f'camera_role={role!r} history not supported by robot worker')
    for role, depth in config.tactile_roles.items():
        if depth <= 1:
            continue
        if capabilities.max_history_per_role < depth:
            raise ValueError(
                f'tactile_role={role!r} requires history depth {depth}, '
                f'robot max_history_per_role={capabilities.max_history_per_role}. '
                'Restart the C-side robot worker with FASTVTAM_TACTILE_HISTORY_LEN/WORLDARENA_TACTILE_HISTORY_LEN '
                'set to the required depth, or use a tactile embodiment preset that advertises enough history.'
            )
    if config.robot_state_len > 1 and not capabilities.supports_robot_state_history:
        raise ValueError('robot_state history requested but embodiment does not support it')


def camera_observation_frame_count(cam: CameraObservation) -> int:
    history_len = len(cam.frame_history_bytes or [])
    return history_len + (1 if cam.frame_bytes else 0)


def stack_camera_frames(cam: CameraObservation):
    """Return all frames oldest-first as uint8 HWC arrays."""
    from real_world_benchmark.worldarena.image_codec import decode_image_bytes

    frames: List[Any] = []
    for blob in cam.frame_history_bytes or []:
        if blob is None:
            continue
        h, w = cam.height, cam.width
        if h <= 0 or w <= 0:
            continue
        frames.append(
            decode_image_bytes(
                blob,
                encoding=cam.encoding,
                shape=[h, w, 3],
                dtype='uint8',
            )
        )
    if cam.frame_bytes is not None and cam.height > 0 and cam.width > 0:
        frames.append(
            decode_image_bytes(
                cam.frame_bytes,
                encoding=cam.encoding,
                shape=[cam.height, cam.width, 3],
                dtype='uint8',
            )
        )
    return frames


def observation_history_actual(packet: ObservationPacket) -> Dict[str, int]:
    """Report actual history lengths returned in a packet."""
    actual: Dict[str, int] = {}
    for cam in packet.camera_observations:
        actual[f'camera:{cam.camera_role}'] = camera_observation_frame_count(cam)
    for tac in packet.tactile_observations:
        history_fields = len(tac.fields_history or [])
        actual[f'tactile:{tac.tactile_role}'] = history_fields + (1 if tac.fields else 0)
    if packet.robot_state_history:
        actual[HISTORY_ROLE_ROBOT_STATE] = len(packet.robot_state_history) + 1
    else:
        actual[HISTORY_ROLE_ROBOT_STATE] = 1
    return actual
