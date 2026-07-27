"""Embodiment profile: active arms, cameras, and tactile topology for robot workers."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

from worldarena.protocol import (
    ARM_ID_LEFT,
    ARM_ID_RIGHT,
    CAMERA_ROLE_GLOBAL,
    CAMERA_ROLE_LEFT_WRIST,
    CAMERA_ROLE_RIGHT_WRIST,
    TACTILE_PROFILE_DERIVED,
    TACTILE_PROFILE_RAW,
    TACTILE_ROLE_LEFT_GRIPPER,
    TACTILE_ROLE_RIGHT_GRIPPER,
)

# Well-known preset names (serve_robot --embodiment-preset).
PRESET_DUAL_ARM = 'dual_arm'
PRESET_SINGLE_ARM_TACTILE = 'single_arm_tactile'
PRESET_SINGLE_ARM_TACTILE_RIGHT = 'single_arm_tactile_right'
PRESET_DUAL_ARM_TACTILE = 'dual_arm_tactile'

EMBODIMENT_TYPE_DUAL_ARM = 'dual_arm'
EMBODIMENT_TYPE_SINGLE_ARM = 'single_arm'

GRIPPER_TYPE_PARALLEL = 'parallel'  # 2 tactile pads per gripper (non-dexhand)
GRIPPER_TYPE_DEXHAND = 'dexhand'

PADS_PER_GRIPPER_PARALLEL = 2


def default_tactile_roles(
    active_arms: List[str],
    *,
    pads_per_gripper: int = PADS_PER_GRIPPER_PARALLEL,
    vital_legacy_names: bool = False,
) -> List[str]:
    """Compute tactile roles: count = len(active_arms) * pads_per_gripper.

    For a single active arm with ``vital_legacy_names=True`` (ViTAL / UniVTAC),
    the two gripper pads map to ``left_gripper`` and ``right_gripper`` (pad sides,
    not robot arms).

    Otherwise roles are ``{arm_id}_gripper_pad_{a|b}`` per arm.
    """
    if not active_arms:
        return []
    if vital_legacy_names and len(active_arms) == 1:
        if pads_per_gripper != 2:
            raise ValueError('vital_legacy_names requires pads_per_gripper=2')
        return [TACTILE_ROLE_LEFT_GRIPPER, TACTILE_ROLE_RIGHT_GRIPPER]

    pad_suffixes = ['a', 'b', 'c', 'd']
    roles: List[str] = []
    for arm_id in active_arms:
        for pad_idx in range(pads_per_gripper):
            suffix = pad_suffixes[pad_idx] if pad_idx < len(pad_suffixes) else str(pad_idx)
            roles.append(f'{arm_id}_gripper_pad_{suffix}')
    return roles


def default_camera_roles(active_arms: List[str], *, include_global: bool = True) -> List[str]:
    roles: List[str] = []
    if include_global:
        roles.append(CAMERA_ROLE_GLOBAL)
    if ARM_ID_LEFT in active_arms:
        roles.append(CAMERA_ROLE_LEFT_WRIST)
    if ARM_ID_RIGHT in active_arms:
        roles.append(CAMERA_ROLE_RIGHT_WRIST)
    return roles


@dataclasses.dataclass(frozen=True)
class EmbodimentProfile:
    """Declarative robot morphology for adapter + robot worker metadata."""

    embodiment_id: str
    embodiment_type: str = EMBODIMENT_TYPE_DUAL_ARM
    active_arms: List[str] = dataclasses.field(default_factory=lambda: [ARM_ID_LEFT, ARM_ID_RIGHT])
    adapter_id: str = 'manifold_msg.agilex'
    adapter_version: str = 'adapter.manifold_msg.1.3.0'

    tactile_enabled: bool = False
    tactile_profile: str = TACTILE_PROFILE_DERIVED
    gripper_type: str = GRIPPER_TYPE_PARALLEL
    pads_per_gripper: int = PADS_PER_GRIPPER_PARALLEL
    tactile_roles: List[str] = dataclasses.field(default_factory=list)
    vital_legacy_tactile_names: bool = False

    camera_roles: List[str] = dataclasses.field(default_factory=list)
    hold_inactive_arms: bool = True
    max_observation_history: int = 1

    def __post_init__(self) -> None:
        if self.gripper_type == GRIPPER_TYPE_PARALLEL and self.pads_per_gripper != PADS_PER_GRIPPER_PARALLEL:
            raise ValueError('parallel gripper requires pads_per_gripper=2')
        if self.tactile_enabled and self.expected_tactile_count() == 0:
            raise ValueError('tactile_enabled requires at least one active arm')

    @property
    def inactive_arms(self) -> List[str]:
        all_arms = [ARM_ID_LEFT, ARM_ID_RIGHT]
        return [arm for arm in all_arms if arm not in self.active_arms]

    def expected_tactile_count(self) -> int:
        if not self.tactile_enabled:
            return 0
        return len(self.active_arms) * self.pads_per_gripper

    def resolved_tactile_roles(self) -> List[str]:
        if self.tactile_roles:
            if len(self.tactile_roles) != self.expected_tactile_count():
                raise ValueError(
                    f'tactile_roles length {len(self.tactile_roles)} != '
                    f'expected {self.expected_tactile_count()} '
                    f'(active_arms={self.active_arms}, pads_per_gripper={self.pads_per_gripper})'
                )
            return list(self.tactile_roles)
        return default_tactile_roles(
            self.active_arms,
            pads_per_gripper=self.pads_per_gripper,
            vital_legacy_names=self.vital_legacy_tactile_names,
        )

    def resolved_camera_roles(self) -> List[str]:
        if self.camera_roles:
            return list(self.camera_roles)
        return default_camera_roles(self.active_arms)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            'embodiment_id': self.embodiment_id,
            'embodiment_type': self.embodiment_type,
            'active_arms': list(self.active_arms),
            'adapter_id': self.adapter_id,
            'adapter_version': self.adapter_version,
            'tactile_enabled': self.tactile_enabled,
            'gripper_type': self.gripper_type,
            'pads_per_gripper': self.pads_per_gripper,
            'hold_inactive_arms': self.hold_inactive_arms,
            'max_observation_history': self.max_observation_history,
        }
        if self.tactile_enabled:
            out['tactile_profile'] = self.tactile_profile
            out['tactile_roles'] = self.resolved_tactile_roles()
            out['expected_tactile_count'] = self.expected_tactile_count()
        if self.camera_roles:
            out['camera_roles'] = list(self.camera_roles)
        else:
            out['camera_roles'] = self.resolved_camera_roles()
        if self.vital_legacy_tactile_names:
            out['vital_legacy_tactile_names'] = True
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbodimentProfile':
        active_arms = [str(x) for x in data.get('active_arms', [ARM_ID_LEFT, ARM_ID_RIGHT])]
        tactile_roles = [str(x) for x in data.get('tactile_roles', [])]
        camera_roles = [str(x) for x in data.get('camera_roles', [])]
        return cls(
            embodiment_id=str(data.get('embodiment_id', 'agilex_dual_arm')),
            embodiment_type=str(data.get('embodiment_type', EMBODIMENT_TYPE_DUAL_ARM)),
            active_arms=active_arms,
            adapter_id=str(data.get('adapter_id', 'manifold_msg.agilex')),
            adapter_version=str(data.get('adapter_version', 'adapter.manifold_msg.1.3.0')),
            tactile_enabled=bool(data.get('tactile_enabled', False)),
            tactile_profile=str(data.get('tactile_profile', TACTILE_PROFILE_DERIVED)),
            gripper_type=str(data.get('gripper_type', GRIPPER_TYPE_PARALLEL)),
            pads_per_gripper=int(data.get('pads_per_gripper', PADS_PER_GRIPPER_PARALLEL)),
            tactile_roles=tactile_roles,
            vital_legacy_tactile_names=bool(data.get('vital_legacy_tactile_names', False)),
            camera_roles=camera_roles,
            hold_inactive_arms=bool(data.get('hold_inactive_arms', True)),
            max_observation_history=int(data.get('max_observation_history', 1)),
        )


def preset_profile(name: str, *, embodiment_id: Optional[str] = None) -> EmbodimentProfile:
    """Built-in morphology presets for Manifold AgileX."""
    if name == PRESET_DUAL_ARM:
        return EmbodimentProfile(
            embodiment_id=embodiment_id or 'agilex_dual_arm',
            embodiment_type=EMBODIMENT_TYPE_DUAL_ARM,
            active_arms=[ARM_ID_LEFT, ARM_ID_RIGHT],
            adapter_id='manifold_msg.agilex_dual_arm',
            tactile_enabled=False,
            hold_inactive_arms=False,
            max_observation_history=5,
        )
    if name == PRESET_SINGLE_ARM_TACTILE:
        return EmbodimentProfile(
            embodiment_id=embodiment_id or 'agilex_single_arm_tactile',
            embodiment_type=EMBODIMENT_TYPE_SINGLE_ARM,
            active_arms=[ARM_ID_LEFT],
            adapter_id='manifold_msg.agilex_single_arm_tactile',
            tactile_enabled=True,
            tactile_profile=TACTILE_PROFILE_RAW,
            vital_legacy_tactile_names=True,
            hold_inactive_arms=True,
        )
    if name == PRESET_SINGLE_ARM_TACTILE_RIGHT:
        return EmbodimentProfile(
            embodiment_id=embodiment_id or 'agilex_single_arm_tactile_right',
            embodiment_type=EMBODIMENT_TYPE_SINGLE_ARM,
            active_arms=[ARM_ID_RIGHT],
            adapter_id='manifold_msg.agilex_single_arm_tactile_right',
            tactile_enabled=True,
            tactile_profile=TACTILE_PROFILE_RAW,
            vital_legacy_tactile_names=True,
            hold_inactive_arms=True,
        )
    if name == PRESET_DUAL_ARM_TACTILE:
        return EmbodimentProfile(
            embodiment_id=embodiment_id or 'agilex_dual_arm_tactile',
            embodiment_type=EMBODIMENT_TYPE_DUAL_ARM,
            active_arms=[ARM_ID_LEFT, ARM_ID_RIGHT],
            adapter_id='manifold_msg.agilex_dual_arm_tactile',
            tactile_enabled=True,
            tactile_profile=TACTILE_PROFILE_DERIVED,
            vital_legacy_tactile_names=False,
            hold_inactive_arms=False,
        )
    raise ValueError(f'Unknown embodiment preset: {name!r}')


def build_profile_from_cli(
    *,
    embodiment_preset: Optional[str],
    embodiment_id: Optional[str],
    embodiment_type: Optional[str],
    active_arms: Optional[List[str]],
    enable_tactile: bool,
    tactile_profile: str,
    tactile_roles: Optional[List[str]],
    gripper_type: str,
    hold_inactive_arms: Optional[bool],
    vital_legacy_tactile_names: bool,
) -> EmbodimentProfile:
    """Resolve CLI flags into a concrete EmbodimentProfile."""
    if embodiment_preset:
        profile = preset_profile(embodiment_preset, embodiment_id=embodiment_id)
    else:
        arms = active_arms or [ARM_ID_LEFT, ARM_ID_RIGHT]
        etype = embodiment_type or (
            EMBODIMENT_TYPE_SINGLE_ARM if len(arms) == 1 else EMBODIMENT_TYPE_DUAL_ARM
        )
        profile = EmbodimentProfile(
            embodiment_id=embodiment_id or f'agilex_{etype}',
            embodiment_type=etype,
            active_arms=arms,
            tactile_enabled=enable_tactile,
            tactile_profile=tactile_profile,
            gripper_type=gripper_type,
            vital_legacy_tactile_names=vital_legacy_tactile_names,
            hold_inactive_arms=hold_inactive_arms if hold_inactive_arms is not None else len(arms) == 1,
        )

    if embodiment_id and embodiment_preset:
        profile = dataclasses.replace(profile, embodiment_id=embodiment_id)
    if embodiment_type and not embodiment_preset:
        profile = dataclasses.replace(profile, embodiment_type=embodiment_type)
    if active_arms and not embodiment_preset:
        profile = dataclasses.replace(profile, active_arms=active_arms)
    if enable_tactile and not profile.tactile_enabled:
        profile = dataclasses.replace(profile, tactile_enabled=True, tactile_profile=tactile_profile)
    if not enable_tactile and embodiment_preset is None:
        profile = dataclasses.replace(profile, tactile_enabled=False)
    if tactile_roles:
        profile = dataclasses.replace(profile, tactile_roles=tactile_roles)
    if vital_legacy_tactile_names:
        profile = dataclasses.replace(profile, vital_legacy_tactile_names=True)
    if hold_inactive_arms is not None:
        profile = dataclasses.replace(profile, hold_inactive_arms=hold_inactive_arms)
    if tactile_profile and profile.tactile_enabled:
        profile = dataclasses.replace(profile, tactile_profile=tactile_profile)

    # Validate resolved roles/count early.
    if profile.tactile_enabled:
        profile.resolved_tactile_roles()
    return profile
