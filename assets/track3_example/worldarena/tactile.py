"""Tactile modality helpers: profile parsing, validation, and ndarray bridging."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Iterable, List, Optional, Set

import numpy as np

from worldarena.protocol import (
    TACTILE_FIELD_FORCE_XYZ,
    TACTILE_FIELD_RECTIFY_BGR,
    TACTILE_FIELD_WRENCH_6D,
    TACTILE_PROFILE_DERIVED,
    TACTILE_PROFILE_RAW,
    TACTILE_PROFILE_RAW_PLUS_DERIVED,
)
from worldarena.schema import TactileField, TactileObservation


@dataclasses.dataclass(frozen=True)
class TactileBenchmarkConfig:
    """Session / benchmark tactile requirements (Notion §2.3)."""

    tactile_required: bool = False
    tactile_profile: str = TACTILE_PROFILE_DERIVED
    tactile_roles: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            'tactile_required': self.tactile_required,
            'tactile_profile': self.tactile_profile,
        }
        if self.tactile_roles:
            out['tactile_roles'] = list(self.tactile_roles)
        return out

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'TactileBenchmarkConfig':
        if not data:
            return cls()
        roles = data.get('tactile_roles') or []
        return cls(
            tactile_required=bool(data.get('tactile_required', False)),
            tactile_profile=str(data.get('tactile_profile', TACTILE_PROFILE_DERIVED)),
            tactile_roles=[str(x) for x in roles],
        )


@dataclasses.dataclass(frozen=True)
class TactileCapabilities:
    """Embodiment / robot worker tactile capability declaration."""

    supported_tactile_roles: List[str] = dataclasses.field(default_factory=list)
    supported_tactile_profiles: List[str] = dataclasses.field(default_factory=list)
    default_tactile_profile: str = TACTILE_PROFILE_DERIVED
    tactile_sensor_vendor: str = ''
    tactile_sdk_version: str = ''

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            'supported_tactile_roles': list(self.supported_tactile_roles),
            'supported_tactile_profiles': list(self.supported_tactile_profiles),
            'default_tactile_profile': self.default_tactile_profile,
        }
        if self.tactile_sensor_vendor:
            out['tactile_sensor_vendor'] = self.tactile_sensor_vendor
        if self.tactile_sdk_version:
            out['tactile_sdk_version'] = self.tactile_sdk_version
        return out

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'TactileCapabilities':
        if not data:
            return cls()
        return cls(
            supported_tactile_roles=[str(x) for x in data.get('supported_tactile_roles', [])],
            supported_tactile_profiles=[str(x) for x in data.get('supported_tactile_profiles', [])],
            default_tactile_profile=str(data.get('default_tactile_profile', TACTILE_PROFILE_DERIVED)),
            tactile_sensor_vendor=str(data.get('tactile_sensor_vendor', '')),
            tactile_sdk_version=str(data.get('tactile_sdk_version', '')),
        )


_PROFILE_FIELD_REQUIREMENTS: Dict[str, Set[str]] = {
    TACTILE_PROFILE_RAW: {TACTILE_FIELD_RECTIFY_BGR},
    TACTILE_PROFILE_DERIVED: {TACTILE_FIELD_FORCE_XYZ, TACTILE_FIELD_WRENCH_6D},
}


def parse_tactile_profiles(profile: str) -> Set[str]:
    """Expand profile string into a set of base profiles."""
    normalized = profile.strip()
    if normalized == TACTILE_PROFILE_RAW_PLUS_DERIVED:
        return {TACTILE_PROFILE_RAW, TACTILE_PROFILE_DERIVED}
    if normalized in (TACTILE_PROFILE_RAW, TACTILE_PROFILE_DERIVED):
        return {normalized}
    raise ValueError(f'Unsupported tactile_profile: {profile!r}')


def required_field_types(profile: str) -> Set[str]:
    """Union of required field types for the given profile string."""
    required: Set[str] = set()
    for base in parse_tactile_profiles(profile):
        required |= _PROFILE_FIELD_REQUIREMENTS[base]
    return required


def make_tactile_field(
    field_type: str,
    array: np.ndarray,
    *,
    units: str = '',
    encoding: str = 'raw',
) -> TactileField:
    array = np.asarray(array)
    return TactileField(
        field_type=field_type,
        shape=[int(x) for x in array.shape],
        dtype=str(array.dtype),
        units=units,
        encoding=encoding,
        data_bytes=array.tobytes(),
    )


def field_to_ndarray(field: TactileField) -> np.ndarray:
    if field.data_bytes is None:
        raise ValueError(f'TactileField {field.field_type!r} has no data_bytes')
    dtype = np.dtype(field.dtype) if field.dtype else np.float32
    array = np.frombuffer(field.data_bytes, dtype=dtype)
    if field.shape:
        array = array.reshape(field.shape)
    return array


def compute_contact_state(
    wrench_6d: Iterable[float],
    *,
    force_threshold_n: float = 0.05,
) -> tuple[bool, float]:
    """Derive contact_state and confidence from wrench magnitude."""
    wrench = np.asarray(list(wrench_6d), dtype=np.float64).ravel()
    if wrench.size < 3:
        return False, 0.0
    force_norm = float(np.linalg.norm(wrench[:3]))
    if force_norm <= force_threshold_n:
        return False, 0.0
    confidence = min(1.0, force_norm / max(force_threshold_n * 4.0, 1e-6))
    return True, confidence


def observation_field_types(observation: TactileObservation) -> Set[str]:
    return {field.field_type for field in observation.fields}


def validate_tactile_observation(
    observation: TactileObservation,
    *,
    profile: str,
    role: str,
) -> None:
    if observation.tactile_role != role:
        raise ValueError(
            f'Expected tactile_role {role!r}, got {observation.tactile_role!r}'
        )
    present = observation_field_types(observation)
    missing = required_field_types(profile) - present
    if missing:
        raise ValueError(
            f'Tactile role {role!r} missing required fields for profile {profile!r}: {sorted(missing)}'
        )
    if TACTILE_FIELD_WRENCH_6D in required_field_types(profile):
        if observation.wrench_6d is None or len(observation.wrench_6d) != 6:
            raise ValueError(f'Tactile role {role!r} missing wrench_6d summary')


def validate_tactile_observations(
    observations: List[TactileObservation],
    config: TactileBenchmarkConfig,
) -> None:
    """Validate observation tactile payload against benchmark config."""
    if not config.tactile_required:
        return

    if not config.tactile_roles:
        raise ValueError('tactile_required=true but tactile_roles is empty')

    by_role = {obs.tactile_role: obs for obs in observations}
    for role in config.tactile_roles:
        if role not in by_role:
            raise ValueError(f'Missing tactile observation for role {role!r}')
        validate_tactile_observation(by_role[role], profile=config.tactile_profile, role=role)


def tactile_observations_to_legacy(observations: List[TactileObservation]) -> Dict[str, Any]:
    """Convert canonical tactile observations to legacy new_obs['tactile'] layout."""
    legacy: Dict[str, Any] = {}
    for obs in observations:
        role_data: Dict[str, Any] = {}
        if obs.wrench_6d is not None:
            role_data['wrench_6d'] = np.asarray(obs.wrench_6d, dtype=np.float32)
        if obs.contact_state is not None:
            role_data['contact_state'] = obs.contact_state
        if obs.contact_confidence is not None:
            role_data['contact_confidence'] = obs.contact_confidence
        for field in obs.fields:
            array = field_to_ndarray(field)
            if field.field_type == TACTILE_FIELD_RECTIFY_BGR:
                role_data['rectify'] = array
            elif field.field_type == TACTILE_FIELD_FORCE_XYZ:
                role_data['force'] = array
            elif field.field_type == TACTILE_FIELD_WRENCH_6D and 'wrench_6d' not in role_data:
                role_data['wrench_6d'] = array.astype(np.float32)
        legacy[obs.tactile_role] = role_data
    return legacy
