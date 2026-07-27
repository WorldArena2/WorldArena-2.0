"""Xense tactile sensor adapter for WorldArena canonical schema."""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np

from worldarena.protocol import (
    TACTILE_FIELD_FORCE_XYZ,
    TACTILE_FIELD_RECTIFY_BGR,
    TACTILE_FIELD_WRENCH_6D,
    TACTILE_PROFILE_DERIVED,
    TACTILE_PROFILE_RAW,
    TACTILE_PROFILE_RAW_PLUS_DERIVED,
    TACTILE_ROLE_LEFT_GRIPPER,
    TACTILE_ROLE_RIGHT_GRIPPER,
    TACTILE_VENDOR_XENSE,
)
from worldarena.schema import TactileObservation
from worldarena.tactile import (
    TactileBenchmarkConfig,
    TactileCapabilities,
    compute_contact_state,
    make_tactile_field,
    parse_tactile_profiles,
)


class _SensorFrame(Protocol):
    rectify: Optional[np.ndarray]
    force: Optional[np.ndarray]
    wrench_6d: Optional[np.ndarray]
    timestamp_ns: int


def _default_role_for_cam_id(cam_id: int) -> str:
    if cam_id == 0:
        return TACTILE_ROLE_LEFT_GRIPPER
    if cam_id == 1:
        return TACTILE_ROLE_RIGHT_GRIPPER
    return f'tactile_cam_{cam_id}'


def _to_uint8_hw3(array: np.ndarray) -> np.ndarray:
    frame = np.asarray(array)
    if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.dtype != np.uint8:
        if np.nanmax(frame) <= 1.0:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        else:
            frame = frame.clip(0, 255).astype(np.uint8)
    return frame


class XenseTactileCollector:
    """Collect Xense sensor frames and emit canonical TactileObservation objects."""

    adapter_id = 'xense.tactile'
    adapter_version = 'adapter.xense.1.0.0'

    def __init__(
        self,
        role_to_cam_id: Dict[str, int],
        *,
        sensor_serials: Optional[Dict[str, str]] = None,
        runtime_config_refs: Optional[Dict[str, str]] = None,
        contact_force_threshold_n: float = 0.05,
        sensor_factory: Optional[Callable[[int], Any]] = None,
    ) -> None:
        self.role_to_cam_id = dict(role_to_cam_id)
        self.sensor_serials = dict(sensor_serials or {})
        self.runtime_config_refs = dict(runtime_config_refs or {})
        self.contact_force_threshold_n = contact_force_threshold_n
        self._sensor_factory = sensor_factory or self._create_live_sensor
        self._sensors: Dict[str, Any] = {}

    @classmethod
    def from_cam_ids(
        cls,
        cam_ids: List[int],
        *,
        roles: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> 'XenseTactileCollector':
        if roles is not None and len(roles) != len(cam_ids):
            raise ValueError('roles length must match cam_ids length')
        role_to_cam_id: Dict[str, int] = {}
        for index, cam_id in enumerate(cam_ids):
            role = roles[index] if roles is not None else _default_role_for_cam_id(cam_id)
            role_to_cam_id[role] = cam_id
        return cls(role_to_cam_id, **kwargs)

    def capabilities(self) -> TactileCapabilities:
        sdk_version = ''
        try:
            import xensesdk

            sdk_version = getattr(xensesdk, '__version__', '')
        except ImportError:
            pass
        return TactileCapabilities(
            supported_tactile_roles=list(self.role_to_cam_id.keys()),
            supported_tactile_profiles=[
                TACTILE_PROFILE_RAW,
                TACTILE_PROFILE_DERIVED,
                TACTILE_PROFILE_RAW_PLUS_DERIVED,
            ],
            default_tactile_profile=TACTILE_PROFILE_DERIVED,
            tactile_sensor_vendor=TACTILE_VENDOR_XENSE,
            tactile_sdk_version=sdk_version,
        )

    def _create_live_sensor(self, cam_id: int) -> Any:
        from xensesdk import Sensor

        return Sensor.create(cam_id=cam_id)

    def _get_sensor(self, role: str) -> Any:
        if role not in self._sensors:
            cam_id = self.role_to_cam_id[role]
            self._sensors[role] = self._sensor_factory(cam_id)
        return self._sensors[role]

    def _read_sensor_frame(self, role: str, profiles: set[str]) -> _SensorFrame:
        sensor = self._get_sensor(role)
        from xensesdk import Sensor

        want_raw = TACTILE_PROFILE_RAW in profiles
        want_derived = TACTILE_PROFILE_DERIVED in profiles

        output_types: List[Any] = []
        if want_raw:
            output_types.append(Sensor.OutputType.Rectify)
        if want_derived:
            output_types.extend([Sensor.OutputType.Force, Sensor.OutputType.ForceResultant])

        if not output_types:
            raise ValueError('No tactile profiles enabled for collection')

        selected = sensor.selectSensorInfo(*output_types)
        idx = 0
        rectify = None
        force = None
        wrench_6d = None

        if want_raw:
            rectify = np.asarray(selected[idx])
            idx += 1
        if want_derived:
            force = np.asarray(selected[idx], dtype=np.float32)
            idx += 1
            wrench_6d = np.asarray(selected[idx], dtype=np.float32).ravel()

        return _SensorFrameImpl(
            rectify=rectify,
            force=force,
            wrench_6d=wrench_6d,
            timestamp_ns=time.time_ns(),
        )

    def collect(
        self,
        config: TactileBenchmarkConfig,
        *,
        step_index: int = 0,
    ) -> List[TactileObservation]:
        if not config.tactile_required and not config.tactile_roles:
            return []

        profiles = parse_tactile_profiles(config.tactile_profile)
        roles = config.tactile_roles or list(self.role_to_cam_id.keys())
        observations: List[TactileObservation] = []

        for role in roles:
            if role not in self.role_to_cam_id:
                raise ValueError(f'Unknown tactile role {role!r} for Xense collector')
            frame = self._read_sensor_frame(role, profiles)
            observations.append(
                self._frame_to_observation(
                    role=role,
                    frame=frame,
                    profiles=profiles,
                    step_index=step_index,
                )
            )
        return observations

    def collect_from_arrays(
        self,
        role_frames: Dict[str, Dict[str, np.ndarray]],
        config: TactileBenchmarkConfig,
        *,
        step_index: int = 0,
        timestamp_ns: Optional[int] = None,
    ) -> List[TactileObservation]:
        """Build observations from precomputed arrays (tests / replay)."""
        profiles = parse_tactile_profiles(config.tactile_profile)
        roles = config.tactile_roles or list(role_frames.keys())
        ts = timestamp_ns if timestamp_ns is not None else time.time_ns()
        observations: List[TactileObservation] = []
        for role in roles:
            if role not in role_frames:
                if config.tactile_required:
                    raise KeyError(
                        f'Missing tactile arrays for role {role!r}; '
                        f'available roles: {sorted(role_frames.keys())}'
                    )
                continue
            arrays = role_frames[role]
            frame = _SensorFrameImpl(
                rectify=arrays.get('rectify'),
                force=arrays.get('force'),
                wrench_6d=arrays.get('wrench_6d'),
                timestamp_ns=ts,
            )
            observations.append(
                self._frame_to_observation(
                    role=role,
                    frame=frame,
                    profiles=profiles,
                    step_index=step_index,
                )
            )
        return observations

    def _frame_to_observation(
        self,
        *,
        role: str,
        frame: _SensorFrame,
        profiles: set[str],
        step_index: int,
    ) -> TactileObservation:
        fields = []
        wrench_6d: Optional[List[float]] = None
        contact_state: Optional[bool] = None
        contact_confidence: Optional[float] = None

        if TACTILE_PROFILE_RAW in profiles and frame.rectify is not None:
            rectify = _to_uint8_hw3(frame.rectify)
            fields.append(
                make_tactile_field(TACTILE_FIELD_RECTIFY_BGR, rectify, encoding='raw')
            )

        if TACTILE_PROFILE_DERIVED in profiles:
            if frame.force is not None:
                force = np.asarray(frame.force, dtype=np.float32)
                fields.append(make_tactile_field(TACTILE_FIELD_FORCE_XYZ, force, units='N'))
            if frame.wrench_6d is not None:
                wrench = np.asarray(frame.wrench_6d, dtype=np.float32).ravel()
                if wrench.size >= 6:
                    wrench_6d = [float(x) for x in wrench[:6]]
                    fields.append(
                        make_tactile_field(TACTILE_FIELD_WRENCH_6D, wrench[:6], units='N_Nm')
                    )
                    contact_state, contact_confidence = compute_contact_state(
                        wrench_6d,
                        force_threshold_n=self.contact_force_threshold_n,
                    )

        runtime_ref = self.runtime_config_refs.get(role, '')
        if TACTILE_PROFILE_RAW in profiles and not runtime_ref:
            serial = self.sensor_serials.get(role, '')
            if serial:
                runtime_ref = f'runtime_{serial}@xense'

        return TactileObservation(
            tactile_role=role,
            sensor_vendor=TACTILE_VENDOR_XENSE,
            sensor_serial=self.sensor_serials.get(role, ''),
            timestamp_ns=frame.timestamp_ns,
            frame_id=f'{role}_tac_{step_index}',
            contact_state=contact_state,
            contact_confidence=contact_confidence,
            wrench_6d=wrench_6d,
            fields=fields,
            runtime_config_ref=runtime_ref,
        )

    def release(self) -> None:
        for sensor in self._sensors.values():
            release = getattr(sensor, 'release', None)
            if callable(release):
                release()
        self._sensors.clear()


@dataclasses.dataclass
class _SensorFrameImpl:
    rectify: Optional[np.ndarray]
    force: Optional[np.ndarray]
    wrench_6d: Optional[np.ndarray]
    timestamp_ns: int
