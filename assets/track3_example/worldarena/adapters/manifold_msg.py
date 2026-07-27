"""Adapter between manifold_msg private protocol and WorldArena canonical schema."""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from worldarena.adapters.base import RobotAdapter
from worldarena.geometry import camera_calibration_for_role
from worldarena.protocol import (
    ACTION_MODE_JOINT_ABSOLUTE,
    ACTION_MODE_TASK_SPACE_ABSOLUTE,
    ACTION_MODE_TASK_SPACE_DELTA,
    ARM_ID_LEFT,
    ARM_ID_RIGHT,
    CAMERA_ROLE_GLOBAL,
    CAMERA_ROLE_LEFT_WRIST,
    CAMERA_ROLE_RIGHT_WRIST,
    SCHEMA_VERSION,
    TACTILE_ROLE_LEFT_GRIPPER,
    TACTILE_ROLE_RIGHT_GRIPPER,
    TACTILE_VENDOR_XENSE,
)
from worldarena.schema import (
    ActionPacket,
    ActionStep,
    ArmAction,
    ArmState,
    CameraObservation,
    GripperState,
    JointState,
    ObservationPacket,
    Pose,
    Quaternion,
    RobotState,
    SafetyState,
    SessionContext,
    TactileObservation,
    Vector3,
)
from worldarena.embodiment import EmbodimentProfile, preset_profile
from worldarena.observation_history import (
    ObservationHistoryCapabilities,
    ObservationHistoryConfig,
    camera_history_depth,
    observation_history_actual,
)
from worldarena.tactile import TactileBenchmarkConfig, TactileCapabilities

_CAMERA_ROLE_TO_FIELD = {
    CAMERA_ROLE_GLOBAL: 'img_front',
    CAMERA_ROLE_LEFT_WRIST: 'img_left',
    CAMERA_ROLE_RIGHT_WRIST: 'img_right',
}

_LEFT_JOINT_NAMES = [f'left_j{i + 1}' for i in range(7)]
_RIGHT_JOINT_NAMES = [f'right_j{i + 1}' for i in range(7)]

_TOUCH_ARM_FIELDS = {
    ARM_ID_LEFT: 'img_left_touch',
    ARM_ID_RIGHT: 'img_right_touch',
}
_PAD_SUFFIXES = ('a', 'b', 'c', 'd')

logger = logging.getLogger(__name__)


def summarize_private_observation(private_obs: Dict[str, Any]) -> str:
    """One-line summary for debug logging."""

    def _frame_count(key: str) -> int:
        value = private_obs.get(key)
        if not value:
            return 0
        return len(value) if isinstance(value, list) else 1

    def _touch_pad_count(key: str) -> int:
        history = private_obs.get(key) or []
        if not history:
            return 0
        latest = history[-1] if isinstance(history, list) else history
        return len(latest) if isinstance(latest, list) else 1

    extra = private_obs.get('extra_info') if isinstance(private_obs.get('extra_info'), dict) else {}
    prompt = str(private_obs.get('prompt', ''))
    if len(prompt) > 80:
        prompt = prompt[:77] + '...'
    parts = [
        f'ts={private_obs.get("timestamp")}',
        f'prompt={prompt!r}',
        f'cam front/left/right={_frame_count("img_front")}/{_frame_count("img_left")}/{_frame_count("img_right")}',
        f'touch left/right pads={_touch_pad_count("img_left_touch")}/{_touch_pad_count("img_right_touch")}',
    ]
    if extra:
        parts.append(f'extra_info={extra}')
    return ' | '.join(parts)


def _touch_history_frames(value: Any) -> List[List[np.ndarray]]:
    """Parse ``img_*_touch`` private obs: outer=history, inner=gripper pads."""
    if value is None:
        return []
    if not isinstance(value, list):
        return [[_to_uint8_hw3(value)]]
    history: List[List[np.ndarray]] = []
    for history_step in value:
        if isinstance(history_step, list):
            pads = [_to_uint8_hw3(item) for item in history_step if item is not None]
        elif history_step is not None:
            pads = [_to_uint8_hw3(history_step)]
        else:
            pads = []
        if pads:
            history.append(pads)
    return history


def _touch_role_series(
    private_obs: Dict[str, Any],
    profile: EmbodimentProfile,
) -> Dict[str, List[np.ndarray]]:
    """Map per-arm touch history to canonical tactile roles (oldest-first)."""
    expected_roles = profile.resolved_tactile_roles()
    series: Dict[str, List[np.ndarray]] = {role: [] for role in expected_roles}

    if profile.vital_legacy_tactile_names and len(profile.active_arms) == 1:
        arm_id = profile.active_arms[0]
        field = _TOUCH_ARM_FIELDS[arm_id]
        pad_roles = [TACTILE_ROLE_LEFT_GRIPPER, TACTILE_ROLE_RIGHT_GRIPPER]
        for step_pads in _touch_history_frames(private_obs.get(field)):
            for pad_idx, role in enumerate(pad_roles):
                if pad_idx < len(step_pads) and role in series:
                    series[role].append(step_pads[pad_idx])
        return {role: frames for role, frames in series.items() if frames}

    for arm_id in profile.active_arms:
        field = _TOUCH_ARM_FIELDS.get(arm_id)
        if field is None:
            continue
        for step_pads in _touch_history_frames(private_obs.get(field)):
            for pad_idx, pad_img in enumerate(step_pads):
                suffix = _PAD_SUFFIXES[pad_idx] if pad_idx < len(_PAD_SUFFIXES) else str(pad_idx)
                role = f'{arm_id}_gripper_pad_{suffix}'
                if role in series:
                    series[role].append(pad_img)

    return {role: frames for role, frames in series.items() if frames}


def _touch_role_frames_latest(
    private_obs: Dict[str, Any],
    profile: EmbodimentProfile,
) -> Dict[str, Dict[str, np.ndarray]]:
    role_frames: Dict[str, Dict[str, np.ndarray]] = {}
    for role, frames in _touch_role_series(private_obs, profile).items():
        role_frames[role] = {'rectify': frames[-1]}
    return role_frames


def _extract_force_only_tactile_observations(
    private_obs: Dict[str, Any],
    tactile_config: TactileBenchmarkConfig,
    image_roles: List[str],
    step_index: int,
    timestamp_ns: int,
) -> List[TactileObservation]:
    """Create TactileObservation objects for force-only roles not backed by images.

    Some embodiments expose additional force/torque sensors (e.g. wrist F/T)
    as independent tactile roles.  This helper looks for those roles in the
    private observation dict and emits canonical TactileObservation objects
    containing only a wrench_6d summary.
    """
    if not tactile_config.tactile_roles:
        return []

    force_root = private_obs.get('tactile_force', private_obs)
    if not isinstance(force_root, dict):
        force_root = private_obs

    observations: List[TactileObservation] = []
    for role in tactile_config.tactile_roles:
        if role in image_roles:
            continue
        value = force_root.get(role)
        if value is None:
            value = private_obs.get(role)
        if value is None:
            continue
        wrench = np.asarray(value, dtype=np.float32).ravel()
        if wrench.size < 6:
            continue
        wrench_6d = [float(x) for x in wrench[:6]]
        observations.append(
            TactileObservation(
                tactile_role=role,
                sensor_vendor='manifold',
                timestamp_ns=timestamp_ns,
                frame_id=f'{role}_force_{step_index}',
                contact_state=None,
                contact_confidence=None,
                wrench_6d=wrench_6d,
                fields=[],
            )
        )
    return observations


def _last_row(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim >= 2:
        return array[-1]
    return array


def _manifold_euler_seq() -> str:
    import os

    return os.environ.get('VITAL_MANIFOLD_EULER_SEQ', 'xyz').strip() or 'xyz'


def _resolve_is_euler(private_obs: Dict[str, Any], arm_id: str) -> bool:
    """Resolve per-arm ``is_euler`` from manifold private obs (bool or list)."""
    flag = private_obs.get('is_euler', False)
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, (list, tuple, np.ndarray)):
        if len(flag) == 0:
            return False
        if len(flag) >= 2:
            idx = 0 if arm_id == ARM_ID_LEFT else 1
            return bool(flag[idx])
        return bool(flag[-1])
    return bool(flag)


def _pose7_from_manifold(raw: np.ndarray, *, is_euler: bool) -> np.ndarray:
    """Convert manifold ``end_pose`` rot3 to xyz+quat xyzw (7D)."""
    from scipy.spatial.transform import Rotation as R

    raw = np.asarray(raw, dtype=np.float64).ravel()
    if raw.size < 7:
        return np.zeros(7, dtype=np.float64)
    xyz = raw[:3]
    if is_euler:
        roll, pitch, yaw = float(raw[3]), float(raw[4]), float(raw[5])
        quat = R.from_euler(_manifold_euler_seq(), [roll, pitch, yaw]).as_quat()
        return np.array([xyz[0], xyz[1], xyz[2], quat[0], quat[1], quat[2], quat[3]], dtype=np.float64)
    return raw[:7].copy()


def _pose_from_manifold(raw: np.ndarray, *, is_euler: bool, frame: str = 'base') -> Pose:
    return _pose_from_array(_pose7_from_manifold(raw, is_euler=is_euler), frame=frame)


def _end_pose8_from_manifold(private_obs: Dict[str, Any], arm_id: str) -> List[float]:
    """Build normalized 8D base-frame pose ``[xyz+quat, gripper]`` from private obs."""
    prefix = 'left' if arm_id == ARM_ID_LEFT else 'right'
    raw = _last_row(private_obs.get(f'{prefix}_end_pose', []))
    if raw.size == 0:
        return []
    is_euler = _resolve_is_euler(private_obs, arm_id)
    pose7 = _pose7_from_manifold(raw, is_euler=is_euler)
    gripper = 0.0
    if raw.size >= 8:
        gripper = float(raw[7])
    else:
        joint = _last_row(private_obs.get(f'{prefix}_arm_joint_state', []))
        if joint.size:
            gripper = float(joint[-1])
    return [float(x) for x in pose7] + [gripper]


def _hold_end_pose_from_private_obs(private_obs: Dict[str, Any], arm_id: str) -> List[float]:
    """Build a consistent 8D base-frame hold pose ``[pose7, gripper]`` for inactive arms."""
    return _end_pose8_from_manifold(private_obs, arm_id)


def _vital_pose_debug_enabled() -> bool:
    import os

    return os.environ.get('VITAL_DEBUG_PIPELINE', '0') == '1' or os.environ.get('VITAL_DEBUG_ACTION', '0') == '1'


def _log_manifold_end_pose_obs_debug(private_obs: Dict[str, Any]) -> None:
    if not _vital_pose_debug_enabled():
        return
    raw_flag = private_obs.get('is_euler', False)
    for arm_id in (ARM_ID_LEFT, ARM_ID_RIGHT):
        prefix = 'left' if arm_id == ARM_ID_LEFT else 'right'
        raw = _last_row(private_obs.get(f'{prefix}_end_pose', []))
        if raw.size < 7:
            continue
        is_euler = _resolve_is_euler(private_obs, arm_id)
        parsed = _pose7_from_manifold(raw, is_euler=is_euler)
        naive = _pose7_from_manifold(raw, is_euler=False)
        logger.info(
            '[VITAL_DEBUG_PIPELINE] obs %s is_euler=%s raw_flag=%r euler_seq=%s '
            'raw_rot=[%.4f,%.4f,%.4f] parsed_quat=[%.4f,%.4f,%.4f,%.4f] '
            'naive_as_quat=[%.4f,%.4f,%.4f,%.4f]',
            arm_id,
            is_euler,
            raw_flag,
            _manifold_euler_seq(),
            float(raw[3]),
            float(raw[4]),
            float(raw[5]),
            float(parsed[3]),
            float(parsed[4]),
            float(parsed[5]),
            float(parsed[6]),
            float(naive[3]),
            float(naive[4]),
            float(naive[5]),
            float(naive[6]),
        )


def _pose_xyz_delta_m(a: List[float], b: np.ndarray) -> float:
    if len(a) < 3 or b.size < 3:
        return float('nan')
    delta = np.asarray(a[:3], dtype=np.float64) - np.asarray(b[:3], dtype=np.float64).ravel()[:3]
    return float(np.linalg.norm(delta))


def _maybe_truncate_action_lists(
    left: List[List[float]],
    right: List[List[float]],
) -> tuple[List[List[float]], List[List[float]]]:
    import os

    raw = os.environ.get('VITAL_ACTION_EXEC_STEPS', '').strip()
    if not raw:
        return left, right
    n = max(1, int(raw))
    return left[:n], right[:n]


def _log_vital_action_debug(
    *,
    action_packet: ActionPacket,
    params: Dict[str, Any],
    private_obs: Dict[str, Any],
    profile: EmbodimentProfile,
) -> None:
    import os

    if os.environ.get('VITAL_DEBUG_ACTION', '0') != '1':
        return

    method = params.get('method')
    left_key = 'left_end_pose' if method == 'end_pose' else 'left_joint_states'
    right_key = 'right_end_pose' if method == 'end_pose' else 'right_joint_states'
    left_cmds = params.get(left_key) or []
    right_cmds = params.get(right_key) or []
    chunk_len = len(action_packet.action_chunk)
    active = list(profile.active_arms)
    inactive = list(profile.inactive_arms)

    lines = [
        (
            f'[VITAL_DEBUG_ACTION] mode={action_packet.action_mode} method={method} '
            f'active_arms={active} inactive_arms={inactive} '
            f'chunk={chunk_len} exec_left={len(left_cmds)} exec_right={len(right_cmds)} '
            f'rate={params.get("action_rate")}'
        ),
    ]

    if action_packet.action_mode == ACTION_MODE_JOINT_ABSOLUTE:
        logger.warning('[VITAL_DEBUG_ACTION] WARNING: joint mode — 10D eef policy may be misparsed')
    if method == 'joint' and 'right' in active and profile.hold_inactive_arms:
        logger.warning('[VITAL_DEBUG_ACTION] WARNING: inactive arm hold uses joint commands (left may move)')

    if method == 'end_pose':
        for arm_id, cmds, obs_key in (
            (ARM_ID_LEFT, left_cmds, 'left_end_pose'),
            (ARM_ID_RIGHT, right_cmds, 'right_end_pose'),
        ):
            if not cmds:
                continue
            obs8 = _end_pose8_from_manifold(private_obs, arm_id)
            obs_pose = np.asarray(obs8[:7], dtype=np.float64) if obs8 else _last_row(private_obs.get(obs_key, []))
            role = 'HOLD' if arm_id in inactive else 'TARGET'
            first = cmds[0]
            dim = len(first)
            pos_delta = _pose_xyz_delta_m(first, obs_pose)
            quat_line = ''
            if len(first) >= 7 and obs8 and len(obs8) >= 7:
                dquat = np.asarray(first[3:7], dtype=np.float64) - np.asarray(obs8[3:7], dtype=np.float64)
                quat_line = (
                    f' dquat=[{dquat[0]:+.4f},{dquat[1]:+.4f},{dquat[2]:+.4f},{dquat[3]:+.4f}]'
                    f' cmd_qy={first[4]:.4f} obs_qy={obs8[4]:.4f}'
                    f' cmd_qw={first[6]:.4f} obs_qw={obs8[6]:.4f}'
                )
            is_euler = _resolve_is_euler(private_obs, arm_id)
            lines.append(
                f'[VITAL_DEBUG_ACTION] {role} {arm_id} dim={dim} obs_is_euler={is_euler} '
                f'obs_delta_xyz_m={pos_delta:.5f} first={first[:4]}...{quat_line}'
            )
            if arm_id in inactive and pos_delta > 1e-4:
                logger.warning(
                    '[VITAL_DEBUG_ACTION] inactive %s hold differs from obs by %.5fm — left may drift',
                    arm_id,
                    pos_delta,
                )
        if right_cmds and ARM_ID_RIGHT in active:
            obs_right = _hold_end_pose_from_private_obs(private_obs, ARM_ID_RIGHT)
            target_right = right_cmds[0]
            if obs_right and len(target_right) >= 3:
                delta = _pose_xyz_delta_m(target_right, np.asarray(obs_right))
                lines.append(
                    f'[VITAL_DEBUG_ACTION] active right step0 vs obs delta_xyz_m={delta:.5f} '
                    f'target={target_right[:4]}... obs={obs_right[:4]}...'
                )
                exec_steps = os.environ.get('VITAL_ACTION_EXEC_STEPS', '').strip()
                if len(right_cmds) > 1 and not exec_steps:
                    lines.append(
                        f'[VITAL_DEBUG_ACTION] sending full chunk ({len(right_cmds)} steps); '
                        'set VITAL_ACTION_EXEC_STEPS=1 to execute only the first step per cycle'
                    )

    for line in lines:
        logger.info(line)


def _collect_frames(value: Any, depth: int) -> List[np.ndarray]:
    """Return up to *depth* frames oldest-first from a private obs field."""
    if depth <= 0:
        depth = 1
    if isinstance(value, list):
        arrays = [_to_uint8_hw3(item) for item in value if item is not None]
    elif value is None:
        arrays = []
    else:
        arrays = [_to_uint8_hw3(value)]
    if not arrays:
        return []
    return arrays[-depth:]


def _history_timestamps(base_ns: int, count: int, interval_s: float) -> List[int]:
    if count <= 0:
        return []
    interval_ns = int(max(0.0, interval_s) * 1_000_000_000)
    if interval_ns <= 0:
        return [base_ns] * count
    return [base_ns - interval_ns * (count - idx) for idx in range(count)]


def _to_uint8_hw3(frame: Any) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        if np.nanmax(array) <= 1.0:
            array = (array * 255).clip(0, 255).astype(np.uint8)
        else:
            array = array.clip(0, 255).astype(np.uint8)
    return array


def _timestamp_to_ns(timestamp: Any) -> int:
    if timestamp is None:
        return time.time_ns()
    try:
        value = float(timestamp)
    except (TypeError, ValueError):
        return time.time_ns()
    if value > 1e15:
        return int(value)
    return int(value * 1_000_000_000)


def _pose_from_array(pose: np.ndarray, frame: str = 'base') -> Pose:
    pose = np.asarray(pose, dtype=np.float64).ravel()
    if pose.size < 7:
        return Pose(frame=frame)
    return Pose(
        position_m=Vector3(float(pose[0]), float(pose[1]), float(pose[2])),
        orientation_xyzw=Quaternion(float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])),
        frame=frame,
    )


def _apply_delta_pose(current: Pose, delta_pos: Optional[Vector3], delta_rot: Optional[Vector3]) -> Pose:
    from scipy.spatial.transform import Rotation as R

    pos = np.array([current.position_m.x, current.position_m.y, current.position_m.z], dtype=np.float64)
    quat = np.array(
        [
            current.orientation_xyzw.x,
            current.orientation_xyzw.y,
            current.orientation_xyzw.z,
            current.orientation_xyzw.w,
        ],
        dtype=np.float64,
    )
    if delta_pos is not None:
        pos += np.array([delta_pos.x, delta_pos.y, delta_pos.z], dtype=np.float64)
    if delta_rot is not None:
        delta = np.array([delta_rot.x, delta_rot.y, delta_rot.z], dtype=np.float64)
        angle = np.linalg.norm(delta)
        if angle > 1e-8:
            rot_delta = R.from_rotvec(delta)
            rot_current = R.from_quat(quat)
            quat = (rot_delta * rot_current).as_quat()
    return Pose(
        position_m=Vector3(float(pos[0]), float(pos[1]), float(pos[2])),
        orientation_xyzw=Quaternion(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
        frame=current.frame,
    )


class ManifoldMsgAdapter(RobotAdapter):
    """Adapter for Manifold AgileX robots using manifold_msg HTTP protocol."""

    adapter_id = 'manifold_msg.agilex'
    adapter_version = 'adapter.manifold_msg.1.3.0'
    embodiment_type = 'dual_arm'

    def __init__(
        self,
        server: Any,
        *,
        embodiment_id: str = 'agilex_dual_arm',
        embodiment_profile: Optional[EmbodimentProfile] = None,
        default_action_rate: int = 45,
        joint_dof: int = 7,
        tactile_config: Optional[TactileBenchmarkConfig] = None,
        debug_observations: bool = False,
    ) -> None:
        self._server = server
        self._debug_observations = debug_observations
        self._profile = embodiment_profile or preset_profile('dual_arm', embodiment_id=embodiment_id)
        self.embodiment_id = self._profile.embodiment_id
        self.embodiment_type = self._profile.embodiment_type
        self.adapter_id = self._profile.adapter_id
        self.adapter_version = self._profile.adapter_version
        self.default_action_rate = default_action_rate
        self.joint_dof = joint_dof
        # Converter-only collector (no local xensesdk); tactile frames come from manifold_agent.
        from worldarena.adapters.xense import XenseTactileCollector

        self._tactile_converter = XenseTactileCollector({})
        if tactile_config is not None:
            self._tactile_config = tactile_config
        elif self._profile.tactile_enabled:
            self._tactile_config = TactileBenchmarkConfig(
                tactile_required=True,
                tactile_profile=self._profile.tactile_profile,
                tactile_roles=self._profile.resolved_tactile_roles(),
            )
        else:
            self._tactile_config = TactileBenchmarkConfig()
        self._history_config = ObservationHistoryConfig()
        self._step_index = 0
        self._context = SessionContext(
            schema_version=SCHEMA_VERSION,
            embodiment_id=embodiment_id,
            embodiment_type=self.embodiment_type,
            adapter_version=self.adapter_version,
        )
        self._episode_events: List[Any] = []

    def on_episode_event(self, event: Any) -> None:
        self._episode_events.append(event)

    def set_context(self, context: SessionContext) -> None:
        self._context = context

    def set_tactile_config(self, config: TactileBenchmarkConfig) -> None:
        self._tactile_config = config

    def set_observation_history_config(self, config: ObservationHistoryConfig) -> None:
        self._history_config = config

    def observation_history_capabilities(self) -> Dict[str, Any]:
        roles = self._profile.resolved_camera_roles()
        return ObservationHistoryCapabilities(
            max_history_per_role=self._profile.max_observation_history,
            supported_camera_history_roles=roles,
            supported_tactile_history_roles=self._profile.resolved_tactile_roles()
            if self._profile.tactile_enabled
            else [],
            supports_robot_state_history=False,
        ).to_dict()

    def set_embodiment_profile(self, profile: EmbodimentProfile) -> None:
        self._profile = profile
        self.embodiment_id = profile.embodiment_id
        self.embodiment_type = profile.embodiment_type
        self.adapter_id = profile.adapter_id

    @property
    def embodiment_profile(self) -> EmbodimentProfile:
        return self._profile

    def tactile_capabilities(self) -> Optional[Dict[str, Any]]:
        if not self._profile.tactile_enabled:
            return None
        profiles = [self._profile.tactile_profile]
        return TactileCapabilities(
            supported_tactile_roles=self._profile.resolved_tactile_roles(),
            supported_tactile_profiles=profiles,
            default_tactile_profile=self._profile.tactile_profile,
            tactile_sensor_vendor=TACTILE_VENDOR_XENSE,
        ).to_dict()

    def _tactile_frames_from_private_obs(self, private_obs: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        return _touch_role_frames_latest(private_obs, self._profile)

    def _collect_tactile_observations(
        self,
        private_obs: Dict[str, Any],
        step_index: int,
    ) -> List[TactileObservation]:
        if not self._tactile_config.tactile_required and not self._tactile_config.tactile_roles:
            return []

        role_series = _touch_role_series(private_obs, self._profile)
        if not role_series and self._tactile_config.tactile_roles:
            fallback_profile = EmbodimentProfile(
                embodiment_id=self._profile.embodiment_id,
                embodiment_type=self._profile.embodiment_type,
                active_arms=list(self._profile.active_arms),
                tactile_enabled=True,
                tactile_roles=list(self._tactile_config.tactile_roles),
                vital_legacy_tactile_names=True,
            )
            role_series = _touch_role_series(private_obs, fallback_profile)
        if not role_series:
            if self._tactile_config.tactile_required:
                raise ValueError(
                    'tactile_required but manifold_msg observation has no img_left_touch/img_right_touch; '
                    'ensure manifold_agent sends frame-aligned tactile in the same observation packet'
                )
            return []

        timestamp_ns = _timestamp_to_ns(private_obs.get('timestamp'))
        interval_s = self._history_config.history_interval_s
        observations: List[TactileObservation] = []

        for role, frames in role_series.items():
            if not frames:
                continue
            history_frames = frames[:-1]
            latest_arrays = {role: {'rectify': frames[-1]}}
            role_config = dataclasses.replace(self._tactile_config, tactile_roles=[role])
            obs_list = self._tactile_converter.collect_from_arrays(
                latest_arrays,
                role_config,
                step_index=step_index,
                timestamp_ns=timestamp_ns,
            )
            observation = obs_list[0]
            if history_frames:
                history_ts = _history_timestamps(timestamp_ns, len(history_frames), interval_s)
                observation.fields_history = [
                    self._tactile_converter.collect_from_arrays(
                        {role: {'rectify': frame}},
                        role_config,
                        step_index=step_index,
                        timestamp_ns=history_ts[idx] if idx < len(history_ts) else timestamp_ns,
                    )[0].fields
                    for idx, frame in enumerate(history_frames)
                ]
                observation.fields_history_timestamps_ns = history_ts
            observations.append(observation)

        # Add force-only roles (e.g. independent wrist F/T sensors) that are not
        # backed by tactile images.
        force_only = _extract_force_only_tactile_observations(
            private_obs,
            self._tactile_config,
            image_roles=list(role_series.keys()),
            step_index=step_index,
            timestamp_ns=timestamp_ns,
        )
        observations.extend(force_only)

        return observations

    def _filter_arms(self, arms: List[ArmState]) -> List[ArmState]:
        active = set(self._profile.active_arms)
        return [arm for arm in arms if arm.arm_id in active]

    def _current_joint_for_arm(self, private_obs: Dict[str, Any], arm_id: str) -> List[float]:
        key = f'{arm_id}_arm_joint_state'
        if key not in private_obs:
            return []
        joint = _last_row(private_obs[key])
        return [float(x) for x in joint]

    def metadata(self) -> Dict[str, Any]:
        meta = super().metadata()
        meta.update(self._profile.to_dict())
        meta.update(self.observation_history_capabilities())
        return meta

    def wait_private_observation(self) -> Dict[str, Any]:
        private_obs = self._server.wait_observation()
        if self._debug_observations:
            logger.info(
                '[serve_robot] manifold_msg obs consumed (%s)',
                summarize_private_observation(private_obs),
            )
        return private_obs

    def wait_observation(self) -> Dict[str, Any]:
        """Compatibility shim for benchmark_runner live loop."""
        return self.wait_private_observation()

    def private_observation_to_canonical(
        self,
        private_obs: Dict[str, Any],
        *,
        context: Optional[SessionContext] = None,
        step_index: Optional[int] = None,
    ) -> ObservationPacket:
        ctx = context or self._context
        if step_index is None:
            step_index = self._step_index
            self._step_index += 1

        timestamp_ns = _timestamp_to_ns(private_obs.get('timestamp'))
        if ctx.task_instruction == '':
            ctx = SessionContext(
                **{**ctx.to_dict(), 'task_instruction': str(private_obs.get('prompt', ''))}
            )

        arms: List[ArmState] = []
        if 'left_end_pose' in private_obs and 'right_end_pose' in private_obs:
            left_pose = _last_row(private_obs['left_end_pose'])
            right_pose = _last_row(private_obs['right_end_pose'])
            left_joint = _last_row(private_obs.get('left_arm_joint_state', []))
            right_joint = _last_row(private_obs.get('right_arm_joint_state', []))
            left_gripper = float(left_joint[-1]) if left_joint.size else 0.0
            right_gripper = float(right_joint[-1]) if right_joint.size else 0.0
            left_is_euler = _resolve_is_euler(private_obs, ARM_ID_LEFT)
            right_is_euler = _resolve_is_euler(private_obs, ARM_ID_RIGHT)
            arms = [
                ArmState(
                    arm_id=ARM_ID_LEFT,
                    joint_state=JointState(
                        joint_names=_LEFT_JOINT_NAMES[: left_joint.size],
                        position_rad=[float(x) for x in left_joint],
                    ),
                    ee_pose_base=_pose_from_manifold(left_pose, is_euler=left_is_euler, frame='base'),
                    gripper=GripperState(open_ratio=left_gripper),
                ),
                ArmState(
                    arm_id=ARM_ID_RIGHT,
                    joint_state=JointState(
                        joint_names=_RIGHT_JOINT_NAMES[: right_joint.size],
                        position_rad=[float(x) for x in right_joint],
                    ),
                    ee_pose_base=_pose_from_manifold(right_pose, is_euler=right_is_euler, frame='base'),
                    gripper=GripperState(open_ratio=right_gripper),
                ),
            ]
            _log_manifold_end_pose_obs_debug(private_obs)
        elif 'left_arm_joint_state' in private_obs and 'right_arm_joint_state' in private_obs:
            left_joint = _last_row(private_obs['left_arm_joint_state'])
            right_joint = _last_row(private_obs['right_arm_joint_state'])
            arms = [
                ArmState(
                    arm_id=ARM_ID_LEFT,
                    joint_state=JointState(
                        joint_names=_LEFT_JOINT_NAMES[: left_joint.size],
                        position_rad=[float(x) for x in left_joint],
                    ),
                    gripper=GripperState(open_ratio=float(left_joint[-1]) if left_joint.size else 0.0),
                ),
                ArmState(
                    arm_id=ARM_ID_RIGHT,
                    joint_state=JointState(
                        joint_names=_RIGHT_JOINT_NAMES[: right_joint.size],
                        position_rad=[float(x) for x in right_joint],
                    ),
                    gripper=GripperState(open_ratio=float(right_joint[-1]) if right_joint.size else 0.0),
                ),
            ]

        arms = self._filter_arms(arms)

        allowed_camera_roles = set(self._profile.resolved_camera_roles())
        camera_observations: List[CameraObservation] = []
        interval_s = self._history_config.history_interval_s
        for role in self._profile.resolved_camera_roles():
            field = _CAMERA_ROLE_TO_FIELD.get(role)
            if field is None or field not in private_obs:
                continue
            depth = min(
                camera_history_depth(self._history_config, role),
                self._profile.max_observation_history,
            )
            frames = _collect_frames(private_obs[field], depth)
            if not frames:
                continue
            history_frames = frames[:-1]
            latest = frames[-1]
            h, w = latest.shape[:2]
            intrinsics, extrinsics = camera_calibration_for_role(role)
            history_ts = _history_timestamps(timestamp_ns, len(history_frames), interval_s)
            camera_observations.append(
                CameraObservation(
                    camera_role=role,
                    modality='rgb',
                    frame_id=f'{role}_{step_index}',
                    timestamp_ns=timestamp_ns,
                    width=w,
                    height=h,
                    encoding='rgb8',
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    frame_bytes=latest.tobytes(),
                    frame_history_bytes=[frame.tobytes() for frame in history_frames],
                    frame_history_timestamps_ns=history_ts,
                )
            )
        _ = allowed_camera_roles  # reserved for future strict validation

        tactile_observations = self._collect_tactile_observations(private_obs, step_index)

        packet = ObservationPacket(
            context=ctx,
            observation_timestamp_ns=timestamp_ns,
            step_index=step_index,
            robot_state=RobotState(
                arms=arms,
                raw_state_refs={'protocol': 'manifold_msg', 'tactile_source': 'manifold_agent'},
            ),
            camera_observations=camera_observations,
            tactile_observations=tactile_observations,
            safety_state=SafetyState(
                velocity_limit_active=True,
                workspace_limit_active=True,
                safety_status='ok',
            ),
        )
        packet.history_actual = observation_history_actual(packet)
        return packet

    def _arms_from_private_obs(self, private_obs: Dict[str, Any]) -> Dict[str, ArmState]:
        arms: List[ArmState] = []
        if 'left_end_pose' in private_obs and 'right_end_pose' in private_obs:
            left_pose = _last_row(private_obs['left_end_pose'])
            right_pose = _last_row(private_obs['right_end_pose'])
            left_joint = _last_row(private_obs.get('left_arm_joint_state', []))
            right_joint = _last_row(private_obs.get('right_arm_joint_state', []))
            left_is_euler = _resolve_is_euler(private_obs, ARM_ID_LEFT)
            right_is_euler = _resolve_is_euler(private_obs, ARM_ID_RIGHT)
            arms = [
                ArmState(
                    arm_id=ARM_ID_LEFT,
                    ee_pose_base=_pose_from_manifold(left_pose, is_euler=left_is_euler, frame='base'),
                    gripper=GripperState(open_ratio=float(left_joint[-1]) if left_joint.size else 0.0),
                ),
                ArmState(
                    arm_id=ARM_ID_RIGHT,
                    ee_pose_base=_pose_from_manifold(right_pose, is_euler=right_is_euler, frame='base'),
                    gripper=GripperState(open_ratio=float(right_joint[-1]) if right_joint.size else 0.0),
                ),
            ]
        return {arm.arm_id: arm for arm in arms}

    def canonical_action_to_private(
        self,
        action_packet: ActionPacket,
        private_obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        timestamp = private_obs.get('timestamp')
        action_rate = self.default_action_rate
        mode = action_packet.action_mode

        if mode == ACTION_MODE_JOINT_ABSOLUTE:
            left_states: List[List[float]] = []
            right_states: List[List[float]] = []
            active = set(self._profile.active_arms)
            for step in action_packet.action_chunk:
                left = next((a for a in step.arm_actions if a.arm_id == ARM_ID_LEFT), None)
                right = next((a for a in step.arm_actions if a.arm_id == ARM_ID_RIGHT), None)
                if left and left.joint_position_rad and ARM_ID_LEFT in active:
                    if len(left.joint_position_rad) < 7:
                        raise RuntimeError(
                            f'Invalid left joint command dim={len(left.joint_position_rad)}; '
                            '10D eef6d must not use joint mode with 5+5 split'
                        )
                    left_states.append([float(x) for x in left.joint_position_rad])
                if right and right.joint_position_rad and ARM_ID_RIGHT in active:
                    if len(right.joint_position_rad) < 7:
                        raise RuntimeError(
                            f'Invalid right joint command dim={len(right.joint_position_rad)}; '
                            '10D eef6d must not use joint mode with 5+5 split'
                        )
                    right_states.append([float(x) for x in right.joint_position_rad])

            if self._profile.hold_inactive_arms:
                chunk_len = max(len(left_states), len(right_states), 0)
                if ARM_ID_LEFT not in self._profile.active_arms:
                    if right_states:
                        hold = self._current_joint_for_arm(private_obs, ARM_ID_LEFT)
                        left_states = [hold] * chunk_len if hold else []
                    else:
                        left_states = []
                if ARM_ID_RIGHT not in self._profile.active_arms:
                    if left_states:
                        hold = self._current_joint_for_arm(private_obs, ARM_ID_RIGHT)
                        right_states = [hold] * chunk_len if hold else []
                    else:
                        right_states = []

            left_states, right_states = _maybe_truncate_action_lists(left_states, right_states)

            return {
                'method': 'joint',
                'timestamp': timestamp,
                'action_rate': action_rate,
                'left_joint_states': left_states,
                'right_joint_states': right_states,
            }

        left_poses: List[List[float]] = []
        right_poses: List[List[float]] = []
        arms_by_id = self._arms_from_private_obs(private_obs)

        for step in action_packet.action_chunk:
            for arm_action in step.arm_actions:
                if arm_action.arm_id not in self._profile.active_arms:
                    continue
                if arm_action.target_pose_base is not None:
                    pose = arm_action.target_pose_base
                elif mode == ACTION_MODE_TASK_SPACE_DELTA:
                    current = arms_by_id.get(arm_action.arm_id)
                    if current is None:
                        continue
                    pose = _apply_delta_pose(
                        current.ee_pose_base,
                        arm_action.delta_position_m,
                        arm_action.delta_rotation_axis_angle,
                    )
                else:
                    continue

                values = [
                    pose.position_m.x,
                    pose.position_m.y,
                    pose.position_m.z,
                    pose.orientation_xyzw.x,
                    pose.orientation_xyzw.y,
                    pose.orientation_xyzw.z,
                    pose.orientation_xyzw.w,
                ]
                if arm_action.gripper_target_open_ratio is not None:
                    values.append(float(arm_action.gripper_target_open_ratio))
                elif arm_action.arm_id in arms_by_id:
                    values.append(float(arms_by_id[arm_action.arm_id].gripper.open_ratio))

                if arm_action.arm_id == ARM_ID_LEFT:
                    left_poses.append(values)
                elif arm_action.arm_id == ARM_ID_RIGHT:
                    right_poses.append(values)

        if self._profile.hold_inactive_arms:
            chunk_len = max(len(left_poses), len(right_poses), 0)
            if ARM_ID_LEFT not in self._profile.active_arms:
                if right_poses:
                    hold = _hold_end_pose_from_private_obs(private_obs, ARM_ID_LEFT)
                    left_poses = [hold] * len(right_poses) if hold else []
                else:
                    left_poses = []
            if ARM_ID_RIGHT not in self._profile.active_arms:
                if left_poses:
                    hold = _hold_end_pose_from_private_obs(private_obs, ARM_ID_RIGHT)
                    right_poses = [hold] * len(left_poses) if hold else []
                else:
                    right_poses = []

        left_poses, right_poses = _maybe_truncate_action_lists(left_poses, right_poses)

        return {
            'method': 'end_pose',
            'timestamp': timestamp,
            'action_rate': action_rate,
            'left_end_pose': left_poses,
            'right_end_pose': right_poses,
            'is_euler': False,
        }

    def apply_action(
        self,
        action_packet: ActionPacket,
        private_obs: Dict[str, Any],
        *,
        debug_raw_actions: Any = None,
    ) -> None:
        from worldarena.bridges.legacy_policy import (
            action_packet_looks_like_misparsed_joint,
            repair_action_packet_if_misparsed,
        )

        action_packet = repair_action_packet_if_misparsed(action_packet)
        if action_packet_looks_like_misparsed_joint(action_packet):
            logger.error(
                'Refusing misparsed 5+5 joint ActionPacket at apply_action; '
                '10D eef6d must use task_space_absolute/end_pose'
            )
            raise RuntimeError(
                'Misparsed 10D eef6d action (5+5 joint split). '
                'Update B/C to latest worldarena and verify action_format=eef6d_single.'
            )
        params = self.canonical_action_to_private(action_packet, private_obs)
        from worldarena.action_debug import log_action_cross_check, log_pipeline_trace

        right0 = None
        if params.get('method') == 'end_pose':
            right_poses = params.get('right_end_pose') or []
            if right_poses:
                right0 = right_poses[0]
        log_action_cross_check(
            action_packet=action_packet,
            raw_actions_10d=debug_raw_actions,
            manifold_right0=right0,
        )
        log_pipeline_trace(
            'C:apply_action',
            action_packet=action_packet,
            raw_actions=debug_raw_actions,
            manifold_params=params,
        )
        import os

        left_key = 'left_end_pose' if params['method'] == 'end_pose' else 'left_joint_states'
        right_key = 'right_end_pose' if params['method'] == 'end_pose' else 'right_joint_states'
        left_n = len(params.get(left_key) or [])
        right_n = len(params.get(right_key) or [])
        active = set(self._profile.active_arms)
        expected = sum(
            1
            for arm in active
            if (arm == ARM_ID_LEFT and left_n > 0) or (arm == ARM_ID_RIGHT and right_n > 0)
        )
        if expected == 0:
            logger.error(
                'apply_action produced no commands for active_arms=%s '
                '(mode=%s method=%s chunk=%d left=%d right=%d)',
                self._profile.active_arms,
                action_packet.action_mode,
                params.get('method'),
                len(action_packet.action_chunk),
                left_n,
                right_n,
            )
            return

        _log_vital_action_debug(
            action_packet=action_packet,
            params=params,
            private_obs=private_obs,
            profile=self._profile,
        )
        if params['method'] == 'joint':
            self._server.send_joint_state_action(
                params['timestamp'],
                params['action_rate'],
                params['left_joint_states'],
                params['right_joint_states'],
            )
            return

        self._server.send_end_pose_action(
            params['timestamp'],
            params['action_rate'],
            params['left_end_pose'],
            params['right_end_pose'],
            is_euler=params.get('is_euler', False),
        )

    def reset(self, reset_info: Optional[Dict[str, Any]] = None) -> None:
        self._step_index = 0
        if reset_info:
            self._context = SessionContext(
                **{
                    **self._context.to_dict(),
                    'session_id': str(reset_info.get('session_id', self._context.session_id)),
                    'episode_id': str(reset_info.get('episode_id', self._context.episode_id)),
                    'task_id': str(reset_info.get('task_id', self._context.task_id)),
                    'task_instruction': str(reset_info.get('prompt', self._context.task_instruction)),
                }
            )

    def wait_observation_packet(self) -> ObservationPacket:
        private_obs = self.wait_private_observation()
        return self.private_observation_to_canonical(private_obs)
