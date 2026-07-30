"""Bridge between legacy new_obs/actions and WorldArena canonical packets."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from real_world_benchmark.worldarena.protocol import (
    ACTION_MODE_JOINT_ABSOLUTE,
    ACTION_MODE_TASK_SPACE_ABSOLUTE,
    ARM_ID_LEFT,
    ARM_ID_RIGHT,
    CAMERA_ROLE_GLOBAL,
    CAMERA_ROLE_LEFT_WRIST,
    CAMERA_ROLE_RIGHT_WRIST,
    SCHEMA_VERSION,
)
from real_world_benchmark.worldarena.observation_history import (
    ObservationHistoryConfig,
    stack_camera_frames,
)
from real_world_benchmark.worldarena.schema import (
    ActionPacket,
    ActionStep,
    ArmAction,
    CameraObservation,
    ObservationPacket,
    Pose,
    Quaternion,
    SessionContext,
    Vector3,
)
from real_world_benchmark.worldarena.tactile import (
    TactileBenchmarkConfig,
    field_to_ndarray,
    tactile_observations_to_legacy,
)


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


def _stack_frames(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 4 and array.shape[1] in (1, 3):
        array = np.transpose(array, (0, 2, 3, 1))
    if array.dtype != np.uint8:
        if np.nanmax(array) <= 1.0:
            array = (array * 255).clip(0, 255).astype(np.uint8)
        else:
            array = array.clip(0, 255).astype(np.uint8)
    return array


def _stack_tactile_history(values: List[np.ndarray]) -> np.ndarray:
    if not values:
        return np.zeros((0,), dtype=np.float32)
    return np.stack([np.asarray(v) for v in values], axis=0)


def _extract_tactile_field_from_snapshot(
    fields: List[Any],
    *,
    field_type: str,
) -> Optional[np.ndarray]:
    for field in fields:
        if getattr(field, 'field_type', '') == field_type:
            return np.asarray(field_to_ndarray(field))
    return None


def _tactile_observation_history_to_legacy(obs: Any) -> Dict[str, Any]:
    role_data: Dict[str, Any] = {}
    history_timestamps = list(getattr(obs, 'fields_history_timestamps_ns', []) or [])
    current_timestamp = int(getattr(obs, 'timestamp_ns', 0))

    for field_type, legacy_key in (
        ('rectify_bgr', 'rectify'),
        ('force_xyz', 'force'),
        ('wrench_6d', 'wrench_6d'),
        ('marker2d', 'marker2d'),
        ('mesh3dflow', 'mesh3dflow'),
    ):
        history_values: List[np.ndarray] = []
        complete = True
        for snapshot in getattr(obs, 'fields_history', []) or []:
            array = _extract_tactile_field_from_snapshot(snapshot, field_type=field_type)
            if array is None:
                complete = False
                break
            history_values.append(array)
        current_value = _extract_tactile_field_from_snapshot(getattr(obs, 'fields', []) or [], field_type=field_type)
        if current_value is None:
            complete = False
        if not complete or current_value is None:
            continue
        history_values.append(current_value)
        role_data[legacy_key] = _stack_tactile_history(history_values)
        role_data['timestamp_ns'] = np.asarray([*history_timestamps, current_timestamp], dtype=np.int64)
    return role_data


from real_world_benchmark.worldarena.geometry import (
    cont6d_to_matrix,
    eef_pose_base_to_camera,
    eef_pose_camera_to_base,
    quat_to_rot6d,
)


def _arm_end_pose_7d(arm: Any) -> np.ndarray:
    pose = arm.ee_pose_base
    return np.array(
        [
            pose.position_m.x,
            pose.position_m.y,
            pose.position_m.z,
            pose.orientation_xyzw.x,
            pose.orientation_xyzw.y,
            pose.orientation_xyzw.z,
            pose.orientation_xyzw.w,
        ],
        dtype=np.float32,
    )


def _populate_legacy_arm_fields(
    new_obs: Dict[str, Any],
    left: Any,
    right: Any,
) -> None:
    """Fill per-arm joint / end-pose keys expected by legacy Policy.infer(new_obs)."""
    if left is not None:
        if left.joint_state.position_rad:
            new_obs['left_arm_joint_state'] = np.asarray(left.joint_state.position_rad, dtype=np.float32)
        if left.ee_pose_base.frame == 'base':
            new_obs['left_end_pose'] = _arm_end_pose_7d(left)
    if right is not None:
        if right.joint_state.position_rad:
            new_obs['right_arm_joint_state'] = np.asarray(right.joint_state.position_rad, dtype=np.float32)
        if right.ee_pose_base.frame == 'base':
            new_obs['right_end_pose'] = _arm_end_pose_7d(right)


def observation_packet_to_new_obs(
    packet: ObservationPacket,
    *,
    use_history: bool = False,
    observation_history: Optional[ObservationHistoryConfig] = None,
    tactile_config: Optional[TactileBenchmarkConfig] = None,
) -> Dict[str, Any]:
    """Convert canonical ObservationPacket to legacy new_obs for existing policies."""
    history_config = observation_history
    if history_config is None and use_history:
        history_config = ObservationHistoryConfig.for_use_history()
    new_obs: Dict[str, Any] = {'images': {}}
    prompt = packet.context.task_instruction or ''

    role_to_legacy = {
        CAMERA_ROLE_GLOBAL: 'cam_high',
        CAMERA_ROLE_LEFT_WRIST: 'cam_left_wrist',
        CAMERA_ROLE_RIGHT_WRIST: 'cam_right_wrist',
    }

    for cam in packet.camera_observations:
        if cam.frame_bytes is None and not cam.frame_history_bytes:
            continue
        frames = stack_camera_frames(cam)
        if not frames:
            continue
        legacy_name = role_to_legacy.get(cam.camera_role)
        if legacy_name:
            new_obs['images'][legacy_name] = frames[-1]
            depth = 1
            if history_config is not None:
                depth = int(history_config.camera_roles.get(cam.camera_role, 1))
            wants_history = depth > 1 or use_history
            if wants_history and len(frames) > 1 and cam.camera_role == CAMERA_ROLE_GLOBAL:
                new_obs['images']['cam_high_memory'] = _stack_frames(frames)

    # ViTAL wrist ACT: single active right arm → cam_right_wrist; alias for encode_new_obs.
    images = new_obs['images']
    if 'cam_wrist' not in images:
        if 'cam_right_wrist' in images:
            images['cam_wrist'] = images['cam_right_wrist']
        elif 'cam_left_wrist' in images:
            images['cam_wrist'] = images['cam_left_wrist']

    arms = {arm.arm_id: arm for arm in packet.robot_state.arms}
    left = arms.get(ARM_ID_LEFT)
    right = arms.get(ARM_ID_RIGHT)

    if left and left.joint_state.position_rad:
        new_obs['joint_qpos_left'] = np.asarray(left.joint_state.position_rad, dtype=np.float32)
    if right and right.joint_state.position_rad:
        new_obs['joint_qpos_right'] = np.asarray(right.joint_state.position_rad, dtype=np.float32)
    if right and right.joint_state.position_rad:
        new_obs['joint_qpos'] = new_obs['joint_qpos_right']
    elif left and left.joint_state.position_rad:
        new_obs['joint_qpos'] = new_obs['joint_qpos_left']

    _populate_legacy_arm_fields(new_obs, left, right)

    if left and right and left.ee_pose_base.frame == 'base':
        left_pose = left.ee_pose_base
        right_pose = right.ee_pose_base
        left_pose_arr = np.array(
            [
                left_pose.position_m.x,
                left_pose.position_m.y,
                left_pose.position_m.z,
                left_pose.orientation_xyzw.x,
                left_pose.orientation_xyzw.y,
                left_pose.orientation_xyzw.z,
                left_pose.orientation_xyzw.w,
            ],
            dtype=np.float32,
        )
        right_pose_arr = np.array(
            [
                right_pose.position_m.x,
                right_pose.position_m.y,
                right_pose.position_m.z,
                right_pose.orientation_xyzw.x,
                right_pose.orientation_xyzw.y,
                right_pose.orientation_xyzw.z,
                right_pose.orientation_xyzw.w,
            ],
            dtype=np.float32,
        )
        left_pose_cam = eef_pose_base_to_camera(left_pose_arr, is_left=True)
        right_pose_cam = eef_pose_base_to_camera(right_pose_arr, is_left=False)
        left_gripper = np.array([left.gripper.open_ratio], dtype=np.float32)
        right_gripper = np.array([right.gripper.open_ratio], dtype=np.float32)
        left_pose_9d = quat_to_rot6d(left_pose_cam)
        right_pose_9d = quat_to_rot6d(right_pose_cam)
        new_obs['state'] = np.concatenate([left_pose_9d, left_gripper, right_pose_9d, right_gripper])
    elif left and right:
        new_obs['state'] = np.concatenate(
            [
                np.asarray(left.joint_state.position_rad, dtype=np.float32),
                np.asarray(right.joint_state.position_rad, dtype=np.float32),
            ]
        )
    else:
        new_obs['state'] = np.zeros((32,), dtype=np.float32)

    if 'cam_high' in new_obs['images']:
        new_obs['first_frame'] = new_obs['images']['cam_high']
    new_obs['prompt'] = prompt
    new_obs['task_id'] = packet.context.task_id or ''

    if packet.tactile_observations:
        profile = tactile_config.tactile_profile if tactile_config else TactileBenchmarkConfig().tactile_profile
        new_obs['tactile'] = tactile_observations_to_legacy(packet.tactile_observations)
        new_obs['tactile_profile'] = profile
        tactile_history: Dict[str, Any] = {}
        for obs in packet.tactile_observations:
            role_history = _tactile_observation_history_to_legacy(obs)
            if role_history:
                tactile_history[str(obs.tactile_role)] = role_history
        if tactile_history:
            new_obs['tactile_history'] = tactile_history

    return new_obs


def _resolve_single_arm_id(control_arm: Optional[str], action_dim: int) -> Optional[str]:
    """Return arm id when *action_dim* is a single-arm joint vector (7/8 DoF)."""
    if action_dim not in (7, 8):
        return None
    arm = (control_arm or 'right').strip().lower()
    if arm == 'left':
        return ARM_ID_LEFT
    if arm == 'right':
        return ARM_ID_RIGHT
    raise ValueError(f"Unsupported control_arm={control_arm!r} for {action_dim}D joint actions")


def _normalize_action_format(action_format: str, action_dim: int) -> str:
    """Resolve action format; 10D vectors are always single-arm camera eef6d."""
    if action_dim >= 20:
        return 'eef6d'
    if action_dim == 10:
        return 'eef6d_single'
    if action_format == 'auto':
        return 'joint'
    return action_format


def _joint_arm_actions_from_vector(
    one: np.ndarray,
    *,
    control_arm: Optional[str] = None,
) -> List[ArmAction]:
    """Map a joint-space action vector to per-arm ArmAction entries."""
    dim = int(len(one))
    if dim == 10:
        raise ValueError(
            '10D actions are eef6d_single, not joint positions. '
            'Use action_format=eef6d_single or auto.'
        )
    single_arm_id = _resolve_single_arm_id(control_arm, dim)
    if single_arm_id is not None:
        return [
            ArmAction(
                arm_id=single_arm_id,
                joint_position_rad=[float(x) for x in one],
                gripper_target_open_ratio=float(one[-1]) if dim > 0 else None,
            ),
        ]

    half = dim // 2
    return [
        ArmAction(
            arm_id=ARM_ID_LEFT,
            joint_position_rad=[float(x) for x in one[:half]],
            gripper_target_open_ratio=float(one[half - 1]) if half > 0 else None,
        ),
        ArmAction(
            arm_id=ARM_ID_RIGHT,
            joint_position_rad=[float(x) for x in one[half:]],
            gripper_target_open_ratio=float(one[-1]) if dim > half else None,
        ),
    ]


def actions_array_to_action_packet(
    actions: np.ndarray,
    *,
    context: SessionContext,
    observation_timestamp_ns: int,
    action_format: str = 'auto',
    control_arm: Optional[str] = None,
    policy_latency_ms: float = 0.0,
) -> ActionPacket:
    """Convert legacy policy actions ndarray to canonical ActionPacket."""
    if actions.ndim == 1:
        actions = actions[None, :]

    if action_format == 'auto':
        dim = int(actions.shape[-1])
        if dim >= 20:
            action_format = 'eef6d'
        elif dim == 10:
            action_format = 'eef6d_single'
        else:
            action_format = 'joint'

    action_format = _normalize_action_format(action_format, int(actions.shape[-1]))

    steps: List[ActionStep] = []
    for rel_step, one in enumerate(actions):
        if action_format == 'eef6d':
            from scipy.spatial.transform import Rotation as R

            left_pos = one[0:3]
            left_rot6d = one[3:9]
            left_gripper = float(one[9])
            right_pos = one[10:13]
            right_rot6d = one[13:19]
            right_gripper = float(one[19])

            left_rotmat_cam = cont6d_to_matrix(left_rot6d)
            right_rotmat_cam = cont6d_to_matrix(right_rot6d)
            left_quat_cam = R.from_matrix(left_rotmat_cam).as_quat()
            right_quat_cam = R.from_matrix(right_rotmat_cam).as_quat()
            left_pose_cam = np.concatenate([left_pos, left_quat_cam], axis=0)
            right_pose_cam = np.concatenate([right_pos, right_quat_cam], axis=0)
            left_pose_base = eef_pose_camera_to_base(left_pose_cam, is_left=True)
            right_pose_base = eef_pose_camera_to_base(right_pose_cam, is_left=False)

            left_pose = Pose(
                position_m=Vector3(*left_pose_base[:3]),
                orientation_xyzw=Quaternion(*left_pose_base[3:7]),
                frame='base',
            )
            right_pose = Pose(
                position_m=Vector3(*right_pose_base[:3]),
                orientation_xyzw=Quaternion(*right_pose_base[3:7]),
                frame='base',
            )
            arm_actions = [
                ArmAction(arm_id=ARM_ID_LEFT, target_pose_base=left_pose, gripper_target_open_ratio=left_gripper),
                ArmAction(arm_id=ARM_ID_RIGHT, target_pose_base=right_pose, gripper_target_open_ratio=right_gripper),
            ]
            mode = ACTION_MODE_TASK_SPACE_ABSOLUTE
        elif action_format in ('eef6d_single', 'end_pose_base'):
            arm_name = (control_arm or 'right').strip().lower()
            if arm_name not in ('left', 'right'):
                arm_name = 'right'
            arm_id = ARM_ID_RIGHT if arm_name == 'right' else ARM_ID_LEFT
            if action_format == 'end_pose_base':
                if len(one) != 8:
                    raise ValueError(f'end_pose_base expects 8D [pose7, gripper], got dim={len(one)}')
                end_pose_8d = np.asarray(one, dtype=np.float64).ravel()
            else:
                from real_world_benchmark.policy.ViTAL.eef6d_utils import eef6d_single_to_base_end_pose

                end_pose_8d = eef6d_single_to_base_end_pose(one, control_arm=arm_name)
            pose = Pose(
                position_m=Vector3(*end_pose_8d[:3]),
                orientation_xyzw=Quaternion(*end_pose_8d[3:7]),
                frame='base',
            )
            arm_actions = [
                ArmAction(
                    arm_id=arm_id,
                    target_pose_base=pose,
                    gripper_target_open_ratio=float(end_pose_8d[7]),
                ),
            ]
            mode = ACTION_MODE_TASK_SPACE_ABSOLUTE
        else:
            arm_actions = _joint_arm_actions_from_vector(one, control_arm=control_arm)
            mode = ACTION_MODE_JOINT_ABSOLUTE

        steps.append(ActionStep(relative_step=rel_step, arm_actions=arm_actions))

    now_ns = time.time_ns()
    return ActionPacket(
        context=context,
        observation_timestamp_ns=observation_timestamp_ns,
        inference_timestamp_ns=now_ns,
        action_apply_timestamp_ns=now_ns,
        action_mode=mode,
        action_chunk=steps,
        expected_horizon_steps=len(steps),
        policy_latency_ms=policy_latency_ms,
        status='ok',
    )


def infer_output_to_action_packet(
    output: Dict[str, Any],
    *,
    context: SessionContext,
    observation_timestamp_ns: int,
    action_format: str = 'auto',
    control_arm: Optional[str] = None,
) -> ActionPacket:
    actions = output.get('actions')
    if actions is None:
        raise ValueError("Policy output must contain 'actions'")
    actions = np.asarray(actions)
    timing = output.get('policy_timing') or {}
    meta = output.get('policy_metadata') or {}
    latency = timing.get('infer_ms')
    policy_latency_ms = float(latency) if latency is not None else 0.0
    if action_format == 'auto':
        action_format = str(meta.get('action_format', 'auto'))
    if control_arm is None:
        control_arm = meta.get('control_arm')
    dim = int(np.asarray(actions).shape[-1])
    if action_format == 'joint' and dim == 10:
        action_format = str(meta.get('action_format', 'eef6d_single'))
    return actions_array_to_action_packet(
        actions,
        context=context,
        observation_timestamp_ns=observation_timestamp_ns,
        action_format=action_format,
        control_arm=control_arm,
        policy_latency_ms=policy_latency_ms,
    )


def action_packet_looks_like_misparsed_joint(packet: ActionPacket) -> bool:
    """Detect 10D eef vectors incorrectly stored as 5+5 joint targets."""
    from real_world_benchmark.worldarena.protocol import ACTION_MODE_JOINT_ABSOLUTE

    if packet.action_mode != ACTION_MODE_JOINT_ABSOLUTE:
        return False
    for step in packet.action_chunk:
        for arm in step.arm_actions:
            dim = len(arm.joint_position_rad or [])
            if 0 < dim < 7:
                return True
    return False


def misparsed_joint_packet_to_eef6d_actions(packet: ActionPacket) -> Optional[np.ndarray]:
    """Rebuild ``(N, 10)`` camera eef6d rows from a 5+5 joint misparsed packet."""
    rows: List[np.ndarray] = []
    for step in packet.action_chunk:
        left = next((a for a in step.arm_actions if a.arm_id == ARM_ID_LEFT), None)
        right = next((a for a in step.arm_actions if a.arm_id == ARM_ID_RIGHT), None)
        left_vals = list(left.joint_position_rad or []) if left else []
        right_vals = list(right.joint_position_rad or []) if right else []
        if len(left_vals) == 5 and len(right_vals) == 5:
            rows.append(np.asarray(left_vals + right_vals, dtype=np.float32))
        elif len(left_vals) == 5 and len(right_vals) == 0:
            return None
        elif len(right_vals) == 5 and len(left_vals) == 0:
            return None
        else:
            return None
    if not rows:
        return None
    return np.stack(rows, axis=0)


def repair_action_packet_if_misparsed(
    packet: ActionPacket,
    *,
    raw_actions: Optional[Any] = None,
    control_arm: Optional[str] = None,
) -> ActionPacket:
    """Convert a 5+5 joint misparsed packet back to single-arm task-space eef6d."""
    if not action_packet_looks_like_misparsed_joint(packet):
        return packet

    actions: Optional[np.ndarray]
    if raw_actions is not None:
        actions = np.asarray(raw_actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
    else:
        actions = misparsed_joint_packet_to_eef6d_actions(packet)

    if actions is None or actions.size == 0:
        raise ValueError(
            'ActionPacket looks like 10D eef6d misparsed as 5+5 joint targets, '
            'but raw 10D actions are unavailable for repair.'
        )
    if int(actions.shape[-1]) != 10:
        raise ValueError(
            f'Cannot repair misparsed joint ActionPacket: expected actions shape (*, 10), got {actions.shape}'
        )

    arm = (control_arm or 'right').strip().lower()
    if arm not in ('left', 'right'):
        arm = 'right'

    return infer_output_to_action_packet(
        {
            'actions': actions,
            'policy_metadata': {'action_format': 'eef6d_single', 'control_arm': arm},
            'policy_timing': {'infer_ms': packet.policy_latency_ms},
        },
        context=packet.context,
        observation_timestamp_ns=packet.observation_timestamp_ns,
        action_format='eef6d_single',
        control_arm=arm,
    )


def parse_policy_infer_response(
    data: Dict[str, Any],
    *,
    context: SessionContext,
    observation_timestamp_ns: int,
) -> ActionPacket:
    """Prefer canonical ActionPacket payloads; rebuild from raw actions only when needed."""
    chunk = data.get('action_chunk')
    if isinstance(chunk, list) and chunk:
        packet = ActionPacket.from_dict(data)
        meta = data.get('policy_metadata') or {}
        return repair_action_packet_if_misparsed(
            packet,
            raw_actions=data.get('actions'),
            control_arm=meta.get('control_arm'),
        )
    if 'actions' in data:
        packet = infer_output_to_action_packet(
            data,
            context=context,
            observation_timestamp_ns=observation_timestamp_ns,
        )
        meta = data.get('policy_metadata') or {}
        return repair_action_packet_if_misparsed(
            packet,
            raw_actions=data.get('actions'),
            control_arm=meta.get('control_arm'),
        )
    packet = ActionPacket.from_dict(data)
    return repair_action_packet_if_misparsed(packet)


def parse_policy_infer_full(
    data: Dict[str, Any],
    *,
    context: SessionContext,
    observation_timestamp_ns: int,
) -> 'PolicyInferResult':
    """Parse infer response into ActionPacket plus optional raw policy actions."""
    from real_world_benchmark.worldarena.action_debug import PolicyInferResult, as_actions_array
    from real_world_benchmark.worldarena.protocol import ACTION_MODE_TASK_SPACE_ABSOLUTE

    action_packet = parse_policy_infer_response(
        data,
        context=context,
        observation_timestamp_ns=observation_timestamp_ns,
    )
    raw_actions = as_actions_array(data.get('actions'))
    meta = dict(data.get('policy_metadata') or {})
    timing = data.get('policy_timing') or {}
    latency = timing.get('infer_ms')
    policy_latency_ms = float(latency) if latency is not None else float(action_packet.policy_latency_ms)

    if action_packet.action_mode == ACTION_MODE_TASK_SPACE_ABSOLUTE and action_packet.action_chunk:
        arm_ids = {
            arm.arm_id
            for step in action_packet.action_chunk
            for arm in step.arm_actions
            if arm.arm_id
        }
        if len(arm_ids) == 1:
            only_arm = next(iter(arm_ids))
            meta.setdefault('action_format', 'eef6d_single')
            meta.setdefault('control_arm', only_arm)

    return PolicyInferResult(
        action_packet=action_packet,
        raw_actions=raw_actions,
        policy_metadata=meta,
        policy_latency_ms=policy_latency_ms,
    )


def make_default_context(
    *,
    session_id: str = '',
    episode_id: str = '',
    task_id: str = '',
    task_instruction: str = '',
    embodiment_id: str = '',
    policy_id: str = '',
    adapter_version: str = '',
) -> SessionContext:
    return SessionContext(
        schema_version=SCHEMA_VERSION,
        session_id=session_id,
        episode_id=episode_id,
        task_id=task_id,
        task_instruction=task_instruction,
        embodiment_id=embodiment_id,
        embodiment_type='dual_arm',
        policy_id=policy_id,
        adapter_version=adapter_version,
    )
