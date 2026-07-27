"""Debug helpers for tracing actions vs ActionPacket through A/B/C pipeline."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from worldarena.schema import ActionPacket, ArmAction

logger = logging.getLogger(__name__)


def pipeline_debug_enabled() -> bool:
    return os.environ.get('VITAL_DEBUG_PIPELINE', '0') == '1'


def as_actions_array(raw: Any) -> Optional[np.ndarray]:
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float32)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _fmt_values(values: Any, *, limit: int = 12) -> str:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return '[]'
    shown = arr[:limit]
    text = np.array2string(shown, precision=6, separator=', ', suppress_small=True)
    if arr.size > limit:
        text = f'{text} ... (+{arr.size - limit} more)'
    return text


def summarize_actions_array(actions: Any) -> str:
    arr = as_actions_array(actions)
    if arr is None:
        return 'actions=None'
    first = arr[0] if arr.ndim >= 2 else arr
    return (
        f'actions shape={list(arr.shape)} dim={int(arr.shape[-1])} '
        f'first={_fmt_values(first)}'
    )


def _summarize_arm_action(arm: ArmAction) -> str:
    parts = [f'arm={arm.arm_id!r}']
    if arm.joint_position_rad:
        joints = list(arm.joint_position_rad)
        parts.append(f'joint_dim={len(joints)} joint={_fmt_values(joints)}')
    if arm.target_pose_base is not None:
        pose = arm.target_pose_base
        pose7 = [
            pose.position_m.x,
            pose.position_m.y,
            pose.position_m.z,
            pose.orientation_xyzw.x,
            pose.orientation_xyzw.y,
            pose.orientation_xyzw.z,
            pose.orientation_xyzw.w,
        ]
        parts.append(f'pose7={_fmt_values(pose7)}')
    if arm.gripper_target_open_ratio is not None:
        parts.append(f'gripper={float(arm.gripper_target_open_ratio):.6f}')
    return ' '.join(parts)


def summarize_action_packet_first_step(action_packet: Optional[ActionPacket]) -> str:
    if action_packet is None or not action_packet.action_chunk:
        return 'action_packet=None'
    step = action_packet.action_chunk[0]
    arm_lines = [_summarize_arm_action(arm) for arm in step.arm_actions]
    arms_text = ' | '.join(arm_lines) if arm_lines else 'no arm_actions'
    return (
        f'action_packet mode={action_packet.action_mode!r} chunk={len(action_packet.action_chunk)} '
        f'step0: {arms_text}'
    )


def summarize_manifold_first_step(params: Dict[str, Any]) -> str:
    method = params.get('method')
    if method == 'end_pose':
        left = params.get('left_end_pose') or []
        right = params.get('right_end_pose') or []
        left0 = left[0] if left else []
        right0 = right[0] if right else []
        return (
            f'manifold method=end_pose left_steps={len(left)} right_steps={len(right)} '
            f'left0_dim={len(left0)} left0={_fmt_values(left0)} '
            f'right0_dim={len(right0)} right0={_fmt_values(right0)}'
        )
    left = params.get('left_joint_states') or []
    right = params.get('right_joint_states') or []
    left0 = left[0] if left else []
    right0 = right[0] if right else []
    return (
        f'manifold method=joint left_steps={len(left)} right_steps={len(right)} '
        f'left0_dim={len(left0)} left0={_fmt_values(left0)} '
        f'right0_dim={len(right0)} right0={_fmt_values(right0)}'
    )


def log_pipeline_trace(
    stage: str,
    *,
    raw_actions: Any = None,
    action_packet: Optional[ActionPacket] = None,
    manifold_params: Optional[Dict[str, Any]] = None,
    extra: Optional[str] = None,
) -> None:
    if not pipeline_debug_enabled():
        return
    lines: List[str] = [f'[VITAL_DEBUG_PIPELINE] {stage}']
    if raw_actions is not None:
        lines.append(f'  {summarize_actions_array(raw_actions)}')
    if action_packet is not None:
        lines.append(f'  {summarize_action_packet_first_step(action_packet)}')
    if manifold_params is not None:
        lines.append(f'  {summarize_manifold_first_step(manifold_params)}')
    if extra:
        lines.append(f'  {extra}')
    logger.info('\n'.join(lines))


def extract_base8d_step0(
    action_packet: ActionPacket,
    *,
    arm_id: str = 'right',
) -> Optional[np.ndarray]:
    """First-step base-frame end pose ``[pose7, gripper]`` from ActionPacket."""
    if not action_packet.action_chunk:
        return None
    for arm in action_packet.action_chunk[0].arm_actions:
        if arm.arm_id != arm_id or arm.target_pose_base is None:
            continue
        pose = arm.target_pose_base
        gripper = float(arm.gripper_target_open_ratio or 0.0)
        return np.array(
            [
                pose.position_m.x,
                pose.position_m.y,
                pose.position_m.z,
                pose.orientation_xyzw.x,
                pose.orientation_xyzw.y,
                pose.orientation_xyzw.z,
                pose.orientation_xyzw.w,
                gripper,
            ],
            dtype=np.float32,
        )
    return None


def base8d_to_eef6d10(end_pose_8d: np.ndarray, *, control_arm: str = 'right') -> np.ndarray:
    """Convert base-frame 8D end pose back to 10D camera-frame eef6d."""
    from worldarena.geometry import eef_pose_base_to_camera, quat_to_rot6d

    vec = np.asarray(end_pose_8d, dtype=np.float32).ravel()
    if vec.size < 7:
        raise ValueError(f'Expected at least 7D base pose, got {vec.size}')
    gripper = float(vec[7]) if vec.size > 7 else 0.0
    pose_cam = eef_pose_base_to_camera(vec[:7], is_left=(control_arm == 'left'))
    row9 = quat_to_rot6d(pose_cam)
    return np.concatenate([row9, [gripper]], axis=0).astype(np.float32)


def _resolve_control_arm(
    action_packet: ActionPacket,
    *,
    fallback: str = 'right',
) -> str:
    if action_packet.action_chunk:
        active = [a.arm_id for a in action_packet.action_chunk[0].arm_actions if a.arm_id]
        if len(active) == 1:
            return active[0]
    return fallback


def log_action_cross_check(
    *,
    action_packet: ActionPacket,
    raw_actions_10d: Any = None,
    manifold_right0: Optional[List[float]] = None,
    control_arm: Optional[str] = None,
) -> None:
    """Compare A-side raw 10D eef6d with ActionPacket base 8D on C."""
    if not pipeline_debug_enabled():
        return

    arm = control_arm or _resolve_control_arm(action_packet)
    base8d = extract_base8d_step0(action_packet, arm_id=arm)
    raw10 = as_actions_array(raw_actions_10d)
    raw_row = raw10[0] if raw10 is not None and raw10.ndim >= 2 else raw10

    lines: List[str] = [f'[VITAL_DEBUG_PIPELINE] C:cross_check arm={arm!r}']
    roundtrip = None
    if base8d is not None:
        lines.append(f'  packet_base8D={_fmt_values(base8d, limit=10)}')
        roundtrip = base8d_to_eef6d10(base8d, control_arm=arm)
        lines.append(f'  packet_roundtrip_10D={_fmt_values(roundtrip, limit=12)}')
    else:
        lines.append('  packet_base8D=missing (not task_space_absolute?)')

    if raw_row is not None and int(np.asarray(raw_row).size) == 10:
        raw_arr = np.asarray(raw_row, dtype=np.float64).ravel()
        lines.append(f'  A_raw_10D={_fmt_values(raw_arr, limit=12)}')
        if roundtrip is not None:
            delta = raw_arr - np.asarray(roundtrip, dtype=np.float64).ravel()
            lines.append(
                f'  delta_A_vs_packet_roundtrip max_abs={float(np.max(np.abs(delta))):.6g} '
                f'l2={float(np.linalg.norm(delta)):.6g}'
            )
    else:
        lines.append('  A_raw_10D=not forwarded (set VITAL_DEBUG_PIPELINE=1 on B to attach debug_raw_actions)')

    if manifold_right0 is not None:
        lines.append(f'  manifold_right0_8D={_fmt_values(manifold_right0, limit=10)}')
        if base8d is not None and len(manifold_right0) >= 8:
            delta8 = np.asarray(manifold_right0[:8], dtype=np.float64) - np.asarray(base8d[:8], dtype=np.float64)
            lines.append(
                f'  delta_packet8D_vs_manifold max_abs={float(np.max(np.abs(delta8))):.6g} '
                f'l2={float(np.linalg.norm(delta8)):.6g}'
            )

    logger.info('\n'.join(lines))


def debug_raw_actions_for_send(raw_actions: Any) -> Optional[Any]:
    """Attach A-side raw actions to robot apply requests when pipeline debug is on."""
    if not pipeline_debug_enabled():
        return None
    arr = as_actions_array(raw_actions)
    if arr is None:
        return None
    return arr.tolist()


@dataclass
class PolicyInferResult:
    action_packet: ActionPacket
    raw_actions: Optional[np.ndarray] = None
    policy_metadata: Dict[str, Any] = field(default_factory=dict)
    policy_latency_ms: float = 0.0
