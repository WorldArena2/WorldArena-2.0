"""Debug helpers for tracing Policy actions vs ActionPacket on the A-side."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from real_world_benchmark.worldarena.schema import ActionPacket, ArmAction

logger = logging.getLogger(__name__)


def pipeline_debug_enabled() -> bool:
    return os.environ.get('VITAL_DEBUG_PIPELINE', '0') == '1' or os.environ.get('WA_DEBUG_PIPELINE', '0') == '1'


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


def _summarize_arm_action(arm: ArmAction) -> str:
    parts: List[str] = [f'arm={arm.arm_id!r}']
    if arm.target_joint_position_rad is not None:
        parts.append(f'qpos={_fmt_values(arm.target_joint_position_rad, limit=8)}')
    if arm.target_pose_base is not None:
        pose = arm.target_pose_base
        parts.append(
            'pose_base='
            + _fmt_values(
                [
                    pose.position_m.x,
                    pose.position_m.y,
                    pose.position_m.z,
                    pose.orientation_xyzw.x,
                    pose.orientation_xyzw.y,
                    pose.orientation_xyzw.z,
                    pose.orientation_xyzw.w,
                ],
                limit=8,
            )
        )
    if arm.gripper_target_open_ratio is not None:
        parts.append(f'gripper={float(arm.gripper_target_open_ratio):.4g}')
    return ' '.join(parts)


def summarize_actions_array(raw_actions: Any) -> str:
    arr = as_actions_array(raw_actions)
    if arr is None:
        return 'actions=None'
    return f'actions shape={arr.shape} step0={_fmt_values(arr[0])}'


def summarize_action_packet_first_step(action_packet: ActionPacket) -> str:
    if not action_packet.action_chunk:
        return f'action_packet mode={action_packet.action_mode!r} chunk=0'
    step = action_packet.action_chunk[0]
    arm_lines = [_summarize_arm_action(arm) for arm in step.arm_actions]
    arms_text = ' | '.join(arm_lines) if arm_lines else 'no arm_actions'
    return (
        f'action_packet mode={action_packet.action_mode!r} chunk={len(action_packet.action_chunk)} '
        f'step0: {arms_text}'
    )


def log_pipeline_trace(
    stage: str,
    *,
    raw_actions: Any = None,
    action_packet: Optional[ActionPacket] = None,
    extra: Optional[str] = None,
) -> None:
    if not pipeline_debug_enabled():
        return
    lines: List[str] = [f'[WA_DEBUG_PIPELINE] {stage}']
    if raw_actions is not None:
        lines.append(f'  {summarize_actions_array(raw_actions)}')
    if action_packet is not None:
        lines.append(f'  {summarize_action_packet_first_step(action_packet)}')
    if extra:
        lines.append(f'  {extra}')
    logger.info('\n'.join(lines))


@dataclass
class PolicyInferResult:
    action_packet: ActionPacket
    raw_actions: Optional[np.ndarray] = None
    policy_metadata: Dict[str, Any] = field(default_factory=dict)
    policy_latency_ms: float = 0.0
