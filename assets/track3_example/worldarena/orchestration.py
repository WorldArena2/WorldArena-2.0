"""Orchestration helpers for canonical WorldArena live evaluation."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from worldarena.action_debug import debug_raw_actions_for_send, log_pipeline_trace
from worldarena.adapters.manifold_msg import ManifoldMsgAdapter
from worldarena.bridges.legacy_policy import (
    action_packet_looks_like_misparsed_joint,
    actions_array_to_action_packet,
    infer_output_to_action_packet,
    make_default_context,
    observation_packet_to_new_obs,
    parse_policy_infer_response,
    repair_action_packet_if_misparsed,
)
from worldarena.hub_orchestrator import HubPolicyClient, HubRobotClient
from worldarena.policy_remote import CanonicalPolicyClient
from worldarena.robot_remote import RemoteRobotClient, observation_packet_to_private
from worldarena.schema import ActionPacket, ObservationPacket, SessionContext
from worldarena.observation_history import (
    ObservationHistoryCapabilities,
    ObservationHistoryConfig,
    validate_observation_history_capabilities,
)
from worldarena.tactile import TactileBenchmarkConfig, validate_tactile_observations

logger = logging.getLogger(__name__)


def uses_canonical_policy(args: Any, policy: Any) -> bool:
    if getattr(args, 'policy_protocol', 'rwb-policy-v1') == 'wa-policy-v1':
        return True
    return callable(getattr(policy, 'infer_packet', None))


def uses_canonical_robot(server: Any) -> bool:
    return isinstance(server, (ManifoldMsgAdapter, RemoteRobotClient, HubRobotClient))


def bind_hub_session(policy: Any, server: Any, *, session_id: str, args: Any) -> None:
    if not getattr(args, 'hub_mode', False):
        return
    policy_key = getattr(args, 'hub_policy_key', '') or getattr(args, 'policy_id', '')
    robot_key = getattr(args, 'hub_robot_key', '') or getattr(args, 'site_id', '')
    if isinstance(policy, HubPolicyClient):
        policy.bind_session(session_id, policy_worker_key=policy_key, robot_worker_key=robot_key)
    if isinstance(server, HubRobotClient):
        server.bind_session(session_id, policy_worker_key=policy_key, robot_worker_key=robot_key)


def robot_metadata(server: Any) -> Dict[str, str]:
    if isinstance(server, (RemoteRobotClient, HubRobotClient)):
        return server.metadata.to_dict()
    if hasattr(server, 'metadata') and callable(server.metadata):
        data = server.metadata()
        return {str(k): str(v) if not isinstance(v, list) else ','.join(str(x) for x in v) for k, v in data.items()}
    if isinstance(server, ManifoldMsgAdapter):
        return {str(k): str(v) if not isinstance(v, list) else ','.join(str(x) for x in v) for k, v in server.metadata().items()}
    return {}


def build_session_context(
    *,
    args: Any,
    task_id: str = '',
    task_instruction: str = '',
    episode_id: str = '',
    session_id: str = '',
    server: Any,
) -> SessionContext:
    robot = robot_metadata(server)
    policy_id = getattr(args, 'policy_id', '') or ''
    return make_default_context(
        session_id=session_id or episode_id,
        episode_id=episode_id,
        task_id=task_id,
        task_instruction=task_instruction,
        embodiment_id=robot.get('embodiment_id', getattr(args, 'site_id', '') or ''),
        policy_id=policy_id,
        adapter_version=robot.get('adapter_version', ''),
    )


def set_server_context(server: Any, context: SessionContext) -> None:
    if hasattr(server, 'set_context'):
        server.set_context(context)


def set_server_tactile_config(server: Any, config: TactileBenchmarkConfig) -> None:
    if hasattr(server, 'set_tactile_config'):
        server.set_tactile_config(config)


def set_server_observation_history_config(server: Any, config: ObservationHistoryConfig) -> None:
    if hasattr(server, 'set_observation_history_config'):
        server.set_observation_history_config(config)


def fetch_live_observation(
    server: Any,
    *,
    context: SessionContext,
    default_prompt: str,
    make_live_new_obs,
    use_history: bool,
    stack_frames,
    tactile_config: Optional[TactileBenchmarkConfig] = None,
    observation_history: Optional[ObservationHistoryConfig] = None,
) -> Tuple[Optional[ObservationPacket], Dict[str, Any], Dict[str, Any]]:
    history_config = observation_history
    if history_config is None and use_history:
        history_config = ObservationHistoryConfig.for_use_history()
    if uses_canonical_robot(server):
        set_server_context(server, context)
        if tactile_config is not None:
            set_server_tactile_config(server, tactile_config)
        if history_config is not None:
            set_server_observation_history_config(server, history_config)
            if isinstance(server, RemoteRobotClient):
                caps = ObservationHistoryCapabilities.from_dict(server.metadata.to_dict())
                validate_observation_history_capabilities(history_config, caps)
            elif isinstance(server, HubRobotClient):
                caps = ObservationHistoryCapabilities.from_dict(server.metadata.to_dict())
                validate_observation_history_capabilities(history_config, caps)
        if hasattr(server, 'wait_observation_packet'):
            obs_packet = server.wait_observation_packet()
        else:
            adapter = server
            private_obs = adapter.wait_private_observation()
            obs_packet = adapter.private_observation_to_canonical(private_obs, context=context)
        if tactile_config is not None:
            validate_tactile_observations(obs_packet.tactile_observations, tactile_config)
        obs = observation_packet_to_private(obs_packet)
        new_obs = observation_packet_to_new_obs(
            obs_packet,
            use_history=use_history,
            observation_history=history_config,
            tactile_config=tactile_config,
        )
        if (
            use_history
            and 'cam_high_memory' not in new_obs.get('images', {})
            and 'img_front' in obs
            and isinstance(obs['img_front'], list)
            and len(obs['img_front']) > 1
        ):
            new_obs['images']['cam_high_memory'] = stack_frames(obs['img_front'])
        if not new_obs.get('prompt'):
            new_obs['prompt'] = default_prompt
        return obs_packet, obs, new_obs

    obs = server.wait_observation()
    new_obs = make_live_new_obs(obs, default_prompt)
    if use_history and 'img_front' in obs and 'cam_high_memory' not in new_obs.get('images', {}):
        new_obs['images']['cam_high_memory'] = stack_frames(obs['img_front'])
    return None, obs, new_obs


def run_policy_step(
    policy: Any,
    args: Any,
    *,
    obs_packet: Optional[ObservationPacket],
    new_obs: Dict[str, Any],
    validate_output,
) -> Tuple[Dict[str, Any], np.ndarray, Optional[ActionPacket]]:
    from worldarena.protocol import ACTION_MODE_TASK_SPACE_ABSOLUTE

    if uses_canonical_policy(args, policy) and obs_packet is not None and isinstance(
        policy, (CanonicalPolicyClient, HubPolicyClient)
    ):
        infer_result = policy.infer_policy_result(obs_packet)
        action_packet = infer_result.action_packet
        meta = dict(infer_result.policy_metadata)
        if infer_result.raw_actions is not None and infer_result.raw_actions.size:
            actions = infer_result.raw_actions
        else:
            logger.warning(
                'Policy infer response missing raw actions; falling back to legacy array rebuild'
            )
            actions = _fallback_actions_from_action_packet(action_packet)
        output: Dict[str, Any] = {
            'actions': actions,
            'policy_timing': {'infer_ms': infer_result.policy_latency_ms},
        }
        if meta:
            output['policy_metadata'] = meta
        elif action_packet.action_mode == ACTION_MODE_TASK_SPACE_ABSOLUTE and action_packet.action_chunk:
            arm_ids = {
                arm.arm_id
                for step in action_packet.action_chunk
                for arm in step.arm_actions
                if arm.arm_id
            }
            if len(arm_ids) == 1:
                only_arm = next(iter(arm_ids))
                output['policy_metadata'] = {
                    'action_format': 'eef6d_single',
                    'control_arm': only_arm,
                }
        log_pipeline_trace(
            'B:after_infer',
            raw_actions=actions,
            action_packet=action_packet,
        )
        return output, actions, action_packet

    output = policy.infer(new_obs)
    actions = validate_output(output)
    action_packet = None
    if obs_packet is not None:
        action_packet = infer_output_to_action_packet(
            output,
            context=obs_packet.context,
            observation_timestamp_ns=obs_packet.observation_timestamp_ns,
        )
        meta = output.get('policy_metadata') or {}
        action_packet = repair_action_packet_if_misparsed(
            action_packet,
            raw_actions=output.get('actions'),
            control_arm=meta.get('control_arm'),
        )
        log_pipeline_trace(
            'B:after_infer',
            raw_actions=actions,
            action_packet=action_packet,
        )
    return output, actions, action_packet


def send_live_action(
    server: Any,
    *,
    obs: Dict[str, Any],
    actions: np.ndarray,
    action_packet: Optional[ActionPacket],
    action_rate: int,
    action_format: str,
    send_actions_to_server,
    policy_output: Optional[Dict[str, Any]] = None,
) -> None:
    if action_packet is not None:
        meta = dict((policy_output or {}).get('policy_metadata') or {})
        raw = (policy_output or {}).get('actions')
        if action_packet_looks_like_misparsed_joint(action_packet):
            import logging

            logging.getLogger(__name__).warning(
                'ActionPacket looks like misparsed 5+5 joint targets; repairing as eef6d_single'
            )
        action_packet = repair_action_packet_if_misparsed(
            action_packet,
            raw_actions=raw,
            control_arm=meta.get('control_arm'),
        )
    if action_packet is not None and uses_canonical_robot(server):
        log_pipeline_trace(
            'B:before_send_robot',
            raw_actions=(policy_output or {}).get('actions'),
            action_packet=action_packet,
        )
        if isinstance(server, (RemoteRobotClient, HubRobotClient)):
            server.send_action_packet(
                action_packet,
                debug_raw_actions=debug_raw_actions_for_send((policy_output or {}).get('actions')),
            )
        elif isinstance(server, ManifoldMsgAdapter):
            server.apply_action(
                action_packet,
                obs,
                debug_raw_actions=debug_raw_actions_for_send((policy_output or {}).get('actions')),
            )
        return
    send_actions_to_server(server, obs, actions, action_rate, action_format)


def _fallback_actions_from_action_packet(action_packet: ActionPacket) -> np.ndarray:
    """Legacy fallback when policy infer response omits raw actions."""
    from worldarena.geometry import eef_pose_base_to_camera, quat_to_rot6d
    from worldarena.protocol import ACTION_MODE_JOINT_ABSOLUTE, ARM_ID_LEFT, ARM_ID_RIGHT

    rows = []
    for step in action_packet.action_chunk:
        if action_packet.action_mode == ACTION_MODE_JOINT_ABSOLUTE:
            left = next((a for a in step.arm_actions if a.arm_id == ARM_ID_LEFT), None)
            right = next((a for a in step.arm_actions if a.arm_id == ARM_ID_RIGHT), None)
            left_vals = list(left.joint_position_rad or []) if left else []
            right_vals = list(right.joint_position_rad or []) if right else []
            if left_vals and right_vals:
                rows.append(np.asarray(left_vals + right_vals, dtype=np.float32))
            elif right_vals:
                rows.append(np.asarray(right_vals, dtype=np.float32))
            elif left_vals:
                rows.append(np.asarray(left_vals, dtype=np.float32))
            continue

        active = [a for a in step.arm_actions if a.target_pose_base is not None]
        if len(active) == 1:
            arm = active[0]
            pose = np.array(
                [
                    arm.target_pose_base.position_m.x,
                    arm.target_pose_base.position_m.y,
                    arm.target_pose_base.position_m.z,
                    arm.target_pose_base.orientation_xyzw.x,
                    arm.target_pose_base.orientation_xyzw.y,
                    arm.target_pose_base.orientation_xyzw.z,
                    arm.target_pose_base.orientation_xyzw.w,
                ],
                dtype=np.float32,
            )
            is_left = arm.arm_id == ARM_ID_LEFT
            pose_cam = eef_pose_base_to_camera(pose, is_left=is_left)
            gripper = float(arm.gripper_target_open_ratio or 0.0)
            row9 = quat_to_rot6d(pose_cam)
            rows.append(np.concatenate([row9, [gripper]], axis=0).astype(np.float32))
            continue

        left = next((a for a in step.arm_actions if a.arm_id == ARM_ID_LEFT), None)
        right = next((a for a in step.arm_actions if a.arm_id == ARM_ID_RIGHT), None)
        if left is None or right is None or left.target_pose_base is None or right.target_pose_base is None:
            continue
        left_pose = np.array(
            [
                left.target_pose_base.position_m.x,
                left.target_pose_base.position_m.y,
                left.target_pose_base.position_m.z,
                left.target_pose_base.orientation_xyzw.x,
                left.target_pose_base.orientation_xyzw.y,
                left.target_pose_base.orientation_xyzw.z,
                left.target_pose_base.orientation_xyzw.w,
            ],
            dtype=np.float32,
        )
        right_pose = np.array(
            [
                right.target_pose_base.position_m.x,
                right.target_pose_base.position_m.y,
                right.target_pose_base.position_m.z,
                right.target_pose_base.orientation_xyzw.x,
                right.target_pose_base.orientation_xyzw.y,
                right.target_pose_base.orientation_xyzw.z,
                right.target_pose_base.orientation_xyzw.w,
            ],
            dtype=np.float32,
        )
        left_pose_cam = eef_pose_base_to_camera(left_pose, is_left=True)
        right_pose_cam = eef_pose_base_to_camera(right_pose, is_left=False)
        left_gripper = float(left.gripper_target_open_ratio or 0.0)
        right_gripper = float(right.gripper_target_open_ratio or 0.0)
        left9 = quat_to_rot6d(left_pose_cam)
        right9 = quat_to_rot6d(right_pose_cam)
        row = np.concatenate([left9, [left_gripper], right9, [right_gripper], np.zeros(12, dtype=np.float32)])
        rows.append(row[:32])

    if not rows:
        return np.zeros((0, 10), dtype=np.float32)
    return np.stack(rows, axis=0)


def make_action_packet_for_send(
    *,
    server: Any,
    obs: Dict[str, Any],
    actions: np.ndarray,
    context: SessionContext,
    action_format: str,
    control_arm: Optional[str] = None,
) -> ActionPacket:
    timestamp_ns = int(float(obs.get('timestamp', 0)) * 1_000_000_000)
    packet = actions_array_to_action_packet(
        actions,
        context=context,
        observation_timestamp_ns=timestamp_ns,
        action_format=action_format,
        control_arm=control_arm,
    )
    return repair_action_packet_if_misparsed(
        packet,
        raw_actions=actions,
        control_arm=control_arm,
    )
