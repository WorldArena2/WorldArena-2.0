"""WorldArena 2.0 canonical schema dataclasses."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

from worldarena.protocol import SCHEMA_VERSION


def _vec3_to_dict(value: Optional['Vector3']) -> Optional[Dict[str, float]]:
    if value is None:
        return None
    return dataclasses.asdict(value)


def _quat_to_dict(value: Optional['Quaternion']) -> Optional[Dict[str, float]]:
    if value is None:
        return None
    return dataclasses.asdict(value)


@dataclasses.dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'Vector3':
        if not data:
            return cls()
        return cls(
            x=float(data.get('x', 0.0)),
            y=float(data.get('y', 0.0)),
            z=float(data.get('z', 0.0)),
        )


@dataclasses.dataclass
class Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'Quaternion':
        if not data:
            return cls()
        return cls(
            x=float(data.get('x', 0.0)),
            y=float(data.get('y', 0.0)),
            z=float(data.get('z', 0.0)),
            w=float(data.get('w', 1.0)),
        )


@dataclasses.dataclass
class Pose:
    position_m: Vector3 = dataclasses.field(default_factory=Vector3)
    orientation_xyzw: Quaternion = dataclasses.field(default_factory=Quaternion)
    frame: str = 'base'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'position_m': _vec3_to_dict(self.position_m),
            'orientation_xyzw': _quat_to_dict(self.orientation_xyzw),
            'frame': self.frame,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'Pose':
        if not data:
            return cls()
        return cls(
            position_m=Vector3.from_dict(data.get('position_m')),
            orientation_xyzw=Quaternion.from_dict(data.get('orientation_xyzw')),
            frame=str(data.get('frame', 'base')),
        )


@dataclasses.dataclass
class SessionContext:
    schema_version: str = SCHEMA_VERSION
    session_id: str = ''
    episode_id: str = ''
    task_id: str = ''
    task_instruction: str = ''
    embodiment_id: str = ''
    embodiment_type: str = ''
    policy_id: str = ''
    policy_interface_version: str = ''
    adapter_version: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'SessionContext':
        if not data:
            return cls()
        return cls(
            schema_version=str(data.get('schema_version', SCHEMA_VERSION)),
            session_id=str(data.get('session_id', '')),
            episode_id=str(data.get('episode_id', '')),
            task_id=str(data.get('task_id', '')),
            task_instruction=str(data.get('task_instruction', '')),
            embodiment_id=str(data.get('embodiment_id', '')),
            embodiment_type=str(data.get('embodiment_type', '')),
            policy_id=str(data.get('policy_id', '')),
            policy_interface_version=str(data.get('policy_interface_version', '')),
            adapter_version=str(data.get('adapter_version', '')),
        )


@dataclasses.dataclass
class JointState:
    joint_names: List[str] = dataclasses.field(default_factory=list)
    position_rad: List[float] = dataclasses.field(default_factory=list)
    velocity_rad_s: List[float] = dataclasses.field(default_factory=list)
    effort: List[float] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'JointState':
        if not data:
            return cls()
        return cls(
            joint_names=[str(x) for x in data.get('joint_names', [])],
            position_rad=[float(x) for x in data.get('position_rad', data.get('joint_position_rad', []))],
            velocity_rad_s=[float(x) for x in data.get('velocity_rad_s', data.get('joint_velocity_rad_s', []))],
            effort=[float(x) for x in data.get('effort', [])],
        )


@dataclasses.dataclass
class GripperState:
    open_ratio: float = 0.0
    width_m: Optional[float] = None
    is_holding_object: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {'open_ratio': self.open_ratio}
        if self.width_m is not None:
            out['width_m'] = self.width_m
        if self.is_holding_object is not None:
            out['is_holding_object'] = self.is_holding_object
        return out

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'GripperState':
        if not data:
            return cls()
        return cls(
            open_ratio=float(data.get('open_ratio', 0.0)),
            width_m=data.get('width_m'),
            is_holding_object=data.get('is_holding_object'),
        )


@dataclasses.dataclass
class ArmState:
    arm_id: str = ''
    joint_state: JointState = dataclasses.field(default_factory=JointState)
    ee_pose_base: Pose = dataclasses.field(default_factory=Pose)
    gripper: GripperState = dataclasses.field(default_factory=GripperState)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'arm_id': self.arm_id,
            'joint_state': self.joint_state.to_dict(),
            'joint_names': self.joint_state.joint_names,
            'joint_position_rad': self.joint_state.position_rad,
            'joint_velocity_rad_s': self.joint_state.velocity_rad_s,
            'ee_pose_base': self.ee_pose_base.to_dict(),
            'gripper': self.gripper.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'ArmState':
        if not data:
            return cls()
        joint_state = JointState.from_dict(data.get('joint_state') or data)
        return cls(
            arm_id=str(data.get('arm_id', '')),
            joint_state=joint_state,
            ee_pose_base=Pose.from_dict(data.get('ee_pose_base')),
            gripper=GripperState.from_dict(data.get('gripper')),
        )


@dataclasses.dataclass
class RobotState:
    arms: List[ArmState] = dataclasses.field(default_factory=list)
    base_pose_world: Optional[Pose] = None
    raw_state_refs: Dict[str, str] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {'arms': [arm.to_dict() for arm in self.arms]}
        if self.base_pose_world is not None:
            out['base_pose_world'] = self.base_pose_world.to_dict()
        if self.raw_state_refs:
            out['raw_state_refs'] = dict(self.raw_state_refs)
        return out

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'RobotState':
        if not data:
            return cls()
        return cls(
            arms=[ArmState.from_dict(item) for item in data.get('arms', [])],
            base_pose_world=Pose.from_dict(data.get('base_pose_world')) if data.get('base_pose_world') else None,
            raw_state_refs={str(k): str(v) for k, v in (data.get('raw_state_refs') or {}).items()},
        )


@dataclasses.dataclass
class CameraIntrinsics:
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    distortion: List[float] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'CameraIntrinsics':
        if not data:
            return cls()
        return cls(
            fx=float(data.get('fx', 0.0)),
            fy=float(data.get('fy', 0.0)),
            cx=float(data.get('cx', 0.0)),
            cy=float(data.get('cy', 0.0)),
            distortion=[float(x) for x in data.get('distortion', [])],
        )


@dataclasses.dataclass
class CameraExtrinsics:
    camera_pose_parent: Optional[Pose] = None
    parent_frame: str = ''

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {'parent_frame': self.parent_frame}
        if self.camera_pose_parent is not None:
            out['camera_pose_parent'] = self.camera_pose_parent.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'CameraExtrinsics':
        if not data:
            return cls()
        pose = data.get('camera_pose_parent')
        return cls(
            camera_pose_parent=Pose.from_dict(pose) if pose else None,
            parent_frame=str(data.get('parent_frame', '')),
        )


@dataclasses.dataclass
class CameraObservation:
    camera_role: str = ''
    modality: str = 'rgb'
    frame_id: str = ''
    timestamp_ns: int = 0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    encoding: str = 'rgb8'
    intrinsics: Optional[CameraIntrinsics] = None
    extrinsics: Optional[CameraExtrinsics] = None
    frame_bytes: Optional[bytes] = None
    transport_ref: str = ''
    # Older frames, oldest-first; ``frame_bytes`` is always the latest frame.
    frame_history_bytes: List[bytes] = dataclasses.field(default_factory=list)
    frame_history_timestamps_ns: List[int] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            'camera_role': self.camera_role,
            'modality': self.modality,
            'frame_id': self.frame_id,
            'timestamp_ns': self.timestamp_ns,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'encoding': self.encoding,
        }
        if self.intrinsics is not None:
            out['intrinsics'] = self.intrinsics.to_dict()
        if self.extrinsics is not None:
            out['extrinsics'] = self.extrinsics.to_dict()
        if self.frame_bytes is not None:
            out['frame_bytes'] = self.frame_bytes
        if self.transport_ref:
            out['transport_ref'] = self.transport_ref
        if self.frame_history_bytes:
            out['frame_history_bytes'] = list(self.frame_history_bytes)
        if self.frame_history_timestamps_ns:
            out['frame_history_timestamps_ns'] = [int(x) for x in self.frame_history_timestamps_ns]
        return out

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'CameraObservation':
        if not data:
            return cls()
        intrinsics = data.get('intrinsics')
        extrinsics = data.get('extrinsics')
        history_bytes = data.get('frame_history_bytes') or []
        history_ts = data.get('frame_history_timestamps_ns') or []
        return cls(
            camera_role=str(data.get('camera_role', '')),
            modality=str(data.get('modality', 'rgb')),
            frame_id=str(data.get('frame_id', '')),
            timestamp_ns=int(data.get('timestamp_ns', 0)),
            width=int(data.get('width', 0)),
            height=int(data.get('height', 0)),
            fps=float(data.get('fps', 0.0)),
            encoding=str(data.get('encoding', 'rgb8')),
            intrinsics=CameraIntrinsics.from_dict(intrinsics) if intrinsics else None,
            extrinsics=CameraExtrinsics.from_dict(extrinsics) if extrinsics else None,
            frame_bytes=data.get('frame_bytes'),
            transport_ref=str(data.get('transport_ref', '')),
            frame_history_bytes=[bytes(item) for item in history_bytes],
            frame_history_timestamps_ns=[int(x) for x in history_ts],
        )


@dataclasses.dataclass
class TactileField:
    field_type: str = ''
    shape: List[int] = dataclasses.field(default_factory=list)
    dtype: str = ''
    units: str = ''
    encoding: str = 'raw'
    data_bytes: Optional[bytes] = None
    transport_ref: str = ''

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            'field_type': self.field_type,
            'shape': list(self.shape),
            'dtype': self.dtype,
            'encoding': self.encoding,
        }
        if self.units:
            out['units'] = self.units
        if self.data_bytes is not None:
            out['data_bytes'] = self.data_bytes
        if self.transport_ref:
            out['transport_ref'] = self.transport_ref
        return out

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'TactileField':
        if not data:
            return cls()
        return cls(
            field_type=str(data.get('field_type', '')),
            shape=[int(x) for x in data.get('shape', [])],
            dtype=str(data.get('dtype', '')),
            units=str(data.get('units', '')),
            encoding=str(data.get('encoding', 'raw')),
            data_bytes=data.get('data_bytes'),
            transport_ref=str(data.get('transport_ref', '')),
        )


@dataclasses.dataclass
class TactileObservation:
    tactile_role: str = ''
    sensor_vendor: str = ''
    sensor_serial: str = ''
    sensor_model: str = ''
    timestamp_ns: int = 0
    frame_id: str = ''
    contact_state: Optional[bool] = None
    contact_confidence: Optional[float] = None
    wrench_6d: Optional[List[float]] = None
    fields: List[TactileField] = dataclasses.field(default_factory=list)
    runtime_config_ref: str = ''
    raw_refs: Dict[str, str] = dataclasses.field(default_factory=dict)
    # Older tactile snapshots (each entry is a full ``fields`` list), oldest-first.
    fields_history: List[List[TactileField]] = dataclasses.field(default_factory=list)
    fields_history_timestamps_ns: List[int] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            'tactile_role': self.tactile_role,
            'sensor_vendor': self.sensor_vendor,
            'timestamp_ns': self.timestamp_ns,
            'fields': [field.to_dict() for field in self.fields],
        }
        if self.sensor_serial:
            out['sensor_serial'] = self.sensor_serial
        if self.sensor_model:
            out['sensor_model'] = self.sensor_model
        if self.frame_id:
            out['frame_id'] = self.frame_id
        if self.contact_state is not None:
            out['contact_state'] = self.contact_state
        if self.contact_confidence is not None:
            out['contact_confidence'] = self.contact_confidence
        if self.wrench_6d is not None:
            out['wrench_6d'] = [float(x) for x in self.wrench_6d]
        if self.runtime_config_ref:
            out['runtime_config_ref'] = self.runtime_config_ref
        if self.raw_refs:
            out['raw_refs'] = dict(self.raw_refs)
        if self.fields_history:
            out['fields_history'] = [[field.to_dict() for field in snapshot] for snapshot in self.fields_history]
        if self.fields_history_timestamps_ns:
            out['fields_history_timestamps_ns'] = [int(x) for x in self.fields_history_timestamps_ns]
        return out

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'TactileObservation':
        if not data:
            return cls()
        wrench = data.get('wrench_6d')
        fields_history_raw = data.get('fields_history') or []
        return cls(
            tactile_role=str(data.get('tactile_role', '')),
            sensor_vendor=str(data.get('sensor_vendor', '')),
            sensor_serial=str(data.get('sensor_serial', '')),
            sensor_model=str(data.get('sensor_model', '')),
            timestamp_ns=int(data.get('timestamp_ns', 0)),
            frame_id=str(data.get('frame_id', '')),
            contact_state=data.get('contact_state'),
            contact_confidence=data.get('contact_confidence'),
            wrench_6d=[float(x) for x in wrench] if wrench is not None else None,
            fields=[TactileField.from_dict(item) for item in data.get('fields', [])],
            runtime_config_ref=str(data.get('runtime_config_ref', '')),
            raw_refs={str(k): str(v) for k, v in (data.get('raw_refs') or {}).items()},
            fields_history=[
                [TactileField.from_dict(item) for item in snapshot]
                for snapshot in fields_history_raw
            ],
            fields_history_timestamps_ns=[int(x) for x in (data.get('fields_history_timestamps_ns') or [])],
        )


@dataclasses.dataclass
class SafetyState:
    emergency_stop_active: bool = False
    velocity_limit_active: bool = False
    workspace_limit_active: bool = False
    action_clipped: bool = False
    safety_status: str = 'ok'
    active_constraints: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'SafetyState':
        if not data:
            return cls()
        return cls(
            emergency_stop_active=bool(data.get('emergency_stop_active', False)),
            velocity_limit_active=bool(data.get('velocity_limit_active', False)),
            workspace_limit_active=bool(data.get('workspace_limit_active', False)),
            action_clipped=bool(data.get('action_clipped', False)),
            safety_status=str(data.get('safety_status', 'ok')),
            active_constraints=[str(x) for x in data.get('active_constraints', [])],
        )


@dataclasses.dataclass
class NetworkState:
    uplink_latency_ms: float = 0.0
    downlink_latency_ms: float = 0.0
    round_trip_time_ms: float = 0.0
    jitter_ms: float = 0.0
    packet_drop_detected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'NetworkState':
        if not data:
            return cls()
        return cls(
            uplink_latency_ms=float(data.get('uplink_latency_ms', 0.0)),
            downlink_latency_ms=float(data.get('downlink_latency_ms', 0.0)),
            round_trip_time_ms=float(data.get('round_trip_time_ms', 0.0)),
            jitter_ms=float(data.get('jitter_ms', 0.0)),
            packet_drop_detected=bool(data.get('packet_drop_detected', False)),
        )


@dataclasses.dataclass
class ObservationPacket:
    context: SessionContext = dataclasses.field(default_factory=SessionContext)
    observation_timestamp_ns: int = 0
    step_index: int = 0
    robot_state: RobotState = dataclasses.field(default_factory=RobotState)
    camera_observations: List[CameraObservation] = dataclasses.field(default_factory=list)
    tactile_observations: List[TactileObservation] = dataclasses.field(default_factory=list)
    safety_state: SafetyState = dataclasses.field(default_factory=SafetyState)
    network_state: NetworkState = dataclasses.field(default_factory=NetworkState)
    robot_state_history: List[RobotState] = dataclasses.field(default_factory=list)
    history_actual: Dict[str, int] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            **self.context.to_dict(),
            'observation_timestamp_ns': self.observation_timestamp_ns,
            'step_index': self.step_index,
            'robot_state': self.robot_state.to_dict(),
            'camera_observations': [cam.to_dict() for cam in self.camera_observations],
            'tactile_observations': [tac.to_dict() for tac in self.tactile_observations],
            'safety_state': self.safety_state.to_dict(),
            'network_state': self.network_state.to_dict(),
        }
        if self.robot_state_history:
            out['robot_state_history'] = [state.to_dict() for state in self.robot_state_history]
        if self.history_actual:
            out['history_actual'] = dict(self.history_actual)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ObservationPacket':
        context = SessionContext.from_dict(data)
        return cls(
            context=context,
            observation_timestamp_ns=int(data.get('observation_timestamp_ns', 0)),
            step_index=int(data.get('step_index', 0)),
            robot_state=RobotState.from_dict(data.get('robot_state')),
            camera_observations=[
                CameraObservation.from_dict(item) for item in data.get('camera_observations', [])
            ],
            tactile_observations=[
                TactileObservation.from_dict(item) for item in data.get('tactile_observations', [])
            ],
            safety_state=SafetyState.from_dict(data.get('safety_state')),
            network_state=NetworkState.from_dict(data.get('network_state')),
            robot_state_history=[
                RobotState.from_dict(item) for item in data.get('robot_state_history', [])
            ],
            history_actual={str(k): int(v) for k, v in (data.get('history_actual') or {}).items()},
        )


@dataclasses.dataclass
class ArmAction:
    arm_id: str = ''
    delta_position_m: Optional[Vector3] = None
    delta_rotation_axis_angle: Optional[Vector3] = None
    target_pose_base: Optional[Pose] = None
    joint_position_rad: Optional[List[float]] = None
    gripper_target_open_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {'arm_id': self.arm_id}
        if self.delta_position_m is not None:
            out['delta_position_m'] = _vec3_to_dict(self.delta_position_m)
        if self.delta_rotation_axis_angle is not None:
            out['delta_rotation_axis_angle'] = _vec3_to_dict(self.delta_rotation_axis_angle)
        if self.target_pose_base is not None:
            out['target_pose_base'] = self.target_pose_base.to_dict()
        if self.joint_position_rad is not None:
            out['joint_position_rad'] = list(self.joint_position_rad)
        if self.gripper_target_open_ratio is not None:
            out['gripper_target_open_ratio'] = self.gripper_target_open_ratio
        return out

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'ArmAction':
        if not data:
            return cls()
        target_pose = data.get('target_pose_base')
        return cls(
            arm_id=str(data.get('arm_id', '')),
            delta_position_m=Vector3.from_dict(data.get('delta_position_m')),
            delta_rotation_axis_angle=Vector3.from_dict(data.get('delta_rotation_axis_angle')),
            target_pose_base=Pose.from_dict(target_pose) if target_pose else None,
            joint_position_rad=[float(x) for x in data['joint_position_rad']] if 'joint_position_rad' in data else None,
            gripper_target_open_ratio=data.get('gripper_target_open_ratio'),
        )


@dataclasses.dataclass
class ActionStep:
    relative_step: int = 0
    arm_actions: List[ArmAction] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'relative_step': self.relative_step,
            'arm_actions': [action.to_dict() for action in self.arm_actions],
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'ActionStep':
        if not data:
            return cls()
        return cls(
            relative_step=int(data.get('relative_step', 0)),
            arm_actions=[ArmAction.from_dict(item) for item in data.get('arm_actions', [])],
        )


@dataclasses.dataclass
class ActionPacket:
    context: SessionContext = dataclasses.field(default_factory=SessionContext)
    observation_timestamp_ns: int = 0
    inference_timestamp_ns: int = 0
    action_apply_timestamp_ns: int = 0
    action_mode: str = ''
    action_chunk: List[ActionStep] = dataclasses.field(default_factory=list)
    expected_horizon_steps: int = 0
    policy_latency_ms: float = 0.0
    confidence: Optional[float] = None
    status: str = 'ok'
    error_message: str = ''

    def to_dict(self) -> Dict[str, Any]:
        out = {
            **self.context.to_dict(),
            'observation_timestamp_ns': self.observation_timestamp_ns,
            'inference_timestamp_ns': self.inference_timestamp_ns,
            'action_apply_timestamp_ns': self.action_apply_timestamp_ns,
            'action_mode': self.action_mode,
            'action_chunk': [step.to_dict() for step in self.action_chunk],
            'expected_horizon_steps': self.expected_horizon_steps,
            'policy_latency_ms': self.policy_latency_ms,
            'status': self.status,
        }
        if self.confidence is not None:
            out['confidence'] = self.confidence
        if self.error_message:
            out['error_message'] = self.error_message
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActionPacket':
        context = SessionContext.from_dict(data)
        return cls(
            context=context,
            observation_timestamp_ns=int(data.get('observation_timestamp_ns', 0)),
            inference_timestamp_ns=int(data.get('inference_timestamp_ns', 0)),
            action_apply_timestamp_ns=int(data.get('action_apply_timestamp_ns', 0)),
            action_mode=str(data.get('action_mode', '')),
            action_chunk=[ActionStep.from_dict(item) for item in data.get('action_chunk', [])],
            expected_horizon_steps=int(data.get('expected_horizon_steps', 0)),
            policy_latency_ms=float(data.get('policy_latency_ms', 0.0)),
            confidence=data.get('confidence'),
            status=str(data.get('status', 'ok')),
            error_message=str(data.get('error_message', '')),
        )


@dataclasses.dataclass
class EpisodeEvent:
    context: SessionContext = dataclasses.field(default_factory=SessionContext)
    event_timestamp_ns: int = 0
    event_type: str = ''
    event_message: str = ''
    metadata: Dict[str, str] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.context.to_dict(),
            'event_timestamp_ns': self.event_timestamp_ns,
            'event_type': self.event_type,
            'event_message': self.event_message,
            'metadata': dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EpisodeEvent':
        context = SessionContext.from_dict(data)
        return cls(
            context=context,
            event_timestamp_ns=int(data.get('event_timestamp_ns', 0)),
            event_type=str(data.get('event_type', '')),
            event_message=str(data.get('event_message', '')),
            metadata={str(k): str(v) for k, v in (data.get('metadata') or {}).items()},
        )
