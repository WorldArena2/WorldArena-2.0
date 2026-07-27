"""wa-robot-v1 WebSocket protocol for remote robot embodiment workers."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple

import websockets.asyncio.server
import websockets.frames

from worldarena import msgpack_numpy

from worldarena.adapters.base import RobotAdapter
from worldarena.protocol import (
    ROBOT_ENDPOINT_APPLY_ACTION,
    ROBOT_ENDPOINT_GET_OBSERVATION,
    ROBOT_ENDPOINT_HEALTH,
    ROBOT_ENDPOINT_REPORT_EVENT,
    ROBOT_ENDPOINT_RESET,
    ROBOT_PROTOCOL_VERSION,
    SCHEMA_VERSION,
)
from worldarena.observation_history import ObservationHistoryConfig
from worldarena.schema import ActionPacket, EpisodeEvent, ObservationPacket, SessionContext
from worldarena.serde import action_from_message, event_from_message, observation_from_message, pack_message, unpack_message
from worldarena.ws_util import connect_websocket, normalize_websocket_uri

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RobotServerMetadata:
    protocol: str = ROBOT_PROTOCOL_VERSION
    schema_version: str = SCHEMA_VERSION
    adapter_id: str = ''
    adapter_version: str = ''
    embodiment_id: str = ''
    embodiment_type: str = ''
    supported_tactile_roles: List[str] = dataclasses.field(default_factory=list)
    supported_tactile_profiles: List[str] = dataclasses.field(default_factory=list)
    default_tactile_profile: str = ''
    tactile_sensor_vendor: str = ''
    tactile_sdk_version: str = ''

    def to_dict(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        if not self.supported_tactile_roles:
            out.pop('supported_tactile_roles', None)
        if not self.supported_tactile_profiles:
            out.pop('supported_tactile_profiles', None)
        if not self.default_tactile_profile:
            out.pop('default_tactile_profile', None)
        if not self.tactile_sensor_vendor:
            out.pop('tactile_sensor_vendor', None)
        if not self.tactile_sdk_version:
            out.pop('tactile_sdk_version', None)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RobotServerMetadata':
        known = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: data.get(k, '') for k in known if k in data}
        for list_field in ('supported_tactile_roles', 'supported_tactile_profiles'):
            if list_field in data:
                kwargs[list_field] = [str(x) for x in data[list_field]]
        defaults = cls()
        for field in dataclasses.fields(cls):
            if field.name not in kwargs:
                kwargs[field.name] = getattr(defaults, field.name)
        return cls(**kwargs)


def normalize_robot_uri(uri: str) -> str:
    return normalize_websocket_uri(uri)


@dataclasses.dataclass
class RobotDispatchState:
    last_private_obs: Optional[Dict[str, Any]] = None
    events: List[EpisodeEvent] = dataclasses.field(default_factory=list)


def build_robot_server_metadata(adapter: RobotAdapter) -> RobotServerMetadata:
    metadata = RobotServerMetadata.from_dict(
        adapter.metadata() if hasattr(adapter, 'metadata') else {}
    )
    metadata.protocol = ROBOT_PROTOCOL_VERSION
    metadata.schema_version = SCHEMA_VERSION
    if not metadata.adapter_id:
        metadata.adapter_id = getattr(adapter, 'adapter_id', '')
    if not metadata.adapter_version:
        metadata.adapter_version = getattr(adapter, 'adapter_version', '')
    if not metadata.embodiment_id:
        metadata.embodiment_id = getattr(adapter, 'embodiment_id', '')
    if not metadata.embodiment_type:
        metadata.embodiment_type = getattr(adapter, 'embodiment_type', '')
    return metadata


def dispatch_robot_endpoint(
    adapter: RobotAdapter,
    request: Dict[str, Any],
    *,
    state: Optional[RobotDispatchState] = None,
) -> Any:
    """Handle a wa-robot-v1 request dict and return the response payload."""
    if state is None:
        state = RobotDispatchState()

    endpoint = request.get('endpoint', ROBOT_ENDPOINT_GET_OBSERVATION)

    if endpoint == ROBOT_ENDPOINT_HEALTH:
        meta = build_robot_server_metadata(adapter)
        response: Any = {'status': 'ok', 'protocol': ROBOT_PROTOCOL_VERSION, 'schema_version': SCHEMA_VERSION}
        response.update(meta.to_dict())
        return response
    if endpoint == ROBOT_ENDPOINT_RESET:
        reset_info = request.get('reset_info', {})
        context_data = request.get('context')
        if context_data:
            adapter.set_context(SessionContext.from_dict(context_data))  # type: ignore[attr-defined]
        adapter.reset(reset_info)
        return {'status': 'reset successful', 'protocol': ROBOT_PROTOCOL_VERSION}
    if endpoint == ROBOT_ENDPOINT_GET_OBSERVATION:
        context_data = request.get('context')
        context = SessionContext.from_dict(context_data) if context_data else None
        step_index = int(request.get('step_index', 0))
        history_data = request.get('observation_history')
        if history_data and hasattr(adapter, 'set_observation_history_config'):
            adapter.set_observation_history_config(  # type: ignore[attr-defined]
                ObservationHistoryConfig.from_dict(history_data)
            )
        private_obs = adapter.wait_private_observation()  # type: ignore[attr-defined]
        state.last_private_obs = private_obs
        obs_packet = adapter.private_observation_to_canonical(
            private_obs,
            context=context,
            step_index=step_index,
        )
        return obs_packet.to_dict()
    if endpoint == ROBOT_ENDPOINT_APPLY_ACTION:
        from worldarena.bridges.legacy_policy import repair_action_packet_if_misparsed

        action_packet = action_from_message(request['action_packet'])
        action_packet = repair_action_packet_if_misparsed(action_packet)
        if state.last_private_obs is None:
            raise RuntimeError('No observation received before apply_action')
        from worldarena.action_debug import log_pipeline_trace

        log_pipeline_trace('C:receive_action_packet', action_packet=action_packet)
        debug_raw = request.get('debug_raw_actions')
        adapter.apply_action(action_packet, state.last_private_obs, debug_raw_actions=debug_raw)
        return {'status': 'action applied', 'protocol': ROBOT_PROTOCOL_VERSION}
    if endpoint == ROBOT_ENDPOINT_REPORT_EVENT:
        event = event_from_message(request['event'])
        state.events.append(event)
        if hasattr(adapter, 'on_episode_event'):
            adapter.on_episode_event(event)  # type: ignore[attr-defined]
        logger.info('Episode event received: %s (%s)', event.event_type, event.event_message)
        return {'status': 'event recorded', 'protocol': ROBOT_PROTOCOL_VERSION}
    raise ValueError(f'Unknown endpoint: {endpoint}')


class WebsocketRobotServer:
    """Expose a RobotAdapter over wa-robot-v1 WebSocket protocol."""

    def __init__(
        self,
        adapter: RobotAdapter,
        *,
        host: str = '0.0.0.0',
        port: int = 9000,
    ) -> None:
        self._adapter = adapter
        self._host = host
        self._port = port
        self._last_private_obs: Optional[Dict[str, Any]] = None
        self._events: List[EpisodeEvent] = []
        self._dispatch_state = RobotDispatchState()
        logging.getLogger('websockets.server').setLevel(logging.INFO)

    @property
    def events(self) -> List[EpisodeEvent]:
        return list(self._events)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection) -> None:
        remote = websocket.remote_address
        logger.info('Robot connection from %s opened', remote)
        packer = msgpack_numpy.Packer()

        metadata = build_robot_server_metadata(self._adapter)
        await websocket.send(packer.pack(metadata.to_dict()))

        while True:
            try:
                request = unpack_message(await websocket.recv())
                response = dispatch_robot_endpoint(
                    self._adapter,
                    request,
                    state=self._dispatch_state,
                )
                self._last_private_obs = self._dispatch_state.last_private_obs
                self._events = self._dispatch_state.events
                await websocket.send(packer.pack(response))
            except websockets.ConnectionClosed:
                logger.info('Robot connection from %s closed', remote)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason='Internal server error. Traceback included in previous frame.',
                )
                raise


class RemoteRobotClient:
    """Client for wa-robot-v1 that mimics manifold_msg Server interface."""

    def __init__(self, uri: str) -> None:
        self._uri = normalize_robot_uri(uri)
        self._packer = msgpack_numpy.Packer()
        self._ws, self._metadata = self._connect()
        self._step_index = 0
        self._context = SessionContext()
        self._last_private_obs: Dict[str, Any] = {}
        self._observation_history = ObservationHistoryConfig()

    @property
    def metadata(self) -> RobotServerMetadata:
        return self._metadata

    @property
    def embodiment_id(self) -> str:
        return self._metadata.embodiment_id

    @property
    def adapter_version(self) -> str:
        return self._metadata.adapter_version

    @property
    def adapter_id(self) -> str:
        return self._metadata.adapter_id

    @property
    def embodiment_type(self) -> str:
        return self._metadata.embodiment_type

    def set_context(self, context: SessionContext) -> None:
        self._context = context

    def set_observation_history_config(self, config: ObservationHistoryConfig) -> None:
        self._observation_history = config

    def _connect(self) -> Tuple[Any, RobotServerMetadata]:
        conn, self._uri = connect_websocket(self._uri)
        metadata_dict = unpack_message(conn.recv())
        metadata = RobotServerMetadata.from_dict(metadata_dict if isinstance(metadata_dict, dict) else {})
        if metadata.protocol != ROBOT_PROTOCOL_VERSION:
            raise ValueError(
                f'Unsupported robot protocol: {metadata.protocol} (expected {ROBOT_PROTOCOL_VERSION})'
            )
        return conn, metadata

    def health_check(self) -> Dict[str, Any]:
        data = self._request({'endpoint': ROBOT_ENDPOINT_HEALTH})
        if not isinstance(data, dict) or data.get('status') != 'ok':
            raise RuntimeError(f'Robot health check failed: {data!r}')
        return data

    def _request(self, payload: Dict[str, Any]) -> Any:
        self._ws.send(self._packer.pack(payload))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f'Error in remote robot server:\n{response}')
        return unpack_message(response)

    def wait_observation(self) -> Dict[str, Any]:
        """Return private-style observation reconstructed from canonical packet."""
        obs_packet = self.wait_observation_packet()
        return observation_packet_to_private(obs_packet)

    def wait_observation_packet(self) -> ObservationPacket:
        request = {
            'endpoint': ROBOT_ENDPOINT_GET_OBSERVATION,
            'context': self._context.to_dict(),
            'step_index': self._step_index,
        }
        if self._observation_history.camera_roles or self._observation_history.robot_state_len > 1:
            request['observation_history'] = self._observation_history.to_dict()
        data = self._request(request)
        self._step_index += 1
        packet = observation_from_message(data)
        self._last_private_obs = observation_packet_to_private(packet)
        return packet

    def observation_packet_to_private(self, packet: ObservationPacket) -> Dict[str, Any]:
        return observation_packet_to_private(packet)

    def send_action_packet(
        self,
        action_packet: ActionPacket,
        *,
        debug_raw_actions: Any = None,
    ) -> None:
        request = {
            'endpoint': ROBOT_ENDPOINT_APPLY_ACTION,
            'action_packet': action_packet.to_dict(),
        }
        if debug_raw_actions is not None:
            request['debug_raw_actions'] = debug_raw_actions
        self._request(request)

    def report_event(self, event: EpisodeEvent) -> None:
        request = {
            'endpoint': ROBOT_ENDPOINT_REPORT_EVENT,
            'event': event.to_dict(),
        }
        self._request(request)

    def reset(self, reset_info: Optional[Dict[str, Any]] = None) -> None:
        request = {
            'endpoint': ROBOT_ENDPOINT_RESET,
            'reset_info': reset_info or {},
            'context': self._context.to_dict(),
        }
        self._request(request)
        self._step_index = 0

    def close(self) -> None:
        self._ws.close()


def observation_packet_to_private(packet: ObservationPacket) -> Dict[str, Any]:
    import numpy as np

    private: Dict[str, Any] = {
        'timestamp': packet.observation_timestamp_ns / 1_000_000_000,
        'prompt': packet.context.task_instruction,
    }
    role_to_field = {
        'global': 'img_front',
        'left_wrist': 'img_left',
        'right_wrist': 'img_right',
    }
    for cam in packet.camera_observations:
        field = role_to_field.get(cam.camera_role)
        if not field:
            continue
        frames = []
        for blob in cam.frame_history_bytes or []:
            if blob is None or cam.height <= 0 or cam.width <= 0:
                continue
            frames.append(np.frombuffer(blob, dtype=np.uint8).reshape(cam.height, cam.width, 3))
        if cam.frame_bytes is not None and cam.height > 0 and cam.width > 0:
            frames.append(np.frombuffer(cam.frame_bytes, dtype=np.uint8).reshape(cam.height, cam.width, 3))
        if frames:
            private[field] = frames

    for arm in packet.robot_state.arms:
        prefix = 'left' if arm.arm_id == 'left' else 'right'
        if arm.joint_state.position_rad:
            private[f'{prefix}_arm_joint_state'] = [np.asarray(arm.joint_state.position_rad, dtype=np.float32)]
        pose = arm.ee_pose_base
        end_pose = np.array(
            [
                pose.position_m.x,
                pose.position_m.y,
                pose.position_m.z,
                pose.orientation_xyzw.x,
                pose.orientation_xyzw.y,
                pose.orientation_xyzw.z,
                pose.orientation_xyzw.w,
                arm.gripper.open_ratio,
            ],
            dtype=np.float32,
        )
        private[f'{prefix}_end_pose'] = [end_pose]

    if packet.tactile_observations:
        from worldarena.tactile import tactile_observations_to_legacy

        private['tactile'] = tactile_observations_to_legacy(packet.tactile_observations)

    return private
