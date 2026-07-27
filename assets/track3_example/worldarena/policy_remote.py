"""wa-policy-v1 WebSocket protocol for remote policy inference."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import traceback
from typing import Any, Dict, Optional, Tuple

import websockets.asyncio.server
import websockets.frames

from worldarena import msgpack_numpy

from worldarena.bridges.legacy_policy import (
    infer_output_to_action_packet,
    observation_packet_to_new_obs,
)
from worldarena.protocol import (
    POLICY_ENDPOINT_HEALTH,
    POLICY_ENDPOINT_INFER,
    POLICY_ENDPOINT_RESET,
    POLICY_PROTOCOL_VERSION,
    SCHEMA_VERSION,
)
from worldarena.schema import ActionPacket, ObservationPacket
from worldarena.serde import action_from_message, observation_from_message, pack_message, unpack_message
from worldarena.ws_util import connect_websocket, normalize_websocket_uri

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PolicyServerMetadata:
    protocol: str = POLICY_PROTOCOL_VERSION
    schema_version: str = SCHEMA_VERSION
    policy_source: str = ''
    supports_reset: bool = False
    supports_legacy_new_obs: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def normalize_policy_uri(uri: str) -> str:
    return normalize_websocket_uri(uri)


def dispatch_policy_endpoint(
    policy: Any,
    request: Dict[str, Any],
    *,
    legacy_bridge: bool = True,
) -> Any:
    """Handle a wa-policy-v1 request dict and return the response payload."""
    from worldarena.bridges.legacy_policy import (
        infer_output_to_action_packet,
        observation_packet_to_new_obs,
    )
    from worldarena.serde import observation_from_message

    endpoint = request.get('endpoint', POLICY_ENDPOINT_INFER)

    if endpoint == POLICY_ENDPOINT_HEALTH:
        return {'status': 'ok', 'protocol': POLICY_PROTOCOL_VERSION, 'schema_version': SCHEMA_VERSION}
    if endpoint == POLICY_ENDPOINT_RESET:
        reset_info = request.get('reset_info', {})
        if callable(getattr(policy, 'reset', None)):
            policy.reset(reset_info)
        return {'status': 'reset successful', 'protocol': POLICY_PROTOCOL_VERSION}
    if endpoint == POLICY_ENDPOINT_INFER:
        from worldarena.action_debug import log_pipeline_trace

        if 'observation_packet' in request:
            obs_packet = observation_from_message(request['observation_packet'])
            new_obs = observation_packet_to_new_obs(obs_packet)
            output = policy.infer(new_obs)
            meta = output.get('policy_metadata') or {}
            action_packet = infer_output_to_action_packet(
                output,
                context=obs_packet.context,
                observation_timestamp_ns=obs_packet.observation_timestamp_ns,
                control_arm=meta.get('control_arm'),
            )
            payload = action_packet.to_dict()
            if 'actions' in output:
                payload['actions'] = output['actions']
            if meta:
                payload['policy_metadata'] = meta
            if output.get('policy_timing'):
                payload['policy_timing'] = output['policy_timing']
            # Forward optional tactile force / auxiliary data so policies can
            # return extra per-timestep information alongside actions.
            if 'tactile_force' in output:
                payload['tactile_force'] = output['tactile_force']
            if 'auxiliary' in output:
                payload['auxiliary'] = output['auxiliary']
            log_pipeline_trace(
                'A:infer_response',
                raw_actions=output.get('actions'),
                action_packet=action_packet,
            )
            return payload
        if 'new_obs' in request and legacy_bridge:
            return policy.infer(request['new_obs'])
        raise ValueError("infer request must include 'observation_packet' or 'new_obs'")
    raise ValueError(f'Unknown endpoint: {endpoint}')


class CanonicalPolicyServer:
    """Serve a policy over wa-policy-v1 using ObservationPacket / ActionPacket."""

    def __init__(
        self,
        policy: Any,
        *,
        policy_source: str = '',
        host: str = '0.0.0.0',
        port: int = 8000,
        legacy_bridge: bool = True,
    ) -> None:
        self._policy = policy
        self._policy_source = policy_source
        self._host = host
        self._port = port
        self._legacy_bridge = legacy_bridge
        self._supports_reset = callable(getattr(policy, 'reset', None))
        logging.getLogger('websockets.server').setLevel(logging.INFO)

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
        logger.info('Policy connection from %s opened', remote)
        packer = msgpack_numpy.Packer()

        metadata = PolicyServerMetadata(
            policy_source=self._policy_source,
            supports_reset=self._supports_reset,
            supports_legacy_new_obs=self._legacy_bridge,
        )
        await websocket.send(packer.pack(metadata.to_dict()))

        while True:
            try:
                request = unpack_message(await websocket.recv())
                response = dispatch_policy_endpoint(
                    self._policy,
                    request,
                    legacy_bridge=self._legacy_bridge,
                )
                await websocket.send(packer.pack(response))
            except websockets.ConnectionClosed:
                logger.info('Policy connection from %s closed', remote)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason='Internal server error. Traceback included in previous frame.',
                )
                raise


class CanonicalPolicyClient:
    """Policy client for wa-policy-v1 that speaks ObservationPacket / ActionPacket."""

    def __init__(self, uri: str) -> None:
        self._uri = normalize_policy_uri(uri)
        self._packer = msgpack_numpy.Packer()
        self._ws, self._metadata = self._connect()

    @property
    def metadata(self) -> PolicyServerMetadata:
        return self._metadata

    def _connect(self) -> Tuple[Any, PolicyServerMetadata]:
        conn, self._uri = connect_websocket(self._uri)
        metadata_dict = unpack_message(conn.recv())
        metadata = PolicyServerMetadata(**{k: metadata_dict.get(k, v) for k, v in PolicyServerMetadata().__dict__.items()})
        if metadata.protocol != POLICY_PROTOCOL_VERSION:
            raise ValueError(
                f'Unsupported policy protocol: {metadata.protocol} (expected {POLICY_PROTOCOL_VERSION})'
            )
        return conn, metadata

    def health_check(self) -> Dict[str, Any]:
        request = {'endpoint': POLICY_ENDPOINT_HEALTH}
        self._ws.send(self._packer.pack(request))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f'Error in canonical policy server:\n{response}')
        data = unpack_message(response)
        if not isinstance(data, dict) or data.get('status') != 'ok':
            raise RuntimeError(f'Policy health check failed: {data!r}')
        return data

    def infer_policy_result(self, observation_packet: ObservationPacket) -> 'PolicyInferResult':
        from worldarena.bridges.legacy_policy import parse_policy_infer_full

        request = {
            'endpoint': POLICY_ENDPOINT_INFER,
            'observation_packet': observation_packet.to_dict(),
        }
        self._ws.send(self._packer.pack(request))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f'Error in canonical policy server:\n{response}')
        data = unpack_message(response)
        return parse_policy_infer_full(
            data,
            context=observation_packet.context,
            observation_timestamp_ns=observation_packet.observation_timestamp_ns,
        )

    def infer_packet(self, observation_packet: ObservationPacket) -> ActionPacket:
        return self.infer_policy_result(observation_packet).action_packet

    def infer(self, new_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy-compatible infer() for benchmark_runner integration."""
        request = {'endpoint': POLICY_ENDPOINT_INFER, 'new_obs': new_obs}
        self._ws.send(self._packer.pack(request))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f'Error in canonical policy server:\n{response}')
        return unpack_message(response)

    def reset(self, reset_info: Optional[Dict[str, Any]] = None) -> str:
        request = {'endpoint': POLICY_ENDPOINT_RESET, 'reset_info': reset_info or {}}
        self._ws.send(self._packer.pack(request))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f'Error in canonical policy server:\n{response}')
        data = unpack_message(response)
        if isinstance(data, dict):
            return str(data.get('status', data))
        return str(data)

    def close(self) -> None:
        self._ws.close()
