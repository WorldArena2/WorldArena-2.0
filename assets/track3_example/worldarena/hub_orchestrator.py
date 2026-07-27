"""Orchestrator HTTP clients for wa-hub-v1 (benchmark_runner on machine B)."""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import requests

from worldarena.bridges.legacy_policy import repair_action_packet_if_misparsed
from worldarena.hub_protocol import (
    HUB_DEFAULT_RPC_TIMEOUT_S,
    HUB_ROUTE_ORCHESTRATOR_POLICY,
    HUB_ROUTE_ORCHESTRATOR_ROBOT,
    HUB_ROUTE_SESSION_BIND,
)
from worldarena.hub_json import hub_json_decode, hub_json_encode
from worldarena.hub_worker import normalize_hub_base_url
from worldarena.observation_history import ObservationHistoryConfig
from worldarena.policy_remote import PolicyServerMetadata
from worldarena.protocol import (
    POLICY_ENDPOINT_HEALTH,
    POLICY_ENDPOINT_INFER,
    POLICY_ENDPOINT_RESET,
    ROBOT_ENDPOINT_APPLY_ACTION,
    ROBOT_ENDPOINT_GET_OBSERVATION,
    ROBOT_ENDPOINT_HEALTH,
    ROBOT_ENDPOINT_REPORT_EVENT,
    ROBOT_ENDPOINT_RESET,
)
from worldarena.robot_remote import RobotServerMetadata, observation_packet_to_private
from worldarena.schema import ActionPacket, EpisodeEvent, ObservationPacket, SessionContext
from worldarena.serde import observation_from_message


class HubOrchestratorClient:
    def __init__(self, base_url: str, *, token: str = '', timeout_s: float = HUB_DEFAULT_RPC_TIMEOUT_S) -> None:
        self._base_url = normalize_hub_base_url(base_url)
        self._token = token.strip()
        self._timeout_s = timeout_s
        self._session = requests.Session()

    def _headers(self) -> Dict[str, str]:
        headers = {'Content-Type': 'application/json', 'X-Hub-Protocol': 'wa-hub-v1'}
        if self._token:
            headers['Authorization'] = f'Bearer {self._token}'
        return headers

    def _url(self, path: str) -> str:
        return urljoin(self._base_url + '/', path.lstrip('/'))

    def bind_session(self, *, session_id: str, policy_worker_key: str, robot_worker_key: str) -> Dict[str, Any]:
        response = self._session.post(
            self._url(HUB_ROUTE_SESSION_BIND),
            json={
                'session_id': session_id,
                'policy_worker_key': policy_worker_key,
                'robot_worker_key': robot_worker_key,
            },
            headers=self._headers(),
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def _rpc(self, orchestrator_prefix: str, endpoint: str, body: Dict[str, Any]) -> Any:
        response = self._session.post(
            self._url(f'{orchestrator_prefix}/{endpoint}'),
            json=hub_json_encode(body),
            headers=self._headers(),
            timeout=self._timeout_s,
        )
        response.raise_for_status()
        return hub_json_decode(response.json())


class HubPolicyClient:
    """Drop-in orchestrator client matching CanonicalPolicyClient surface."""

    def __init__(
        self,
        base_url: str,
        *,
        policy_worker_key: str = '',
        token: str = '',
        timeout_s: float = HUB_DEFAULT_RPC_TIMEOUT_S,
    ) -> None:
        self._client = HubOrchestratorClient(base_url, token=token, timeout_s=timeout_s)
        self._policy_worker_key = policy_worker_key
        self._session_id = ''
        self.metadata = PolicyServerMetadata()

    def bind_session(self, session_id: str, *, policy_worker_key: str, robot_worker_key: str) -> None:
        self._session_id = session_id
        self._policy_worker_key = policy_worker_key
        self._client.bind_session(
            session_id=session_id,
            policy_worker_key=policy_worker_key,
            robot_worker_key=robot_worker_key,
        )

    def _body(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            'request_id': str(uuid.uuid4()),
            'session_id': self._session_id,
        }
        if self._policy_worker_key:
            body['worker_key'] = self._policy_worker_key
        if extra:
            body.update(extra)
        return body

    def health_check(self) -> Dict[str, Any]:
        data = self._client._rpc(HUB_ROUTE_ORCHESTRATOR_POLICY, POLICY_ENDPOINT_HEALTH, self._body())
        if not isinstance(data, dict) or data.get('status') != 'ok':
            raise RuntimeError(f'Policy health check failed: {data!r}')
        return data

    def infer_policy_result(self, observation_packet: ObservationPacket) -> 'PolicyInferResult':
        from worldarena.bridges.legacy_policy import parse_policy_infer_full

        data = self._client._rpc(
            HUB_ROUTE_ORCHESTRATOR_POLICY,
            POLICY_ENDPOINT_INFER,
            self._body({'observation_packet': observation_packet.to_dict()}),
        )
        return parse_policy_infer_full(
            data,
            context=observation_packet.context,
            observation_timestamp_ns=observation_packet.observation_timestamp_ns,
        )

    def infer_packet(self, observation_packet: ObservationPacket) -> ActionPacket:
        return self.infer_policy_result(observation_packet).action_packet

    def infer(self, new_obs: Dict[str, Any]) -> Dict[str, Any]:
        return self._client._rpc(
            HUB_ROUTE_ORCHESTRATOR_POLICY,
            POLICY_ENDPOINT_INFER,
            self._body({'new_obs': new_obs}),
        )

    def reset(self, reset_info: Optional[Dict[str, Any]] = None) -> str:
        data = self._client._rpc(
            HUB_ROUTE_ORCHESTRATOR_POLICY,
            POLICY_ENDPOINT_RESET,
            self._body({'reset_info': reset_info or {}}),
        )
        if isinstance(data, dict):
            return str(data.get('status', data))
        return str(data)

    def close(self) -> None:
        self._session_id = ''


class HubRobotClient:
    """Drop-in orchestrator client matching RemoteRobotClient surface."""

    def __init__(
        self,
        base_url: str,
        *,
        robot_worker_key: str = '',
        token: str = '',
        timeout_s: float = HUB_DEFAULT_RPC_TIMEOUT_S,
    ) -> None:
        self._client = HubOrchestratorClient(base_url, token=token, timeout_s=timeout_s)
        self._robot_worker_key = robot_worker_key
        self._session_id = ''
        self._step_index = 0
        self._context = SessionContext()
        self._last_private_obs: Dict[str, Any] = {}
        self._observation_history = ObservationHistoryConfig()
        self._metadata = RobotServerMetadata()

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

    def bind_session(self, session_id: str, *, policy_worker_key: str, robot_worker_key: str) -> None:
        self._session_id = session_id
        self._robot_worker_key = robot_worker_key

    def set_context(self, context: SessionContext) -> None:
        self._context = context

    def set_observation_history_config(self, config: ObservationHistoryConfig) -> None:
        self._observation_history = config

    def _body(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            'request_id': str(uuid.uuid4()),
            'session_id': self._session_id,
        }
        if self._robot_worker_key:
            body['worker_key'] = self._robot_worker_key
        if extra:
            body.update(extra)
        return body

    def health_check(self) -> Dict[str, Any]:
        data = self._client._rpc(HUB_ROUTE_ORCHESTRATOR_ROBOT, ROBOT_ENDPOINT_HEALTH, self._body())
        if not isinstance(data, dict) or data.get('status') != 'ok':
            raise RuntimeError(f'Robot health check failed: {data!r}')
        self._metadata = RobotServerMetadata.from_dict(data)
        return data

    def wait_observation(self) -> Dict[str, Any]:
        return observation_packet_to_private(self.wait_observation_packet())

    def wait_observation_packet(self) -> ObservationPacket:
        request = {
            'context': self._context.to_dict(),
            'step_index': self._step_index,
        }
        if self._observation_history.camera_roles or self._observation_history.robot_state_len > 1:
            request['observation_history'] = self._observation_history.to_dict()
        data = self._client._rpc(
            HUB_ROUTE_ORCHESTRATOR_ROBOT,
            ROBOT_ENDPOINT_GET_OBSERVATION,
            self._body(request),
        )
        self._step_index += 1
        packet = observation_from_message(data)
        self._last_private_obs = observation_packet_to_private(packet)
        return packet

    def send_action_packet(
        self,
        action_packet: ActionPacket,
        *,
        debug_raw_actions: Any = None,
    ) -> None:
        step_index = max(0, self._step_index - 1)
        idempotency_key = f'{self._session_id}:{step_index}:apply_action'
        payload: Dict[str, Any] = {
            'action_packet': action_packet.to_dict(),
            'idempotency_key': idempotency_key,
        }
        if debug_raw_actions is not None:
            payload['debug_raw_actions'] = debug_raw_actions
        self._client._rpc(
            HUB_ROUTE_ORCHESTRATOR_ROBOT,
            ROBOT_ENDPOINT_APPLY_ACTION,
            self._body(payload),
        )

    def report_event(self, event: EpisodeEvent) -> None:
        self._client._rpc(
            HUB_ROUTE_ORCHESTRATOR_ROBOT,
            ROBOT_ENDPOINT_REPORT_EVENT,
            self._body({'event': event.to_dict()}),
        )

    def reset(self, reset_info: Optional[Dict[str, Any]] = None) -> None:
        self._client._rpc(
            HUB_ROUTE_ORCHESTRATOR_ROBOT,
            ROBOT_ENDPOINT_RESET,
            self._body({'reset_info': reset_info or {}, 'context': self._context.to_dict()}),
        )
        self._step_index = 0

    def close(self) -> None:
        self._session_id = ''
