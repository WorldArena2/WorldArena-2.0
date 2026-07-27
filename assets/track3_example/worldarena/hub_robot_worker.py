"""Hub robot worker loop (wa-hub-v1 on machine C)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from worldarena.adapters.base import RobotAdapter
from worldarena.hub_protocol import WORKER_ROLE_ROBOT
from worldarena.hub_worker import HubWorkerClient
from worldarena.robot_remote import RobotDispatchState, build_robot_server_metadata, dispatch_robot_endpoint


class HubRobotWorkerState:
    """Tracks robot dispatch state and apply_action idempotency."""

    def __init__(self) -> None:
        self.dispatch_state = RobotDispatchState()
        self._applied_keys: set[str] = set()

    def handle(self, adapter: RobotAdapter, request: Dict[str, Any]) -> Any:
        from worldarena.protocol import ROBOT_ENDPOINT_APPLY_ACTION

        if request.get('endpoint') == ROBOT_ENDPOINT_APPLY_ACTION:
            key = str(request.get('idempotency_key') or request.get('request_id') or '')
            if key and key in self._applied_keys:
                return {'status': 'applied', 'idempotent': True, 'protocol': 'wa-robot-v1'}
            result = dispatch_robot_endpoint(adapter, request, state=self.dispatch_state)
            if key:
                self._applied_keys.add(key)
            return result
        return dispatch_robot_endpoint(adapter, request, state=self.dispatch_state)


def run_robot_hub_worker(
    adapter: RobotAdapter,
    *,
    hub_url: str,
    worker_key: str,
    token: str = '',
) -> None:
    metadata = build_robot_server_metadata(adapter).to_dict()
    client = HubWorkerClient(
        hub_url,
        role=WORKER_ROLE_ROBOT,
        worker_key=worker_key,
        metadata=metadata,
        token=token,
    )
    state = HubRobotWorkerState()

    def handler(request: Dict[str, Any]) -> Any:
        return state.handle(adapter, request)

    client.run_forever(handler)
