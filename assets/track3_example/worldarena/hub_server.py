"""Flask HTTP server for wa-hub-v1 (dual-port policy/robot)."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request
from werkzeug.serving import make_server

from worldarena.hub_core import HubCore, HubError, HubTask
from worldarena.hub_json import hub_json_encode
from worldarena.hub_protocol import (
    HUB_DEFAULT_POLL_TIMEOUT_S,
    HUB_ROUTE_HEALTH,
    HUB_ROUTE_ORCHESTRATOR_POLICY,
    HUB_ROUTE_ORCHESTRATOR_ROBOT,
    HUB_ROUTE_SESSION_BIND,
    HUB_ROUTE_WORKER_HEARTBEAT,
    HUB_ROUTE_WORKER_POLL,
    HUB_ROUTE_WORKER_REGISTER,
    HUB_ROUTE_WORKER_RESULT,
    WORKER_ROLE_POLICY,
    WORKER_ROLE_ROBOT,
)

logger = logging.getLogger(__name__)


def _is_localhost_request() -> bool:
    addr = request.remote_addr or ''
    return addr in ('127.0.0.1', '::1', 'localhost') or addr.startswith('127.')


def _require_localhost() -> Optional[Any]:
    if not _is_localhost_request():
        return jsonify({'error': 'orchestrator endpoints are localhost-only'}), 403
    return None


def _task_to_dict(task: HubTask) -> Dict[str, Any]:
    return hub_json_encode(
        {
            'request_id': task.request_id,
            'session_id': task.session_id,
            'role': task.role,
            'endpoint': task.endpoint,
            'deadline_ms': task.deadline_ms,
            'payload': task.payload,
        }
    )


def _json_hub_error(exc: HubError) -> Any:
    return jsonify({'error': str(exc), 'code': exc.code}), exc.http_status


def _build_app(
    core: HubCore,
    *,
    role: str,
    orchestrator_prefix: str,
    url_prefix: str = '',
    include_session_bind: bool = False,
) -> Flask:
    app = Flask(f'hub_{role}_{id(core)}')
    p = url_prefix.rstrip('/')

    def route(path: str, *, gateway: bool = True) -> str:
        """Apply gateway prefix to worker-facing routes only (orchestrator stays unprefixed for localhost)."""
        if gateway and p:
            return f'{p}{path}'
        return path

    @app.get(route(HUB_ROUTE_HEALTH))
    def health() -> Any:
        return jsonify(core.health())

    @app.post(route(HUB_ROUTE_WORKER_REGISTER))
    def register_worker() -> Any:
        body = request.get_json(silent=True) or {}
        try:
            return jsonify(
                core.register_worker(
                    role=str(body.get('role', role)),
                    worker_key=str(body.get('worker_key', '')),
                    metadata=body.get('metadata') if isinstance(body.get('metadata'), dict) else {},
                )
            )
        except HubError as exc:
            return _json_hub_error(exc)

    @app.post(route(HUB_ROUTE_WORKER_HEARTBEAT))
    def heartbeat() -> Any:
        body = request.get_json(silent=True) or {}
        try:
            core.heartbeat(str(body.get('worker_id', '')))
            return jsonify({'status': 'ok'})
        except HubError as exc:
            return _json_hub_error(exc)

    @app.get(route(HUB_ROUTE_WORKER_POLL))
    def poll_worker() -> Any:
        worker_id = str(request.args.get('worker_id', ''))
        timeout_s = float(request.args.get('timeout_s', HUB_DEFAULT_POLL_TIMEOUT_S))
        try:
            task = core.poll_task(worker_id, timeout_s=timeout_s)
            if task is None:
                return ('', 204)
            return jsonify({'task': _task_to_dict(task)})
        except HubError as exc:
            return _json_hub_error(exc)

    @app.post(route(HUB_ROUTE_WORKER_RESULT))
    def worker_result() -> Any:
        body = request.get_json(silent=True) or {}
        try:
            core.submit_result(
                request_id=str(body.get('request_id', '')),
                worker_id=str(body.get('worker_id', '')),
                status=str(body.get('status', 'ok')),
                result=body.get('result'),
                error=body.get('error') if isinstance(body.get('error'), dict) else None,
            )
            return jsonify({'status': 'ok'})
        except HubError as exc:
            return _json_hub_error(exc)

    @app.post(f'{route(orchestrator_prefix, gateway=False)}/<endpoint>')
    def orchestrator_rpc(endpoint: str) -> Any:
        denied = _require_localhost()
        if denied is not None:
            return denied
        body = request.get_json(silent=True) or {}
        try:
            result = core.orchestrator_rpc(role=role, endpoint=endpoint, body=body)
            return jsonify(hub_json_encode(result))
        except HubError as exc:
            return _json_hub_error(exc)

    if include_session_bind:

        @app.post(route(HUB_ROUTE_SESSION_BIND, gateway=False))
        def bind_session() -> Any:
            denied = _require_localhost()
            if denied is not None:
                return denied
            body = request.get_json(silent=True) or {}
            try:
                return jsonify(
                    core.bind_session(
                        session_id=str(body.get('session_id', '')),
                        policy_worker_key=str(body.get('policy_worker_key', '')),
                        robot_worker_key=str(body.get('robot_worker_key', '')),
                    )
                )
            except HubError as exc:
                return _json_hub_error(exc)

    return app


class HubServer:
    """Run policy and robot Flask apps on separate ports sharing one HubCore."""

    def __init__(
        self,
        core: HubCore,
        *,
        policy_port: int = 8000,
        robot_port: int = 9000,
        host: str = '0.0.0.0',
        policy_url_prefix: str = '',
        robot_url_prefix: str = '',
    ) -> None:
        self._host = host
        self._policy_app = _build_app(
            core,
            role=WORKER_ROLE_POLICY,
            orchestrator_prefix=HUB_ROUTE_ORCHESTRATOR_POLICY,
            url_prefix=policy_url_prefix,
            include_session_bind=True,
        )
        self._robot_app = _build_app(
            core,
            role=WORKER_ROLE_ROBOT,
            orchestrator_prefix=HUB_ROUTE_ORCHESTRATOR_ROBOT,
            url_prefix=robot_url_prefix,
        )
        self._policy_server = make_server(host, policy_port, self._policy_app, threaded=True)
        self._robot_server = make_server(host, robot_port, self._robot_app, threaded=True)
        self._threads: list[threading.Thread] = []

    def serve_forever(self) -> None:
        for name, server in ('policy', self._policy_server), ('robot', self._robot_server):
            thread = threading.Thread(target=server.serve_forever, name=f'hub-{name}', daemon=True)
            thread.start()
            self._threads.append(thread)
            logger.info('Hub %s listening on %s:%s', name, self._host, server.server_port)
        for thread in self._threads:
            thread.join()
