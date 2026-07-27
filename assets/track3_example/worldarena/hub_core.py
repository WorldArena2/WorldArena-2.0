"""In-memory HubCore for wa-hub-v1 long-poll worker orchestration."""

from __future__ import annotations

import collections
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from worldarena.hub_json import hub_json_decode
from worldarena.hub_protocol import (
    HUB_DEFAULT_POLL_TIMEOUT_S,
    WORKER_ROLE_POLICY,
    WORKER_ROLE_ROBOT,
)
from worldarena.protocol import (
    POLICY_ENDPOINT_HEALTH,
    POLICY_ENDPOINT_INFER,
    POLICY_ENDPOINT_RESET,
    ROBOT_ENDPOINT_APPLY_ACTION,
    ROBOT_ENDPOINT_GET_OBSERVATION,
    ROBOT_ENDPOINT_HEALTH,
    ROBOT_ENDPOINT_REPORT_EVENT,
    ROBOT_ENDPOINT_RESET,
    ROBOT_PROTOCOL_VERSION,
    SCHEMA_VERSION,
)


class HubError(Exception):
    """Base Hub error with optional HTTP status code."""

    def __init__(self, message: str, *, code: str = 'HUB_ERROR', http_status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status


@dataclass
class WorkerRecord:
    worker_id: str
    role: str
    worker_key: str
    metadata: Dict[str, Any]
    last_heartbeat: float = field(default_factory=time.time)


@dataclass
class HubTask:
    request_id: str
    session_id: str
    role: str
    endpoint: str
    deadline_ms: int
    payload: Dict[str, Any]


@dataclass
class PendingRpc:
    event: threading.Event = field(default_factory=threading.Event)
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


@dataclass
class SessionBinding:
    session_id: str
    policy_worker_key: str
    robot_worker_key: str
    policy_worker_id: str = ''
    robot_worker_id: str = ''


class HubCore:
    """Thread-safe registry, task queues, and orchestrator RPC coordination."""

    def __init__(self, *, worker_ttl_s: float = 90.0) -> None:
        self._lock = threading.RLock()
        self._worker_ttl_s = worker_ttl_s
        self._workers: Dict[str, WorkerRecord] = {}
        self._worker_keys: Dict[str, str] = {}  # role:worker_key -> worker_id
        self._queues: Dict[str, Deque[HubTask]] = collections.defaultdict(collections.deque)
        self._queue_events: Dict[str, threading.Event] = collections.defaultdict(threading.Event)
        self._pending: Dict[str, PendingRpc] = {}
        self._sessions: Dict[str, SessionBinding] = {}

    def health(self) -> Dict[str, Any]:
        with self._lock:
            self._prune_stale_workers()
            policy_count = sum(1 for w in self._workers.values() if w.role == WORKER_ROLE_POLICY)
            robot_count = sum(1 for w in self._workers.values() if w.role == WORKER_ROLE_ROBOT)
        from worldarena.hub_protocol import HUB_PROTOCOL_VERSION, HUB_TRANSPORT_HTTP
        from worldarena.protocol import SCHEMA_VERSION

        return {
            'status': 'ok',
            'protocol': HUB_PROTOCOL_VERSION,
            'schema_version': SCHEMA_VERSION,
            'transport': HUB_TRANSPORT_HTTP,
            'registered_workers': {'policy': policy_count, 'robot': robot_count},
        }

    def register_worker(
        self,
        *,
        role: str,
        worker_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if role not in (WORKER_ROLE_POLICY, WORKER_ROLE_ROBOT):
            raise HubError(f'Invalid worker role: {role}', code='INVALID_ROLE')
        if not worker_key:
            raise HubError('worker_key is required', code='INVALID_WORKER_KEY')

        worker_id = f'{role}-{worker_key}-{uuid.uuid4().hex[:8]}'
        record = WorkerRecord(
            worker_id=worker_id,
            role=role,
            worker_key=worker_key,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._prune_stale_workers()
            map_key = f'{role}:{worker_key}'
            old_id = self._worker_keys.get(map_key)
            if old_id and old_id in self._workers:
                del self._workers[old_id]
            self._workers[worker_id] = record
            self._worker_keys[map_key] = worker_id

        return {
            'worker_id': worker_id,
            'role': role,
            'worker_key': worker_key,
            'poll_path': f'/workers/poll?worker_id={worker_id}',
            'heartbeat_interval_s': 15,
        }

    def heartbeat(self, worker_id: str) -> None:
        with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                raise HubError('Unknown worker_id', code='UNKNOWN_WORKER', http_status=404)
            record.last_heartbeat = time.time()

    def poll_task(self, worker_id: str, *, timeout_s: float = HUB_DEFAULT_POLL_TIMEOUT_S) -> Optional[HubTask]:
        deadline = time.time() + max(0.1, min(timeout_s, 55.0))
        while True:
            with self._lock:
                record = self._workers.get(worker_id)
                if record is None:
                    raise HubError('Unknown worker_id', code='UNKNOWN_WORKER', http_status=404)
                record.last_heartbeat = time.time()
                queue = self._queues[worker_id]
                if queue:
                    return queue.popleft()
                event = self._queue_events[worker_id]

            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            event.wait(timeout=min(remaining, 1.0))
            with self._lock:
                event.clear()

    def submit_result(
        self,
        *,
        request_id: str,
        worker_id: str,
        status: str,
        result: Any = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            if worker_id not in self._workers:
                raise HubError('Unknown worker_id', code='UNKNOWN_WORKER', http_status=404)
            pending = self._pending.get(request_id)
            if pending is None:
                raise HubError('Unknown or expired request_id', code='UNKNOWN_REQUEST', http_status=404)
            pending.result = {
                'request_id': request_id,
                'worker_id': worker_id,
                'status': status,
                'result': result,
                'error': error,
            }
            if status != 'ok':
                pending.error = error or {'code': 'WORKER_ERROR', 'message': 'worker returned error status'}
            pending.event.set()

    def bind_session(
        self,
        *,
        session_id: str,
        policy_worker_key: str,
        robot_worker_key: str,
    ) -> Dict[str, Any]:
        with self._lock:
            self._prune_stale_workers()
            policy_id = self._worker_keys.get(f'{WORKER_ROLE_POLICY}:{policy_worker_key}')
            robot_id = self._worker_keys.get(f'{WORKER_ROLE_ROBOT}:{robot_worker_key}')
            if not policy_id:
                raise HubError(
                    f'No policy worker registered for key {policy_worker_key!r}',
                    code='POLICY_WORKER_NOT_FOUND',
                    http_status=404,
                )
            if not robot_id:
                raise HubError(
                    f'No robot worker registered for key {robot_worker_key!r}',
                    code='ROBOT_WORKER_NOT_FOUND',
                    http_status=404,
                )
            binding = SessionBinding(
                session_id=session_id,
                policy_worker_key=policy_worker_key,
                robot_worker_key=robot_worker_key,
                policy_worker_id=policy_id,
                robot_worker_id=robot_id,
            )
            self._sessions[session_id] = binding
        return {
            'session_id': session_id,
            'policy_worker_id': policy_id,
            'robot_worker_id': robot_id,
            'status': 'bound',
        }

    def orchestrator_rpc(
        self,
        *,
        role: str,
        endpoint: str,
        body: Dict[str, Any],
        timeout_s: float = 120.0,
    ) -> Any:
        session_id = str(body.get('session_id', '') or body.get('context', {}).get('session_id', ''))
        request_id = str(body.get('request_id') or uuid.uuid4())
        deadline_ms = int(body.get('deadline_ms', timeout_s * 1000))

        if endpoint in (POLICY_ENDPOINT_HEALTH, ROBOT_ENDPOINT_HEALTH):
            with self._lock:
                worker_id = self._resolve_worker_id(role, session_id, body)
                record = self._workers[worker_id]
            response: Dict[str, Any] = {
                'status': 'ok',
                'protocol': ROBOT_PROTOCOL_VERSION if role == WORKER_ROLE_ROBOT else 'wa-policy-v1',
                'schema_version': SCHEMA_VERSION,
            }
            response.update(record.metadata)
            return response

        with self._lock:
            worker_id = self._resolve_worker_id(role, session_id, body)

        payload = hub_json_decode(dict(body))
        payload.pop('request_id', None)
        payload.pop('deadline_ms', None)

        task = HubTask(
            request_id=request_id,
            session_id=session_id,
            role=role,
            endpoint=endpoint,
            deadline_ms=deadline_ms,
            payload=payload,
        )
        pending = PendingRpc()
        with self._lock:
            self._pending[request_id] = pending
            self._enqueue(worker_id, task)

        if not pending.event.wait(timeout=timeout_s):
            with self._lock:
                self._pending.pop(request_id, None)
            raise HubError('Orchestrator RPC timed out', code='RPC_TIMEOUT', http_status=504)

        with self._lock:
            stored = self._pending.pop(request_id, None)
        if stored is None:
            raise HubError('RPC state lost', code='RPC_STATE_LOST', http_status=500)
        if stored.error:
            message = stored.error.get('message', 'worker error')
            raise HubError(message, code=stored.error.get('code', 'WORKER_ERROR'), http_status=500)
        if stored.result is None or stored.result.get('status') != 'ok':
            err = (stored.result or {}).get('error') or {'message': 'unknown worker failure'}
            raise HubError(str(err.get('message', err)), code='WORKER_ERROR', http_status=500)
        return hub_json_decode(stored.result.get('result'))

    def _resolve_worker_id(self, role: str, session_id: str, body: Dict[str, Any]) -> str:
        worker_key = body.get('worker_key')
        if session_id and session_id in self._sessions:
            binding = self._sessions[session_id]
            if role == WORKER_ROLE_POLICY:
                return binding.policy_worker_id
            return binding.robot_worker_id
        if worker_key:
            worker_id = self._worker_keys.get(f'{role}:{worker_key}')
            if worker_id:
                return worker_id
        if role == WORKER_ROLE_POLICY:
            candidates = [w.worker_id for w in self._workers.values() if w.role == WORKER_ROLE_POLICY]
        else:
            candidates = [w.worker_id for w in self._workers.values() if w.role == WORKER_ROLE_ROBOT]
        if len(candidates) == 1:
            return candidates[0]
        raise HubError(
            f'Cannot resolve {role} worker; bind session or provide worker_key',
            code='WORKER_NOT_RESOLVED',
            http_status=404,
        )

    def _enqueue(self, worker_id: str, task: HubTask) -> None:
        self._queues[worker_id].append(task)
        self._queue_events[worker_id].set()

    def _prune_stale_workers(self) -> None:
        now = time.time()
        stale = [wid for wid, rec in self._workers.items() if now - rec.last_heartbeat > self._worker_ttl_s]
        for wid in stale:
            rec = self._workers.pop(wid, None)
            if rec:
                self._worker_keys.pop(f'{rec.role}:{rec.worker_key}', None)
            self._queues.pop(wid, None)
            self._queue_events.pop(wid, None)
