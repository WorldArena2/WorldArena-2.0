"""HTTP long-poll worker client for wa-hub-v1."""

from __future__ import annotations

import logging
import threading
import time
import traceback
import json
from typing import Any, Callable, Dict, Optional
from urllib.parse import urljoin, urlparse

import requests

from real_world_benchmark.worldarena.hub_codec import (
    HUB_JSON_MIME,
    HUB_MSGPACK_MIME,
    is_msgpack_content_type,
    pack_hub_msgpack,
    unpack_hub_msgpack,
)
from real_world_benchmark.worldarena.hub_json import hub_json_decode
from real_world_benchmark.worldarena.hub_protocol import (
    HUB_DEFAULT_HEARTBEAT_INTERVAL_S,
    HUB_DEFAULT_POLL_TIMEOUT_S,
    HUB_ROUTE_WORKER_HEARTBEAT,
    HUB_ROUTE_WORKER_POLL,
    HUB_ROUTE_WORKER_REGISTER,
    HUB_ROUTE_WORKER_RESULT,
)

logger = logging.getLogger(__name__)


def normalize_hub_base_url(url: str) -> str:
    url = url.strip().rstrip('/')
    parsed = urlparse(url)
    if not parsed.scheme:
        url = f'https://{url}'
    return url.rstrip('/')


def is_local_hub_url(url: str) -> bool:
    hostname = urlparse(url).hostname or ''
    return hostname in ('localhost', '127.0.0.1', '::1') or hostname.startswith('127.')


class HubWorkerClient:
    """Register with Hub and run a long-poll loop."""

    def __init__(
        self,
        hub_url: str,
        *,
        role: str,
        worker_key: str,
        metadata: Optional[Dict[str, Any]] = None,
        token: str = '',
        poll_timeout_s: float = HUB_DEFAULT_POLL_TIMEOUT_S,
        on_register: Optional[Callable[[Dict[str, Any], bool], None]] = None,
    ) -> None:
        self._base_url = normalize_hub_base_url(hub_url)
        self._role = role
        self._worker_key = worker_key
        self._metadata = dict(metadata or {})
        self._token = token.strip()
        self._poll_timeout_s = poll_timeout_s
        self._on_register = on_register
        self._worker_id = ''
        # Main loop (register / long-poll / submit_result) and heartbeat run on
        # different threads; requests.Session is not thread-safe.
        self._session = requests.Session()
        self._heartbeat_session = requests.Session()
        if is_local_hub_url(self._base_url):
            self._session.trust_env = False
            self._heartbeat_session.trust_env = False
        self._stop = threading.Event()
        self._heartbeat_interval_s = HUB_DEFAULT_HEARTBEAT_INTERVAL_S
        self._register_lock = threading.Lock()

    @property
    def worker_id(self) -> str:
        return self._worker_id

    def _headers(self, *, content_type: str = HUB_JSON_MIME, accept_msgpack: bool = False) -> Dict[str, str]:
        headers = {'Content-Type': content_type, 'X-Hub-Protocol': 'wa-hub-v1'}
        if accept_msgpack:
            headers['Accept'] = HUB_MSGPACK_MIME
        if self._token:
            headers['Authorization'] = f'Bearer {self._token}'
        return headers

    def _url(self, path: str) -> str:
        return urljoin(self._base_url + '/', path.lstrip('/'))

    def register(self) -> Dict[str, Any]:
        with self._register_lock:
            response = self._session.post(
                self._url(HUB_ROUTE_WORKER_REGISTER),
                json={'role': self._role, 'worker_key': self._worker_key, 'metadata': self._metadata},
                headers=self._headers(),
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            old_worker_id = self._worker_id
            self._worker_id = str(data['worker_id'])
            self._heartbeat_interval_s = int(data.get('heartbeat_interval_s', HUB_DEFAULT_HEARTBEAT_INTERVAL_S))
            reregistered = bool(old_worker_id and old_worker_id != self._worker_id)
            if reregistered:
                logger.warning(
                    'Re-registered hub worker key=%s old_worker_id=%s new_worker_id=%s',
                    self._worker_key,
                    old_worker_id,
                    self._worker_id,
                )
            else:
                logger.info('Registered hub worker %s key=%s', self._worker_id, self._worker_key)
            if self._on_register is not None:
                self._on_register(data, reregistered)
            return data

    @staticmethod
    def _response_json_safe(response: requests.Response) -> Dict[str, Any]:
        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _decode_response(response: requests.Response) -> Dict[str, Any]:
        if is_msgpack_content_type(response.headers.get('Content-Type', '')):
            data = unpack_hub_msgpack(response.content)
            return data if isinstance(data, dict) else {}
        return hub_json_decode(response.json())

    def _should_reregister(self, exc: requests.RequestException) -> bool:
        response = getattr(exc, 'response', None)
        if response is None:
            return False
        if response.status_code != 404:
            return False
        data = self._response_json_safe(response)
        return str(data.get('code', '')) == 'UNKNOWN_WORKER'

    def _reregister_after_worker_loss(self, source: str) -> bool:
        if self._stop.is_set():
            return False
        try:
            logger.warning(
                'Hub worker lost registration during %s; re-registering key=%s',
                source,
                self._worker_key,
            )
            self.register()
            return True
        except requests.RequestException as reg_exc:
            logger.warning('Hub worker re-register failed during %s: %s', source, reg_exc)
            return False

    def heartbeat(self) -> None:
        if not self._worker_id:
            return
        try:
            self._heartbeat_session.post(
                self._url(HUB_ROUTE_WORKER_HEARTBEAT),
                json={'worker_id': self._worker_id},
                headers=self._headers(),
                timeout=10,
            ).raise_for_status()
        except requests.RequestException as exc:
            if self._should_reregister(exc) and self._reregister_after_worker_loss('heartbeat'):
                return
            raise

    def poll_once(self) -> Optional[Dict[str, Any]]:
        if not self._worker_id:
            raise RuntimeError('Worker not registered')
        try:
            response = self._session.get(
                self._url(HUB_ROUTE_WORKER_POLL),
                params={'worker_id': self._worker_id, 'timeout_s': self._poll_timeout_s},
                headers=self._headers(accept_msgpack=True),
                timeout=self._poll_timeout_s + 10,
            )
            if response.status_code == 204:
                return None
            response.raise_for_status()
        except requests.RequestException as exc:
            if self._should_reregister(exc) and self._reregister_after_worker_loss('poll'):
                return None
            raise
        data = self._decode_response(response)
        task = data.get('task')
        return task if isinstance(task, dict) else None

    def submit_result(
        self,
        *,
        request_id: str,
        status: str,
        result: Any = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            self._session.post(
                self._url(HUB_ROUTE_WORKER_RESULT),
                data=pack_hub_msgpack(
                    {
                        'request_id': request_id,
                        'worker_id': self._worker_id,
                        'status': status,
                        'result': result,
                        'error': error,
                    }
                ),
                headers=self._headers(content_type=HUB_MSGPACK_MIME),
                timeout=60,
            ).raise_for_status()
        except requests.RequestException as exc:
            if self._should_reregister(exc) and self._reregister_after_worker_loss('submit_result'):
                raise exc
            raise

    def run_forever(self, handler: Callable[[Dict[str, Any]], Any]) -> None:
        self.register()
        heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        heartbeat_thread.start()
        logger.info('Hub worker poll loop started for %s', self._worker_id)
        while not self._stop.is_set():
            try:
                task = self.poll_once()
                if task is None:
                    continue
                request_id = str(task.get('request_id', ''))
                endpoint = str(task.get('endpoint', ''))
                payload = task.get('payload') if isinstance(task.get('payload'), dict) else {}
                request = {'endpoint': endpoint, **payload}
                logger.info('Hub task started endpoint=%s request_id=%s', endpoint, request_id)
                try:
                    result = handler(request)
                except Exception as exc:
                    logger.exception('Task %s endpoint=%s failed', request_id, endpoint)
                    result = None
                    status = 'error'
                    error = {
                        'code': 'WORKER_EXECUTION_ERROR',
                        'message': str(exc),
                        'traceback': traceback.format_exc(),
                    }
                else:
                    status = 'ok'
                    error = None

                self.submit_result(
                    request_id=request_id,
                    status=status,
                    result=result,
                    error=error,
                )
                if status == 'ok':
                    logger.info('Hub task completed endpoint=%s request_id=%s', endpoint, request_id)
                else:
                    logger.info(
                        'Hub task error reported endpoint=%s request_id=%s code=%s',
                        endpoint,
                        request_id,
                        error.get('code', ''),
                    )
            except requests.RequestException as exc:
                logger.warning('Hub poll/request failed: %s', exc)
                time.sleep(1.0)

    def stop(self) -> None:
        self._stop.set()

    def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.heartbeat()
            except requests.RequestException as exc:
                logger.warning('Hub heartbeat failed: %s', exc)
            self._stop.wait(self._heartbeat_interval_s)
