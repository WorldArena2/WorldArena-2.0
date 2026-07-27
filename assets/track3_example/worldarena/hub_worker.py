"""HTTP long-poll worker client for wa-hub-v1."""

from __future__ import annotations

import logging
import threading
import time
import traceback
from typing import Any, Callable, Dict, Optional
from urllib.parse import urljoin, urlparse

import requests

from worldarena.hub_json import hub_json_decode, hub_json_encode

from worldarena.hub_protocol import (
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
    ) -> None:
        self._base_url = normalize_hub_base_url(hub_url)
        self._role = role
        self._worker_key = worker_key
        self._metadata = dict(metadata or {})
        self._token = token.strip()
        self._poll_timeout_s = poll_timeout_s
        self._worker_id = ''
        # Main loop (register / long-poll / submit_result) and heartbeat run on
        # different threads; requests.Session is not thread-safe.
        self._session = requests.Session()
        self._heartbeat_session = requests.Session()
        self._stop = threading.Event()
        self._heartbeat_interval_s = HUB_DEFAULT_HEARTBEAT_INTERVAL_S

    @property
    def worker_id(self) -> str:
        return self._worker_id

    def _headers(self) -> Dict[str, str]:
        headers = {'Content-Type': 'application/json', 'X-Hub-Protocol': 'wa-hub-v1'}
        if self._token:
            headers['Authorization'] = f'Bearer {self._token}'
        return headers

    def _url(self, path: str) -> str:
        return urljoin(self._base_url + '/', path.lstrip('/'))

    def register(self) -> Dict[str, Any]:
        response = self._session.post(
            self._url(HUB_ROUTE_WORKER_REGISTER),
            json={'role': self._role, 'worker_key': self._worker_key, 'metadata': self._metadata},
            headers=self._headers(),
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        self._worker_id = str(data['worker_id'])
        self._heartbeat_interval_s = int(data.get('heartbeat_interval_s', HUB_DEFAULT_HEARTBEAT_INTERVAL_S))
        logger.info('Registered hub worker %s key=%s', self._worker_id, self._worker_key)
        return data

    def heartbeat(self) -> None:
        if not self._worker_id:
            return
        self._heartbeat_session.post(
            self._url(HUB_ROUTE_WORKER_HEARTBEAT),
            json={'worker_id': self._worker_id},
            headers=self._headers(),
            timeout=10,
        ).raise_for_status()

    def poll_once(self) -> Optional[Dict[str, Any]]:
        if not self._worker_id:
            raise RuntimeError('Worker not registered')
        response = self._session.get(
            self._url(HUB_ROUTE_WORKER_POLL),
            params={'worker_id': self._worker_id, 'timeout_s': self._poll_timeout_s},
            headers=self._headers(),
            timeout=self._poll_timeout_s + 10,
        )
        if response.status_code == 204:
            return None
        response.raise_for_status()
        data = response.json()
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
        self._session.post(
            self._url(HUB_ROUTE_WORKER_RESULT),
            json=hub_json_encode(
                {
                    'request_id': request_id,
                    'worker_id': self._worker_id,
                    'status': status,
                    'result': result,
                    'error': error,
                }
            ),
            headers=self._headers(),
            timeout=60,
        ).raise_for_status()

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
                request = hub_json_decode({'endpoint': endpoint, **payload})
                logger.info('Hub task started endpoint=%s request_id=%s', endpoint, request_id)
                try:
                    result = handler(request)
                    self.submit_result(request_id=request_id, status='ok', result=result)
                    logger.info('Hub task completed endpoint=%s request_id=%s', endpoint, request_id)
                except Exception as exc:
                    logger.exception('Task %s endpoint=%s failed', request_id, endpoint)
                    self.submit_result(
                        request_id=request_id,
                        status='error',
                        result=None,
                        error={
                            'code': 'WORKER_EXECUTION_ERROR',
                            'message': str(exc),
                            'traceback': traceback.format_exc(),
                        },
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
