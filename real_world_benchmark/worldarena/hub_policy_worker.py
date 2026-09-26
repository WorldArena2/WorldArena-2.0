"""Hub policy worker loop (wa-hub-v1 on machine A)."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Optional

from real_world_benchmark.worldarena.hub_worker import HubWorkerClient
from real_world_benchmark.worldarena.hub_protocol import WORKER_ROLE_POLICY
from real_world_benchmark.worldarena.policy_remote import PolicyServerMetadata, dispatch_policy_endpoint

logger = logging.getLogger(__name__)


def _reset_on_register_enabled() -> bool:
    value = os.environ.get('WA_POLICY_RESET_ON_REGISTER', '1').strip().lower()
    return value not in ('0', 'false', 'no', 'off')


def run_policy_hub_worker(
    policy: Any,
    *,
    hub_url: str,
    worker_key: str,
    policy_source: str = '',
    legacy_bridge: bool = True,
    token: str = '',
) -> None:
    metadata = PolicyServerMetadata(
        policy_source=policy_source,
        supports_reset=callable(getattr(policy, 'reset', None)),
        supports_legacy_new_obs=legacy_bridge,
    ).to_dict()
    reset_lock = threading.Lock()
    reset_pending = threading.Event()
    reset_info: Dict[str, Any] = {}

    def on_register(register_response: Dict[str, Any], reregistered: bool) -> None:
        if not _reset_on_register_enabled() or not callable(getattr(policy, 'reset', None)):
            return
        with reset_lock:
            reset_info.clear()
            reset_info.update(
                {
                    'reason': 'hub_worker_reregister' if reregistered else 'hub_worker_register',
                    'worker_id': str(register_response.get('worker_id', '')),
                    'worker_key': worker_key,
                    'reregistered': bool(reregistered),
                }
            )
            reset_pending.set()

    client = HubWorkerClient(
        hub_url,
        role=WORKER_ROLE_POLICY,
        worker_key=worker_key,
        metadata=metadata,
        token=token,
        on_register=on_register,
    )

    def handler(request: Dict[str, Any]) -> Any:
        if reset_pending.is_set():
            with reset_lock:
                info = dict(reset_info)
                reset_pending.clear()
            logger.info(
                'Resetting policy state after hub %s worker_id=%s',
                info.get('reason', 'register'),
                info.get('worker_id', ''),
            )
            policy.reset(info)
        return dispatch_policy_endpoint(policy, request, legacy_bridge=legacy_bridge)

    client.run_forever(handler)
