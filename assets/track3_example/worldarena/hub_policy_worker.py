"""Hub policy worker loop (wa-hub-v1 on machine A)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from worldarena.hub_worker import HubWorkerClient
from worldarena.hub_protocol import WORKER_ROLE_POLICY
from worldarena.policy_remote import PolicyServerMetadata, dispatch_policy_endpoint


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
    client = HubWorkerClient(
        hub_url,
        role=WORKER_ROLE_POLICY,
        worker_key=worker_key,
        metadata=metadata,
        token=token,
    )

    def handler(request: Dict[str, Any]) -> Any:
        return dispatch_policy_endpoint(policy, request, legacy_bridge=legacy_bridge)

    client.run_forever(handler)
