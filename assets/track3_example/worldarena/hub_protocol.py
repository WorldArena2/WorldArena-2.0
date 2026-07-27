"""wa-hub-v1 HTTP Hub protocol constants (long-poll worker registration)."""

from __future__ import annotations

HUB_PROTOCOL_VERSION = 'wa-hub-v1'
HUB_TRANSPORT_HTTP = 'http'

# Volcano gateway path prefixes (external URL); container routes omit these when gateway strips prefix.
HUB_GATEWAY_POLICY_PREFIX = '/policy'
HUB_GATEWAY_ROBOT_PREFIX = '/robot'

# In-container route paths (default when gateway strips /policy and /robot).
HUB_ROUTE_HEALTH = '/health'
HUB_ROUTE_WORKER_REGISTER = '/workers/register'
HUB_ROUTE_WORKER_HEARTBEAT = '/workers/heartbeat'
HUB_ROUTE_WORKER_POLL = '/workers/poll'
HUB_ROUTE_WORKER_RESULT = '/workers/result'
HUB_ROUTE_SESSION_BIND = '/sessions/bind'
HUB_ROUTE_ORCHESTRATOR_POLICY = '/orchestrator/policy'
HUB_ROUTE_ORCHESTRATOR_ROBOT = '/orchestrator/robot'

# Hub management endpoint names (for logging).
HUB_ENDPOINT_HEALTH = 'health'
HUB_ENDPOINT_WORKER_REGISTER = 'workers/register'
HUB_ENDPOINT_WORKER_HEARTBEAT = 'workers/heartbeat'
HUB_ENDPOINT_WORKER_POLL = 'workers/poll'
HUB_ENDPOINT_WORKER_RESULT = 'workers/result'
HUB_ENDPOINT_SESSION_BIND = 'sessions/bind'
HUB_ENDPOINT_ORCHESTRATOR_POLICY = 'orchestrator/policy'
HUB_ENDPOINT_ORCHESTRATOR_ROBOT = 'orchestrator/robot'

# Worker roles (Hub routing).
WORKER_ROLE_POLICY = 'policy'
WORKER_ROLE_ROBOT = 'robot'

# Recommended long-poll timeout (seconds). Keep below API gateway read timeout.
HUB_DEFAULT_POLL_TIMEOUT_S = 25
HUB_DEFAULT_POLL_MAX_TIMEOUT_S = 55
HUB_DEFAULT_HEARTBEAT_INTERVAL_S = 15
HUB_DEFAULT_RPC_TIMEOUT_S = 120.0
