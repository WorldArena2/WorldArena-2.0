"""CLI entry point for serving a policy over wa-policy-v1 or wa-hub-v1."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Iterable

from real_world_benchmark.policy_loader import load_policy
from real_world_benchmark.worldarena.hub_policy_worker import run_policy_hub_worker
from real_world_benchmark.worldarena.policy_remote import CanonicalPolicyServer


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Serve a Policy over wa-policy-v1 or wa-hub-v1')
    parser.add_argument(
        'policy',
        nargs='?',
        default='real_world_benchmark.examples.policy_template.policy',
        help='module path or .py file defining Policy',
    )
    parser.add_argument('--host', default='0.0.0.0', help='WebSocket bind host')
    parser.add_argument('--port', type=int, default=8000, help='WebSocket bind port')
    parser.add_argument(
        '--hub-url',
        default=None,
        help='Hub worker base URL, e.g. https://gateway.example.com/policy',
    )
    parser.add_argument('--worker-key', default='', help='Hub worker_key (defaults to policy module name)')
    parser.add_argument('--hub-token', default='', help='optional Bearer token for Hub HTTP requests')
    parser.add_argument(
        '--no-legacy-bridge',
        action='store_true',
        help='disable legacy new_obs bridge (only accept observation_packet)',
    )
    return parser.parse_args(list(argv))


def main(argv: list[str]) -> int:
    args = parse_args(argv[1:])
    logging.basicConfig(level=logging.INFO)

    print(f'Loading policy from: {args.policy}')
    mod = load_policy(args.policy)
    if not hasattr(mod, 'Policy'):
        print('ERROR: module does not define class Policy')
        return 2

    policy_cls = getattr(mod, 'Policy')
    policy = policy_cls()
    worker_key = args.worker_key or args.policy

    if args.hub_url:
        print(f'Registering policy hub worker at {args.hub_url} key={worker_key}')
        run_policy_hub_worker(
            policy,
            hub_url=args.hub_url,
            worker_key=worker_key,
            policy_source=args.policy,
            legacy_bridge=not args.no_legacy_bridge,
            token=args.hub_token,
        )
        return 0

    server = CanonicalPolicyServer(
        policy,
        policy_source=args.policy,
        host=args.host,
        port=args.port,
        legacy_bridge=not args.no_legacy_bridge,
    )
    print(f'Serving policy at ws://{args.host}:{args.port} (protocol wa-policy-v1, schema worldarena.v1)')
    server.serve_forever()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
