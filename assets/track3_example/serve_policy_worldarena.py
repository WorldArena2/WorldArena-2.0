"""CLI entry point for serving a policy over wa-policy-v1 or wa-hub-v1.

This local copy lives in track3_example/ and imports the local worldarena/
package, so the folder can be used standalone without the full repo.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Iterable

from worldarena.hub_policy_worker import run_policy_hub_worker
from worldarena.policy_remote import CanonicalPolicyServer


def load_policy(path_or_module: str):
    """Load a Policy class from a .py file or dotted module name."""
    p = Path(path_or_module)
    if p.suffix == '.py' and not p.exists():
        raise FileNotFoundError(
            f'Policy file not found: {path_or_module}\n'
            'Use a real path (e.g. ./dummy_policy.py) or module '
            '(e.g. dummy_policy).'
        )
    if p.exists() and p.suffix == '.py':
        spec = importlib.util.spec_from_file_location('user_policy', str(p))
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot import policy from file: {path_or_module}')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[arg-type]
        return mod
    try:
        return importlib.import_module(path_or_module)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f'Cannot import policy module {path_or_module!r}. '
            'Pass a .py file path or a dotted module name '
            '(e.g. dummy_policy).'
        ) from exc


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Serve a Policy over wa-policy-v1 or wa-hub-v1')
    parser.add_argument(
        'policy',
        nargs='?',
        default='dummy_policy',
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
