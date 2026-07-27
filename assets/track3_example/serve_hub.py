"""CLI entry point for wa-hub-v1 HTTP Hub (dual-port policy/robot)."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Iterable

from worldarena.hub_core import HubCore
from worldarena.hub_server import HubServer


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Serve WorldArena wa-hub-v1 HTTP Hub')
    parser.add_argument('--host', default='0.0.0.0', help='bind host for policy and robot ports')
    parser.add_argument('--policy-port', type=int, default=8000, help='policy worker + orchestrator port')
    parser.add_argument('--robot-port', type=int, default=9000, help='robot worker + orchestrator port')
    parser.add_argument(
        '--policy-url-prefix',
        default='',
        help='optional in-container URL prefix if gateway does not strip /policy (default: empty)',
    )
    parser.add_argument(
        '--robot-url-prefix',
        default='',
        help='optional in-container URL prefix if gateway does not strip /robot (default: empty)',
    )
    return parser.parse_args(list(argv))


def main(argv: list[str]) -> int:
    args = parse_args(argv[1:])
    logging.basicConfig(level=logging.INFO)

    core = HubCore()
    server = HubServer(
        core,
        host=args.host,
        policy_port=args.policy_port,
        robot_port=args.robot_port,
        policy_url_prefix=args.policy_url_prefix,
        robot_url_prefix=args.robot_url_prefix,
    )
    print(
        f'Serving wa-hub-v1 on {args.host}:{args.policy_port} (policy) '
        f'and {args.host}:{args.robot_port} (robot)'
    )
    server.serve_forever()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
