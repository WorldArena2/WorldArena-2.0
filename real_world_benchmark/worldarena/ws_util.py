"""Shared WebSocket connection helpers."""

from __future__ import annotations

import logging
from typing import Tuple
from urllib.parse import urlparse

import websockets.sync.client

logger = logging.getLogger(__name__)


def normalize_websocket_uri(uri: str) -> str:
    uri = uri.strip()
    if '://' not in uri:
        return f'ws://{uri}'
    parsed = urlparse(uri)
    if parsed.scheme in ('ws', 'wss'):
        return uri
    if parsed.scheme == 'http':
        return f'ws://{parsed.netloc}{parsed.path}'
    if parsed.scheme == 'https':
        return f'wss://{parsed.netloc}{parsed.path}'
    raise ValueError(f'Unsupported WebSocket URI scheme: {parsed.scheme}')


def connect_websocket(uri: str) -> Tuple[websockets.sync.client.ClientConnection, str]:
    """Connect to a WebSocket server, retrying with wss:// if ws:// fails."""
    normalized = normalize_websocket_uri(uri)
    logger.info('Connecting to WebSocket server at %s', normalized)
    try:
        conn = websockets.sync.client.connect(normalized, compression=None, max_size=None)
        return conn, normalized
    except Exception:
        if normalized.startswith('ws://'):
            fallback = 'wss://' + normalized[len('ws://') :]
            logger.info('ws:// connection failed, retrying with %s', fallback)
            conn = websockets.sync.client.connect(fallback, compression=None, max_size=None)
            return conn, fallback
        raise
