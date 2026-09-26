"""HTTP body codecs for wa-hub-v1.

Heavy Hub RPC payloads can carry image bytes.  JSON remains supported for
compatibility, while msgpack avoids base64 expansion on binary fields.
"""

from __future__ import annotations

from typing import Any

from real_world_benchmark.worldarena import msgpack_numpy


HUB_JSON_MIME = 'application/json'
HUB_MSGPACK_MIME = 'application/msgpack'
HUB_MSGPACK_MIME_ALT = 'application/x-msgpack'


def pack_hub_msgpack(value: Any) -> bytes:
    return msgpack_numpy.Packer().pack(value)


def unpack_hub_msgpack(data: bytes) -> Any:
    return msgpack_numpy.unpackb(data)


def is_msgpack_content_type(content_type: str) -> bool:
    key = (content_type or '').split(';', 1)[0].strip().lower()
    return key in (HUB_MSGPACK_MIME, HUB_MSGPACK_MIME_ALT)


def accepts_msgpack(accept_header: str) -> bool:
    return any(
        part.split(';', 1)[0].strip().lower() in (HUB_MSGPACK_MIME, HUB_MSGPACK_MIME_ALT)
        for part in (accept_header or '').split(',')
    )
