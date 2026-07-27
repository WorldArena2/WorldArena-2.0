"""Serialization helpers for WorldArena protocol messages."""

from __future__ import annotations

from typing import Any, Dict, Type, TypeVar

from worldarena import msgpack_numpy

from worldarena.protocol import SCHEMA_VERSION
from worldarena.schema import (
    ActionPacket,
    EpisodeEvent,
    ObservationPacket,
)

T = TypeVar('T')


def pack_message(data: Any) -> bytes:
    return msgpack_numpy.Packer().pack(data)


def unpack_message(data: bytes | Any) -> Any:
    if isinstance(data, bytes):
        return msgpack_numpy.unpackb(data)
    return data


def packet_to_bytes(packet: ObservationPacket | ActionPacket | EpisodeEvent) -> bytes:
    return pack_message(packet.to_dict())


def observation_from_message(data: Dict[str, Any]) -> ObservationPacket:
    validate_schema_version(data, SCHEMA_VERSION)
    return ObservationPacket.from_dict(data)


def action_from_message(data: Dict[str, Any]) -> ActionPacket:
    validate_schema_version(data, SCHEMA_VERSION)
    return ActionPacket.from_dict(data)


def event_from_message(data: Dict[str, Any]) -> EpisodeEvent:
    validate_schema_version(data, SCHEMA_VERSION)
    return EpisodeEvent.from_dict(data)


def validate_schema_version(data: Dict[str, Any], expected: str) -> None:
    version = data.get('schema_version')
    if version and version != expected:
        raise ValueError(f'Unsupported schema version: {version} (expected {expected})')
