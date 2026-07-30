"""JSON-safe encoding for wa-hub-v1 HTTP transport (bytes → base64)."""

from __future__ import annotations

import base64
from typing import Any


def _is_numpy_scalar(value: Any) -> bool:
    type_name = type(value).__name__
    module = getattr(type(value), '__module__', '')
    return module.startswith('numpy') and type_name.endswith(('float32', 'float64', 'int32', 'int64', 'bool_'))

def _is_numpy_array(value: Any) -> bool:
    module = getattr(type(value), '__module__', '')
    return module.startswith('numpy') and type(value).__name__ == 'ndarray'


def hub_json_encode(value: Any) -> Any:
    if isinstance(value, bytes):
        return {'$b64$': base64.b64encode(value).decode('ascii')}
    if _is_numpy_array(value):
        return hub_json_encode(value.tolist())
    if _is_numpy_scalar(value):
        return hub_json_encode(value.item())
    if isinstance(value, dict):
        return {str(k): hub_json_encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [hub_json_encode(v) for v in value]
    return value


def hub_json_decode(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value.keys()) == {'$b64$'}:
            return base64.b64decode(str(value['$b64$']))
        return {str(k): hub_json_decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [hub_json_decode(v) for v in value]
    return value
