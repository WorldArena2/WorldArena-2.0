"""Image byte codecs used by WorldArena canonical transport."""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np


JPEG_ENCODING = 'jpeg'
DEFAULT_JPEG_QUALITY = 80


def jpeg_quality_from_env(*names: str, default: int = DEFAULT_JPEG_QUALITY) -> int:
    for name in names:
        value = os.environ.get(name)
        if value is None or value == '':
            continue
        try:
            return max(1, min(100, int(value)))
        except ValueError as exc:
            raise ValueError(f'{name} must be an integer JPEG quality in [1, 100], got {value!r}') from exc
    return default


def camera_jpeg_quality() -> int:
    return jpeg_quality_from_env('WA_CAMERA_JPEG_QUALITY', 'WA_JPEG_QUALITY')


def tactile_jpeg_quality() -> int:
    return jpeg_quality_from_env('WA_TACTILE_JPEG_QUALITY', 'WA_JPEG_QUALITY')


def to_uint8_hwc(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        if array.size and np.nanmax(array) <= 1.0:
            array = (array * 255.0).clip(0, 255).astype(np.uint8)
        else:
            array = array.clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def encode_jpeg(image: Any, *, quality: Optional[int] = None) -> bytes:
    import cv2

    array = to_uint8_hwc(image)
    quality = jpeg_quality_from_env(default=DEFAULT_JPEG_QUALITY) if quality is None else quality
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, encoded = cv2.imencode('.jpg', array, params)
    if not ok:
        raise RuntimeError('Failed to encode image as JPEG')
    return encoded.tobytes()


def decode_image_bytes(
    data: bytes,
    *,
    encoding: str,
    shape: Optional[list[int]] = None,
    dtype: str = 'uint8',
) -> np.ndarray:
    key = (encoding or 'raw').strip().lower()
    if key in ('jpeg', 'jpg'):
        import cv2

        encoded = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Failed to decode JPEG image bytes')
        return image

    array = np.frombuffer(data, dtype=np.dtype(dtype) if dtype else np.uint8)
    if shape:
        array = array.reshape(shape)
    return array
