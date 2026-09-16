# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from . import configs, distributed, modules
from .image2video import WanI2V
from .text2video import WanT2V
from .textimage2video import WanTI2V
from .tactile_model import WanTactile

try:
	from .animate import WanAnimate
except Exception:  # pragma: no cover - optional dependency (e.g., peft)
	WanAnimate = None

try:
	from .speech2video import WanS2V
except Exception:  # pragma: no cover - optional dependency (e.g., librosa)
	WanS2V = None