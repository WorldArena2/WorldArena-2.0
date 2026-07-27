"""Robot adapters for WorldArena 2.0."""

from worldarena.adapters.base import RobotAdapter
from worldarena.adapters.manifold_msg import ManifoldMsgAdapter
from worldarena.adapters.xense import XenseTactileCollector

__all__ = ['RobotAdapter', 'ManifoldMsgAdapter', 'XenseTactileCollector']
