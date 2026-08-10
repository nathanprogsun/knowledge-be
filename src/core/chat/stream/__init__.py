"""Chat stream manager: append-only event store for SSE streaming."""

from __future__ import annotations

from src.core.chat.stream.manager import (
    DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    DEFAULT_REDIS_PREFIX,
    DEFAULT_REDIS_TTL_SECONDS,
    MemoryStreamManager,
    RedisStreamManager,
    StreamManager,
    StreamManagerConfig,
    create_stream_manager,
)
from src.core.chat.stream.types import Event, StreamData, StreamKey

__all__ = [
    "DEFAULT_HEARTBEAT_INTERVAL_SECONDS",
    "DEFAULT_REDIS_PREFIX",
    "DEFAULT_REDIS_TTL_SECONDS",
    "Event",
    "MemoryStreamManager",
    "RedisStreamManager",
    "StreamData",
    "StreamKey",
    "StreamManager",
    "StreamManagerConfig",
    "create_stream_manager",
]
