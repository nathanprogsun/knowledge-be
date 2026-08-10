"""Chat stream manager — append-only event store for SSE streaming.

Mirrors the upstream stream-manager abstraction: one interface with an
in-memory backend and an optional Redis-backed backend, selected at
runtime by configuration. The manager owns the append-only event list per
(session, message) pair, incremental reads from an offset, per-stream
cancellation, and an optional heartbeat that appends ``ping`` events so
long-lived SSE connections stay alive.

The handler layer depends on the ``StreamManager`` interface only; the
concrete backend is chosen by ``create_stream_manager`` from
``StreamManagerConfig`` (backend type from the ``STREAM_MANAGER_TYPE``
environment variable, Redis connection from the shared application
settings).
"""

from __future__ import annotations

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from redis.asyncio import Redis

from src.core.chat.stream.types import Event, StreamData, StreamKey
from src.settings import get_settings

logger = logging.getLogger(__name__)

#: Default Redis key prefix for stream event lists.
DEFAULT_REDIS_PREFIX = "stream:events"
#: Default TTL for Redis stream keys (24 hours).
DEFAULT_REDIS_TTL_SECONDS = 24 * 60 * 60
#: Default interval between heartbeat ``ping`` events.
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 15.0
#: Environment variable selecting the backend ("memory" or "redis").
STREAM_MANAGER_TYPE_ENV = "STREAM_MANAGER_TYPE"

#: Backend identifiers accepted by ``StreamManagerConfig``.
_BACKENDS = frozenset({"memory", "redis"})


class StreamManagerConfig(BaseModel):
    """Runtime configuration selecting the stream-manager backend."""

    model_config = ConfigDict(frozen=True)

    backend: Literal["memory", "redis"] = "memory"
    redis_url: str = "redis://localhost:6379"
    redis_prefix: str = DEFAULT_REDIS_PREFIX
    redis_ttl_seconds: int = DEFAULT_REDIS_TTL_SECONDS
    heartbeat_interval_seconds: float = Field(
        default=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        gt=0,
    )

    @classmethod
    def from_settings(cls) -> StreamManagerConfig:
        """Build the config from process settings and the backend env var.

        The backend is read from ``STREAM_MANAGER_TYPE`` (defaults to
        ``memory``); the Redis connection string comes from the shared
        application settings. Unknown backend values fall back to memory.
        """
        settings = get_settings()
        backend = os.environ.get(STREAM_MANAGER_TYPE_ENV, "memory").strip().lower()
        if backend not in _BACKENDS:
            backend = "memory"
        return cls(
            backend=backend,  # type: ignore[arg-type]
            redis_url=settings.redis_url,
        )


class StreamManager(ABC):
    """Append-only event store for chat streams.

    Concrete backends implement the storage operations; cancellation and
    heartbeat are shared in-process concerns provided by this base class.
    """

    def __init__(
        self,
        *,
        heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        if heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._cancel_events: dict[StreamKey, asyncio.Event] = {}

    @abstractmethod
    async def append_event(
        self,
        session_id: str,
        message_id: str,
        event: Event,
    ) -> None:
        """Append a single event to the stream, creating it if needed."""

    @abstractmethod
    async def get_events(
        self,
        session_id: str,
        message_id: str,
        offset: int,
        limit: int | None = None,
    ) -> tuple[list[Event], int]:
        """Return events from ``offset`` (up to ``limit``) and the next offset.

        A missing stream yields an empty list and the unchanged offset.
        ``limit`` of ``None`` returns every event from ``offset`` onward.
        """

    @abstractmethod
    async def get_or_create_stream(
        self,
        session_id: str,
        message_id: str,
    ) -> StreamData:
        """Return the stream state, creating an empty one if absent."""

    def cancel(self, session_id: str, message_id: str) -> None:
        """Mark the stream cancelled; heartbeat and readers observe it."""
        key = StreamKey(session_id=session_id, message_id=message_id)
        event = self._cancel_events.get(key)
        if event is None:
            event = asyncio.Event()
            self._cancel_events[key] = event
        event.set()

    def is_cancelled(self, session_id: str, message_id: str) -> bool:
        """Return whether the stream has been cancelled."""
        key = StreamKey(session_id=session_id, message_id=message_id)
        event = self._cancel_events.get(key)
        return event is not None and event.is_set()

    async def wait_cancelled(self, session_id: str, message_id: str) -> None:
        """Wait until the stream is cancelled (returns immediately if so)."""
        key = StreamKey(session_id=session_id, message_id=message_id)
        event = self._cancel_events.get(key)
        if event is None:
            event = asyncio.Event()
            self._cancel_events[key] = event
        await event.wait()

    def start_heartbeat(self, session_id: str, message_id: str) -> asyncio.Task[None]:
        """Start a background task appending ``ping`` events every interval.

        The task stops when cancelled or when the stream is cancelled.
        """
        return asyncio.create_task(self._heartbeat_loop(session_id, message_id))

    async def _heartbeat_loop(self, session_id: str, message_id: str) -> None:
        try:
            while True:
                await asyncio.sleep(self._heartbeat_interval_seconds)
                if self.is_cancelled(session_id, message_id):
                    return
                await self.append_event(
                    session_id,
                    message_id,
                    Event(id=f"ping-{uuid4().hex}", type="ping"),
                )
        except asyncio.CancelledError:
            return

    @abstractmethod
    async def close(self) -> None:
        """Release backend resources."""


class MemoryStreamManager(StreamManager):
    """In-memory stream backend (mirrors the upstream memory manager)."""

    def __init__(
        self,
        *,
        heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        super().__init__(heartbeat_interval_seconds=heartbeat_interval_seconds)
        self._streams: dict[StreamKey, StreamData] = {}

    async def get_or_create_stream(
        self,
        session_id: str,
        message_id: str,
    ) -> StreamData:
        key = StreamKey(session_id=session_id, message_id=message_id)
        stream = self._streams.get(key)
        if stream is None:
            stream = StreamData()
            self._streams[key] = stream
        return stream

    async def append_event(
        self,
        session_id: str,
        message_id: str,
        event: Event,
    ) -> None:
        key = StreamKey(session_id=session_id, message_id=message_id)
        stream = self._streams.get(key)
        if stream is None:
            stream = StreamData()
        self._streams[key] = StreamData(
            events=(*stream.events, event),
            last_updated=datetime.now(UTC),
        )

    async def get_events(
        self,
        session_id: str,
        message_id: str,
        offset: int,
        limit: int | None = None,
    ) -> tuple[list[Event], int]:
        key = StreamKey(session_id=session_id, message_id=message_id)
        stream = self._streams.get(key)
        if stream is None:
            return [], offset
        events = stream.events
        if offset >= len(events):
            return [], offset
        end = len(events) if limit is None else min(offset + limit, len(events))
        return list(events[offset:end]), end

    async def close(self) -> None:
        """Release backend resources (no-op: nothing to release)."""


class RedisStreamManager(StreamManager):
    """Redis-list stream backend (mirrors the upstream Redis manager).

    Events are appended with ``RPush`` (O(1)) and read incrementally with
    ``LRange``; the key carries a TTL so stale streams expire. The Redis
    connection is injected so tests can substitute a fake client.
    """

    def __init__(
        self,
        *,
        redis: Redis[str],
        prefix: str = DEFAULT_REDIS_PREFIX,
        ttl_seconds: int = DEFAULT_REDIS_TTL_SECONDS,
        heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        super().__init__(heartbeat_interval_seconds=heartbeat_interval_seconds)
        self._redis = redis
        self._prefix = prefix
        self._ttl_seconds = ttl_seconds

    def _build_key(self, session_id: str, message_id: str) -> str:
        return f"{self._prefix}:{session_id}:{message_id}"

    async def get_or_create_stream(
        self,
        session_id: str,
        message_id: str,
    ) -> StreamData:
        # Redis has no explicit stream record; the key is created lazily on
        # first append. Return an empty snapshot so callers get a uniform
        # shape regardless of backend.
        events, _ = await self.get_events(session_id, message_id, 0)
        return StreamData(events=tuple(events))

    async def append_event(
        self,
        session_id: str,
        message_id: str,
        event: Event,
    ) -> None:
        key = self._build_key(session_id, message_id)
        await self._redis.rpush(key, event.model_dump_json())
        await self._redis.expire(key, self._ttl_seconds)

    async def get_events(
        self,
        session_id: str,
        message_id: str,
        offset: int,
        limit: int | None = None,
    ) -> tuple[list[Event], int]:
        key = self._build_key(session_id, message_id)
        if limit is None:
            raw = await self._redis.lrange(key, offset, -1)
        else:
            raw = await self._redis.lrange(key, offset, offset + limit - 1)
        if not raw:
            return [], offset
        events: list[Event] = []
        for item in raw:
            try:
                events.append(Event.model_validate_json(item))
            except Exception:
                logger.warning("Skipping unparseable stream event at key %s", key)
                continue
        return events, offset + len(events)

    async def close(self) -> None:
        # ``aclose`` is the non-deprecated asyncio API; the pinned
        # types-redis stubs only declare ``close``, so the gap is ignored.
        await self._redis.aclose()  # type: ignore[attr-defined]


def create_stream_manager(config: StreamManagerConfig | None = None) -> StreamManager:
    """Create a stream manager for the configured backend.

    With no explicit config the backend is read from the environment via
    ``StreamManagerConfig.from_settings``. The Redis client is created
    lazily (no connection is opened until the first command); callers own
    the returned manager and should ``close()`` it when done.
    """
    if config is None:
        config = StreamManagerConfig.from_settings()
    if config.backend == "redis":
        return RedisStreamManager(
            redis=Redis.from_url(config.redis_url, decode_responses=True),
            prefix=config.redis_prefix,
            ttl_seconds=config.redis_ttl_seconds,
            heartbeat_interval_seconds=config.heartbeat_interval_seconds,
        )
    return MemoryStreamManager(
        heartbeat_interval_seconds=config.heartbeat_interval_seconds,
    )


__all__ = [
    "DEFAULT_HEARTBEAT_INTERVAL_SECONDS",
    "DEFAULT_REDIS_PREFIX",
    "DEFAULT_REDIS_TTL_SECONDS",
    "STREAM_MANAGER_TYPE_ENV",
    "MemoryStreamManager",
    "RedisStreamManager",
    "StreamManager",
    "StreamManagerConfig",
    "create_stream_manager",
]
