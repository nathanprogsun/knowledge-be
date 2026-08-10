"""Unit tests for the chat stream manager.

Covers the ``Event`` / ``StreamKey`` / ``StreamData`` value types, the
in-memory backend (append, incremental reads with offset/limit, stream
creation), per-stream cancellation, the heartbeat task, the Redis backend
against a fake client, and the factory backend selection.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError as PydanticValidationError

from src.core.chat.stream.manager import (
    DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    DEFAULT_REDIS_PREFIX,
    DEFAULT_REDIS_TTL_SECONDS,
    STREAM_MANAGER_TYPE_ENV,
    MemoryStreamManager,
    RedisStreamManager,
    StreamManagerConfig,
    create_stream_manager,
)
from src.core.chat.stream.types import Event, StreamData, StreamKey


def _event(
    *,
    event_id: str = "e1",
    event_type: str = "answer",
    content: str = "hi",
) -> Event:
    return Event(id=event_id, type=event_type, content=content)


# ── value types ────────────────────────────────────────────────────────


def test_event_defaults_are_frozen_and_timestamped() -> None:
    event = Event(id="e1", type="thinking")
    assert event.content == ""
    assert event.done is False
    assert event.data is None
    assert event.timestamp is not None
    with pytest.raises(PydanticValidationError):
        event.content = "mutated"  # type: ignore[misc]


def test_event_round_trips_through_json() -> None:
    event = _event(event_id="e1", event_type="tool_call", content="ls")
    restored = Event.model_validate_json(event.model_dump_json())
    assert restored == event
    assert restored.timestamp == event.timestamp


def test_stream_key_identifies_session_and_message() -> None:
    key = StreamKey(session_id="s1", message_id="m1")
    assert key.session_id == "s1"
    assert key.message_id == "m1"
    assert StreamKey(session_id="s1", message_id="m1") == key
    assert hash(StreamKey(session_id="s1", message_id="m1")) == hash(key)


def test_stream_data_is_frozen() -> None:
    data = StreamData()
    assert data.events == ()
    assert data.last_updated is not None
    with pytest.raises(PydanticValidationError):
        data.events = (Event(id="x", type="ping"),)  # type: ignore[misc]


# ── memory backend ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_append_and_read_events() -> None:
    manager = MemoryStreamManager()
    await manager.append_event("s1", "m1", _event(event_id="e1", content="a"))
    await manager.append_event("s1", "m1", _event(event_id="e2", content="b"))

    events, next_offset = await manager.get_events("s1", "m1", 0)

    assert [e.id for e in events] == ["e1", "e2"]
    assert [e.content for e in events] == ["a", "b"]
    assert next_offset == 2


@pytest.mark.asyncio
async def test_get_events_from_offset() -> None:
    manager = MemoryStreamManager()
    for i in range(3):
        await manager.append_event("s1", "m1", _event(event_id=f"e{i}"))

    events, next_offset = await manager.get_events("s1", "m1", 1)

    assert [e.id for e in events] == ["e1", "e2"]
    assert next_offset == 3


@pytest.mark.asyncio
async def test_get_events_with_limit() -> None:
    manager = MemoryStreamManager()
    for i in range(5):
        await manager.append_event("s1", "m1", _event(event_id=f"e{i}"))

    events, next_offset = await manager.get_events("s1", "m1", 1, limit=2)

    assert [e.id for e in events] == ["e1", "e2"]
    assert next_offset == 3


@pytest.mark.asyncio
async def test_get_events_unknown_stream_returns_empty() -> None:
    manager = MemoryStreamManager()

    events, next_offset = await manager.get_events("s1", "m1", 0)

    assert events == []
    assert next_offset == 0


@pytest.mark.asyncio
async def test_get_events_beyond_end_returns_empty() -> None:
    manager = MemoryStreamManager()
    await manager.append_event("s1", "m1", _event())

    events, next_offset = await manager.get_events("s1", "m1", 5)

    assert events == []
    assert next_offset == 5


@pytest.mark.asyncio
async def test_get_or_create_stream_creates_and_reuses() -> None:
    manager = MemoryStreamManager()

    first = await manager.get_or_create_stream("s1", "m1")
    assert first.events == ()

    await manager.append_event("s1", "m1", _event())

    second = await manager.get_or_create_stream("s1", "m1")
    assert len(second.events) == 1


@pytest.mark.asyncio
async def test_streams_are_isolated_by_session_and_message() -> None:
    manager = MemoryStreamManager()
    await manager.append_event("s1", "m1", _event(event_id="a"))
    await manager.append_event("s1", "m2", _event(event_id="b"))
    await manager.append_event("s2", "m1", _event(event_id="c"))

    events, _ = await manager.get_events("s1", "m1", 0)
    assert [e.id for e in events] == ["a"]


# ── cancellation ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_marks_stream_cancelled() -> None:
    manager = MemoryStreamManager()
    assert manager.is_cancelled("s1", "m1") is False

    manager.cancel("s1", "m1")

    assert manager.is_cancelled("s1", "m1") is True


@pytest.mark.asyncio
async def test_cancel_is_scoped_to_one_stream() -> None:
    manager = MemoryStreamManager()
    manager.cancel("s1", "m1")

    assert manager.is_cancelled("s1", "m1") is True
    assert manager.is_cancelled("s1", "m2") is False


@pytest.mark.asyncio
async def test_wait_cancelled_returns_when_cancelled() -> None:
    manager = MemoryStreamManager()
    waiter = asyncio.create_task(manager.wait_cancelled("s1", "m1"))
    await asyncio.sleep(0)
    assert not waiter.done()

    manager.cancel("s1", "m1")

    await asyncio.wait_for(waiter, timeout=1)


# ── heartbeat ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_heartbeat_appends_ping_events() -> None:
    manager = MemoryStreamManager(heartbeat_interval_seconds=0.01)
    task = manager.start_heartbeat("s1", "m1")
    try:
        await asyncio.sleep(0.05)
        events, _ = await manager.get_events("s1", "m1", 0)
        assert any(e.type == "ping" for e in events)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_heartbeat_stops_after_cancel() -> None:
    manager = MemoryStreamManager(heartbeat_interval_seconds=0.01)
    task = manager.start_heartbeat("s1", "m1")

    manager.cancel("s1", "m1")
    await asyncio.sleep(0.05)

    assert task.done()


def test_heartbeat_interval_must_be_positive() -> None:
    with pytest.raises(ValueError):
        MemoryStreamManager(heartbeat_interval_seconds=0)


# ── redis backend ──────────────────────────────────────────────────────


class _FakeRedis:
    """Minimal in-memory stand-in for ``redis.asyncio.Redis``."""

    def __init__(self) -> None:
        self._lists: dict[str, list[str]] = {}
        self._ttls: dict[str, int] = {}
        self.closed = False

    async def rpush(self, key: str, value: str) -> int:
        self._lists.setdefault(key, []).append(value)
        return len(self._lists[key])

    async def expire(self, key: str, seconds: int) -> bool:
        self._ttls[key] = seconds
        return True

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        values = self._lists.get(key, [])
        if start < 0:
            start = max(len(values) + start, 0)
        if end < 0:
            end = len(values) + end
        return values[start : end + 1]

    async def aclose(self) -> None:
        self.closed = True


def _redis_manager(fake: _FakeRedis) -> RedisStreamManager:
    return RedisStreamManager(redis=fake)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_redis_append_and_read_events() -> None:
    fake = _FakeRedis()
    manager = _redis_manager(fake)
    await manager.append_event("s1", "m1", _event(event_id="e1", content="a"))
    await manager.append_event("s1", "m1", _event(event_id="e2", content="b"))

    events, next_offset = await manager.get_events("s1", "m1", 0)

    assert [e.id for e in events] == ["e1", "e2"]
    assert [e.content for e in events] == ["a", "b"]
    assert next_offset == 2
    assert fake._ttls[f"{DEFAULT_REDIS_PREFIX}:s1:m1"] == DEFAULT_REDIS_TTL_SECONDS


@pytest.mark.asyncio
async def test_redis_get_events_from_offset_with_limit() -> None:
    fake = _FakeRedis()
    manager = _redis_manager(fake)
    for i in range(5):
        await manager.append_event("s1", "m1", _event(event_id=f"e{i}"))

    events, next_offset = await manager.get_events("s1", "m1", 1, limit=2)

    assert [e.id for e in events] == ["e1", "e2"]
    assert next_offset == 3


@pytest.mark.asyncio
async def test_redis_get_events_unknown_stream_returns_empty() -> None:
    manager = _redis_manager(_FakeRedis())

    events, next_offset = await manager.get_events("s1", "m1", 0)

    assert events == []
    assert next_offset == 0


@pytest.mark.asyncio
async def test_redis_get_or_create_stream_returns_snapshot() -> None:
    fake = _FakeRedis()
    manager = _redis_manager(fake)
    await manager.append_event("s1", "m1", _event(event_id="e1"))

    data = await manager.get_or_create_stream("s1", "m1")

    assert [e.id for e in data.events] == ["e1"]


@pytest.mark.asyncio
async def test_redis_close_releases_client() -> None:
    fake = _FakeRedis()
    manager = _redis_manager(fake)

    await manager.close()

    assert fake.closed is True


# ── factory ───────────────────────────────────────────────────────────


def test_factory_defaults_to_memory() -> None:
    manager = create_stream_manager(StreamManagerConfig(backend="memory"))
    assert isinstance(manager, MemoryStreamManager)


@pytest.mark.asyncio
async def test_factory_selects_redis_backend() -> None:
    manager = create_stream_manager(
        StreamManagerConfig(backend="redis", redis_url="redis://localhost:6379")
    )
    assert isinstance(manager, RedisStreamManager)
    await manager.close()


def test_config_from_settings_uses_env_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(STREAM_MANAGER_TYPE_ENV, "redis")

    config = StreamManagerConfig.from_settings()

    assert config.backend == "redis"
    assert config.heartbeat_interval_seconds == DEFAULT_HEARTBEAT_INTERVAL_SECONDS


def test_config_from_settings_falls_back_to_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(STREAM_MANAGER_TYPE_ENV, "bogus")

    config = StreamManagerConfig.from_settings()

    assert config.backend == "memory"
