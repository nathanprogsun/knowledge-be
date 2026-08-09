"""Unit tests for the chat-layer event bus.

Covers the event vocabulary, ``Event`` payload handling, synchronous
subscribe/publish ordering, fire-and-forget asynchronous dispatch with
error isolation, concurrent emit-and-wait, and the process-wide
singleton helpers.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import FrozenInstanceError

import pytest

from src.core.chat.bus import (
    Event,
    EventBus,
    EventBusError,
    clear,
    emit,
    emit_and_wait,
    get_global_event_bus,
    get_handler_count,
    has_handlers,
    off,
    on,
    set_global_event_bus,
)
from src.core.chat.types import EventType


@pytest.fixture(autouse=True)
def _isolated_global_bus() -> Iterator[None]:
    """Give every test a fresh process-wide bus and restore it afterwards."""
    original = get_global_event_bus()
    set_global_event_bus(EventBus())
    yield
    set_global_event_bus(original)


# ── event vocabulary ────────────────────────────────────────────────────


def test_event_type_values_match_upstream_vocabulary() -> None:
    assert EventType.QUERY_RECEIVED == "query.received"
    assert EventType.QUERY_REWRITE == "query.rewrite"
    assert EventType.RETRIEVAL_START == "retrieval.start"
    assert EventType.RERANK_COMPLETE == "rerank.complete"
    assert EventType.MERGE_START == "merge.start"
    assert EventType.CHAT_START == "chat.start"
    assert EventType.CHAT_STREAM == "chat.stream"
    assert EventType.AGENT_PLAN == "agent.plan"
    assert EventType.AGENT_FINAL_ANSWER == "final_answer"
    assert EventType.TOOL_APPROVAL_REQUIRED == "tool_approval_required"
    assert EventType.MCP_OAUTH_REQUIRED == "mcp_oauth_required"
    assert EventType.ERROR == "error"
    assert EventType.SESSION_TITLE == "session_title"
    assert EventType.STOP == "stop"


def test_event_type_is_a_str_subclass() -> None:
    assert str(EventType.CHAT_COMPLETE) == "chat.complete"
    assert EventType("retrieval.vector") is EventType.RETRIEVAL_VECTOR


# ── event payload ───────────────────────────────────────────────────────


def test_event_with_id_generates_an_immutable_id() -> None:
    event = Event(type=EventType.CHAT_START, session_id="s-1")

    stamped = event.with_id()

    assert stamped.id
    assert stamped.session_id == "s-1"
    assert stamped is not event
    assert event.id == ""  # original untouched


def test_event_with_id_keeps_an_existing_id() -> None:
    event = Event(type=EventType.CHAT_START, id="evt-1")

    assert event.with_id() is event


def test_event_is_frozen() -> None:
    event = Event(type=EventType.ERROR)

    with pytest.raises(FrozenInstanceError):
        event.id = "forbidden"  # type: ignore[misc]


# ── synchronous subscribe / publish ─────────────────────────────────────


async def test_publish_delivers_event_to_subscribed_handler() -> None:
    bus = EventBus()
    received: list[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    bus.subscribe(EventType.CHAT_START, handler)

    await bus.publish(
        Event(
            type=EventType.CHAT_START,
            session_id="s-1",
            request_id="r-1",
            data={"query": "hello"},
        )
    )

    assert len(received) == 1
    assert received[0].type is EventType.CHAT_START
    assert received[0].session_id == "s-1"
    assert received[0].request_id == "r-1"
    assert received[0].data == {"query": "hello"}
    assert received[0].id  # auto-generated before delivery


async def test_publish_runs_multiple_handlers_in_registration_order() -> None:
    bus = EventBus()
    calls: list[str] = []

    async def first(event: Event) -> None:
        calls.append("first")

    async def second(event: Event) -> None:
        calls.append("second")

    bus.subscribe(EventType.CHAT_START, first)
    bus.subscribe(EventType.CHAT_START, second)

    await bus.publish(Event(type=EventType.CHAT_START))

    assert calls == ["first", "second"]


async def test_publish_without_handlers_is_a_noop() -> None:
    bus = EventBus()

    await bus.publish(Event(type=EventType.CHAT_START))

    assert bus.get_handler_count(EventType.CHAT_START) == 0


async def test_publish_generates_a_unique_id_per_event() -> None:
    bus = EventBus()
    ids: list[str] = []

    async def handler(event: Event) -> None:
        ids.append(event.id)

    bus.subscribe(EventType.RETRIEVAL_START, handler)

    await bus.publish(Event(type=EventType.RETRIEVAL_START))
    await bus.publish(Event(type=EventType.RETRIEVAL_START))

    assert len(ids) == 2
    assert ids[0] != ids[1]


async def test_sync_mode_propagates_handler_failure() -> None:
    bus = EventBus()

    async def broken(event: Event) -> None:
        raise RuntimeError("boom")

    bus.subscribe(EventType.CHAT_START, broken)

    with pytest.raises(EventBusError) as excinfo:
        await bus.publish(Event(type=EventType.CHAT_START))

    assert "chat.start" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


async def test_sync_mode_stops_at_the_first_failure() -> None:
    bus = EventBus()
    calls: list[str] = []

    async def broken(event: Event) -> None:
        calls.append("broken")
        raise RuntimeError("boom")

    async def after(event: Event) -> None:
        calls.append("after")

    bus.subscribe(EventType.CHAT_START, broken)
    bus.subscribe(EventType.CHAT_START, after)

    with pytest.raises(EventBusError):
        await bus.publish(Event(type=EventType.CHAT_START))

    assert calls == ["broken"]


# ── asynchronous (fire-and-forget) dispatch ─────────────────────────────


async def test_async_mode_schedules_every_handler() -> None:
    bus = EventBus(async_mode=True)
    done = asyncio.Event()
    calls: list[str] = []

    async def handler(event: Event) -> None:
        calls.append(event.type.value)
        done.set()

    bus.subscribe(EventType.RERANK_START, handler)

    await bus.publish(Event(type=EventType.RERANK_START))
    await asyncio.wait_for(done.wait(), timeout=2)
    await asyncio.sleep(0)  # let the scheduled task fully finish

    assert calls == ["rerank.start"]


async def test_async_mode_isolates_handler_errors(caplog: pytest.LogCaptureFixture) -> None:
    bus = EventBus(async_mode=True)
    done = asyncio.Event()
    calls: list[str] = []

    async def broken(event: Event) -> None:
        raise RuntimeError("boom")

    async def healthy(event: Event) -> None:
        calls.append("healthy")
        done.set()

    bus.subscribe(EventType.CHAT_COMPLETE, broken)
    bus.subscribe(EventType.CHAT_COMPLETE, healthy)

    # Fire-and-forget: a failing handler must not propagate or block others.
    await bus.publish(Event(type=EventType.CHAT_COMPLETE))
    await asyncio.wait_for(done.wait(), timeout=2)
    await asyncio.sleep(0)  # let the failing task fully finish

    assert calls == ["healthy"]
    assert "chat.complete" in caplog.text


# ── concurrent emit-and-wait ────────────────────────────────────────────


async def test_emit_and_wait_runs_handlers_concurrently() -> None:
    bus = EventBus()
    entered = asyncio.Event()
    release = asyncio.Event()
    calls: list[str] = []

    async def blocker(event: Event) -> None:
        calls.append("blocker")
        entered.set()
        await asyncio.wait_for(release.wait(), timeout=2)

    async def releaser(event: Event) -> None:
        calls.append("releaser")
        release.set()

    bus.subscribe(EventType.RERANK_START, blocker)
    bus.subscribe(EventType.RERANK_START, releaser)

    # blocker waits on an event only releaser can set — concurrent
    # execution is required for this to complete without timing out.
    await bus.emit_and_wait(Event(type=EventType.RERANK_START))

    assert calls == ["blocker", "releaser"]


async def test_emit_and_wait_reports_failure_after_all_handlers_run() -> None:
    bus = EventBus()
    calls: list[str] = []

    async def broken(event: Event) -> None:
        calls.append("broken")
        raise RuntimeError("boom")

    async def healthy(event: Event) -> None:
        calls.append("healthy")

    bus.subscribe(EventType.CHAT_COMPLETE, broken)
    bus.subscribe(EventType.CHAT_COMPLETE, healthy)

    with pytest.raises(EventBusError) as excinfo:
        await bus.emit_and_wait(Event(type=EventType.CHAT_COMPLETE))

    assert "chat.complete" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert sorted(calls) == ["broken", "healthy"]


async def test_emit_and_wait_works_in_async_mode() -> None:
    bus = EventBus(async_mode=True)
    calls: list[str] = []

    async def handler(event: Event) -> None:
        calls.append(event.type.value)

    bus.subscribe(EventType.MERGE_COMPLETE, handler)

    await bus.emit_and_wait(Event(type=EventType.MERGE_COMPLETE))

    assert calls == ["merge.complete"]


async def test_emit_and_wait_auto_generates_event_id() -> None:
    bus = EventBus()
    received: list[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    bus.subscribe(EventType.CHAT_START, handler)

    await bus.emit_and_wait(Event(type=EventType.CHAT_START))

    assert received[0].id


# ── registry introspection ──────────────────────────────────────────────


def test_handler_registry_queries() -> None:
    bus = EventBus()

    async def handler(event: Event) -> None:
        pass

    assert bus.has_handlers(EventType.ERROR) is False
    assert bus.get_handler_count(EventType.ERROR) == 0

    bus.on(EventType.ERROR, handler)
    bus.on(EventType.ERROR, handler)

    assert bus.has_handlers(EventType.ERROR) is True
    assert bus.get_handler_count(EventType.ERROR) == 2

    bus.off(EventType.ERROR)

    assert bus.has_handlers(EventType.ERROR) is False
    assert bus.get_handler_count(EventType.ERROR) == 0


async def test_unsubscribe_only_removes_the_targeted_event_type() -> None:
    bus = EventBus()
    calls: list[str] = []

    async def error_handler(event: Event) -> None:
        calls.append("error")

    async def stop_handler(event: Event) -> None:
        calls.append("stop")

    bus.subscribe(EventType.ERROR, error_handler)
    bus.subscribe(EventType.STOP, stop_handler)
    bus.unsubscribe(EventType.ERROR)

    await bus.publish(Event(type=EventType.ERROR))
    await bus.publish(Event(type=EventType.STOP))

    assert calls == ["stop"]


def test_clear_drops_every_handler() -> None:
    bus = EventBus()

    async def handler(event: Event) -> None:
        pass

    bus.subscribe(EventType.STOP, handler)
    bus.subscribe(EventType.ERROR, handler)

    bus.clear()

    assert bus.has_handlers(EventType.STOP) is False
    assert bus.has_handlers(EventType.ERROR) is False


# ── process-wide singleton ──────────────────────────────────────────────


def test_get_global_event_bus_is_a_singleton() -> None:
    assert get_global_event_bus() is get_global_event_bus()


def test_set_global_event_bus_swaps_the_instance() -> None:
    replacement = EventBus(async_mode=True)

    set_global_event_bus(replacement)

    assert get_global_event_bus() is replacement


async def test_module_level_helpers_route_to_the_global_bus() -> None:
    received: list[EventType] = []

    async def handler(event: Event) -> None:
        received.append(event.type)

    on(EventType.ERROR, handler)

    assert has_handlers(EventType.ERROR) is True
    assert get_handler_count(EventType.ERROR) == 1

    await emit(Event(type=EventType.ERROR))

    assert received == [EventType.ERROR]

    off(EventType.ERROR)

    assert has_handlers(EventType.ERROR) is False
    await emit(Event(type=EventType.ERROR))
    assert received == [EventType.ERROR]  # nothing delivered after off


async def test_module_level_emit_and_wait_delivers_to_the_global_bus() -> None:
    calls: list[str] = []

    async def handler(event: Event) -> None:
        calls.append(event.type.value)

    on(EventType.CHAT_START, handler)

    await emit_and_wait(Event(type=EventType.CHAT_START))

    assert calls == ["chat.start"]


async def test_module_level_clear_drops_all_handlers() -> None:
    async def handler(event: Event) -> None:
        pass

    on(EventType.STOP, handler)
    on(EventType.ERROR, handler)

    clear()

    assert has_handlers(EventType.STOP) is False
    assert has_handlers(EventType.ERROR) is False
