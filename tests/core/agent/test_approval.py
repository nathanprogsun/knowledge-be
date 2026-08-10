"""Unit tests for the agent approval gate.

Drives ``ApprovalGate`` with a scripted stub checker, a real ``EventBus``
whose handlers complete pending waits mid-emit, and short timeouts — no
database, no network, no Redis. Covers the pre-check (fail-close/fail-open),
the tool-approval and OAuth wait/resume cycles (approve, deny, timeout,
cancel), the resolution ownership checks, and the idempotent single-delivery
race.
"""

from __future__ import annotations

import asyncio

import pytest

from src.core.agents.engine.approval import (
    DEFAULT_APPROVAL_TIMEOUT_SECONDS,
    FAIL_OPEN_ENV,
    AlreadyResolvedError,
    ApprovalChecker,
    ApprovalGate,
    ApprovalGateError,
    PendingNotFoundError,
    TenantMismatchError,
    UserMismatchError,
)
from src.core.agents.tools.mcp_tool import ApprovalDecision, OAuthPendingRequest, PendingRequest
from src.core.chat.bus import Event, EventBus
from src.core.chat.types import EventType


class StubChecker:
    """Scripted checker: returns ``required`` or raises ``error``."""

    def __init__(self, *, required: bool = False) -> None:
        self.required = required
        self.error: Exception | None = None

    def is_required(self, tenant_id: int, service_id: str, tool_name: str) -> bool:
        if self.error is not None:
            raise self.error
        return self.required


class RecordingBus(EventBus):
    """Event bus that also records every published event."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)
        await super().emit(event)


def make_request(
    *,
    tenant_id: int = 1,
    user_id: str = "",
    session_id: str = "s1",
    assistant_message_id: str = "m1",
    request_id: str = "r1",
    service_id: str = "svc",
    service_name: str = "svcname",
    mcp_tool_name: str = "danger_tool",
    registered_tool_name: str = "mcp_svcname_danger_tool",
    description: str = "desc",
    args: str = '{"a": 1}',
    tool_call_id: str = "tc1",
    event_bus: EventBus | None = None,
) -> PendingRequest:
    return PendingRequest(
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        assistant_message_id=assistant_message_id,
        request_id=request_id,
        service_id=service_id,
        service_name=service_name,
        mcp_tool_name=mcp_tool_name,
        registered_tool_name=registered_tool_name,
        description=description,
        args=args,
        tool_call_id=tool_call_id,
        event_bus=event_bus,
    )


def make_oauth_request(
    *,
    tenant_id: int = 1,
    user_id: str = "alice",
    wait_timeout_seconds: int = 0,
    event_bus: EventBus | None = None,
) -> OAuthPendingRequest:
    return OAuthPendingRequest(
        tenant_id=tenant_id,
        user_id=user_id,
        session_id="s1",
        assistant_message_id="m1",
        request_id="r1",
        service_id="svc",
        service_name="svcname",
        mcp_tool_name="t",
        tool_call_id="tc1",
        wait_timeout_seconds=wait_timeout_seconds,
        event_bus=event_bus,
    )


# ── pre-check (needs_approval) ──────────────────────────────────────────


def test_needs_approval_without_checker_returns_false() -> None:
    gate = ApprovalGate()
    assert gate.needs_approval(tenant_id=1, service_id="svc", tool_name="t") is False


def test_needs_approval_reflects_checker_decision() -> None:
    assert (
        ApprovalGate(checker=StubChecker(required=True)).needs_approval(
            tenant_id=1, service_id="svc", tool_name="t"
        )
        is True
    )
    assert (
        ApprovalGate(checker=StubChecker(required=False)).needs_approval(
            tenant_id=1, service_id="svc", tool_name="t"
        )
        is False
    )


def test_needs_approval_skips_empty_identity() -> None:
    gate = ApprovalGate(checker=StubChecker(required=True))
    assert gate.needs_approval(tenant_id=0, service_id="svc", tool_name="t") is False
    assert gate.needs_approval(tenant_id=1, service_id="", tool_name="t") is False
    assert gate.needs_approval(tenant_id=1, service_id="svc", tool_name="") is False


def test_needs_approval_fails_closed_on_checker_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(FAIL_OPEN_ENV, raising=False)
    checker = StubChecker(required=False)
    checker.error = RuntimeError("db down")
    gate = ApprovalGate(checker=checker)
    assert gate.needs_approval(tenant_id=1, service_id="svc", tool_name="t") is True


def test_needs_approval_fails_open_when_configured() -> None:
    checker = StubChecker(required=False)
    checker.error = RuntimeError("db down")
    gate = ApprovalGate(checker=checker, fail_open=True)
    assert gate.needs_approval(tenant_id=1, service_id="svc", tool_name="t") is False


def test_needs_approval_fails_open_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(FAIL_OPEN_ENV, "true")
    checker = StubChecker(required=False)
    checker.error = RuntimeError("db down")
    gate = ApprovalGate(checker=checker)
    assert gate.needs_approval(tenant_id=1, service_id="svc", tool_name="t") is False


def test_gate_accepts_protocol_checker() -> None:
    def accepts(checker: ApprovalChecker) -> None:
        del checker

    accepts(StubChecker(required=True))


# ── request_and_wait ────────────────────────────────────────────────────


async def test_request_and_wait_without_checker_approves_immediately() -> None:
    gate = ApprovalGate()
    decision = await gate.request_and_wait(make_request(event_bus=None))
    assert decision.approved is True
    assert decision.timed_out is False


async def test_request_and_wait_requires_event_bus() -> None:
    gate = ApprovalGate(checker=StubChecker(required=True))
    with pytest.raises(ApprovalGateError):
        await gate.request_and_wait(make_request(event_bus=None))


async def test_request_and_wait_approves_with_modified_args() -> None:
    bus = EventBus()
    gate = ApprovalGate(checker=StubChecker(required=True), timeout_seconds=2)

    async def on_required(event: Event) -> None:
        data = event.data
        assert isinstance(data, dict)
        gate.resolve(1, "", str(data["pending_id"]), ApprovalDecision(
            approved=True, modified_args='{"a": 2}'
        ))

    bus.on(EventType.TOOL_APPROVAL_REQUIRED, on_required)
    decision = await gate.request_and_wait(make_request(event_bus=bus))
    assert decision.approved is True
    assert decision.modified_args == '{"a": 2}'


async def test_request_and_wait_denies() -> None:
    bus = EventBus()
    gate = ApprovalGate(checker=StubChecker(required=True), timeout_seconds=2)

    async def on_required(event: Event) -> None:
        data = event.data
        assert isinstance(data, dict)
        gate.resolve(1, "", str(data["pending_id"]), ApprovalDecision(
            approved=False, reason="no"
        ))

    bus.on(EventType.TOOL_APPROVAL_REQUIRED, on_required)
    decision = await gate.request_and_wait(make_request(event_bus=bus))
    assert decision.approved is False
    assert decision.reason == "no"


async def test_request_and_wait_times_out() -> None:
    bus = EventBus()
    gate = ApprovalGate(checker=StubChecker(required=True), timeout_seconds=0.05)
    decision = await gate.request_and_wait(make_request(event_bus=bus))
    assert decision.approved is False
    assert decision.timed_out is True
    assert decision.reason == "approval timeout"


async def test_request_and_wait_cancelled() -> None:
    bus = EventBus()
    captured: dict[str, str] = {}

    async def on_required(event: Event) -> None:
        data = event.data
        assert isinstance(data, dict)
        captured["pending_id"] = str(data["pending_id"])

    bus.on(EventType.TOOL_APPROVAL_REQUIRED, on_required)
    gate = ApprovalGate(checker=StubChecker(required=True), timeout_seconds=30)
    task = asyncio.create_task(gate.request_and_wait(make_request(event_bus=bus)))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # The cancelled wait must not leave a dangling pending entry.
    with pytest.raises(PendingNotFoundError):
        gate.resolve(1, "", captured["pending_id"], ApprovalDecision(approved=True))


async def test_request_and_wait_emits_required_and_resolved_events() -> None:
    bus = RecordingBus()
    gate = ApprovalGate(checker=StubChecker(required=True), timeout_seconds=2)

    async def on_required(event: Event) -> None:
        data = event.data
        assert isinstance(data, dict)
        gate.resolve(1, "", str(data["pending_id"]), ApprovalDecision(
            approved=True, reason="ok"
        ))

    bus.on(EventType.TOOL_APPROVAL_REQUIRED, on_required)
    await gate.request_and_wait(make_request(args='{"a": 1}', event_bus=bus))

    types = [evt.type for evt in bus.events]
    assert EventType.TOOL_APPROVAL_REQUIRED in types
    assert EventType.TOOL_APPROVAL_RESOLVED in types

    required = next(evt for evt in bus.events if evt.type == EventType.TOOL_APPROVAL_REQUIRED)
    required_data = required.data
    assert isinstance(required_data, dict)
    assert required_data["service_id"] == "svc"
    assert required_data["mcp_tool_name"] == "danger_tool"
    assert required_data["registered_tool_name"] == "mcp_svcname_danger_tool"
    assert required_data["timeout_seconds"] == 2
    assert required_data["args"] == {"a": 1}
    assert required_data["args_json"] == '{"a": 1}'

    resolved = next(evt for evt in bus.events if evt.type == EventType.TOOL_APPROVAL_RESOLVED)
    resolved_data = resolved.data
    assert isinstance(resolved_data, dict)
    assert resolved_data["approved"] is True
    assert resolved_data["reason"] == "ok"
    assert resolved_data["timed_out"] is False
    assert resolved_data["canceled"] is False


# ── OAuth authorization wait ────────────────────────────────────────────


async def test_oauth_wait_authorizes() -> None:
    bus = EventBus()
    gate = ApprovalGate(timeout_seconds=2)

    async def on_required(event: Event) -> None:
        data = event.data
        assert isinstance(data, dict)
        gate.resolve(1, "alice", str(data["pending_id"]), ApprovalDecision(approved=True))

    bus.on(EventType.MCP_OAUTH_REQUIRED, on_required)
    decision = await gate.request_oauth_and_wait(make_oauth_request(event_bus=bus))
    assert decision.approved is True


async def test_oauth_wait_times_out() -> None:
    bus = EventBus()
    gate = ApprovalGate(timeout_seconds=0.05)
    decision = await gate.request_oauth_and_wait(make_oauth_request(event_bus=bus))
    assert decision.approved is False
    assert decision.timed_out is True


async def test_oauth_wait_requires_event_bus() -> None:
    gate = ApprovalGate()
    with pytest.raises(ApprovalGateError):
        await gate.request_oauth_and_wait(make_oauth_request(event_bus=None))


# ── resolution ownership ────────────────────────────────────────────────


def test_resolve_pending_not_found() -> None:
    gate = ApprovalGate(checker=StubChecker(required=True))
    with pytest.raises(PendingNotFoundError):
        gate.resolve(1, "", "no-such-id", ApprovalDecision(approved=True))


async def test_resolve_tenant_mismatch() -> None:
    bus = EventBus()
    gate = ApprovalGate(checker=StubChecker(required=True), timeout_seconds=2)

    async def on_required(event: Event) -> None:
        data = event.data
        assert isinstance(data, dict)
        pending_id = str(data["pending_id"])
        with pytest.raises(TenantMismatchError):
            gate.resolve(999, "", pending_id, ApprovalDecision(approved=True))
        gate.resolve(1, "", pending_id, ApprovalDecision(approved=False, reason="no"))

    bus.on(EventType.TOOL_APPROVAL_REQUIRED, on_required)
    decision = await gate.request_and_wait(make_request(event_bus=bus))
    assert decision.approved is False


async def test_resolve_user_mismatch() -> None:
    bus = EventBus()
    gate = ApprovalGate(checker=StubChecker(required=True), timeout_seconds=2)

    async def on_required(event: Event) -> None:
        data = event.data
        assert isinstance(data, dict)
        pending_id = str(data["pending_id"])
        with pytest.raises(UserMismatchError):
            gate.resolve(1, "bob", pending_id, ApprovalDecision(approved=True))
        gate.resolve(1, "alice", pending_id, ApprovalDecision(approved=True))

    bus.on(EventType.TOOL_APPROVAL_REQUIRED, on_required)
    decision = await gate.request_and_wait(make_request(user_id="alice", event_bus=bus))
    assert decision.approved is True


async def test_resolve_empty_user_rejected_when_waiter_has_user() -> None:
    bus = EventBus()
    gate = ApprovalGate(checker=StubChecker(required=True), timeout_seconds=2)

    async def on_required(event: Event) -> None:
        data = event.data
        assert isinstance(data, dict)
        pending_id = str(data["pending_id"])
        with pytest.raises(UserMismatchError):
            gate.resolve(1, "", pending_id, ApprovalDecision(approved=True))
        gate.resolve(1, "alice", pending_id, ApprovalDecision(approved=False, reason="no"))

    bus.on(EventType.TOOL_APPROVAL_REQUIRED, on_required)
    decision = await gate.request_and_wait(make_request(user_id="alice", event_bus=bus))
    assert decision.approved is False


async def test_resolve_after_timeout_returns_not_found() -> None:
    bus = EventBus()
    gate = ApprovalGate(checker=StubChecker(required=True), timeout_seconds=0.02)
    captured: dict[str, str] = {}

    async def on_required(event: Event) -> None:
        data = event.data
        assert isinstance(data, dict)
        captured["pending_id"] = str(data["pending_id"])

    bus.on(EventType.TOOL_APPROVAL_REQUIRED, on_required)
    decision = await gate.request_and_wait(make_request(event_bus=bus))
    assert decision.timed_out is True
    with pytest.raises(PendingNotFoundError):
        gate.resolve(1, "", captured["pending_id"], ApprovalDecision(approved=True))


async def test_resolve_race_second_call_already_resolved() -> None:
    bus = EventBus()
    gate = ApprovalGate(checker=StubChecker(required=True), timeout_seconds=30)

    async def on_required(event: Event) -> None:
        data = event.data
        assert isinstance(data, dict)
        pending_id = str(data["pending_id"])
        gate.resolve(1, "", pending_id, ApprovalDecision(approved=True))
        with pytest.raises(AlreadyResolvedError):
            gate.resolve(1, "", pending_id, ApprovalDecision(approved=False))

    bus.on(EventType.TOOL_APPROVAL_REQUIRED, on_required)
    decision = await gate.request_and_wait(make_request(event_bus=bus))
    assert decision.approved is True


# ── defaults ────────────────────────────────────────────────────────────


def test_default_timeout_constant_is_positive() -> None:
    assert DEFAULT_APPROVAL_TIMEOUT_SECONDS > 0


def test_constructor_ignores_non_positive_timeout() -> None:
    gate = ApprovalGate(timeout_seconds=0)
    assert gate._timeout == DEFAULT_APPROVAL_TIMEOUT_SECONDS
