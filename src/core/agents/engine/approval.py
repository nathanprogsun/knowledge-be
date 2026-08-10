"""Agent approval gate: pause high-risk tool operations for manual approval.

The gate decides which tool calls need a human sign-off through a pluggable
checker, then pauses execution with a UI event until an operator either
approves (optionally with modified arguments), denies, or the wait times out
or the caller is cancelled. A parallel flow pauses for in-conversation OAuth
authorization of a connected service.

Pending waiters live in-memory on the instance that started the wait;
``resolve`` validates tenant (and, when registered, user) ownership before
delivering the decision. All methods must be called from the single event
loop that started the wait — wait and resolution always share one process.

The resolution error vocabulary mirrors the upstream gate contract:
``PendingNotFoundError``, ``TenantMismatchError``, ``UserMismatchError`` and
``AlreadyResolvedError`` map to the HTTP status codes surfaced by the
approval endpoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Protocol
from uuid import uuid4

from src.common.exception import ApplicationError
from src.common.json import JsonObject
from src.core.agents.tools.mcp_tool import (
    ApprovalDecision,
    OAuthPendingRequest,
    PendingRequest,
)
from src.core.chat.bus import Event, EventBus
from src.core.chat.types import EventType

logger = logging.getLogger(__name__)

#: Default wait window (seconds) for one approval round-trip.
DEFAULT_APPROVAL_TIMEOUT_SECONDS: float = 600.0

#: Environment variable that opts into fail-open: a checker error skips the
#: gate instead of requiring approval. Unset defaults to fail-close.
FAIL_OPEN_ENV = "WEKNORA_AGENT_TOOL_APPROVAL_FAIL_OPEN"

#: Reason text attached to a timed-out wait.
TIMEOUT_REASON = "approval timeout"
#: Reason text attached to a cancelled wait.
CANCELED_REASON = "request canceled"


class ApprovalError(ApplicationError):
    """Base error for approval-gate failures."""

    code = "approval.failed"
    message = "Tool approval failed"


class ApprovalGateError(ApprovalError):
    """Raised when the gate cannot start a wait (missing bus, emit failure)."""

    code = "approval.gate_failed"
    message = "Tool approval gate failed"


class PendingNotFoundError(ApprovalError):
    """Raised when resolving a pending id the gate does not know."""

    code = "approval.pending_not_found"
    message = "Tool approval pending not found"


class TenantMismatchError(ApprovalError):
    """Raised when the resolving tenant differs from the wait's tenant."""

    code = "approval.tenant_mismatch"
    message = "Workspace mismatch for tool approval"


class UserMismatchError(ApprovalError):
    """Raised when the resolving user does not own the originating session."""

    code = "approval.user_mismatch"
    message = "User mismatch for tool approval"


class AlreadyResolvedError(ApprovalError):
    """Raised when a previous resolve (or timeout/cancel) won the race."""

    code = "approval.already_resolved"
    message = "Tool approval already resolved"


class ApprovalChecker(Protocol):
    """Answers whether a concrete tool requires human approval."""

    def is_required(self, tenant_id: int, service_id: str, tool_name: str) -> bool: ...


class _Waiter:
    """One blocked approval wait: a single-slot channel plus a delivery flag."""

    __slots__ = ("_delivered", "ch", "tenant_id", "user_id")

    def __init__(self, tenant_id: int, user_id: str) -> None:
        self.ch: asyncio.Queue[ApprovalDecision] = asyncio.Queue[ApprovalDecision](
            maxsize=1
        )
        self.tenant_id = tenant_id
        # Empty user_id means "skip the user check" on resolve.
        self.user_id = user_id
        self._delivered = False

    def deliver(self, decision: ApprovalDecision) -> bool:
        """Return True when this call won the race and delivered the decision."""
        if self._delivered:
            return False
        try:
            self.ch.put_nowait(decision)
        except asyncio.QueueFull:
            return False
        self._delivered = True
        return True


class ApprovalGate:
    """Coordinates wait/resolve for tool-approval and OAuth prompts.

    ``needs_approval`` consults the checker (fail-close on checker errors by
    default). ``request_and_wait`` emits a UI event, then blocks until an
    operator resolves the pending id, the timeout elapses, or the caller is
    cancelled. ``request_oauth_and_wait`` is the authorization variant and
    never consults the checker. ``resolve`` completes a pending wait after
    validating tenant and (when registered) user ownership.
    """

    def __init__(
        self,
        *,
        checker: ApprovalChecker | None = None,
        timeout_seconds: float = DEFAULT_APPROVAL_TIMEOUT_SECONDS,
        fail_open: bool | None = None,
    ) -> None:
        self._checker = checker
        self._timeout = (
            timeout_seconds if timeout_seconds > 0 else DEFAULT_APPROVAL_TIMEOUT_SECONDS
        )
        if fail_open is None:
            fail_open = os.environ.get(FAIL_OPEN_ENV, "").strip().lower() == "true"
        self._fail_close = not fail_open
        self._pending: dict[str, _Waiter] = {}

    # ── pre-check ───────────────────────────────────────────────────

    def needs_approval(self, *, tenant_id: int, service_id: str, tool_name: str) -> bool:
        """Return whether execution should pause for human confirmation."""
        if self._checker is None or tenant_id == 0 or service_id == "" or tool_name == "":
            return False
        try:
            return self._checker.is_required(tenant_id, service_id, tool_name)
        except Exception as exc:
            # Fail-close by default: a transient checker failure must not
            # silently let a dangerous tool run. Fail-open is opt-in.
            if self._fail_close:
                logger.warning(
                    "tool approval check failed (fail-close: requiring approval): %s",
                    exc,
                )
                return True
            logger.warning("tool approval check failed (fail-open: skip gate): %s", exc)
            return False

    # ── tool approval wait ─────────────────────────────────────────

    async def request_and_wait(self, request: PendingRequest) -> ApprovalDecision:
        """Emit a UI prompt, then block until resolve, timeout, or cancel."""
        if self._checker is None:
            return ApprovalDecision(approved=True)
        if request.event_bus is None:
            raise ApprovalGateError("tool approval: EventBus is nil")
        event_bus = request.event_bus

        pending_id = str(uuid4())
        waiter = _Waiter(request.tenant_id, request.user_id)
        self._pending[pending_id] = waiter
        try:
            try:
                await self._emit_tool_approval_required(event_bus, request, pending_id)
            except Exception as exc:
                raise ApprovalGateError(f"emit tool approval required: {exc}") from exc
            decision = await self._wait_for_decision(waiter, self._timeout)
        finally:
            self._pending.pop(pending_id, None)
        await self._emit_tool_approval_resolved(event_bus, request, pending_id, decision)
        return decision

    # ── OAuth authorization wait ───────────────────────────────────

    async def request_oauth_and_wait(self, request: OAuthPendingRequest) -> ApprovalDecision:
        """Pause for in-conversation authorization; no checker consultation."""
        if request.event_bus is None:
            raise ApprovalGateError("oauth gate: EventBus is nil")
        event_bus = request.event_bus

        pending_id = str(uuid4())
        waiter = _Waiter(request.tenant_id, request.user_id)
        self._pending[pending_id] = waiter
        timeout = (
            request.wait_timeout_seconds if request.wait_timeout_seconds > 0 else self._timeout
        )
        try:
            try:
                await self._emit_oauth_required(event_bus, request, pending_id, timeout)
            except Exception as exc:
                raise ApprovalGateError(f"emit mcp oauth required: {exc}") from exc
            decision = await self._wait_for_decision(waiter, timeout)
        finally:
            self._pending.pop(pending_id, None)
        await self._emit_oauth_resolved(event_bus, request, pending_id, decision)
        return decision

    # ── resolution ─────────────────────────────────────────────────

    def resolve(
        self,
        tenant_id: int,
        user_id: str,
        pending_id: str,
        decision: ApprovalDecision,
    ) -> None:
        """Complete a pending approval after validating ownership."""
        waiter = self._pending.get(pending_id)
        if waiter is None:
            raise PendingNotFoundError()
        if waiter.tenant_id != tenant_id:
            raise TenantMismatchError()
        # A registered user check is fail-close: a missing caller user_id can
        # never skip it (no auth middleware can bypass the per-user check).
        if waiter.user_id != "" and waiter.user_id != user_id:
            raise UserMismatchError()
        if not waiter.deliver(decision):
            raise AlreadyResolvedError()

    # ── internals ──────────────────────────────────────────────────

    async def _wait_for_decision(
        self, waiter: _Waiter, timeout: float
    ) -> ApprovalDecision:
        """Block until a decision arrives, the window elapses, or cancel."""
        try:
            return await asyncio.wait_for(waiter.ch.get(), timeout=timeout)
        except TimeoutError:
            # Deliver the timeout decision and re-read it so the returned value
            # reflects whichever decision actually won the race.
            timeout_decision = ApprovalDecision(
                approved=False, reason=TIMEOUT_REASON, timed_out=True
            )
            waiter.deliver(timeout_decision)
            return await waiter.ch.get()
        except asyncio.CancelledError:
            # Best-effort: mark the wait as cancelled so a concurrent resolve
            # cannot deliver after the caller is gone.
            waiter.deliver(
                ApprovalDecision(
                    approved=False, reason=CANCELED_REASON, context_canceled=True
                )
            )
            raise

    async def _emit_tool_approval_required(
        self, event_bus: EventBus, request: PendingRequest, pending_id: str
    ) -> None:
        args_obj: JsonObject | None = None
        if request.args:
            try:
                parsed = json.loads(request.args)
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, dict):
                args_obj = parsed
        data: JsonObject = {
            "pending_id": pending_id,
            "tenant_id": request.tenant_id,
            "session_id": request.session_id,
            "assistant_message_id": request.assistant_message_id,
            "service_id": request.service_id,
            "service_name": request.service_name,
            "mcp_tool_name": request.mcp_tool_name,
            "registered_tool_name": request.registered_tool_name,
            "description": request.description,
            "args_json": request.args,
            "timeout_seconds": max(1, int(self._timeout)),
            "requested_at": int(time.time()),
            "tool_call_id": request.tool_call_id,
            "request_id": request.request_id,
        }
        if args_obj is not None:
            data["args"] = args_obj
        await event_bus.emit(
            Event(
                id=f"{pending_id}-approval-required",
                type=EventType.TOOL_APPROVAL_REQUIRED,
                session_id=request.session_id or None,
                data=data,
                metadata={
                    "assistant_message_id": request.assistant_message_id,
                    "pending_id": pending_id,
                },
                request_id=request.request_id or None,
            )
        )

    async def _emit_tool_approval_resolved(
        self,
        event_bus: EventBus,
        request: PendingRequest,
        pending_id: str,
        decision: ApprovalDecision,
    ) -> None:
        data: JsonObject = {
            "pending_id": pending_id,
            "approved": decision.approved,
            "reason": decision.reason,
            "timed_out": decision.timed_out,
            "canceled": decision.context_canceled,
        }
        try:
            await event_bus.emit(
                Event(
                    id=f"{pending_id}-approval-resolved",
                    type=EventType.TOOL_APPROVAL_RESOLVED,
                    session_id=request.session_id or None,
                    data=data,
                    metadata={"assistant_message_id": request.assistant_message_id},
                    request_id=request.request_id or None,
                )
            )
        except Exception as exc:
            # The decision is already final; a failed notice must not fail the
            # caller. Log and continue (best-effort, mirrors the upstream emit).
            logger.warning("tool approval resolved event failed: %s", exc)

    async def _emit_oauth_required(
        self,
        event_bus: EventBus,
        request: OAuthPendingRequest,
        pending_id: str,
        timeout: float,
    ) -> None:
        data: JsonObject = {
            "pending_id": pending_id,
            "tenant_id": request.tenant_id,
            "session_id": request.session_id,
            "assistant_message_id": request.assistant_message_id,
            "service_id": request.service_id,
            "service_name": request.service_name,
            "mcp_tool_name": request.mcp_tool_name,
            "timeout_seconds": max(1, int(timeout)),
            "requested_at": int(time.time()),
            "tool_call_id": request.tool_call_id,
            "request_id": request.request_id,
        }
        await event_bus.emit(
            Event(
                id=f"{pending_id}-mcp-oauth-required",
                type=EventType.MCP_OAUTH_REQUIRED,
                session_id=request.session_id or None,
                data=data,
                metadata={
                    "assistant_message_id": request.assistant_message_id,
                    "pending_id": pending_id,
                },
                request_id=request.request_id or None,
            )
        )

    async def _emit_oauth_resolved(
        self,
        event_bus: EventBus,
        request: OAuthPendingRequest,
        pending_id: str,
        decision: ApprovalDecision,
    ) -> None:
        data: JsonObject = {
            "pending_id": pending_id,
            "service_id": request.service_id,
            "authorized": decision.approved,
            "reason": decision.reason,
            "timed_out": decision.timed_out,
            "canceled": decision.context_canceled,
        }
        try:
            await event_bus.emit(
                Event(
                    id=f"{pending_id}-mcp-oauth-resolved",
                    type=EventType.MCP_OAUTH_RESOLVED,
                    session_id=request.session_id or None,
                    data=data,
                    metadata={"assistant_message_id": request.assistant_message_id},
                    request_id=request.request_id or None,
                )
            )
        except Exception as exc:
            logger.warning("mcp oauth resolved event failed: %s", exc)


__all__ = [
    "CANCELED_REASON",
    "DEFAULT_APPROVAL_TIMEOUT_SECONDS",
    "FAIL_OPEN_ENV",
    "TIMEOUT_REASON",
    "AlreadyResolvedError",
    "ApprovalChecker",
    "ApprovalDecision",
    "ApprovalError",
    "ApprovalGate",
    "ApprovalGateError",
    "OAuthPendingRequest",
    "PendingNotFoundError",
    "PendingRequest",
    "TenantMismatchError",
    "UserMismatchError",
]
