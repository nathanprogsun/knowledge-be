"""Connection pool for MCP transport clients.

Mirrors ``internal/mcp/manager.go``. The manager owns one
:class:`MCPSession` per remote service, reuses the live session while
it is connected, runs a periodic sweep that drops sessions whose
underlying transport is no longer connected, and exposes the
high-level JSON-RPC operations (``initialize``, ``tools/list``,
``resources/list``, ``tools/call``, ``resources/read``).

SSE transport only. The HTTP-streamable client plugs in through
:data:`_default_transport_factory` once its module ships; until then
the factory rejects ``"http-streamable"`` with a clear error so
callers do not silently fall back.

Wire-level errors
-----------------

The Go code distinguishes:

- transport-level errors with a known message (``Invalid session ID``,
  ``No active connection``) — these force a disconnect so the next
  caller rebuilds the session;
- OAuth-required errors — wrapped as :class:`OAuthRequiredError` so the
  service layer can show the user a "switch auth strategy" hint.

The Python surface raises the same exceptions and follows the same
heuristic for session invalidation (see
:func:`src.ai.mcp_transport.jsonrpc.is_session_invalid_error`).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from src.ai.mcp_transport.errors import (
    MCPError,
    MCPTransportError,
    SessionNotConnectedError,
)
from src.ai.mcp_transport.http_streamable_client import HTTPStreamableClient
from src.ai.mcp_transport.jsonrpc import (
    METHOD_INITIALIZE,
    METHOD_RESOURCES_LIST,
    METHOD_RESOURCES_READ,
    METHOD_TOOLS_CALL,
    METHOD_TOOLS_LIST,
    JSONRPCResponse,
)
from src.ai.mcp_transport.sse_client import SSEClient

# Strings copied verbatim from the Go layer. A match on the lower-cased
# error text is the operational signal that the server-side session is
# gone and the client must drop its ``Mcp-Session-Id``.
SESSION_INVALID_HINTS: frozenset[str] = frozenset(
    {"invalid session id", "no active connection"},
)


@dataclass
class MCPSession:
    """One live connection to a remote MCP server.

    ``client`` is the underlying transport; ``session_id`` is the
    server-assigned ``Mcp-Session-Id`` (empty until ``initialize``).
    ``connected`` and ``initialized`` mirror the transport- and
    handshake-level liveness flags.
    """

    service_id: str
    transport_type: str
    client: Any  # ``SSEClient`` (``HTTPStreamableClient`` once implemented)
    session_id: str = ""
    connected: bool = False
    initialized: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def is_alive(self) -> bool:
        """Return whether the transport reports itself as connected."""
        return self.connected and bool(self.client.is_connected())

    async def close(self) -> None:
        """Disconnect and mark the session for re-creation."""
        await self.client.disconnect()
        self.connected = False
        self.initialized = False


class _TransportFactory(Protocol):
    """Factory protocol for transport clients.

    The manager asks the factory for a fresh client when the cache
    misses. ``url`` is the service's configured URL; ``headers`` is the
    per-request header map (auth + custom). The return type is the
    union of the SSE / HTTP-streamable client classes — typed as
    ``Any`` here so the protocol does not force a static dependency on
    the transport module.
    """

    def __call__(
        self,
        *,
        service_id: str,
        transport_type: str,
        url: str,
        headers: dict[str, str] | None,
        timeout_seconds: float,
    ) -> Any: ...


def _default_transport_factory(
    *,
    service_id: str,
    transport_type: str,
    url: str,
    headers: dict[str, str] | None,
    timeout_seconds: float,
) -> Any:
    """Build a transport client by inspecting ``transport_type``.

    Mirrors the Go switch in ``NewMCPClient`` — ``sse`` gets an
    :class:`src.ai.mcp_transport.sse_client.SSEClient`,
    ``http-streamable`` (PR-17.5b) gets an
    :class:`src.ai.mcp_transport.http_streamable_client.HTTPStreamableClient`;
    ``stdio`` is intentionally rejected (disabled for security, same as
    the Go side).
    """
    if transport_type == "sse":
        return SSEClient(
            url=url,
            headers=headers or {},
            timeout_seconds=timeout_seconds,
        )
    if transport_type == "http-streamable":
        return HTTPStreamableClient(
            url=url,
            headers=headers or {},
            timeout_seconds=timeout_seconds,
        )
    raise MCPTransportError(
        f"unsupported transport_type: {transport_type!r}; "
        "only 'sse' and 'http-streamable' are accepted",
    )


class MCPConnectionManager:
    """Pool of live MCP sessions.

    The manager exposes the same three operations as the Go side:

    - :meth:`get_or_create` — return the live session or build one;
    - :meth:`close_service` — drop every session for one service;
    - :meth:`shutdown` — drop everything and stop the cleanup loop.

    The cleanup loop runs every ``cleanup_interval_seconds`` and evicts
    sessions whose transport reports itself disconnected, mirroring the
    5-minute ticker in Go.
    """

    def __init__(
        self,
        *,
        transport_factory: _TransportFactory | None = None,
        cleanup_interval_seconds: float = 300.0,
        default_timeout_seconds: float = 30.0,
        max_timeout_seconds: float = 60.0,
    ) -> None:
        self._factory: _TransportFactory = transport_factory or _default_transport_factory
        self._cleanup_interval = cleanup_interval_seconds
        self._default_timeout = default_timeout_seconds
        self._max_timeout = max_timeout_seconds
        self._sessions: dict[str, MCPSession] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    # ── Session lifecycle ──────────────────────────────────────────

    async def get_or_create(
        self,
        *,
        service_id: str,
        transport_type: str,
        url: str,
        headers: dict[str, str] | None,
        advanced_timeout_seconds: int | None = None,
        service_name: str | None = None,
    ) -> MCPSession:
        """Return a connected, initialized session for the service.

        Re-uses a live session; otherwise creates one, performs the
        MCP ``initialize`` handshake, and caches it.
        """
        if not url:
            raise MCPTransportError(
                f"service {service_id!r}: URL is required for transport_type={transport_type!r}",
            )
        async with self._lock:
            session = self._sessions.get(service_id)
            if session is not None and session.is_alive() and session.initialized:
                return session
            if session is not None:
                # The session is stale — drop it before we rebuild.
                await _safe_close(session)

            timeout = self._resolve_timeout(advanced_timeout_seconds)
            client = self._factory(
                service_id=service_id,
                transport_type=transport_type,
                url=url,
                headers=headers,
                timeout_seconds=timeout,
            )
            session = MCPSession(
                service_id=service_id,
                transport_type=transport_type,
                client=client,
            )
            self._sessions[service_id] = session

        try:
            await session.client.connect()
            session.connected = True
            await self._initialize(session)
            session.initialized = True
        except MCPError:
            await self._evict(service_id)
            raise
        except Exception as exc:
            await self._evict(service_id)
            raise MCPTransportError(
                f"failed to initialize MCP session for {service_id!r}: {exc}",
            ) from exc
        return session

    async def ping(
        self,
        *,
        session: MCPSession,
    ) -> JSONRPCResponse:
        """Send a ``ping`` over the cached session.

        Used by the heartbeat loop to detect dead sessions before a
        real call lands; mirrors the Go ``Ping`` helper.
        """
        return await self._invoke(
            session,
            method="ping",
            params={},
            evict_on_session_invalid=False,
        )

    async def list_tools(
        self,
        *,
        session: MCPSession,
    ) -> JSONRPCResponse:
        """Invoke ``tools/list`` on the cached session."""
        return await self._invoke(
            session,
            method=METHOD_TOOLS_LIST,
            params={},
            evict_on_session_invalid=True,
        )

    async def list_resources(
        self,
        *,
        session: MCPSession,
    ) -> JSONRPCResponse:
        """Invoke ``resources/list`` on the cached session."""
        return await self._invoke(
            session,
            method=METHOD_RESOURCES_LIST,
            params={},
            evict_on_session_invalid=True,
        )

    async def call_tool(
        self,
        *,
        session: MCPSession,
        tool_name: str,
        arguments: dict[str, object] | None,
    ) -> JSONRPCResponse:
        """Invoke ``tools/call`` on the cached session."""
        return await self._invoke(
            session,
            method=METHOD_TOOLS_CALL,
            params={"name": tool_name, "arguments": arguments or {}},
            evict_on_session_invalid=True,
        )

    async def read_resource(
        self,
        *,
        session: MCPSession,
        uri: str,
    ) -> JSONRPCResponse:
        """Invoke ``resources/read`` on the cached session."""
        return await self._invoke(
            session,
            method=METHOD_RESOURCES_READ,
            params={"uri": uri},
            evict_on_session_invalid=True,
        )

    async def _invoke(
        self,
        session: MCPSession,
        *,
        method: str,
        params: dict[str, object],
        evict_on_session_invalid: bool,
    ) -> JSONRPCResponse:
        """Send one JSON-RPC request and surface the response.

        On a session-invalid error (server says ``Invalid session ID``
        or ``No active connection``) the session is evicted so the next
        caller rebuilds it. The call itself is not retried here —
        callers that want a retry-once-on-fresh-session flow re-invoke
        ``_invoke`` themselves. Other transport errors bubble up
        unchanged.
        """
        if not session.is_alive():
            raise SessionNotConnectedError(
                f"session for {session.service_id!r} is not connected",
            )
        try:
            response = await session.client.request(method=method, params=params)
            return cast("JSONRPCResponse", response)
        except MCPTransportError as exc:
            if evict_on_session_invalid and _looks_like_session_invalid(exc):
                await self._evict(session.service_id)
                raise
            raise

    async def _initialize(self, session: MCPSession) -> None:
        """Send the MCP ``initialize`` handshake."""
        params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "knowledge-be", "version": "0.1.0"},
        }
        response = await session.client.request(
            method=METHOD_INITIALIZE,
            params=params,
        )
        if response.error is not None:
            raise MCPTransportError(
                f"initialize failed: {response.error.message} (code={response.error.code})",
            )

    # ── Cache management ────────────────────────────────────────────

    async def close_service(self, service_id: str) -> None:
        """Drop every session for one service (idempotent)."""
        async with self._lock:
            keys: Iterable[str] = [
                key
                for key in list(self._sessions)
                if key == service_id or key.startswith(f"{service_id}\x00")
            ]
            for key in keys:
                session = self._sessions.pop(key, None)
                if session is not None:
                    await _safe_close(session)

    async def _evict(self, service_id: str) -> None:
        """Close + remove one session; swallow errors during teardown."""
        async with self._lock:
            session = self._sessions.pop(service_id, None)
        if session is not None:
            await _safe_close(session)

    async def shutdown(self) -> None:
        """Stop the cleanup loop and close every cached session."""
        self._stop_event.set()
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, BaseException):
                await self._cleanup_task
            self._cleanup_task = None
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            await _safe_close(session)

    def start_cleanup(self) -> None:
        """Start the background sweep that evicts dead sessions.

        Mirrors the goroutine started by Go ``NewMCPManager``. Safe to
        call repeatedly; the second call is a no-op.
        """
        if self._cleanup_task is not None and not self._cleanup_task.done():
            return
        self._stop_event.clear()
        self._cleanup_task = asyncio.create_task(
            self._cleanup_loop(),
            name="mcp-connection-cleanup",
        )

    async def _cleanup_loop(self) -> None:
        """Periodically drop sessions whose transport is dead."""
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._cleanup_interval,
                )
                # Event signalled — exit without another sweep.
                return
            except TimeoutError:
                pass
            await self._sweep_dead_sessions()

    async def _sweep_dead_sessions(self) -> None:
        """Remove every session whose transport reports disconnected."""
        async with self._lock:
            stale = [key for key, session in self._sessions.items() if not session.is_alive()]
            for key in stale:
                session = self._sessions.pop(key, None)
                if session is not None:
                    await _safe_close(session)

    def active_sessions(self) -> list[str]:
        """Return the list of service ids that currently have a live session."""
        return [key for key, session in self._sessions.items() if session.is_alive()]

    # ── Internals ──────────────────────────────────────────────────

    def _resolve_timeout(self, advanced_timeout_seconds: int | None) -> float:
        """Compute the per-request timeout honoring ``advanced_config``."""
        if advanced_timeout_seconds is None or advanced_timeout_seconds <= 0:
            return self._default_timeout
        upper = min(float(advanced_timeout_seconds), self._max_timeout)
        return max(upper, 1.0)


async def _safe_close(session: MCPSession) -> None:
    """Close a session, swallowing any error during teardown."""
    try:
        await session.close()
    except (MCPError, asyncio.CancelledError):
        pass
    except Exception:
        # Teardown must never raise — the manager cleanup path is best
        # effort.
        pass


def _looks_like_session_invalid(error: MCPTransportError) -> bool:
    """True if ``error`` text matches a known session-invalidation hint.

    Wraps :func:`src.ai.mcp_transport.jsonrpc.is_session_invalid_error`
    for the transport-error path (where we only have a string).
    """
    needle = error.message_text.lower()
    return any(hint in needle for hint in SESSION_INVALID_HINTS)


__all__ = [
    "SESSION_INVALID_HINTS",
    "MCPConnectionManager",
    "MCPSession",
    "_default_transport_factory",
    "_looks_like_session_invalid",
]
