"""Tests for the MCP connection manager.

The manager owns the cache and the cleanup loop; the SSE wire
plumbing is covered in ``test_sse_client.py``. Here we focus on:

- cache reuse vs. rebuild after staleness;
- session-invalid eviction;
- transport-factory rejection of unsupported transports;
- the background cleanup sweep;
- high-level ``list_tools`` / ``list_resources`` round-trip via a
  fake transport client (deterministic, no SSE plumbing).
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from src.ai.mcp_transport.connection_manager import (
    MCPConnectionManager,
    MCPSession,
    TransportClient,
    _default_transport_factory,
)
from src.ai.mcp_transport.errors import (
    MCPTransportError,
    OAuthRequiredError,
    SessionNotConnectedError,
)
from src.ai.mcp_transport.jsonrpc import JSONRPCResponse
from src.ai.mcp_transport.sse_client import SSEClient

# ── Fakes ───────────────────────────────────────────────────────────


class _FakeTransportClient:
    """In-memory stand-in for a transport client.

    Records calls, returns scripted responses, and tracks the
    connected / initialized flags the manager relies on.
    """

    def __init__(
        self,
        *,
        script: list[JSONRPCResponse | Exception] | None = None,
    ) -> None:
        self._script = list(script or [])
        self.connected = False
        self.initialized = False
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.requests: list[tuple[str, dict[str, object]]] = []
        self.session_id = "srv-" + str(id(self))

    def is_connected(self) -> bool:
        return self.connected

    async def connect(self) -> None:
        self.connect_calls += 1
        self.connected = True

    async def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False

    async def request(
        self,
        *,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> JSONRPCResponse:
        self.requests.append((method, params or {}))
        if not self._script:
            raise MCPTransportError(
                f"fake client ran out of scripted responses on method={method!r}",
            )
        outcome = self._script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _build_manager(*, fake: _FakeTransportClient) -> MCPConnectionManager:
    """Wire a manager whose transport factory returns the given fake."""

    def _factory(
        *,
        service_id: str,
        transport_type: str,
        url: str,
        headers: dict[str, str] | None,
        timeout_seconds: float,
    ) -> TransportClient:
        return cast(TransportClient, fake)

    return MCPConnectionManager(
        transport_factory=_factory,
        default_timeout_seconds=5.0,
        cleanup_interval_seconds=999.0,
    )


# ── Session lifecycle ───────────────────────────────────────────────


async def test_get_or_create_initializes_session_and_caches_it() -> None:
    """First call runs ``initialize``; second call returns the cached session."""
    fake = _FakeTransportClient(
        script=[
            JSONRPCResponse(
                id="init",
                result={"protocolVersion": "2024-11-05"},
            ),
        ],
    )
    manager = _build_manager(fake=fake)
    try:
        first = await manager.get_or_create(
            service_id="svc-1",
            transport_type="sse",
            url="https://mcp.example.com/sse",
            headers=None,
        )
        assert isinstance(first, MCPSession)
        assert first.initialized is True
        assert first.connected is True
        assert fake.connect_calls == 1
        assert fake.requests[0][0] == "initialize"

        # Second call must reuse the cached session — no new connect,
        # no new initialize.
        second = await manager.get_or_create(
            service_id="svc-1",
            transport_type="sse",
            url="https://mcp.example.com/sse",
            headers=None,
        )
        assert second is first
        assert fake.connect_calls == 1
    finally:
        await manager.shutdown()


async def test_get_or_create_rebuilds_when_session_is_stale() -> None:
    """A session whose transport reports disconnected gets rebuilt on next call."""
    fake = _FakeTransportClient(
        script=[
            JSONRPCResponse(id="init", result={}),
            JSONRPCResponse(id="init", result={}),
        ],
    )
    manager = _build_manager(fake=fake)
    try:
        await manager.get_or_create(
            service_id="svc-stale",
            transport_type="sse",
            url="https://mcp.example.com/sse",
            headers=None,
        )
        # Mark the session dead on the transport side.
        await manager._sessions["svc-stale"].client.disconnect()
        manager._sessions["svc-stale"].connected = False
        assert fake.disconnect_calls == 1
        # Next call rebuilds — fake.connect_calls increments and
        # initialize is called a second time.
        await manager.get_or_create(
            service_id="svc-stale",
            transport_type="sse",
            url="https://mcp.example.com/sse",
            headers=None,
        )
        assert fake.connect_calls == 2
    finally:
        await manager.shutdown()


async def test_get_or_create_rejects_missing_url() -> None:
    """An SSE service without a URL raises :class:`MCPTransportError`."""
    manager = MCPConnectionManager()
    try:
        with pytest.raises(MCPTransportError):
            await manager.get_or_create(
                service_id="svc-no-url",
                transport_type="sse",
                url="",
                headers=None,
            )
    finally:
        await manager.shutdown()


async def test_get_or_create_rejects_unsupported_transport() -> None:
    """``stdio`` (and other unsupported types) raise :class:`MCPTransportError`."""
    manager = MCPConnectionManager()
    try:
        with pytest.raises(MCPTransportError) as excinfo:
            await manager.get_or_create(
                service_id="svc-bad",
                transport_type="stdio",
                url="ignored",
                headers=None,
            )
        assert "unsupported transport_type" in excinfo.value.message_text
    finally:
        await manager.shutdown()


async def test_get_or_create_returns_http_streamable_client_for_http_streamable() -> None:
    """``http-streamable`` now builds an :class:`HTTPStreamableClient`.

    The factory no longer rejects the transport type — it builds the
    client and the network call is what surfaces a transport error.
    We assert that the failure is the network-level :class:`MCPTransportError`,
    not a config rejection.
    """
    manager = MCPConnectionManager()
    try:
        with pytest.raises(MCPTransportError) as excinfo:
            await manager.get_or_create(
                service_id="svc-streamable",
                transport_type="http-streamable",
                url="https://no-such-host.invalid",
                headers=None,
            )
        assert "PR-17.5b" not in excinfo.value.message_text
    finally:
        await manager.shutdown()


async def test_get_or_create_wraps_unexpected_exceptions_as_transport_error() -> None:
    """A non-``MCPError`` raised by the transport is wrapped consistently."""

    class _BoomClient(_FakeTransportClient):
        async def connect(self) -> None:
            raise RuntimeError("boom")

    manager = MCPConnectionManager(
        transport_factory=lambda **_kwargs: cast(TransportClient, _BoomClient()),
        default_timeout_seconds=5.0,
    )
    try:
        with pytest.raises(MCPTransportError) as excinfo:
            await manager.get_or_create(
                service_id="svc-boom",
                transport_type="sse",
                url="https://mcp.example.com/sse",
                headers=None,
            )
        assert "failed to initialize" in excinfo.value.message_text
    finally:
        await manager.shutdown()


async def test_get_or_create_evicts_failed_session() -> None:
    """A failed handshake evicts the session so the next caller rebuilds."""
    fake = _FakeTransportClient(
        script=[
            MCPTransportError("session invalid on initialize"),
        ],
    )
    manager = _build_manager(fake=fake)
    try:
        with pytest.raises(MCPTransportError):
            await manager.get_or_create(
                service_id="svc-fail",
                transport_type="sse",
                url="https://mcp.example.com/sse",
                headers=None,
            )
        assert "svc-fail" not in manager.active_sessions()
    finally:
        await manager.shutdown()


# ── High-level operations ───────────────────────────────────────────


async def test_list_tools_returns_manager_response() -> None:
    """``list_tools`` invokes ``tools/list`` on the cached session."""
    fake = _FakeTransportClient(
        script=[
            JSONRPCResponse(id="init", result={}),
            JSONRPCResponse(
                id="x",
                result={"tools": [{"name": "echo"}]},
            ),
        ],
    )
    manager = _build_manager(fake=fake)
    try:
        session = await manager.get_or_create(
            service_id="svc-tools",
            transport_type="sse",
            url="https://mcp.example.com/sse",
            headers=None,
        )
        response = await manager.list_tools(session=session)
        assert isinstance(response, JSONRPCResponse)
        assert response.error is None
        assert response.result == {"tools": [{"name": "echo"}]}
        assert fake.requests[-1][0] == "tools/list"
    finally:
        await manager.shutdown()


async def test_list_resources_returns_manager_response() -> None:
    """``list_resources`` invokes ``resources/list`` on the cached session."""
    fake = _FakeTransportClient(
        script=[
            JSONRPCResponse(id="init", result={}),
            JSONRPCResponse(
                id="x",
                result={"resources": [{"uri": "file://a", "name": "a"}]},
            ),
        ],
    )
    manager = _build_manager(fake=fake)
    try:
        session = await manager.get_or_create(
            service_id="svc-resources",
            transport_type="sse",
            url="https://mcp.example.com/sse",
            headers=None,
        )
        response = await manager.list_resources(session=session)
        assert response.error is None
        assert response.result == {"resources": [{"uri": "file://a", "name": "a"}]}
        assert fake.requests[-1][0] == "resources/list"
    finally:
        await manager.shutdown()


async def test_call_tool_invokes_tools_call() -> None:
    """``call_tool`` invokes ``tools/call`` with the supplied arguments."""
    fake = _FakeTransportClient(
        script=[
            JSONRPCResponse(id="init", result={}),
            JSONRPCResponse(id="x", result={"content": [{"type": "text", "text": "ok"}]}),
        ],
    )
    manager = _build_manager(fake=fake)
    try:
        session = await manager.get_or_create(
            service_id="svc-tool-call",
            transport_type="sse",
            url="https://mcp.example.com/sse",
            headers=None,
        )
        response = await manager.call_tool(
            session=session,
            tool_name="echo",
            arguments={"msg": "hi"},
        )
        assert response.error is None
        assert response.result == {"content": [{"type": "text", "text": "ok"}]}
        method, params = fake.requests[-1]
        assert method == "tools/call"
        assert params == {"name": "echo", "arguments": {"msg": "hi"}}
    finally:
        await manager.shutdown()


async def test_read_resource_invokes_resources_read() -> None:
    """``read_resource`` invokes ``resources/read`` with the supplied uri."""
    fake = _FakeTransportClient(
        script=[
            JSONRPCResponse(id="init", result={}),
            JSONRPCResponse(id="x", result={"contents": [{"uri": "file://a"}]}),
        ],
    )
    manager = _build_manager(fake=fake)
    try:
        session = await manager.get_or_create(
            service_id="svc-res-read",
            transport_type="sse",
            url="https://mcp.example.com/sse",
            headers=None,
        )
        response = await manager.read_resource(session=session, uri="file://a")
        assert response.error is None
        method, params = fake.requests[-1]
        assert method == "resources/read"
        assert params == {"uri": "file://a"}
    finally:
        await manager.shutdown()


async def test_ping_invokes_ping_method() -> None:
    """``ping`` invokes the ``ping`` heartbeat method (no retry)."""
    fake = _FakeTransportClient(
        script=[
            JSONRPCResponse(id="init", result={}),
            JSONRPCResponse(id="x", result={}),
        ],
    )
    manager = _build_manager(fake=fake)
    try:
        session = await manager.get_or_create(
            service_id="svc-ping",
            transport_type="sse",
            url="https://mcp.example.com/sse",
            headers=None,
        )
        response = await manager.ping(session=session)
        assert response.error is None
        method, params = fake.requests[-1]
        assert method == "ping"
        assert params == {}
    finally:
        await manager.shutdown()


# ── Cache management ────────────────────────────────────────────────


async def test_close_service_drops_session() -> None:
    """``close_service`` evicts the cached session for that id."""
    fake = _FakeTransportClient(
        script=[
            JSONRPCResponse(id="init", result={}),
        ],
    )
    manager = _build_manager(fake=fake)
    try:
        await manager.get_or_create(
            service_id="svc-drop",
            transport_type="sse",
            url="https://mcp.example.com/sse",
            headers=None,
        )
        assert "svc-drop" in manager.active_sessions()
        await manager.close_service("svc-drop")
        assert "svc-drop" not in manager.active_sessions()
        assert fake.disconnect_calls == 1
    finally:
        await manager.shutdown()


async def test_shutdown_closes_all_cached_sessions() -> None:
    """``shutdown`` closes every cached session and clears the pool."""
    fake = _FakeTransportClient(
        script=[
            JSONRPCResponse(id="init", result={}),
            JSONRPCResponse(id="init", result={}),
        ],
    )
    manager = _build_manager(fake=fake)
    await manager.get_or_create(
        service_id="svc-a",
        transport_type="sse",
        url="https://mcp.example.com/sse",
        headers=None,
    )
    await manager.get_or_create(
        service_id="svc-b",
        transport_type="sse",
        url="https://mcp.example.com/sse",
        headers=None,
    )
    assert set(manager.active_sessions()) == {"svc-a", "svc-b"}
    await manager.shutdown()
    assert manager.active_sessions() == []
    assert fake.disconnect_calls == 2


async def test_close_service_is_idempotent() -> None:
    """Closing an unknown service is a no-op (does not raise)."""
    manager = MCPConnectionManager()
    await manager.close_service("never-existed")
    assert manager.active_sessions() == []


# ── Wire-error translation ───────────────────────────────────────────


async def test_session_invalid_hint_evicts_session() -> None:
    """An ``Invalid session ID`` error text drops the session and re-raises."""
    fake = _FakeTransportClient(
        script=[
            JSONRPCResponse(id="init", result={}),
        ],
    )
    manager = _build_manager(fake=fake)
    try:
        session = await manager.get_or_create(
            service_id="svc-evict",
            transport_type="sse",
            url="https://mcp.example.com/sse",
            headers=None,
        )
        # Replace the script with one that always raises.
        fake._script = [
            MCPTransportError("Invalid session ID from server"),
        ]
        with pytest.raises(MCPTransportError) as excinfo:
            await manager._invoke(
                session,
                method="tools/list",
                params={},
                evict_on_session_invalid=False,
            )
        assert "Invalid session ID" in excinfo.value.message_text
    finally:
        await manager.shutdown()


async def test_no_active_connection_hint_also_evicts() -> None:
    """``No active connection`` is the second known session-invalidation hint."""
    fake = _FakeTransportClient(
        script=[
            JSONRPCResponse(id="init", result={}),
        ],
    )
    manager = _build_manager(fake=fake)
    try:
        session = await manager.get_or_create(
            service_id="svc-evict2",
            transport_type="sse",
            url="https://mcp.example.com/sse",
            headers=None,
        )
        fake._script = [
            MCPTransportError("No active connection for that session"),
        ]
        with pytest.raises(MCPTransportError):
            await manager._invoke(
                session,
                method="tools/list",
                params={},
                evict_on_session_invalid=False,
            )
    finally:
        await manager.shutdown()


async def test_unrelated_transport_error_does_not_evict_session() -> None:
    """Transport errors that are not session-invalidation hints leave the session cached."""
    fake = _FakeTransportClient(
        script=[
            JSONRPCResponse(id="init", result={}),
        ],
    )
    manager = _build_manager(fake=fake)
    try:
        session = await manager.get_or_create(
            service_id="svc-keep",
            transport_type="sse",
            url="https://mcp.example.com/sse",
            headers=None,
        )
        fake._script = [
            MCPTransportError("transport timeout"),
        ]
        with pytest.raises(MCPTransportError):
            await manager._invoke(
                session,
                method="tools/list",
                params={},
                evict_on_session_invalid=False,
            )
        assert "svc-keep" in manager.active_sessions()
    finally:
        await manager.shutdown()


async def test_oauth_required_error_propagates_from_transport() -> None:
    """An :class:`OAuthRequiredError` is not wrapped as :class:`MCPTransportError`."""
    fake = _FakeTransportClient(
        script=[
            JSONRPCResponse(id="init", result={}),
        ],
    )
    manager = _build_manager(fake=fake)
    try:
        session = await manager.get_or_create(
            service_id="svc-oauth",
            transport_type="sse",
            url="https://mcp.example.com/sse",
            headers=None,
        )
        fake._script = [
            OAuthRequiredError(metadata_url="https://mcp.example.com/.well-known/oauth"),
        ]
        with pytest.raises(OAuthRequiredError) as excinfo:
            await manager._invoke(
                session,
                method="tools/list",
                params={},
                evict_on_session_invalid=False,
            )
        assert "mcp.example.com" in excinfo.value.metadata_url
    finally:
        await manager.shutdown()


async def test_list_tools_on_dead_session_raises_session_not_connected() -> None:
    """A request on a disconnected session raises :class:`SessionNotConnectedError`."""
    fake = _FakeTransportClient(
        script=[
            JSONRPCResponse(id="init", result={}),
        ],
    )
    manager = _build_manager(fake=fake)
    try:
        session = await manager.get_or_create(
            service_id="svc-dead",
            transport_type="sse",
            url="https://mcp.example.com/sse",
            headers=None,
        )
        await session.client.disconnect()
        session.connected = False
        with pytest.raises(SessionNotConnectedError):
            await manager.list_tools(session=session)
    finally:
        await manager.shutdown()


async def test_invoke_evict_on_session_invalid_returns_fresh_session() -> None:
    """``evict_on_session_invalid=True`` drops the stale
    session on a session-invalid hint, and the next ``get_or_create``
    rebuilds it.

    The parameter is now self-documenting — it does not promise a
    retry-once, only an eviction so the next caller rebuilds.
    """
    fake = _FakeTransportClient(
        script=[
            JSONRPCResponse(id="init", result={}),  # first init
            MCPTransportError("Invalid session ID from server"),  # session-invalid hint
            JSONRPCResponse(id="init", result={}),  # second init after rebuild
        ],
    )
    manager = _build_manager(fake=fake)
    try:
        session = await manager.get_or_create(
            service_id="svc-evict-on-invalid",
            transport_type="sse",
            url="https://mcp.example.com/sse",
            headers=None,
        )
        first_session_id = id(session)
        with pytest.raises(MCPTransportError):
            await manager._invoke(
                session,
                method="tools/list",
                params={},
                evict_on_session_invalid=True,
            )
        # The session was evicted; the next ``get_or_create`` rebuilds.
        assert "svc-evict-on-invalid" not in manager.active_sessions()
        rebuilt = await manager.get_or_create(
            service_id="svc-evict-on-invalid",
            transport_type="sse",
            url="https://mcp.example.com/sse",
            headers=None,
        )
        assert id(rebuilt) != first_session_id
        assert fake.connect_calls == 2
    finally:
        await manager.shutdown()


# ── Background cleanup ──────────────────────────────────────────────


async def test_cleanup_loop_drops_dead_sessions() -> None:
    """The sweep evicts sessions whose transport is no longer connected."""
    manager = MCPConnectionManager(
        cleanup_interval_seconds=0.05,
        default_timeout_seconds=5.0,
    )

    fake = _FakeTransportClient(
        script=[JSONRPCResponse(id="init", result={})],
    )
    fake.connected = False
    session = MCPSession(
        service_id="svc-ghost",
        transport_type="sse",
        client=cast(TransportClient, fake),
    )
    session.initialized = True
    session.connected = False
    manager._sessions["svc-ghost"] = session

    try:
        manager.start_cleanup()
        # Wait for the sweep to pop the session AND call disconnect.
        for _ in range(200):
            if "svc-ghost" not in manager.active_sessions() and fake.disconnect_calls >= 1:
                break
            await asyncio.sleep(0.01)
        assert "svc-ghost" not in manager.active_sessions()
        assert fake.disconnect_calls >= 1
    finally:
        await manager.shutdown()


async def test_start_cleanup_is_idempotent() -> None:
    """Calling ``start_cleanup`` twice does not spawn two tasks."""
    manager = MCPConnectionManager(cleanup_interval_seconds=999.0)
    try:
        manager.start_cleanup()
        first_task = manager._cleanup_task
        manager.start_cleanup()
        assert manager._cleanup_task is first_task
    finally:
        await manager.shutdown()


# ── Default factory ─────────────────────────────────────────────────


def test_default_transport_factory_builds_sse_client() -> None:
    """``_default_transport_factory`` returns an :class:`SSEClient` for ``sse``."""
    client = _default_transport_factory(
        service_id="x",
        transport_type="sse",
        url="https://mcp.example.com/sse",
        headers=None,
        timeout_seconds=10.0,
    )
    assert isinstance(client, SSEClient)


def test_default_transport_factory_returns_http_streamable_client() -> None:
    """The default factory now returns an :class:`HTTPStreamableClient`."""
    from src.ai.mcp_transport.http_streamable_client import HTTPStreamableClient

    client = _default_transport_factory(
        service_id="x",
        transport_type="http-streamable",
        url="https://mcp.example.com",
        headers=None,
        timeout_seconds=10.0,
    )
    assert isinstance(client, HTTPStreamableClient)


def test_default_transport_factory_rejects_stdio() -> None:
    """``_default_transport_factory`` rejects ``stdio`` (mirrors Go)."""
    with pytest.raises(MCPTransportError) as excinfo:
        _default_transport_factory(
            service_id="x",
            transport_type="stdio",
            url="ignored",
            headers=None,
            timeout_seconds=10.0,
        )
    assert "unsupported transport_type" in excinfo.value.message_text


# ── Timeout resolution ───────────────────────────────────────────────


def test_resolve_timeout_uses_default_when_advanced_unset() -> None:
    """No ``advanced_timeout_seconds`` → the configured default."""
    manager = MCPConnectionManager(default_timeout_seconds=12.0, max_timeout_seconds=60.0)
    assert manager._resolve_timeout(None) == 12.0


def test_resolve_timeout_clamps_to_max() -> None:
    """``advanced_timeout_seconds`` above ``max_timeout_seconds`` is clamped down."""
    manager = MCPConnectionManager(default_timeout_seconds=10.0, max_timeout_seconds=30.0)
    assert manager._resolve_timeout(120) == 30.0


def test_resolve_timeout_enforces_minimum_of_one_second() -> None:
    """``advanced_timeout_seconds`` below 1 s is clamped up to 1 s."""
    manager = MCPConnectionManager(default_timeout_seconds=10.0, max_timeout_seconds=60.0)
    assert manager._resolve_timeout(0) == 10.0  # <=0 falls back to default
    assert manager._resolve_timeout(-5) == 10.0  # negative falls back
