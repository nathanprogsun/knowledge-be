"""SSE transport for MCP — long-lived ``GET /sse`` + ``POST /messages``.

Mirrors the SSE branch in ``internal/mcp/client.go`` (the Go side
delegates to ``mark3labs/mcp-go``'s SSE client which itself implements
the same handshake). The remote MCP server exposes a
``text/event-stream`` endpoint; the client opens a GET to that URL,
parses the first ``endpoint`` event to discover the ``/messages``
POST URL, then sends JSON-RPC requests over POST and reads
``message`` events off the SSE stream for responses.

This module is wire-only: it does not run a cleanup loop, does not
pool sessions, and does not translate transport errors into MCP
domain errors. Those concerns live in
:mod:`src.ai.mcp_transport.connection_manager`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from contextlib import AbstractAsyncContextManager
from urllib.parse import urlsplit, urlunsplit

import httpx
from httpx_sse import EventSource, ServerSentEvent, aconnect_sse
from httpx_sse._exceptions import SSEError

from src.ai.mcp_transport.errors import MCPTransportError, OAuthRequiredError
from src.ai.mcp_transport.jsonrpc import JSONRPCResponse, build_request
from src.common.json import JsonValue

# Match the RFC 9728 resource-metadata link advertised in the
# ``WWW-Authenticate`` header by an MCP server that requires OAuth.
_RFC9728_LINK_RE = re.compile(
    r'resource_metadata\s*=\s*"([^"]+)"',
    re.IGNORECASE,
)


class SSEClient:
    """One MCP-over-SSE connection.

    The constructor is intentionally side-effect free; the caller
    invokes :meth:`connect` to open the SSE stream and discover the
    POST endpoint, then :meth:`request` for every JSON-RPC call, and
    finally :meth:`disconnect` to release the underlying HTTP client.
    """

    def __init__(
        self,
        *,
        url: str,
        headers: dict[str, str] | None = None,
        timeout_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = url
        self._headers = dict(headers or {})
        self._timeout = timeout_seconds
        self._client = client
        self._owns_client = client is None
        self._event_source: EventSource | None = None
        self._post_endpoint: str = ""
        self._connected = False
        # The ``aconnect_sse`` async context manager stays open until
        # :meth:`disconnect` exits it. ``httpx_sse`` types it as an
        # ``AbstractAsyncContextManager[EventSource]`` so we use the same
        # shape; ``__aexit__`` swallows the connection error so the
        # ``event_source`` payload never escapes through us.
        self._context_manager: AbstractAsyncContextManager[EventSource] | None = None
        # The ``httpx`` stream iterator is single-pass; we drain it
        # into this queue so :meth:`_wait_for_endpoint` and
        # :meth:`_await_message` can read sequentially without
        # exhausting the response.
        self._event_queue: asyncio.Queue[ServerSentEvent | BaseException] | None = None
        self._drain_task: asyncio.Task[None] | None = None

    # ── Lifecycle ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open the SSE stream and wait for the ``endpoint`` event."""
        if self._connected:
            return
        client = self._ensure_client()
        try:
            context = aconnect_sse(
                client,
                "GET",
                self._url,
                headers=self._headers,
                timeout=self._timeout,
            )
            event_source = await context.__aenter__()
        except httpx.HTTPStatusError as exc:
            raise MCPTransportError(
                f"SSE handshake failed: HTTP {exc.response.status_code}",
                status_code=exc.response.status_code,
            ) from exc
        except SSEError as exc:
            # ``httpx_sse`` raises when the response is not
            # ``text/event-stream`` (e.g. a 502 with ``text/plain``);
            # surface it as a transport error so callers do not have
            # to depend on the ``httpx_sse`` exception type.
            raise MCPTransportError(
                f"SSE handshake failed: {type(exc).__name__}: {exc}",
            ) from exc
        except httpx.HTTPError as exc:
            raise MCPTransportError(
                f"SSE handshake failed: {type(exc).__name__}: {exc}",
            ) from exc
        self._context_manager = context
        self._event_source = event_source
        self._event_queue = asyncio.Queue()
        self._drain_task = asyncio.create_task(
            self._drain_events(event_source),
            name="mcp-sse-drain",
        )
        try:
            await self._wait_for_endpoint()
        except MCPTransportError:
            await self.disconnect()
            raise
        self._connected = True

    async def disconnect(self) -> None:
        """Close the SSE stream and release the underlying client."""
        # Cancel the drain task first so it stops reading the
        # ``aconnect_sse`` context.
        if self._drain_task is not None:
            self._drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._drain_task
            self._drain_task = None
        context = self._context_manager
        self._context_manager = None
        self._event_source = None
        self._event_queue = None
        self._post_endpoint = ""
        self._connected = False
        if context is not None:
            # ``httpx_sse`` uses ``httpx_sse.ServerSentEvent`` to
            # surface SSE errors; suppress them on teardown so callers
            # never see a half-closed stream.
            with contextlib.suppress(httpx.HTTPError, Exception):
                await context.__aexit__(None, None, None)
        if self._owns_client and self._client is not None:
            with contextlib.suppress(httpx.HTTPError, Exception):
                await self._client.aclose()
            self._client = None

    def is_connected(self) -> bool:
        """Return whether the SSE stream is open."""
        return self._connected

    @property
    def post_endpoint(self) -> str:
        """Return the resolved POST URL the server advertised."""
        return self._post_endpoint

    # ── Request flow ───────────────────────────────────────────────

    async def request(
        self,
        *,
        method: str,
        params: dict[str, JsonValue] | None = None,
        request_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> JSONRPCResponse:
        """Send one JSON-RPC request via POST and await its SSE response.

        The response is correlated by id via a small waiter registry.
        The MCP-over-SSE spec mandates that the server answers with an
        SSE ``message`` event carrying the JSON-RPC response on the
        same stream the GET was opened on.
        """
        if not self._connected:
            raise MCPTransportError("SSE client is not connected")
        envelope = build_request(
            method=method,
            params=params,
            request_id=request_id,
        )
        client = self._ensure_client()
        try:
            response = await client.post(
                self._post_endpoint,
                json=json.loads(envelope.model_dump_json()),
                headers=self._headers,
                timeout=timeout_seconds if timeout_seconds is not None else self._timeout,
            )
        except httpx.HTTPError as exc:
            raise MCPTransportError(
                f"SSE POST failed: {type(exc).__name__}: {exc}",
            ) from exc
        if response.status_code in {401, 403}:
            _raise_oauth_required_if_advertised(response)
        if response.status_code >= 400:
            raise MCPTransportError(
                f"SSE POST returned HTTP {response.status_code}",
                status_code=response.status_code,
                body=response.text,
            )
        # Per the MCP-over-SSE spec, the server answers with an SSE
        # ``message`` event carrying the JSON-RPC response. We stream
        # events off the open SSE stream until one matches our id.
        return await self._await_message(envelope.id)

    # ── Internals ──────────────────────────────────────────────────

    def _ensure_client(self) -> httpx.AsyncClient:
        """Return the shared ``httpx.AsyncClient`` (lazily creating one)."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def _drain_events(self, event_source: EventSource) -> None:
        """Background task that pumps ``event_source.aiter_sse()`` into ``_event_queue``.

        The ``httpx`` response stream is single-pass; running a
        background drain keeps the iterator alive across multiple
        consumer awaits (the initial endpoint read and subsequent
        message reads).
        """
        queue = self._event_queue
        assert queue is not None
        try:
            async for event in event_source.aiter_sse():
                await queue.put(event)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            await queue.put(exc)
            return
        # Sentinel: stream ended cleanly.
        await queue.put(_STREAM_CLOSED)  # type: ignore[arg-type]

    async def _wait_for_endpoint(self) -> None:
        """Wait for the ``endpoint`` SSE event from the server.

        The MCP-over-SSE spec mandates that the server emit an event
        with ``event: endpoint`` and ``data: <post-url>`` as the first
        event on every SSE stream.
        """
        queue = self._event_queue
        assert queue is not None
        while True:
            event = await queue.get()
            if isinstance(event, BaseException):
                if isinstance(event, SSEError):
                    raise MCPTransportError(
                        f"SSE handshake failed: {type(event).__name__}: {event}",
                    ) from event
                raise MCPTransportError(
                    f"SSE handshake failed: {type(event).__name__}: {event}",
                ) from event
            if event is _STREAM_CLOSED:
                raise MCPTransportError("SSE stream closed before endpoint event")
            if event.event == "endpoint":
                post_url = (event.data or "").strip()
                if not post_url:
                    raise MCPTransportError(
                        "SSE server returned an empty endpoint event",
                    )
                self._post_endpoint = _resolve_post_url(self._url, post_url)
                return

    async def _await_message(self, request_id: str) -> JSONRPCResponse:
        """Read SSE ``message`` events until one matches ``request_id``."""
        queue = self._event_queue
        if queue is None:
            raise MCPTransportError("SSE stream was closed mid-call")
        while True:
            event = await queue.get()
            if isinstance(event, BaseException):
                raise MCPTransportError(
                    f"SSE stream invalidated mid-call: {type(event).__name__}: {event}",
                ) from event
            if event is _STREAM_CLOSED:
                raise MCPTransportError(
                    f"SSE stream closed before response for id={request_id!r}",
                )
            if event.event != "message":
                continue
            payload = (event.data or "").strip()
            if not payload:
                continue
            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise MCPTransportError(
                    f"SSE message event was not valid JSON: {exc}",
                ) from exc
            envelope = JSONRPCResponse.model_validate(decoded)
            if envelope.id != request_id:
                continue
            return envelope


# Internal sentinel pushed onto the event queue when the SSE stream
# ends cleanly. ``_StreamClosed`` is intentionally a private type
# so consumers cannot accidentally inject an instance from outside.
class _StreamClosed:
    """Marker instance pushed onto the SSE event queue at stream end."""

    pass


_STREAM_CLOSED = _StreamClosed()


def _raise_oauth_required_if_advertised(response: httpx.Response) -> None:
    """Translate a 401 ``WWW-Authenticate: resource_metadata=…`` into ``OAuthRequiredError``."""
    header = response.headers.get("WWW-Authenticate")
    if not header:
        return
    match = _RFC9728_LINK_RE.search(header)
    if match is None:
        return
    raise OAuthRequiredError(
        metadata_url=match.group(1),
        message=(
            "the MCP server requires OAuth authorization "
            f"(advertised metadata URL: {match.group(1)})"
        ),
    )


def _resolve_post_url(base_url: str, post_url: str) -> str:
    """Resolve ``post_url`` against ``base_url`` when it is relative.

    The MCP-over-SSE spec allows the server to return either an
    absolute URL or a relative path; we resolve against the original
    SSE URL so POST calls do not depend on the SSE handler keeping
    the connection open.
    """
    if post_url.startswith(("http://", "https://")):
        return post_url
    parts = urlsplit(base_url)
    return urlunsplit((parts.scheme, parts.netloc, post_url, "", ""))


__all__ = ["SSEClient"]
