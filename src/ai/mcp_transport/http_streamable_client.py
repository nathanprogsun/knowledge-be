"""HTTP-streamable transport for MCP — single POST endpoint per request.

Mirrors the ``http-streamable`` branch in ``internal/mcp/client.go`` (the
Go side delegates to ``mark3labs/mcp-go``'s streamable-HTTP client). The
remote MCP server exposes one URL; every JSON-RPC request is a single
``POST`` to that URL whose response is either:

- a JSON body (``application/json``) when the server can answer
  synchronously (``initialize`` / ``tools/list`` / small ``tools/call``),
- an SSE stream (``text/event-stream``) when the response is large or
  the server decided to stream back. Each ``message`` SSE event carries
  one JSON-RPC envelope; the response with the matching id is the one
  the caller awaits.

Notifications and ``Mcp-Session-Id`` round-tripping follow the MCP HTTP
transport spec: the server assigns a session id on ``initialize`` and
the client forwards the header on every subsequent call so the server
can look the session back up.

This module is wire-only: it does not run a cleanup loop, does not pool
sessions, and does not translate transport errors into MCP domain
errors. Those concerns live in
:mod:`src.ai.mcp_transport.connection_manager`.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from src.ai.mcp_transport.errors import MCPTransportError, OAuthRequiredError
from src.ai.mcp_transport.jsonrpc import JSONRPCResponse, build_request
from src.ai.mcp_transport.sse_client import (
    _raise_oauth_required_if_advertised,
    _resolve_post_url,
)

# The MCP HTTP-streamable transport lets the server choose between a JSON
# response and an SSE response by inspecting the request ``Accept``
# header. Sending the wildcard forces the server to negotiate per
# request rather than assume the client wants one shape or the other.
_ACCEPT_HEADER = "application/json, text/event-stream"
_SESSION_HEADER = "Mcp-Session-Id"
_CONTENT_TYPE_JSON = "application/json"


class HTTPStreamableClient:
    """One MCP-over-HTTP-streamable client.

    Constructor is intentionally side-effect free. Call :meth:`connect`
    first (records state; the HTTP client is created lazily) and
    :meth:`disconnect` to release the underlying ``httpx.AsyncClient``.
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
        self._connected = False
        self._session_id: str = ""

    # ── Lifecycle ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Record the client as ready.

        HTTP-streamable is connectionless from the transport's point of
        view (each request carries its own state in the ``Mcp-Session-Id``
        header). We still expose a ``connect`` / ``disconnect`` pair so
        the connection-manager protocol can treat this client the same
        way it treats :class:`SSEClient`.
        """
        if self._connected:
            return
        self._ensure_client()
        self._connected = True

    async def disconnect(self) -> None:
        """Release the underlying client and clear the session id."""
        self._connected = False
        self._session_id = ""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def is_connected(self) -> bool:
        """Return whether :meth:`connect` has been called."""
        return self._connected

    @property
    def session_id(self) -> str:
        """Return the latest ``Mcp-Session-Id`` the server advertised."""
        return self._session_id

    # ── Request flow ───────────────────────────────────────────────

    async def request(
        self,
        *,
        method: str,
        params: dict[str, Any] | None = None,
        request_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> JSONRPCResponse:
        """Send one JSON-RPC request via POST and return the response.

        Mirrors Go's ``mcpGoClient`` behaviour for the
        ``http-streamable`` transport: a single ``POST`` to the
        configured URL carries the envelope; the response is either a
        JSON body or an SSE stream whose ``message`` events carry the
        JSON-RPC reply.
        """
        if not self._connected:
            raise MCPTransportError("HTTP-streamable client is not connected")
        envelope = build_request(
            method=method,
            params=params,
            request_id=request_id,
        )
        client = self._ensure_client()
        request_headers = self._build_headers()
        try:
            response = await client.post(
                self._url,
                content=envelope.model_dump_json(),
                headers=request_headers,
                timeout=timeout_seconds if timeout_seconds is not None else self._timeout,
            )
        except httpx.HTTPError as exc:
            raise MCPTransportError(
                f"HTTP-streamable POST failed: {type(exc).__name__}: {exc}",
            ) from exc

        self._ingest_session_header(response)
        if response.status_code in {401, 403}:
            _raise_oauth_required_if_advertised(response)
        if response.status_code >= 400:
            raise MCPTransportError(
                f"HTTP-streamable POST returned HTTP {response.status_code}",
                status_code=response.status_code,
                body=response.text,
            )

        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type.lower():
            return await _read_sse_response(response, envelope.id)
        return _read_json_response(response, envelope.id)

    # ── Internals ──────────────────────────────────────────────────

    def _ensure_client(self) -> httpx.AsyncClient:
        """Return the shared ``httpx.AsyncClient`` (lazily creating one)."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    def _build_headers(self) -> dict[str, str]:
        """Compose the request headers (user headers + Accept + session id)."""
        merged = dict(self._headers)
        merged.setdefault("Accept", _ACCEPT_HEADER)
        if self._session_id:
            merged[_SESSION_HEADER] = self._session_id
        return merged

    def _ingest_session_header(self, response: httpx.Response) -> None:
        """Capture the ``Mcp-Session-Id`` the server may have assigned."""
        session_id = response.headers.get(_SESSION_HEADER)
        if session_id:
            self._session_id = session_id


# ── Helpers ──────────────────────────────────────────────────────────


def _read_json_response(response: httpx.Response, request_id: str) -> JSONRPCResponse:
    """Decode a non-streaming JSON body into a :class:`JSONRPCResponse`."""
    try:
        decoded = response.json()
    except json.JSONDecodeError as exc:
        raise MCPTransportError(
            f"HTTP-streamable response was not valid JSON: {exc}",
        ) from exc
    if not isinstance(decoded, dict):
        raise MCPTransportError(
            "HTTP-streamable JSON response must be a JSON object",
        )
    envelope = JSONRPCResponse.model_validate(decoded)
    if envelope.id != request_id:
        raise MCPTransportError(
            "HTTP-streamable response id does not match the request id",
        )
    return envelope


async def _read_sse_response(
    response: httpx.Response,
    request_id: str,
) -> JSONRPCResponse:
    """Parse an SSE response body for the ``message`` event whose id matches."""
    async for line in response.aiter_lines():
        if not line:
            continue
        if line.startswith("event: "):
            event_name = line[len("event: ") :].strip()
            if event_name != "message":
                continue
            continue
        if line.startswith("data: "):
            payload = line[len("data: ") :].strip()
            if not payload:
                continue
            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise MCPTransportError(
                    f"HTTP-streamable SSE payload was not valid JSON: {exc}",
                ) from exc
            envelope = JSONRPCResponse.model_validate(decoded)
            if envelope.id != request_id:
                continue
            return envelope
    raise MCPTransportError(
        f"HTTP-streamable SSE stream closed before response for id={request_id!r}",
    )


def resolve_post_url(base_url: str, post_url: str) -> str:
    """Public re-export of :func:`src.ai.mcp_transport.sse_client._resolve_post_url`.

    The HTTP-streamable client reuses the resolve helper from the SSE
    module; we re-export it here so callers do not need to reach into
    a "private" helper they did not import.
    """
    return _resolve_post_url(base_url, post_url)


__all__ = [
    "HTTPStreamableClient",
    "OAuthRequiredError",
    "resolve_post_url",
]
