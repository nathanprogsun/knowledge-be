"""Tests for the HTTP-streamable MCP transport.

Mirrors ``tests/ai/mcp_transport/test_sse_client.py`` — ``respx``
mocks the ``httpx`` transport layer so the synchronous POST + JSON
response round-trip, the POST + SSE response round-trip, the
``Mcp-Session-Id`` header round-tripping, and the 401-with-OAuth-
challenge translation can be exercised without a live MCP server.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from src.ai.mcp_transport.errors import MCPTransportError, OAuthRequiredError
from src.ai.mcp_transport.http_streamable_client import HTTPStreamableClient
from src.ai.mcp_transport.jsonrpc import JSONRPCResponse

_STREAMABLE_URL = "https://mcp.example.com/mcp"


def _sse_body(events: list[tuple[str, str]]) -> bytes:
    """Build a ``text/event-stream`` body from (event, data) pairs."""
    return "".join(f"event: {event}\ndata: {data}\n\n" for event, data in events).encode("utf-8")


# ── Lifecycle ────────────────────────────────────────────────────────


async def test_connect_marks_client_ready_without_network_call() -> None:
    """HTTP-streamable is connectionless; ``connect`` only flips a flag."""
    client = HTTPStreamableClient(url=_STREAMABLE_URL, timeout_seconds=5.0)
    assert client.is_connected() is False
    await client.connect()
    assert client.is_connected() is True
    await client.disconnect()
    assert client.is_connected() is False


async def test_disconnect_releases_owned_client() -> None:
    """``disconnect`` closes the internal ``httpx.AsyncClient``."""
    client = HTTPStreamableClient(url=_STREAMABLE_URL, timeout_seconds=5.0)
    await client.connect()
    await client.disconnect()
    # Second disconnect is a no-op.
    await client.disconnect()
    assert client.is_connected() is False


async def test_request_without_connect_raises_transport_error() -> None:
    """A ``request`` before ``connect`` raises :class:`MCPTransportError`."""
    client = HTTPStreamableClient(url=_STREAMABLE_URL, timeout_seconds=5.0)
    with pytest.raises(MCPTransportError) as excinfo:
        await client.request(method="tools/list", params={})
    assert "not connected" in excinfo.value.message_text


# ── Synchronous JSON response ────────────────────────────────────────


async def test_request_posts_envelope_and_decodes_json_response() -> None:
    """The request POST carries the JSON-RPC envelope; the response is JSON."""
    request_id = "req-1"
    response_payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"tools": [{"name": "echo"}]},
    }

    with respx.mock(base_url="https://mcp.example.com") as router:
        post_route = router.post("/mcp").respond(
            200,
            headers={"content-type": "application/json"},
            json=response_payload,
        )
        client = HTTPStreamableClient(
            url=_STREAMABLE_URL,
            headers={"X-Test": "yes"},
            timeout_seconds=5.0,
        )
        await client.connect()
        response = await client.request(
            method="tools/list",
            params={},
            request_id=request_id,
        )

    assert isinstance(response, JSONRPCResponse)
    assert response.error is None
    assert response.result == {"tools": [{"name": "echo"}]}
    assert post_route.called


async def test_request_captures_session_id_from_response_header() -> None:
    """The ``Mcp-Session-Id`` round-trips onto subsequent requests."""
    init_response = httpx.Response(
        200,
        headers={
            "content-type": "application/json",
            "Mcp-Session-Id": "srv-abc",
        },
        json={"jsonrpc": "2.0", "id": "i", "result": {}},
    )
    tools_response = httpx.Response(
        200,
        headers={"content-type": "application/json"},
        json={"jsonrpc": "2.0", "id": "t", "result": {"tools": []}},
    )

    with respx.mock(assert_all_called=False) as router:
        post_route = router.post("/mcp").mock(
            side_effect=[init_response, tools_response],
        )

        client = HTTPStreamableClient(url=_STREAMABLE_URL, timeout_seconds=5.0)
        await client.connect()
        await client.request(method="initialize", params={}, request_id="i")
        assert client.session_id == "srv-abc"
        await client.request(method="tools/list", params={}, request_id="t")

    assert post_route.call_count == 2
    # The second request sends the captured session id.
    headers = post_route.calls.last.request.headers
    assert headers.get("Mcp-Session-Id") == "srv-abc"


# ── Streaming SSE response ───────────────────────────────────────────


async def test_request_decodes_message_event_with_matching_id_from_sse() -> None:
    """An SSE response with ``message`` events yields the matching envelope."""
    request_id = "req-target"
    other_payload = {
        "jsonrpc": "2.0",
        "id": "req-other",
        "result": {"tools": [{"name": "other"}]},
    }
    target_payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"tools": [{"name": "target"}]},
    }

    with respx.mock(base_url="https://mcp.example.com") as router:
        router.post("/mcp").respond(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                _sse_body([("message", json.dumps(other_payload))])
                + _sse_body([("message", json.dumps(target_payload))])
            ),
        )

        client = HTTPStreamableClient(url=_STREAMABLE_URL, timeout_seconds=5.0)
        await client.connect()
        response = await client.request(
            method="tools/list",
            params={},
            request_id=request_id,
        )

    assert response.id == request_id
    assert response.result == {"tools": [{"name": "target"}]}


async def test_request_raises_when_sse_response_closes_without_match() -> None:
    """An SSE stream that closes before the matching id surfaces as :class:`MCPTransportError`."""
    with respx.mock(base_url="https://mcp.example.com") as router:
        router.post("/mcp").respond(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse_body([("message", json.dumps({"id": "missing"}))]),
        )

        client = HTTPStreamableClient(url=_STREAMABLE_URL, timeout_seconds=5.0)
        await client.connect()
        with pytest.raises(MCPTransportError) as excinfo:
            await client.request(method="tools/list", params={}, request_id="req-1")

    assert "closed before response" in excinfo.value.message_text


# ── Error translation ────────────────────────────────────────────────


async def test_request_translates_401_with_resource_metadata_to_oauth_error() -> None:
    """A 401 carrying ``WWW-Authenticate: resource_metadata=`` becomes :class:`OAuthRequiredError`."""
    with respx.mock(base_url="https://mcp.example.com") as router:
        router.post("/mcp").respond(
            401,
            headers={
                "WWW-Authenticate": (
                    'Bearer resource_metadata="https://mcp.example.com/'
                    '.well-known/oauth-protected-resource"'
                ),
            },
        )

        client = HTTPStreamableClient(url=_STREAMABLE_URL, timeout_seconds=5.0)
        await client.connect()
        with pytest.raises(OAuthRequiredError) as excinfo:
            await client.request(method="tools/list", params={})

    assert "mcp.example.com" in excinfo.value.metadata_url


async def test_request_raises_on_non_2xx_json_response() -> None:
    """A non-2xx JSON response surfaces as :class:`MCPTransportError`."""
    with respx.mock(base_url="https://mcp.example.com") as router:
        router.post("/mcp").respond(500, text="server boom")

        client = HTTPStreamableClient(url=_STREAMABLE_URL, timeout_seconds=5.0)
        await client.connect()
        with pytest.raises(MCPTransportError) as excinfo:
            await client.request(method="tools/list", params={})

    assert excinfo.value.status_code == 500


async def test_request_raises_on_malformed_json_body() -> None:
    """A 2xx with a non-JSON body surfaces as :class:`MCPTransportError`."""
    with respx.mock(base_url="https://mcp.example.com") as router:
        router.post("/mcp").respond(
            200,
            headers={"content-type": "application/json"},
            content=b"not-json-at-all",
        )

        client = HTTPStreamableClient(url=_STREAMABLE_URL, timeout_seconds=5.0)
        await client.connect()
        with pytest.raises(MCPTransportError) as excinfo:
            await client.request(method="tools/list", params={})

    assert "JSON" in excinfo.value.message_text


async def test_request_rejects_response_with_mismatched_id() -> None:
    """A JSON response whose id does not match the request id is rejected."""
    with respx.mock(base_url="https://mcp.example.com") as router:
        router.post("/mcp").respond(
            200,
            headers={"content-type": "application/json"},
            json={"jsonrpc": "2.0", "id": "different", "result": {}},
        )

        client = HTTPStreamableClient(url=_STREAMABLE_URL, timeout_seconds=5.0)
        await client.connect()
        with pytest.raises(MCPTransportError) as excinfo:
            await client.request(method="tools/list", params={}, request_id="self")

    assert "id" in excinfo.value.message_text.lower()


# ── Client injection ────────────────────────────────────────────────


async def test_request_accepts_injected_httpx_client() -> None:
    """A caller-supplied ``httpx.AsyncClient`` is reused (no internal client)."""
    injected = httpx.AsyncClient(timeout=5.0)
    try:
        with respx.mock(assert_all_called=False) as router:
            route = router.post("/mcp").respond(
                200,
                headers={"content-type": "application/json"},
                json={"jsonrpc": "2.0", "id": "i", "result": {}},
            )
            client = HTTPStreamableClient(
                url=_STREAMABLE_URL,
                timeout_seconds=5.0,
                client=injected,
            )
            await client.connect()
            await client.request(method="initialize", params={}, request_id="i")
            assert route.called
    finally:
        await injected.aclose()


async def test_post_sends_content_type_application_json() -> None:
    """The POST carries ``Content-Type: application/json``.

    Without this header some MCP servers reject the request with a
    415 because ``httpx.post(content=...)`` does not set the type
    automatically.
    """
    with respx.mock(base_url="https://mcp.example.com") as router:
        route = router.post("/mcp").respond(
            200,
            headers={"content-type": "application/json"},
            json={"jsonrpc": "2.0", "id": "1", "result": {}},
        )
        client = HTTPStreamableClient(url=_STREAMABLE_URL, timeout_seconds=5.0)
        await client.connect()
        await client.request(method="initialize", params={}, request_id="1")

    headers = route.calls.last.request.headers
    assert headers.get("Content-Type") == "application/json"


async def test_request_wraps_httpx_connect_error_as_transport_error() -> None:
    """An ``httpx.ConnectError`` becomes ``MCPTransportError``."""
    import httpx as _httpx

    with respx.mock(base_url="https://mcp.example.com") as router:
        router.post("/mcp").mock(
            side_effect=_httpx.ConnectError("connection refused"),
        )
        client = HTTPStreamableClient(url=_STREAMABLE_URL, timeout_seconds=5.0)
        await client.connect()
        with pytest.raises(MCPTransportError) as excinfo:
            await client.request(method="tools/list", params={})

    assert "ConnectError" in excinfo.value.message_text


async def test_request_wraps_httpx_timeout_exception_as_transport_error() -> None:
    """An ``httpx.TimeoutException`` becomes ``MCPTransportError``."""
    import httpx as _httpx

    with respx.mock(base_url="https://mcp.example.com") as router:
        router.post("/mcp").mock(
            side_effect=_httpx.TimeoutException("read timed out"),
        )
        client = HTTPStreamableClient(url=_STREAMABLE_URL, timeout_seconds=5.0)
        await client.connect()
        with pytest.raises(MCPTransportError) as excinfo:
            await client.request(method="tools/list", params={})

    assert "TimeoutException" in excinfo.value.message_text


async def test_403_with_resource_metadata_becomes_oauth_required() -> None:
    """A 403 carrying ``WWW-Authenticate: resource_metadata``
    becomes :class:`OAuthRequiredError` (same path as 401)."""
    with respx.mock(base_url="https://mcp.example.com") as router:
        router.post("/mcp").respond(
            403,
            headers={
                "WWW-Authenticate": (
                    'Bearer resource_metadata="https://mcp.example.com/'
                    '.well-known/oauth-protected-resource"'
                ),
            },
        )
        client = HTTPStreamableClient(url=_STREAMABLE_URL, timeout_seconds=5.0)
        await client.connect()
        with pytest.raises(OAuthRequiredError) as excinfo:
            await client.request(method="tools/list", params={})

    assert "mcp.example.com" in excinfo.value.metadata_url
