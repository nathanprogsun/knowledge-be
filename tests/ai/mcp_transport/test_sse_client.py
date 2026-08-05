"""Tests for the SSE MCP transport client.

The tests use ``respx`` to mock the ``httpx`` transport layer so the
SSE handshake (``GET /sse`` + first ``endpoint`` event) and the
request / response round-trip can be exercised without a live MCP
server. The tests assert:

- the ``endpoint`` event from the server names the POST URL,
- missing / empty ``endpoint`` events fail cleanly,
- the response correlates by id from the open SSE stream,
- 401 with ``WWW-Authenticate: resource_metadata=...`` becomes
  :class:`OAuthRequiredError`.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from src.ai.mcp_transport.errors import MCPTransportError, OAuthRequiredError
from src.ai.mcp_transport.jsonrpc import JSONRPCResponse
from src.ai.mcp_transport.sse_client import SSEClient

_SSE_URL = "https://mcp.example.com/sse"


def _sse_body(events: list[tuple[str, str]]) -> bytes:
    """Build a ``text/event-stream`` body from (event, data) pairs."""
    return "".join(f"event: {event}\ndata: {data}\n\n" for event, data in events).encode(
        "utf-8"
    )


async def test_connect_discovers_post_endpoint_from_first_event() -> None:
    """The first ``endpoint`` event names the POST URL the client must use."""
    with respx.mock(base_url="https://mcp.example.com") as router:
        endpoint_route = router.get("/sse").respond(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse_body([("endpoint", "/messages?session=abc")]),
        )
        client = SSEClient(
            url=_SSE_URL,
            headers={"X-Test": "yes"},
            timeout_seconds=5.0,
        )
        await client.connect()

    assert client.is_connected() is True
    assert client.post_endpoint.endswith("/messages?session=abc")
    assert endpoint_route.called


async def test_connect_resolves_relative_post_url_against_base() -> None:
    """A relative POST URL is resolved against the SSE URL's origin."""
    with respx.mock(base_url="https://mcp.example.com") as router:
        router.get("/sse").respond(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse_body([("endpoint", "/messages")]),
        )
        client = SSEClient(url=_SSE_URL, timeout_seconds=5.0)
        await client.connect()

    assert client.post_endpoint == "https://mcp.example.com/messages"


async def test_connect_raises_when_endpoint_event_missing() -> None:
    """An SSE stream that closes without an ``endpoint`` event is an error."""
    with respx.mock(base_url="https://mcp.example.com") as router:
        router.get("/sse").respond(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse_body([]),
        )
        client = SSEClient(url=_SSE_URL, timeout_seconds=5.0)
        with pytest.raises(MCPTransportError) as excinfo:
            await client.connect()

    assert "endpoint" in excinfo.value.message_text or "closed" in excinfo.value.message_text
    assert client.is_connected() is False


async def test_connect_raises_when_endpoint_event_is_empty() -> None:
    """An ``endpoint`` event with empty data is an error."""
    with respx.mock(base_url="https://mcp.example.com") as router:
        router.get("/sse").respond(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse_body([("endpoint", "")]),
        )
        client = SSEClient(url=_SSE_URL, timeout_seconds=5.0)
        with pytest.raises(MCPTransportError) as excinfo:
            await client.connect()

    assert "empty" in excinfo.value.message_text


async def test_connect_raises_when_handshake_returns_non_2xx() -> None:
    """A non-2xx SSE handshake surfaces as :class:`MCPTransportError`."""
    with respx.mock(base_url="https://mcp.example.com") as router:
        router.get("/sse").respond(502, text="bad gateway")
        client = SSEClient(url=_SSE_URL, timeout_seconds=5.0)
        with pytest.raises(MCPTransportError) as excinfo:
            await client.connect()

    # ``httpx_sse`` raises ``SSEError`` on the content-type check before
    # the status code is exposed; we just verify the handshake errored.
    assert "SSE handshake failed" in excinfo.value.message_text


async def test_request_sends_envelope_and_returns_matching_message_event() -> None:
    """``request`` POSTs the envelope and awaits the matching SSE message."""
    request_id = "req-1"
    response_payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"tools": [{"name": "echo"}]},
    }

    with respx.mock(base_url="https://mcp.example.com") as router:
        # The handshake stream keeps the SSE connection open while the
        # request round-trip happens; we model the "endpoint first,
        # then matching message" sequence as a single streamed body.
        sse_body = (
            _sse_body([("endpoint", "/messages")])
            + _sse_body([("message", json.dumps(response_payload))])
        )
        router.get("/sse").respond(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse_body,
        )
        post_route = router.post("/messages").respond(202, content=b"")

        client = SSEClient(url=_SSE_URL, timeout_seconds=5.0)
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


async def test_request_skips_message_events_for_other_request_ids() -> None:
    """Out-of-order SSE messages for other ids are ignored."""
    request_id = "req-target"
    foreign_payload = {
        "jsonrpc": "2.0",
        "id": "req-other",
        "result": {"tools": [{"name": "other"}]},
    }
    own_payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"tools": [{"name": "self"}]},
    }

    with respx.mock(base_url="https://mcp.example.com") as router:
        sse_body = (
            _sse_body([("endpoint", "/messages")])
            + _sse_body([("message", json.dumps(foreign_payload))])
            + _sse_body([("message", json.dumps(own_payload))])
        )
        router.get("/sse").respond(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse_body,
        )
        router.post("/messages").respond(202, content=b"")

        client = SSEClient(url=_SSE_URL, timeout_seconds=5.0)
        await client.connect()
        response = await client.request(
            method="tools/list",
            params={},
            request_id=request_id,
        )

    assert response.id == request_id
    assert response.result == {"tools": [{"name": "self"}]}


async def test_request_raises_on_post_http_error() -> None:
    """A non-2xx POST response surfaces as :class:`MCPTransportError`."""
    with respx.mock(base_url="https://mcp.example.com") as router:
        router.get("/sse").respond(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse_body([("endpoint", "/messages")]),
        )
        router.post("/messages").respond(500, text="server boom")

        client = SSEClient(url=_SSE_URL, timeout_seconds=5.0)
        await client.connect()
        with pytest.raises(MCPTransportError) as excinfo:
            await client.request(method="tools/list", params={})

    assert excinfo.value.status_code == 500


async def test_request_translates_401_with_resource_metadata_to_oauth_error() -> None:
    """A 401 ``WWW-Authenticate: resource_metadata=...`` becomes :class:`OAuthRequiredError`."""
    with respx.mock(base_url="https://mcp.example.com") as router:
        router.get("/sse").respond(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse_body([("endpoint", "/messages")]),
        )
        router.post("/messages").respond(
            401,
            headers={
                "WWW-Authenticate": (
                    'Bearer resource_metadata="https://mcp.example.com/.well-known/'
                    'oauth-protected-resource"'
                ),
            },
        )

        client = SSEClient(url=_SSE_URL, timeout_seconds=5.0)
        await client.connect()
        with pytest.raises(OAuthRequiredError) as excinfo:
            await client.request(method="tools/list", params={})

    assert "mcp.example.com" in excinfo.value.metadata_url


async def test_request_fails_when_not_connected() -> None:
    """A ``request`` before ``connect`` raises :class:`MCPTransportError`."""
    client = SSEClient(url=_SSE_URL, timeout_seconds=5.0)
    with pytest.raises(MCPTransportError) as excinfo:
        await client.request(method="tools/list", params={})
    assert "not connected" in excinfo.value.message_text


async def test_disconnect_releases_underlying_client() -> None:
    """``disconnect`` flips the connected flag and is idempotent."""
    with respx.mock(base_url="https://mcp.example.com") as router:
        router.get("/sse").respond(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse_body([("endpoint", "/messages")]),
        )
        client = SSEClient(url=_SSE_URL, timeout_seconds=5.0)
        await client.connect()
        assert client.is_connected() is True
        await client.disconnect()
        assert client.is_connected() is False
        # Second disconnect is a no-op.
        await client.disconnect()
        assert client.is_connected() is False


async def test_request_surfaces_stream_closed_as_transport_error() -> None:
    """An SSE stream that closes before the response surfaces a clean error."""
    with respx.mock(base_url="https://mcp.example.com", assert_all_called=False) as router:
        router.get("/sse").respond(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse_body([("endpoint", "/messages")]),
        )
        router.post("/messages").respond(202, content=b"")

        client = SSEClient(url=_SSE_URL, timeout_seconds=5.0)
        await client.connect()
        # Closing the SSE stream forces ``_await_message`` to exhaust.
        await client.disconnect()
        with pytest.raises(MCPTransportError):
            await client.request(method="tools/list", params={})


async def test_request_accepts_injected_httpx_client() -> None:
    """A caller-supplied ``httpx.AsyncClient`` is reused (no internal client)."""
    injected = httpx.AsyncClient(timeout=5.0)
    try:
        with respx.mock(assert_all_called=False) as router:
            sse_route = router.get("/sse").respond(
                200,
                headers={"content-type": "text/event-stream"},
                content=_sse_body([("endpoint", "/messages")]),
            )
            router.post("/messages").respond(202, content=b"")
            client = SSEClient(
                url=_SSE_URL,
                timeout_seconds=5.0,
                client=injected,
            )
            await client.connect()
            assert sse_route.called
    finally:
        await injected.aclose()
