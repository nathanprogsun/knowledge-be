"""Tests for the JSON-RPC 2.0 envelope types (PR-17.5a).

Covers the wire-shape contract every MCP message must conform to
before the SSE / HTTP-streamable clients serialise or parse it.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.ai.mcp_transport.jsonrpc import (
    INTERNAL_ERROR_CODE,
    INVALID_PARAMS_CODE,
    METHOD_INITIALIZE,
    METHOD_TOOLS_LIST,
    PARSE_ERROR_CODE,
    JSONRPCError,
    JSONRPCNotification,
    JSONRPCResponse,
    build_error_response,
    build_request,
    is_session_invalid_error,
)

# ── Build helpers ───────────────────────────────────────────────────


def test_build_request_generates_unique_ids() -> None:
    """``build_request`` produces a fresh id on each call when none is supplied."""
    seen: set[str] = set()
    for _ in range(50):
        req = build_request(method=METHOD_TOOLS_LIST, params={})
        assert req.jsonrpc == "2.0"
        assert req.method == METHOD_TOOLS_LIST
        assert req.id not in seen
        seen.add(req.id)


def test_build_request_reuses_supplied_id() -> None:
    """When the caller supplies ``request_id``, the envelope carries it verbatim."""
    req = build_request(
        method=METHOD_INITIALIZE,
        params={"protocolVersion": "2024-11-05"},
        request_id="caller-supplied",
    )
    assert req.id == "caller-supplied"


def test_build_request_rejects_empty_method() -> None:
    """``build_request`` validates the method name before serialising."""
    with pytest.raises(ValueError):
        build_request(method="", params={})


def test_build_error_response_requires_request_id() -> None:
    """An error response without an id is rejected (JSON-RPC 2.0 §5)."""
    with pytest.raises(ValueError):
        build_error_response(
            request_id="",
            code=INVALID_PARAMS_CODE,
            message="bad params",
        )


def test_build_error_response_round_trips() -> None:
    """The envelope serialises and parses back into the same shape."""
    envelope = build_error_response(
        request_id="abc",
        code=PARSE_ERROR_CODE,
        message="parse failure",
        data={"line": 3},
    )
    decoded = JSONRPCResponse.model_validate_json(envelope.model_dump_json())
    assert decoded.error is not None
    assert decoded.error.code == PARSE_ERROR_CODE
    assert decoded.error.message == "parse failure"
    assert decoded.error.data == {"line": 3}
    assert decoded.result is None


# ── Serialisation round-trips ───────────────────────────────────────


def test_request_serialises_to_json_rpc_2_0_envelope() -> None:
    """The wire shape matches the JSON-RPC 2.0 spec exactly."""
    request = build_request(method=METHOD_TOOLS_LIST, params={"limit": 5})
    raw = json.loads(request.model_dump_json())
    assert raw["jsonrpc"] == "2.0"
    assert raw["method"] == METHOD_TOOLS_LIST
    assert raw["params"] == {"limit": 5}
    assert isinstance(raw["id"], str) and raw["id"]


def test_notification_serialises_without_id() -> None:
    """Notifications do not carry an id per JSON-RPC 2.0 §4.1."""
    note = JSONRPCNotification(method="notifications/cancelled", params={"requestId": "x"})
    raw = json.loads(note.model_dump_json())
    assert raw == {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": "x"}}


def test_response_accepts_result_only() -> None:
    """A response may carry ``result`` without ``error``."""
    payload = {
        "jsonrpc": "2.0",
        "id": "1",
        "result": {"tools": [{"name": "search"}]},
    }
    envelope = JSONRPCResponse.model_validate(payload)
    assert envelope.error is None
    assert envelope.result == {"tools": [{"name": "search"}]}


def test_response_accepts_error_only() -> None:
    """A response may carry ``error`` without ``result``."""
    payload = {
        "jsonrpc": "2.0",
        "id": "1",
        "error": {"code": -32601, "message": "method not found"},
    }
    envelope = JSONRPCResponse.model_validate(payload)
    assert envelope.result is None
    assert envelope.error is not None
    assert envelope.error.code == -32601


def test_response_accepts_neither_result_nor_error_for_compatibility() -> None:
    """Both fields may be missing — JSON-RPC 2.0 §4.1 only requires ``id`` + ``jsonrpc``.

    The Go mark3labs library may produce responses with neither
    ``result`` nor ``error`` (e.g. notification-style pings that do
    not actually answer). The envelope must parse without raising;
    downstream callers check ``response.error`` and ``response.result``
    themselves.
    """
    envelope = JSONRPCResponse.model_validate({"jsonrpc": "2.0", "id": "1"})
    assert envelope.id == "1"
    assert envelope.result is None
    assert envelope.error is None


def test_request_is_frozen() -> None:
    """Frozen dataclass — mutation must raise."""
    req = build_request(method=METHOD_TOOLS_LIST, params={})
    with pytest.raises(ValidationError):
        req.method = "tools/call"


# ── Session-invalid detection ───────────────────────────────────────


@pytest.mark.parametrize(
    "code, message",
    [
        (PARSE_ERROR_CODE, "anything"),
        (INTERNAL_ERROR_CODE, "Invalid session ID from server"),
        (INTERNAL_ERROR_CODE, "no active connection for that session"),
        (INTERNAL_ERROR_CODE, "transport timeout"),
    ],
)
def test_is_session_invalid_error_detection(code: int, message: str) -> None:
    """``is_session_invalid_error`` matches Go's session-invalidation heuristic."""
    err = JSONRPCError(code=code, message=message)
    if "session" in message.lower() and (
        "invalid" in message.lower() or "no active" in message.lower()
    ):
        assert is_session_invalid_error(err) is True
    else:
        assert is_session_invalid_error(err) is False


def test_is_session_invalid_error_handles_none() -> None:
    """``None`` input is not a session invalidation."""
    assert is_session_invalid_error(None) is False
