"""JSON-RPC 2.0 envelope types for the MCP wire protocol.

The MCP spec rides on JSON-RPC 2.0 — every request, response and
notification follows that envelope shape. This module is the single
source of truth for the request id, the request / response / error
naming, the error code namespace, and the helpers used by the SSE
client (PR-17.5a) and the HTTP-streamable client (PR-17.5b).

The Go side uses the ``github.com/mark3labs/mcp-go/mcp`` package for
the same purpose; here we keep it minimal — frozen Pydantic models
with the exact JSON field names from JSON-RPC 2.0:

- :class:`JSONRPCRequest` — ``jsonrpc`` / ``id`` / ``method`` / ``params``
- :class:`JSONRPCResponse` — ``jsonrpc`` / ``id`` / ``result`` / ``error``
- :class:`JSONRPCNotification` — ``jsonrpc`` / ``method`` / ``params``
- :class:`JSONRPCError` — ``code`` / ``message`` / ``data``
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# JSON-RPC 2.0 reserved error codes (RFC 7464 §7.1). The MCP spec
# reuses the standard codes and adds a small extension namespace
# (``-32000..-32099``).
PARSE_ERROR_CODE: int = -32700
INVALID_REQUEST_CODE: int = -32600
METHOD_NOT_FOUND_CODE: int = -32601
INVALID_PARAMS_CODE: int = -32602
INTERNAL_ERROR_CODE: int = -32603

# MCP-specific error codes (mirrors ``internal/mcp/errors.go``):
#   -32001  Resource not found
#   -32002  Tool execution rejected by the user / approval flow
#   -32003  Session no longer valid on the server side
# The connection manager uses ``-32003`` to detect session-invalidation
# errors and proactively drop the cached session; per-method errors are
# surfaced via ``JSONRPCError.data``.
MCP_SESSION_INVALID_CODE: int = -32003

# Method names from the MCP wire spec — exposed as constants so the
# connection manager and per-call helpers stay typo-proof.
METHOD_INITIALIZE: str = "initialize"
METHOD_TOOLS_LIST: str = "tools/list"
METHOD_TOOLS_CALL: str = "tools/call"
METHOD_RESOURCES_LIST: str = "resources/list"
METHOD_RESOURCES_READ: str = "resources/read"
METHOD_PROMPTS_LIST: str = "prompts/list"
METHOD_PING: str = "ping"

# Notification method namespace (``notifications/*``).
NOTIFICATION_CANCELLED: str = "notifications/cancelled"
NOTIFICATION_PROGRESS: str = "notifications/progress"
NOTIFICATION_RESOURCES_UPDATED: str = "notifications/resources/updated"
NOTIFICATION_RESOURCES_LIST_CHANGED: str = "notifications/resources/list_changed"
NOTIFICATION_TOOLS_LIST_CHANGED: str = "notifications/tools/list_changed"


class JSONRPCError(BaseModel):
    """Wire shape for one ``error`` object inside a JSON-RPC response.

    ``code`` follows the JSON-RPC 2.0 reserved range plus the MCP
    extensions; ``message`` is short, ``data`` is the optional structured
    detail (``None`` when the server omits it).
    """

    model_config = ConfigDict(frozen=True)

    code: int
    message: str
    data: dict[str, Any] | None = Field(default=None)


class JSONRPCRequest(BaseModel):
    """A JSON-RPC 2.0 request envelope (with id, awaiting response)."""

    model_config = ConfigDict(frozen=True)

    jsonrpc: str = Field(default="2.0")
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    method: str
    params: dict[str, Any] | None = Field(default=None)


class JSONRPCNotification(BaseModel):
    """A JSON-RPC 2.0 notification (no id, no response expected)."""

    model_config = ConfigDict(frozen=True)

    jsonrpc: str = Field(default="2.0")
    method: str
    params: dict[str, Any] | None = Field(default=None)


class JSONRPCResponse(BaseModel):
    """A JSON-RPC 2.0 response envelope (either ``result`` or ``error``)."""

    model_config = ConfigDict(frozen=True)

    jsonrpc: str = Field(default="2.0")
    id: str
    result: dict[str, Any] | None = Field(default=None)
    error: JSONRPCError | None = Field(default=None)


def build_request(
    *,
    method: str,
    params: dict[str, Any] | None,
    request_id: str | None = None,
) -> JSONRPCRequest:
    """Build a JSON-RPC 2.0 request envelope.

    A request id is generated on demand when not supplied; passing one
    explicitly is useful when the caller must correlate a follow-up
    response back to its initiating call (e.g. when the id is shared
    with a notification that cancels the same call).
    """
    if not method:
        raise ValueError("JSON-RPC method name must be non-empty")
    return JSONRPCRequest(
        id=request_id or uuid.uuid4().hex,
        method=method,
        params=params,
    )


def build_error_response(
    *,
    request_id: str,
    code: int,
    message: str,
    data: dict[str, Any] | None = None,
) -> JSONRPCResponse:
    """Build a JSON-RPC 2.0 error response envelope.

    Used by the transport layer when a transport-level failure
    (timeout, parse failure, server-side session invalidation) must be
    surfaced to the caller as a structured error rather than a Python
    exception.
    """
    if not request_id:
        raise ValueError("JSON-RPC error response requires a request id")
    return JSONRPCResponse(
        id=request_id,
        error=JSONRPCError(code=code, message=message, data=data),
    )


def is_session_invalid_error(error: JSONRPCError | None) -> bool:
    """True when ``error`` looks like the server-side session is gone.

    Mirrors Go's ``checkErrorAndDisconnectIfNeeded`` heuristic: the MCP
    spec uses code ``-32003`` for ``SESSION_INVALID``; the upstream
    mark3labs library additionally returns ``Invalid session ID`` /
    ``No active connection`` as plain text inside the error payload.
    """
    if error is None:
        return False
    if error.code == MCP_SESSION_INVALID_CODE:
        return True
    needle = error.message.lower()
    return "invalid session id" in needle or "no active connection" in needle


__all__ = [
    "INTERNAL_ERROR_CODE",
    "INVALID_PARAMS_CODE",
    "INVALID_REQUEST_CODE",
    "MCP_SESSION_INVALID_CODE",
    "METHOD_INITIALIZE",
    "METHOD_NOT_FOUND_CODE",
    "METHOD_PING",
    "METHOD_PROMPTS_LIST",
    "METHOD_RESOURCES_LIST",
    "METHOD_RESOURCES_READ",
    "METHOD_TOOLS_CALL",
    "METHOD_TOOLS_LIST",
    "NOTIFICATION_CANCELLED",
    "NOTIFICATION_PROGRESS",
    "NOTIFICATION_RESOURCES_LIST_CHANGED",
    "NOTIFICATION_RESOURCES_UPDATED",
    "NOTIFICATION_TOOLS_LIST_CHANGED",
    "PARSE_ERROR_CODE",
    "JSONRPCError",
    "JSONRPCNotification",
    "JSONRPCRequest",
    "JSONRPCResponse",
    "build_error_response",
    "build_request",
    "is_session_invalid_error",
]
