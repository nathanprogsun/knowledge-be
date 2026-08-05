"""MCP live transport layer — JSON-RPC 2.0 over SSE.

- :mod:`src.ai.mcp_transport.jsonrpc` — JSON-RPC 2.0 envelope types
  (``JSONRPCRequest`` / ``JSONRPCResponse`` / ``JSONRPCError`` /
  ``JSONRPCNotification``) + reserved error code constants.
- :mod:`src.ai.mcp_transport.sse_client` — long-lived ``GET /sse`` +
  ``POST /messages`` transport on top of ``httpx`` + ``httpx_sse``.
- :mod:`src.ai.mcp_transport.connection_manager` — pooled sessions
  per remote service, background sweep, session-invalidation
  detection.

The HTTP-streamable transport and the OAuth lifecycle are not yet
implemented; the manager's default factory raises for
``http-streamable`` so callers do not silently fall back.
"""

from __future__ import annotations

from src.ai.mcp_transport.connection_manager import (
    MCPConnectionManager,
    MCPSession,
)
from src.ai.mcp_transport.errors import (
    MCPError,
    MCPTransportError,
    OAuthRequiredError,
    SessionNotConnectedError,
)
from src.ai.mcp_transport.jsonrpc import (
    JSONRPCError,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    build_error_response,
    build_request,
    is_session_invalid_error,
)
from src.ai.mcp_transport.sse_client import SSEClient

__all__ = [
    "JSONRPCError",
    "JSONRPCNotification",
    "JSONRPCRequest",
    "JSONRPCResponse",
    "MCPConnectionManager",
    "MCPError",
    "MCPSession",
    "MCPTransportError",
    "OAuthRequiredError",
    "SSEClient",
    "SessionNotConnectedError",
    "build_error_response",
    "build_request",
    "is_session_invalid_error",
]
