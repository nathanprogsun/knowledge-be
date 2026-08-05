"""MCP live transport layer — JSON-RPC 2.0 over SSE / HTTP-streamable.

PR-17.5 scope (17.5a + 17.5b):

- :mod:`src.ai.mcp_transport.jsonrpc` — JSON-RPC 2.0 envelope types
  (``JSONRPCRequest`` / ``JSONRPCResponse`` / ``JSONRPCError`` /
  ``JSONRPCNotification``) + reserved error code constants.
- :mod:`src.ai.mcp_transport.sse_client` — long-lived ``GET /sse`` +
  ``POST /messages`` transport on top of ``httpx`` + ``httpx_sse``.
- :mod:`src.ai.mcp_transport.http_streamable_client` — single ``POST``
  endpoint transport that answers either a JSON body or an SSE
  stream.
- :mod:`src.ai.mcp_transport.connection_manager` — pooled sessions
  per remote service, background sweep, session-invalidation
  detection.
- :mod:`src.core.infra.mcp_services.oauth` — full OAuth 2.0 lifecycle:
  authorize URL, code exchange, refresh, revoke, in-memory secret
  store + CSRF state store.
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
from src.ai.mcp_transport.http_streamable_client import HTTPStreamableClient
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
    "HTTPStreamableClient",
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
