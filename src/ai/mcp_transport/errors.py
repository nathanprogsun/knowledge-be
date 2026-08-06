"""Error types raised by the MCP transport layer.

Standalone module so the connection manager and individual transport
clients can share exception types without risking a circular import.

The same three exception families (transport-level failure,
OAuth-required, session no-longer-connected) surface on the Python side.

Hierarchy: ``MCPError`` inherits :class:`src.common.exception.ExternalServiceError`
so the web layer's exception handler maps it to HTTP 502 — the right
status for "we called an upstream service and it failed" — without
each subclass having to opt in. :class:`OAuthRequiredError` keeps
:class:`UnauthorizedError` in its MRO so it still maps to HTTP 401 when
the server-side auth challenge is the headline failure.
"""

from __future__ import annotations

from src.common.exception import ExternalServiceError, UnauthorizedError


class MCPError(ExternalServiceError):
    """Base class for every MCP transport error.

    Inherits :class:`ExternalServiceError` so the web-layer exception
    handler resolves ``_STATUS_BY_TYPE[MCPError]`` to HTTP 502 via the
    MRO walk in :mod:`src.web.exception_handler`. ``code`` /
    ``message`` are inherited but explicitly set so the wire-side error
    payload carries the canonical "mcp_service.transport_error" code.
    """

    code: str = "mcp_service.transport_error"
    message: str = "MCP transport error"


class MCPTransportError(MCPError):
    """Wire-level transport failure that does not require re-authorization.

    The remote server returned a structured JSON-RPC error, the
    transport itself raised (network failure, parse failure, timeout),
    or the underlying HTTP call returned a non-2xx status without the
    RFC 9728 ``resource_metadata`` advertisement that would translate
    the failure into :class:`OAuthRequiredError`.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message_text = message
        self.status_code = status_code
        self.body = body


class OAuthRequiredError(UnauthorizedError, MCPError):
    """The remote MCP server demands OAuth authorization.

    Mirrors Go's ``OAuthRequiredError``. Raised when the
    ``WWW-Authenticate`` header advertises an RFC 9728
    ``resource_metadata`` URL the service was not configured to use;
    the UI surfaces this so the user can switch auth strategy rather
    than see a generic 401.

    Inherits :class:`src.common.exception.UnauthorizedError` (which
    maps to HTTP 401 via the standard exception handler) and
    :class:`MCPError` so the existing ``except MCPError`` clauses in
    the discovery + connectivity paths continue to match.

    The ``code`` defaults to ``UnauthorizedError.code`` ("unauthorized")
    rather than ``MCPError.code`` ("mcp_service.transport_error") so
    the API response carries the more specific signal.
    """

    def __init__(self, *, metadata_url: str, message: str | None = None) -> None:
        text = message or f"MCP server requires OAuth (resource metadata: {metadata_url})"
        # Initialize both bases so MRO walks see ``UnauthorizedError``
        # code/message and ``MCPError`` is satisfied for isinstance.
        UnauthorizedError.__init__(self, text)
        MCPError.__init__(self, text)
        # ``UnauthorizedError`` is the more specific code, so restore it
        # after ``MCPError.__init__`` set ``code`` to its default.
        self.code = UnauthorizedError.code
        self.metadata_url = metadata_url


class SessionNotConnectedError(MCPError):
    """A call was attempted against a closed or evicted session."""


__all__ = [
    "MCPError",
    "MCPTransportError",
    "OAuthRequiredError",
    "SessionNotConnectedError",
]
