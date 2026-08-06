"""Tests for the MCP transport error hierarchy.

Keeps the cross-layer ``MCPError`` ↔ ``ApplicationError`` ↔
``OAuthRequiredError`` contract pinned so the discovery + connectivity
paths can use it without a 500 leaking through.
"""

from __future__ import annotations

from src.ai.mcp_transport.errors import (
    MCPError,
    MCPTransportError,
    OAuthRequiredError,
    SessionNotConnectedError,
)
from src.common.exception import (
    ApplicationError,
    ExternalServiceError,
    UnauthorizedError,
)


def test_mcp_error_is_an_external_service_error() -> None:
    """``MCPError`` inherits ``ExternalServiceError`` so the web
    handler maps it to HTTP 502 via the standard MRO walk."""
    err = MCPError("boom")
    assert isinstance(err, Exception)
    assert isinstance(err, ApplicationError)
    assert isinstance(err, ExternalServiceError)
    assert err.code == "mcp_service.transport_error"


def test_oauth_required_error_is_application_error_with_401() -> None:
    """``OAuthRequiredError`` inherits ``ApplicationError``
    and the exception handler resolves it to HTTP 401.

    Without this the route layer would surface a 500 when an MCP
    server's 401 leaked past the connectivity path.
    """
    err = OAuthRequiredError(metadata_url="https://idp.example.com/.well-known/oauth")
    assert isinstance(err, ApplicationError)
    assert isinstance(err, MCPError)
    assert isinstance(err, UnauthorizedError)
    # The OAuth-specific code wins over the MCP transport default so
    # the API response advertises the auth challenge, not transport.
    assert err.code == UnauthorizedError.code

    # The handler's MRO walk resolves it to the explicit 401 entry.
    from src.web.exception_handler import _status_for

    assert _status_for(err) == 401


def test_transport_error_maps_to_external_service_error() -> None:
    """``MCPTransportError`` inherits ``ExternalServiceError`` so the
    web handler maps it to HTTP 502 (the upstream-call-failed
    status)."""
    err = MCPTransportError("boom", status_code=502)
    assert isinstance(err, MCPError)
    assert isinstance(err, ExternalServiceError)

    from src.web.exception_handler import _status_for

    assert _status_for(err) == 502


def test_session_not_connected_error_is_an_application_error() -> None:
    """``SessionNotConnectedError`` also inherits ``MCPError`` →
    ``ExternalServiceError``."""
    err = SessionNotConnectedError("session is dead")
    assert isinstance(err, MCPError)
    assert isinstance(err, ApplicationError)


__all__ = [
    "test_mcp_error_is_an_external_service_error",
    "test_oauth_required_error_is_application_error_with_401",
    "test_session_not_connected_error_is_an_application_error",
    "test_transport_error_maps_to_external_service_error",
]
