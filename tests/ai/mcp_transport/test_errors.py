"""Tests for the MCP transport error hierarchy.

Keeps the cross-layer ``OAuthRequiredError`` ↔ ``ApplicationError``
contract pinned so the discovery + connectivity paths can use it
without a 500 leaking through.
"""

from __future__ import annotations

from src.ai.mcp_transport.errors import (
    MCPError,
    MCPTransportError,
    OAuthRequiredError,
    SessionNotConnectedError,
)
from src.common.exception import ApplicationError, UnauthorizedError


def test_mcp_error_is_a_plain_exception() -> None:
    """``MCPError`` is the transport-side base; it is intentionally
    NOT an ``ApplicationError`` so transport errors raised from
    layers below ``web`` stay decoupled from the HTTP handler."""
    err = MCPError("boom")
    assert isinstance(err, Exception)
    assert not isinstance(err, ApplicationError)


def test_oauth_required_error_is_application_error_with_401() -> None:
    """PR-17.5c H5: ``OAuthRequiredError`` inherits ``ApplicationError``
    and the exception handler resolves it to HTTP 401.

    Without this the route layer would surface a 500 when an MCP
    server's 401 leaked past the connectivity path.
    """
    err = OAuthRequiredError(metadata_url="https://idp.example.com/.well-known/oauth")
    assert isinstance(err, ApplicationError)
    assert isinstance(err, MCPError)
    assert isinstance(err, UnauthorizedError)

    # The handler's MRO walk resolves it to the explicit 401 entry.
    from src.web.exception_handler import _status_for

    assert _status_for(err) == 401


def test_transport_error_remains_outside_application_hierarchy() -> None:
    """``MCPTransportError`` stays an ``MCPError`` only — non-OAuth
    transport failures remain generic 500s unless callers wrap them."""
    err = MCPTransportError("boom", status_code=502)
    assert isinstance(err, MCPError)
    assert not isinstance(err, ApplicationError)


def test_session_not_connected_error_remains_outside_application_hierarchy() -> None:
    err = SessionNotConnectedError("session is dead")
    assert isinstance(err, MCPError)
    assert not isinstance(err, ApplicationError)


__all__ = [
    "test_mcp_error_is_a_plain_exception",
    "test_oauth_required_error_is_application_error_with_401",
    "test_session_not_connected_error_remains_outside_application_hierarchy",
    "test_transport_error_remains_outside_application_hierarchy",
]
