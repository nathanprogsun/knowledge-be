"""Unit tests for the MCP service OAuth flow."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from src.common.exception import NotFoundError, ValidationError
from src.common.json import JsonObject
from src.core.infra.mcp_services.oauth import OAuthManager
from src.core.infra.mcp_services.types import MCPServiceInfo


def _now() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def _info(auth_config: JsonObject | None = None) -> MCPServiceInfo:
    return MCPServiceInfo(
        id="svc-1",
        tenant_id=1,
        name="acme",
        transport_type="sse",
        auth_config=auth_config,
        created_at=_now(),
        updated_at=_now(),
    )


def test_manager_requires_auth_config_for_authorization() -> None:
    manager = OAuthManager(service=_info(auth_config=None))
    with pytest.raises(ValidationError) as excinfo:
        manager.start_authorization(
            redirect_uri="https://example.com/oauth/callback",
            frontend_redirect="/",
            user_id="alice",
        )
    assert excinfo.value.code == "mcp_service.oauth_not_configured"


def test_manager_requires_redirect_uri_for_authorization() -> None:
    manager = OAuthManager(
        service=_info(auth_config={"auth_type": "oauth", "scopes": ["read"]}),
    )
    with pytest.raises(ValidationError) as excinfo:
        manager.start_authorization(
            redirect_uri="",
            frontend_redirect="/",
            user_id="alice",
        )
    assert excinfo.value.code == "mcp_service.redirect_uri_required"


def test_manager_returns_authorize_url_when_oauth_configured() -> None:
    manager = OAuthManager(
        service=_info(auth_config={"auth_type": "oauth", "scopes": ["read"]}),
    )
    outcome = manager.start_authorization(
        redirect_uri="https://example.com/oauth/callback",
        frontend_redirect="/",
        user_id="alice",
    )
    assert outcome.authorization_url.startswith("https://example.com/oauth/callback")
    assert outcome.authorization_attempt


def test_manager_status_raises_for_missing_user() -> None:
    manager = OAuthManager(
        service=_info(auth_config={"auth_type": "oauth"}),
    )
    with pytest.raises(NotFoundError) as excinfo:
        manager.authorization_status(user_id="")
    assert excinfo.value.code == "mcp_service.user_missing"


def test_manager_status_default_is_pending() -> None:
    manager = OAuthManager(
        service=_info(auth_config={"auth_type": "oauth"}),
    )
    status = manager.authorization_status(user_id="alice")
    assert status.authorized is False
    assert status.state == "pending"


def test_manager_revoke_raises_for_missing_user() -> None:
    manager = OAuthManager(
        service=_info(auth_config={"auth_type": "oauth"}),
    )
    with pytest.raises(NotFoundError) as excinfo:
        manager.revoke(user_id="")
    assert excinfo.value.code == "mcp_service.user_missing"


def test_manager_revoke_is_a_noop_for_valid_user() -> None:
    manager = OAuthManager(
        service=_info(auth_config={"auth_type": "oauth"}),
    )
    manager.revoke(user_id="alice")  # returns None implicitly


def test_legacy_oauth_manager_uses_injected_http_client_not_creates_new_one() -> None:
    """When ``http_client`` is injected the manager does NOT
    create a fresh ``httpx.AsyncClient`` (one per request leak).

    The legacy router path used to construct ``OAuthManager(service=info)``
    which auto-created a new client when ``auth_config`` was OAuth — that
    leaked one TCP/TLS connection per request and was never closed.
    Lifespan-wired factories now inject the shared http_client; this
    test pins both the "no auto-creation" path and the "owns=False"
    flag the ``aclose`` semantics depend on.
    """
    auth_config: JsonObject = {
        "auth_type": "oauth",
        "authorization_endpoint": "https://idp.example.com/authorize",
        "token_endpoint": "https://idp.example.com/token",
        "client_id": "client-abc",
    }
    injected = httpx.AsyncClient(timeout=10.0)
    manager = OAuthManager(
        service=_info(auth_config=auth_config),
        http_client=injected,
    )
    # The injected client is the one wired in; the manager does not
    # own it (so ``aclose`` will leave it alone).
    assert manager._http_client is injected
    assert manager._owns_http_client is False
