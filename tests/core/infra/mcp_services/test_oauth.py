"""Unit tests for the MCP service OAuth flow."""

from __future__ import annotations

from datetime import UTC, datetime

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
