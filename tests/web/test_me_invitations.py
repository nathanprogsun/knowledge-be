"""Web-layer tests for the ``/me/invitations`` inbox."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.app_context import request_context
from src.core.tenants.types import MembershipInfo, TenantInvitationInfo
from src.web.api.me.router import router as me_router
from src.web.deps.tenants import get_tenant_invitation_service
from src.web.exception_handler import register_exception_handlers
from src.web.middleware.auth import require_auth

_NOW = datetime(2026, 9, 5, tzinfo=UTC)


def _invitation() -> TenantInvitationInfo:
    return TenantInvitationInfo(
        id=3,
        tenant_id=9,
        invitee_user_id="u-1",
        invited_by="owner",
        role="viewer",
        status="pending",
        message=None,
        expires_at=_NOW + timedelta(days=2),
        responded_at=None,
        accepted_count=0,
        is_share_link=False,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _membership() -> MembershipInfo:
    return MembershipInfo(
        id=1,
        user_id="u-1",
        tenant_id=9,
        role="viewer",
        status="active",
        invited_by="owner",
        joined_at=_NOW,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _build_app(invitation_service: AsyncMock) -> FastAPI:
    async def _noop_auth() -> None:
        request_context.set_user_id("u-1")
        request_context.set_tenant_id("7")

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(me_router, prefix="/api/v1")
    app.dependency_overrides[require_auth] = _noop_auth
    app.dependency_overrides[get_tenant_invitation_service] = lambda: invitation_service
    return app


def test_list_my_invitations() -> None:
    fake = AsyncMock()
    fake.list_by_invitee = AsyncMock(return_value=[_invitation()])
    app = _build_app(fake)

    with TestClient(app) as client:
        response = client.get("/api/v1/me/invitations")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["total"] == 1
    assert body["data"]["invitations"][0]["id"] == 3
    fake.list_by_invitee.assert_awaited_once_with("u-1", include_terminal=False)


def test_accept_my_invitation() -> None:
    fake = AsyncMock()
    fake.accept = AsyncMock(return_value=_membership())
    app = _build_app(fake)

    with TestClient(app) as client:
        response = client.post("/api/v1/me/invitations/3/accept")

    assert response.status_code == 200
    membership = response.json()["data"]["membership"]
    assert membership["tenant_id"] == 9
    assert membership["role"] == "viewer"
    fake.accept.assert_awaited_once_with(3, user_id="u-1")


def test_decline_my_invitation() -> None:
    fake = AsyncMock()
    fake.decline = AsyncMock(return_value=None)
    app = _build_app(fake)

    with TestClient(app) as client:
        response = client.post("/api/v1/me/invitations/3/decline")

    assert response.status_code == 200
    assert response.json() == {"success": True, "message": "Invitation declined"}
    fake.decline.assert_awaited_once_with(3, user_id="u-1")
