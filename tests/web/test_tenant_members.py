"""Web-layer tests for workspace members, leave, invitations, and invite-links."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.params import Depends
from fastapi.testclient import TestClient

from src.app_context import request_context
from src.common.exception import ConflictError
from src.core.tenants.types import MembershipInfo, TenantInvitationInfo
from src.web.api.tenants.router import router as tenants_router
from src.web.deps import RoleOwnerDep, RoleViewerDep
from src.web.deps.auth import get_auth_service
from src.web.deps.rbac import PathTenantMatchDep
from src.web.deps.tenants import get_tenant_invitation_service, get_tenant_member_service
from src.web.exception_handler import register_exception_handlers
from src.web.middleware.auth import require_auth

_NOW = datetime(2026, 9, 5, tzinfo=UTC)
_TENANT = 7
_USER = "u-1"


def _membership(**overrides: object) -> MembershipInfo:
    defaults: dict[str, object] = {
        "id": 1,
        "user_id": "u-2",
        "tenant_id": _TENANT,
        "role": "viewer",
        "status": "active",
        "invited_by": _USER,
        "joined_at": _NOW,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    defaults.update(overrides)
    return MembershipInfo.model_validate(defaults)


def _invitation(**overrides: object) -> TenantInvitationInfo:
    defaults: dict[str, object] = {
        "id": 3,
        "tenant_id": _TENANT,
        "invitee_user_id": "u-2",
        "invited_by": _USER,
        "role": "viewer",
        "status": "pending",
        "message": None,
        "expires_at": _NOW + timedelta(days=2),
        "responded_at": None,
        "accepted_count": 0,
        "is_share_link": False,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    defaults.update(overrides)
    return TenantInvitationInfo.model_validate(defaults)


def _noop_gates(app: FastAPI) -> None:
    def _noop() -> None:
        return None

    for dep in (RoleViewerDep, RoleOwnerDep, PathTenantMatchDep):
        for metadata in getattr(dep, "__metadata__", ()):
            if isinstance(metadata, Depends) and metadata.dependency is not None:
                app.dependency_overrides[metadata.dependency] = _noop


def _build_app(
    *,
    member_service: AsyncMock,
    invitation_service: AsyncMock,
    auth_service: AsyncMock,
) -> FastAPI:
    async def _noop_auth() -> None:
        request_context.set_user_id(_USER)
        request_context.set_tenant_id(str(_TENANT))

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(tenants_router, prefix="/api/v1")
    app.dependency_overrides[require_auth] = _noop_auth
    app.dependency_overrides[get_tenant_member_service] = lambda: member_service
    app.dependency_overrides[get_tenant_invitation_service] = lambda: invitation_service
    app.dependency_overrides[get_auth_service] = lambda: auth_service
    _noop_gates(app)
    return app


def test_list_members_uses_existing_envelope() -> None:
    members = AsyncMock()
    members.list_members_page = AsyncMock(return_value=([_membership()], 1))
    app = _build_app(
        member_service=members,
        invitation_service=AsyncMock(),
        auth_service=AsyncMock(),
    )

    with TestClient(app) as client:
        response = client.get(f"/api/v1/tenants/{_TENANT}/members?q=ann&page=1&page_size=20")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["total"] == 1
    assert body["data"]["page"] == 1
    assert body["data"]["page_size"] == 20
    assert body["data"]["members"][0]["user_id"] == "u-2"
    assert body["data"]["members"][0]["email"] == ""
    members.list_members_page.assert_awaited_once_with(
        _TENANT,
        query="ann",
        page=1,
        page_size=20,
    )


def test_add_member_resolves_email() -> None:
    members = AsyncMock()
    members.add_member = AsyncMock(return_value=_membership())
    auth = AsyncMock()
    auth.get_user_row_by_email = AsyncMock(
        return_value=SimpleNamespace(
            id="u-2",
            email="ann@example.com",
            username="ann",
            avatar=None,
        )
    )
    app = _build_app(
        member_service=members,
        invitation_service=AsyncMock(),
        auth_service=auth,
    )

    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/tenants/{_TENANT}/members",
            json={"email": "ann@example.com", "role": "viewer"},
        )

    assert response.status_code == 201
    data = response.json()["data"]
    assert data["email"] == "ann@example.com"
    assert data["username"] == "ann"
    members.add_member.assert_awaited_once_with(
        user_id="u-2",
        tenant_id=_TENANT,
        role="viewer",
        invited_by=_USER,
    )


def test_add_member_unknown_email_is_404() -> None:
    auth = AsyncMock()
    auth.get_user_row_by_email = AsyncMock(return_value=None)
    members = AsyncMock()
    app = _build_app(
        member_service=members,
        invitation_service=AsyncMock(),
        auth_service=auth,
    )

    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/tenants/{_TENANT}/members",
            json={"email": "missing@example.com", "role": "viewer"},
        )

    assert response.status_code == 404
    members.add_member.assert_not_called()


def test_update_and_delete_member() -> None:
    members = AsyncMock()
    members.update_role = AsyncMock(return_value=_membership(role="admin"))
    members.remove_member = AsyncMock(return_value=None)
    app = _build_app(
        member_service=members,
        invitation_service=AsyncMock(),
        auth_service=AsyncMock(),
    )

    with TestClient(app) as client:
        updated = client.put(
            f"/api/v1/tenants/{_TENANT}/members/u-2",
            json={"role": "admin"},
        )
        deleted = client.delete(f"/api/v1/tenants/{_TENANT}/members/u-2")

    assert updated.status_code == 200
    assert updated.json()["success"] is True
    assert deleted.status_code == 200
    members.update_role.assert_awaited_once_with(
        user_id="u-2",
        tenant_id=_TENANT,
        role="admin",
    )
    members.remove_member.assert_awaited_once_with(user_id="u-2", tenant_id=_TENANT)


def test_leave_wraps_remove_member_for_caller() -> None:
    members = AsyncMock()
    members.remove_member = AsyncMock(return_value=None)
    app = _build_app(
        member_service=members,
        invitation_service=AsyncMock(),
        auth_service=AsyncMock(),
    )

    with TestClient(app) as client:
        response = client.post(f"/api/v1/tenants/{_TENANT}/leave")

    assert response.status_code == 200
    assert response.json()["success"] is True
    members.remove_member.assert_awaited_once_with(user_id=_USER, tenant_id=_TENANT)


def test_leave_last_owner_is_conflict() -> None:
    members = AsyncMock()
    members.remove_member = AsyncMock(
        side_effect=ConflictError(
            code="tenant_member.last_owner",
            message="Workspace must keep at least one owner",
        )
    )
    app = _build_app(
        member_service=members,
        invitation_service=AsyncMock(),
        auth_service=AsyncMock(),
    )

    with TestClient(app) as client:
        response = client.post(f"/api/v1/tenants/{_TENANT}/leave")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "tenant_member.last_owner"


def test_create_invitation_lands_on_invitee() -> None:
    invitations = AsyncMock()
    invitations.create_invitation = AsyncMock(return_value=_invitation())
    auth = AsyncMock()
    auth.get_user_row_by_email = AsyncMock(
        return_value=SimpleNamespace(
            id="u-2",
            email="ann@example.com",
            username="ann",
            avatar=None,
        )
    )
    app = _build_app(
        member_service=AsyncMock(),
        invitation_service=invitations,
        auth_service=auth,
    )

    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/tenants/{_TENANT}/invitations",
            json={"email": "ann@example.com", "role": "viewer"},
        )

    assert response.status_code == 201
    body = response.json()["data"]
    assert body["invitee_user_id"] == "u-2"
    assert body["invitee_email"] == "ann@example.com"
    invitations.create_invitation.assert_awaited_once_with(
        tenant_id=_TENANT,
        invitee_user_id="u-2",
        role="viewer",
        invited_by=_USER,
        message=None,
    )


def test_list_and_revoke_invitations() -> None:
    invitations = AsyncMock()
    invitations.list_tenant_invitations_page = AsyncMock(return_value=([_invitation()], 1))
    invitations.get_invitation = AsyncMock(return_value=_invitation())
    invitations.revoke = AsyncMock(return_value=None)
    app = _build_app(
        member_service=AsyncMock(),
        invitation_service=invitations,
        auth_service=AsyncMock(),
    )

    with TestClient(app) as client:
        listed = client.get(f"/api/v1/tenants/{_TENANT}/invitations?page=1&page_size=20")
        revoked = client.delete(f"/api/v1/tenants/{_TENANT}/invitations/3")

    assert listed.status_code == 200
    data = listed.json()["data"]
    assert data["total"] == 1
    assert data["invitations"][0]["id"] == 3
    assert revoked.status_code == 200
    invitations.revoke.assert_awaited_once_with(3)


def test_revoke_hides_other_tenant_invitation() -> None:
    invitations = AsyncMock()
    invitations.get_invitation = AsyncMock(return_value=_invitation(tenant_id=99))
    invitations.revoke = AsyncMock()
    app = _build_app(
        member_service=AsyncMock(),
        invitation_service=invitations,
        auth_service=AsyncMock(),
    )

    with TestClient(app) as client:
        response = client.delete(f"/api/v1/tenants/{_TENANT}/invitations/3")

    assert response.status_code == 404
    invitations.revoke.assert_not_called()


def test_revoke_missing_invitation_is_404() -> None:
    invitations = AsyncMock()
    invitations.get_invitation = AsyncMock(return_value=None)
    invitations.revoke = AsyncMock()
    app = _build_app(
        member_service=AsyncMock(),
        invitation_service=invitations,
        auth_service=AsyncMock(),
    )

    with TestClient(app) as client:
        response = client.delete(f"/api/v1/tenants/{_TENANT}/invitations/9")

    assert response.status_code == 404
    invitations.revoke.assert_not_called()


def test_create_invite_link_returns_url() -> None:
    invitations = AsyncMock()
    invitations.create_share_link = AsyncMock(
        return_value=(_invitation(invitee_user_id="", is_share_link=True), "tok-1")
    )
    app = _build_app(
        member_service=AsyncMock(),
        invitation_service=invitations,
        auth_service=AsyncMock(),
    )

    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/tenants/{_TENANT}/invite-links",
            json={"role": "viewer"},
        )

    assert response.status_code == 201
    assert response.json()["data"]["invite_url"] == "/register?token=tok-1"
    invitations.create_share_link.assert_awaited_once_with(
        tenant_id=_TENANT,
        role="viewer",
        invited_by=_USER,
        message=None,
    )


def test_update_role_last_owner_is_conflict() -> None:
    members = AsyncMock()
    members.update_role = AsyncMock(
        side_effect=ConflictError(
            code="tenant_member.last_owner",
            message="Workspace must keep at least one owner",
        )
    )
    app = _build_app(
        member_service=members,
        invitation_service=AsyncMock(),
        auth_service=AsyncMock(),
    )

    with TestClient(app) as client:
        response = client.put(
            f"/api/v1/tenants/{_TENANT}/members/{_USER}",
            json={"role": "viewer"},
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "tenant_member.last_owner"
