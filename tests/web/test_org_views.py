"""Unit tests for the organization web endpoints.

The router is mounted on a minimal FastAPI app; the auth gate and the
organization / tenant service deps are overridden so tests exercise
routing, request validation, and response shapes without a database.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.app_context import request_context
from src.common.exception import (
    ConflictError,
    NotFoundError,
    ValidationError,
)
from src.core.organizations.service.organization_service import (
    JOIN_REQUEST_STATUS_PENDING,
    JOIN_REQUEST_TYPE_JOIN,
    ORG_ROLE_ADMIN,
    OrganizationService,
)
from src.core.organizations.types import (
    OrganizationInfo,
    OrganizationJoinRequestInfo,
    OrganizationMemberInfo,
)
from src.core.tenants.service import TenantService
from src.core.tenants.types import TenantInfo
from src.web.api.organizations.router import router
from src.web.deps.organizations import get_organization_service
from src.web.deps.tenants import get_tenant_service
from src.web.exception_handler import register_exception_handlers
from src.web.middleware.auth import require_auth

# Principal constants the fake auth stamps on every request.
_TENANT: int = 42
_USER: str = "user-1"

# Module-level service holders so dependency overrides can resolve them
# per request without threading mutable state through the FastAPI app.
_holder: dict[str, object] = {}


def _org_info(**overrides: object) -> OrganizationInfo:
    """Build an OrganizationInfo DTO with sensible defaults."""
    defaults: dict[str, object] = {
        "id": "org-1",
        "name": "Acme Org",
        "description": "A test organization",
        "avatar": "",
        "owner_id": _USER,
        "owner_tenant_id": _TENANT,
        "invite_code_expires_at": None,
        "invite_code_validity_days": 7,
        "require_approval": False,
        "searchable": False,
        "member_limit": 50,
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "updated_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    defaults.update(overrides)
    return OrganizationInfo(**defaults)  # type: ignore[arg-type]


def _member_info(**overrides: object) -> OrganizationMemberInfo:
    """Build an OrganizationMemberInfo DTO."""
    defaults: dict[str, object] = {
        "id": "mem-1",
        "organization_id": "org-1",
        "tenant_id": _TENANT,
        "role": ORG_ROLE_ADMIN,
        "representative_user_id": _USER,
        "joined_at": datetime(2026, 1, 1, tzinfo=UTC),
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "updated_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    defaults.update(overrides)
    return OrganizationMemberInfo(**defaults)  # type: ignore[arg-type]


def _join_request_info(**overrides: object) -> OrganizationJoinRequestInfo:
    """Build an OrganizationJoinRequestInfo DTO."""
    defaults: dict[str, object] = {
        "id": "req-1",
        "organization_id": "org-1",
        "user_id": _USER,
        "tenant_id": _TENANT,
        "status": JOIN_REQUEST_STATUS_PENDING,
        "requested_role": "viewer",
        "request_type": JOIN_REQUEST_TYPE_JOIN,
        "prev_role": None,
        "message": "Please let me in",
        "reviewed_at": None,
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "updated_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    defaults.update(overrides)
    return OrganizationJoinRequestInfo(**defaults)  # type: ignore[arg-type]


def _tenant_info(**overrides: object) -> TenantInfo:
    """Build a TenantInfo DTO with minimal required fields."""
    defaults: dict[str, object] = {
        "id": 99,
        "name": "Workspace X",
        "status": "active",
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "updated_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    defaults.update(overrides)
    return TenantInfo(**defaults)  # type: ignore[arg-type]


def _happy_path_org_service() -> AsyncMock:
    """Build an OrganizationService mock preconfigured for the happy path.

    Methods the router calls resolve to the canned data below; individual
    tests can override any of them to inject errors or custom returns.
    """
    org = _org_info()
    member = _member_info()
    request = _join_request_info()
    svc = AsyncMock(spec=OrganizationService)
    svc.create_organization.return_value = org
    svc.get_organization.return_value = org
    svc.list_tenant_organizations.return_value = [org]
    svc.list_tenant_members.return_value = [member]
    # ``get_tenant_member`` returns the canned membership only when the
    # caller matches the test principal; other workspaces (e.g. an
    # invite target) read as non-members, which is what the invite and
    # search flows expect.
    async def _smart_get_member(*, org_id: str, tenant_id: int):
        if tenant_id == _TENANT:
            return member
        raise NotFoundError(
            code="organization.tenant_not_member",
            message="tenant not member",
        )

    svc.get_tenant_member.side_effect = _smart_get_member
    svc.get_tenant_role_in_org.return_value = ORG_ROLE_ADMIN
    svc.count_pending_join_requests.return_value = 0
    # No pending upgrade request by default — the router catches the
    # ``NotFoundError`` and reports ``has_pending_upgrade=False``.
    svc.get_pending_upgrade_request.side_effect = NotFoundError(
        code="organization.upgrade_request_not_found",
        message="pending upgrade request not found",
    )
    svc.update_organization.return_value = org
    svc.delete_organization.return_value = None
    svc.generate_invite_code.return_value = "aabbccdd00112233"
    svc.join_by_invite_code.return_value = org
    svc.submit_join_request.return_value = request
    svc.join_by_organization_id.return_value = org
    svc.get_organization_by_invite_code.return_value = org
    svc.search_searchable_organizations.return_value = [org]
    svc.request_role_upgrade.return_value = request
    svc.update_tenant_member_role.return_value = member
    svc.remove_tenant_member.return_value = None
    svc.is_tenant_org_admin.return_value = True
    svc.list_join_requests.return_value = [request]
    svc.review_join_request.return_value = request
    svc.add_tenant_member.return_value = member
    return svc


def _happy_path_tenant_service() -> AsyncMock:
    """Build a TenantService mock preconfigured for the happy path."""
    tenant = _tenant_info()
    svc = AsyncMock(spec=TenantService)
    svc.search_tenants.return_value = ([tenant], 1)
    svc.get_tenants.return_value = {tenant.id: tenant}
    svc.get_tenant.return_value = tenant
    return svc


async def _fake_auth(request: Request) -> None:
    """Stand-in for ``require_auth`` that stamps an admin principal."""
    request.state.tenant_id = str(_TENANT)
    request.state.tenant_role = "admin"
    request.state.user_info = {
        "id": _USER,
        "username": "alice",
        "email": "alice@example.com",
        "is_active": "1",
        "can_access_all_tenants": "0",
        "is_system_admin": "0",
    }
    request.state.is_system_admin = False
    request.state.api_key_scope = None
    request_context.set_tenant_id(str(_TENANT))
    request_context.set_user_id(_USER)


def _get_org_service() -> AsyncMock:
    """DI override factory: return the per-test organization service."""
    return _holder["org"]  # type: ignore[return-value]


def _get_tenant_service() -> AsyncMock:
    """DI override factory: return the per-test tenant service."""
    return _holder["tenant"]  # type: ignore[return-value]


@pytest.fixture
def client() -> TestClient:
    """A ``TestClient`` bound to a minimal app with the org router."""
    _holder["org"] = _happy_path_org_service()
    _holder["tenant"] = _happy_path_tenant_service()
    application = FastAPI()
    register_exception_handlers(application)
    application.include_router(router)
    application.dependency_overrides[require_auth] = _fake_auth
    application.dependency_overrides[get_organization_service] = _get_org_service
    application.dependency_overrides[get_tenant_service] = _get_tenant_service
    return TestClient(application)


def org_service(client: TestClient) -> AsyncMock:
    """Return the per-test organization service mock."""
    return _holder["org"]  # type: ignore[return-value]


def tenant_service(client: TestClient) -> AsyncMock:
    """Return the per-test tenant service mock."""
    return _holder["tenant"]  # type: ignore[return-value]


# ── CRUD ─────────────────────────────────────────────────────────────


def test_create_organization_returns_envelope_with_201(client: TestClient) -> None:
    """POST /organizations returns 201 and the success envelope."""
    response = client.post(
        "/organizations",
        json={"name": "New Org", "description": "d", "member_limit": 25},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True
    data = body["data"]
    assert data["id"] == "org-1"
    assert data["name"] == "Acme Org"
    assert data["member_limit"] == 50  # service DTO wins over request default
    assert data["member_count"] == 1  # creator enrolled
    assert data["my_role"] == "admin"
    assert data["is_owner"] is True
    assert data["pending_join_request_count"] == 0
    assert data["invite_code"] is None  # service drops the credential
    org_service(client).create_organization.assert_awaited_once()


def test_create_organization_propagates_validation_error(client: TestClient) -> None:
    """Service ValidationError surfaces as 422."""
    org_service(client).create_organization.side_effect = ValidationError(
        code="organization.invite_validity_invalid",
        message="invalid",
    )
    response = client.post("/organizations", json={"name": "x"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "organization.invite_validity_invalid"


def test_create_organization_requires_name(client: TestClient) -> None:
    """Missing the required ``name`` yields 422 from request validation."""
    response = client.post("/organizations", json={})
    assert response.status_code == 422


def test_list_my_organizations_returns_envelope(client: TestClient) -> None:
    """GET /organizations returns the list envelope with items + total."""
    response = client.get("/organizations")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    data = body["data"]
    assert isinstance(data["items"], list)
    assert data["total"] == 1
    assert data["items"][0]["id"] == "org-1"
    assert data["resource_counts"] is None  # deferred seam


def test_get_organization_returns_envelope(client: TestClient) -> None:
    """GET /organizations/{id} returns one enriched org."""
    response = client.get("/organizations/org-1")
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["id"] == "org-1"
    assert body["data"]["member_count"] == 1


def test_get_organization_private_not_member_returns_404(client: TestClient) -> None:
    """Private org + non-member caller returns 404 (visibility gate)."""
    org_service(client).get_tenant_member.side_effect = NotFoundError(
        code="organization.tenant_not_member",
        message="not a member",
    )
    response = client.get("/organizations/org-1")
    assert response.status_code == 404


def test_update_organization_returns_envelope(client: TestClient) -> None:
    """PUT /organizations/{id} returns the updated org envelope."""
    response = client.put(
        "/organizations/org-1",
        json={"name": "Renamed", "require_approval": True},
    )
    assert response.status_code == 200
    assert response.json()["data"]["id"] == "org-1"
    org_service(client).update_organization.assert_awaited_once()


def test_delete_organization_returns_ack(client: TestClient) -> None:
    """DELETE /organizations/{id} returns the success + message ack."""
    response = client.delete("/organizations/org-1")
    assert response.status_code == 200
    body = response.json()
    assert body == {"success": True, "message": "Organization deleted successfully"}


# ── Invite code & leave ──────────────────────────────────────────────


def test_generate_invite_code_returns_envelope(client: TestClient) -> None:
    """POST /{id}/invite-code wraps the fresh code in the success envelope."""
    response = client.post("/organizations/org-1/invite-code")
    assert response.status_code == 200
    body = response.json()
    assert body == {"success": True, "data": {"invite_code": "aabbccdd00112233"}}


def test_leave_organization_returns_ack(client: TestClient) -> None:
    """POST /{id}/leave returns the success + message ack."""
    response = client.post("/organizations/org-1/leave")
    assert response.status_code == 200
    body = response.json()
    assert body == {"success": True, "message": "Left organization successfully"}
    org_service(client).remove_tenant_member.assert_awaited_once()


def test_leave_organization_owner_conflict_returns_409(client: TestClient) -> None:
    """Owner cannot leave: service ConflictError surfaces as 409."""
    org_service(client).remove_tenant_member.side_effect = ConflictError(
        code="organization.cannot_remove_owner",
        message="cannot remove organization owner tenant",
    )
    response = client.post("/organizations/org-1/leave")
    assert response.status_code == 409


def test_request_role_upgrade_returns_envelope(client: TestClient) -> None:
    """POST /{id}/request-upgrade returns the join-request envelope."""
    response = client.post(
        "/organizations/org-1/request-upgrade",
        json={"requested_role": "editor", "message": "please"},
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["request_type"] == "join"  # default
    assert data["requested_role"] == "viewer"
    org_service(client).request_role_upgrade.assert_awaited_once()


# ── Membership ───────────────────────────────────────────────────────


def test_list_members_returns_envelope(client: TestClient) -> None:
    """GET /{id}/members returns the member list envelope."""
    response = client.get("/organizations/org-1/members")
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["total"] == 1
    assert body["data"]["members"][0]["tenant_id"] == _TENANT
    assert body["data"]["members"][0]["role"] == "admin"


def test_list_members_non_member_returns_403(client: TestClient) -> None:
    """Non-member caller gets 403 (member roster is sensitive)."""
    org_service(client).get_tenant_member.side_effect = NotFoundError(
        code="organization.tenant_not_member",
        message="not a member",
    )
    response = client.get("/organizations/org-1/members")
    assert response.status_code == 403


def test_update_member_role_returns_ack(client: TestClient) -> None:
    """PUT /{id}/members/{tenant_id} returns the success + message ack."""
    response = client.put(
        "/organizations/org-1/members/77",
        json={"role": "editor"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body == {"success": True, "message": "Member role updated successfully"}


def test_remove_member_returns_ack(client: TestClient) -> None:
    """DELETE /{id}/members/{tenant_id} returns the success + message ack."""
    response = client.delete("/organizations/org-1/members/77")
    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "message": "Member removed successfully",
    }


# ── Join requests (admin) ────────────────────────────────────────────


def test_list_join_requests_returns_envelope(client: TestClient) -> None:
    """GET /{id}/join-requests returns the request list envelope."""
    response = client.get("/organizations/org-1/join-requests")
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["total"] == 1
    assert body["data"]["requests"][0]["id"] == "req-1"


def test_list_join_requests_non_admin_returns_403(client: TestClient) -> None:
    """Non-admin caller gets 403."""
    org_service(client).is_tenant_org_admin.return_value = False
    response = client.get("/organizations/org-1/join-requests")
    assert response.status_code == 403


def test_review_join_request_returns_ack(client: TestClient) -> None:
    """PUT /{id}/join-requests/{request_id}/review returns the success ack."""
    response = client.put(
        "/organizations/org-1/join-requests/req-1/review",
        json={"approved": True, "role": "editor", "message": "welcome"},
    )
    assert response.status_code == 200
    assert response.json() == {"success": True, "message": "Review completed"}
    org_service(client).review_join_request.assert_awaited_once()


def test_review_join_request_invalid_role_returns_422(client: TestClient) -> None:
    """Invalid role on review is rejected by the router."""
    response = client.put(
        "/organizations/org-1/join-requests/req-1/review",
        json={"approved": True, "role": "bogus"},
    )
    assert response.status_code == 422


# ── Join-by-code / preview / search ──────────────────────────────────


def test_preview_by_invite_code_returns_envelope(client: TestClient) -> None:
    """GET /preview/{code} returns the preview envelope."""
    response = client.get("/organizations/preview/abc12345")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["id"] == "org-1"
    assert data["name"] == "Acme Org"
    assert data["member_count"] == 1
    assert data["is_already_member"] is True
    assert data["require_approval"] is False


def test_join_by_invite_code_returns_envelope(client: TestClient) -> None:
    """POST /join returns the org envelope on successful join."""
    response = client.post("/organizations/join", json={"invite_code": "abc12345"})
    assert response.status_code == 200
    assert response.json()["data"]["id"] == "org-1"


def test_submit_join_request_returns_envelope(client: TestClient) -> None:
    """POST /join-request returns the join-request envelope."""
    org_service(client).get_organization_by_invite_code.return_value = _org_info(
        require_approval=True
    )
    # Submitting a join request implies the caller is not yet a member.
    org_service(client).get_tenant_member.side_effect = NotFoundError(
        code="organization.tenant_not_member",
        message="tenant not member",
    )
    response = client.post(
        "/organizations/join-request",
        json={"invite_code": "abc12345", "message": "hi", "role": "editor"},
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["request_type"] == "join"


def test_submit_join_request_not_required_returns_422(client: TestClient) -> None:
    """Submitting to a no-approval org is rejected with a 422."""
    response = client.post(
        "/organizations/join-request",
        json={"invite_code": "abc12345"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "organization.join_not_required"


def test_join_by_organization_id_returns_envelope(client: TestClient) -> None:
    """POST /join-by-id returns the org envelope."""
    response = client.post(
        "/organizations/join-by-id",
        json={"organization_id": "org-1", "role": "viewer"},
    )
    assert response.status_code == 200
    assert response.json()["data"]["id"] == "org-1"


def test_search_organizations_returns_envelope(client: TestClient) -> None:
    """GET /search returns the search envelope with items + total."""
    response = client.get("/organizations/search?q=acme&limit=5")
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["total"] == 1
    assert body["data"]["items"][0]["id"] == "org-1"


def test_search_organizations_invalid_limit_falls_back_to_default(
    client: TestClient,
) -> None:
    """Invalid limit values fall back to the default of 20."""
    response = client.get("/organizations/search?q=acme&limit=0")
    assert response.status_code == 200
    org_service(client).search_searchable_organizations.assert_awaited_with(
        tenant_id=_TENANT, query="acme", limit=20
    )


# ── Direct invites ───────────────────────────────────────────────────


def test_search_tenants_for_invite_returns_envelope(client: TestClient) -> None:
    """GET /{id}/search-tenants returns the candidate envelope."""
    response = client.get("/organizations/org-1/search-tenants?q=work&limit=5")
    assert response.status_code == 200
    data = response.json()["data"]
    assert isinstance(data, list)
    assert data[0]["tenant_id"] == 99
    assert data[0]["tenant_name"] == "Workspace X"


def test_search_tenants_for_invite_empty_query_returns_empty(client: TestClient) -> None:
    """Empty query returns an empty list."""
    response = client.get("/organizations/org-1/search-tenants?q=")
    assert response.status_code == 200
    assert response.json()["data"] == []


def test_search_tenants_for_invite_non_admin_returns_403(client: TestClient) -> None:
    """Non-admin caller is rejected."""
    org_service(client).is_tenant_org_admin.return_value = False
    response = client.get("/organizations/org-1/search-tenants?q=work")
    assert response.status_code == 403


def test_search_users_for_invite_alias(client: TestClient) -> None:
    """The /search-users alias forwards to /search-tenants."""
    response = client.get("/organizations/org-1/search-users?q=work")
    assert response.status_code == 200
    assert response.json()["data"][0]["tenant_id"] == 99


def test_invite_member_returns_ack(client: TestClient) -> None:
    """POST /{id}/invite adds a workspace and returns the ack."""
    response = client.post(
        "/organizations/org-1/invite",
        json={
            "tenant_id": 99,
            "representative_user_id": _USER,
            "role": "editor",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"success": True, "message": "Member added successfully"}


def test_invite_member_invalid_role_returns_422(client: TestClient) -> None:
    """Invalid role on invite is rejected by the router."""
    response = client.post(
        "/organizations/org-1/invite",
        json={"tenant_id": 99, "role": "bogus"},
    )
    assert response.status_code == 422


def test_invite_member_missing_target_returns_422(client: TestClient) -> None:
    """Missing both tenant_id and user_id yields 422."""
    response = client.post(
        "/organizations/org-1/invite",
        json={"role": "editor"},
    )
    assert response.status_code == 422


def test_invite_member_legacy_user_only_returns_422(client: TestClient) -> None:
    """Legacy user-only path needs a user service (deferred seam)."""
    response = client.post(
        "/organizations/org-1/invite",
        json={"user_id": "u-1", "role": "editor"},
    )
    assert response.status_code == 422


def test_invite_member_already_member_returns_422(client: TestClient) -> None:
    """Inviting an existing member is rejected with 422."""
    response = client.post(
        "/organizations/org-1/invite",
        json={"tenant_id": _TENANT, "role": "viewer"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "organization.already_member"


# ── Route ordering ───────────────────────────────────────────────────


def test_search_path_does_not_capture_id(client: TestClient) -> None:
    """/search is matched before /{id}; the org service is not consulted."""
    response = client.get("/organizations/search?q=acme")
    assert response.status_code == 200
    org_service(client).get_organization.assert_not_awaited()


def test_preview_path_does_not_capture_id(client: TestClient) -> None:
    """/preview/{code} is matched before /{id}; org service not consulted."""
    response = client.get("/organizations/preview/abc12345")
    assert response.status_code == 200
    org_service(client).get_organization.assert_not_awaited()


# ── Tenant context ───────────────────────────────────────────────────


def test_missing_tenant_context_returns_401(client: TestClient) -> None:
    """A request with no tenant id in the principal fails closed."""
    async def _no_tenant_auth(request: Request) -> None:
        request.state.tenant_id = "0"
        request.state.tenant_role = "admin"
        request.state.user_info = {"id": _USER}
        request.state.is_system_admin = False
        request.state.api_key_scope = None

    app = client.app
    app.dependency_overrides[require_auth] = _no_tenant_auth
    response = client.get("/organizations")
    assert response.status_code == 401