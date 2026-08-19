"""Unit tests for the middleware modules.

Tests cover the RBAC guards (require_role / require_system_admin /
require_ownership_or_role), the API Key Gate (route policy + authorizer),
and the audit middleware (set/get on request state).

The KB-access and embed-auth stubs are tested only for their
``NotImplementedError`` contract.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi import Request

from src.common.exception import PermissionDeniedError
from src.core.auth.permissions import (
    APIKeyCapability,
    APIKeyRoutePolicy,
    APIKeyScopeType,
    TenantAPIKeyScope,
    TenantRole,
)
from src.core.system.audit_actions import AuditAction
from src.core.system.audit_service import AuditLogService
from src.db.models.system.audit_log import AuditLog
from src.web.middleware.api_key_gate import APIKeyRouteAuthorizer, api_key_gate
from src.web.middleware.audit import get_audit_service, set_audit_service
from src.web.middleware.embed_auth import embed_auth
from src.web.middleware.kb_access import require_kb_access
from src.web.middleware.rbac import (
    ResourceNotFoundError,
    require_ownership_or_role,
    require_role,
    require_system_admin,
)

# ── Helpers ────────────────────────────────────────────────────────


def _make_request(
    *,
    method: str = "GET",
    path: str = "/api/v1/test",
    tenant_role: str | None = None,
    is_system_admin: bool = False,
    api_key_scope: TenantAPIKeyScope | None = None,
    user_id: str | None = None,
    tenant_id: str | None = None,
) -> Request:
    """Build a minimal Starlette ``Request`` with state pre-populated."""
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": [],
        "query_string": b"",
    }
    request = Request(scope)
    if tenant_role is not None:
        request.state.tenant_role = tenant_role
    request.state.is_system_admin = is_system_admin
    request.state.api_key_scope = api_key_scope
    if user_id is not None:
        request.state.user_id = user_id
    if tenant_id is not None:
        request.state.tenant_id = tenant_id
    return request


class _FakeAuditRepo:
    """Minimal AuditLogRepository stub for testing."""

    def __init__(self) -> None:
        self.rows: list[AuditLog] = []
        self._next_id: int = 1

    async def create(self, entry: AuditLog) -> AuditLog:
        persisted = entry.model_copy(update={"id": self._next_id})
        self._next_id += 1
        self.rows.append(persisted)
        return persisted

    async def count_since_for_dedup(
        self,
        *,
        tenant_id: int,
        actor_user_id: str,
        action: str,
        request_path: str,
        since: datetime,
    ) -> int:
        return 0  # never dedup in unit tests

    async def delete_older_than(self, cutoff: datetime) -> int:
        return 0

    async def list_for_tenant(self, **kwargs: object) -> list[AuditLog]:
        return []


def _make_audit_svc() -> tuple[AuditLogService, _FakeAuditRepo]:
    repo = _FakeAuditRepo()
    return AuditLogService(audit_repo=repo), repo  # type: ignore[arg-type]


# ── TenantRole tests ──────────────────────────────────────────────


def test_role_hierarchy() -> None:
    assert TenantRole.has_permission(TenantRole.OWNER, TenantRole.ADMIN)
    assert TenantRole.has_permission(TenantRole.ADMIN, TenantRole.CONTRIBUTOR)
    assert TenantRole.has_permission(TenantRole.CONTRIBUTOR, TenantRole.VIEWER)
    assert not TenantRole.has_permission(TenantRole.VIEWER, TenantRole.ADMIN)
    assert not TenantRole.has_permission("", TenantRole.VIEWER)
    assert TenantRole.is_valid(TenantRole.OWNER)
    assert not TenantRole.is_valid("bogus")


# ── TenantAPIKeyScope tests ────────────────────────────────────────


def test_api_key_scope_platform() -> None:
    scope = TenantAPIKeyScope(
        key_id=1,
        scope_type=APIKeyScopeType.PLATFORM,
        full_access=True,
    )
    assert scope.is_platform()
    assert scope.full_access


def test_api_key_scope_has_capability() -> None:
    scope = TenantAPIKeyScope(
        key_id=2,
        capabilities=[APIKeyCapability.CHAT, APIKeyCapability.RETRIEVE],
    )
    assert scope.has_capability(APIKeyCapability.CHAT)
    assert not scope.has_capability(APIKeyCapability.MANAGE_AGENTS)
    assert not scope.has_capability("")


def test_api_key_scope_kb_restricted() -> None:
    scope = TenantAPIKeyScope(key_id=3, knowledge_base_ids=["kb-1"])
    assert scope.is_knowledge_base_restricted()
    scope2 = TenantAPIKeyScope(key_id=4)
    assert not scope2.is_knowledge_base_restricted()


# ── APIKeyRoutePolicy tests ─────────────────────────────────────────


def test_route_policy_with_capability() -> None:
    p = APIKeyRoutePolicy(capabilities=[APIKeyCapability.CHAT])
    p2 = p.with_capability(APIKeyCapability.RETRIEVE)
    assert APIKeyCapability.CHAT in p2.capabilities
    assert APIKeyCapability.RETRIEVE in p2.capabilities
    # Original is not mutated.
    assert APIKeyCapability.RETRIEVE not in p.capabilities


# ── APIKeyRouteAuthorizer tests ─────────────────────────────────────


def test_authorizer_register_and_lookup() -> None:
    auth = APIKeyRouteAuthorizer()
    policy = APIKeyRoutePolicy(capabilities=[APIKeyCapability.CHAT])
    auth.register("GET", "/api/v1/knowledge-bases/{id}", policy)
    found = auth.lookup("GET", "/api/v1/knowledge-bases/{id}")
    assert found is not None
    assert APIKeyCapability.CHAT in found.capabilities


def test_authorizer_lookup_missing_returns_none() -> None:
    auth = APIKeyRouteAuthorizer()
    assert auth.lookup("POST", "/api/v1/unknown") is None


# ── api_key_gate tests ──────────────────────────────────────────────


async def test_api_key_gate_jwt_principal_passes() -> None:
    request = _make_request(api_key_scope=None)
    auth = APIKeyRouteAuthorizer()
    # No policy registered; JWT principal should pass anyway.
    await api_key_gate(request=request, authorizer=auth)


async def test_api_key_gate_no_policy_denies() -> None:
    scope = TenantAPIKeyScope(key_id=1, full_access=True)
    request = _make_request(api_key_scope=scope)
    auth = APIKeyRouteAuthorizer()
    with pytest.raises(PermissionDeniedError):
        await api_key_gate(request=request, authorizer=auth)


async def test_api_key_gate_full_access_passes() -> None:
    scope = TenantAPIKeyScope(key_id=1, full_access=True)
    request = _make_request(api_key_scope=scope, method="GET")
    auth = APIKeyRouteAuthorizer()
    auth.register("GET", "/api/v1/test", APIKeyRoutePolicy())
    await api_key_gate(request=request, authorizer=auth)


async def test_api_key_gate_scoped_key_with_capability_passes() -> None:
    scope = TenantAPIKeyScope(
        key_id=2,
        capabilities=[APIKeyCapability.CHAT],
    )
    request = _make_request(api_key_scope=scope, method="POST")
    auth = APIKeyRouteAuthorizer()
    auth.register(
        "POST",
        "/api/v1/test",
        APIKeyRoutePolicy(capabilities=[APIKeyCapability.CHAT]),
    )
    await api_key_gate(request=request, authorizer=auth)


async def test_api_key_gate_scoped_key_wrong_capability_denies() -> None:
    scope = TenantAPIKeyScope(
        key_id=2,
        capabilities=[APIKeyCapability.RETRIEVE],
    )
    request = _make_request(api_key_scope=scope, method="POST")
    auth = APIKeyRouteAuthorizer()
    auth.register(
        "POST",
        "/api/v1/test",
        APIKeyRoutePolicy(capabilities=[APIKeyCapability.CHAT]),
    )
    with pytest.raises(PermissionDeniedError):
        await api_key_gate(request=request, authorizer=auth)


async def test_api_key_gate_platform_only_denies_tenant_scope() -> None:
    scope = TenantAPIKeyScope(
        key_id=3,
        scope_type=APIKeyScopeType.TENANT,
        full_access=True,
    )
    request = _make_request(api_key_scope=scope, method="GET")
    auth = APIKeyRouteAuthorizer()
    auth.register(
        "GET",
        "/api/v1/test",
        APIKeyRoutePolicy(platform_only=True),
    )
    with pytest.raises(PermissionDeniedError):
        await api_key_gate(request=request, authorizer=auth)


# ── RBAC require_role tests ─────────────────────────────────────────


async def test_require_role_sufficient_passes() -> None:
    request = _make_request(tenant_role=TenantRole.ADMIN)
    audit_svc, _ = _make_audit_svc()
    await require_role(
        min_role=TenantRole.CONTRIBUTOR,
        request=request,
        audit_svc=audit_svc,
    )


async def test_require_role_insufficient_denies() -> None:
    request = _make_request(
        tenant_role=TenantRole.VIEWER,
        user_id="usr-1",
        tenant_id="7",
    )
    audit_svc, _ = _make_audit_svc()
    with pytest.raises(PermissionDeniedError):
        await require_role(
            min_role=TenantRole.ADMIN,
            request=request,
            audit_svc=audit_svc,
        )


async def test_require_role_api_key_short_circuits() -> None:
    scope = TenantAPIKeyScope(key_id=1)
    request = _make_request(
        tenant_role="",  # no role
        api_key_scope=scope,
    )
    audit_svc, _ = _make_audit_svc()
    await require_role(
        min_role=TenantRole.OWNER,
        request=request,
        audit_svc=audit_svc,
    )


async def test_require_role_system_admin_bypasses() -> None:
    request = _make_request(
        tenant_role=TenantRole.VIEWER,
        is_system_admin=True,
    )
    audit_svc, _ = _make_audit_svc()
    await require_role(
        min_role=TenantRole.OWNER,
        request=request,
        audit_svc=audit_svc,
    )


async def test_require_role_emits_audit_on_deny() -> None:
    request = _make_request(
        tenant_role=TenantRole.VIEWER,
        user_id="usr-1",
        tenant_id="7",
    )
    audit_svc, repo = _make_audit_svc()
    with pytest.raises(PermissionDeniedError):
        await require_role(
            min_role=TenantRole.ADMIN,
            request=request,
            audit_svc=audit_svc,
        )
    assert len(repo.rows) == 1
    assert repo.rows[0].action == AuditAction.ACCESS_DENIED


# ── RBAC require_system_admin tests ────────────────────────────────


async def test_require_system_admin_passes() -> None:
    request = _make_request(is_system_admin=True)
    audit_svc, _ = _make_audit_svc()
    await require_system_admin(request=request, audit_svc=audit_svc)


async def test_require_system_admin_denies_non_admin() -> None:
    request = _make_request(is_system_admin=False, user_id="usr-1")
    audit_svc, _ = _make_audit_svc()
    with pytest.raises(PermissionDeniedError):
        await require_system_admin(request=request, audit_svc=audit_svc)


async def test_require_system_admin_platform_key_passes() -> None:
    scope = TenantAPIKeyScope(
        key_id=1,
        scope_type=APIKeyScopeType.PLATFORM,
    )
    request = _make_request(api_key_scope=scope)
    audit_svc, _ = _make_audit_svc()
    await require_system_admin(request=request, audit_svc=audit_svc)


async def test_require_system_admin_tenant_key_denies() -> None:
    scope = TenantAPIKeyScope(key_id=1, scope_type=APIKeyScopeType.TENANT)
    request = _make_request(api_key_scope=scope)
    audit_svc, _ = _make_audit_svc()
    with pytest.raises(PermissionDeniedError):
        await require_system_admin(request=request, audit_svc=audit_svc)


# ── RBAC require_ownership_or_role tests ───────────────────────────


async def test_ownership_role_sufficient_passes() -> None:
    request = _make_request(tenant_role=TenantRole.ADMIN)
    audit_svc, _ = _make_audit_svc()

    async def lookup(req: Request) -> tuple[str, Exception | None]:
        return "", None

    await require_ownership_or_role(
        min_role=TenantRole.ADMIN,
        lookup=lookup,
        request=request,
        audit_svc=audit_svc,
    )


async def test_ownership_role_creator_match_passes() -> None:
    request = _make_request(
        tenant_role=TenantRole.VIEWER,
        user_id="usr-owner",
    )
    audit_svc, _ = _make_audit_svc()

    async def lookup(req: Request) -> tuple[str, Exception | None]:
        return "usr-owner", None

    await require_ownership_or_role(
        min_role=TenantRole.ADMIN,
        lookup=lookup,
        request=request,
        audit_svc=audit_svc,
    )


async def test_ownership_role_not_creator_denies() -> None:
    request = _make_request(
        tenant_role=TenantRole.VIEWER,
        user_id="usr-1",
        tenant_id="7",
    )
    audit_svc, _ = _make_audit_svc()

    async def lookup(req: Request) -> tuple[str, Exception | None]:
        return "usr-other", None

    with pytest.raises(PermissionDeniedError):
        await require_ownership_or_role(
            min_role=TenantRole.ADMIN,
            lookup=lookup,
            request=request,
            audit_svc=audit_svc,
        )


async def test_ownership_role_resource_not_found_passes_through() -> None:
    request = _make_request(
        tenant_role=TenantRole.VIEWER,
        user_id="usr-1",
    )
    audit_svc, _ = _make_audit_svc()

    async def lookup(req: Request) -> tuple[str, Exception | None]:
        return "", ResourceNotFoundError()

    await require_ownership_or_role(
        min_role=TenantRole.ADMIN,
        lookup=lookup,
        request=request,
        audit_svc=audit_svc,
    )


# ── Audit middleware tests ─────────────────────────────────────────


def test_audit_set_and_get() -> None:
    request = _make_request()
    assert get_audit_service(request) is None
    svc, _ = _make_audit_svc()
    set_audit_service(request, svc)
    assert get_audit_service(request) is svc


def test_audit_set_none() -> None:
    request = _make_request()
    set_audit_service(request, None)
    assert get_audit_service(request) is None


# ── Stub tests ─────────────────────────────────────────────────────


async def test_kb_access_stub_raises_not_implemented() -> None:
    request = _make_request()

    async def resolver(req: Request) -> str:
        return "kb-1"

    with pytest.raises(NotImplementedError):
        await require_kb_access(resolver=resolver, request=request)


async def test_embed_auth_stub_raises_not_implemented() -> None:
    request = _make_request()
    with pytest.raises(NotImplementedError):
        await embed_auth(request=request)


__all__ = []


# ── Path tenant match ────────────────────────────────────────────────


async def test_path_tenant_match_passes_when_match() -> None:
    from src.web.deps.rbac import require_path_tenant_match_dep

    request = _make_request(tenant_id="42")
    request.scope["path_params"] = {"tenant_id": "42"}
    # Should not raise.
    await require_path_tenant_match_dep(request)


async def test_path_tenant_match_raises_on_mismatch() -> None:
    from src.common.exception import PermissionDeniedError
    from src.web.deps.rbac import require_path_tenant_match_dep

    request = _make_request(tenant_id="42")
    request.scope["path_params"] = {"tenant_id": "99"}
    with pytest.raises(PermissionDeniedError) as excinfo:
        await require_path_tenant_match_dep(request)
    assert excinfo.value.code == "rbac.not_a_member"


async def test_path_tenant_match_noop_when_no_path_param() -> None:
    from src.web.deps.rbac import require_path_tenant_match_dep

    request = _make_request(tenant_id="42")
    # No ``tenant_id`` in path_params → dep is a no-op so it can be
    # wired into both per-tenant and non-tenant routes.
    await require_path_tenant_match_dep(request)


async def test_path_tenant_match_bypassed_for_system_admin() -> None:
    from src.web.deps.rbac import require_path_tenant_match_dep

    request = _make_request(tenant_id="42", is_system_admin=True)
    request.scope["path_params"] = {"tenant_id": "99"}
    # System admins can address any workspace.
    await require_path_tenant_match_dep(request)
