"""Tenant-scoped RBAC gate dependencies.

Wire the role-guard functions in ``src/web/middleware/rbac.py`` as FastAPI
dependencies so routers can declare ``RoleViewerDep`` / ``RoleOwnerDep`` /
``SystemAdminDep`` inline. The RBAC enforcement switch
(``settings.rbac_enforced``) is honoured: when off, the guards log but do
not reject.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends, Request

from src.common.exception import PermissionDeniedError, ValidationError
from src.core.auth.permissions import TenantRole
from src.settings import get_settings
from src.web.middleware.audit import get_audit_service
from src.web.middleware.context import (
    get_api_key_scope,
    get_is_system_admin,
    get_tenant_id,
    get_user_info,
)
from src.web.middleware.rbac import require_role, require_system_admin


async def require_role_dep(
    request: Request,
    min_role: str,
) -> None:
    """Gate: caller's tenant role must be at least ``min_role``.

    Honours ``settings.rbac_enforced``; when false the guard logs but
    does not reject (rollout behaviour, mirroring WeKnora).
    """
    audit_svc = get_audit_service(request)
    if get_api_key_scope(request) is not None:
        return  # API-key principals authorized by the API Key Gate.

    role = getattr(request.state, "tenant_role", "") or ""
    if TenantRole.has_permission(role, min_role):
        return
    if get_is_system_admin(request):
        return

    if not get_settings().rbac_enforced:
        return

    await require_role(
        min_role=min_role,
        request=request,
        audit_svc=audit_svc,
    )


async def require_system_admin_dep(request: Request) -> None:
    """Gate: caller must be a system administrator. Always enforced."""
    audit_svc = get_audit_service(request)
    if get_api_key_scope(request) is not None:
        scope = get_api_key_scope(request)
        if scope is not None and scope.is_platform():
            return
    if get_is_system_admin(request):
        return
    await require_system_admin(request=request, audit_svc=audit_svc)


def make_role_dep(min_role: str) -> Callable[[Request], Awaitable[None]]:
    """Return a FastAPI dependency factory gating on ``min_role``."""

    async def _dep(request: Request) -> None:
        await require_role_dep(request, min_role)

    return _dep


async def require_cross_tenant_dep(request: Request) -> None:
    """Gate: caller must be an org-level superuser for cross-tenant ops.

    Requires BOTH the cluster-wide ``cross_tenant_access_enabled`` flag
    and the caller's ``can_access_all_tenants``. Platform-scope API keys
    pass.
    """
    scope = get_api_key_scope(request)
    if scope is not None and scope.is_platform():
        return

    if not get_settings().cross_tenant_access_enabled:
        raise PermissionDeniedError(
            code="rbac.cross_tenant_disabled",
            message="Cross-workspace access is disabled",
        )

    info = get_user_info(request)
    can_access_all = bool(info and info.get("can_access_all_tenants") == "1")
    if not can_access_all:
        raise PermissionDeniedError(
            code="rbac.cross_tenant_required",
            message="Insufficient permissions for cross-workspace operation",
        )


async def require_path_tenant_match_dep(request: Request) -> None:
    """Gate: URL ``tenant_id`` must equal the caller's active tenant.

    Cross-tenant superusers bypass. Reads ``tenant_id`` from the request
    path params.
    """
    raw = request.path_params.get("tenant_id")
    if raw is None or raw == "":
        raise ValidationError(
            code="tenant.id_required",
            message="Workspace id is required",
        )
    try:
        path_tenant = int(raw)
    except (TypeError, ValueError):
        raise ValidationError(
            code="tenant.id_invalid",
            message="Workspace id must be a positive integer",
        ) from None
    ctx_tenant = get_tenant_id(request)
    if ctx_tenant == 0:
        raise ValidationError(
            code="auth.tenant_context_missing",
            message="Workspace context missing",
        )
    if path_tenant == ctx_tenant:
        return
    if get_is_system_admin(request):
        return
    raise PermissionDeniedError(
        code="rbac.path_tenant_mismatch",
        message="Access denied: URL workspace does not match the active workspace",
    )


CrossTenantDep = Annotated[None, Depends(require_cross_tenant_dep)]
PathTenantMatchDep = Annotated[None, Depends(require_path_tenant_match_dep)]


RoleViewerDep = Annotated[None, Depends(make_role_dep("viewer"))]
RoleContributorDep = Annotated[None, Depends(make_role_dep("contributor"))]
RoleAdminDep = Annotated[None, Depends(make_role_dep("admin"))]
RoleOwnerDep = Annotated[None, Depends(make_role_dep("owner"))]
SystemAdminDep = Annotated[None, Depends(require_system_admin_dep)]


__all__ = [
    "CrossTenantDep",
    "PathTenantMatchDep",
    "RoleAdminDep",
    "RoleContributorDep",
    "RoleOwnerDep",
    "RoleViewerDep",
    "SystemAdminDep",
    "make_role_dep",
    "require_cross_tenant_dep",
    "require_path_tenant_match_dep",
    "require_role_dep",
    "require_system_admin_dep",
]
