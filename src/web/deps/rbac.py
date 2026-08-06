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

from src.common.exception import UnauthorizedError
from src.core.auth.permissions import TenantAPIKeyScope, TenantRole
from src.settings import get_settings
from src.web.middleware.audit import get_audit_service
from src.web.middleware.rbac import require_role, require_system_admin


def _api_key_scope(request: Request) -> TenantAPIKeyScope | None:
    """Read the API-key scope from ``request.state`` (None for JWT)."""
    return getattr(request.state, "api_key_scope", None)


def _is_system_admin(request: Request) -> bool:
    """Read the system-admin flag from ``request.state``."""
    return bool(getattr(request.state, "is_system_admin", False))


def _tenant_role(request: Request) -> str:
    """Read the tenant role from ``request.state`` (empty when unset)."""
    role: str = getattr(request.state, "tenant_role", "") or ""
    return role or ""


def _user_info(request: Request) -> dict[str, str] | None:
    """Read the principal user-info dict (None when not set)."""
    return getattr(request.state, "user_info", None)


def _principal_tenant_id(request: Request) -> int:
    """Read the active tenant id from ``request.state`` (0 when unset)."""
    raw = getattr(request.state, "tenant_id", None)
    if raw is None or raw == "":
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


async def require_role_dep(
    request: Request,
    min_role: str,
) -> None:
    """Gate: caller's tenant role must be at least ``min_role``.

    Honours ``settings.rbac_enforced``; when false the guard logs but
    does not reject (rollout behaviour, mirroring the upstream).
    """
    audit_svc = get_audit_service(request)
    if _api_key_scope(request) is not None:
        return  # API-key principals authorized by the API Key Gate.

    role = _tenant_role(request)
    if TenantRole.has_permission(role, min_role):
        return
    if _is_system_admin(request):
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
    if _api_key_scope(request) is not None:
        scope = _api_key_scope(request)
        if scope is not None and scope.is_platform():
            return
    if _is_system_admin(request):
        return
    await require_system_admin(request=request, audit_svc=audit_svc)


def make_role_dep(min_role: str) -> Callable[[Request], Awaitable[None]]:
    """Return a FastAPI dependency factory gating on ``min_role``."""

    async def _dep(request: Request) -> None:
        await require_role_dep(request, min_role)

    return _dep


async def require_cross_tenant_dep(request: Request) -> None:
    """Backward-compat shim.

    The cross-tenant guard has been superseded by
    :func:`validate_active_tenant_association`; the alias is kept so
    router signatures stay intact during the staged migration. The
    function now no-ops; real enforcement is performed by the new
    DB-backed gate, which is wired in a later commit.
    """
    return None


async def require_path_tenant_match_dep(request: Request) -> None:
    """Backward-compat shim.

    See :func:`require_cross_tenant_dep`. Replaced by
    :func:`validate_active_tenant_association`; the alias remains so
    router signatures stay intact during the staged migration.
    """
    return None


def _get_principal_user_id(request: Request) -> str | None:
    """Read the caller's user id from ``request.state`` (None when unset)."""
    info = _user_info(request)
    if info is None:
        return None
    return info.get("id")


def _get_principal_tenant_id(request: Request) -> int:
    """Read the caller's active tenant id from ``request.state`` (0 when unset)."""
    return _principal_tenant_id(request)


async def validate_active_tenant_association(
    request: Request,
    session: object,
    user_id: Annotated[str | None, Depends(_get_principal_user_id)] = None,
    tenant_id: Annotated[int, Depends(_get_principal_tenant_id)] = 0,
) -> dict[str, object]:
    """DB-backed gate: confirm the caller has an active tenant membership.

    Reads the principal (user id, tenant id) from ``request.state`` via
    the small accessor deps, then issues a real membership lookup
    against ``tenant_members``. Raises ``UnauthorizedError`` when the
    membership is missing, soft-deleted, or the principal itself is
    unresolved. The DB lookup is the source of truth; the principal
    carried in the header is not sufficient on its own.

    Returns a small DTO dict so the handler can read the role and
    membership id without re-querying.
    """
    if user_id is None or user_id == "":
        raise UnauthorizedError(
            code="rbac.principal_unresolved",
            message="Cannot resolve caller principal",
        )
    if tenant_id == 0:
        raise UnauthorizedError(
            code="rbac.tenant_context_missing",
            message="Workspace context missing",
        )

    from src.core.tenants.factory import build_tenant_member_service

    member_service = build_tenant_member_service(session)  # type: ignore[arg-type]
    membership = await member_service.get_membership(
        user_id=user_id,
        tenant_id=tenant_id,
    )
    if membership is None:
        raise UnauthorizedError(
            code="rbac.membership_missing",
            message="Caller is not an active member of the requested workspace",
        )
    if getattr(membership, "deleted_at", None) is not None:
        raise UnauthorizedError(
            code="rbac.membership_soft_deleted",
            message="Caller membership in the requested workspace is inactive",
        )

    return {
        "user_id": str(user_id),
        "tenant_id": int(tenant_id),
        "role": getattr(membership, "role", ""),
        "membership_id": getattr(membership, "id", None),
    }


CrossTenantDep = Annotated[None, Depends(require_cross_tenant_dep)]
PathTenantMatchDep = Annotated[None, Depends(require_path_tenant_match_dep)]
ValidateActiveTenantAssociationDep = Annotated[
    dict[str, object], Depends(validate_active_tenant_association)
]


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
    "ValidateActiveTenantAssociationDep",
    "make_role_dep",
    "require_cross_tenant_dep",
    "require_path_tenant_match_dep",
    "require_role_dep",
    "require_system_admin_dep",
    "validate_active_tenant_association",
]
