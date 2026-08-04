"""Request-context accessors for auth/RBAC state.

FastAPI middleware sets the authenticated principal's state on the
``request.state`` attribute (Starlette's per-request store). This
module centralises the read accessors so middleware/guards don't
scatter ``getattr`` calls.

When a value is absent (unauthenticated request, or a guard that
runs before auth middleware has populated the state), accessors
return safe defaults: ``None`` for principal, ``""`` for role,
``False`` for system-admin. The defaults fail closed for role-based
guards (unknown role → level 0 → strictly less than any defined
role).
"""

from __future__ import annotations

from fastapi import Request

from src.core.auth.permissions import TenantAPIKeyScope


def set_tenant_role(request: Request, role: str) -> None:
    """Stash the resolved tenant role on ``request.state``."""
    request.state.tenant_role = role


def set_is_system_admin(request: Request, is_admin: bool) -> None:
    """Stash the platform system-admin flag on ``request.state``."""
    request.state.is_system_admin = is_admin


def set_api_key_scope(request: Request, scope: TenantAPIKeyScope | None) -> None:
    """Stash the resolved API-key scope on ``request.state``."""
    request.state.api_key_scope = scope


def set_user_info(request: Request, info: dict[str, str] | None) -> None:
    """Stash the authenticated user's info dict on ``request.state``.

    Keys: ``id``, ``username``, ``email``, ``is_active``,
    ``can_access_all_tenants``, ``is_system_admin``.
    """
    request.state.user_info = info


def set_tenant_id(request: Request, tenant_id: int) -> None:
    """Stash the active tenant id on ``request.state``."""
    request.state.tenant_id = str(tenant_id)


def get_tenant_id(request: Request) -> int:
    """Read the active tenant id from ``request.state`` (0 when unset)."""
    raw = getattr(request.state, "tenant_id", None)
    if raw is None or raw == "":
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def get_tenant_role(request: Request) -> str:
    """Return the caller's tenant role, or ``""`` when unset.

    Returns empty string so ``TenantRole.has_permission("", min)``
    returns ``False`` for any non-empty min — fail closed.
    """
    role: str = request.state.tenant_role if hasattr(request.state, "tenant_role") else ""
    return role or ""


def get_is_system_admin(request: Request) -> bool:
    """Return ``True`` when the caller is a platform system admin."""
    return bool(getattr(request.state, "is_system_admin", False))


def get_api_key_scope(request: Request) -> TenantAPIKeyScope | None:
    """Return the resolved API-key scope, or ``None`` for JWT principals."""
    scope: TenantAPIKeyScope | None = getattr(request.state, "api_key_scope", None)
    return scope


def get_user_info(request: Request) -> dict[str, str] | None:
    """Return the authenticated user's info dict, or ``None``.

    Keys: ``id``, ``username``, ``email``, ``is_active``,
    ``can_access_all_tenants``, ``is_system_admin``.
    """
    return getattr(request.state, "user_info", None)


__all__ = [
    "get_api_key_scope",
    "get_is_system_admin",
    "get_tenant_id",
    "get_tenant_role",
    "get_user_info",
    "set_api_key_scope",
    "set_is_system_admin",
    "set_tenant_id",
    "set_tenant_role",
    "set_user_info",
]
