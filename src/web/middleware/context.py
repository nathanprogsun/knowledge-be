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


def get_tenant_role(request: Request) -> str:
    """Return the caller's tenant role, or ``""`` when unset.

    Defaults to viewer-level when absent would be safer, but the Go
    upstream uses ``TenantRoleViewer`` as the default. Here we return
    empty string so ``TenantRole.has_permission("", min)`` returns
    ``False`` for any non-empty min — fail closed.
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
    "get_tenant_role",
    "get_user_info",
]
