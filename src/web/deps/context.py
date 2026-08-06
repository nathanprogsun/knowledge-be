"""FastAPI dependency wrappers for the ``request.state`` principal fields.

These deps read the authenticated principal from ``request.state`` and
return the typed value. They are the function-arg-style replacement for
the ``get_tenant_id(request)`` / ``get_tenant_role(request)`` /
``get_is_system_admin(request)`` accessors.

The underlying ``request.state`` storage is preserved so the auth
middleware keeps working unchanged. The auth middleware stores the
principal directly on ``request.state``; no helper module is required.
"""

from __future__ import annotations

from fastapi import Request

from src.core.auth.permissions import TenantAPIKeyScope


def get_tenant_id_dep(request: Request) -> int:
    """Resolve the active tenant id from the auth context.

    Returns ``0`` when the principal is not authenticated; the
    downstream router typically treats ``0`` as a missing context.
    """
    raw = getattr(request.state, "tenant_id", None)
    if raw is None or raw == "":
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def get_tenant_role_dep(request: Request) -> str:
    """Resolve the caller's tenant role from the auth context.

    Returns the empty string when the principal is not authenticated.
    Returns empty string so ``TenantRole.has_permission("", min)``
    returns ``False`` for any non-empty min — fail closed.
    """
    role: str = getattr(request.state, "tenant_role", "") or ""
    return role or ""


def get_is_system_admin_dep(request: Request) -> bool:
    """Return True when the caller is a platform system administrator."""
    return bool(getattr(request.state, "is_system_admin", False))


def get_user_id_dep(request: Request) -> str | None:
    """Return the authenticated user's id, or ``None`` when unset.

    Reads from ``request.state.user_info`` (a dict with key ``id``).
    Returns the principal user id or ``None`` so caller code can
    distinguish "no principal" from a real user id.
    """
    info = getattr(request.state, "user_info", None)
    if info is None:
        return None
    return info.get("id")


def get_user_info_dep(request: Request) -> dict[str, str] | None:
    """Return the principal user-info dict, or None when not set."""
    return getattr(request.state, "user_info", None)


def get_api_key_scope_dep(request: Request) -> TenantAPIKeyScope | None:
    """Return the API-key scope, or None for non-API-key principals."""
    scope: TenantAPIKeyScope | None = getattr(request.state, "api_key_scope", None)
    return scope


__all__ = [
    "get_api_key_scope_dep",
    "get_is_system_admin_dep",
    "get_tenant_id_dep",
    "get_tenant_role_dep",
    "get_user_id_dep",
    "get_user_info_dep",
]
