"""FastAPI dependency wrappers for the ``request.state`` principal fields.

These deps read the authenticated principal from ``request.state`` and
return the typed value. They are the function-arg-style replacement for
the ``get_tenant_id(request)`` / ``get_tenant_role(request)`` /
``get_is_system_admin(request)`` accessors in
``src.web.middleware.context``.

Routers migrate to these deps one router at a time; the underlying
``request.state`` storage is preserved so the auth middleware keeps
working unchanged. A later commit removes
``src.web.middleware.context`` entirely once every router has
migrated.
"""

from __future__ import annotations

from fastapi import Request

from src.core.auth.permissions import TenantAPIKeyScope
from src.web.middleware.context import (
    get_api_key_scope,
    get_is_system_admin,
    get_tenant_id,
    get_tenant_role,
    get_user_info,
)


def get_tenant_id_dep(request: Request) -> int:
    """Resolve the active tenant id from the auth context.

    Returns ``0`` when the principal is not authenticated; the
    downstream router typically treats ``0`` as a missing context.
    """
    return get_tenant_id(request)


def get_tenant_role_dep(request: Request) -> str:
    """Resolve the caller's tenant role from the auth context.

    Returns the empty string when the principal is not authenticated.
    """
    return get_tenant_role(request)


def get_is_system_admin_dep(request: Request) -> bool:
    """Return True when the caller is a platform system administrator."""
    return get_is_system_admin(request)


def get_user_info_dep(request: Request) -> dict[str, str] | None:
    """Return the principal user-info dict, or None when not set."""
    return get_user_info(request)


def get_api_key_scope_dep(request: Request) -> TenantAPIKeyScope | None:
    """Return the API-key scope, or None for non-API-key principals."""
    return get_api_key_scope(request)


__all__ = [
    "get_api_key_scope_dep",
    "get_is_system_admin_dep",
    "get_tenant_id_dep",
    "get_tenant_role_dep",
    "get_user_info_dep",
]
