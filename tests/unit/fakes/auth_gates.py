"""Shared web-test helpers to neutralise auth/RBAC gates.

View tests exercise routing + service wiring, not the auth layer. These
helpers override the global auth dependency so every request is treated
as a full-access owner principal, letting all role/system-admin/cross-
tenant gates pass while the tests focus on a single domain service.
"""

from __future__ import annotations

from fastapi import FastAPI, Request

from src.web.deps.rbac import require_cross_tenant_dep
from src.web.middleware.auth import require_auth
from src.web.middleware.context import (
    set_is_system_admin,
    set_tenant_id,
    set_tenant_role,
    set_user_info,
)


async def _bypass_auth(request: Request) -> None:
    """Populate request.state with a full-access owner principal."""
    set_user_info(
        request,
        {
            "id": "test-user",
            "username": "test",
            "email": "test@example.com",
            "is_active": "1",
            "can_access_all_tenants": "1",
            "is_system_admin": "1",
        },
    )
    set_is_system_admin(request, True)
    set_tenant_id(request, 1)
    set_tenant_role(request, "owner")


async def _bypass_cross_tenant() -> None:
    return None


def override_auth_gates(app: FastAPI) -> None:
    """Override the auth + cross-tenant gate deps with full-access principals."""
    app.dependency_overrides[require_auth] = _bypass_auth
    app.dependency_overrides[require_cross_tenant_dep] = _bypass_cross_tenant


__all__ = ["override_auth_gates"]
