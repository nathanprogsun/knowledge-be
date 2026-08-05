"""MCP-domain FastAPI dependency factories.

One-line forwarder to ``src.core.infra.mcp_services.factory``:
repositories are built in ``core`` on the request-scoped
``AsyncSession`` so the request's reads and writes share one
transactional unit of work. ``web`` never imports ``db``.

Adds two FastAPI dependency factories for ``tenant_id`` and ``user_id``
so the router can read them from the per-request auth state without
reaching into the context layer directly.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from src.app_context import request_context
from src.common.exception import ValidationError
from src.core.infra.mcp_services.factory import build_mcp_service
from src.core.infra.mcp_services.service import MCPServiceService
from src.web.deps.session import SessionDep
from src.web.middleware.context import get_tenant_id as _gtid
from src.web.middleware.context import get_user_info as _gui


def get_mcp_service(session: SessionDep) -> MCPServiceService:
    """Build a per-request ``MCPServiceService`` on the shared session."""
    return build_mcp_service(session)


def get_request_tenant_id(request: Request) -> int:
    """Return the active tenant id, or 0 when unset.

    The auth middleware populates ``request.state.tenant_id``; the
    contextvar mirror is read for endpoints that don't go through the
    middleware path (e.g. test-only routes).
    """
    state_value = _gtid(request)
    if state_value:
        return state_value
    raw = request_context.get_tenant_id()
    if raw is None or raw == "":
        return 0
    try:
        return int(raw)
    except ValueError:
        return 0


def get_request_user_id(request: Request) -> str:
    """Return the authenticated user id, or ``""`` when unset.

    Empty string is the fail-closed sentinel — endpoints that actually
    require a user (oauth management, ... ) check it explicitly.
    """
    info = _gui(request)
    if info is not None and isinstance(info, dict):
        user_id = info.get("id")
        if isinstance(user_id, str):
            return user_id
    raw = request_context.get_user_id()
    return raw or ""


MCPServiceDep = Annotated[MCPServiceService, Depends(get_mcp_service)]
RequestTenantIdDep = Annotated[int, Depends(get_request_tenant_id)]
RequestUserIdDep = Annotated[str, Depends(get_request_user_id)]


def require_tenant_id(request: Request) -> int:
    """Same as ``RequestTenantIdDep`` but raises when no tenant is set."""
    value = get_request_tenant_id(request)
    if value <= 0:
        raise ValidationError(
            code="tenant.context_missing",
            message="No active workspace in request context",
        )
    return value


def require_user_id(request: Request) -> str:
    """Same as ``RequestUserIdDep`` but raises when no user is set."""
    value = get_request_user_id(request)
    if not value:
        raise ValidationError(
            code="auth.user_missing",
            message="Authenticated user is required",
        )
    return value


RequireTenantIdDep = Annotated[int, Depends(require_tenant_id)]
RequireUserIdDep = Annotated[str, Depends(require_user_id)]


__all__ = [
    "MCPServiceDep",
    "RequestTenantIdDep",
    "RequestUserIdDep",
    "RequireTenantIdDep",
    "RequireUserIdDep",
    "get_mcp_service",
    "get_request_tenant_id",
    "get_request_user_id",
    "require_tenant_id",
    "require_user_id",
]
