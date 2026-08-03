"""Web-layer FastAPI dependency factories.

One module per domain: ``session`` owns the request-scoped ``AsyncSession``;
every other module builds that domain's repositories + service on top of
it. This package re-exports the public names so callers keep importing
from ``src.web.deps`` regardless of which module a dependency lives in.
"""

from __future__ import annotations

from src.web.deps.auth import AuthServiceDep, get_auth_service
from src.web.deps.session import SessionDep, get_async_session
from src.web.deps.tenants import TenantServiceDep, get_tenant_service

__all__ = [
    "AuthServiceDep",
    "SessionDep",
    "TenantServiceDep",
    "get_async_session",
    "get_auth_service",
    "get_tenant_service",
]
