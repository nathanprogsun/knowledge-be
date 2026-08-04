"""Auth-domain FastAPI dependency factories.

One-line forwarders to the ``core`` builders: repositories are assembled
in ``src.core.auth.factory`` on the request-scoped ``AsyncSession`` (the
request's reads and writes form a single unit of work); APP-scope
singletons come from the lifespan registry. ``web`` never imports ``db``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from src.app_context.registry import get_oidc_client_from_lifespan
from src.core.auth.factory import build_auth_service, build_oidc_service
from src.core.auth.oidc import OidcService
from src.core.auth.service import AuthService
from src.web.deps.session import SessionDep
from src.web.middleware.auth import require_auth


def get_auth_service(session: SessionDep) -> AuthService:
    """Build a per-request ``AuthService`` on the shared session."""
    return build_auth_service(session)


def get_oidc_service(request: Request, session: SessionDep) -> OidcService:
    """Build a per-request ``OidcService``.

    Repos are request-scoped (they bind the per-request session); the
    ``OidcClient`` is the APP-scope singleton from the lifespan registry
    so its pooled ``httpx.AsyncClient`` is shared across requests.
    """
    return build_oidc_service(session, oidc_client=get_oidc_client_from_lifespan(request.app))


AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
OidcServiceDep = Annotated[OidcService, Depends(get_oidc_service)]

# Resolve + populate the authenticated principal's request context
# (user, tenant, role, api-key scope) via the global auth dependency.
CurrentUserContextDep = Annotated[None, Depends(require_auth)]


__all__ = [
    "AuthServiceDep",
    "CurrentUserContextDep",
    "OidcServiceDep",
    "get_auth_service",
    "get_oidc_service",
]
