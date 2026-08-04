"""Auth-domain FastAPI dependency factories.

Builds a per-request ``AuthService`` from fresh ``UserRepository`` and
``AuthTokenRepository`` instances sharing the request-scoped
``AsyncSession``. The service is request-scoped and never shared
across requests.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header

from src.ai.oidc_client import OidcClient
from src.app_context import request_context
from src.common.exception import UnauthorizedError, ValidationError
from src.core.auth.oidc import OidcService
from src.core.auth.service import AuthService
from src.core.auth.types import UserInfo
from src.db.dao.auth_tokens_repository import AuthTokenRepository
from src.db.dao.users_repository import UserRepository
from src.web.deps.session import SessionDep


def get_auth_service(session: SessionDep) -> AuthService:
    """Build a per-request ``AuthService`` with fresh repos sharing the session."""
    users_repo = UserRepository(session)
    tokens_repo = AuthTokenRepository(session)
    return AuthService(users_repo=users_repo, tokens_repo=tokens_repo)


def get_oidc_service(session: SessionDep) -> OidcService:
    """Build a per-request ``OidcService`` with fresh repos + a shared client."""
    users_repo = UserRepository(session)
    tokens_repo = AuthTokenRepository(session)
    return OidcService(
        users_repo=users_repo,
        tokens_repo=tokens_repo,
        oidc_client=OidcClient(),
    )


AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
OidcServiceDep = Annotated[OidcService, Depends(get_oidc_service)]


def _bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise ValidationError(
            code="auth.missing_authorization",
            message="Authorization header is required",
        )
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0] != "Bearer":
        raise ValidationError(
            code="auth.invalid_authorization",
            message="Invalid Authorization header format",
        )
    return parts[1]


async def get_current_user_context(
    session: SessionDep,
    authorization: str | None = Header(default=None),
) -> None:
    """Resolve the Bearer principal and populate the request context.

    Sets ``request_context`` tenant_id / user_id from the validated access
    token so tenant-scoped endpoints (KV, GET /tenants) can read the
    authenticated principal without a dedicated auth middleware pass.
    """
    token = _bearer_token(authorization)
    auth_service = get_auth_service(session)
    try:
        info, tenant_id = await auth_service.get_me(token=token)
    except UnauthorizedError:
        raise ValidationError(
            code="auth.invalid_token",
            message="Token is invalid",
        ) from None
    _set_context(info, tenant_id)


def _set_context(info: UserInfo, tenant_id: int | None) -> None:
    request_context.set_tenant_id(str(tenant_id) if tenant_id is not None else "")
    request_context.set_user_id(info.id)


CurrentUserContextDep = Annotated[None, Depends(get_current_user_context)]


__all__ = [
    "AuthServiceDep",
    "CurrentUserContextDep",
    "OidcServiceDep",
    "get_auth_service",
    "get_current_user_context",
    "get_oidc_service",
]
