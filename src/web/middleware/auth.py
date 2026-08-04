"""Global authentication dependency — resolve JWT / API-key principals.

Populates ``request.state`` with the authenticated principal so the RBAC
guards and API-key gate can read it. Mirrors WeKnora's ``middleware.Auth``:

- OPTIONS / whitelisted paths pass through without auth.
- A valid ``Authorization: Bearer <jwt>`` resolves the user + active
  tenant + tenant role and stashes them on ``request.state``.
- A valid ``X-API-Key`` resolves the API-key scope and stashes it on
  ``request.state`` (the API Key Gate authorizes the route afterwards).
- No successful channel → HTTP 401.

This is a FastAPI dependency (not ASGI middleware) so it runs per request
with access to the request-scoped ``AsyncSession`` and the DI services.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, Request

from src.app_context import request_context
from src.common.exception import UnauthorizedError, ValidationError
from src.core.auth.permissions import APIKeyScopeType, TenantAPIKeyScope
from src.core.auth.service import AuthService
from src.core.auth.types import UserInfo
from src.core.tenants.api_key_service import TenantAPIKeyService
from src.core.tenants.member_service import TenantMemberService
from src.db.dao.auth_tokens_repository import AuthTokenRepository
from src.db.dao.tenant_api_keys_repository import TenantAPIKeyRepository
from src.db.dao.tenant_members_repository import TenantMemberRepository
from src.db.dao.users_repository import UserRepository
from src.web.deps.session import SessionDep
from src.web.middleware.context import (
    set_api_key_scope,
    set_is_system_admin,
    set_tenant_id,
    set_tenant_role,
    set_user_info,
)

# Routes that need no authentication. FastAPI paths use ``{param}``;
# exact-match entries here must equal the registered route path.
PUBLIC_PATHS: dict[str, set[str]] = {
    "/health": {"GET"},
    "/auth/register": {"POST"},
    "/auth/login": {"POST"},
    "/auth/refresh": {"POST"},
    "/auth/oidc/config": {"GET"},
    "/auth/oidc/url": {"GET"},
    "/auth/oidc/callback": {"GET"},
}


def is_public_path(method: str, path: str) -> bool:
    """True when the (method, path) needs no authentication."""
    allowed = PUBLIC_PATHS.get(path)
    return allowed is not None and method.upper() in allowed


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


def _build_user_info(info: UserInfo) -> dict[str, str]:
    """Project a ``UserInfo`` to the compact ``request.state`` dict."""
    return {
        "id": info.id,
        "username": info.username,
        "email": info.email,
        "is_active": "1" if info.is_active else "0",
        "can_access_all_tenants": "1" if info.can_access_all_tenants else "0",
        "is_system_admin": "1" if info.is_system_admin else "0",
    }


async def _resolve_jwt(
    *,
    request: Request,
    token: str,
    session: SessionDep,
) -> bool:
    """Attempt JWT authentication; return True on success."""
    try:
        users_repo = UserRepository(session)
        tokens_repo = AuthTokenRepository(session)
        auth_service = AuthService(users_repo=users_repo, tokens_repo=tokens_repo)
        info, tenant_id = await auth_service.get_me(token=token)
    except UnauthorizedError:
        return False

    # Resolve the user's role in the active tenant (if any).
    role = ""
    if tenant_id is not None:
        try:
            member_service = TenantMemberService(members_repo=TenantMemberRepository(session))
            membership = await member_service.get_membership(
                user_id=info.id,
                tenant_id=tenant_id,
            )
            role = membership.role if membership is not None else ""
        except Exception:
            # The membership table may not exist in minimal test schemas,
            # or the session lacks a live membership. Roll back so the
            # failed query does not poison the request's transaction, and
            # fail to an empty role (fail-closed for role gates).
            await session.rollback()
            role = ""

    set_user_info(request, _build_user_info(info))
    set_is_system_admin(request, info.is_system_admin)
    if tenant_id is not None:
        set_tenant_id(request, tenant_id)
        set_tenant_role(request, role)
    else:
        set_tenant_id(request, 0)
        set_tenant_role(request, "")

    # Populate the contextvar store for context-aware endpoints.
    request_context.set_tenant_id(str(tenant_id) if tenant_id is not None else "")
    request_context.set_user_id(info.id)
    return True


async def _resolve_api_key(
    *,
    request: Request,
    api_key: str,
    session: SessionDep,
) -> bool:
    """Attempt X-API-Key authentication; return True on success."""
    try:
        api_key_service = TenantAPIKeyService(api_keys_repo=TenantAPIKeyRepository(session))
        key_info = await api_key_service.authenticate(api_key)
    except Exception:
        return False

    scope = TenantAPIKeyScope(
        key_id=key_info.id,
        scope_type=(
            APIKeyScopeType.PLATFORM
            if key_info.scope_type == APIKeyScopeType.PLATFORM
            else APIKeyScopeType.TENANT
        ),
        full_access=key_info.full_access,
        knowledge_base_ids=key_info.knowledge_base_ids,
        capabilities=key_info.capabilities,
    )
    set_api_key_scope(request, scope)
    if key_info.tenant_id is not None:
        set_tenant_id(request, key_info.tenant_id)
        request_context.set_tenant_id(str(key_info.tenant_id))
    return True


async def require_auth(
    request: Request,
    session: SessionDep,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Authenticate the request and populate ``request.state``.

    Whitelisted paths pass through. Otherwise a valid JWT Bearer or
    X-API-Key is required; failure yields HTTP 401.
    """
    if request.method == "OPTIONS" or is_public_path(request.method, request.url.path):
        return

    if authorization:
        if await _resolve_jwt(request=request, token=_bearer_token(authorization), session=session):
            return
        raise UnauthorizedError(
            code="auth.invalid_token",
            message="Unauthorized: invalid or expired token",
        )

    if x_api_key:
        if await _resolve_api_key(request=request, api_key=x_api_key, session=session):
            return
        raise UnauthorizedError(
            code="auth.invalid_api_key",
            message="Unauthorized: invalid API key",
        )

    raise UnauthorizedError(
        code="auth.missing_authentication",
        message="Unauthorized: missing authentication",
    )


AuthDep = Annotated[None, Depends(require_auth)]

__all__ = [
    "PUBLIC_PATHS",
    "AuthDep",
    "is_public_path",
    "require_auth",
]
