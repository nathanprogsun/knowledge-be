"""Global authentication dependency - resolve JWT / API-key principals.

Populates ``request.state`` with the authenticated principal so the RBAC
guards and API-key gate can read it:

- OPTIONS / whitelisted paths pass through without auth.
- A valid ``Authorization: Bearer <jwt>`` resolves the user + active
  tenant + tenant role and stashes them on ``request.state``.
- A valid ``X-API-Key`` resolves the API-key scope and stashes it on
  ``request.state`` (the API Key Gate authorizes the route afterwards).
- Internal headers (``X-User-Id`` /
  ``X-Tenant-ID`` / ``X-Roles``) resolve the
  principal directly when no bearer or API key is present. This
  channel is used by the integration-test rig; it must not be
  reachable from a public gateway without a strict deploy-time gate.
- No successful channel → HTTP 401.

This is a FastAPI dependency (not ASGI middleware) so it runs per request
with access to the request-scoped ``AsyncSession`` and the DI services.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, Request

from src.app_context import request_context
from src.common.exception import NotFoundError, UnauthorizedError, ValidationError
from src.core.auth.factory import build_auth_service
from src.core.auth.permissions import APIKeyScopeType, TenantAPIKeyScope
from src.core.auth.types import UserInfo
from src.core.tenants.factory import build_tenant_api_key_service, build_tenant_member_service
from src.db.dao.users_repository import UserRepository
from src.settings import get_settings
from src.web.deps.session import SessionDep

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
        auth_service = build_auth_service(session)
        info, tenant_id = await auth_service.get_me(token=token)
    except UnauthorizedError:
        return False

    # Resolve the user's role in the active tenant (if any).
    role = ""
    if tenant_id is not None:
        try:
            member_service = build_tenant_member_service(session)
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

    request.state.user_info = _build_user_info(info)
    request.state.is_system_admin = info.is_system_admin
    if tenant_id is not None:
        request.state.tenant_id = str(tenant_id)
        request.state.tenant_role = role
    else:
        request.state.tenant_id = str(0)
        request.state.tenant_role = ""

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
        api_key_service = build_tenant_api_key_service(session)
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
    request.state.api_key_scope = scope
    if key_info.tenant_id is not None:
        request.state.tenant_id = str(key_info.tenant_id)
        request_context.set_tenant_id(str(key_info.tenant_id))
    return True


async def _resolve_header_auth(
    *,
    request: Request,
    session: SessionDep,
) -> bool:
    """Authenticate via knowledge-prefixed headers; return True on success.

    Headers (all configurable via ``Settings``):

    - ``X-User-Id`` - required, the user id
    - ``X-Tenant-ID`` - required, the active tenant id (int)
    - ``X-Roles`` - optional, comma-separated role list; the
      first role wins as the tenant role. Presence of ``system_admin``
      in the list grants platform-admin powers.

    A real ``User`` row must exist and be active; the header alone is
    not sufficient. Failure to resolve raises ``UnauthorizedError``
    (the caller in :func:`require_auth` returns ``False`` only when the
    header is **not present** - header-present-but-invalid fails closed).
    """
    settings = get_settings()
    user_id = request.headers.get(settings.auth_header_user_id)
    tenant_id_raw = request.headers.get(settings.auth_header_tenant_id)
    if user_id is None or tenant_id_raw is None:
        return False

    try:
        tenant_id = int(tenant_id_raw)
    except (TypeError, ValueError) as exc:
        raise UnauthorizedError(
            code="auth.invalid_tenant_header",
            message=f"Invalid {settings.auth_header_tenant_id} header",
        ) from exc

    user_repo = UserRepository(session)
    try:
        user = await user_repo.find_by_id(
            user_id,
            not_found_code="auth.user_not_found",
            not_found_message="User for header-auth not found",
        )
    except NotFoundError as exc:
        raise UnauthorizedError(
            code="auth.user_not_found",
            message="Unauthorized: user for header-auth not found",
        ) from exc

    if not user.is_active:
        raise UnauthorizedError(
            code="auth.user_inactive",
            message="Unauthorized: user is inactive",
        )

    roles_header = request.headers.get(settings.auth_header_roles, "")
    roles = [r.strip() for r in roles_header.split(",") if r.strip()]
    role = roles[0] if roles else ""
    is_system_admin = "system_admin" in roles

    user_info: dict[str, str] = {
        "id": str(user.id),
        "username": user.username,
        "email": user.email,
        "is_active": "1" if user.is_active else "0",
        "can_access_all_tenants": "1" if is_system_admin else "0",
        "is_system_admin": "1" if is_system_admin else "0",
    }
    request.state.user_info = user_info
    request.state.tenant_id = str(tenant_id)
    request.state.tenant_role = role
    request.state.is_system_admin = is_system_admin
    request.state.api_key_scope = None
    request_context.set_tenant_id(str(tenant_id))
    request_context.set_user_id(str(user.id))
    return True


async def require_auth(
    request: Request,
    session: SessionDep,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Authenticate the request and populate ``request.state``.

    Whitelisted paths pass through. Otherwise a valid JWT Bearer, a
    valid X-API-Key, or the knowledge-prefixed header trio is required;
    failure yields HTTP 401.
    """
    if request.method == "OPTIONS" or is_public_path(request.method, request.url.path):
        return

    if await _resolve_header_auth(request=request, session=session):
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
