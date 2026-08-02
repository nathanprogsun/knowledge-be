"""Auth HTTP endpoints - login, refresh, logout.

Maps the three auth endpoints from ``internal/handler/auth.go`` whose
service-layer support already exists in ``AuthService`` (PR-2):

- ``POST /auth/login`` - email + password -> ``LoginResponse``
- ``POST /auth/refresh`` - refresh token -> ``RefreshTokenResponse``
- ``POST /auth/logout`` - Bearer token -> ``{success, message}``

The remaining auth endpoints (register, /me, /change-password, OIDC)
depend on tenant service (PR-5), auth middleware (PR-12), or
``AuthService`` extensions not yet implemented; they land in later PRs.

Wire-shape conversion (``UserInfo`` -> ``AuthUser``, ``LoginResult`` ->
``LoginResponse``) lives in this module so the router stays declarative.
"""

from __future__ import annotations

from fastapi import APIRouter, Header
from pydantic import BaseModel, ConfigDict

from src.core.auth.types import UserInfo
from src.core.contracts.auth import (
    AuthUser,
    LoginRequest,
    LoginResponse,
    RefreshTokenRequest,
    RefreshTokenResponse,
)
from src.web.deps import AuthServiceDep

router = APIRouter(prefix="/auth", tags=["auth"])


# ── View models (inline; not contracts because they're trivial wrappers) ──


class LogoutResponse(BaseModel):
    """Wire shape for ``POST /auth/logout`` - matches Go's inline ``gin.H``."""

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str


# ── Conversion helpers ─────────────────────────────────────────────


def _user_info_to_auth_user(info: UserInfo) -> AuthUser:
    """Project the service-layer ``UserInfo`` to the wire ``AuthUser``.

    ``deleted_at`` is always ``None`` on the wire - the service strips it
    before returning ``UserInfo``. ``preferences`` is re-serialised from
    the typed ``UserPreferences`` model to the wire ``dict[str, object]``.
    """
    return AuthUser(
        id=info.id,
        username=info.username,
        email=info.email,
        avatar=info.avatar,
        tenant_id=info.tenant_id,
        is_active=info.is_active,
        can_access_all_tenants=info.can_access_all_tenants,
        is_system_admin=info.is_system_admin,
        preferences=info.preferences.model_dump(mode="json"),
        created_at=info.created_at,
        updated_at=info.updated_at,
        deleted_at=None,
    )


# ── Endpoints ──────────────────────────────────────────────────────


@router.post("/login", response_model=LoginResponse)
async def login(
    body: LoginRequest,
    auth_service: AuthServiceDep,
) -> LoginResponse:
    """Authenticate with email + password and receive an access/refresh pair.

    The ``active_tenant`` field is ``null`` until the tenant service
    (PR-5) is wired in; the Go handler populates it from the user's
    active tenant, which we don't yet resolve.
    """
    result = await auth_service.login(email=body.email, password=body.password)
    return LoginResponse(
        success=True,
        message="Login successful",
        user=_user_info_to_auth_user(result.user),
        active_tenant=None,
        memberships=[],
        token=result.access_token,
        refresh_token=result.refresh_token,
    )


@router.post("/refresh", response_model=RefreshTokenResponse)
async def refresh_token(
    body: RefreshTokenRequest,
    auth_service: AuthServiceDep,
) -> RefreshTokenResponse:
    """Exchange a refresh token for a new access/refresh pair."""
    result = await auth_service.refresh(refresh_token=body.refresh_token)
    return RefreshTokenResponse(
        success=True,
        message="Token refreshed successfully",
        access_token=result.access_token,
        refresh_token=result.refresh_token,
    )


@router.post("/logout", response_model=LogoutResponse)
async def logout(
    auth_service: AuthServiceDep,
    authorization: str | None = Header(default=None),
) -> LogoutResponse:
    """Revoke every outstanding token for the Bearer token's owner.

    Accepts ``Authorization: Bearer <jwt>``. Expired or invalid tokens
    are accepted as long as they decode - mirroring the Go handler so
    clients can log out even after the access token TTL.
    """
    if not authorization:
        raise _missing_auth_header()
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0] != "Bearer":
        raise _invalid_auth_header()
    token = parts[1]
    await auth_service.logout(token=token)
    return LogoutResponse(success=True, message="Logout successful")


# ── Local error helpers (avoid circular import with exception_handler) ──


def _missing_auth_header() -> Exception:
    from src.common.exception import ValidationError

    return ValidationError(
        code="auth.missing_authorization",
        message="Authorization header is required",
    )


def _invalid_auth_header() -> Exception:
    from src.common.exception import ValidationError

    return ValidationError(
        code="auth.invalid_authorization",
        message="Invalid Authorization header format",
    )


__all__ = ["router"]
