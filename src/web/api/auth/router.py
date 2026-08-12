"""Auth HTTP endpoints - login, refresh, logout, register, me, OIDC."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Header, Query
from pydantic import BaseModel, ConfigDict

from src.common.exception import UnauthorizedError, ValidationError
from src.core.auth.types import UserInfo
from src.core.contracts.auth import (
    AuthUser,
    ChangePasswordRequest,
    LoginRequest,
    LoginResponse,
    MeCapabilities,
    MeData,
    MeResponse,
    OIDCAuthorizeURLResponse,
    OIDCCallbackResponse,
    OIDCMetaConfig,
    RefreshTokenRequest,
    RefreshTokenResponse,
    RegisterRequest,
    RegisterResponse,
    ValidateTokenResponse,
)
from src.settings import get_settings
from src.web.deps import AuthDep, AuthServiceDep, OidcServiceDep

router = APIRouter(prefix="/auth", tags=["auth"])


# ── View models (inline; not contracts because they're trivial wrappers) ──


class LogoutResponse(BaseModel):
    """Wire shape for ``POST /auth/logout``."""

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str


# ── Conversion helpers ─────────────────────────────────────────────


def _user_info_to_auth_user(info: UserInfo) -> AuthUser:
    """Project the service-layer ``UserInfo`` to the wire ``AuthUser``."""
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
    """Authenticate with email + password and return an access/refresh pair."""
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
    _auth: AuthDep,
    auth_service: AuthServiceDep,
    authorization: str | None = Header(default=None),
) -> LogoutResponse:
    """Revoke every outstanding token for the Bearer token's owner.

    Accepts ``Authorization: Bearer <jwt>``. Expired or invalid tokens
    are accepted as long as they decode, so clients can log out even
    after the access token TTL.
    """
    if not authorization:
        raise _missing_auth_header()
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0] != "Bearer":
        raise _invalid_auth_header()
    token = parts[1]
    await auth_service.logout(token=token)
    return LogoutResponse(success=True, message="Logout successful")


# ── Registration / profile / token validation ─────────────────────────


@router.post("/register", response_model=RegisterResponse)
async def register(
    body: RegisterRequest,
    auth_service: AuthServiceDep,
) -> RegisterResponse:
    """Create a user and establish a session."""
    result = await auth_service.register(
        username=body.username,
        email=body.email,
        password=body.password,
    )
    return RegisterResponse(
        success=True,
        message="Registration successful",
        user=_user_info_to_auth_user(result.user),
        tenant=None,
        memberships=[],
    )


@router.get("/me", response_model=MeResponse)
async def me(
    _auth: AuthDep,
    auth_service: AuthServiceDep,
    authorization: str | None = Header(default=None),
) -> MeResponse:
    """Return the authenticated user's profile, memberships, and tenant context.

    Response is wrapped in ``data`` per Go's
    ``internal/handler/auth.go`` ``GetMyInfo`` handler. Includes
    ``tenant_required`` (true when the user has no active tenant yet)
    and ``capabilities.can_create_tenant`` so the frontend can drive
    the post-login redirect.
    """
    token = _require_bearer(authorization)
    info, _tenant_id = await auth_service.get_me(token=token)
    tenant_required = info.tenant_id is None
    return MeResponse(
        success=True,
        data=MeData(
            user=_user_info_to_auth_user(info),
            tenant=None,
            memberships=[],
            tenant_required=tenant_required,
            capabilities=MeCapabilities(
                can_create_tenant=info.can_access_all_tenants,
            ),
        ),
    )


@router.post("/change-password", response_model=LogoutResponse)
async def change_password(
    _auth: AuthDep,
    body: ChangePasswordRequest,
    auth_service: AuthServiceDep,
    authorization: str | None = Header(default=None),
) -> LogoutResponse:
    """Verify the current password and replace it with a new one."""
    token = _require_bearer(authorization)
    info, _ = await auth_service.get_me(token=token)
    await auth_service.change_password(
        user_id=info.id,
        old_password=body.old_password,
        new_password=body.new_password,
    )
    return LogoutResponse(success=True, message="Password changed")


@router.get("/validate", response_model=ValidateTokenResponse)
async def validate_token(
    _auth: AuthDep,
    auth_service: AuthServiceDep,
    authorization: str | None = Header(default=None),
) -> ValidateTokenResponse:
    """Validate a Bearer access token.

    On success the full ``user`` object is returned (mirrors Go's
    ``ValidateToken`` handler in ``internal/handler/auth.go``); the
    legacy ``{valid, user_id, tenant_id}`` minimal shape is dropped.
    """
    token = _require_bearer(authorization)
    try:
        info, _tenant_id = await auth_service.validate_token(token=token)
    except UnauthorizedError:
        return ValidateTokenResponse(success=True, message="Token is invalid", user=None)
    return ValidateTokenResponse(
        success=True,
        message="Token is valid",
        user=_user_info_to_auth_user(info),
    )


# ── OIDC SSO ───────────────────────────────────────────────────────────


@router.get("/oidc/config", response_model=OIDCMetaConfig)
async def oidc_config() -> OIDCMetaConfig:
    """Return OIDC provider metadata.

    Mirrors Go's ``OIDCConfigResponse``: when OIDC is not enabled,
    ``provider_display_name`` is omitted (Go's ``omitempty`` / Python
    serialises ``None`` as ``null`` and we route the value through
    ``None`` so the JSON shape matches).
    """
    settings = get_settings()
    display_name = (
        settings.oidc_provider_display_name
        if settings.oidc_enable and settings.oidc_provider_display_name
        else None
    )
    return OIDCMetaConfig(
        success=True,
        enabled=settings.oidc_enable,
        provider_display_name=display_name,
    )


@router.get("/oidc/url", response_model=OIDCAuthorizeURLResponse)
async def oidc_url(
    oidc_service: OidcServiceDep,
    redirect_uri: str = Query(...),
) -> OIDCAuthorizeURLResponse:
    """Build the provider authorize URL + signed state."""
    result = await oidc_service.get_authorization_url(redirect_uri=redirect_uri)
    return OIDCAuthorizeURLResponse(
        success=True,
        provider_display_name=result.provider_display_name,
        authorization_url=result.authorization_url,
        state=result.state,
    )


@router.get("/oidc/callback", response_model=OIDCCallbackResponse)
async def oidc_callback(
    oidc_service: OidcServiceDep,
    code: str = Query(...),
    redirect_uri: str = Query(...),
) -> OIDCCallbackResponse:
    """Exchange an OIDC authorization code for a session."""
    result = await oidc_service.login_with_oidc(
        code=code,
        redirect_uri=redirect_uri,
    )
    if not result.success or result.user is None:
        return OIDCCallbackResponse(
            success=False,
            message=result.message,
            user=_empty_auth_user(),
            token="",
            refresh_token="",
            is_new_user=False,
        )
    return OIDCCallbackResponse(
        success=True,
        message=result.message,
        user=_user_info_to_auth_user(result.user),
        tenant=None,
        memberships=[],
        token=result.access_token,
        refresh_token=result.refresh_token,
        is_new_user=bool(getattr(result, "is_new_user", False)),
    )


# ── Local error helpers (avoid circular import with exception_handler) ──


def _require_bearer(authorization: str | None) -> str:
    """Return the Bearer token, or raise a 422-style validation error."""
    if not authorization:
        raise _missing_auth_header()
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0] != "Bearer":
        raise _invalid_auth_header()
    return parts[1]


def _empty_auth_user() -> AuthUser:
    return AuthUser(
        id="",
        username="",
        email="",
        is_active=False,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _missing_auth_header() -> Exception:
    return ValidationError(
        code="auth.missing_authorization",
        message="Authorization header is required",
    )


def _invalid_auth_header() -> Exception:
    return ValidationError(
        code="auth.invalid_authorization",
        message="Invalid Authorization header format",
    )


__all__ = ["router"]
