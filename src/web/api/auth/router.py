"""Auth HTTP endpoints - login, refresh, logout, register, me, OIDC."""

from __future__ import annotations

import secrets
from datetime import UTC, datetime

from fastapi import APIRouter, Header, Query, Request
from pydantic import BaseModel, ConfigDict

from src.common.exception import GoneError, NotFoundError, UnauthorizedError, ValidationError
from src.core.auth.types import UserInfo, UserPreferences
from src.core.contracts.auth import (
    AuthConfigResponse,
    AuthUser,
    ChangePasswordRequest,
    InvitationLookupRequest,
    InvitationLookupResponse,
    InviteLookup,
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
    RegisterByInviteRequest,
    RegisterRequest,
    RegisterResponse,
    UpdatePreferencesRequest,
    UpdatePreferencesResponse,
    ValidateTokenResponse,
)
from src.core.contracts.tenants import Membership
from src.core.tenants.service import TenantService
from src.core.tenants.types import MembershipInfo
from src.settings import get_settings
from src.web.api.tenants.views import TenantEnvelope, tenant_info_to_contract
from src.web.deps import AuthDep, AuthServiceDep, OidcServiceDep
from src.web.deps.system import SystemSettingServiceDep
from src.web.deps.tenants import (
    TenantInvitationServiceDep,
    TenantMemberServiceDep,
    TenantServiceDep,
)

router = APIRouter(prefix="/auth", tags=["auth"])

#: Default account bootstrapped by the Lite auto-setup flow (mirrors the
#: upstream contract's ``admin@weknora.local``).
AUTO_SETUP_DEFAULT_EMAIL = "admin@weknora.local"


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
    member_service: TenantMemberServiceDep,
    tenant_service: TenantServiceDep,
    request: Request,
    authorization: str | None = Header(default=None),
) -> MeResponse:
    """Return the authenticated user's profile, memberships, and tenant context.

    Response is wrapped in ``data`` per Go's
    ``internal/handler/auth.go`` ``GetMyInfo`` handler. Includes the
    *active* tenant (the one the auth middleware resolved against the
    X-Tenant-ID header, falling back to the user's home tenant), the
    user's memberships (so the frontend can restore ``currentTenantRole``
    on page refresh), ``tenant_required``, and ``capabilities``.
    """
    token = _require_bearer(authorization)
    info, _tenant_id = await auth_service.get_me(token=token)
    active_tenant_id = int(request.state.tenant_id or 0)
    if active_tenant_id <= 0:
        active_tenant_id = info.tenant_id or 0
    tenant = None
    if active_tenant_id > 0:
        try:
            tenant = tenant_info_to_contract(
                await tenant_service.get_tenant(active_tenant_id),
            )
        except NotFoundError:
            tenant = None
    memberships = await _memberships_to_contract(
        await member_service.list_by_user(info.id),
        tenant_service,
    )
    return MeResponse(
        success=True,
        data=MeData(
            user=_user_info_to_auth_user(info),
            tenant=tenant,
            memberships=memberships,
            tenant_required=tenant is None,
            capabilities=MeCapabilities(
                can_create_tenant=info.can_access_all_tenants or tenant is None,
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


# ── Registration-mode config / Lite auto-setup ────────────────────────


@router.get("/config", response_model=AuthConfigResponse)
async def auth_config(
    system_setting_service: SystemSettingServiceDep,
) -> AuthConfigResponse:
    """Return the public registration mode (no auth).

    Mirrors Go's ``GetAuthConfig``: the frontend reads it on app load
    to decide whether to show the Register tab. Only ``registration_mode``
    is exposed; other config stays internal.
    """
    mode = await system_setting_service.get_string(
        "auth.registration_mode",
        "",
        "self_serve",
    )
    return AuthConfigResponse(success=True, registration_mode=mode)


@router.post("/auto-setup", response_model=LoginResponse)
async def auto_setup(
    auth_service: AuthServiceDep,
    tenant_service: TenantServiceDep,
    member_service: TenantMemberServiceDep,
) -> LoginResponse:
    """Lite-mode transparent bootstrap: create the default user/workspace
    on first boot, then just sign them in on every subsequent boot.

    Mirrors Go's ``AutoSetup``: idempotent — the default account is
    created once, and later calls reuse it and mint a fresh token pair.
    """
    user_row = await auth_service.get_user_row_by_email(AUTO_SETUP_DEFAULT_EMAIL)
    if user_row is None:
        user_row = await auth_service.create_user(
            username=f"user_{secrets.token_hex(4)}",
            email=AUTO_SETUP_DEFAULT_EMAIL,
            password=secrets.token_urlsafe(24),
        )
    if user_row.tenant_id is None:
        tenant_info = await tenant_service.create_tenant(name=user_row.username)
        await member_service.ensure_owner(
            user_id=user_row.id,
            tenant_id=tenant_info.id,
        )
        user_row = await auth_service.update_home_tenant(
            user_id=user_row.id,
            tenant_id=tenant_info.id,
        )
    result = await auth_service.mint_pair_for_user_row(user_row)
    tenant = await tenant_service.get_tenant(user_row.tenant_id)
    memberships = await _memberships_to_contract(
        await member_service.list_by_user(user_row.id),
        tenant_service,
    )
    return LoginResponse(
        success=True,
        message="Auto-setup successful",
        user=_user_info_to_auth_user(result.user),
        active_tenant=tenant_info_to_contract(tenant),
        memberships=memberships,
        token=result.access_token,
        refresh_token=result.refresh_token,
    )


@router.get("/tenant", response_model=TenantEnvelope)
async def current_tenant(
    _auth: AuthDep,
    tenant_service: TenantServiceDep,
    request: Request,
) -> TenantEnvelope:
    """Return the authenticated user's active workspace."""
    tenant_id = int(request.state.tenant_id or 0)
    if tenant_id <= 0:
        raise NotFoundError(
            code="tenant.not_found",
            message="No active tenant",
        )
    tenant = await tenant_service.get_tenant(tenant_id)
    return TenantEnvelope(success=True, data=tenant_info_to_contract(tenant))


@router.put("/me/preferences", response_model=UpdatePreferencesResponse)
async def update_my_preferences(
    _auth: AuthDep,
    body: UpdatePreferencesRequest,
    auth_service: AuthServiceDep,
    authorization: str | None = Header(default=None),
) -> UpdatePreferencesResponse:
    """PATCH-merge the current user's preferences."""
    token = _require_bearer(authorization)
    info, _ = await auth_service.get_me(token=token)
    prefs = await auth_service.update_my_preferences(
        user_id=info.id,
        patch=UserPreferences(last_active_tenant_id=body.last_active_tenant_id),
    )
    return UpdatePreferencesResponse(success=True, data=prefs.model_dump(mode="json"))


# ── Share-link registration ───────────────────────────────────────────


@router.post("/invitations/lookup", response_model=InvitationLookupResponse)
async def invitation_lookup(
    body: InvitationLookupRequest,
    invitation_service: TenantInvitationServiceDep,
    tenant_service: TenantServiceDep,
) -> InvitationLookupResponse:
    """Resolve a share-link token into the registration-page context.

    No auth. Invalid / expired / revoked tokens collapse to a single
    410 so a stolen token's failure mode does not leak which slot it
    occupied.
    """
    try:
        invite = await invitation_service.lookup_by_token(body.token)
    except NotFoundError as exc:
        raise GoneError(
            message="Invitation link is invalid or has been revoked",
        ) from exc
    tenant_name: str | None = None
    try:
        tenant = await tenant_service.get_tenant(invite.tenant_id)
        tenant_name = tenant.name
    except NotFoundError:
        pass
    return InvitationLookupResponse(
        success=True,
        data=InviteLookup(
            tenant_id=invite.tenant_id,
            tenant_name=tenant_name,
            role=invite.role,
            expires_at=invite.expires_at,
        ),
    )


@router.post("/register-by-invite", response_model=LoginResponse, status_code=201)
async def register_by_invite(
    body: RegisterByInviteRequest,
    auth_service: AuthServiceDep,
    invitation_service: TenantInvitationServiceDep,
    tenant_service: TenantServiceDep,
    member_service: TenantMemberServiceDep,
) -> LoginResponse:
    """Complete registration via a share-link token.

    The invitee supplies their own email — the token is the
    authorisation, not an identity lock. Not subject to the
    invite-only gate: the token IS the authorisation.
    """
    try:
        invite = await invitation_service.lookup_by_token(body.token)
    except NotFoundError as exc:
        raise GoneError(
            message="Invitation link is invalid or has been revoked",
        ) from exc
    user_row = await auth_service.create_user(
        username=body.username,
        email=body.email,
        password=body.password,
    )
    user_row = await auth_service.update_home_tenant(
        user_id=user_row.id,
        tenant_id=invite.tenant_id,
    )
    await invitation_service.accept_by_token(body.token, user_id=user_row.id)
    result = await auth_service.mint_pair_for_user_row(user_row)
    tenant = await tenant_service.get_tenant(user_row.tenant_id)
    memberships = await _memberships_to_contract(
        await member_service.list_by_user(user_row.id),
        tenant_service,
    )
    return LoginResponse(
        success=True,
        message="Registration successful",
        user=_user_info_to_auth_user(result.user),
        active_tenant=tenant_info_to_contract(tenant),
        memberships=memberships,
        token=result.access_token,
        refresh_token=result.refresh_token,
    )


async def _memberships_to_contract(
    memberships: list[MembershipInfo],
    tenant_service: TenantService,
) -> list[Membership]:
    """Hydrate membership rows with tenant names into the wire contract."""
    result: list[Membership] = []
    for membership in memberships:
        tenant_name = ""
        try:
            tenant = await tenant_service.get_tenant(membership.tenant_id)
            tenant_name = tenant.name
        except NotFoundError:
            pass
        result.append(
            Membership(
                tenant_id=membership.tenant_id,
                tenant_name=tenant_name,
                role=membership.role,
            ),
        )
    return result


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
