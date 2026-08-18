from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject
from src.core.contracts.tenants import (
    Membership,
    Tenant,
)


class RegisterRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    username: str
    email: str
    # Go ``types/user.go``: ``binding:"required,min=6"``.
    password: str = Field(min_length=6)


class LoginRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    email: str
    password: str


class RefreshTokenRequest(BaseModel):
    """Body for ``POST /auth/refresh``.

    Strict camelCase to match Go's
    ``{RefreshToken string json:"refreshToken" binding:"required"}``.
    The ``refresh_token`` snake_case form is rejected to keep the wire
    contract identical on both sides.
    """

    model_config = ConfigDict(frozen=True)

    refresh_token: str = Field(alias="refreshToken")


class ChangePasswordRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    old_password: str
    # Go ``handler/auth.go``: ``binding:"required,min=6"``.
    new_password: str = Field(min_length=6)


class AuthUser(BaseModel):
    """HTTP wire representation of an authenticated user.

    The password hash is intentionally omitted from this serialized model.
    ``is_system_admin``, ``preferences``, and ``deleted_at`` are included in
    the wire representation.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    username: str
    email: str
    avatar: str | None = Field(default=None)
    tenant_id: int | None = Field(default=None)
    is_active: bool
    can_access_all_tenants: bool = False
    is_system_admin: bool = False
    preferences: JsonObject = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = Field(default=None)


class RegisterResponse(BaseModel):
    """Registration response. Field names mirror Go's contract."""

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str
    user: AuthUser
    tenant: Tenant | None = Field(default=None)
    memberships: list[Membership] = Field(default_factory=list)


class LoginResponse(BaseModel):
    """Login response containing session tokens and tenant context.

    The active tenant is the workspace whose ID is encoded in the
    issued JWT; future requests are scoped to it until the client
    calls /auth/switch-tenant. Field name mirrors Go's
    ``LoginResponse`` (active_tenant, not tenant).
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str
    user: AuthUser
    active_tenant: Tenant | None = Field(default=None)
    memberships: list[Membership] = Field(default_factory=list)
    token: str
    refresh_token: str


class OIDCCallbackResponse(BaseModel):
    """OIDC callback response containing session tokens and tenant context.

    Mirrors Go's ``OIDCCallbackResponse`` in ``internal/types/user.go``.
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str
    user: AuthUser
    tenant: Tenant | None = Field(default=None)
    memberships: list[Membership] = Field(default_factory=list)
    token: str
    refresh_token: str
    is_new_user: bool = False


class RefreshTokenResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool
    message: str
    access_token: str
    refresh_token: str


class OIDCMetaConfig(BaseModel):
    """Body for ``GET /auth/oidc/config``.

    Mirrors Go's ``OIDCConfigResponse`` — ``provider_display_name`` is
    omitted (omitempty) when the OIDC provider is not configured, so
    the field is ``Optional`` here and serialised as ``None`` when the
    value is missing.
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    enabled: bool
    provider_display_name: str | None = None


class OIDCAuthorizeURLResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool
    provider_display_name: str
    authorization_url: str
    state: str


class ValidateTokenResponse(BaseModel):
    """Token validation response. Returns the full user object on success.

    Mirrors Go's ``internal/handler/auth.go`` ``ValidateToken`` which
    emits ``{success, message, user}`` rather than the legacy
    ``{valid, user_id, tenant_id}`` minimal shape.
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str | None = Field(default=None)
    user: AuthUser | None = Field(default=None)


class MeCapabilities(BaseModel):
    """Capability flags attached to ``/auth/me`` responses."""

    model_config = ConfigDict(frozen=True)

    can_create_tenant: bool = False


class MeData(BaseModel):
    """The ``data`` envelope for ``/auth/me`` responses.

    Mirrors Go's ``gin.H{"data": {...}}`` shape in
    ``internal/handler/auth.go``: ``user``, ``tenant``, ``memberships``,
    ``tenant_required`` (true when the user has no active tenant yet),
    and ``capabilities``.
    """

    model_config = ConfigDict(frozen=True)

    user: AuthUser
    tenant: Tenant | None = Field(default=None)
    memberships: list[Membership] = Field(default_factory=list)
    tenant_required: bool = False
    capabilities: MeCapabilities = Field(default_factory=MeCapabilities)


class MeResponse(BaseModel):
    """``/auth/me`` response. The substantive payload is wrapped in ``data``."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: MeData


class AuthConfigResponse(BaseModel):
    """Public registration-mode config read on app load (no auth).

    Mirrors Go's ``GetAuthConfig``: only ``registration_mode`` is
    exposed; the frontend uses it to decide whether to show the
    Register tab.
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    registration_mode: str


class UpdatePreferencesRequest(BaseModel):
    """PATCH body for ``PUT /auth/me/preferences``.

    Fields are optional so the handler can distinguish "key not
    present" (preserve existing value) from "explicit value"; send
    ``last_active_tenant_id=0`` to clear the preference. Mirrors Go's
    ``updateMyPreferencesRequest``.
    """

    model_config = ConfigDict(frozen=True)

    last_active_tenant_id: int | None = Field(default=None)


class UpdatePreferencesResponse(BaseModel):
    """The merged preferences after a PATCH update."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: JsonObject = Field(default_factory=dict)


class InvitationLookupRequest(BaseModel):
    """Body for ``POST /auth/invitations/lookup``.

    The token travels in the body (not the path) so it never lands in
    access logs / browser history / tracing spans.
    """

    model_config = ConfigDict(frozen=True)

    token: str


class InviteLookup(BaseModel):
    """Public projection of a share-link row for the registration page.

    Narrow on purpose: just enough to render "X invited you to Y"
    without leaking inviter audit fields.
    """

    model_config = ConfigDict(frozen=True)

    tenant_id: int
    tenant_name: str | None = Field(default=None)
    role: str
    expires_at: datetime


class InvitationLookupResponse(BaseModel):
    """``POST /auth/invitations/lookup`` response."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: InviteLookup | None = Field(default=None)
    message: str | None = Field(default=None)


class RegisterByInviteRequest(BaseModel):
    """Body for ``POST /auth/register-by-invite``."""

    model_config = ConfigDict(frozen=True)

    token: str
    email: str
    username: str
    password: str = Field(min_length=6)


__all__ = [
    "AuthConfigResponse",
    "AuthUser",
    "ChangePasswordRequest",
    "InvitationLookupRequest",
    "InvitationLookupResponse",
    "InviteLookup",
    "LoginRequest",
    "LoginResponse",
    "MeCapabilities",
    "MeData",
    "MeResponse",
    "OIDCAuthorizeURLResponse",
    "OIDCCallbackResponse",
    "OIDCMetaConfig",
    "RefreshTokenRequest",
    "RefreshTokenResponse",
    "RegisterByInviteRequest",
    "RegisterRequest",
    "RegisterResponse",
    "UpdatePreferencesRequest",
    "UpdatePreferencesResponse",
    "ValidateTokenResponse",
]
