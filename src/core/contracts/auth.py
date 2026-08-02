from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.core.contracts.tenants import (
    Membership,
    Tenant,
)


class RegisterRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    email: str
    password: str


class RefreshTokenRequest(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    refresh_token: str = Field(alias="refreshToken")


class ChangePasswordRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    old_password: str
    new_password: str


class AuthUser(BaseModel):
    """HTTP wire user — mirrors Go ``types.User`` as serialized on the wire.

    The upstream ``AuthLoginResponse.User`` is typed ``*types.User``; the
    ``password_hash`` column is ``json:"-"`` (never serialized), so the
    wire shape is the full storage struct minus the hash. ``is_system_admin``,
    ``preferences`` and ``deleted_at`` all round-trip over the wire.
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
    preferences: dict[str, object] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = Field(default=None)


class RegisterResponse(BaseModel):
    """Mirrors ``handler/dto/auth.go::AuthRegisterResponse``.

    Field names match the Go DTO (``active_tenant`` not ``tenant``); the
    ``memberships`` list carries the caller's role across every tenant
    they belong to.
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str
    user: AuthUser
    active_tenant: Tenant
    memberships: list[Membership] = Field(default_factory=list)


class LoginResponse(BaseModel):
    """Mirrors ``handler/dto/auth.go::AuthLoginResponse``."""

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str
    user: AuthUser
    active_tenant: Tenant | None = Field(default=None)
    memberships: list[Membership] = Field(default_factory=list)
    token: str
    refresh_token: str


class OIDCCallbackResponse(BaseModel):
    """Mirrors ``handler/dto/auth.go::AuthOIDCCallbackResponse``.

    Same shape as ``LoginResponse`` — the OIDC code-exchange path ends
    in the same downstream session-establishment state.
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str
    user: AuthUser
    active_tenant: Tenant
    memberships: list[Membership] = Field(default_factory=list)
    token: str
    refresh_token: str


class RefreshTokenResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool
    message: str
    access_token: str
    refresh_token: str


class OIDCMetaConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool
    enabled: bool
    provider_display_name: str


class OIDCAuthorizeURLResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool
    provider_display_name: str
    authorization_url: str
    state: str


class ValidateTokenResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool
    valid: bool
    user_id: str | None = Field(default=None)
    tenant_id: int | None = Field(default=None)


class MeResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool
    user: AuthUser
    active_tenant: Tenant | None = Field(default=None)
    memberships: list[Membership] = Field(default_factory=list)


__all__ = [
    "AuthUser",
    "ChangePasswordRequest",
    "LoginRequest",
    "LoginResponse",
    "MeResponse",
    "OIDCAuthorizeURLResponse",
    "OIDCCallbackResponse",
    "OIDCMetaConfig",
    "RefreshTokenRequest",
    "RefreshTokenResponse",
    "RegisterRequest",
    "RegisterResponse",
    "ValidateTokenResponse",
]
