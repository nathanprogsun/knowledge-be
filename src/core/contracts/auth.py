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
    model_config = ConfigDict(frozen=True, populate_by_name=True)

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
    """Registration response containing the user, active tenant, and memberships.

    ``active_tenant`` identifies the current tenant, while ``memberships``
    carries the caller's role for each tenant they belong to.
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str
    user: AuthUser
    active_tenant: Tenant | None = Field(default=None)
    memberships: list[Membership] = Field(default_factory=list)


class LoginResponse(BaseModel):
    """Login response containing session tokens and tenant context."""

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

    It has the same shape as ``LoginResponse`` because both flows establish
    an authenticated session.
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str
    user: AuthUser
    active_tenant: Tenant | None = Field(default=None)
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
