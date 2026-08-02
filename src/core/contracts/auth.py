from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


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


class AuthTenant(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    name: str
    api_key: str


class AuthUser(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    username: str
    email: str
    avatar: str | None = Field(default=None)
    tenant_id: int
    is_active: bool
    can_access_all_tenants: bool = False
    created_at: datetime
    updated_at: datetime


class RegisterResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool
    message: str
    user: AuthUser
    tenant: AuthTenant


class LoginResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool
    message: str
    user: AuthUser
    tenant: AuthTenant
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


__all__ = [
    "AuthTenant",
    "AuthUser",
    "ChangePasswordRequest",
    "LoginRequest",
    "LoginResponse",
    "MeResponse",
    "OIDCAuthorizeURLResponse",
    "OIDCMetaConfig",
    "RefreshTokenRequest",
    "RefreshTokenResponse",
    "RegisterRequest",
    "RegisterResponse",
    "ValidateTokenResponse",
]
