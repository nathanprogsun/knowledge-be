"""Internal DTOs for the `auth` domain.

These are service-output projections, not the HTTP wire shape. The
HTTP wire shape is in ``src/core/contracts/auth.py::AuthUser``.

`UserInfo` mirrors ``internal/types/user.go::UserInfo``: a stripped view
of the `users` row with the sensitive fields (``password_hash``,
``deleted_at``) removed. Services return ``UserInfo`` to the web layer
once authentication is complete; the password hash never leaves the
auth service.

`UserPreferences` mirrors ``types.UserPreferences``.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class UserPreferences(BaseModel):
    """Per-user UI/feature preferences. Stored as JSONB on Postgres."""

    model_config = ConfigDict(frozen=True)

    last_active_tenant_id: int | None = None


class UserInfo(BaseModel):
    """Wire-side projection of a `users` row.

    Mirrors ``internal/types/user.go::UserInfo``: the same field set as
    the storage ``User`` minus the sensitive columns
    (``password_hash``, ``deleted_at``). The service owns the
    ``User`` row during the login flow, then returns this
    ``UserInfo`` to the web layer.
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
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    created_at: datetime
    updated_at: datetime


__all__ = ["UserInfo", "UserPreferences"]
