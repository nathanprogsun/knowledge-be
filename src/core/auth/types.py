"""Internal DTOs for the `auth` domain.

`UserDTO` is the read-side view of `UserRow`: same column set, but with
the password hash exposed as optional (services should never pass a hash
through a DTO unless they have already authenticated the caller).

`UserPreferences` mirrors the Go `types.UserPreferences` struct: today
it carries only `last_active_tenant_id`, with room for future UI knobs.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from src.common.table_model import TableModel


class UserPreferences(TableModel):
    """Per-user UI/feature preferences. Stored as JSONB on Postgres."""

    last_active_tenant_id: int | None = None


class UserDTO(TableModel):
    """Read-side projection of a `users` row."""

    id: str
    username: str
    email: str
    password_hash: str | None = None
    avatar: str | None = None
    tenant_id: int | None = None
    is_active: bool = True
    can_access_all_tenants: bool = False
    is_system_admin: bool = False
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


__all__ = ["UserDTO", "UserPreferences"]
