"""Row shape for the `users` table.

Mirrors `internal/types/user.go` field-for-field: a globally-unique user
with username + email as natural keys, an opaque bcrypt hash, a nullable
`tenant_id` for users that exist before being provisioned into a
workspace, an `is_system_admin` flag independent of tenant roles, and a
`preferences` JSON blob (Postgres `JSONB`) carrying per-user UI knobs.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from src.common.table_model import TableModel


class UserRow(TableModel):
    """One row of the `users` table."""

    table: str = "users"

    id: str
    username: str
    email: str
    password_hash: str
    avatar: str | None = None
    tenant_id: int | None = None
    is_active: bool = True
    can_access_all_tenants: bool = False
    is_system_admin: bool = False
    preferences: dict[str, object] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


__all__ = ["UserRow"]
