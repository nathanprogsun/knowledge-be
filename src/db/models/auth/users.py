"""Storage row for the `users` table.

The Python `User` and the wire-side projection `UserInfo` (in
`src/core/auth/types.py`) are layered: `User` is the full storage row
(including `password_hash`); `UserInfo` drops `password_hash` and
`deleted_at`. The boundary translation lives in the auth service.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import Field

from src.common.table_model import TableModel


class User(TableModel):
    """One row of the `users` table."""

    table: ClassVar[str] = "users"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ("preferences",)

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


__all__ = ["User"]
