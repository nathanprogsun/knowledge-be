"""Storage row for the `users` table.

Mirrors `internal/types/user.go::User`. The Python `User` and the Python
`UserInfo` (in `src/core/auth/types.py`) are layered: ``User`` is the
full storage row (including ``password_hash``) used inside the
repository and service; ``UserInfo`` is the wire-side projection (no
``password_hash``, no ``deleted_at``) returned to the web layer and
external clients. The boundary translation lives in
``UserRepository._to_info``.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from src.common.table_model import TableModel


class User(TableModel):
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


__all__ = ["User"]
