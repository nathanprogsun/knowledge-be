"""Storage row for the `tenant_members` table.

One row per (user, workspace) pair carrying that user's role inside the
workspace. The pair is unique among live rows — the partial unique
index on `(user_id, tenant_id) WHERE deleted_at IS NULL` makes a
removed member re-addable.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from src.common.table_model import TableModel


class TenantMember(TableModel):
    """One row of the `tenant_members` table."""

    table: ClassVar[str] = "tenant_members"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ()

    id: int = 0
    user_id: str
    tenant_id: int
    role: str = "contributor"
    status: str = "active"
    invited_by: str | None = None
    joined_at: datetime
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


__all__ = ["TenantMember"]
