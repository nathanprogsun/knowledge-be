"""Storage row for the `user_kb_pins` table.

Per-(user, tenant, knowledge_base) pin state. Replaces the legacy
tenant-wide ``knowledge_bases.is_pinned`` / ``pinned_at`` columns as
the source of truth for ordering the knowledge-base list per viewer.

The composite primary key ``(tenant_id, user_id, kb_id)`` keeps the
upsert path idempotent (``INSERT ... ON CONFLICT DO NOTHING``) and
makes per-user pin lookups an equality hit on the PK index. The
table carries no ``updated_at`` / ``deleted_at``: pinning is a
single instant in time, and unpinning is a row delete.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from src.common.table_model import TableModel


class UserKBPin(TableModel):
    """One row of the `user_kb_pins` table."""

    table: ClassVar[str] = "user_kb_pins"
    primary_keys: ClassVar[tuple[str, ...]] = (
        "tenant_id",
        "user_id",
        "kb_id",
    )
    json_columns: ClassVar[tuple[str, ...]] = ()
    # All columns are caller-supplied (PK + the timestamp the
    # application stamps before insert).
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    tenant_id: int
    user_id: str
    kb_id: str
    pinned_at: datetime


__all__ = ["UserKBPin"]
