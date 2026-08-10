"""Storage row for the ``user_kb_pins`` table.

Per-``(user, tenant, kb)`` pin state for the knowledge-base list. The
previous tenant-wide pin model (``knowledge_bases.is_pinned`` /
``pinned_at``) hid the affordance from non-admin members; this table
restores per-user pinning and is keyed by the full triple.

Column notes
------------

- The composite primary key is the upsert target; conflict resolution
  is ``ON CONFLICT DO NOTHING`` so the backfill from the legacy
  tenant-wide pins does not duplicate rows.
- The secondary ``(tenant_id, user_id, pinned_at DESC)`` index backs
  the "list my pinned KBs newest first" listing without an in-memory
  sort step.
- ``pinned_at`` is caller-supplied (the application stamps it on the
  upsert path); no DB default is needed.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from src.common.table_model import TableModel


class UserKBPin(TableModel):
    """One row of the ``user_kb_pins`` table."""

    table: ClassVar[str] = "user_kb_pins"
    primary_keys: ClassVar[tuple[str, ...]] = (
        "tenant_id",
        "user_id",
        "kb_id",
    )
    json_columns: ClassVar[tuple[str, ...]] = ()
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    tenant_id: int
    user_id: str
    kb_id: str
    pinned_at: datetime


__all__ = ["UserKBPin"]