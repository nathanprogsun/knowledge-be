"""Storage row for the `chunk_revisions` table.

An immutable snapshot of a superseded chunk revision: each user edit or
rollback appends one row, while the current content stays on the
`chunks` row. Column names match the upstream migration (and therefore
the wire projections produced by the service layer).

The default values mirror the SQL ``DEFAULT`` clauses (``content`` /
``editor_id`` / ``edit_source`` / ``is_enabled``), so a snapshot can be
built without spelling out every superseded field. ``id`` is a
caller-assigned UUID — it must take part in the INSERT column list, so
``db_generated_columns`` is empty.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from src.common.table_model import TableModel


class ChunkRevision(TableModel):
    """One row of the ``chunk_revisions`` table."""

    table: ClassVar[str] = "chunk_revisions"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ()
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    id: str
    tenant_id: int
    knowledge_base_id: str
    knowledge_id: str
    chunk_id: str
    revision: int
    content: str = ""
    is_enabled: bool = True
    editor_id: str = ""
    edit_source: str = "user"
    edited_at: datetime
    created_at: datetime


__all__ = ["ChunkRevision"]
