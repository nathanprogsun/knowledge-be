"""Storage row for the ``wiki_log_entries`` table.

Append-only event log for wiki ingest / retract / fix operations. The
event log replaces the legacy "one giant TEXT column on ``slug='log'``
wiki_pages row" model that caused ``O(n^2)`` write amplification as KBs
grew; here each event is a single INSERT and reads paginate by
``(kb_id, id DESC)``.

Column notes
------------

- ``id`` is the database-assigned BIGSERIAL and is used as the cursor
  for newest-first pagination (sidesteps duplicate-timestamp ties).
- ``pages_affected`` is a JSONB array of page ids; ``json_columns``
  binds it with the JSONB bind type.
- The DB-assigned id is excluded from INSERT and read back via
  ``RETURNING *``.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import Field

from src.common.table_model import TableModel


class WikiLogEntry(TableModel):
    """One row of the ``wiki_log_entries`` table."""

    table: ClassVar[str] = "wiki_log_entries"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ("pages_affected",)
    # ``id`` is DB-assigned (BIGSERIAL); ``created_at`` carries a DB
    # default and is also stamped by the application on insert.
    db_generated_columns: ClassVar[tuple[str, ...]] = ("id",)

    id: int = 0
    tenant_id: int
    knowledge_base_id: str
    action: str
    knowledge_id: str = ""
    doc_title: str = ""
    summary: str = ""
    pages_affected: list[str] = Field(default_factory=list)
    created_at: datetime


__all__ = ["WikiLogEntry"]