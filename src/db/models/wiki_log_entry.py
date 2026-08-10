"""Storage row for the `wiki_log_entries` table.

Append-only event log for wiki operations (ingest / retract / lint
runs). One row per event; reads cursor-paginate by
``(knowledge_base_id, id DESC)`` so the ``id`` BIGSERIAL doubles as
the cursor (monotonic, no timestamp tie-breaker required).

``pages_affected`` carries the slugs of pages the event touched
(JSON array). ``knowledge_id`` and ``doc_title`` are denormalised
proxies of the source document so a log feed can render the entry
without joining ``knowledges``.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import Field

from src.common.table_model import TableModel


class WikiLogEntry(TableModel):
    """One row of the `wiki_log_entries` table."""

    table: ClassVar[str] = "wiki_log_entries"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ("pages_affected",)
    # ``id`` is DB-assigned (BIGSERIAL), so it is excluded from INSERT
    # and read back via RETURNING.
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
