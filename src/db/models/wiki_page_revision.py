"""Storage row for the ``wiki_page_revisions`` table.

Immutable per-version snapshot of a wiki page. The CURRENT version of a
page lives only in ``wiki_pages``; when an edit replaces version ``V``,
the pre-edit state is inserted here as ``(page_id, V)`` before the row
is rewritten, so every historical version stays diffable and revertable.

Column notes
------------

- ``id`` is a caller-assigned UUID; the database never assigns columns.
- ``aliases`` is a JSONB array of slug aliases; ``json_columns`` binds
  it with the JSONB bind type.
- ``edit_source`` / ``editor_id`` carry the same provenance semantics as
  ``wiki_pages.last_edit_source`` / ``wiki_pages.last_editor_id``.
- The unique index ``(page_id, version)`` keeps the snapshot-then-update
  write path idempotent under retries.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import Field

from src.common.table_model import TableModel


class WikiPageRevision(TableModel):
    """One row of the ``wiki_page_revisions`` table."""

    table: ClassVar[str] = "wiki_page_revisions"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ("aliases",)
    # ``id`` is a caller-assigned UUID; the database never assigns columns.
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    id: str
    tenant_id: int
    knowledge_base_id: str
    page_id: str
    slug: str
    version: int
    title: str = ""
    page_type: str = "summary"
    status: str = "published"
    content: str = ""
    summary: str = ""
    aliases: list[str] = Field(default_factory=list)
    edit_source: str = ""
    editor_id: str = ""
    edited_at: datetime
    created_at: datetime


__all__ = ["WikiPageRevision"]