"""Storage row for the ``wiki_page_issues`` table.

Issue reports against generated wiki pages (LLM-flagged or user-flagged).
The status lifecycle is ``pending -> resolved / dismissed``; the lifecycle
columns support the curator dashboard and the page-level "issues" badge.

Column notes
------------

- ``id`` is a caller-assigned UUID; the database never assigns columns.
- ``suspected_knowledge_ids`` is a JSONB array of knowledge ids;
  ``json_columns`` binds it with the JSONB bind type.
- The DB-assigned id is excluded from INSERT via the
  ``db_generated_columns`` override being empty; the application always
  supplies the UUID at insert time.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from src.common.table_model import TableModel


class WikiPageIssue(TableModel):
    """One row of the ``wiki_page_issues`` table."""

    table: ClassVar[str] = "wiki_page_issues"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ("suspected_knowledge_ids",)
    # ``id`` is a caller-assigned UUID; the database never assigns columns.
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    id: str
    tenant_id: int
    knowledge_base_id: str
    slug: str
    issue_type: str
    description: str
    suspected_knowledge_ids: list[str] | None = None
    status: str = "pending"
    reported_by: str
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


__all__ = ["WikiPageIssue"]