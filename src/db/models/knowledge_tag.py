"""Storage rows for the `tags` and `document_tags` tables.

A tag belongs to exactly one knowledge base and is scoped by tenant.
``id`` is a caller-assigned UUID string; ``seq_id`` is the
DB-assigned autoincrement integer id exposed to external APIs (FAQ
chunks reference tags by ``seq_id``).

``document_tags`` is the many-to-many association between a document
knowledge entry and a tag. Its composite primary key
``(knowledge_id, tag_id)`` makes re-binding idempotent; ``created_at``
records when the binding was made. The association carries no
``deleted_at``: unbinding is a hard delete of the row.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from src.common.table_model import TableModel


class KnowledgeTag(TableModel):
    """One row of the `tags` table."""

    table: ClassVar[str] = "tags"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ()
    # ``id`` is caller-assigned, so it participates in INSERT; ``seq_id``
    # is minted by the DB sequence and read back via RETURNING.
    db_generated_columns: ClassVar[tuple[str, ...]] = ("seq_id",)

    id: str
    seq_id: int = 0
    tenant_id: int
    knowledge_base_id: str
    name: str
    color: str | None = None
    sort_order: int = 0
    created_at: datetime
    updated_at: datetime


class DocumentTag(TableModel):
    """One row of the `document_tags` association table."""

    table: ClassVar[str] = "document_tags"
    primary_keys: ClassVar[tuple[str, ...]] = ("knowledge_id", "tag_id")
    json_columns: ClassVar[tuple[str, ...]] = ()
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    knowledge_id: str
    tag_id: str
    created_at: datetime


__all__ = ["DocumentTag", "KnowledgeTag"]
