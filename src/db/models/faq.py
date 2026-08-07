"""Storage row for the `faq` table.

One row is one FAQ entry of a knowledge base: the standard question,
the similar / negative question aliases, the answers, and the
entry-level flags and scope columns. The column set mirrors the FAQ
entry shape of the upstream contract; the search-only result fields
(score, match type, matched question) are computed at query time and
are not columns.

``id`` is a database-assigned identity (the entry sequence id), so it
is excluded from INSERT and read back via ``RETURNING *``. Rows are
tenant-scoped and belong to exactly one knowledge base. ``chunk_id``
carries the backing chunk reference and is unique per entry.

The question alias lists and answers are JSONB; ``json_columns`` binds
them with the JSONB bind type. There is no ``deleted_at``: entries are
hard-deleted, matching the delete semantics of the FAQ service.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import Field

from src.common.table_model import TableModel


class Faq(TableModel):
    """One row of the `faq` table."""

    table: ClassVar[str] = "faq"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = (
        "similar_questions",
        "negative_questions",
        "answers",
    )
    # ``id`` is DB-assigned (identity), excluded from INSERT.
    db_generated_columns: ClassVar[tuple[str, ...]] = ("id",)

    id: int = 0
    tenant_id: int
    chunk_id: str
    knowledge_id: str
    knowledge_base_id: str
    tag_id: int | None = None
    tag_name: str | None = None
    is_enabled: bool = True
    is_recommended: bool = False
    standard_question: str
    similar_questions: list[str] = Field(default_factory=list)
    negative_questions: list[str] = Field(default_factory=list)
    answers: list[str] = Field(default_factory=list)
    answer_strategy: str = "all"
    index_mode: str | None = None
    chunk_type: str = "faq"
    created_at: datetime
    updated_at: datetime


__all__ = ["Faq"]
