"""Storage row for the `knowledge_processing_spans` table.

One row is one span in a per-(knowledge, attempt) progress tree recorded
by the processing pipeline. The tree mirrors Langfuse's vocabulary:
a single ``root`` span per (knowledge_id, attempt), ``stage`` children
(the five canonical pipeline stages), and free-form ``subspan`` /
``generation`` rows hanging off any parent.

Column notes
------------

- ``id`` is a DB-assigned BIGSERIAL and is excluded from INSERT.
- ``input`` / ``output`` / ``metadata`` are JSONB. ``metadata`` carries
  the optional ``langfuse_trace_id`` used to stitch generations to
  external tracing.
- ``error_code`` / ``error_message`` / ``error_detail`` are nullable but
  the tracker writes ``""`` (empty string) when unset, matching the
  upstream write semantics where the error columns are always updated.
- ``started_at`` / ``finished_at`` / ``duration_ms`` are null until the
  span transitions; ``duration_ms`` is computed from the in-process start
  cache when available and falls back to ``finished_at - started_at``.
- Rows are never soft-deleted; the unique ``(knowledge_id, attempt,
  span_id)`` constraint makes every state transition an upsert.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from src.common.json import JsonObject
from src.common.table_model import TableModel


class KnowledgeProcessingSpan(TableModel):
    """One row of the ``knowledge_processing_spans`` table."""

    table: ClassVar[str] = "knowledge_processing_spans"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = (
        "input",
        "output",
        "metadata",
    )
    # ``id`` is a DB-assigned BIGSERIAL identity; every other column is
    # caller-supplied.
    db_generated_columns: ClassVar[tuple[str, ...]] = ("id",)

    id: int = 0
    knowledge_id: str
    attempt: int = 1
    span_id: str
    parent_span_id: str | None = None
    name: str
    kind: str
    status: str
    input: JsonObject | None = None
    output: JsonObject | None = None
    metadata: JsonObject | None = None
    error_code: str | None = None
    error_message: str | None = None
    error_detail: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: int | None = None
    created_at: datetime
    updated_at: datetime


__all__ = ["KnowledgeProcessingSpan"]
