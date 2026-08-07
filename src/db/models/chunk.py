"""Storage row for the `chunks` table.

Mirrors the upstream chunk contract: a chunk is the basic retrieval unit
split out of a document, carrying its positional relationship with the
source text (``start_at`` / ``end_at`` / ``chunk_index``) plus the
indexing bookkeeping (``index_status``, ``content_revision``,
``last_editor_id``).

Column notes
------------

- ``seq_id`` is a DB-assigned auto-increment value (used for FAQ-style
  external references) and is excluded from INSERT; every other column is
  caller-supplied (the application assigns the UUID ``id`` before insert).
- ``relation_chunks`` / ``indirect_relation_chunks`` / ``metadata`` are
  JSONB. The relation lists are raw JSON in storage and are narrowed to
  ``list[str]`` on the wire projection.
- ``image_info`` stays a plain ``text`` column; its JSON payload is parsed
  at the service layer on read.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from src.common.json import JsonObject, JsonValue
from src.common.table_model import TableModel


class Chunk(TableModel):
    """One row of the ``chunks`` table."""

    table: ClassVar[str] = "chunks"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = (
        "relation_chunks",
        "indirect_relation_chunks",
        "metadata",
    )
    # ``id`` is a caller-assigned UUID; only ``seq_id`` is DB-generated
    # (an auto-increment sequence) so it stays out of the INSERT column list.
    db_generated_columns: ClassVar[tuple[str, ...]] = ("seq_id",)

    id: str
    tenant_id: int
    knowledge_base_id: str
    knowledge_id: str
    content: str
    chunk_index: int
    is_enabled: bool = True
    start_at: int
    end_at: int
    pre_chunk_id: str | None = None
    next_chunk_id: str | None = None
    chunk_type: str = "text"
    parent_chunk_id: str | None = None
    image_info: str | None = None
    relation_chunks: JsonValue | None = None
    indirect_relation_chunks: JsonValue | None = None
    metadata: JsonObject | None = None
    tag_id: str | None = None
    status: int = 0
    content_hash: str | None = None
    flags: int = 1
    seq_id: int = 0
    source_content: str = ""
    content_revision: int = 0
    index_status: str = "ready"
    last_editor_id: str = ""
    context_header: str = ""
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


__all__ = ["Chunk"]
