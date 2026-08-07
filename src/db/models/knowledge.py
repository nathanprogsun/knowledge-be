"""Storage row for the `documents` table.

The SQL shape is captured from the upstream persistence contract for
knowledge entries. Column names (and therefore the JSON names produced
by the wire projections) match the domain entity.

Two upstream entity fields are NOT columns and are absent here:

- ``Tags`` (``gorm:"-"``) — stored via the tag-relation table.
- ``KnowledgeBaseName`` (``gorm:"-"``) — joined per query.

``metadata`` / ``custom_metadata`` / ``last_faq_import_result`` are
JSONB. ``custom_metadata`` is user-authored descriptive metadata and is
deliberately kept separate from ``metadata``, which carries internal
ingestion state and IDs.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import Field

from src.common.json import JsonObject
from src.common.table_model import TableModel


class Document(TableModel):
    """One row of the ``documents`` table."""

    table: ClassVar[str] = "documents"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = (
        "metadata",
        "custom_metadata",
        "last_faq_import_result",
    )
    # ``id`` is a caller-assigned UUID (generated at the service edge),
    # not a server default — it must take part in the INSERT column list.
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    id: str
    tenant_id: int
    knowledge_base_id: str
    type: str
    title: str
    description: str | None = None
    source: str
    channel: str = "web"
    parse_status: str = "unprocessed"
    pending_subtasks_count: int = 0
    summary_status: str = "none"
    enable_status: str = "enabled"
    embedding_model_id: str | None = None
    file_name: str | None = None
    file_type: str | None = None
    file_size: int | None = None
    file_hash: str | None = None
    file_path: str | None = None
    storage_size: int = 0
    metadata: JsonObject | None = None
    custom_metadata: JsonObject = Field(default_factory=dict)
    last_faq_import_result: JsonObject | None = None
    created_at: datetime
    updated_at: datetime
    processed_at: datetime | None = None
    error_message: str | None = None
    deleted_at: datetime | None = None


__all__ = ["Document"]
