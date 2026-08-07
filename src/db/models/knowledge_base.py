"""Storage row for the `knowledge_bases` table.

The column shape is captured from the upstream contract during the
initial migration; column names mirror the storage layer exactly.
JSON configuration blobs are carried as opaque ``JsonObject`` values
here — typed parsing happens in the domain layer.

``vector_store_id`` is bound once at creation time and never modified
afterwards (enforced by the repository's update path). ``cos_config``
is the legacy COS configuration column; it surfaces on the wire under
its renamed ``storage_config`` field via the domain projection.

Several response-only fields (``is_pinned``, ``pinned_at``,
``knowledge_count``, ``chunk_count``, ``is_processing``,
``processing_count``, ``share_count``) are NOT columns and are absent
here — the service computes them per query.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from src.common.json import JsonObject
from src.common.table_model import TableModel


class KnowledgeBase(TableModel):
    """One row of the `knowledge_bases` table."""

    table: ClassVar[str] = "knowledge_bases"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = (
        "chunking_config",
        "image_processing_config",
        "vlm_config",
        "asr_config",
        "storage_provider_config",
        "cos_config",
        "extract_config",
        "faq_config",
        "question_generation_config",
        "wiki_config",
        "indexing_strategy",
    )
    # ``id`` is a caller-assigned UUID (set by the service before insert),
    # not a server default — it must take part in the INSERT column list.
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    id: str
    name: str
    type: str = "document"
    is_temporary: bool = False
    description: str | None = None
    tenant_id: int
    creator_id: str | None = None
    chunking_config: JsonObject | None = None
    image_processing_config: JsonObject | None = None
    embedding_model_id: str = ""
    summary_model_id: str = ""
    vlm_config: JsonObject | None = None
    asr_config: JsonObject | None = None
    storage_provider_config: JsonObject | None = None
    storage_backend_id: str | None = None
    cos_config: JsonObject | None = None
    vector_store_id: str | None = None
    extract_config: JsonObject | None = None
    faq_config: JsonObject | None = None
    question_generation_config: JsonObject | None = None
    wiki_config: JsonObject | None = None
    indexing_strategy: JsonObject | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


__all__ = ["KnowledgeBase"]
