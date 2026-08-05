"""Storage row for the `vector_stores` table.

Mirrors ``internal/types/vectorstore.go::VectorStore``. Each row is a
tenant-scoped configuration of a vector database (Elasticsearch, Qdrant,
Milvus, Tencent VectorDB, Weaviate, Doris, OpenSearch). Agents reference
vector stores by UUID ``id``.

``connection_config`` and ``index_config`` are JSONB blobs carrying
engine-specific credentials and index settings. The service layer
controls what crosses the wire; the row carries everything that was
persisted.

``source`` is the classifier from the Go contract:
``"user"`` for DB-managed rows, ``"env"`` for virtual entries synthesised
from ``RETRIEVE_DRIVER``. The Python side only persists ``"user"`` rows;
``"env"`` rows are computed at request time by the service from the
environment. ``readonly`` is mirrored for parity with the wire contract.

``deleted_at`` is the soft-delete marker. Mirrors the Go entity's
``gorm.DeletedAt``.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from src.common.json import JsonObject
from src.common.table_model import TableModel


class VectorStore(TableModel):
    """One row of the `vector_stores` table."""

    table: ClassVar[str] = "vector_stores"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ("connection_config", "index_config")
    db_generated_columns: ClassVar[tuple[str, ...]] = ()  # id is caller-assigned (UUID).

    id: str
    tenant_id: int
    name: str
    engine_type: str
    connection_config: JsonObject | None = None
    index_config: JsonObject | None = None
    source: str = "user"
    readonly: bool = False
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


__all__ = ["VectorStore"]
