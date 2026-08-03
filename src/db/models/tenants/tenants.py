"""Storage row for the `tenants` table.

The wire-side projection (`TenantInfo` in `src/core/tenants/types.py`)
strips the secret-bearing columns; the boundary translation lives on
that DTO.

Column notes
------------

- `id` is DB-assigned, so it is excluded from INSERT and read back via
  `RETURNING *`.
- Every config column is JSONB; `json_columns` binds them with the JSONB
  bind type.
- `retriever_engines` accepts both the object form `{"engines": [...]}`
  and the legacy bare array `[...]`.
- `api_principal_config` carries an HMAC secret and must never reach
  the wire layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import Field

from src.common.table_model import TableModel

# Default storage quota: 10 GiB, in bytes.
DEFAULT_STORAGE_QUOTA_BYTES: int = 10737418240

# Columns the database assigns itself; excluded from INSERT.
_DB_GENERATED_COLUMNS: frozenset[str] = frozenset({"id"})


class Tenant(TableModel):
    """One row of the `tenants` table."""

    table: ClassVar[str] = "tenants"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = (
        "retriever_engines",
        "agent_config",
        "context_config",
        "conversation_config",
        "web_search_config",
        "parser_engine_config",
        "storage_engine_config",
        "credentials",
        "chat_history_config",
        "retrieval_config",
        "api_principal_config",
    )

    id: int = 0
    name: str
    description: str | None = None
    retriever_engines: dict[str, object] | list[dict[str, object]] = Field(default_factory=dict)
    status: str = "active"
    business: str = ""
    storage_quota: int = DEFAULT_STORAGE_QUOTA_BYTES
    storage_used: int = 0
    agent_config: dict[str, object] | None = None
    context_config: dict[str, object] | None = None
    conversation_config: dict[str, object] | None = None
    web_search_config: dict[str, object] | None = None
    parser_engine_config: dict[str, object] | None = None
    storage_engine_config: dict[str, object] | None = None
    default_storage_backend_id: str | None = None
    credentials: dict[str, object] | None = None
    chat_history_config: dict[str, object] | None = None
    retrieval_config: dict[str, object] | None = None
    api_principal_config: dict[str, object] | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None

    @classmethod
    def insert_sql_column_list(cls) -> tuple[str, ...]:
        """Every column except the DB-generated `id`."""
        return tuple(c for c in cls.column_fields() if c not in _DB_GENERATED_COLUMNS)


__all__ = ["DEFAULT_STORAGE_QUOTA_BYTES", "Tenant"]
