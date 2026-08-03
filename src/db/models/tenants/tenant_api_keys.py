"""Storage row for the `tenant_api_keys` table.

A revocable machine credential. Tenant-scoped and platform keys share
the table — platform keys carry `tenant_id = NULL`, enforced by the
CHECK constraint that ties `tenant_id` to `scope_type`.

Authentication resolves a key by `key_hash` (SHA-256 of the token), so
that column is unique. `api_key` keeps the token for display and the
hash backfill; encryption at rest is gated on a deployment key.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import Field

from src.common.table_model import TableModel

# Columns the database assigns itself; excluded from INSERT.
_DB_GENERATED_COLUMNS: frozenset[str] = frozenset({"id"})


class TenantAPIKey(TableModel):
    """One row of the `tenant_api_keys` table."""

    table: ClassVar[str] = "tenant_api_keys"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ("knowledge_base_ids", "capabilities")

    id: int = 0
    tenant_id: int | None = None
    scope_type: str = "tenant"
    name: str
    key_hash: str
    api_key: str = ""
    full_access: bool = False
    knowledge_base_ids: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    last_used_at: datetime | None = None
    expires_at: datetime | None = None
    revoked_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def insert_sql_column_list(cls) -> tuple[str, ...]:
        """Every column except the DB-generated `id`."""
        return tuple(c for c in cls.column_fields() if c not in _DB_GENERATED_COLUMNS)


__all__ = ["TenantAPIKey"]
