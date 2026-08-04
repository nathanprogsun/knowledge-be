"""Storage row for the `tenant_kv` table.

One row binds a JSON value to a (tenant, key) pair. The value is JSONB and
schema-less; the wire contract types (`TenantKV*Config` in
`src/core/contracts/tenants.py`) shape the decoded value at the boundary.

Soft-deleted rows are excluded on every read; the partial unique index
makes a removed key re-addable.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from src.common.json import JsonValue
from src.common.table_model import TableModel


class TenantKV(TableModel):
    """One row of the `tenant_kv` table."""

    table: ClassVar[str] = "tenant_kv"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ("value",)

    id: int = 0
    tenant_id: int
    key: str
    value: JsonValue
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


__all__ = ["TenantKV"]
