"""Tenant key-value persistence — raw SQL only, no ORM.

A generic JSON value bound to a (tenant, key) pair. Soft-deleted rows are
filtered on every read; upsert (``ON CONFLICT (tenant_id, key) DO UPDATE``)
makes a write idempotent and revives a soft-deleted row atomically.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

from sqlalchemy import JSON, CursorResult, bindparam, text
from sqlalchemy.dialects.postgresql import JSONB

from src.common.json import JsonValue
from src.db.dao.generic_repository import GenericRepository
from src.db.models.tenants.tenant_kv import TenantKV

_JSON_BIND_TYPE = JSON().with_variant(JSONB(), "postgresql")

_LIVE = "deleted_at is null"


class TenantKVRepository(GenericRepository[TenantKV]):
    """`tenant_kv`-table SQL — upsert + point read by (tenant, key)."""

    model_class = TenantKV

    async def find_value(self, *, tenant_id: int, key: str) -> TenantKV | None:
        """Return the live row for the pair, or ``None``."""
        return await self.find_unique_by_column_values(
            {"tenant_id": tenant_id, "key": key},
        )

    async def upsert(self, *, tenant_id: int, key: str, value: JsonValue) -> TenantKV:
        """Insert or update the (tenant, key) row, returning the stored row.

        Revives a soft-deleted row by clearing ``deleted_at`` so the
        partial unique index stays satisfiable.
        """
        now = datetime.now(UTC)
        stmt = text(
            "insert into tenant_kv (tenant_id, key, value, created_at, updated_at) "
            "values (:tenant_id, :key, :value, :created_at, :updated_at) "
            "on conflict (tenant_id, key) where deleted_at is null "
            "do update set value = excluded.value, updated_at = excluded.updated_at, "
            "deleted_at = null "
            "returning *"
        ).bindparams(bindparam("value", type_=_JSON_BIND_TYPE))
        result = await self._session.execute(
            stmt,
            {
                "tenant_id": tenant_id,
                "key": key,
                "value": value,
                "created_at": now,
                "updated_at": now,
            },
        )
        mapping = result.mappings().first()
        if mapping is None:
            raise RuntimeError("tenant_kv upsert returned no row")
        return self._hydrate(mapping)

    async def delete(self, *, tenant_id: int, key: str) -> bool:
        """Soft-delete the live (tenant, key) row. Returns whether one existed."""
        stmt = text(
            "update tenant_kv set deleted_at = :now, updated_at = :now "
            f"where tenant_id = :tenant_id and key = :key and {_LIVE}"
        )
        result = await self._session.execute(
            stmt,
            {"tenant_id": tenant_id, "key": key, "now": datetime.now(UTC)},
        )
        return (cast(CursorResult[object], result).rowcount or 0) > 0


__all__ = ["TenantKVRepository"]
