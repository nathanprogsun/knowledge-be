"""Tenant key-value service — generic JSON config per (tenant, key).

Reads and writes an arbitrary JSON value bound to a workspace and key.
The repository owns the per-request session; this service is stateless.
"""

from __future__ import annotations

from src.common.json import JsonValue
from src.db.dao.tenant_kv_repository import TenantKVRepository


class TenantKVService:
    """Stateless tenant KV service."""

    def __init__(self, *, kv_repo: TenantKVRepository) -> None:
        self._kv_repo = kv_repo

    async def get(self, *, tenant_id: int, key: str) -> JsonValue | None:
        """Return the stored value for the pair, or ``None`` when absent."""
        row = await self._kv_repo.find_value(tenant_id=tenant_id, key=key)
        if row is None:
            return None
        return row.value

    async def set(self, *, tenant_id: int, key: str, value: JsonValue) -> JsonValue:
        """Upsert the value and return it (mirrors the stored JSON)."""
        row = await self._kv_repo.upsert(tenant_id=tenant_id, key=key, value=value)
        return row.value

    async def delete(self, *, tenant_id: int, key: str) -> bool:
        """Soft-delete the live row. Returns whether one existed."""
        return await self._kv_repo.delete(tenant_id=tenant_id, key=key)


__all__ = ["TenantKVService"]
