"""Shared in-memory fake for the tenant API-key repository.

Method signatures mirror ``TenantAPIKeyRepository``; revoked rows stay
in the store but are filtered from every read, matching the real repo's
``revoked_at IS NULL`` predicate.
"""

from __future__ import annotations

from datetime import datetime

from src.common.exception import NotFoundError
from src.db.dao.tenant_api_keys_repository import PLACEHOLDER_KEY_HASH_PREFIX
from src.db.models.tenants.tenant_api_keys import TenantAPIKey

_NOT_FOUND_CODE = "tenant_api_key.not_found"


class FakeTenantAPIKeyRepository:
    """In-memory replacement for `TenantAPIKeyRepository`."""

    def __init__(self) -> None:
        self.rows: dict[int, TenantAPIKey] = {}
        self._next_id = 1

    # ── Writes ──────────────────────────────────────────────────────

    async def insert(self, row: TenantAPIKey) -> TenantAPIKey:
        stored = row.model_copy(update={"id": self._next_id})
        self.rows[stored.id] = stored
        self._next_id += 1
        return stored

    async def revoke(self, key_id: int, *, tenant_id: int, revoked_at: datetime) -> None:
        row = self._live().get(key_id)
        if row is None or row.tenant_id != tenant_id:
            raise NotFoundError(code=_NOT_FOUND_CODE, message="Tenant API key not found")
        self.rows[key_id] = row.model_copy(update={"revoked_at": revoked_at})

    async def revoke_platform(self, key_id: int, *, revoked_at: datetime) -> None:
        row = self._live().get(key_id)
        if row is None or row.scope_type != "platform":
            raise NotFoundError(code=_NOT_FOUND_CODE, message="Tenant API key not found")
        self.rows[key_id] = row.model_copy(update={"revoked_at": revoked_at})

    async def touch_last_used(self, key_id: int, *, used_at: datetime) -> int:
        row = self._live().get(key_id)
        if row is None:
            return 0
        self.rows[key_id] = row.model_copy(update={"last_used_at": used_at})
        return 1

    async def update_hash(self, key_id: int, *, key_hash: str) -> int:
        row = self._live().get(key_id)
        if row is None:
            return 0
        self.rows[key_id] = row.model_copy(update={"key_hash": key_hash})
        return 1

    # ── Reads ───────────────────────────────────────────────────────

    async def find_by_hash(self, key_hash: str) -> TenantAPIKey:
        for row in self._live().values():
            if row.key_hash == key_hash:
                return row
        raise NotFoundError(code=_NOT_FOUND_CODE, message="Tenant API key not found")

    async def list_for_tenant(self, tenant_id: int) -> list[TenantAPIKey]:
        return self._sorted([r for r in self._live().values() if r.tenant_id == tenant_id])

    async def list_platform(self) -> list[TenantAPIKey]:
        return self._sorted([r for r in self._live().values() if r.scope_type == "platform"])

    async def list_with_placeholder_hash(self) -> list[TenantAPIKey]:
        return self._sorted(
            [r for r in self._live().values() if r.key_hash.startswith(PLACEHOLDER_KEY_HASH_PREFIX)]
        )

    async def has_placeholder_hash(self) -> bool:
        return bool(await self.list_with_placeholder_hash())

    # ── Helpers ─────────────────────────────────────────────────────

    def _live(self) -> dict[int, TenantAPIKey]:
        return {i: r for i, r in self.rows.items() if r.revoked_at is None}

    @staticmethod
    def _sorted(rows: list[TenantAPIKey]) -> list[TenantAPIKey]:
        return sorted(rows, key=lambda r: r.created_at, reverse=True)


__all__ = ["FakeTenantAPIKeyRepository"]
