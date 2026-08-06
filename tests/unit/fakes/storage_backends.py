"""In-memory ``StorageBackendRepository`` double.

Mirrors the repository's async surface without SQL so the service tests
exercise the domain rules (validation, immutability, delete guards,
resolution precedence) rather than the DAO. Rows are stored as frozen
``StorageBackend`` models, so a test that mutates the returned row cannot
corrupt the fake's state.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.common.json import BindParams
from src.db.models.storage_backend import StorageBackend


class FakeStorageBackendRepository:
    """Dict-backed stand-in for ``StorageBackendRepository``."""

    def __init__(self) -> None:
        self.rows: dict[str, StorageBackend] = {}
        self.default_backend_id: dict[int, str] = {}
        # Reference counts the delete/disable guards consult. Tests set
        # these directly; the real repository queries other tables.
        self.knowledge_base_references: int = 0
        self.active_resource_references: int = 0

    # ── Reads ───────────────────────────────────────────────────────

    async def get_by_id(self, *, tenant_id: int, id: str) -> StorageBackend | None:
        row = self.rows.get(id)
        if row is None or row.tenant_id != tenant_id or row.deleted_at is not None:
            return None
        return row

    async def list_for_tenant(self, tenant_id: int) -> list[StorageBackend]:
        live = [
            row
            for row in self.rows.values()
            if row.tenant_id == tenant_id and row.deleted_at is None
        ]
        return sorted(live, key=lambda r: (r.created_at, r.id))

    async def find_legacy_alias(self, *, tenant_id: int, provider: str) -> StorageBackend | None:
        candidates = [
            row
            for row in await self.list_for_tenant(tenant_id)
            if row.provider == provider and row.legacy_alias
        ]
        return candidates[0] if candidates else None

    async def find_by_name(self, *, tenant_id: int, name: str) -> StorageBackend | None:
        for row in await self.list_for_tenant(tenant_id):
            if row.name == name:
                return row
        return None

    # ── Writes ──────────────────────────────────────────────────────

    async def create(self, row: StorageBackend) -> StorageBackend:
        self.rows[row.id] = row
        return row

    async def update_columns(
        self, *, tenant_id: int, id: str, columns: BindParams
    ) -> StorageBackend | None:
        existing = await self.get_by_id(tenant_id=tenant_id, id=id)
        if existing is None:
            return None
        updated = existing.model_copy(update=dict(columns))
        self.rows[id] = updated
        return updated

    async def soft_delete(self, *, tenant_id: int, id: str) -> bool:
        existing = await self.get_by_id(tenant_id=tenant_id, id=id)
        if existing is None:
            return False
        now = datetime.now(UTC)
        self.rows[id] = existing.model_copy(update={"deleted_at": now, "updated_at": now})
        return True

    # ── Workspace default pointer ───────────────────────────────────

    async def get_default_backend_id(self, tenant_id: int) -> str | None:
        return self.default_backend_id.get(tenant_id)

    async def set_default_backend_id(self, *, tenant_id: int, id: str) -> bool:
        self.default_backend_id[tenant_id] = id
        return True

    # ── Reference counts ────────────────────────────────────────────

    async def count_knowledge_base_references(self, *, tenant_id: int, id: str) -> int:
        return self.knowledge_base_references

    async def count_active_resource_references(self, *, tenant_id: int, id: str) -> int:
        return self.active_resource_references


__all__ = ["FakeStorageBackendRepository"]
