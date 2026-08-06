"""Shared in-memory fakes for the tenants domain.

Extracted here once the second consumer appeared (the service unit
tests and the web view tests both need a repository stand-in). Method
signatures mirror the real ``TenantRepository`` so a drift between the
two surfaces shows up as a type error rather than a passing test.
"""

from __future__ import annotations

from datetime import datetime

from src.common.exception import NotFoundError
from src.db.models.tenants.tenants import Tenant


class FakeTenantRepository:
    """In-memory replacement for `TenantRepository`.

    Stores rows in a dict keyed by id and assigns ids from a counter,
    standing in for the Postgres sequence. Soft-deleted rows stay in the
    dict but are filtered from every read, mirroring the real repo's
    ``deleted_at IS NULL`` predicate.
    """

    def __init__(self) -> None:
        self.rows: dict[int, Tenant] = {}
        self._next_id = 1

    # ── Writes ──────────────────────────────────────────────────────

    async def insert(self, row: Tenant) -> Tenant:
        stored = row.model_copy(update={"id": self._next_id})
        self.rows[stored.id] = stored
        self._next_id += 1
        return stored

    async def update_by_primary_key(
        self,
        primary_key_to_value: dict[str, object],
        column_to_update: dict[str, object],
    ) -> Tenant | None:
        tenant_id = primary_key_to_value["id"]
        row = self.rows.get(int(str(tenant_id)))
        if row is None or row.deleted_at is not None:
            return None
        updated = row.model_copy(update=column_to_update)
        self.rows[updated.id] = updated
        return updated

    async def adjust_storage_used(
        self,
        tenant_id: int,
        *,
        delta: int,
        updated_at: datetime,
    ) -> int:
        row = self._live().get(tenant_id)
        if row is None:
            raise NotFoundError(code="tenant.not_found", message="Tenant not found")
        used = max(row.storage_used + delta, 0)
        self.rows[tenant_id] = row.model_copy(
            update={"storage_used": used, "updated_at": updated_at}
        )
        return used

    async def bulk_set_storage_quota(self, *, quota_bytes: int, updated_at: datetime) -> int:
        live = self._live()
        for tenant_id, row in live.items():
            self.rows[tenant_id] = row.model_copy(
                update={"storage_quota": quota_bytes, "updated_at": updated_at}
            )
        return len(live)

    # ── Reads ───────────────────────────────────────────────────────

    async def find_by_id(self, id: str | int) -> Tenant:
        row = self._live().get(int(str(id)))
        if row is None:
            raise NotFoundError(code="tenant.not_found", message=f"Tenant {id} not found")
        return row

    async def find_by_ids(self, ids: list[int]) -> list[Tenant]:
        if not ids:
            return []
        wanted = set(ids)
        return self._sorted([r for r in self._live().values() if r.id in wanted])

    async def list_all(self, *, limit: int | None = None, offset: int = 0) -> list[Tenant]:
        rows = self._sorted(list(self._live().values()))
        return rows[offset : offset + limit] if limit is not None else rows

    async def search(
        self,
        *,
        keyword: str | None = None,
        tenant_id: int | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> tuple[list[Tenant], int]:
        matches = self._sorted(
            [r for r in self._live().values() if self._matches(r, keyword, tenant_id)]
        )
        page = matches[offset : offset + limit] if limit is not None else matches
        return page, len(matches)

    # ── Helpers ─────────────────────────────────────────────────────

    def _live(self) -> dict[int, Tenant]:
        return {i: r for i, r in self.rows.items() if r.deleted_at is None}

    @staticmethod
    def _sorted(rows: list[Tenant]) -> list[Tenant]:
        return sorted(rows, key=lambda r: r.created_at, reverse=True)

    @staticmethod
    def _matches(row: Tenant, keyword: str | None, tenant_id: int | None) -> bool:
        if tenant_id is None and not keyword:
            return True
        if tenant_id is not None and tenant_id > 0 and row.id == tenant_id:
            return True
        if not keyword:
            return False
        return keyword in row.name or keyword in (row.description or "")


__all__ = ["FakeTenantRepository"]
