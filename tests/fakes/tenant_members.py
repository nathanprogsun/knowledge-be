"""Shared in-memory fake for the tenant membership repository.

Soft-deleted rows stay in the store but are filtered from every read,
so re-adding a removed member behaves like the partial unique index.
"""

from __future__ import annotations

from datetime import datetime

from src.db.models.tenants.tenant_members import TenantMember


class FakeTenantMemberRepository:
    """In-memory replacement for `TenantMemberRepository`."""

    def __init__(self) -> None:
        self.rows: dict[int, TenantMember] = {}
        self._next_id = 1
        # Emails/usernames keyed by user id, standing in for the join
        # the real search performs against `users`.
        self.user_search_index: dict[str, str] = {}

    # ── Writes ──────────────────────────────────────────────────────

    async def insert_live_or_none(self, row: TenantMember) -> TenantMember | None:
        if await self.find_membership(user_id=row.user_id, tenant_id=row.tenant_id):
            return None
        stored = row.model_copy(update={"id": self._next_id})
        self.rows[stored.id] = stored
        self._next_id += 1
        return stored

    async def update_role(
        self,
        *,
        user_id: str,
        tenant_id: int,
        role: str,
        updated_at: datetime,
    ) -> int:
        row = await self.find_membership(user_id=user_id, tenant_id=tenant_id)
        if row is None:
            return 0
        self.rows[row.id] = row.model_copy(update={"role": role, "updated_at": updated_at})
        return 1

    async def soft_delete(
        self,
        *,
        user_id: str,
        tenant_id: int,
        deleted_at: datetime,
    ) -> int:
        row = await self.find_membership(user_id=user_id, tenant_id=tenant_id)
        if row is None:
            return 0
        self.rows[row.id] = row.model_copy(
            update={"deleted_at": deleted_at, "updated_at": deleted_at}
        )
        return 1

    # ── Reads ───────────────────────────────────────────────────────

    async def find_membership(self, *, user_id: str, tenant_id: int) -> TenantMember | None:
        for row in self._live():
            if row.user_id == user_id and row.tenant_id == tenant_id:
                return row
        return None

    async def list_by_user(self, user_id: str) -> list[TenantMember]:
        return self._sorted([r for r in self._live() if r.user_id == user_id])

    async def list_by_tenant(self, tenant_id: int) -> list[TenantMember]:
        return self._sorted([r for r in self._live() if r.tenant_id == tenant_id])

    async def has_any_members(self, tenant_id: int) -> bool:
        return any(r.tenant_id == tenant_id and r.status == "active" for r in self._live())

    async def count_by_tenant(self, tenant_id: int, *, search: str | None = None) -> int:
        return len(self._matching(tenant_id, search))

    async def list_page_by_tenant(
        self,
        tenant_id: int,
        *,
        search: str | None = None,
        limit: int,
        offset: int,
    ) -> list[TenantMember]:
        return self._matching(tenant_id, search)[offset : offset + limit]

    async def count_active_owners(self, tenant_id: int) -> int:
        return len(
            [
                r
                for r in self._live()
                if r.tenant_id == tenant_id and r.role == "owner" and r.status == "active"
            ]
        )

    async def count_other_active_owners_for_update(
        self,
        *,
        tenant_id: int,
        exclude_user_id: str,
    ) -> int:
        return len(
            [
                r
                for r in self._live()
                if r.tenant_id == tenant_id
                and r.user_id != exclude_user_id
                and r.role == "owner"
                and r.status == "active"
            ]
        )

    # ── Helpers ─────────────────────────────────────────────────────

    def _live(self) -> list[TenantMember]:
        return [r for r in self.rows.values() if r.deleted_at is None]

    def _matching(self, tenant_id: int, search: str | None) -> list[TenantMember]:
        rows = [r for r in self._live() if r.tenant_id == tenant_id]
        term = (search or "").strip().lower()
        if term:
            rows = [r for r in rows if term in self.user_search_index.get(r.user_id, "").lower()]
        return self._sorted(rows)

    @staticmethod
    def _sorted(rows: list[TenantMember]) -> list[TenantMember]:
        return sorted(rows, key=lambda r: (r.joined_at, r.id))


__all__ = ["FakeTenantMemberRepository"]
