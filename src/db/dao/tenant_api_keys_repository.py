"""Tenant API-key persistence — raw SQL only, no ORM.

Revocation is the soft delete: the table has no ``deleted_at`` and
every read filters ``revoked_at IS NULL`` so a revoked key behaves as
if it no longer exists. The authentication lookup goes through
``find_by_hash``; the plaintext token is never a query predicate.
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from sqlalchemy import CursorResult, text

from src.common.exception import NotFoundError
from src.db.dao.generic_repository import GenericRepository
from src.db.models.tenants.tenant_api_keys import TenantAPIKey

# Placeholder hash carried over from the legacy per-tenant `api_key`
# column; the real SHA-256 is filled in by the backfill.
PLACEHOLDER_KEY_HASH_PREFIX = "migrated-tenant-"

_NOT_FOUND_CODE = "tenant_api_key.not_found"


class TenantAPIKeyRepository(GenericRepository[TenantAPIKey]):
    """`tenant_api_keys`-table SQL."""

    model_class = TenantAPIKey

    # ── Reads ───────────────────────────────────────────────────────

    async def find_by_hash(self, key_hash: str) -> TenantAPIKey:
        """Resolve a key by its SHA-256 hash; revoked keys are not found."""
        row = await self.find_unique_by_column_values({"key_hash": key_hash})
        if row is None or row.revoked_at is not None:
            raise NotFoundError(code=_NOT_FOUND_CODE, message="Tenant API key not found")
        return row

    async def list_for_tenant(self, tenant_id: int) -> list[TenantAPIKey]:
        """Live keys of one workspace, newest first."""
        return await self._select_live(
            "tenant_id = :tenant_id",
            {"tenant_id": tenant_id},
        )

    async def list_platform(self) -> list[TenantAPIKey]:
        """Live platform-scoped keys, newest first."""
        return await self._select_live(
            "scope_type = :scope_type",
            {"scope_type": "platform"},
        )

    async def list_with_placeholder_hash(self) -> list[TenantAPIKey]:
        """Live keys whose ``key_hash`` is still the migration placeholder."""
        return await self._select_live(
            "key_hash like :prefix",
            {"prefix": f"{PLACEHOLDER_KEY_HASH_PREFIX}%"},
        )

    async def has_placeholder_hash(self) -> bool:
        """Whether any live key still carries the placeholder hash."""
        stmt = text(
            f"select 1 from {self._table} "
            "where key_hash like :prefix and revoked_at is null limit 1"
        ).bindparams(prefix=f"{PLACEHOLDER_KEY_HASH_PREFIX}%")
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none() is not None

    # ── Mutations ───────────────────────────────────────────────────

    async def revoke(self, key_id: int, *, tenant_id: int, revoked_at: datetime) -> None:
        """Revoke one workspace-scoped key; raise when nothing matched."""
        await self._revoke_where(
            "id = :id and tenant_id = :tenant_id",
            {"id": key_id, "tenant_id": tenant_id},
            revoked_at,
        )

    async def revoke_platform(self, key_id: int, *, revoked_at: datetime) -> None:
        """Revoke one platform-scoped key; raise when nothing matched."""
        await self._revoke_where(
            "id = :id and scope_type = :scope_type",
            {"id": key_id, "scope_type": "platform"},
            revoked_at,
        )

    async def touch_last_used(self, key_id: int, *, used_at: datetime) -> int:
        """Stamp ``last_used_at`` on a live key; return rows affected."""
        return await self._update_live(
            "last_used_at = :used_at",
            "id = :id",
            {"id": key_id, "used_at": used_at},
        )

    async def update_hash(self, key_id: int, *, key_hash: str) -> int:
        """Replace a live key's hash (backfill); return rows affected."""
        return await self._update_live(
            "key_hash = :key_hash",
            "id = :id",
            {"id": key_id, "key_hash": key_hash},
        )

    # ── Query builders ──────────────────────────────────────────────

    async def _select_live(
        self,
        conditions: str,
        params: dict[str, object],
    ) -> list[TenantAPIKey]:
        stmt = text(
            f"select * from {self._table} "
            f"where {conditions} and revoked_at is null order by created_at desc"
        ).bindparams(**params)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def _update_live(
        self,
        set_clause: str,
        conditions: str,
        params: dict[str, object],
    ) -> int:
        stmt = text(
            f"update {self._table} set {set_clause} where {conditions} and revoked_at is null"
        ).bindparams(**params)
        result = await self._session.execute(stmt)
        return cast("CursorResult[object]", result).rowcount

    async def _revoke_where(
        self,
        conditions: str,
        params: dict[str, object],
        revoked_at: datetime,
    ) -> None:
        affected = await self._update_live(
            "revoked_at = :revoked_at",
            conditions,
            {**params, "revoked_at": revoked_at},
        )
        if affected == 0:
            raise NotFoundError(code=_NOT_FOUND_CODE, message="Tenant API key not found")


__all__ = ["PLACEHOLDER_KEY_HASH_PREFIX", "TenantAPIKeyRepository"]
