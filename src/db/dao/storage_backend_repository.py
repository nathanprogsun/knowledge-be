"""Storage-backend persistence — raw SQL only, no ORM.

Mirrors ``internal/types/interfaces/storagebackend.go::StorageBackendRepository``:
``Create`` / ``GetByID`` / ``List`` / ``Update`` / ``Delete`` /
``FindLegacyAlias``. Every read is tenant-scoped and filters soft-deleted
rows; the workspace-default pointer lives on the ``tenants`` row
(``default_storage_backend_id``) and is read/written here so the service
never reaches across into the tenants repository.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

from sqlalchemy import JSON, CursorResult, bindparam, text
from sqlalchemy.dialects.postgresql import JSONB

from src.common.json import BindParams, SqlValue
from src.db.dao.generic_repository import GenericRepository
from src.db.models.storage_backend import StorageBackend

_JSON_BIND_TYPE = JSON().with_variant(JSONB(), "postgresql")

_LIVE = "deleted_at is null"

# Module-level alias for the table name — used in every ``text(f"...")``
# in this file; user input is bound via ``:tenant_id`` / ``:id``.
_TABLE_NAME = "storage_backends"


class StorageBackendRepository(GenericRepository[StorageBackend]):
    """`storage_backends`-table SQL — tenant-scoped CRUD + reference counts."""

    model_class = StorageBackend

    # ── Reads ───────────────────────────────────────────────────────

    async def get_by_id(self, *, tenant_id: int, id: str) -> StorageBackend | None:
        """Return the live row for ``(tenant_id, id)``, or ``None``."""
        return await self.find_unique_by_column_values(
            {"tenant_id": tenant_id, "id": id},
        )

    async def list_for_tenant(self, tenant_id: int) -> list[StorageBackend]:
        """Return every live backend of the workspace, newest first.

        Go's ``List`` orders by ``created_at DESC`` so the UI keeps a
        stable ordering across reloads.
        """
        stmt = text(
            f"select * from {self._table} where tenant_id = :tenant_id and {_LIVE} "
            "order by created_at desc, id desc"
        ).bindparams(tenant_id=tenant_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def find_legacy_alias(self, *, tenant_id: int, provider: str) -> StorageBackend | None:
        """Return the legacy-alias row for ``provider``, or ``None``.

        The legacy alias is the row projected from the pre-instance
        workspace singleton config; resolution falls back to it when a
        caller passes a bare provider name and no backend id.
        """
        stmt = text(
            f"select * from {self._table} where tenant_id = :tenant_id "
            f"and provider = :provider and legacy_alias = true and {_LIVE} "
            "order by created_at asc limit 1"
        ).bindparams(tenant_id=tenant_id, provider=provider)
        result = await self._session.execute(stmt)
        return self._hydrate_opt(result.mappings().first())

    async def find_by_name(self, *, tenant_id: int, name: str) -> StorageBackend | None:
        """Return the live row with this name in the workspace, or ``None``.

        Backs the pre-insert uniqueness probe: Go relies on the unique
        index and maps the driver error to a conflict, which we cannot
        do portably across dialects.
        """
        return await self.find_unique_by_column_values(
            {"tenant_id": tenant_id, "name": name},
        )

    # ── Writes ──────────────────────────────────────────────────────

    async def create(self, row: StorageBackend) -> StorageBackend:
        """Insert ``row`` and return the persisted row."""
        return await self.insert(row)

    async def update_columns(
        self,
        *,
        tenant_id: int,
        id: str,
        columns: BindParams,
    ) -> StorageBackend | None:
        """Update ``columns`` of one tenant-scoped row, returning it.

        Tenant scoping is part of the WHERE clause (not just the pk) so a
        crafted id from another workspace cannot be written.
        """
        self.model_class.validate_in_columns(columns)
        set_clause = ", ".join(f'"{k}" = :u_{k}' for k in columns)
        params: BindParams = {f"u_{k}": v for k, v in columns.items()}
        json_columns = self.model_class.get_json_columns()
        json_bps = [
            bindparam(f"u_{col}", type_=_JSON_BIND_TYPE) for col in columns if col in json_columns
        ]
        stmt = text(
            f"update {self._table} set {set_clause} "
            f"where tenant_id = :tenant_id and id = :id and {_LIVE} returning *"
        ).bindparams(*json_bps, tenant_id=tenant_id, id=id, **params)
        result = await self._session.execute(stmt)
        return self._hydrate_opt(result.mappings().first())

    async def soft_delete(self, *, tenant_id: int, id: str) -> bool:
        """Soft-delete one tenant-scoped row. Returns whether one existed."""
        now = datetime.now(UTC)
        stmt = text(
            f"update {self._table} set deleted_at = :now, updated_at = :now "
            f"where tenant_id = :tenant_id and id = :id and {_LIVE}"
        ).bindparams(tenant_id=tenant_id, id=id, now=now)
        result = await self._session.execute(stmt)
        return (cast("CursorResult[SqlValue]", result).rowcount or 0) > 0

    # ── Workspace default pointer (tenants.default_storage_backend_id) ──

    async def get_default_backend_id(self, tenant_id: int) -> str | None:
        """Return the workspace's ``default_storage_backend_id``, or ``None``."""
        stmt = text(
            "select default_storage_backend_id from tenants "
            "where id = :tenant_id and deleted_at is null"
        ).bindparams(tenant_id=tenant_id)
        result = await self._session.execute(stmt)
        row = result.mappings().first()
        if row is None:
            return None
        raw = row["default_storage_backend_id"]
        return str(raw) if raw is not None else None

    async def set_default_backend_id(self, *, tenant_id: int, id: str) -> bool:
        """Point the workspace default at ``id``. Returns whether a row matched."""
        stmt = text(
            "update tenants set default_storage_backend_id = :id, updated_at = :now "
            "where id = :tenant_id and deleted_at is null"
        ).bindparams(tenant_id=tenant_id, id=id, now=datetime.now(UTC))
        result = await self._session.execute(stmt)
        return (cast("CursorResult[SqlValue]", result).rowcount or 0) > 0

    # ── Reference counts (delete / disable guards) ──────────────────

    async def count_knowledge_base_references(self, *, tenant_id: int, id: str) -> int:
        """Count knowledge bases bound to this backend.

        The ``knowledge_bases`` table is not created yet; until then a
        missing relation means zero bindings rather than a hard failure,
        so the guard degrades safely instead of blocking every delete.
        """
        return await self._count_optional_relation(
            "select count(*) as n from knowledge_bases "
            "where tenant_id = :tenant_id and storage_backend_id = :id",
            {"tenant_id": tenant_id, "id": id},
        )

    async def count_active_resource_references(self, *, tenant_id: int, id: str) -> int:
        """Count active stored resources held by this backend."""
        return await self._count_optional_relation(
            "select count(*) as n from stored_resources "
            "where tenant_id = :tenant_id and storage_backend_id = :id and state = 'active'",
            {"tenant_id": tenant_id, "id": id},
        )

    async def _count_optional_relation(self, sql: str, params: BindParams) -> int:
        """Run a ``count(*)`` whose table may not exist yet.

        Returns 0 when the relation is absent. The savepoint keeps the
        surrounding transaction usable after Postgres aborts it on the
        undefined-table error — the exception must be caught OUTSIDE the
        ``begin_nested`` block so ``ROLLBACK TO SAVEPOINT`` runs first;
        catching inside leaves the savepoint aborted and the release
        raises ``InFailedSQLTransactionError``.
        """
        try:
            async with self._session.begin_nested():
                result = await self._session.execute(text(sql).bindparams(**params))
        except Exception:  # relation absent until the table exists
            return 0
        row = result.mappings().first()
        return int(row["n"]) if row is not None else 0


__all__ = ["StorageBackendRepository"]
