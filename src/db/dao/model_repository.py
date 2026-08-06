"""Model persistence — raw SQL only, no ORM.

Same method set: ``Create``, ``GetByID(tenantID, id)``,
``List(tenantID, type, source)``, ``Update``, ``Delete``,
``ClearDefaultByType``.

The repository only knows about the ``models`` table; the soft-delete
filter lives in :meth:`GenericRepository._soft_delete_where_fragment`
and is applied automatically by every read on this table.
"""

from __future__ import annotations

from typing import cast

from sqlalchemy import JSON, bindparam, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import CursorResult, RowMapping

from src.common.exception import NotFoundError, ValidationError
from src.common.json import BindParams
from src.db.dao.generic_repository import GenericRepository
from src.db.models.infra.model import Model

# JSONB on Postgres, JSON on other dialects (e.g. SQLite in tests).
_JSON = JSON().with_variant(JSONB(), "postgresql")

# Module-level alias for the table name — used in every ``text(f"...")``
# in this file; user input is bound via ``:tenant_id`` / ``:id`` / etc.
_TABLE_NAME = "models"


class ModelRepository(GenericRepository[Model]):
    """`models`-table SQL — domain wrappers on the generic CRUD base."""

    model_class = Model

    # ── Writes ──────────────────────────────────────────────────────

    async def insert(self, row: Model) -> Model:
        """Insert a row; the application supplies the id (UUID)."""
        return await super().insert(row)

    # ── Reads ───────────────────────────────────────────────────────

    async def find_by_tenant_and_id(
        self,
        *,
        tenant_id: int,
        id: str,
        include_builtin: bool = True,
    ) -> Model | None:
        """Return one row by id, scoped to a tenant.

        ``include_builtin=True`` lets callers resolve ``is_builtin=true``
        rows from the system tenant (the Go equivalent is the
        ``(tenant_id = ? OR is_builtin = true)`` predicate in
        ``GetByID``). Tenant-scoped endpoints leave it on; the
        cross-tenant lookup helper sets it explicitly.
        """
        if include_builtin:
            where = "(tenant_id = :tenant_id OR is_builtin = true) and id = :id"
        else:
            where = "tenant_id = :tenant_id and id = :id"
        stmt = text(f"select * from {_TABLE_NAME} where {where} and deleted_at is null").bindparams(
            tenant_id=tenant_id, id=id
        )
        result = await self._session.execute(stmt)
        return self._hydrate_opt(result.mappings().first())

    async def find_by_tenant_and_id_or_fail(
        self,
        *,
        tenant_id: int,
        id: str,
        include_builtin: bool = True,
    ) -> Model:
        """Same as :meth:`find_by_tenant_and_id` but raise on absence."""
        row = await self.find_by_tenant_and_id(
            tenant_id=tenant_id,
            id=id,
            include_builtin=include_builtin,
        )
        if row is None:
            raise NotFoundError(
                code="model.not_found",
                message=f"Model {id} not found",
            )
        return row

    async def list_by_tenant(
        self,
        *,
        tenant_id: int,
        model_type: str | None = None,
        source: str | None = None,
        include_builtin: bool = True,
    ) -> list[Model]:
        """List every row visible to ``tenant_id``.

        Mirrors ``ModelRepository.List`` on the Go side. When
        ``include_builtin`` is True (the default) the ``is_builtin``
        rows from the system tenant are returned alongside the
        tenant-owned rows; when False the result is tenant-scoped
        only.
        """
        params: BindParams = {"tenant_id": tenant_id}
        where_parts: list[str] = []
        if include_builtin:
            where_parts.append("(tenant_id = :tenant_id OR is_builtin = true)")
        else:
            where_parts.append("tenant_id = :tenant_id")
        if model_type:
            where_parts.append("type = :model_type")
            params["model_type"] = model_type
        if source:
            where_parts.append("source = :source")
            params["source"] = source
        where_parts.append("deleted_at is null")
        where = " and ".join(where_parts)
        stmt = text(f"select * from {_TABLE_NAME} where {where}").bindparams(**params)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    # ── Updates / deletes ───────────────────────────────────────────

    async def update_row(self, row: Model) -> Model | None:
        """Update an existing row, returning the refreshed record.

        Mirrors ``ModelRepository.Update``: WHERE is scoped to
        ``(id, tenant_id)`` so a caller cannot stomp another tenant's
        row. ``RETURNING *`` makes the call atomic and hands back the
        persisted state.
        """
        if row.tenant_id == 0:
            raise ValidationError(
                code="model.tenant_id_required",
                message="Model.tenant_id must be set for update",
            )
        columns = self.model_class.insert_sql_column_list()
        update_cols = tuple(c for c in columns if c not in self._pk_columns)
        set_clause = ", ".join(f'"{c}" = :u_{c}' for c in update_cols)
        params: BindParams = {
            **{f"u_{c}": getattr(row, c) for c in update_cols},
            "id": row.id,
            "tenant_id": row.tenant_id,
        }
        bps = [bindparam(f"u_{c}", type_=_JSON) for c in update_cols if c in self._json_columns]
        stmt_text = (
            f"update {_TABLE_NAME} set {set_clause} "
            "where id = :id and tenant_id = :tenant_id returning *"
        )
        stmt = text(stmt_text).bindparams(*bps, **params)
        result = await self._session.execute(stmt)
        mapping = result.mappings().first()
        return self._hydrate_opt(mapping)

    async def delete_by_tenant_and_id(
        self,
        *,
        tenant_id: int,
        id: str,
    ) -> int:
        """Delete a row scoped to ``(tenant_id, id)``; return rows affected.

        The Go guard that refuses to delete ``is_builtin = true`` rows
        lives on the service (``ModelService.delete_model``); this
        repository does not enforce it so the same method can clean
        up rows from any tenant scope.
        """
        stmt_text = f"delete from {_TABLE_NAME} where id = :id and tenant_id = :tenant_id"
        stmt = text(stmt_text).bindparams(id=id, tenant_id=tenant_id)
        result = cast(
            "CursorResult[RowMapping]",
            await self._session.execute(stmt),
        )
        return result.rowcount or 0

    async def clear_default_by_type(
        self,
        *,
        tenant_id: int,
        model_type: str,
        exclude_id: str | None = None,
    ) -> int:
        """Flip ``is_default`` off for every row of one type.

        Mirrors ``ModelRepository.ClearDefaultByType``. Optional
        ``exclude_id`` keeps the freshly-set default from being
        cleared in the same transaction.
        """
        where_parts = [
            "tenant_id = :tenant_id",
            "type = :model_type",
            "is_default = true",
            "deleted_at is null",
        ]
        params: BindParams = {"tenant_id": tenant_id, "model_type": model_type}
        if exclude_id:
            where_parts.append("id != :exclude_id")
            params["exclude_id"] = exclude_id
        where = " and ".join(where_parts)
        stmt = text(f"update {_TABLE_NAME} set is_default = false where {where}").bindparams(
            **params
        )
        result = cast(
            "CursorResult[RowMapping]",
            await self._session.execute(stmt),
        )
        return result.rowcount or 0


__all__ = ["ModelRepository"]
