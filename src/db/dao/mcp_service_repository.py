"""MCP-service persistence — raw SQL only, no ORM.

Mirrors the Go-side ``MCPServiceRepository`` interface. Soft delete is
the only lifecycle: a deleted row's ``id`` is reserved forever, but
every read filters ``deleted_at is null``. Builtin rows are visible to
every tenant — that filter lives in the service layer, not here, so
the repository stays a single-tenant lookup primitive.
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from sqlalchemy import CursorResult, bindparam, text

from src.common.exception import NotFoundError
from src.common.json import BindParams, SqlValue
from src.db.dao.generic_repository import _JSON_BIND_TYPE, GenericRepository
from src.db.models.infra.mcp_services import MCPService

# Code used when ``find_by_id`` cannot find a row. Matches the Go
# side's ``fmt.Errorf("MCP service not found")`` shape, surfaced to the
# web layer as a 404 by the standard exception handler.
_NOT_FOUND_CODE = "mcp_service.not_found"

# Module-level alias for the table name — used in every ``text(f"...")``
# in this file; user input is bound via ``:tenant_id`` / ``:id`` / etc.
_TABLE_NAME = "mcp_services"


class MCPServiceRepository(GenericRepository[MCPService]):
    """``mcp_services``-table SQL."""

    model_class = MCPService

    async def find_for_tenant(
        self,
        tenant_id: int,
        id: str,
    ) -> MCPService | None:
        """Return the live row for (tenant_id, id) or ``None``.

        Named differently from the inherited ``find_by_id`` (which
        looks up by primary key only) because the MCP service lookup
        is tenant-scoped — the bare service id is not unique across
        workspaces.
        """
        return await self.find_unique_by_column_values(
            {"tenant_id": tenant_id, "id": id},
        )

    async def get_by_id(self, tenant_id: int, id: str) -> MCPService:
        """Like :meth:`find_for_tenant` but raises ``mcp_service.not_found``."""
        row = await self.find_for_tenant(tenant_id, id)
        if row is None:
            raise NotFoundError(
                code=_NOT_FOUND_CODE,
                message=f"MCP service {id} not found",
            )
        return row

    async def list_for_tenant(self, tenant_id: int) -> list[MCPService]:
        """Live tenant-scoped rows (excludes builtin), newest first."""
        stmt = text(
            f"select * from {_TABLE_NAME} "
            "where tenant_id = :tenant_id and is_builtin = false "
            "and deleted_at is null "
            "order by created_at desc"
        ).bindparams(tenant_id=tenant_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def find_builtin(self, id: str) -> MCPService | None:
        """Return the live builtin row for an id (cross-tenant)."""
        return await self.find_unique_by_column_values(
            {"is_builtin": True, "id": id},
        )

    async def exists_by_tenant_and_name(
        self,
        tenant_id: int,
        name: str,
    ) -> bool:
        """Return whether a live row with ``(tenant_id, name)`` exists.

        Mirrors Go's ``MCPServiceRepository.FindByName`` used by the
        service-layer 409 path. The DB-level unique constraint on
        ``(tenant_id, name)`` is the authoritative race-condition guard;
        this method is a fast pre-check that surfaces the conflict
        without paying the insert cost.
        """
        row = await self.find_unique_by_column_values(
            {"tenant_id": tenant_id, "name": name},
        )
        return row is not None

    async def soft_delete(
        self,
        tenant_id: int,
        id: str,
        *,
        deleted_at: datetime,
    ) -> bool:
        """Soft-delete a live row; return whether one existed."""
        affected = await self._update_live(
            "tenant_id = :tenant_id and id = :id",
            {"tenant_id": tenant_id, "id": id},
            {"deleted_at": deleted_at, "updated_at": deleted_at},
        )
        return affected > 0

    async def update(
        self,
        tenant_id: int,
        id: str,
        *,
        columns: BindParams,
    ) -> MCPService | None:
        """Update a live row by primary key; return the refreshed row.

        ``columns`` carries only the mutable columns (NOT including the
        primary key). ``updated_at`` should be set by the caller for a
        consistent audit trail.
        """
        where_params: BindParams = {"tenant_id": tenant_id, "id": id}
        return await self._update_live_with_where(
            columns=columns,
            where_params=where_params,
        )

    # ── Query helpers ─────────────────────────────────────────────

    async def _update_live(
        self,
        where_sql: str,
        where_params: BindParams,
        columns: BindParams,
    ) -> int:
        """Internal: ``update ... set ... where ... and deleted_at is null``."""
        set_clause = ", ".join(f'"{c}" = :u_{c}' for c in columns)
        update_params: BindParams = {f"u_{k}": v for k, v in columns.items()}
        stmt_text = (
            f"update {_TABLE_NAME} set {set_clause} where {where_sql} and deleted_at is null"
        )
        # JSONB columns must be bound with the JSON type so asyncpg
        # serialises dict values; without it the driver rejects the
        # raw dict (``'dict' object has no attribute 'encode'``).
        json_bps = [
            bindparam(f"u_{col}", type_=_JSON_BIND_TYPE)
            for col in columns
            if col in self._json_columns
        ]
        stmt = text(stmt_text).bindparams(*json_bps, **update_params, **where_params)
        result = await self._session.execute(stmt)
        return cast("CursorResult[SqlValue]", result).rowcount

    async def _update_live_with_where(
        self,
        *,
        columns: BindParams,
        where_params: BindParams,
    ) -> MCPService | None:
        """Update + return the refreshed row."""
        where_sql = " and ".join(f'"{k}" = :{k}' for k in where_params)
        affected = await self._update_live(where_sql, where_params, columns)
        if affected == 0:
            return None
        return await self.find_unique_by_column_values(where_params)


__all__ = ["MCPServiceRepository"]
