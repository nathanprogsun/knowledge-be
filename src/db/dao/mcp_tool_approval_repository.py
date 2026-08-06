"""MCP tool-approval persistence — raw SQL only, no ORM.

The Go repository surface is small: list approvals for a service and
upsert one (tenant, service, tool) record. There is no soft-delete —
clearing a flag is a write of ``require_approval = false``.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import text

from src.common.exception import DataError
from src.db.dao.generic_repository import GenericRepository
from src.db.models.infra.mcp_services import MCPToolApproval


class MCPToolApprovalRepository(GenericRepository[MCPToolApproval]):
    """``mcp_tool_approvals``-table SQL."""

    model_class = MCPToolApproval

    async def list_by_service(
        self,
        tenant_id: int,
        service_id: str,
    ) -> list[MCPToolApproval]:
        """Return every persisted approval override for the (tenant, service)."""
        stmt = text(
            f"select * from {self._table} "
            "where tenant_id = :tenant_id and service_id = :service_id "
            "order by tool_name"
        ).bindparams(tenant_id=tenant_id, service_id=service_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def upsert(
        self,
        *,
        row: MCPToolApproval,
    ) -> MCPToolApproval:
        """Insert or update by (tenant_id, service_id, tool_name).

        The Go side uses ``BeforeCreate`` to assign the UUID when
        omitted; here we require the row's ``id`` to be set explicitly
        so the repository stays a plain SQL adapter.
        """
        now = datetime.now(UTC)
        stmt = text(
            "insert into mcp_tool_approvals ("
            "id, tenant_id, service_id, tool_name, require_approval, "
            "created_at, updated_at"
            ") values ("
            ":id, :tenant_id, :service_id, :tool_name, :require_approval, "
            ":created_at, :updated_at"
            ") on conflict (tenant_id, service_id, tool_name) do update set "
            "require_approval = excluded.require_approval, updated_at = excluded.updated_at "
            "returning *"
        ).bindparams(
            id=row.id,
            tenant_id=row.tenant_id,
            service_id=row.service_id,
            tool_name=row.tool_name,
            require_approval=row.require_approval,
            created_at=row.created_at,
            updated_at=now,
        )
        result = await self._session.execute(stmt)
        mapping = result.mappings().first()
        # Upsert returning * yields exactly one row.
        if mapping is None:  # pragma: no cover — defensive
            raise DataError(
                code="db.upsert_no_row",
                message="mcp_tool_approvals upsert returned no row",
            )
        return self._hydrate(mapping)


__all__ = ["MCPToolApprovalRepository"]
