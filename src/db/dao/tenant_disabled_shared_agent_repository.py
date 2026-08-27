"""Per-tenant shared-agent hide preference persistence — raw SQL only.

The `tenant_disabled_shared_agents` table records that a tenant has
hidden a shared agent from its conversation dropdown. The composite
primary key ``(tenant_id, agent_id, source_tenant_id)`` lets a tenant
disable the same agent shared by multiple source tenants independently.

Rows are append-only: disabling inserts a row, re-enabling deletes it.
There is no mutation path other than insert and delete, matching the
upstream repository surface.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import text

from src.db.dao.generic_repository import GenericRepository
from src.db.models.tenant_disabled_shared_agent import TenantDisabledSharedAgent

_TABLE = "tenant_disabled_shared_agents"


class TenantDisabledSharedAgentRepository(GenericRepository[TenantDisabledSharedAgent]):
    """`tenant_disabled_shared_agents`-table SQL — list / add / remove."""

    model_class = TenantDisabledSharedAgent

    # ── Reads ───────────────────────────────────────────────────────

    async def list_by_tenant(self, tenant_id: int) -> list[TenantDisabledSharedAgent]:
        """Every hide row of one tenant, oldest first."""
        stmt = text(
            f"select * from {_TABLE} where tenant_id = :tenant_id order by created_at asc"
        ).bindparams(tenant_id=tenant_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    # ── Writes ──────────────────────────────────────────────────────

    async def add(
        self,
        *,
        tenant_id: int,
        agent_id: str,
        source_tenant_id: int,
    ) -> None:
        """Record a hide, suppressing a duplicate (idempotent).

        The conflict target matches the composite primary key, so a
        second insert for the same tuple is a no-op rather than an error.
        """
        row = TenantDisabledSharedAgent(
            tenant_id=tenant_id,
            agent_id=agent_id,
            source_tenant_id=source_tenant_id,
            created_at=datetime.now(UTC),
        )
        await self.insert_or_none(
            row,
            on_conflict_do_nothing_target_columns=[
                "tenant_id",
                "agent_id",
                "source_tenant_id",
            ],
        )

    async def remove(
        self,
        *,
        tenant_id: int,
        agent_id: str,
        source_tenant_id: int,
    ) -> None:
        """Drop the hide row for the (tenant, agent, source) tuple."""
        stmt = text(
            f"delete from {_TABLE} "
            "where tenant_id = :tenant_id "
            "  and agent_id = :agent_id "
            "  and source_tenant_id = :source_tenant_id"
        ).bindparams(
            tenant_id=tenant_id,
            agent_id=agent_id,
            source_tenant_id=source_tenant_id,
        )
        await self._session.execute(stmt)


__all__ = ["TenantDisabledSharedAgentRepository"]
