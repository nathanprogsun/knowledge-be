"""Agent share persistence — raw SQL only, no ORM.

One row per (agent, source tenant, organization) share. The tuple is
unique among live rows (partial unique index on ``deleted_at is null``),
so ``create_or_none`` suppresses a duplicate share instead of raising —
the service layer treats that as an idempotent no-op.

Every read filters soft-deleted rows (``deleted_at is null``).
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from sqlalchemy import CursorResult, text

from src.common.exception import DataError
from src.common.json import SqlValue
from src.db.dao.generic_repository import GenericRepository
from src.db.models.agent_share import AgentShare

_LIVE = "deleted_at is null"

# Newest first: the share list reads as a reverse-chronological feed.
_SHARE_ORDER = "created_at desc, id desc"

# Module-level alias for the table name. Every ``text(f"...{...}")`` in
# this file interpolates this constant; user input never reaches the SQL
# string.
_AGENT_SHARE_TABLE = "agent_shares"


class AgentShareRepository(GenericRepository[AgentShare]):
    """`agent_shares`-table SQL — CRUD + per-scope lists and counts."""

    model_class = AgentShare

    # ── Writes ──────────────────────────────────────────────────────

    async def create(self, row: AgentShare) -> AgentShare:
        """Insert a share row and return the persisted row."""
        return await self.insert(row)

    async def create_or_none(self, row: AgentShare) -> AgentShare | None:
        """Insert a share, returning ``None`` on a live duplicate.

        The conflict target matches the partial unique index on
        ``(agent_id, source_tenant_id, organization_id) WHERE deleted_at
        IS NULL``, so a second live share for the same tuple is
        suppressed while soft-deleted rows insert freely.
        """
        columns = self.model_class.insert_sql_column_list()
        column_list = ", ".join(f'"{c}"' for c in columns)
        value_list = ", ".join(f":{c}" for c in columns)
        stmt = text(
            f"insert into {_AGENT_SHARE_TABLE} ({column_list}) values ({value_list}) "
            "on conflict (agent_id, source_tenant_id, organization_id) "
            f"where {_LIVE} do nothing returning *"
        ).bindparams(**row.insert_bind_params())
        result = await self._session.execute(stmt)
        return self._hydrate_opt(result.mappings().first())

    async def update(self, row: AgentShare) -> AgentShare:
        """Overwrite every mutable column of the row, returning the result.

        ``id`` / ``agent_id`` / ``organization_id`` / ``source_tenant_id``
        / ``created_at`` are immutable by contract, so they stay out of
        the SET clause.
        """
        immutable = {"id", "agent_id", "organization_id", "source_tenant_id", "created_at"}
        updates = {k: v for k, v in row.model_dump().items() if k not in immutable}
        persisted = await self.update_by_primary_key({"id": row.id}, updates)
        if persisted is None:
            raise DataError(
                code="agent_share.update_no_row",
                message=f"agent share {row.id} not found for update",
            )
        return persisted

    async def soft_delete(self, *, id: str, now: datetime) -> bool:
        """Mark the row deleted. Returns whether a live row was affected."""
        stmt = text(
            f"update {_AGENT_SHARE_TABLE} set deleted_at = :now, updated_at = :now "
            f"where id = :id and {_LIVE}"
        ).bindparams(id=id, now=now)
        result = cast("CursorResult[SqlValue]", await self._session.execute(stmt))
        return (result.rowcount or 0) > 0

    async def delete_by_agent(self, *, agent_id: str, source_tenant_id: int, now: datetime) -> int:
        """Soft-delete every share of an agent. Returns rows affected."""
        stmt = text(
            f"update {_AGENT_SHARE_TABLE} set deleted_at = :now, updated_at = :now "
            f"where agent_id = :agent_id and source_tenant_id = :source_tenant_id and {_LIVE}"
        ).bindparams(agent_id=agent_id, source_tenant_id=source_tenant_id, now=now)
        result = cast("CursorResult[SqlValue]", await self._session.execute(stmt))
        return result.rowcount or 0

    async def delete_by_organization(self, *, organization_id: str, now: datetime) -> int:
        """Soft-delete every share into an organization. Returns rows affected."""
        stmt = text(
            f"update {_AGENT_SHARE_TABLE} set deleted_at = :now, updated_at = :now "
            f"where organization_id = :organization_id and {_LIVE}"
        ).bindparams(organization_id=organization_id, now=now)
        result = cast("CursorResult[SqlValue]", await self._session.execute(stmt))
        return result.rowcount or 0

    # ── Reads ───────────────────────────────────────────────────────

    async def get_by_id_or_none(self, id: str) -> AgentShare | None:
        """Return the live row for ``id``, or ``None`` when absent."""
        return await self.find_by_primary_key({"id": id})

    async def get_by_agent_and_org_or_none(
        self,
        *,
        agent_id: str,
        organization_id: str,
    ) -> AgentShare | None:
        """Return the live share for the (agent, org) pair, or ``None``."""
        return await self.find_unique_by_column_values(
            {"agent_id": agent_id, "organization_id": organization_id},
        )

    async def list_by_agent(self, agent_id: str) -> list[AgentShare]:
        """Every live share of one agent, newest first."""
        stmt = text(
            f"select * from {_AGENT_SHARE_TABLE} "
            f"where agent_id = :agent_id and {_LIVE} "
            f"order by {_SHARE_ORDER}"
        ).bindparams(agent_id=agent_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def list_by_organization(self, organization_id: str) -> list[AgentShare]:
        """Every live share into one organization, newest first."""
        stmt = text(
            f"select * from {_AGENT_SHARE_TABLE} "
            f"where organization_id = :organization_id and {_LIVE} "
            f"order by {_SHARE_ORDER}"
        ).bindparams(organization_id=organization_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def count_by_agent(self, agent_id: str) -> int:
        """Count live shares of one agent."""
        stmt = text(
            f"select count(*) from {_AGENT_SHARE_TABLE} "
            f"where agent_id = :agent_id and {_LIVE}"
        ).bindparams(agent_id=agent_id)
        return int((await self._session.execute(stmt)).scalar_one())

    async def list_shared_for_tenant(self, tenant_id: int) -> list[AgentShare]:
        """Every live share into an organization the tenant belongs to.

        The member-org join narrows the sweep to shares the tenant can
        actually reach; the org and agent joins drop shares whose owning
        organization or agent was soft-deleted (mirrors the upstream
        tenant-scoped share list).
        """
        stmt = text(
            f"select ags.* from {_AGENT_SHARE_TABLE} ags "
            "join organization_tenant_members otm "
            "  on otm.organization_id = ags.organization_id "
            "join organizations o "
            "  on o.id = ags.organization_id and o.deleted_at is null "
            "join custom_agents ca "
            "  on ca.id = ags.agent_id "
            "  and ca.tenant_id = ags.source_tenant_id "
            "  and ca.deleted_at is null "
            "where otm.tenant_id = :tenant_id and ags.deleted_at is null "
            f"order by {_SHARE_ORDER}"
        ).bindparams(tenant_id=tenant_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def get_share_for_tenant(
        self,
        *,
        tenant_id: int,
        agent_id: str,
        exclude_source_tenant_id: int,
    ) -> AgentShare | None:
        """Return one live share of ``agent_id`` reachable by the tenant.

        The caller's own tenant is excluded as a source so an agent the
        caller owns never resolves through a share row; the member-org
        join keeps the lookup tenant-scoped.
        """
        stmt = text(
            f"select ags.* from {_AGENT_SHARE_TABLE} ags "
            "join organization_tenant_members otm "
            "  on otm.organization_id = ags.organization_id "
            "where otm.tenant_id = :tenant_id "
            "  and ags.agent_id = :agent_id "
            "  and ags.source_tenant_id != :exclude_source_tenant_id "
            f"  and ags.{_LIVE} "
            "order by ags.id "
            "limit 1"
        ).bindparams(
            tenant_id=tenant_id,
            agent_id=agent_id,
            exclude_source_tenant_id=exclude_source_tenant_id,
        )
        result = await self._session.execute(stmt)
        return self._hydrate_opt(result.mappings().first())


__all__ = ["AgentShareRepository"]
