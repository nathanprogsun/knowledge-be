"""IM-channel persistence — raw SQL only, no ORM.

The Python side adds a clean repository abstraction for ``im_channels``;
upstream stores these via inline queries in the IM service package, so
there is no separate repository file on the Go side. Every method here
maps one of those inline queries.

Each method scopes by ``tenant_id`` where the upstream query does, so a
caller can never read or mutate another workspace's rows. Reads filter
``deleted_at IS NULL`` (via the base ``GenericRepository`` helpers) so a
soft-deleted row behaves as if it no longer exists.

``find_by_bot_identity`` backs the duplicate-bot guard: the service
derives a channel's bot identity from platform + mode + credential
fields, and the DB unique index on ``bot_identity`` (partial, live rows
only) is the safety net.
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from sqlalchemy import text
from sqlalchemy.engine import CursorResult

from src.common.exception import DataError
from src.common.json import BindParams, SqlValue
from src.db.dao.generic_repository import GenericRepository
from src.db.models.im_channel import IMChannel


class IMChannelRepository(GenericRepository[IMChannel]):
    """`im_channels`-table SQL — tenant-scoped CRUD + bot-identity lookup."""

    model_class = IMChannel

    # ── Reads ───────────────────────────────────────────────────────

    async def get_by_id(self, tenant_id: int, channel_id: str) -> IMChannel | None:
        """Return one live channel by primary key + tenant scope."""
        return await self.find_unique_by_column_values(
            {"id": channel_id, "tenant_id": tenant_id},
        )

    async def get_by_id_global(self, channel_id: str) -> IMChannel | None:
        """Return one live channel by id, ignoring tenant scope.

        Used by the supervisor, which runs across tenants and looks up
        channels by id alone.
        """
        return await self.find_unique_by_column_values({"id": channel_id})

    async def list_by_agent(self, tenant_id: int, agent_id: str) -> list[IMChannel]:
        """Return every live channel of ``agent_id`` within the tenant, newest first."""
        stmt = text(
            "select * from im_channels "
            "where tenant_id = :tenant_id and agent_id = :agent_id and deleted_at is null "
            "order by created_at desc"
        ).bindparams(tenant_id=tenant_id, agent_id=agent_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def list_by_tenant(self, tenant_id: int) -> list[IMChannel]:
        """Return every live channel of the tenant, newest first."""
        stmt = text(
            "select * from im_channels "
            "where tenant_id = :tenant_id and deleted_at is null "
            "order by created_at desc"
        ).bindparams(tenant_id=tenant_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def list_enabled(self) -> list[IMChannel]:
        """Return every live, enabled channel — the supervisor startup set."""
        stmt = text(
            "select * from im_channels "
            "where enabled = true and deleted_at is null "
            "order by created_at asc"
        )
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def find_by_bot_identity(
        self,
        bot_identity: str,
        *,
        exclude_id: str = "",
    ) -> IMChannel | None:
        """Return the live channel bound to ``bot_identity``, or ``None``.

        ``exclude_id`` skips the caller's own row so an update does not
        trip the duplicate-bot guard against itself.
        """
        params: BindParams = {"bot_identity": bot_identity}
        where = "bot_identity = :bot_identity and deleted_at is null"
        if exclude_id:
            where += " and id != :exclude_id"
            params["exclude_id"] = exclude_id
        stmt = text(f"select * from im_channels where {where}").bindparams(**params)
        result = await self._session.execute(stmt)
        return self._hydrate_opt(result.mappings().first())

    # ── Mutations ───────────────────────────────────────────────────

    async def create(self, row: IMChannel) -> IMChannel:
        """Insert a channel and return the persisted row."""
        return await self.insert(row)

    async def update(self, row: IMChannel) -> IMChannel:
        """Overwrite every mutable column of the row, returning the result.

        ``id`` / ``tenant_id`` / ``created_at`` are immutable by contract
        and stay out of the SET clause.
        """
        immutable = {"id", "tenant_id", "created_at"}
        updates = {k: v for k, v in row.model_dump().items() if k not in immutable}
        persisted = await self.update_by_primary_key(
            {"id": row.id, "tenant_id": row.tenant_id},
            updates,
        )
        if persisted is None:
            raise DataError(
                code="im_channel.update_no_row",
                message=f"im channel {row.id} not found for update",
            )
        return persisted

    async def soft_delete(
        self,
        *,
        channel_id: str,
        tenant_id: int,
        now: datetime,
    ) -> bool:
        """Mark the channel deleted. Returns whether a live row was affected."""
        stmt = text(
            "update im_channels set deleted_at = :now, updated_at = :now "
            "where id = :channel_id and tenant_id = :tenant_id and deleted_at is null"
        ).bindparams(channel_id=channel_id, tenant_id=tenant_id, now=now)
        result = cast("CursorResult[SqlValue]", await self._session.execute(stmt))
        return (result.rowcount or 0) > 0

    async def soft_delete_by_agent(
        self,
        *,
        agent_id: str,
        tenant_id: int,
        now: datetime,
    ) -> int:
        """Soft-delete every live channel bound to ``agent_id`` in the tenant.

        Used when a custom agent is removed so overview lists and running
        adapters do not outlive the agent. Returns the row count affected.
        """
        stmt = text(
            "update im_channels set deleted_at = :now, updated_at = :now "
            "where agent_id = :agent_id and tenant_id = :tenant_id and deleted_at is null"
        ).bindparams(agent_id=agent_id, tenant_id=tenant_id, now=now)
        result = cast("CursorResult[SqlValue]", await self._session.execute(stmt))
        return int(result.rowcount or 0)

    async def toggle_enabled(
        self,
        *,
        channel_id: str,
        tenant_id: int,
        now: datetime,
    ) -> IMChannel | None:
        """Flip ``enabled`` on a live channel, returning the updated row.

        Returns ``None`` when no live row matched the tenant-scoped id.
        """
        stmt = text(
            "update im_channels set enabled = not enabled, updated_at = :now "
            "where id = :channel_id and tenant_id = :tenant_id and deleted_at is null "
            "returning *"
        ).bindparams(channel_id=channel_id, tenant_id=tenant_id, now=now)
        result = await self._session.execute(stmt)
        return self._hydrate_opt(result.mappings().first())


__all__ = ["IMChannelRepository"]
