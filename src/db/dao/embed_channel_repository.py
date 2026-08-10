"""Embed-channel persistence — raw SQL only, no ORM.

Maps the upstream embed-channel repository: every method here mirrors
one of its queries. Reads filter ``deleted_at IS NULL`` (via the base
``GenericRepository`` helpers) so a soft-deleted row behaves as if it
no longer exists.

``get_by_publish_token`` backs the anonymous embed client: the publish
token is the public handle embedded in the widget script, so the lookup
must work without a tenant scope.
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from sqlalchemy import text
from sqlalchemy.engine import CursorResult

from src.common.exception import DataError
from src.common.json import SqlValue
from src.db.dao.generic_repository import GenericRepository
from src.db.models.embed_channel import EmbedChannel


class EmbedChannelRepository(GenericRepository[EmbedChannel]):
    """`embed_channels`-table SQL — CRUD + publish-token lookup."""

    model_class = EmbedChannel

    # ── Reads ───────────────────────────────────────────────────────

    async def get_by_id(self, channel_id: str) -> EmbedChannel | None:
        """Return one live channel by primary key."""
        return await self.find_unique_by_column_values({"id": channel_id})

    async def get_by_publish_token(self, token: str) -> EmbedChannel | None:
        """Return the live channel bound to ``token``, or ``None``."""
        return await self.find_unique_by_column_values({"publish_token": token})

    async def list_by_agent(self, tenant_id: int, agent_id: str) -> list[EmbedChannel]:
        """Return every live channel of ``agent_id`` within the tenant, newest first."""
        stmt = text(
            "select * from embed_channels "
            "where tenant_id = :tenant_id and agent_id = :agent_id and deleted_at is null "
            "order by created_at desc"
        ).bindparams(tenant_id=tenant_id, agent_id=agent_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def list_by_tenant(self, tenant_id: int) -> list[EmbedChannel]:
        """Return every live channel of the tenant, newest first."""
        stmt = text(
            "select * from embed_channels "
            "where tenant_id = :tenant_id and deleted_at is null "
            "order by created_at desc"
        ).bindparams(tenant_id=tenant_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    # ── Mutations ───────────────────────────────────────────────────

    async def create(self, row: EmbedChannel) -> EmbedChannel:
        """Insert a channel and return the persisted row."""
        return await self.insert(row)

    async def update(self, row: EmbedChannel) -> EmbedChannel:
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
                code="embed_channel.update_no_row",
                message=f"embed channel {row.id} not found for update",
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
            "update embed_channels set deleted_at = :now, updated_at = :now "
            "where id = :channel_id and tenant_id = :tenant_id and deleted_at is null"
        ).bindparams(channel_id=channel_id, tenant_id=tenant_id, now=now)
        result = cast("CursorResult[SqlValue]", await self._session.execute(stmt))
        return (result.rowcount or 0) > 0


__all__ = ["EmbedChannelRepository"]
