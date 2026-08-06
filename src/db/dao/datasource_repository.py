"""Data-source and sync-log persistence — raw SQL only, no ORM.

Maps the methods declared upstream in
``internal/types/interfaces/datasource.go`` (``DataSourceRepository`` /
``SyncLogRepository``):

``DataSourceRepository``
    ``Create`` / ``FindByID`` / ``FindByKnowledgeBase`` / ``Update`` /
    ``Delete`` (soft) — plus ``count_items_synced`` which backs the
    ``total_items_synced`` field the Go service computes per query.

``SyncLogRepository``
    ``Create`` / ``FindByID`` / ``FindByDataSource`` / ``FindLatest`` /
    ``Update`` / ``CancelPendingByDataSource``.

Every query is ``sqlalchemy.text()`` with named ``bindparams``; JSON
columns are bound through the ``GenericRepository`` JSONB helper. Reads
filter soft-deleted data sources (``deleted_at is null``).
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from sqlalchemy import CursorResult, text

from src.common.exception import DataError
from src.common.json import SqlValue
from src.db.dao.generic_repository import GenericRepository
from src.db.models.datasource import DataSource, SyncLog

# Sync-log statuses whose asynq task may still be queued; cancelled when
# the owning data source is deleted so a retry cannot resurrect it.
_PENDING_SYNC_STATUSES: tuple[str, ...] = ("running",)

_CANCELED = "canceled"


class DataSourceRepository(GenericRepository[DataSource]):
    """`data_sources`-table SQL — CRUD + KB-scoped listing."""

    model_class = DataSource

    async def create(self, row: DataSource) -> DataSource:
        """Insert a data source and return the persisted row."""
        return await self.insert(row)

    async def find_by_id_or_none(self, id: str) -> DataSource | None:
        """Return the live row for ``id``, or ``None`` when absent."""
        return await self.find_by_primary_key({"id": id})

    async def find_by_knowledge_base(self, knowledge_base_id: str) -> list[DataSource]:
        """Return every live data source of a knowledge base, newest first."""
        stmt = text(
            "select * from data_sources where knowledge_base_id = :kb_id "
            "and deleted_at is null order by created_at desc"
        ).bindparams(kb_id=knowledge_base_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def update(self, row: DataSource) -> DataSource:
        """Overwrite every mutable column of the row, returning the result.

        ``id`` / ``tenant_id`` / ``knowledge_base_id`` / ``created_at``
        are immutable by contract (the Go service rejects changes to the
        latter two before reaching the repo), so they stay out of the SET
        clause.
        """
        immutable = {"id", "tenant_id", "knowledge_base_id", "created_at"}
        updates = {k: v for k, v in row.model_dump().items() if k not in immutable}
        persisted = await self.update_by_primary_key({"id": row.id}, updates)
        if persisted is None:
            raise DataError(
                code="datasource.update_no_row",
                message=f"data source {row.id} not found for update",
            )
        return persisted

    async def soft_delete(self, *, id: str, now: datetime) -> bool:
        """Mark the row deleted. Returns whether a live row was affected."""
        stmt = text(
            "update data_sources set deleted_at = :now, updated_at = :now "
            "where id = :id and deleted_at is null"
        ).bindparams(id=id, now=now)
        result = await self._session.execute(stmt)
        return (cast("CursorResult[SqlValue]", result).rowcount or 0) > 0

    async def count_items_synced(self, data_source_id: str) -> int:
        """Sum created+updated items across the source's successful syncs.

        Backs the non-persisted ``total_items_synced`` response field
        (Go computes it on query rather than storing it).
        """
        stmt = text(
            "select coalesce(sum(items_created + items_updated), 0) as total "
            "from sync_logs where data_source_id = :ds_id "
            "and status in ('success', 'partial')"
        ).bindparams(ds_id=data_source_id)
        result = await self._session.execute(stmt)
        total = result.scalar_one()
        return int(total) if total is not None else 0


class SyncLogRepository(GenericRepository[SyncLog]):
    """`sync_logs`-table SQL — append, update, and per-source history."""

    model_class = SyncLog

    async def create(self, row: SyncLog) -> SyncLog:
        """Insert a sync-log row and return the persisted row."""
        return await self.insert(row)

    async def find_by_id_or_none(self, id: str) -> SyncLog | None:
        """Return the sync log for ``id``, or ``None`` when absent."""
        return await self.find_by_primary_key({"id": id})

    async def find_by_data_source(
        self,
        data_source_id: str,
        *,
        limit: int,
        offset: int,
    ) -> list[SyncLog]:
        """Return a page of the source's sync history, newest first."""
        stmt = text(
            "select * from sync_logs where data_source_id = :ds_id "
            "order by started_at desc limit :limit offset :offset"
        ).bindparams(ds_id=data_source_id, limit=limit, offset=offset)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def find_latest(self, data_source_id: str) -> SyncLog | None:
        """Return the source's most recent sync log, or ``None``."""
        stmt = text(
            "select * from sync_logs where data_source_id = :ds_id order by started_at desc limit 1"
        ).bindparams(ds_id=data_source_id)
        result = await self._session.execute(stmt)
        return self._hydrate_opt(result.mappings().first())

    async def update(self, row: SyncLog) -> SyncLog:
        """Overwrite every mutable column of the sync-log row."""
        immutable = {"id", "data_source_id", "tenant_id", "created_at"}
        updates = {k: v for k, v in row.model_dump().items() if k not in immutable}
        persisted = await self.update_by_primary_key({"id": row.id}, updates)
        if persisted is None:
            raise DataError(
                code="datasource.sync_log_update_no_row",
                message=f"sync log {row.id} not found for update",
            )
        return persisted

    async def cancel_pending_by_data_source(
        self,
        *,
        data_source_id: str,
        now: datetime,
    ) -> int:
        """Cancel still-running sync logs of a source. Returns the count.

        Called when the owning data source is deleted so a queued task
        retry cannot report progress against a dead source.
        """
        placeholders = ", ".join(f":s{i}" for i in range(len(_PENDING_SYNC_STATUSES)))
        params: dict[str, str | datetime] = {
            f"s{i}": s for i, s in enumerate(_PENDING_SYNC_STATUSES)
        }
        params["ds_id"] = data_source_id
        params["now"] = now
        params["canceled"] = _CANCELED
        stmt = text(
            "update sync_logs set status = :canceled, finished_at = :now, updated_at = :now "
            f"where data_source_id = :ds_id and status in ({placeholders})"
        ).bindparams(**params)
        result = await self._session.execute(stmt)
        return cast("CursorResult[SqlValue]", result).rowcount or 0


__all__ = ["DataSourceRepository", "SyncLogRepository"]
