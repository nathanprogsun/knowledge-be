"""Sync engine — ``ManualSync`` / ``ProcessSync``.

Mixed into ``DataSourceService``. Two halves of one flow:

``manual_sync``
    The request-time half. Opens a ``running`` sync log and hands back its
    id so the caller can poll. Go additionally enqueues an asynq task
    here; there is no task queue in this codebase yet, so the log is
    opened and the worker PR will pick up ``running`` logs. The failure
    bookkeeping Go performs when the enqueue itself fails (log →
    ``failed``, source → ``error``) is implemented in
    :meth:`SyncMixin.fail_sync` so the worker PR can call it unchanged.

``process_sync``
    The worker half. Resolves the connector, walks the source
    (incremental when a cursor exists and the mode allows, full
    otherwise), ingests each item through an injected
    :class:`ItemIngestor`, and closes the log with the tallied
    ``SyncResult``. Terminal status follows Go: ``success`` when nothing
    failed, ``partial`` when some items failed but not all, ``failed``
    when every item failed.

Knowledge ingestion itself (parse → chunk → embed) belongs to the
knowledge domain, which is a later stage. ``ItemIngestor`` is the seam:
this module owns the state machine and the counters; the ingestor owns
what a fetched item becomes. With no ingestor injected, items are
counted as skipped, which is what makes the sync state machine testable
today without the knowledge stack.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from src.common.exception import ApplicationError, ValidationError
from src.core.infra.datasources.connector_base import ConnectorRegistry
from src.core.infra.datasources.types import (
    DATA_SOURCE_STATUS_ERROR,
    DATA_SOURCE_STATUS_PAUSED,
    MANUAL_SYNC_ALLOWED_STATUSES,
    SYNC_LOG_STATUS_FAILED,
    SYNC_LOG_STATUS_PARTIAL,
    SYNC_LOG_STATUS_RUNNING,
    SYNC_LOG_STATUS_SUCCESS,
    SYNC_MODE_FULL,
    FetchedItem,
    SyncCursor,
    SyncItemError,
    SyncLogInfo,
    SyncResult,
    parse_config,
)
from src.db.dao.datasource_repository import DataSourceRepository, SyncLogRepository
from src.db.models.datasource import DataSource, SyncLog

# Per-item failure samples kept on a sync result. Uncapped, a feed whose
# every item fails would write a multi-megabyte JSONB blob.
MAX_SYNC_ERROR_SAMPLES = 20


@runtime_checkable
class ItemIngestor(Protocol):
    """Writes one fetched item into a knowledge base.

    Returns ``True`` when an existing knowledge item was replaced (an
    update) and ``False`` when a new one was created, so the sync tally
    can separate ``created`` from ``updated`` — the same
    ``(isUpdate, error)`` contract as Go's ``ingestItem``.
    """

    async def ingest(self, *, data_source: DataSource, item: FetchedItem) -> bool: ...

    async def delete(self, *, data_source: DataSource, external_id: str) -> bool:
        """Remove a knowledge item whose source counterpart was deleted.

        Returns whether something was actually removed.
        """
        ...


class SyncMixin:
    """``ManualSync`` / ``ProcessSync`` for ``DataSourceService``."""

    _ds_repo: DataSourceRepository
    _sync_log_repo: SyncLogRepository
    _connector_registry: ConnectorRegistry
    _ingestor: ItemIngestor | None = None

    async def _require_owned(self, *, id: str, tenant_id: int) -> DataSource:  # pragma: no cover
        raise NotImplementedError

    # ── Request-time half ─────────────────────────────────────────────

    async def manual_sync(self, *, id: str, tenant_id: int) -> SyncLogInfo:
        """Open a ``running`` sync log for an immediate sync.

        Rejects a source in a status that cannot sync. ``paused`` is
        allowed on purpose (matching Go): a manual run is an explicit
        override of the schedule, not a resume.
        """
        row = await self._require_owned(id=id, tenant_id=tenant_id)
        if row.status not in MANUAL_SYNC_ALLOWED_STATUSES:
            raise ValidationError(
                code="datasource.not_active",
                message="data source is not active",
            )
        now = datetime.now(UTC)
        log = await self._sync_log_repo.create(
            SyncLog(
                id=str(uuid.uuid4()),
                data_source_id=row.id,
                tenant_id=row.tenant_id,
                status=SYNC_LOG_STATUS_RUNNING,
                started_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        return SyncLogInfo.map_from_db(log)

    async def fail_sync(
        self,
        *,
        sync_log_id: str,
        data_source: DataSource,
        message: str,
    ) -> SyncLogInfo:
        """Close a sync log as ``failed`` and flag its source.

        Used when the run could not even start (Go: enqueue failure). A
        ``paused`` source keeps its status — pausing is a user decision
        that a failed dispatch must not silently undo.
        """
        now = datetime.now(UTC)
        log = await self._sync_log_repo.find_by_id_or_none(sync_log_id)
        if log is None:
            raise ValidationError(
                code="datasource.sync_log_not_found",
                message="sync log not found",
            )
        closed = await self._sync_log_repo.update(
            log.model_copy(
                update={
                    "status": SYNC_LOG_STATUS_FAILED,
                    "finished_at": now,
                    "error_message": message,
                    "updated_at": now,
                }
            )
        )
        await self._ds_repo.update(
            data_source.model_copy(
                update={
                    "status": (
                        DATA_SOURCE_STATUS_PAUSED
                        if data_source.status == DATA_SOURCE_STATUS_PAUSED
                        else DATA_SOURCE_STATUS_ERROR
                    ),
                    "error_message": message,
                    "updated_at": now,
                }
            )
        )
        return SyncLogInfo.map_from_db(closed)

    # ── Worker half ───────────────────────────────────────────────────

    async def process_sync(
        self,
        *,
        data_source_id: str,
        sync_log_id: str,
        force_full: bool = False,
        max_items: int = 0,
    ) -> SyncLogInfo:
        """Run one sync to completion and close its log.

        ``force_full`` ignores the stored cursor. ``max_items`` caps how
        many items are ingested (``0`` = unlimited), which is what lets a
        first run be bounded while a schema is still being tuned.
        """
        row = await self._ds_repo.find_by_id_or_none(data_source_id)
        if row is None:
            raise ValidationError(
                code="datasource.not_found",
                message="data source not found",
            )
        log = await self._sync_log_repo.find_by_id_or_none(sync_log_id)
        if log is None:
            raise ValidationError(
                code="datasource.sync_log_not_found",
                message="sync log not found",
            )
        config = parse_config(row.config)
        if config is None:
            raise ValidationError(
                code="datasource.invalid_config",
                message="invalid configuration",
            )
        connector = self._connector_registry.get(row.type)

        use_incremental = (
            not force_full and row.sync_mode != SYNC_MODE_FULL and row.last_sync_cursor is not None
        )
        next_cursor: SyncCursor | None = None
        if use_incremental:
            prior = SyncCursor.model_validate(row.last_sync_cursor)
            items, next_cursor = await connector.fetch_incremental(config, prior)
        else:
            items = await connector.fetch_all(config, config.resource_ids)

        if max_items > 0:
            items = items[:max_items]

        result = await self._apply_items(row, items)
        return await self._close_sync_run(
            row=row,
            log=log,
            result=result,
            next_cursor=next_cursor,
        )

    # ── Internals ─────────────────────────────────────────────────────

    async def _apply_items(self, row: DataSource, items: list[FetchedItem]) -> SyncResult:
        """Ingest every item, tallying the run into a ``SyncResult``.

        A per-item failure is recorded and the walk continues — one
        unparseable document must not abort a 10k-item sync. Deletions are
        honoured only when the source opted into ``sync_deletions``;
        otherwise a removed upstream item is counted as skipped rather
        than silently dropping knowledge the user may still want.
        """
        created = updated = deleted = skipped = failed = 0
        errors: list[SyncItemError] = []
        for item in items:
            try:
                if item.is_deleted:
                    if not row.sync_deletions:
                        skipped += 1
                        continue
                    if await self._delete_item(row, item):
                        deleted += 1
                    else:
                        skipped += 1
                    continue
                outcome = await self._ingest_item(row, item)
                if outcome is None:
                    skipped += 1
                elif outcome:
                    updated += 1
                else:
                    created += 1
            except ApplicationError as exc:
                failed += 1
                if len(errors) < MAX_SYNC_ERROR_SAMPLES:
                    errors.append(
                        SyncItemError(
                            title=item.title,
                            code=exc.code,
                            message=exc.message,
                        )
                    )
        return SyncResult(
            total=len(items),
            created=created,
            updated=updated,
            deleted=deleted,
            skipped=skipped,
            failed=failed,
            errors=errors,
        )

    async def _ingest_item(self, row: DataSource, item: FetchedItem) -> bool | None:
        """Ingest one item. ``None`` when no ingestor is wired up."""
        if self._ingestor is None:
            return None
        return await self._ingestor.ingest(data_source=row, item=item)

    async def _delete_item(self, row: DataSource, item: FetchedItem) -> bool:
        """Delete one item's knowledge counterpart, if an ingestor exists."""
        if self._ingestor is None:
            return False
        return await self._ingestor.delete(data_source=row, external_id=item.external_id)

    async def _close_sync_run(
        self,
        *,
        row: DataSource,
        log: SyncLog,
        result: SyncResult,
        next_cursor: SyncCursor | None,
    ) -> SyncLogInfo:
        """Persist the terminal sync log and the source's sync state."""
        now = datetime.now(UTC)
        status = _terminal_status(result)
        error_message = "; ".join(e.display() for e in result.errors) if result.failed else ""
        closed = await self._sync_log_repo.update(
            log.model_copy(
                update={
                    "status": status,
                    "finished_at": now,
                    "items_total": result.total,
                    "items_created": result.created,
                    "items_updated": result.updated,
                    "items_deleted": result.deleted,
                    "items_skipped": result.skipped,
                    "items_failed": result.failed,
                    "error_message": error_message,
                    "result": result.model_dump(mode="json"),
                    "updated_at": now,
                }
            )
        )
        source_status = (
            DATA_SOURCE_STATUS_ERROR if status == SYNC_LOG_STATUS_FAILED else _resume_status(row)
        )
        await self._ds_repo.update(
            row.model_copy(
                update={
                    "status": source_status,
                    "error_message": error_message,
                    "last_sync_at": now,
                    "last_sync_cursor": (
                        next_cursor.model_dump(mode="json")
                        if next_cursor is not None
                        else row.last_sync_cursor
                    ),
                    "last_sync_result": result.model_dump(mode="json"),
                    "updated_at": now,
                }
            )
        )
        return SyncLogInfo.map_from_db(closed)


def _terminal_status(result: SyncResult) -> str:
    """Map a tally onto the sync-log terminal status.

    ``failed`` requires that *every* fetched item failed; a run with some
    successes is ``partial`` so the user sees what did land.
    """
    if result.failed == 0:
        return SYNC_LOG_STATUS_SUCCESS
    if result.total > 0 and result.failed >= result.total:
        return SYNC_LOG_STATUS_FAILED
    return SYNC_LOG_STATUS_PARTIAL


def _resume_status(row: DataSource) -> str:
    """Status a source returns to after a non-failed run.

    ``paused`` survives a manual run; anything else settles on ``active``.
    """
    if row.status == DATA_SOURCE_STATUS_PAUSED:
        return DATA_SOURCE_STATUS_PAUSED
    return "active"


__all__ = ["MAX_SYNC_ERROR_SAMPLES", "ItemIngestor", "SyncMixin"]
