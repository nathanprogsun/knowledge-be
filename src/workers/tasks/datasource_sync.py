"""ARQ worker task: ``datasource_sync``.

Maps the upstream datasource-sync task: receives the serialized
``DataSourceSyncPayload`` over ARQ, validates it, and dispatches it to
the core sync engine :meth:`DataSourceService.process_sync`.

The handler stays thin — payload parsing, logging, and result shaping
live here; the fetch / ingest / sync-log-close orchestration lives in
the core layer. Wire field names mirror the upstream contract so
payloads enqueued by the existing web/CLI paths deserialize without
translation.

A session-bound :class:`DataSourceService` is injected by the worker
wiring layer (the worker context carries no database engine). Until
that wiring lands, an uninjected seam raises rather than silently
dropping a sync, so a miswired worker fails loudly.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.app_logging import logger
from src.core.infra.datasources.service.datasource_service import DataSourceService
from src.core.infra.datasources.types import SyncLogInfo
from src.workers.base import WorkerContext
from src.workers.registry import JsonValue, register_task


class DatasourceSyncPayload(BaseModel):
    """ARQ-side payload for the ``datasource_sync`` task.

    Mirrors the upstream wire contract: the data-source id and tenant
    are mandatory; the sync-log id, trigger and tuning flags are
    optional, matching the ``omitempty`` JSON tags on the upstream side.
    Field names use snake_case so ARQ's JSON deserializer maps
    transparently onto this model.

    The upstream tracing-context fields are accepted via the JSON
    payload but not modelled here (no consumer reads them yet), so
    ``extra="ignore"`` drops them without failing the parse.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    data_source_id: str
    tenant_id: int
    sync_log_id: str = ""
    force_full: bool = False
    max_items: int = 0
    trigger: str = ""


@register_task("datasource_sync")
async def task_datasource_sync(
    ctx: WorkerContext,
    *,
    service: DataSourceService | None = None,
    **payload: JsonValue,
) -> dict[str, JsonValue]:
    """ARQ handler for the ``datasource_sync`` task.

    Parses the ARQ payload into :class:`DatasourceSyncPayload` and
    delegates to :func:`process_datasource_sync`. A session-bound
    ``service`` may be injected by the worker wiring layer; when
    omitted, the seam raises so a miswired sync fails loudly. ``ctx``
    is currently unused — the worker context carries the ARQ-Redis
    pool, which the sync seam does not need at this stage.
    """
    parsed = DatasourceSyncPayload.model_validate(payload)
    return await process_datasource_sync(
        data_source_id=parsed.data_source_id,
        tenant_id=parsed.tenant_id,
        sync_log_id=parsed.sync_log_id,
        force_full=parsed.force_full,
        max_items=parsed.max_items,
        service=service,
    )


async def process_datasource_sync(
    *,
    data_source_id: str,
    tenant_id: int,
    sync_log_id: str,
    force_full: bool = False,
    max_items: int = 0,
    service: DataSourceService | None = None,
) -> dict[str, JsonValue]:
    """Run one data-source sync to completion via the core sync engine.

    Delegates to :meth:`DataSourceService.process_sync` and returns the
    terminal sync-log record as a JSON-serialisable dict. ``tenant_id``
    is carried for log correlation; the core engine resolves ownership
    from the stored data-source row.

    ``service`` is injected by the worker wiring layer — a
    :class:`DataSourceService` bound to a per-job ``AsyncSession``. No
    service can be constructed here (the worker context carries no
    database engine), so an uninjected call raises ``NotImplementedError``
    instead of silently skipping the sync.
    """
    if service is None:
        raise NotImplementedError(
            "data-source sync requires a session-bound DataSourceService "
            "injected by the worker wiring layer",
        )
    logger.info(
        "datasource_sync: tenant={} ds={} sync_log={} force_full={} max_items={}",
        tenant_id,
        data_source_id,
        sync_log_id,
        force_full,
        max_items,
    )
    info = await service.process_sync(
        data_source_id=data_source_id,
        sync_log_id=sync_log_id,
        force_full=force_full,
        max_items=max_items,
    )
    return _serialise_sync_log(info)


def _serialise_sync_log(info: SyncLogInfo) -> dict[str, JsonValue]:
    """Project a sync-log record onto a JSON-serialisable dict."""
    return info.model_dump(mode="json")


__all__ = [
    "DatasourceSyncPayload",
    "process_datasource_sync",
    "task_datasource_sync",
]
