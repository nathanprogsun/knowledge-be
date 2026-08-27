"""Unit tests for the ``datasource_sync`` worker task.

Covers the worker-side surface: the registered handler is the expected
function, the payload parses cleanly into the contract model, the
handler delegates to the core sync engine with the parsed arguments
(through an injected session-bound ``DataSourceService``), and the
un-injected seam raises so a miswired worker fails loudly. The core
dispatch is exercised through a mocked core datasource service, so no
real database or connector is needed.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from arq.connections import ArqRedis
from pydantic import ValidationError

from src.core.infra.datasources.service.datasource_service import DataSourceService
from src.core.infra.datasources.types import SyncLogInfo
from src.workers.base import WorkerContext
from src.workers.registry import get_task
from src.workers.tasks import datasource_sync as datasource_sync_module
from src.workers.tasks.datasource_sync import (
    DatasourceSyncPayload,
    process_datasource_sync,
    task_datasource_sync,
)

NOW = datetime(2026, 3, 1, tzinfo=UTC)
FINISHED = datetime(2026, 3, 1, 1, 0, 0, tzinfo=UTC)


def make_ctx() -> WorkerContext:
    """Build a context dict matching what ARQ passes to tasks."""
    return WorkerContext(
        redis=cast(ArqRedis, None),
        job_id="job-1",
        job_try=1,
        enqueue_time=datetime.now(UTC),
        score=0,
    )


@pytest.fixture
def ctx() -> WorkerContext:
    """Worker context for ad-hoc task invocations."""
    return make_ctx()


@pytest.fixture
def valid_payload() -> dict[str, object]:
    """A representative JSON payload for the datasource-sync task."""
    return {
        "data_source_id": "ds-1",
        "tenant_id": 42,
        "sync_log_id": "log-1",
        "force_full": True,
        "max_items": 5,
        "trigger": "schedule",
    }


@pytest.fixture
def sync_log_info() -> SyncLogInfo:
    """A terminal sync-log record as returned by the core engine."""
    return SyncLogInfo(
        id="log-1",
        data_source_id="ds-1",
        tenant_id=42,
        status="success",
        started_at=NOW,
        finished_at=FINISHED,
        items_total=3,
        items_created=2,
        items_updated=1,
        items_deleted=0,
        items_skipped=0,
        items_failed=0,
        error_message="",
        result={"created": 2, "updated": 1},
        created_at=NOW,
        updated_at=FINISHED,
    )


@pytest.fixture
def mock_service(sync_log_info: SyncLogInfo) -> AsyncMock:
    """A mocked core datasource service bound to the sync seam."""
    service = AsyncMock(spec=DataSourceService)
    service.process_sync.return_value = sync_log_info
    return service


# ── Registration ────────────────────────────────────────────────────


def test_datasource_sync_registered_under_task_name() -> None:
    """The handler is registered under the upstream task name."""
    assert get_task("datasource_sync") is task_datasource_sync


def test_datasource_sync_unknown_name_returns_none() -> None:
    """A typo in the registered name must not silently resolve."""
    assert get_task("datasource_syn") is None


# ── Payload contract ────────────────────────────────────────────────


def test_payload_parses_full() -> None:
    """A complete payload round-trips through the contract model."""
    payload = DatasourceSyncPayload.model_validate(
        {
            "data_source_id": "ds-1",
            "tenant_id": 7,
            "sync_log_id": "log-9",
            "force_full": True,
            "max_items": 5,
            "trigger": "manual",
        }
    )
    assert payload.data_source_id == "ds-1"
    assert payload.tenant_id == 7
    assert payload.sync_log_id == "log-9"
    assert payload.force_full is True
    assert payload.max_items == 5
    assert payload.trigger == "manual"


def test_payload_defaults_optional_fields() -> None:
    """The optional fields default to their no-op values."""
    payload = DatasourceSyncPayload.model_validate({"data_source_id": "ds-1", "tenant_id": 1})
    assert payload.sync_log_id == ""
    assert payload.force_full is False
    assert payload.max_items == 0
    assert payload.trigger == ""


def test_payload_ignores_upstream_tracing_fields() -> None:
    """Tracing / initiator fields are accepted but not modelled."""
    payload = DatasourceSyncPayload.model_validate(
        {
            "data_source_id": "ds-1",
            "tenant_id": 1,
            "lf_trace_id": "trace-1",
            "lf_traceparent": "00-abc-def-01",
            "initiator": {"user_id": "u-1", "role": "admin"},
        }
    )
    assert payload.data_source_id == "ds-1"
    assert payload.tenant_id == 1


def test_payload_rejects_missing_data_source_id() -> None:
    """The data-source id is mandatory."""
    with pytest.raises(ValidationError):
        DatasourceSyncPayload.model_validate({"tenant_id": 1})


def test_payload_rejects_missing_tenant_id() -> None:
    """The tenant id is mandatory."""
    with pytest.raises(ValidationError):
        DatasourceSyncPayload.model_validate({"data_source_id": "ds-1"})


def test_payload_is_frozen() -> None:
    """The payload model is immutable; mutations must raise."""
    payload = DatasourceSyncPayload.model_validate({"data_source_id": "ds-1", "tenant_id": 1})
    with pytest.raises(ValidationError):
        payload.data_source_id = "tampered"


# ── Worker dispatch ─────────────────────────────────────────────────


async def test_task_datasource_sync_delegates_to_core_service(
    ctx: WorkerContext,
    valid_payload: dict[str, object],
    mock_service: AsyncMock,
) -> None:
    """The handler parses and forwards the payload to the core engine."""
    result = await task_datasource_sync(
        ctx,
        service=cast(DataSourceService, mock_service),
        **valid_payload,  # type: ignore[arg-type]
    )

    mock_service.process_sync.assert_awaited_once_with(
        data_source_id="ds-1",
        sync_log_id="log-1",
        force_full=True,
        max_items=5,
    )
    assert result == {
        "id": "log-1",
        "data_source_id": "ds-1",
        "tenant_id": 42,
        "status": "success",
        "started_at": "2026-03-01T00:00:00Z",
        "finished_at": "2026-03-01T01:00:00Z",
        "items_total": 3,
        "items_created": 2,
        "items_updated": 1,
        "items_deleted": 0,
        "items_skipped": 0,
        "items_failed": 0,
        "error_message": "",
        "result": {"created": 2, "updated": 1},
        "created_at": "2026-03-01T00:00:00Z",
        "updated_at": "2026-03-01T01:00:00Z",
    }


async def test_task_datasource_sync_uses_defaults_for_optional_fields(
    ctx: WorkerContext,
    mock_service: AsyncMock,
) -> None:
    """Omitted ``sync_log_id`` / ``force_full`` / ``max_items`` fall back."""
    await task_datasource_sync(
        ctx,
        service=cast(DataSourceService, mock_service),
        data_source_id="ds-1",  # type: ignore[arg-type]
        tenant_id=1,  # type: ignore[arg-type]
    )

    mock_service.process_sync.assert_awaited_once_with(
        data_source_id="ds-1",
        sync_log_id="",
        force_full=False,
        max_items=0,
    )


async def test_task_datasource_sync_rejects_invalid_payload(
    ctx: WorkerContext,
    mock_service: AsyncMock,
) -> None:
    """A payload missing required fields surfaces as ``ValidationError``."""
    with pytest.raises(ValidationError):
        await task_datasource_sync(
            ctx,
            service=cast(DataSourceService, mock_service),
            tenant_id=1,  # type: ignore[arg-type]
        )


async def test_task_datasource_sync_propagates_core_errors(
    ctx: WorkerContext,
    valid_payload: dict[str, object],
    mock_service: AsyncMock,
) -> None:
    """Errors raised by the core engine surface to the worker caller."""
    mock_service.process_sync.side_effect = RuntimeError("sync exploded")
    with pytest.raises(RuntimeError, match="sync exploded"):
        await task_datasource_sync(
            ctx,
            service=cast(DataSourceService, mock_service),
            **valid_payload,  # type: ignore[arg-type]
        )


# ── Core seam without injected service ──────────────────────────────


async def test_process_datasource_sync_raises_without_service() -> None:
    """An uninjected seam raises so a miswired sync is never silent."""
    with pytest.raises(NotImplementedError, match="DataSourceService"):
        await process_datasource_sync(
            data_source_id="ds-1",
            tenant_id=1,
            sync_log_id="log-1",
        )


async def test_process_datasource_sync_delegates_to_injected_service(
    mock_service: AsyncMock,
    sync_log_info: SyncLogInfo,
) -> None:
    """An injected service runs the sync and its record is serialised."""
    result = await process_datasource_sync(
        data_source_id="ds-1",
        tenant_id=42,
        sync_log_id="log-1",
        force_full=True,
        max_items=5,
        service=cast(DataSourceService, mock_service),
    )

    mock_service.process_sync.assert_awaited_once_with(
        data_source_id="ds-1",
        sync_log_id="log-1",
        force_full=True,
        max_items=5,
    )
    assert result["status"] == sync_log_info.status
    assert result["items_created"] == sync_log_info.items_created
    assert result["data_source_id"] == "ds-1"


# ── Re-registration guard ──────────────────────────────────────────


@pytest.fixture(autouse=True)
def _restore_registry() -> Iterator[None]:
    """Snapshot the registry around each test to avoid cross-test pollution.

    The ``register_task`` decorator mutates a module-level dict. Tests
    that import the module leave the registration in place, but a
    future test that re-registers under the same name would silently
    overwrite it. The fixture is defensive — a no-op today.
    """
    import src.workers.tasks  # noqa: F401 — pre-warm so the snapshot captures registered handlers.
    from src.workers import registry as registry_module

    snapshot = dict(registry_module.all_tasks())
    # Drop any test-only handler that was registered by ``@register_task``
    # decorators at import time of this module (e.g. ``test_base``'s
    # ``test_task``). The canonical handler set is what downstream
    # invariant tests assert against.
    baseline = {name: handler for name, handler in snapshot.items() if not name.startswith("test_")}
    yield
    # Restore the baseline after the test: drop anything newly added
    # (incl. handlers the test itself registered) and re-assert the
    # canonical handlers from the snapshot.
    current = registry_module.all_tasks()
    for name in list(current.keys()):
        if name not in baseline:
            current.pop(name, None)
    for name, handler in baseline.items():
        current[name] = handler


# ── Patchability guard ──────────────────────────────────────────────


async def test_task_datasource_sync_patchable_core_seam(
    ctx: WorkerContext,
    valid_payload: dict[str, object],
) -> None:
    """The core seam is patchable for callers wiring the worker later."""
    with patch.object(
        datasource_sync_module,
        "process_datasource_sync",
        new_callable=AsyncMock,
        return_value={"status": "dispatched"},
    ) as mock:
        result = await task_datasource_sync(ctx, **valid_payload)  # type: ignore[arg-type]

    mock.assert_awaited_once_with(
        data_source_id="ds-1",
        tenant_id=42,
        sync_log_id="log-1",
        force_full=True,
        max_items=5,
        service=None,
    )
    assert result == {"status": "dispatched"}
