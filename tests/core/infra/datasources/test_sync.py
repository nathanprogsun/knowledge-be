"""Unit tests for the data-source sync engine + connectivity.

Covers the three modules mixed into ``DataSourceService``:

- ``sync.py``            — ``manual_sync`` / ``process_sync`` / ``fail_sync``
- ``connectivity.py``    — ``validate_connection`` / ``validate_credentials``
- ``resource_listing.py``— ``list_available_resources`` /
  ``resolve_resource_ancestors``

The behaviour that matters most is the terminal-status state machine: a
run with zero failures is ``success``, all-failed is ``failed``, and
anything between is ``partial`` — because ``failed`` is what flips the
source itself into an error state and stops the schedule.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.common.exception import ExternalServiceError, NotFoundError, ValidationError
from src.core.infra.datasources.connector_base import ConnectorRegistry
from src.core.infra.datasources.service.datasource_service import DataSourceService
from src.core.infra.datasources.sync import MAX_SYNC_ERROR_SAMPLES
from src.core.infra.datasources.types import (
    DATA_SOURCE_STATUS_ACTIVE,
    DATA_SOURCE_STATUS_ERROR,
    DATA_SOURCE_STATUS_PAUSED,
    SYNC_LOG_STATUS_FAILED,
    SYNC_LOG_STATUS_PARTIAL,
    SYNC_LOG_STATUS_RUNNING,
    SYNC_LOG_STATUS_SUCCESS,
    SYNC_MODE_FULL,
    FetchedItem,
    Resource,
    SyncCursor,
)
from src.core.system.audit_service import AuditLogService
from src.db.models.datasource import DataSource, SyncLog
from tests.unit.fakes.datasources import (
    FakeAuditRepo,
    FakeDataSourceRepo,
    FakeSyncLogRepo,
    RecordingIngestor,
    StubConnector,
    unreachable_error,
)

TENANT_ID = 7
KB_ID = "kb-1"
NOW = datetime(2026, 3, 1, tzinfo=UTC)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def ds_repo() -> FakeDataSourceRepo:
    return FakeDataSourceRepo()


@pytest.fixture
def sync_log_repo() -> FakeSyncLogRepo:
    return FakeSyncLogRepo()


@pytest.fixture
def connector() -> StubConnector:
    return StubConnector("notion")


@pytest.fixture
def service(
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
    connector: StubConnector,
) -> DataSourceService:
    registry = ConnectorRegistry()
    registry.register(connector)
    return DataSourceService(
        ds_repo=ds_repo,  # type: ignore[arg-type]
        sync_log_repo=sync_log_repo,  # type: ignore[arg-type]
        connector_registry=registry,
        audit_service=AuditLogService(audit_repo=FakeAuditRepo()),  # type: ignore[arg-type]
    )


def _row(
    *,
    id: str = "ds-1",
    status: str = DATA_SOURCE_STATUS_ACTIVE,
    sync_mode: str = "incremental",
    sync_deletions: bool = True,
    config: dict[str, object] | None = None,
    last_sync_cursor: dict[str, object] | None = None,
) -> DataSource:
    return DataSource(
        id=id,
        tenant_id=TENANT_ID,
        knowledge_base_id=KB_ID,
        name="my source",
        type="notion",
        config=config if config is not None else {"resource_ids": ["r-1"]},  # type: ignore[arg-type]
        status=status,
        sync_mode=sync_mode,
        sync_deletions=sync_deletions,
        last_sync_cursor=last_sync_cursor,  # type: ignore[arg-type]
        created_at=NOW,
        updated_at=NOW,
    )


def _log(*, id: str = "log-1", data_source_id: str = "ds-1") -> SyncLog:
    return SyncLog(
        id=id,
        data_source_id=data_source_id,
        tenant_id=TENANT_ID,
        status=SYNC_LOG_STATUS_RUNNING,
        started_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )


def _item(external_id: str, *, is_deleted: bool = False, title: str = "") -> FetchedItem:
    return FetchedItem(
        external_id=external_id,
        title=title or external_id,
        is_deleted=is_deleted,
    )


# ── manual_sync ──────────────────────────────────────────────────────


async def test_manual_sync_opens_running_log(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row

    info = await service.manual_sync(id=row.id, tenant_id=TENANT_ID)

    assert info.status == SYNC_LOG_STATUS_RUNNING
    assert info.data_source_id == row.id
    assert info.finished_at is None
    assert sync_log_repo.rows[info.id].status == SYNC_LOG_STATUS_RUNNING


async def test_manual_sync_allowed_on_paused_source(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
) -> None:
    # A manual run is an explicit override of the schedule, not a resume.
    row = _row(status=DATA_SOURCE_STATUS_PAUSED)
    ds_repo.rows[row.id] = row

    info = await service.manual_sync(id=row.id, tenant_id=TENANT_ID)

    assert info.status == SYNC_LOG_STATUS_RUNNING
    assert ds_repo.rows[row.id].status == DATA_SOURCE_STATUS_PAUSED


async def test_manual_sync_allowed_on_errored_source(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
) -> None:
    row = _row(status=DATA_SOURCE_STATUS_ERROR)
    ds_repo.rows[row.id] = row

    info = await service.manual_sync(id=row.id, tenant_id=TENANT_ID)

    assert info.status == SYNC_LOG_STATUS_RUNNING


async def test_manual_sync_rejects_unsyncable_status(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
) -> None:
    row = _row(status="deleted")
    ds_repo.rows[row.id] = row

    with pytest.raises(ValidationError) as excinfo:
        await service.manual_sync(id=row.id, tenant_id=TENANT_ID)
    assert excinfo.value.code == "datasource.not_active"


async def test_manual_sync_on_foreign_source_raises_not_found(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row

    with pytest.raises(NotFoundError):
        await service.manual_sync(id=row.id, tenant_id=999)


# ── fail_sync ────────────────────────────────────────────────────────


async def test_fail_sync_closes_log_and_errors_source(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row
    sync_log_repo.rows["log-1"] = _log()

    info = await service.fail_sync(
        sync_log_id="log-1",
        data_source=row,
        message="could not dispatch",
    )

    assert info.status == SYNC_LOG_STATUS_FAILED
    assert info.finished_at is not None
    assert info.error_message == "could not dispatch"
    assert ds_repo.rows[row.id].status == DATA_SOURCE_STATUS_ERROR


async def test_fail_sync_keeps_paused_status(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
) -> None:
    # Pausing is a user decision a failed dispatch must not undo.
    row = _row(status=DATA_SOURCE_STATUS_PAUSED)
    ds_repo.rows[row.id] = row
    sync_log_repo.rows["log-1"] = _log()

    await service.fail_sync(sync_log_id="log-1", data_source=row, message="boom")

    assert ds_repo.rows[row.id].status == DATA_SOURCE_STATUS_PAUSED


# ── process_sync: terminal status machine ────────────────────────────


async def test_process_sync_success_tallies_created_and_updated(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
    connector: StubConnector,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row
    sync_log_repo.rows["log-1"] = _log()
    connector.items = [_item("a"), _item("b"), _item("c")]
    service._ingestor = RecordingIngestor(updates={"b"})

    info = await service.process_sync(data_source_id=row.id, sync_log_id="log-1")

    assert info.status == SYNC_LOG_STATUS_SUCCESS
    assert info.items_total == 3
    assert info.items_created == 2
    assert info.items_updated == 1
    assert info.items_failed == 0
    assert info.finished_at is not None


async def test_process_sync_partial_when_some_items_fail(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
    connector: StubConnector,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row
    sync_log_repo.rows["log-1"] = _log()
    connector.items = [_item("a"), _item("b")]
    service._ingestor = RecordingIngestor(failures={"b": unreachable_error("parse failed")})

    info = await service.process_sync(data_source_id=row.id, sync_log_id="log-1")

    assert info.status == SYNC_LOG_STATUS_PARTIAL
    assert info.items_created == 1
    assert info.items_failed == 1
    # A partial run must not flip the source into an error state.
    assert ds_repo.rows[row.id].status == DATA_SOURCE_STATUS_ACTIVE


async def test_process_sync_failed_when_every_item_fails(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
    connector: StubConnector,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row
    sync_log_repo.rows["log-1"] = _log()
    connector.items = [_item("a"), _item("b")]
    service._ingestor = RecordingIngestor(
        failures={
            "a": unreachable_error("nope"),
            "b": unreachable_error("nope"),
        }
    )

    info = await service.process_sync(data_source_id=row.id, sync_log_id="log-1")

    assert info.status == SYNC_LOG_STATUS_FAILED
    assert info.items_failed == 2
    assert ds_repo.rows[row.id].status == DATA_SOURCE_STATUS_ERROR
    assert ds_repo.rows[row.id].error_message != ""


async def test_process_sync_empty_run_is_success(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row
    sync_log_repo.rows["log-1"] = _log()

    info = await service.process_sync(data_source_id=row.id, sync_log_id="log-1")

    assert info.status == SYNC_LOG_STATUS_SUCCESS
    assert info.items_total == 0


async def test_process_sync_caps_error_samples(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
    connector: StubConnector,
) -> None:
    # Uncapped, a feed whose every item fails writes a huge JSONB blob.
    count = MAX_SYNC_ERROR_SAMPLES + 5
    ids = [f"item-{i}" for i in range(count)]
    row = _row()
    ds_repo.rows[row.id] = row
    sync_log_repo.rows["log-1"] = _log()
    connector.items = [_item(i) for i in ids]
    service._ingestor = RecordingIngestor(failures={i: unreachable_error("nope") for i in ids})

    info = await service.process_sync(data_source_id=row.id, sync_log_id="log-1")

    assert info.items_failed == count
    assert info.result is not None
    errors = info.result["errors"]
    assert isinstance(errors, list)
    assert len(errors) == MAX_SYNC_ERROR_SAMPLES


# ── process_sync: deletions ──────────────────────────────────────────


async def test_process_sync_applies_deletions_when_enabled(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
    connector: StubConnector,
) -> None:
    row = _row(sync_deletions=True)
    ds_repo.rows[row.id] = row
    sync_log_repo.rows["log-1"] = _log()
    connector.items = [_item("gone", is_deleted=True)]
    ingestor = RecordingIngestor()
    service._ingestor = ingestor

    info = await service.process_sync(data_source_id=row.id, sync_log_id="log-1")

    assert info.items_deleted == 1
    assert ingestor.deleted == ["gone"]


async def test_process_sync_skips_deletions_when_disabled(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
    connector: StubConnector,
) -> None:
    # Opting out means upstream removals must not delete local knowledge.
    row = _row(sync_deletions=False)
    ds_repo.rows[row.id] = row
    sync_log_repo.rows["log-1"] = _log()
    connector.items = [_item("gone", is_deleted=True)]
    ingestor = RecordingIngestor()
    service._ingestor = ingestor

    info = await service.process_sync(data_source_id=row.id, sync_log_id="log-1")

    assert info.items_deleted == 0
    assert info.items_skipped == 1
    assert ingestor.deleted == []


# ── process_sync: incremental vs full ────────────────────────────────


async def test_process_sync_uses_incremental_when_cursor_present(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
    connector: StubConnector,
) -> None:
    row = _row(last_sync_cursor={"connector_cursor": {"page": 2}})
    ds_repo.rows[row.id] = row
    sync_log_repo.rows["log-1"] = _log()
    connector.incremental_items = [_item("changed")]
    connector.next_cursor = SyncCursor(connector_cursor={"page": 3})

    info = await service.process_sync(data_source_id=row.id, sync_log_id="log-1")

    assert info.items_total == 1
    stored_cursor = ds_repo.rows[row.id].last_sync_cursor
    assert stored_cursor is not None
    assert stored_cursor["connector_cursor"] == {"page": 3}


async def test_process_sync_force_full_ignores_cursor(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
    connector: StubConnector,
) -> None:
    row = _row(last_sync_cursor={"connector_cursor": {"page": 2}})
    ds_repo.rows[row.id] = row
    sync_log_repo.rows["log-1"] = _log()
    connector.items = [_item("full-1"), _item("full-2")]
    connector.incremental_items = [_item("changed")]

    info = await service.process_sync(
        data_source_id=row.id,
        sync_log_id="log-1",
        force_full=True,
    )

    assert info.items_total == 2


async def test_process_sync_full_mode_ignores_cursor(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
    connector: StubConnector,
) -> None:
    row = _row(sync_mode=SYNC_MODE_FULL, last_sync_cursor={"connector_cursor": {"page": 2}})
    ds_repo.rows[row.id] = row
    sync_log_repo.rows["log-1"] = _log()
    connector.items = [_item("full-1")]
    connector.incremental_items = [_item("a"), _item("b")]

    info = await service.process_sync(data_source_id=row.id, sync_log_id="log-1")

    assert info.items_total == 1


async def test_process_sync_respects_max_items(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
    connector: StubConnector,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row
    sync_log_repo.rows["log-1"] = _log()
    connector.items = [_item(f"i-{i}") for i in range(10)]

    info = await service.process_sync(
        data_source_id=row.id,
        sync_log_id="log-1",
        max_items=3,
    )

    assert info.items_total == 3


async def test_process_sync_advances_last_sync_at(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
    connector: StubConnector,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row
    sync_log_repo.rows["log-1"] = _log()
    connector.items = [_item("a")]

    await service.process_sync(data_source_id=row.id, sync_log_id="log-1")

    assert ds_repo.rows[row.id].last_sync_at is not None
    assert ds_repo.rows[row.id].last_sync_result is not None


async def test_process_sync_keeps_paused_status_after_manual_run(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
    connector: StubConnector,
) -> None:
    row = _row(status=DATA_SOURCE_STATUS_PAUSED)
    ds_repo.rows[row.id] = row
    sync_log_repo.rows["log-1"] = _log()
    connector.items = [_item("a")]

    await service.process_sync(data_source_id=row.id, sync_log_id="log-1")

    assert ds_repo.rows[row.id].status == DATA_SOURCE_STATUS_PAUSED


async def test_process_sync_propagates_fetch_failure(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
    connector: StubConnector,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row
    sync_log_repo.rows["log-1"] = _log()
    connector.fetch_error = unreachable_error()

    with pytest.raises(ExternalServiceError):
        await service.process_sync(data_source_id=row.id, sync_log_id="log-1")


async def test_process_sync_rejects_missing_source(service: DataSourceService) -> None:
    with pytest.raises(ValidationError) as excinfo:
        await service.process_sync(data_source_id="nope", sync_log_id="log-1")
    assert excinfo.value.code == "datasource.not_found"


async def test_process_sync_rejects_missing_log(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row

    with pytest.raises(ValidationError) as excinfo:
        await service.process_sync(data_source_id=row.id, sync_log_id="nope")
    assert excinfo.value.code == "datasource.sync_log_not_found"


async def test_process_sync_rejects_unconfigured_source(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
) -> None:
    row = _row(config={})
    ds_repo.rows[row.id] = row
    sync_log_repo.rows["log-1"] = _log()

    with pytest.raises(ValidationError) as excinfo:
        await service.process_sync(data_source_id=row.id, sync_log_id="log-1")
    assert excinfo.value.code == "datasource.invalid_config"


async def test_process_sync_without_ingestor_counts_skipped(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
    connector: StubConnector,
) -> None:
    # The knowledge stack is a later stage; items are counted, not lost.
    row = _row()
    ds_repo.rows[row.id] = row
    sync_log_repo.rows["log-1"] = _log()
    connector.items = [_item("a"), _item("b")]

    info = await service.process_sync(data_source_id=row.id, sync_log_id="log-1")

    assert info.items_skipped == 2
    assert info.status == SYNC_LOG_STATUS_SUCCESS


# ── validate_connection ──────────────────────────────────────────────


async def test_validate_connection_clears_prior_error(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
) -> None:
    row = _row(status=DATA_SOURCE_STATUS_ERROR).model_copy(update={"error_message": "auth failed"})
    ds_repo.rows[row.id] = row

    info = await service.validate_connection(id=row.id, tenant_id=TENANT_ID)

    assert info.status == DATA_SOURCE_STATUS_ACTIVE
    assert info.error_message == ""


async def test_validate_connection_records_failure_then_reraises(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    connector: StubConnector,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row
    connector.validate_error = unreachable_error("token expired")

    with pytest.raises(ExternalServiceError):
        await service.validate_connection(id=row.id, tenant_id=TENANT_ID)

    # Persisted before the raise, so the list view shows the problem.
    assert ds_repo.rows[row.id].status == DATA_SOURCE_STATUS_ERROR
    assert ds_repo.rows[row.id].error_message == "token expired"


async def test_validate_connection_leaves_healthy_source_untouched(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row
    before = ds_repo.rows[row.id].updated_at

    await service.validate_connection(id=row.id, tenant_id=TENANT_ID)

    assert ds_repo.rows[row.id].updated_at == before


async def test_validate_connection_rejects_unconfigured_source(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
) -> None:
    row = _row(config={})
    ds_repo.rows[row.id] = row

    with pytest.raises(ValidationError) as excinfo:
        await service.validate_connection(id=row.id, tenant_id=TENANT_ID)
    assert excinfo.value.code == "datasource.invalid_config"


# ── validate_credentials (nothing persisted) ─────────────────────────


async def test_validate_credentials_persists_nothing(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    connector: StubConnector,
) -> None:
    await service.validate_credentials(type="notion", credentials={"api_key": "k"})

    assert ds_repo.rows == {}
    assert connector.validate_calls[0].credentials == {"api_key": "k"}


async def test_validate_credentials_rejects_unknown_type(service: DataSourceService) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        await service.validate_credentials(type="nope", credentials={})
    assert excinfo.value.code == "datasource.connector_not_found"


async def test_validate_credentials_propagates_connector_error(
    service: DataSourceService,
    connector: StubConnector,
) -> None:
    connector.validate_error = unreachable_error()

    with pytest.raises(ExternalServiceError):
        await service.validate_credentials(type="notion", credentials={"api_key": "bad"})


# ── resource listing ─────────────────────────────────────────────────


async def test_list_available_resources_passes_parent_id(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    connector: StubConnector,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row
    connector.resources = [
        Resource(external_id="child-1", name="Child", type="page", parent_id="root-1")
    ]

    resources = await service.list_available_resources(
        id=row.id,
        tenant_id=TENANT_ID,
        parent_id="root-1",
    )

    assert [r.external_id for r in resources] == ["child-1"]
    assert connector.list_resources_calls == ["root-1"]


async def test_list_available_resources_on_foreign_source_raises_not_found(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row

    with pytest.raises(NotFoundError):
        await service.list_available_resources(id=row.id, tenant_id=999)


async def test_resolve_resource_ancestors_returns_connector_result(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    connector: StubConnector,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row
    connector.ancestors = ["root-1", "mid-2"]

    ancestors = await service.resolve_resource_ancestors(
        id=row.id,
        tenant_id=TENANT_ID,
        resource_ids=["leaf-3"],
    )

    assert ancestors == ["root-1", "mid-2"]


async def test_resolve_resource_ancestors_short_circuits_on_empty_request(
    service: DataSourceService,
) -> None:
    # The picker calls this on every edit-form open, selection or not; an
    # empty request must not touch the repo or the connector.
    ancestors = await service.resolve_resource_ancestors(
        id="does-not-exist",
        tenant_id=TENANT_ID,
        resource_ids=[],
    )

    assert ancestors == []
