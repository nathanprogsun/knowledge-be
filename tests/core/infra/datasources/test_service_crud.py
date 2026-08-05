"""Unit tests for ``DataSourceService`` CRUD.

Core services are tested with Protocol-based fakes (AGENTS.md §9): the
fakes mirror the repository contracts, the service projects storage rows
to ``DataSourceInfo`` via ``map_from_db``.

The invariants under test, in order of importance:

1. Credentials never appear in a projection, and an update body cannot
   overwrite them.
2. A cross-workspace id is indistinguishable from a miss (404, not 403).
3. The connector-validation gate fires on create/update only when
   credentials are actually configured.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.common.exception import NotFoundError, ValidationError
from src.core.infra.datasources.connector_base import (
    CONNECTOR_METADATA_REGISTRY,
    ConnectorRegistry,
    list_available_connectors,
)
from src.core.infra.datasources.service.datasource_service import (
    MAX_SYNC_LOG_LIMIT,
    DataSourceService,
)
from src.core.infra.datasources.types import (
    CONNECTOR_TYPE_RSS,
    DATA_SOURCE_STATUS_ACTIVE,
    DATA_SOURCE_STATUS_PAUSED,
    DataSourceInfo,
)
from src.core.system.audit_actions import AuditAction
from src.core.system.audit_service import AuditLogService
from src.db.models.datasource import DataSource, SyncLog
from tests.fakes.datasources import (
    FakeAuditRepo,
    FakeDataSourceRepo,
    FakeSyncLogRepo,
    StubConnector,
    unreachable_error,
)

TENANT_ID = 7
OTHER_TENANT_ID = 8
KB_ID = "kb-1"


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def ds_repo() -> FakeDataSourceRepo:
    return FakeDataSourceRepo()


@pytest.fixture
def sync_log_repo() -> FakeSyncLogRepo:
    return FakeSyncLogRepo()


@pytest.fixture
def audit_repo() -> FakeAuditRepo:
    return FakeAuditRepo()


@pytest.fixture
def connector() -> StubConnector:
    return StubConnector("notion")


@pytest.fixture
def registry(connector: StubConnector) -> ConnectorRegistry:
    reg = ConnectorRegistry()
    reg.register(connector)
    return reg


@pytest.fixture
def service(
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
    audit_repo: FakeAuditRepo,
    registry: ConnectorRegistry,
) -> DataSourceService:
    return DataSourceService(
        ds_repo=ds_repo,  # type: ignore[arg-type]
        sync_log_repo=sync_log_repo,  # type: ignore[arg-type]
        connector_registry=registry,
        audit_service=AuditLogService(audit_repo=audit_repo),  # type: ignore[arg-type]
    )


def _row(
    *,
    id: str = "ds-1",
    tenant_id: int = TENANT_ID,
    type: str = "notion",
    status: str = DATA_SOURCE_STATUS_ACTIVE,
    config: dict[str, object] | None = None,
) -> DataSource:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return DataSource(
        id=id,
        tenant_id=tenant_id,
        knowledge_base_id=KB_ID,
        name="my source",
        type=type,
        config=config,  # type: ignore[arg-type]
        status=status,
        created_at=now,
        updated_at=now,
    )


# ── create ───────────────────────────────────────────────────────────


async def test_create_persists_row_and_returns_projection(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
) -> None:
    info = await service.create(
        tenant_id=TENANT_ID,
        knowledge_base_id=KB_ID,
        name="notion sync",
        type="notion",
    )

    assert info.tenant_id == TENANT_ID
    assert info.knowledge_base_id == KB_ID
    assert info.status == DATA_SOURCE_STATUS_ACTIVE
    # Defaults mirror the SQL column defaults.
    assert info.sync_mode == "incremental"
    assert info.conflict_strategy == "overwrite"
    assert info.sync_deletions is True
    assert info.sync_log_retention_days == 30
    assert ds_repo.rows[info.id].name == "notion sync"


async def test_create_rejects_unknown_connector_type(service: DataSourceService) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        await service.create(
            tenant_id=TENANT_ID,
            knowledge_base_id=KB_ID,
            name="mystery",
            type="does_not_exist",
        )
    assert excinfo.value.code == "datasource.connector_not_found"


async def test_create_rejects_blank_name(service: DataSourceService) -> None:
    with pytest.raises(ValidationError) as excinfo:
        await service.create(
            tenant_id=TENANT_ID,
            knowledge_base_id=KB_ID,
            name="   ",
            type="notion",
        )
    assert excinfo.value.code == "datasource.name_required"


async def test_create_rejects_missing_knowledge_base_id(service: DataSourceService) -> None:
    with pytest.raises(ValidationError) as excinfo:
        await service.create(
            tenant_id=TENANT_ID,
            knowledge_base_id="",
            name="notion sync",
            type="notion",
        )
    assert excinfo.value.code == "datasource.knowledge_base_id_required"


async def test_create_validates_connection_when_credentials_present(
    service: DataSourceService,
    connector: StubConnector,
) -> None:
    await service.create(
        tenant_id=TENANT_ID,
        knowledge_base_id=KB_ID,
        name="notion sync",
        type="notion",
        config={"credentials": {"api_key": "secret"}},
    )
    assert len(connector.validate_calls) == 1
    assert connector.validate_calls[0].credentials == {"api_key": "secret"}


async def test_create_skips_validation_without_credentials(
    service: DataSourceService,
    connector: StubConnector,
) -> None:
    # A form saved before the credentials are entered must still create:
    # a validator with no token to present would always fail.
    await service.create(
        tenant_id=TENANT_ID,
        knowledge_base_id=KB_ID,
        name="notion sync",
        type="notion",
        config={"resource_ids": ["page-1"]},
    )
    assert connector.validate_calls == []


async def test_create_propagates_validation_failure_and_persists_nothing(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    connector: StubConnector,
) -> None:
    connector.validate_error = unreachable_error()

    with pytest.raises(Exception, match="upstream unreachable"):
        await service.create(
            tenant_id=TENANT_ID,
            knowledge_base_id=KB_ID,
            name="notion sync",
            type="notion",
            config={"credentials": {"api_key": "bad"}},
        )
    assert ds_repo.rows == {}


async def test_create_strips_non_secret_rss_credentials(
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
    audit_repo: FakeAuditRepo,
) -> None:
    registry = ConnectorRegistry()
    registry.register(StubConnector(CONNECTOR_TYPE_RSS))
    service = DataSourceService(
        ds_repo=ds_repo,  # type: ignore[arg-type]
        sync_log_repo=sync_log_repo,  # type: ignore[arg-type]
        connector_registry=registry,
        audit_service=AuditLogService(audit_repo=audit_repo),  # type: ignore[arg-type]
    )

    info = await service.create(
        tenant_id=TENANT_ID,
        knowledge_base_id=KB_ID,
        name="feed",
        type=CONNECTOR_TYPE_RSS,
        config={"credentials": {"feed_urls": "https://example.com/rss"}},
    )

    stored = ds_repo.rows[info.id].config
    assert stored is not None
    assert stored["credentials"] == {}
    # feed_urls is not a secret, so the source is NOT "credentialed".
    assert info.credentials_configured is False


async def test_create_emits_audit_row(
    service: DataSourceService,
    audit_repo: FakeAuditRepo,
) -> None:
    info = await service.create(
        tenant_id=TENANT_ID,
        knowledge_base_id=KB_ID,
        name="notion sync",
        type="notion",
        actor_user_id="user-9",
    )

    assert len(audit_repo.rows) == 1
    entry = audit_repo.rows[0]
    assert entry.action == AuditAction.DATASOURCE_CREATED
    assert entry.tenant_id == TENANT_ID
    assert entry.actor_user_id == "user-9"
    assert entry.target_type == "data_source"
    assert entry.target_id == info.id
    assert entry.scope_id == KB_ID


# ── credential redaction ─────────────────────────────────────────────


async def test_projection_never_exposes_credentials(
    service: DataSourceService,
) -> None:
    info = await service.create(
        tenant_id=TENANT_ID,
        knowledge_base_id=KB_ID,
        name="notion sync",
        type="notion",
        config={
            "credentials": {"api_key": "super-secret"},
            "resource_ids": ["page-1"],
            "settings": {"depth": 2},
        },
    )

    assert info.config is not None
    assert info.config.credentials == {}
    assert info.config.resource_ids == ["page-1"]
    assert info.config.settings == {"depth": 2}
    assert info.credentials_configured is True
    assert "super-secret" not in info.model_dump_json()


async def test_rss_feed_urls_surface_through_settings(
    ds_repo: FakeDataSourceRepo,
) -> None:
    # Legacy row: feed_urls still sits in the credential blob.
    row = _row(
        type=CONNECTOR_TYPE_RSS,
        config={"credentials": {"feed_urls": "https://example.com/rss"}},
    )
    info = DataSourceInfo.map_from_db(row)

    assert info.config is not None
    assert info.config.settings["feed_urls"] == "https://example.com/rss"
    assert info.config.credentials == {}


# ── get ──────────────────────────────────────────────────────────────


async def test_get_returns_row_with_sync_aggregates(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row
    ds_repo.items_synced[row.id] = 42
    now = datetime(2026, 2, 1, tzinfo=UTC)
    sync_log_repo.rows["log-1"] = SyncLog(
        id="log-1",
        data_source_id=row.id,
        tenant_id=TENANT_ID,
        status="success",
        started_at=now,
        created_at=now,
        updated_at=now,
    )

    info = await service.get(id=row.id, tenant_id=TENANT_ID)

    assert info.total_items_synced == 42
    assert info.latest_sync_log is not None
    assert info.latest_sync_log.id == "log-1"


async def test_get_missing_row_raises_not_found(service: DataSourceService) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        await service.get(id="nope", tenant_id=TENANT_ID)
    assert excinfo.value.code == "datasource.not_found"


async def test_get_cross_tenant_row_reads_as_not_found(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
) -> None:
    # 404 rather than 403 on purpose: a 403 would confirm the id exists.
    row = _row(tenant_id=OTHER_TENANT_ID)
    ds_repo.rows[row.id] = row

    with pytest.raises(NotFoundError):
        await service.get(id=row.id, tenant_id=TENANT_ID)


# ── list ─────────────────────────────────────────────────────────────


async def test_list_returns_only_own_tenant_rows(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
) -> None:
    mine = _row(id="ds-mine")
    theirs = _row(id="ds-theirs", tenant_id=OTHER_TENANT_ID)
    ds_repo.rows[mine.id] = mine
    ds_repo.rows[theirs.id] = theirs

    infos = await service.list_by_knowledge_base(
        knowledge_base_id=KB_ID,
        tenant_id=TENANT_ID,
    )

    assert [i.id for i in infos] == ["ds-mine"]


async def test_list_rejects_blank_kb_id(service: DataSourceService) -> None:
    with pytest.raises(ValidationError) as excinfo:
        await service.list_by_knowledge_base(knowledge_base_id="", tenant_id=TENANT_ID)
    assert excinfo.value.code == "datasource.knowledge_base_id_required"


# ── update ───────────────────────────────────────────────────────────


async def test_update_patches_only_supplied_fields(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row

    info = await service.update(id=row.id, tenant_id=TENANT_ID, name="renamed")

    assert info.name == "renamed"
    # Untouched fields keep their stored value.
    assert info.sync_mode == row.sync_mode
    assert info.conflict_strategy == row.conflict_strategy


async def test_update_preserves_stored_credentials_against_body(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
) -> None:
    row = _row(config={"credentials": {"api_key": "original"}, "settings": {"depth": 1}})
    ds_repo.rows[row.id] = row

    await service.update(
        id=row.id,
        tenant_id=TENANT_ID,
        config={"credentials": {"api_key": "attacker"}, "settings": {"depth": 9}},
    )

    stored = ds_repo.rows[row.id].config
    assert stored is not None
    # Body credentials are ignored; non-credential fields flow through.
    assert stored["credentials"] == {"api_key": "original"}
    assert stored["settings"] == {"depth": 9}


async def test_update_drops_credentials_when_none_were_stored(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
) -> None:
    row = _row(config={"settings": {"depth": 1}})
    ds_repo.rows[row.id] = row

    await service.update(
        id=row.id,
        tenant_id=TENANT_ID,
        config={"credentials": {"api_key": "attacker"}},
    )

    stored = ds_repo.rows[row.id].config
    assert stored is not None
    assert stored["credentials"] == {}


async def test_update_revalidates_when_config_changed_and_credentialed(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    connector: StubConnector,
) -> None:
    row = _row(config={"credentials": {"api_key": "k"}, "resource_ids": ["a"]})
    ds_repo.rows[row.id] = row

    await service.update(
        id=row.id,
        tenant_id=TENANT_ID,
        config={"resource_ids": ["a", "b"]},
    )

    assert len(connector.validate_calls) == 1


async def test_update_skips_revalidation_when_config_unchanged(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    connector: StubConnector,
) -> None:
    row = _row(
        config={
            "type": "notion",
            "credentials": {"api_key": "k"},
            "resource_ids": ["a"],
            "settings": {},
            "multimodal_enabled": False,
        }
    )
    ds_repo.rows[row.id] = row

    await service.update(
        id=row.id,
        tenant_id=TENANT_ID,
        config={"type": "notion", "resource_ids": ["a"], "settings": {}},
    )

    assert connector.validate_calls == []


async def test_update_cross_tenant_row_raises_not_found(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
) -> None:
    row = _row(tenant_id=OTHER_TENANT_ID)
    ds_repo.rows[row.id] = row

    with pytest.raises(NotFoundError):
        await service.update(id=row.id, tenant_id=TENANT_ID, name="hijack")


async def test_update_emits_audit_row(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    audit_repo: FakeAuditRepo,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row

    await service.update(id=row.id, tenant_id=TENANT_ID, name="renamed")

    assert audit_repo.rows[-1].action == AuditAction.DATASOURCE_UPDATED


# ── delete ───────────────────────────────────────────────────────────


async def test_delete_soft_deletes_and_cancels_running_syncs(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row
    now = datetime(2026, 2, 1, tzinfo=UTC)
    sync_log_repo.rows["log-1"] = SyncLog(
        id="log-1",
        data_source_id=row.id,
        tenant_id=TENANT_ID,
        status="running",
        started_at=now,
        created_at=now,
        updated_at=now,
    )

    await service.delete(id=row.id, tenant_id=TENANT_ID)

    assert ds_repo.rows[row.id].deleted_at is not None
    # A queued task retry must not report progress against a dead source.
    assert sync_log_repo.rows["log-1"].status == "canceled"
    assert await ds_repo.find_by_id_or_none(row.id) is None


async def test_delete_missing_row_raises_not_found(service: DataSourceService) -> None:
    with pytest.raises(NotFoundError):
        await service.delete(id="nope", tenant_id=TENANT_ID)


async def test_delete_emits_audit_row(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    audit_repo: FakeAuditRepo,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row

    await service.delete(id=row.id, tenant_id=TENANT_ID)

    assert audit_repo.rows[-1].action == AuditAction.DATASOURCE_DELETED


# ── pause / resume ───────────────────────────────────────────────────


async def test_pause_sets_paused_status(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    audit_repo: FakeAuditRepo,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row

    info = await service.pause(id=row.id, tenant_id=TENANT_ID)

    assert info.status == DATA_SOURCE_STATUS_PAUSED
    assert audit_repo.rows[-1].action == AuditAction.DATASOURCE_PAUSED


async def test_resume_clears_error_message(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
) -> None:
    row = _row(status="error").model_copy(update={"error_message": "auth failed"})
    ds_repo.rows[row.id] = row

    info = await service.resume(id=row.id, tenant_id=TENANT_ID)

    assert info.status == DATA_SOURCE_STATUS_ACTIVE
    assert info.error_message == ""


# ── sync logs ────────────────────────────────────────────────────────


async def test_list_sync_logs_returns_newest_first(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row
    for idx, day in enumerate((1, 5, 3), start=1):
        started = datetime(2026, 2, day, tzinfo=UTC)
        sync_log_repo.rows[f"log-{idx}"] = SyncLog(
            id=f"log-{idx}",
            data_source_id=row.id,
            tenant_id=TENANT_ID,
            status="success",
            started_at=started,
            created_at=started,
            updated_at=started,
        )

    infos = await service.list_sync_logs(id=row.id, tenant_id=TENANT_ID)

    assert [i.id for i in infos] == ["log-2", "log-3", "log-1"]


async def test_list_sync_logs_rejects_out_of_range_limit(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row

    with pytest.raises(ValidationError) as excinfo:
        await service.list_sync_logs(
            id=row.id,
            tenant_id=TENANT_ID,
            limit=MAX_SYNC_LOG_LIMIT + 1,
        )
    assert excinfo.value.code == "datasource.sync_log_limit_invalid"


async def test_list_sync_logs_rejects_negative_offset(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row

    with pytest.raises(ValidationError) as excinfo:
        await service.list_sync_logs(id=row.id, tenant_id=TENANT_ID, offset=-1)
    assert excinfo.value.code == "datasource.sync_log_offset_invalid"


async def test_get_sync_log_of_foreign_source_raises_not_found(
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
) -> None:
    row = _row(tenant_id=OTHER_TENANT_ID)
    ds_repo.rows[row.id] = row
    now = datetime(2026, 2, 1, tzinfo=UTC)
    sync_log_repo.rows["log-1"] = SyncLog(
        id="log-1",
        data_source_id=row.id,
        tenant_id=OTHER_TENANT_ID,
        status="success",
        started_at=now,
        created_at=now,
        updated_at=now,
    )

    with pytest.raises(NotFoundError):
        await service.get_sync_log(log_id="log-1", tenant_id=TENANT_ID)


async def test_get_missing_sync_log_raises_not_found(service: DataSourceService) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        await service.get_sync_log(log_id="nope", tenant_id=TENANT_ID)
    assert excinfo.value.code == "datasource.sync_log_not_found"


# ── connector registry / metadata ────────────────────────────────────


def test_registry_rejects_blank_connector_type() -> None:
    reg = ConnectorRegistry()
    with pytest.raises(ValidationError) as excinfo:
        reg.register(StubConnector(""))
    assert excinfo.value.code == "datasource.connector_type_empty"


def test_available_connectors_cover_all_upstream_types() -> None:
    metas = list_available_connectors()
    assert len(metas) == len(CONNECTOR_METADATA_REGISTRY) == 13


def test_available_connectors_sorted_by_priority() -> None:
    priorities = [m.priority for m in list_available_connectors()]
    assert priorities == sorted(priorities)
    assert priorities[0] == 0
