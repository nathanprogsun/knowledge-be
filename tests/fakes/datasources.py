"""In-memory doubles for the data-source repositories + connectors.

Mirror the real repository contracts: finders return storage rows and the
service projects them via ``map_from_db``. Shared by the CRUD, sync, and
web-view suites so all three exercise one behaviour model.
"""

from __future__ import annotations

from datetime import datetime

from src.common.datasource_protocol import (
    Connector,
    DataSourceConfig,
    FetchedItem,
    Resource,
    SyncCursor,
)
from src.common.exception import ExternalServiceError, ValidationError
from src.db.models.datasource import DataSource, SyncLog
from src.db.models.system.audit_log import AuditLog


class FakeDataSourceRepo:
    """In-memory ``DataSourceRepository`` replacement."""

    def __init__(self) -> None:
        self.rows: dict[str, DataSource] = {}
        self.items_synced: dict[str, int] = {}

    async def create(self, row: DataSource) -> DataSource:
        self.rows[row.id] = row
        return row

    async def find_by_id_or_none(self, id: str) -> DataSource | None:
        row = self.rows.get(id)
        if row is None or row.deleted_at is not None:
            return None
        return row

    async def find_by_knowledge_base(self, knowledge_base_id: str) -> list[DataSource]:
        rows = [
            r
            for r in self.rows.values()
            if r.knowledge_base_id == knowledge_base_id and r.deleted_at is None
        ]
        return sorted(rows, key=lambda r: r.created_at, reverse=True)

    async def update(self, row: DataSource) -> DataSource:
        existing = self.rows.get(row.id)
        if existing is None:
            raise ValidationError(code="db.not_found", message="row missing")
        # Immutable columns are preserved exactly as the real repo does.
        persisted = row.model_copy(
            update={
                "tenant_id": existing.tenant_id,
                "knowledge_base_id": existing.knowledge_base_id,
                "created_at": existing.created_at,
            }
        )
        self.rows[row.id] = persisted
        return persisted

    async def soft_delete(self, *, id: str, now: datetime) -> bool:
        existing = self.rows.get(id)
        if existing is None or existing.deleted_at is not None:
            return False
        self.rows[id] = existing.model_copy(update={"deleted_at": now, "updated_at": now})
        return True

    async def count_items_synced(self, data_source_id: str) -> int:
        return self.items_synced.get(data_source_id, 0)


class FakeSyncLogRepo:
    """In-memory ``SyncLogRepository`` replacement."""

    def __init__(self) -> None:
        self.rows: dict[str, SyncLog] = {}

    async def create(self, row: SyncLog) -> SyncLog:
        self.rows[row.id] = row
        return row

    async def find_by_id_or_none(self, id: str) -> SyncLog | None:
        return self.rows.get(id)

    async def find_by_data_source(
        self,
        data_source_id: str,
        *,
        limit: int,
        offset: int,
    ) -> list[SyncLog]:
        rows = [r for r in self.rows.values() if r.data_source_id == data_source_id]
        rows = sorted(rows, key=lambda r: r.started_at, reverse=True)
        return rows[offset : offset + limit]

    async def find_latest(self, data_source_id: str) -> SyncLog | None:
        rows = [r for r in self.rows.values() if r.data_source_id == data_source_id]
        if not rows:
            return None
        return max(rows, key=lambda r: r.started_at)

    async def update(self, row: SyncLog) -> SyncLog:
        self.rows[row.id] = row
        return row

    async def cancel_pending_by_data_source(
        self,
        *,
        data_source_id: str,
        now: datetime,
    ) -> int:
        count = 0
        for log_id, row in list(self.rows.items()):
            if row.data_source_id == data_source_id and row.status == "running":
                self.rows[log_id] = row.model_copy(
                    update={"status": "canceled", "finished_at": now, "updated_at": now}
                )
                count += 1
        return count


class FakeAuditRepo:
    """In-memory ``AuditLogRepository`` replacement (create path only)."""

    def __init__(self) -> None:
        self.rows: list[AuditLog] = []
        self._next_id = 1

    async def create(self, entry: AuditLog) -> AuditLog:
        persisted = entry.model_copy(update={"id": self._next_id})
        self._next_id += 1
        self.rows.append(persisted)
        return persisted


class StubConnector(Connector):
    """Scriptable connector for service tests.

    ``validate_error`` makes ``validate`` raise, so the create/update
    validation gate and the ``ValidateConnection`` status side effects can
    be driven without any network.
    """

    def __init__(
        self,
        connector_type: str = "notion",
        *,
        validate_error: Exception | None = None,
        resources: list[Resource] | None = None,
        ancestors: list[str] | None = None,
        items: list[FetchedItem] | None = None,
        incremental_items: list[FetchedItem] | None = None,
        next_cursor: SyncCursor | None = None,
        fetch_error: Exception | None = None,
    ) -> None:
        self._type = connector_type
        self.validate_error = validate_error
        self.resources = resources or []
        self.ancestors = ancestors or []
        self.items = items or []
        self.incremental_items = incremental_items
        self.next_cursor = next_cursor
        self.fetch_error = fetch_error
        self.validate_calls: list[DataSourceConfig] = []
        self.list_resources_calls: list[str] = []

    @property
    def type(self) -> str:
        return self._type

    async def validate(self, config: DataSourceConfig) -> None:
        self.validate_calls.append(config)
        if self.validate_error is not None:
            raise self.validate_error

    async def list_resources(
        self,
        config: DataSourceConfig,
        parent_id: str = "",
    ) -> list[Resource]:
        self.list_resources_calls.append(parent_id)
        return self.resources

    async def resolve_resource_ancestors(
        self,
        config: DataSourceConfig,
        resource_ids: list[str],
    ) -> list[str]:
        return self.ancestors

    async def fetch_all(
        self,
        config: DataSourceConfig,
        resource_ids: list[str],
    ) -> list[FetchedItem]:
        if self.fetch_error is not None:
            raise self.fetch_error
        return self.items

    async def fetch_incremental(
        self,
        config: DataSourceConfig,
        cursor: SyncCursor | None,
    ) -> tuple[list[FetchedItem], SyncCursor | None]:
        if self.fetch_error is not None:
            raise self.fetch_error
        items = self.incremental_items if self.incremental_items is not None else self.items
        return items, self.next_cursor


class RecordingIngestor:
    """``ItemIngestor`` double that records calls and scripts outcomes.

    ``updates`` names the external ids that should report "replaced an
    existing item"; ``failures`` maps an external id to the error raised
    for it, so partial-failure tallies can be driven precisely.
    """

    def __init__(
        self,
        *,
        updates: set[str] | None = None,
        failures: dict[str, Exception] | None = None,
        deletable: set[str] | None = None,
    ) -> None:
        self.updates = updates or set()
        self.failures = failures or {}
        self.deletable = deletable
        self.ingested: list[str] = []
        self.deleted: list[str] = []

    async def ingest(self, *, data_source: DataSource, item: FetchedItem) -> bool:
        failure = self.failures.get(item.external_id)
        if failure is not None:
            raise failure
        self.ingested.append(item.external_id)
        return item.external_id in self.updates

    async def delete(self, *, data_source: DataSource, external_id: str) -> bool:
        self.deleted.append(external_id)
        if self.deletable is None:
            return True
        return external_id in self.deletable


def unreachable_error(message: str = "upstream unreachable") -> ExternalServiceError:
    """Build the error a connector raises when the remote API is down."""
    return ExternalServiceError(code="datasource.unreachable", message=message)


__all__ = [
    "FakeAuditRepo",
    "FakeDataSourceRepo",
    "FakeSyncLogRepo",
    "RecordingIngestor",
    "StubConnector",
    "unreachable_error",
]
