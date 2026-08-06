"""Protocol doubles for the external-datasource integration.

These are not repository fakes (there are no longer any of those in
the suite). They implement the ``Connector`` and ``ItemIngestor``
protocols that the sync service consumes, so tests can script
remote-service behaviour without HTTP.

Lives under ``tests/util/`` so the ``tests/unit/fakes/`` directory can
be deleted entirely.
"""

from __future__ import annotations

from src.common.datasource_protocol import (
    Connector,
    DataSourceConfig,
    FetchedItem,
    Resource,
    SyncCursor,
)
from src.common.exception import ExternalServiceError
from src.db.models.datasource import DataSource


class StubConnector(Connector):
    """Scriptable connector for service tests.

    ``validate_error`` makes ``validate`` raise, so the create/update
    validation gate and the ``ValidateConnection`` status side effects
    can be driven without any network.
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
    "RecordingIngestor",
    "StubConnector",
    "unreachable_error",
]
