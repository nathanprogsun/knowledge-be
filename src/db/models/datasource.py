"""Storage rows for the `data_sources` and `sync_logs` tables.

Mirrors ``internal/types/datasource.go::DataSource`` and
``internal/types/datasource.go::SyncLog``. The SQL shape is
``migrations/versioned/000029_datasource_tables.up.sql``.

Column names (and therefore the JSON names produced by the wire
projections in ``src/core/infra/datasources/types.py``) match the Go
structs exactly.

Two Go fields are NOT columns and are absent here:

- ``TotalItemsSynced`` (``gorm:"-"``) — aggregated per query.
- ``LatestSyncLog`` (``gorm:"-"``) — joined per query.

``config`` holds the AES-encrypted connector configuration blob. Nothing
in ``db`` decrypts it: the service parses / redacts on the way out (see
``DataSourceInfo``), which is why the raw column type is a plain JSON
object.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from src.common.json import JsonObject
from src.common.table_model import TableModel


class DataSource(TableModel):
    """One row of the ``data_sources`` table."""

    table: ClassVar[str] = "data_sources"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = (
        "config",
        "last_sync_cursor",
        "last_sync_result",
    )
    # ``id`` is a caller-assigned UUID (Go: BeforeCreate hook), not a
    # server default — it must take part in the INSERT column list.
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    id: str
    tenant_id: int
    knowledge_base_id: str
    name: str
    type: str
    config: JsonObject | None = None
    sync_schedule: str = ""
    sync_mode: str = "incremental"
    status: str = "active"
    conflict_strategy: str = "overwrite"
    sync_deletions: bool = True
    last_sync_at: datetime | None = None
    last_sync_cursor: JsonObject | None = None
    last_sync_result: JsonObject | None = None
    error_message: str = ""
    sync_log_retention_days: int = 30
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


class SyncLog(TableModel):
    """One row of the ``sync_logs`` table."""

    table: ClassVar[str] = "sync_logs"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ("result",)
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    id: str
    data_source_id: str
    tenant_id: int
    status: str
    started_at: datetime
    finished_at: datetime | None = None
    items_total: int = 0
    items_created: int = 0
    items_updated: int = 0
    items_deleted: int = 0
    items_skipped: int = 0
    items_failed: int = 0
    error_message: str = ""
    result: JsonObject | None = None
    created_at: datetime
    updated_at: datetime


__all__ = ["DataSource", "SyncLog"]
