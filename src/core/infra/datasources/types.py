"""Internal DTOs + constants for the data-source domain.

Service-output projections. The HTTP wire shapes live in
``src/core/contracts/infra.py`` (frozen); the connector protocol —
``DataSourceConfig`` with credentials, ``Resource``, ``FetchedItem``,
``SyncCursor``, ``SyncResult``, ``SyncItemError`` — lives in
``src/common/datasource_protocol.py`` (there, not here, because concrete
connectors live in ``src/ai/`` and ``ai`` may not import ``core``). Those
names are re-exported below so the domain has one import surface.

What this module adds:

- status / sync-mode / conflict-strategy constants
  (``internal/types/datasource.go``)
- ``DataSourceInfo`` / ``SyncLogInfo`` — the ``map_from_db`` projections
  (AGENTS.md §9) that strip the credential map before a row can leave the
  service
- ``parse_config`` — the lenient ``DataSource.ParseConfig`` decode

Field and JSON names match ``internal/types/datasource.go`` exactly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self, cast

from pydantic import BaseModel, ConfigDict

from src.common.datasource_protocol import (
    CONNECTOR_TYPE_CONFLUENCE,
    CONNECTOR_TYPE_DINGTALK,
    CONNECTOR_TYPE_FEISHU,
    CONNECTOR_TYPE_GITHUB,
    CONNECTOR_TYPE_GOOGLE_DRIVE,
    CONNECTOR_TYPE_IMAP,
    CONNECTOR_TYPE_LARK,
    CONNECTOR_TYPE_NOTION,
    CONNECTOR_TYPE_ONEDRIVE,
    CONNECTOR_TYPE_RSS,
    CONNECTOR_TYPE_SLACK,
    CONNECTOR_TYPE_WEB_CRAWLER,
    CONNECTOR_TYPE_YUQUE,
    CREDENTIALS_FIELD,
    Connector,
    DataSourceConfig,
    FetchedItem,
    Resource,
    StreamHandler,
    StreamingConnector,
    SyncCursor,
    SyncItemError,
    SyncResult,
)
from src.common.json import JsonObject
from src.db.models.datasource import DataSource, SyncLog
from src.util.crypto import decrypt_stored_secret_lenient, encrypt_aesgcm, get_aes_key

# ── Sync modes ───────────────────────────────────────────────────────

SYNC_MODE_INCREMENTAL = "incremental"
SYNC_MODE_FULL = "full"

# ── Data-source status ───────────────────────────────────────────────

DATA_SOURCE_STATUS_ACTIVE = "active"
DATA_SOURCE_STATUS_PAUSED = "paused"
DATA_SOURCE_STATUS_ERROR = "error"
DATA_SOURCE_STATUS_DELETED = "deleted"

# ── Sync-log status ──────────────────────────────────────────────────

SYNC_LOG_STATUS_RUNNING = "running"
SYNC_LOG_STATUS_SUCCESS = "success"
SYNC_LOG_STATUS_PARTIAL = "partial"
SYNC_LOG_STATUS_FAILED = "failed"
SYNC_LOG_STATUS_CANCELED = "canceled"

# ── Conflict strategies ──────────────────────────────────────────────

CONFLICT_STRATEGY_OVERWRITE = "overwrite"
CONFLICT_STRATEGY_SKIP = "skip"

# ── Defaults (mirror the SQL column defaults) ────────────────────────

DEFAULT_SYNC_LOG_RETENTION_DAYS = 30

# ``ManualSync`` accepts these three; anything else is "not active".
# ``paused`` is allowed on purpose: a manual run is an explicit override
# of the schedule, not a resume.
MANUAL_SYNC_ALLOWED_STATUSES: frozenset[str] = frozenset(
    {
        DATA_SOURCE_STATUS_ACTIVE,
        DATA_SOURCE_STATUS_ERROR,
        DATA_SOURCE_STATUS_PAUSED,
    }
)


# ── Service-output projections ───────────────────────────────────────


class SyncLogInfo(BaseModel):
    """Wire-side projection of a ``sync_logs`` row."""

    model_config = ConfigDict(frozen=True)

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

    @classmethod
    def map_from_db(cls, db: SyncLog) -> Self:
        """Project a storage row onto the wire shape."""
        return cls.model_validate(db.model_dump())


class DataSourceInfo(BaseModel):
    """Wire-side projection of a ``data_sources`` row.

    ``config`` is the credential-free view: ``map_from_db`` keeps only
    ``type`` / ``resource_ids`` / ``settings``, matching
    ``dto.DataSourceConfigDTO``. ``credentials_configured`` reports
    whether a secret is stored, so the response can carry
    ``{"credentials": {"configured": ...}}`` without exposing any value.

    ``total_items_synced`` and ``latest_sync_log`` are not columns
    (Go: ``gorm:"-"``) — the service fills them per query.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: int
    knowledge_base_id: str
    name: str
    type: str
    config: DataSourceConfig | None = None
    sync_schedule: str = ""
    sync_mode: str = SYNC_MODE_INCREMENTAL
    status: str = DATA_SOURCE_STATUS_ACTIVE
    conflict_strategy: str = CONFLICT_STRATEGY_OVERWRITE
    sync_deletions: bool = True
    last_sync_at: datetime | None = None
    last_sync_cursor: JsonObject | None = None
    last_sync_result: JsonObject | None = None
    error_message: str = ""
    sync_log_retention_days: int = DEFAULT_SYNC_LOG_RETENTION_DAYS
    created_at: datetime
    updated_at: datetime
    total_items_synced: int = 0
    latest_sync_log: SyncLogInfo | None = None
    credentials_configured: bool = False

    @classmethod
    def map_from_db(
        cls,
        db: DataSource,
        *,
        total_items_synced: int = 0,
        latest_sync_log: SyncLog | None = None,
    ) -> Self:
        """Project a storage row onto the wire shape, credentials stripped."""
        parsed = parse_config(db.config)
        redacted: DataSourceConfig | None = None
        configured = False
        if parsed is not None:
            configured = parsed.has_configured_credentials(db.type)
            redacted = DataSourceConfig(
                type=parsed.type,
                resource_ids=parsed.resource_ids,
                settings=_enrich_rss_feed_urls(db.type, parsed),
            )
        record = db.model_dump(exclude={"config", "deleted_at"})
        record["config"] = redacted.model_dump() if redacted is not None else None
        record["total_items_synced"] = total_items_synced
        record["latest_sync_log"] = (
            SyncLogInfo.map_from_db(latest_sync_log).model_dump()
            if latest_sync_log is not None
            else None
        )
        record["credentials_configured"] = configured
        return cls.model_validate(record)


def parse_config(raw: JsonObject | None) -> DataSourceConfig | None:
    """Decode a stored ``config`` blob — ``DataSource.ParseConfig``.

    Returns ``None`` for an absent/empty blob so callers can distinguish
    "never configured" from "configured empty". Unknown keys are dropped
    rather than rejected: the column is written by other revisions too,
    and a strict decode would make a whole data source unreadable over
    one stray key.

    Credential strings are decrypted in place (Go ``ParseConfig``):
    legacy plaintext passes through, ``enc:v1:`` blobs are decrypted
    with ``SYSTEM_AES_KEY``, and a decrypt failure blanks the field so
    the row stays visible.
    """
    if not raw:
        return None
    known = set(DataSourceConfig.model_fields)
    parsed = DataSourceConfig.model_validate({k: v for k, v in raw.items() if k in known})
    if parsed.credentials:
        decrypted: JsonObject = {}
        for key, value in parsed.credentials.items():
            if isinstance(value, str) and value:
                plain, ok = decrypt_stored_secret_lenient(value)
                decrypted[key] = plain if ok else ""
            else:
                decrypted[key] = value
        parsed = parsed.model_copy(update={"credentials": decrypted})
    return parsed


def encrypt_config_credentials(config: JsonObject) -> JsonObject:
    """Encrypt every credential string in a config blob before storage.

    Mirrors Go ``DataSourceConfig.ToJSON``: only string values inside
    ``credentials`` are encrypted (non-strings pass through), and only
    when ``SYSTEM_AES_KEY`` is configured. Operates on a copy — the
    caller's dict is never mutated.
    """
    if not config or not isinstance(config.get(CREDENTIALS_FIELD), dict):
        return config
    key = get_aes_key()
    if key is None:
        return config
    credentials = cast("JsonObject", config[CREDENTIALS_FIELD])
    encrypted: JsonObject = {}
    for k, v in credentials.items():
        if isinstance(v, str) and v:
            encrypted[k] = encrypt_aesgcm(v, key)
        else:
            encrypted[k] = v
    return {**config, CREDENTIALS_FIELD: encrypted}


def _enrich_rss_feed_urls(connector_type: str, parsed: DataSourceConfig) -> JsonObject:
    """Surface RSS ``feed_urls`` through ``settings`` for responses.

    Feed URLs are not secrets but may still sit in the credential blob on
    rows created before they moved to ``settings``
    (``enrichRSSFeedURLsInSettings``).
    """
    if connector_type != CONNECTOR_TYPE_RSS:
        return parsed.settings
    existing = parsed.settings.get("feed_urls")
    if isinstance(existing, str) and existing.strip() != "":
        return parsed.settings
    raw = parsed.credentials.get("feed_urls")
    if not isinstance(raw, str) or raw.strip() == "":
        return parsed.settings
    return {**parsed.settings, "feed_urls": raw}


__all__ = [
    "CONFLICT_STRATEGY_OVERWRITE",
    "CONFLICT_STRATEGY_SKIP",
    "CONNECTOR_TYPE_CONFLUENCE",
    "CONNECTOR_TYPE_DINGTALK",
    "CONNECTOR_TYPE_FEISHU",
    "CONNECTOR_TYPE_GITHUB",
    "CONNECTOR_TYPE_GOOGLE_DRIVE",
    "CONNECTOR_TYPE_IMAP",
    "CONNECTOR_TYPE_LARK",
    "CONNECTOR_TYPE_NOTION",
    "CONNECTOR_TYPE_ONEDRIVE",
    "CONNECTOR_TYPE_RSS",
    "CONNECTOR_TYPE_SLACK",
    "CONNECTOR_TYPE_WEB_CRAWLER",
    "CONNECTOR_TYPE_YUQUE",
    "CREDENTIALS_FIELD",
    "DATA_SOURCE_STATUS_ACTIVE",
    "DATA_SOURCE_STATUS_DELETED",
    "DATA_SOURCE_STATUS_ERROR",
    "DATA_SOURCE_STATUS_PAUSED",
    "DEFAULT_SYNC_LOG_RETENTION_DAYS",
    "MANUAL_SYNC_ALLOWED_STATUSES",
    "SYNC_LOG_STATUS_CANCELED",
    "SYNC_LOG_STATUS_FAILED",
    "SYNC_LOG_STATUS_PARTIAL",
    "SYNC_LOG_STATUS_RUNNING",
    "SYNC_LOG_STATUS_SUCCESS",
    "SYNC_MODE_FULL",
    "SYNC_MODE_INCREMENTAL",
    "Connector",
    "DataSourceConfig",
    "DataSourceInfo",
    "FetchedItem",
    "Resource",
    "StreamHandler",
    "StreamingConnector",
    "SyncCursor",
    "SyncItemError",
    "SyncLogInfo",
    "SyncResult",
    "parse_config",
]
