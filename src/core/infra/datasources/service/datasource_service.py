"""Data-source CRUD service — ``DataSourceService`` (create/read/update/delete).

Maps the CRUD half of
``internal/application/service/datasource_service.go``:
``CreateDataSource`` / ``GetDataSource`` / ``ListDataSources`` /
``UpdateDataSource`` / ``DeleteDataSource``, plus the sync-log readers
(``GetSyncLogs`` / ``GetSyncLog``) and the pause/resume status toggles.

Connectivity (``ValidateConnection`` / ``ValidateCredentials``), resource
listing (``ListAvailableResources`` / ``ResolveResourceAncestors``) and
the sync engine (``ManualSync`` / ``ProcessSync``) live in sibling
modules — ``connectivity.py``, ``resource_listing.py``, ``sync.py`` —
and are mixed into this class so the public surface stays one service
object, matching the Go type, without a 1300-line file.

Deliberate scope deviations from Go, in order of visibility:

1. **Knowledge-base ownership checks.** Go's ``CreateDataSource``
   resolves the KB through ``kbService`` and rejects a cross-tenant KB.
   The KB domain is not migrated yet, so this service validates
   ``tenant_id`` consistency on its own rows and leaves the KB existence
   check to the caller. Tenant isolation itself is enforced here.
2. **Cron scheduler.** Go registers/removes an entry on a
   ``datasource.Scheduler`` for every mutation. There is no scheduler
   process yet; ``sync_schedule`` is persisted verbatim so the scheduler
   can pick the rows up.
3. **Credential encryption.** Go AES-GCM encrypts each credential string
   in ``DataSourceConfig.ToJSON``. The shared crypto helper is not
   migrated yet, so the blob is stored as given. The *redaction* boundary
   (credentials never leave the service) is fully in place via
   ``DataSourceInfo.map_from_db``.
4. **Audit rows.** Go emits KB-activity audit entries per mutation. The
   audit repository is wired in and used; the KB-scope fields it needs
   (``scope_type``/``scope_id``) are set to the data source's KB id.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from src.common.exception import NotFoundError, ValidationError
from src.common.json import JsonObject
from src.core.infra.datasources.connectivity import ConnectivityMixin
from src.core.infra.datasources.connector_base import ConnectorRegistry
from src.core.infra.datasources.resource_listing import ResourceListingMixin
from src.core.infra.datasources.sync import SyncMixin
from src.core.infra.datasources.types import (
    CONFLICT_STRATEGY_OVERWRITE,
    DATA_SOURCE_STATUS_ACTIVE,
    DATA_SOURCE_STATUS_PAUSED,
    SYNC_MODE_INCREMENTAL,
    DataSourceInfo,
    SyncLogInfo,
    parse_config,
)
from src.core.system.audit_actions import AuditAction, AuditOutcome
from src.core.system.audit_service import AuditLogService
from src.db.dao.datasource_repository import DataSourceRepository, SyncLogRepository
from src.db.models.datasource import DataSource
from src.db.models.system.audit_log import AuditLog

# Sync-log page size bounds for ``list_sync_logs`` (Go: default 10,
# capped by the handler's ``maxListPageSize``).
DEFAULT_SYNC_LOG_LIMIT = 10
MAX_SYNC_LOG_LIMIT = 100


class DataSourceService(ConnectivityMixin, ResourceListingMixin, SyncMixin):
    """External data-source configuration + sync orchestration.

    Request-scoped: the repositories hold the per-request ``AsyncSession``,
    so a mutation and its audit row land in one transaction. Built only by
    ``src.core.infra.datasources.factory.build_datasource_service``.
    """

    def __init__(
        self,
        *,
        ds_repo: DataSourceRepository,
        sync_log_repo: SyncLogRepository,
        connector_registry: ConnectorRegistry,
        audit_service: AuditLogService,
    ) -> None:
        self._ds_repo = ds_repo
        self._sync_log_repo = sync_log_repo
        self._connector_registry = connector_registry
        self._audit_service = audit_service

    # ── Create ────────────────────────────────────────────────────────

    async def create(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        name: str,
        type: str,
        config: JsonObject | None = None,
        sync_schedule: str | None = None,
        sync_mode: str | None = None,
        conflict_strategy: str | None = None,
        sync_deletions: bool | None = None,
        sync_log_retention_days: int | None = None,
        actor_user_id: str = "",
    ) -> DataSourceInfo:
        """Create a data source after validating type + connectivity.

        The connector type must be registered (``NotFoundError``
        otherwise) and, when the config carries credentials, the live
        connection is validated before anything is persisted — Go's
        ``validateDataSourceConfig`` gate. Non-secret values are stripped
        out of the credential map first.
        """
        if not name.strip():
            raise ValidationError(
                code="datasource.name_required",
                message="data source name is required",
            )
        if not knowledge_base_id:
            raise ValidationError(
                code="datasource.knowledge_base_id_required",
                message="knowledge_base_id is required",
            )
        # Raises when the connector type is unknown.
        self._connector_registry.get(type)

        stored_config = self._normalize_config(config, type)
        await self._validate_config_if_credentialed(stored_config, type)

        now = datetime.now(UTC)
        row = DataSource(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            name=name,
            type=type,
            config=stored_config,
            sync_schedule=sync_schedule or "",
            sync_mode=sync_mode or SYNC_MODE_INCREMENTAL,
            status=DATA_SOURCE_STATUS_ACTIVE,
            conflict_strategy=conflict_strategy or CONFLICT_STRATEGY_OVERWRITE,
            sync_deletions=True if sync_deletions is None else sync_deletions,
            sync_log_retention_days=(
                30 if sync_log_retention_days is None else sync_log_retention_days
            ),
            created_at=now,
            updated_at=now,
        )
        persisted = await self._ds_repo.create(row)
        await self._audit(
            row=persisted,
            action=AuditAction.DATASOURCE_CREATED,
            outcome=AuditOutcome.SUCCESS,
            actor_user_id=actor_user_id,
            details={"name": persisted.name, "type": persisted.type},
        )
        return DataSourceInfo.map_from_db(persisted)

    # ── Read ──────────────────────────────────────────────────────────

    async def get(self, *, id: str, tenant_id: int) -> DataSourceInfo:
        """Return one data source, enriched with sync aggregates.

        Raises ``NotFoundError`` when the row is absent or belongs to
        another workspace — a cross-tenant read is indistinguishable from
        a miss by design, so the id space is not enumerable.
        """
        row = await self._require_owned(id=id, tenant_id=tenant_id)
        latest = await self._sync_log_repo.find_latest(row.id)
        total = await self._ds_repo.count_items_synced(row.id)
        return DataSourceInfo.map_from_db(
            row,
            total_items_synced=total,
            latest_sync_log=latest,
        )

    async def list_by_knowledge_base(
        self,
        *,
        knowledge_base_id: str,
        tenant_id: int,
    ) -> list[DataSourceInfo]:
        """Return the workspace's data sources for one knowledge base.

        Each entry carries its latest sync log, matching Go's
        ``ListDataSources`` fan-out.
        """
        if not knowledge_base_id:
            raise ValidationError(
                code="datasource.knowledge_base_id_required",
                message="kb_id is required",
            )
        rows = await self._ds_repo.find_by_knowledge_base(knowledge_base_id)
        infos: list[DataSourceInfo] = []
        for row in rows:
            if row.tenant_id != tenant_id:
                continue
            latest = await self._sync_log_repo.find_latest(row.id)
            total = await self._ds_repo.count_items_synced(row.id)
            infos.append(
                DataSourceInfo.map_from_db(
                    row,
                    total_items_synced=total,
                    latest_sync_log=latest,
                )
            )
        return infos

    # ── Update ────────────────────────────────────────────────────────

    async def update(
        self,
        *,
        id: str,
        tenant_id: int,
        name: str | None = None,
        config: JsonObject | None = None,
        sync_schedule: str | None = None,
        sync_mode: str | None = None,
        conflict_strategy: str | None = None,
        sync_deletions: bool | None = None,
        sync_log_retention_days: int | None = None,
        actor_user_id: str = "",
    ) -> DataSourceInfo:
        """Patch mutable fields; ``None`` leaves a field untouched.

        ``knowledge_base_id`` and ``tenant_id`` are immutable (Go rejects
        a change outright). Credentials NEVER flow through this path:
        whatever the incoming ``config`` says, the stored credential map
        is force-preserved and only ``type`` / ``resource_ids`` /
        ``settings`` are taken from the body — mirroring Go's merge, where
        credential writes belong to the ``/credentials`` subresource.
        """
        existing = await self._require_owned(id=id, tenant_id=tenant_id)

        merged_config = existing.config
        config_changed = False
        if config is not None:
            merged_config = self._merge_config_preserving_credentials(
                incoming=config,
                existing=existing.config,
                connector_type=existing.type,
            )
            config_changed = merged_config != existing.config

        if config_changed:
            await self._validate_config_if_credentialed(merged_config, existing.type)

        updated = existing.model_copy(
            update={
                "name": existing.name if name is None else name,
                "config": merged_config,
                "sync_schedule": (
                    existing.sync_schedule if sync_schedule is None else sync_schedule
                ),
                "sync_mode": existing.sync_mode if sync_mode is None else sync_mode,
                "conflict_strategy": (
                    existing.conflict_strategy if conflict_strategy is None else conflict_strategy
                ),
                "sync_deletions": (
                    existing.sync_deletions if sync_deletions is None else sync_deletions
                ),
                "sync_log_retention_days": (
                    existing.sync_log_retention_days
                    if sync_log_retention_days is None
                    else sync_log_retention_days
                ),
                "updated_at": datetime.now(UTC),
            }
        )
        persisted = await self._ds_repo.update(updated)
        await self._audit(
            row=persisted,
            action=AuditAction.DATASOURCE_UPDATED,
            outcome=AuditOutcome.SUCCESS,
            actor_user_id=actor_user_id,
            details={
                "name": persisted.name,
                "type": persisted.type,
                "changed_fields": ["settings"],
            },
        )
        return DataSourceInfo.map_from_db(persisted)

    # ── Delete ────────────────────────────────────────────────────────

    async def delete(self, *, id: str, tenant_id: int, actor_user_id: str = "") -> None:
        """Soft-delete a data source and cancel its in-flight syncs.

        Cancelling the running sync logs is what stops a queued task retry
        from reporting progress against a dead source (Go:
        ``CancelPendingByDataSource``).
        """
        existing = await self._require_owned(id=id, tenant_id=tenant_id)
        now = datetime.now(UTC)
        await self._ds_repo.soft_delete(id=existing.id, now=now)
        await self._sync_log_repo.cancel_pending_by_data_source(
            data_source_id=existing.id,
            now=now,
        )
        await self._audit(
            row=existing,
            action=AuditAction.DATASOURCE_DELETED,
            outcome=AuditOutcome.SUCCESS,
            actor_user_id=actor_user_id,
            details={"name": existing.name, "type": existing.type},
        )

    # ── Pause / resume ────────────────────────────────────────────────

    async def pause(self, *, id: str, tenant_id: int, actor_user_id: str = "") -> DataSourceInfo:
        """Set status to ``paused`` so scheduled syncs stop firing."""
        return await self._set_status(
            id=id,
            tenant_id=tenant_id,
            status=DATA_SOURCE_STATUS_PAUSED,
            action=AuditAction.DATASOURCE_PAUSED,
            actor_user_id=actor_user_id,
        )

    async def resume(self, *, id: str, tenant_id: int, actor_user_id: str = "") -> DataSourceInfo:
        """Set status back to ``active`` and clear any error message."""
        return await self._set_status(
            id=id,
            tenant_id=tenant_id,
            status=DATA_SOURCE_STATUS_ACTIVE,
            action=AuditAction.DATASOURCE_RESUMED,
            actor_user_id=actor_user_id,
        )

    # ── Sync logs ─────────────────────────────────────────────────────

    async def list_sync_logs(
        self,
        *,
        id: str,
        tenant_id: int,
        limit: int = DEFAULT_SYNC_LOG_LIMIT,
        offset: int = 0,
    ) -> list[SyncLogInfo]:
        """Return a page of the source's sync history, newest first."""
        if limit < 1 or limit > MAX_SYNC_LOG_LIMIT:
            raise ValidationError(
                code="datasource.sync_log_limit_invalid",
                message=f"limit must be between 1 and {MAX_SYNC_LOG_LIMIT}",
            )
        if offset < 0:
            raise ValidationError(
                code="datasource.sync_log_offset_invalid",
                message="offset must not be negative",
            )
        row = await self._require_owned(id=id, tenant_id=tenant_id)
        logs = await self._sync_log_repo.find_by_data_source(
            row.id,
            limit=limit,
            offset=offset,
        )
        return [SyncLogInfo.map_from_db(log) for log in logs]

    async def get_sync_log(self, *, log_id: str, tenant_id: int) -> SyncLogInfo:
        """Return one sync log, after checking the owning source is ours.

        Go looks the log up first, then re-checks ownership of its data
        source; same order here so a foreign log id reads as not-found.
        """
        log = await self._sync_log_repo.find_by_id_or_none(log_id)
        if log is None:
            raise NotFoundError(
                code="datasource.sync_log_not_found",
                message="sync log not found",
            )
        await self._require_owned(id=log.data_source_id, tenant_id=tenant_id)
        return SyncLogInfo.map_from_db(log)

    # ── Internal helpers ──────────────────────────────────────────────

    async def _require_owned(self, *, id: str, tenant_id: int) -> DataSource:
        """Fetch a live row and assert it belongs to ``tenant_id``."""
        row = await self._ds_repo.find_by_id_or_none(id)
        if row is None or row.tenant_id != tenant_id:
            raise NotFoundError(
                code="datasource.not_found",
                message="data source not found",
            )
        return row

    async def _set_status(
        self,
        *,
        id: str,
        tenant_id: int,
        status: str,
        action: str,
        actor_user_id: str,
    ) -> DataSourceInfo:
        """Persist a status transition and emit its audit row."""
        existing = await self._require_owned(id=id, tenant_id=tenant_id)
        updated = existing.model_copy(
            update={
                "status": status,
                "error_message": (
                    "" if status == DATA_SOURCE_STATUS_ACTIVE else existing.error_message
                ),
                "updated_at": datetime.now(UTC),
            }
        )
        persisted = await self._ds_repo.update(updated)
        await self._audit(
            row=persisted,
            action=action,
            outcome=AuditOutcome.SUCCESS,
            actor_user_id=actor_user_id,
            details={"name": persisted.name, "type": persisted.type},
        )
        return DataSourceInfo.map_from_db(persisted)

    @staticmethod
    def _normalize_config(config: JsonObject | None, connector_type: str) -> JsonObject | None:
        """Strip non-secret values out of the incoming credential map."""
        parsed = parse_config(config)
        if parsed is None:
            return None
        return parsed.strip_non_secret_credentials(connector_type).model_dump()

    @staticmethod
    def _merge_config_preserving_credentials(
        *,
        incoming: JsonObject,
        existing: JsonObject | None,
        connector_type: str,
    ) -> JsonObject | None:
        """Take non-credential fields from ``incoming``, credentials from
        ``existing``.

        Credentials in an update body are ignored by contract; dropping
        them here (rather than rejecting the request) matches Go, which
        logs a deprecation warning and proceeds.
        """
        incoming_cfg = parse_config(incoming)
        if incoming_cfg is None:
            return existing
        existing_cfg = parse_config(existing)
        merged = incoming_cfg.model_copy(
            update={
                "credentials": existing_cfg.credentials if existing_cfg is not None else {},
            }
        )
        return merged.strip_non_secret_credentials(connector_type).model_dump()

    async def _audit(
        self,
        *,
        row: DataSource,
        action: str,
        outcome: str,
        actor_user_id: str,
        details: JsonObject,
    ) -> None:
        """Emit one KB-scoped audit row for a data-source mutation."""
        await self._audit_service.log(
            AuditLog(
                id=0,
                tenant_id=row.tenant_id,
                actor_user_id=actor_user_id,
                action=action,
                scope_type="knowledge_base",
                scope_id=row.knowledge_base_id,
                target_type="data_source",
                target_id=row.id,
                outcome=outcome,
                details=details,
                created_at=datetime.now(UTC),
            )
        )


__all__ = ["DEFAULT_SYNC_LOG_LIMIT", "MAX_SYNC_LOG_LIMIT", "DataSourceService"]
