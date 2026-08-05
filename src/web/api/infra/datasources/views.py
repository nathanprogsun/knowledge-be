"""Wire-shape conversion for the data-source endpoints.

Projects the service DTOs (``DataSourceInfo`` / ``SyncLogInfo`` /
``ConnectorMetadata`` / ``Resource``) onto the frozen contracts in
``src/core/contracts/infra.py``.

The credential boundary is enforced here as well as in the service: the
contract's ``config`` type has no ``credentials`` field, and
``credentials`` on the response is the presence-only
``{"credentials": {"configured": bool}}`` map — one logical field,
because connector credentials are a per-connector atomic set and a
per-field view would advertise half-configured states that cannot
authenticate (``dto.NewDataSourceResponse``).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject
from src.core.contracts.infra import (
    CredentialFieldMetadata,
    DataSource,
    DataSourceConfig,
    DataSourceConnectorMetadata,
    SyncLog,
)
from src.core.infra.datasources.connector_base import ConnectorMetadata
from src.core.infra.datasources.types import (
    CREDENTIALS_FIELD,
    DataSourceInfo,
    Resource,
    SyncLogInfo,
)


class ResourceResponse(BaseModel):
    """Wire shape for one listed external resource — ``types.Resource``.

    Not in the frozen contracts: it is connector-protocol output rather
    than a persisted entity, so it is shaped here alongside its endpoint.
    """

    model_config = ConfigDict(frozen=True)

    external_id: str
    name: str
    type: str
    description: str = ""
    url: str = ""
    modified_at: datetime | None = None
    parent_id: str = ""
    has_children: bool = False
    metadata: JsonObject = Field(default_factory=dict)


class ResolveResourceAncestorsResponse(BaseModel):
    """``{"ancestors": [...]}`` — the resource-ancestors response."""

    model_config = ConfigDict(frozen=True)

    ancestors: list[str] = Field(default_factory=list)


class ConnectionStatusResponse(BaseModel):
    """``{"status": "connected"}`` — validate / pause / resume ack.

    Go answers these three with a bare status string rather than the
    entity, so the shape is shared.
    """

    model_config = ConfigDict(frozen=True)

    status: str


def sync_log_to_contract(info: SyncLogInfo) -> SyncLog:
    """Project a sync-log DTO onto the frozen wire contract."""
    return SyncLog(
        id=info.id,
        data_source_id=info.data_source_id,
        tenant_id=info.tenant_id,
        status=info.status,
        started_at=info.started_at,
        finished_at=info.finished_at,
        items_total=info.items_total,
        items_created=info.items_created,
        items_updated=info.items_updated,
        items_deleted=info.items_deleted,
        items_skipped=info.items_skipped,
        items_failed=info.items_failed,
        error_message=info.error_message,
        result=info.result,
        created_at=info.created_at,
        updated_at=info.updated_at,
    )


def datasource_to_contract(info: DataSourceInfo) -> DataSource:
    """Project a data-source DTO onto the frozen wire contract.

    ``credentials`` carries presence only. The contract ``config`` type
    has no credential field at all, so secrets cannot leak through this
    conversion even if a caller hands in an unredacted DTO.
    """
    config = (
        DataSourceConfig(
            type=info.config.type,
            resource_ids=info.config.resource_ids,
            settings=info.config.settings,
        )
        if info.config is not None
        else None
    )
    return DataSource(
        id=info.id,
        tenant_id=info.tenant_id,
        knowledge_base_id=info.knowledge_base_id,
        name=info.name,
        type=info.type,
        config=config,
        sync_schedule=info.sync_schedule,
        sync_mode=info.sync_mode,
        status=info.status,
        conflict_strategy=info.conflict_strategy,
        sync_deletions=info.sync_deletions,
        last_sync_at=info.last_sync_at,
        last_sync_cursor=info.last_sync_cursor,
        last_sync_result=info.last_sync_result,
        error_message=info.error_message,
        sync_log_retention_days=info.sync_log_retention_days,
        created_at=info.created_at,
        updated_at=info.updated_at,
        total_items_synced=info.total_items_synced,
        latest_sync_log=(
            sync_log_to_contract(info.latest_sync_log) if info.latest_sync_log is not None else None
        ),
        credentials={
            CREDENTIALS_FIELD: CredentialFieldMetadata(configured=info.credentials_configured)
        },
    )


def connector_metadata_to_contract(meta: ConnectorMetadata) -> DataSourceConnectorMetadata:
    """Project a connector descriptor onto the frozen wire contract."""
    return DataSourceConnectorMetadata(
        type=meta.type,
        name=meta.name,
        description=meta.description,
        icon=meta.icon,
        priority=meta.priority,
        auth_type=meta.auth_type,
        capabilities=meta.capabilities,
    )


def resource_to_response(resource: Resource) -> ResourceResponse:
    """Project a connector resource onto its response shape."""
    return ResourceResponse(
        external_id=resource.external_id,
        name=resource.name,
        type=resource.type,
        description=resource.description,
        url=resource.url,
        modified_at=resource.modified_at,
        parent_id=resource.parent_id,
        has_children=resource.has_children,
        metadata=resource.metadata,
    )


__all__ = [
    "ConnectionStatusResponse",
    "ResolveResourceAncestorsResponse",
    "ResourceResponse",
    "connector_metadata_to_contract",
    "datasource_to_contract",
    "resource_to_response",
    "sync_log_to_contract",
]
