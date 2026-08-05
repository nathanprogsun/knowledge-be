"""Data-source HTTP endpoints — external connector configuration + sync.

Maps ``internal/handler/datasource.go``, registered in Go by
``RegisterDataSourceRoutes`` (``internal/router/routes_infra.go``).

Data sources hold external-service credentials and trigger syncs that
mutate knowledge-base content workspace-wide, so the permission split
follows upstream exactly: reads are Viewer+, everything else Admin+.

============================================  ========
Route                                          Role
============================================  ========
``GET    /datasources/types``                  Viewer
``POST   /datasources/validate-credentials``   Admin
``POST   /datasources``                        Admin
``GET    /datasources``                        Viewer
``GET    /datasources/{id}``                   Viewer
``PUT    /datasources/{id}``                   Admin
``DELETE /datasources/{id}``                   Admin
``POST   /datasources/{id}/validate``          Admin
``GET    /datasources/{id}/resources``         Admin
``POST   /datasources/{id}/resource-ancestors``Admin
``POST   /datasources/{id}/sync``              Admin
``POST   /datasources/{id}/pause``             Admin
``POST   /datasources/{id}/resume``            Admin
``GET    /datasources/{id}/logs``              Viewer
``GET    /datasources/logs/{log_id}``          Viewer
============================================  ========

Route order matters: ``/logs/{log_id}`` and the two static paths
(``/types``, ``/validate-credentials``) are declared before
``/{id}``-shaped routes so a literal segment is never captured as an id.

Tenant scoping is not optional here — every service call takes the
caller's ``tenant_id`` from the request context, and a cross-workspace id
reads as 404 rather than 403 so the id space is not enumerable.

Query-parameter ``description`` strings are intentionally Chinese
(mirrors the upstream Go swagger annotations). RUF001 flags the
full-width punctuation; suppressed file-wide for the same reason as
``src/web/api/system/router.py``.
"""
# ruff: noqa: RUF001

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from src.common.exception import UnauthorizedError
from src.core.contracts.infra import (
    CreateDataSourceRequest,
    DataSource,
    DataSourceConnectorMetadata,
    ResolveResourceAncestorsRequest,
    SyncLog,
    UpdateDataSourceRequest,
    ValidateCredentialsRequest,
)
from src.core.infra.datasources.connector_base import list_available_connectors
from src.core.infra.datasources.service.datasource_service import (
    DEFAULT_SYNC_LOG_LIMIT,
    MAX_SYNC_LOG_LIMIT,
)
from src.web.api.infra.datasources.views import (
    ConnectionStatusResponse,
    ResolveResourceAncestorsResponse,
    ResourceResponse,
    connector_metadata_to_contract,
    datasource_to_contract,
    resource_to_response,
    sync_log_to_contract,
)
from src.web.deps import AuthDep, RoleAdminDep, RoleViewerDep
from src.web.deps.infra_datasources import DataSourceServiceDep
from src.web.middleware.context import get_tenant_id, get_user_info

router = APIRouter(prefix="/datasources", tags=["datasources"])

# Go answers validate / pause / resume with a bare status string.
_STATUS_CONNECTED = "connected"
_STATUS_PAUSED = "paused"
_STATUS_ACTIVE = "active"


def _require_tenant(request: Request) -> int:
    """Return the caller's active workspace id, or fail.

    A data source is always workspace-scoped; without a tenant context
    there is no safe default (tenant 0 is the system scope, which owns no
    data sources), so this rejects rather than guessing.
    """
    tenant_id = get_tenant_id(request)
    if tenant_id == 0:
        raise UnauthorizedError(
            code="auth.tenant_context_missing",
            message="unauthorized: workspace context missing",
        )
    return tenant_id


def _actor(request: Request) -> str:
    """Return the acting user's id for audit rows (``""`` when absent)."""
    info = get_user_info(request)
    return info.get("id", "") if info else ""


# ── Static routes (declared before /{id} to avoid capture) ───────────


@router.get("/types", response_model=list[DataSourceConnectorMetadata])
async def list_connector_types(
    _auth: AuthDep,
    _role: RoleViewerDep,
) -> list[DataSourceConnectorMetadata]:
    """List every available connector type, ordered for the UI picker.

    Static metadata — no workspace scoping, no service call.
    """
    return [connector_metadata_to_contract(m) for m in list_available_connectors()]


@router.post("/validate-credentials", response_model=ConnectionStatusResponse)
async def validate_credentials(
    request: Request,
    _auth: AuthDep,
    _role: RoleAdminDep,
    body: ValidateCredentialsRequest,
    service: DataSourceServiceDep,
) -> ConnectionStatusResponse:
    """Test a raw credential map without persisting anything.

    Backs the "Test Connection" button on the creation form, so the user
    learns the credentials are wrong before a source row exists.
    """
    _require_tenant(request)
    await service.validate_credentials(type=body.type, credentials=body.credentials)
    return ConnectionStatusResponse(status=_STATUS_CONNECTED)


@router.get("/logs/{log_id}", response_model=SyncLog)
async def get_sync_log(
    request: Request,
    _auth: AuthDep,
    _role: RoleViewerDep,
    log_id: str,
    service: DataSourceServiceDep,
) -> SyncLog:
    """Get one sync-log entry.

    Ownership is checked through the log's data source, so a log id from
    another workspace reads as not-found.
    """
    tenant_id = _require_tenant(request)
    info = await service.get_sync_log(log_id=log_id, tenant_id=tenant_id)
    return sync_log_to_contract(info)


# ── CRUD ────────────────────────────────────────────────────────────


@router.post("", response_model=DataSource, status_code=201)
async def create_datasource(
    request: Request,
    _auth: AuthDep,
    _role: RoleAdminDep,
    body: CreateDataSourceRequest,
    service: DataSourceServiceDep,
) -> DataSource:
    """Create a data source for a knowledge base.

    The connector type must be registered, and when the config carries
    credentials the live connection is validated before anything is
    persisted — so a source never lands in a state that cannot sync.
    """
    tenant_id = _require_tenant(request)
    info = await service.create(
        tenant_id=tenant_id,
        knowledge_base_id=body.knowledge_base_id,
        name=body.name,
        type=body.type,
        config=body.config,
        sync_schedule=body.sync_schedule,
        sync_mode=body.sync_mode,
        conflict_strategy=body.conflict_strategy,
        sync_deletions=body.sync_deletions,
        sync_log_retention_days=body.sync_log_retention_days,
        actor_user_id=_actor(request),
    )
    return datasource_to_contract(info)


@router.get("", response_model=list[DataSource])
async def list_datasources(
    request: Request,
    _auth: AuthDep,
    _role: RoleViewerDep,
    service: DataSourceServiceDep,
    kb_id: str = Query(default="", description="知识库 ID"),
) -> list[DataSource]:
    """List a knowledge base's data sources, each with its latest sync log."""
    tenant_id = _require_tenant(request)
    infos = await service.list_by_knowledge_base(
        knowledge_base_id=kb_id,
        tenant_id=tenant_id,
    )
    return [datasource_to_contract(i) for i in infos]


@router.get("/{id}", response_model=DataSource)
async def get_datasource(
    request: Request,
    _auth: AuthDep,
    _role: RoleViewerDep,
    id: str,
    service: DataSourceServiceDep,
) -> DataSource:
    """Get one data source, enriched with its sync aggregates."""
    tenant_id = _require_tenant(request)
    info = await service.get(id=id, tenant_id=tenant_id)
    return datasource_to_contract(info)


@router.put("/{id}", response_model=DataSource)
async def update_datasource(
    request: Request,
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    body: UpdateDataSourceRequest,
    service: DataSourceServiceDep,
) -> DataSource:
    """Update a data source's mutable fields.

    ``knowledge_base_id`` cannot change. Credentials in the body are
    ignored by contract — the stored map is preserved — because a
    credential write belongs to the credential subresource.
    """
    tenant_id = _require_tenant(request)
    info = await service.update(
        id=id,
        tenant_id=tenant_id,
        name=body.name,
        config=body.config,
        sync_schedule=body.sync_schedule,
        sync_mode=body.sync_mode,
        conflict_strategy=body.conflict_strategy,
        sync_deletions=body.sync_deletions,
        sync_log_retention_days=body.sync_log_retention_days,
        actor_user_id=_actor(request),
    )
    return datasource_to_contract(info)


@router.delete("/{id}", status_code=204)
async def delete_datasource(
    request: Request,
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    service: DataSourceServiceDep,
) -> None:
    """Soft-delete a data source and cancel its in-flight syncs."""
    tenant_id = _require_tenant(request)
    await service.delete(id=id, tenant_id=tenant_id, actor_user_id=_actor(request))


# ── Connectivity + resource browsing ────────────────────────────────


@router.post("/{id}/validate", response_model=ConnectionStatusResponse)
async def validate_connection(
    request: Request,
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    service: DataSourceServiceDep,
) -> ConnectionStatusResponse:
    """Test a stored source's connection, recording the outcome.

    A failure persists ``status=error`` plus the message before the error
    propagates, so the list view reflects the problem without a re-test.
    """
    tenant_id = _require_tenant(request)
    await service.validate_connection(id=id, tenant_id=tenant_id)
    return ConnectionStatusResponse(status=_STATUS_CONNECTED)


@router.get("/{id}/resources", response_model=list[ResourceResponse])
async def list_available_resources(
    request: Request,
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    service: DataSourceServiceDep,
    parent_id: str = Query(default="", description="父资源 ExternalID，留空表示顶层"),
) -> list[ResourceResponse]:
    """Browse the external system's syncable resources, one level deep.

    Admin+ despite being a read: it calls the external API with the
    workspace's credentials.
    """
    tenant_id = _require_tenant(request)
    resources = await service.list_available_resources(
        id=id,
        tenant_id=tenant_id,
        parent_id=parent_id,
    )
    return [resource_to_response(r) for r in resources]


@router.post("/{id}/resource-ancestors", response_model=ResolveResourceAncestorsResponse)
async def resolve_resource_ancestors(
    request: Request,
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    body: ResolveResourceAncestorsRequest,
    service: DataSourceServiceDep,
) -> ResolveResourceAncestorsResponse:
    """Resolve which tree nodes a lazy picker must expand.

    Used when reopening an edit form to reveal a pre-existing, possibly
    deeply nested selection without walking the whole tree.
    """
    tenant_id = _require_tenant(request)
    ancestors = await service.resolve_resource_ancestors(
        id=id,
        tenant_id=tenant_id,
        resource_ids=body.resource_ids,
    )
    return ResolveResourceAncestorsResponse(ancestors=ancestors)


# ── Sync control ────────────────────────────────────────────────────


@router.post("/{id}/sync", response_model=SyncLog)
async def manual_sync(
    request: Request,
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    service: DataSourceServiceDep,
) -> SyncLog:
    """Trigger an immediate sync, returning its opened sync log.

    Allowed on a ``paused`` source: a manual run is an explicit override
    of the schedule, not a resume.
    """
    tenant_id = _require_tenant(request)
    info = await service.manual_sync(id=id, tenant_id=tenant_id)
    return sync_log_to_contract(info)


@router.post("/{id}/pause", response_model=ConnectionStatusResponse)
async def pause_datasource(
    request: Request,
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    service: DataSourceServiceDep,
) -> ConnectionStatusResponse:
    """Pause a data source so scheduled syncs stop firing."""
    tenant_id = _require_tenant(request)
    await service.pause(id=id, tenant_id=tenant_id, actor_user_id=_actor(request))
    return ConnectionStatusResponse(status=_STATUS_PAUSED)


@router.post("/{id}/resume", response_model=ConnectionStatusResponse)
async def resume_datasource(
    request: Request,
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    service: DataSourceServiceDep,
) -> ConnectionStatusResponse:
    """Resume a paused data source and clear any recorded error."""
    tenant_id = _require_tenant(request)
    await service.resume(id=id, tenant_id=tenant_id, actor_user_id=_actor(request))
    return ConnectionStatusResponse(status=_STATUS_ACTIVE)


# ── Sync history ────────────────────────────────────────────────────


@router.get("/{id}/logs", response_model=list[SyncLog])
async def list_sync_logs(
    request: Request,
    _auth: AuthDep,
    _role: RoleViewerDep,
    id: str,
    service: DataSourceServiceDep,
    limit: int = Query(
        default=DEFAULT_SYNC_LOG_LIMIT,
        description=f"页大小，1-{MAX_SYNC_LOG_LIMIT}",
    ),
    offset: int = Query(default=0, description="偏移量"),
) -> list[SyncLog]:
    """List a data source's sync history, newest first."""
    tenant_id = _require_tenant(request)
    infos = await service.list_sync_logs(
        id=id,
        tenant_id=tenant_id,
        limit=limit,
        offset=offset,
    )
    return [sync_log_to_contract(i) for i in infos]


__all__ = ["router"]
