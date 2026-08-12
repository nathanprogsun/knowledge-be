"""System admin HTTP endpoints — settings CRUD + audit log viewer.

Maps the system-admin endpoints from ``internal/handler/system.go``
and ``internal/handler/audit_log.go``:

- ``GET    /system/admin/settings``         — list every known setting
- ``GET    /system/admin/settings/{key}``   — get one setting by key
- ``PUT    /system/admin/settings/{key}``   — update a setting value
- ``DELETE /system/admin/settings/{key}``   — reset a setting to ENV/default
- ``GET    /system/admin/audit-log``        — system-scope audit feed
- ``GET    /system/info``                   — build + DB metadata (Viewer-gated)

Tenant-scoped audit-log endpoints (``GET /tenants/{id}/audit-log``) and
KB activity (``GET /knowledge-bases/{id}/activity``) are not yet
implemented — they require RBAC middleware and the KB domain
respectively.

Wire-shape conversion (``SystemSettingInfo`` → response model) lives in
this module so the router stays declarative.

Query-parameter ``description`` strings are intentionally Chinese
(mirrors the upstream Go swagger annotations). RUF001 flags the
full-width punctuation; suppressed file-wide for the same reason as
``src/core/system/registry.py``.
"""
# ruff: noqa: RUF001

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject, JsonValue
from src.core.system.audit_service import AuditLogListResult
from src.core.system.info_service import SystemInfoSnapshot
from src.core.system.types import AuditLogInfo, SystemSettingInfo
from src.web.deps import (
    AuditLogServiceDep,
    AuthDep,
    RoleViewerDep,
    SystemAdminDep,
    SystemInfoServiceDep,
    SystemSettingServiceDep,
    get_request_user_id,
)

router = APIRouter(prefix="/system/admin", tags=["system-admin"])
info_router = APIRouter(prefix="/system", tags=["system"])


# ── View models (wire shape) ─────────────────────────────────────────


class SystemSettingResponse(BaseModel):
    """Wire shape for one system setting."""

    model_config = ConfigDict(frozen=True)

    id: int
    key: str
    value: JsonObject | list[str] | list[JsonValue] | str | int | bool
    value_type: str
    category: str
    description: str = ""
    is_secret: bool = False
    requires_restart: bool = False
    last_modified_by: str = ""
    created_at: datetime
    updated_at: datetime
    enum: list[str] = Field(default_factory=list)
    last_modified_by_name: str = ""


class SystemInfoWireResponse(BaseModel):
    """Wire shape for ``GET /system/info``.

    Mirrors the upstream ``GetSystemInfoResponse`` struct — the
    frozen ``SystemInfo`` contract in ``src/core.contracts.system``
    holds the build-metadata fields, and the runtime fields the
    upstream handler reports (``db_migration_error``, ``started_at``,
    ``uptime_seconds``) are added inline so the wire payload stays
    backward-compatible with the Go response without touching the
    immutable contract shape.
    """

    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: SystemInfoWireData


class SystemInfoWireData(BaseModel):
    """Inner data shape for the system info envelope."""

    model_config = ConfigDict(frozen=True)

    version: str
    edition: str
    commit_id: str
    build_time: datetime
    go_version: str
    keyword_index_engine: str
    vector_store_engine: str
    graph_database_engine: str
    minio_enabled: bool
    db_version: str
    db_migration_error: str | None = None
    started_at: datetime
    uptime_seconds: int


class UpdateSystemSettingRequest(BaseModel):
    """Body for ``PUT /system/admin/settings/{key}``.

    ``value`` carries the new value as raw JSON — int / string / bool /
    list[str] depending on the registry-declared ``value_type``. The
    service validates the type strictly and rejects mismatches with
    400.
    """

    model_config = ConfigDict(frozen=True)

    value: int | str | bool | list[str]


class ResetSystemSettingResponse(BaseModel):
    """Wire shape for ``DELETE /system/admin/settings/{key}``."""

    model_config = ConfigDict(frozen=True)

    success: bool = True


class AuditLogEntryResponse(BaseModel):
    """Wire shape for one audit-log entry."""

    model_config = ConfigDict(frozen=True)

    id: int
    tenant_id: int
    actor_user_id: str = ""
    actor_role: str = ""
    action: str
    scope_type: str = ""
    scope_id: str = ""
    target_type: str = ""
    target_id: str = ""
    target_user_id: str = ""
    request_path: str = ""
    request_method: str = ""
    outcome: str = "success"
    details: JsonObject = Field(default_factory=dict)
    created_at: datetime


class AuditLogListResponse(BaseModel):
    """Wire shape for ``GET /system/admin/audit-log``."""

    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: list[AuditLogEntryResponse]
    next_cursor: int


# ── Conversion helpers ──────────────────────────────────────────────


def _setting_to_response(info: SystemSettingInfo) -> SystemSettingResponse:
    return SystemSettingResponse(
        id=info.id,
        key=info.key,
        value=info.value,
        value_type=info.value_type,
        category=info.category,
        description=info.description,
        is_secret=info.is_secret,
        requires_restart=info.requires_restart,
        last_modified_by=info.last_modified_by,
        created_at=info.created_at,
        updated_at=info.updated_at,
        enum=info.enum,
        last_modified_by_name=info.last_modified_by_name,
    )


def _audit_to_response(info: AuditLogInfo) -> AuditLogEntryResponse:
    return AuditLogEntryResponse(
        id=info.id,
        tenant_id=info.tenant_id,
        actor_user_id=info.actor_user_id,
        actor_role=info.actor_role,
        action=info.action,
        scope_type=info.scope_type,
        scope_id=info.scope_id,
        target_type=info.target_type,
        target_id=info.target_id,
        target_user_id=info.target_user_id,
        request_path=info.request_path,
        request_method=info.request_method,
        outcome=info.outcome,
        details=info.details,
        created_at=info.created_at,
    )


def _audit_list_to_response(result: AuditLogListResult) -> AuditLogListResponse:
    """Render an :class:`AuditLogListResult` to the wire shape.

    The service already projects each row into
    :class:`AuditLogInfo` via ``map_from_db``; this layer just renders
    the wire shape — it no longer performs the §9 projection.
    """
    return AuditLogListResponse(
        success=True,
        data=[_audit_to_response(e) for e in result.entries],
        next_cursor=result.next_cursor,
    )


# ── Conversion helpers ──────────────────────────────────────────────


def _snapshot_to_response(snapshot: SystemInfoSnapshot) -> SystemInfoWireResponse:
    """Project a service snapshot onto the ``/system/info`` wire shape.

    Keeps the conversion in the router so the contract stays declarative
    (mirrors the ``_setting_to_response`` pattern used by the admin
    endpoints).
    """
    info = snapshot.info
    return SystemInfoWireResponse(
        success=True,
        data=SystemInfoWireData(
            version=info.version,
            edition=info.edition,
            commit_id=info.commit_id,
            build_time=info.build_time,
            go_version=info.go_version,
            keyword_index_engine=info.keyword_index_engine,
            vector_store_engine=info.vector_store_engine,
            graph_database_engine=info.graph_database_engine,
            minio_enabled=info.minio_enabled,
            db_version=info.db_version,
            db_migration_error=snapshot.db_migration_error,
            started_at=snapshot.started_at,
            uptime_seconds=snapshot.uptime_seconds,
        ),
    )


# ── Endpoints ───────────────────────────────────────────────────────


@info_router.get("/info", response_model=SystemInfoWireResponse)
async def get_system_info(
    info_svc: SystemInfoServiceDep,
    _auth: AuthDep,
    _viewer: RoleViewerDep,
) -> SystemInfoWireResponse:
    """Return build metadata, DB state, and process uptime.

    Mirrors ``GET /system/info`` in the upstream Go handler — gated by
    ``g.Viewer()``: any authenticated principal with at least Viewer
    role in a workspace may read it. Anonymous / tenant-less principals
    are rejected with ``auth.missing_authentication`` /
    ``rbac.insufficient_role``. The route reads the boot instant from
    ``app.state.started_at`` (set during lifespan startup) and queries
    ``alembic_version`` + ``SELECT version()`` on the request session
    to populate ``db_version`` and ``db_migration_error``.
    """
    snapshot = await info_svc.get_info()
    return _snapshot_to_response(snapshot)


@router.get("/settings", response_model=list[SystemSettingResponse])
async def list_settings(
    _auth: AuthDep,
    _admin: SystemAdminDep,
    settings_svc: SystemSettingServiceDep,
) -> list[SystemSettingResponse]:
    """List every known system setting.

    Returns DB-persisted rows enriched with registry metadata, plus
    virtual rows for registered keys that have no DB row yet.
    """
    infos = await settings_svc.list_settings()
    return [_setting_to_response(i) for i in infos]


@router.get("/settings/{key}", response_model=SystemSettingResponse)
async def get_setting(
    _auth: AuthDep,
    _admin: SystemAdminDep,
    key: str,
    settings_svc: SystemSettingServiceDep,
) -> SystemSettingResponse:
    """Get a single system setting by key."""
    info = await settings_svc.get(key)
    return _setting_to_response(info)


@router.put("/settings/{key}", response_model=SystemSettingResponse)
async def update_setting(
    _auth: AuthDep,
    _admin: SystemAdminDep,
    request: Request,
    key: str,
    body: UpdateSystemSettingRequest,
    settings_svc: SystemSettingServiceDep,
) -> SystemSettingResponse:
    """Update a system setting value.

    The service validates the value against the registry's declared
    ``value_type`` and rejects mismatches with 400. Emits an audit row
    (``system.setting_changed``) on success, stamped with the acting
    user (Go reads ``actorID`` from the request context).
    """
    info = await settings_svc.update(
        key=key,
        raw_value=body.value,
        actor_user_id=get_request_user_id(request),
    )
    return _setting_to_response(info)


@router.delete("/settings/{key}", response_model=ResetSystemSettingResponse)
async def reset_setting(
    _auth: AuthDep,
    _admin: SystemAdminDep,
    key: str,
    settings_svc: SystemSettingServiceDep,
) -> ResetSystemSettingResponse:
    """Reset a system setting to ENV / built-in default.

    Removes the DB override so the 3-tier resolver falls back. Idempotent
    — resetting a key that was never persisted returns 200.
    """
    await settings_svc.reset(key)
    return ResetSystemSettingResponse(success=True)


@router.get("/audit-log", response_model=AuditLogListResponse)
async def list_system_audit_log(
    _auth: AuthDep,
    _admin: SystemAdminDep,
    audit_svc: AuditLogServiceDep,
    after_id: int = Query(default=0, ge=0, description="游标：返回 id 小于此值的记录"),
    limit: int = Query(default=50, ge=1, le=100, description="页大小，1-100"),
    action: str | None = Query(default=None, description="按 action 精确过滤"),
    outcome: str | None = Query(default=None, description="按 outcome 精确过滤"),
    actor: str | None = Query(default=None, description="按 actor_user_id 精确过滤"),
) -> AuditLogListResponse:
    """System-scope audit log feed (tenant_id=0).

    Unlike tenant-scoped audit-log, this route is NOT tenant-scoped —
    it surfaces rows emitted by system settings changes, admin
    promote/revoke, and the apply-default-storage-quota bulk write.
    """
    result = await audit_svc.list_entries(
        tenant_id=0,
        after_id=after_id,
        limit=limit,
        action=action,
        outcome=outcome,
        actor_user_id=actor,
    )
    return _audit_list_to_response(result)


__all__ = ["info_router", "router"]
