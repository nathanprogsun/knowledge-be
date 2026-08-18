"""System-admin management endpoints — API keys, roles, password, runtime.

Maps the remaining ``/system/admin/*`` endpoints from the upstream
system handler, complementing the settings + audit-log endpoints in
``src.web.api.system.router``:

==============================================  ====
Route                                           Action
==============================================  ====
``GET    /system/admin/api-keys``               List platform API keys
``POST   /system/admin/api-keys``               Create a platform API key
``DELETE /system/admin/api-keys/{key_id}``      Revoke a platform API key
``POST   /system/admin/promote``                Promote a user to system admin
``POST   /system/admin/revoke``                 Revoke system admin
``POST   /system/admin/users/reset-password``   Admin password reset
``GET    /system/admin/runtime/queues``         Queue runtime dashboard
==============================================  ====

Every route is gated with ``SystemAdminDep`` — the same gate the
settings / audit-log admin endpoints use.

Success envelopes match the upstream Go handlers exactly: the key
list/create/delete endpoints wrap in ``{"success", "data"}`` while
promote / revoke / reset-password / runtime-queues return their payload
directly (no envelope), matching the frontend's per-call type casts.

The runtime-queues payload is the static pool topology with
``available=false``: the API process does not hold live ARQ queue state
(the worker runs as a separate process), so queue depths and worker
heartbeats are reported as empty rather than fabricated.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from src.common.exception import ValidationError
from src.core.auth.types import UserInfo
from src.core.tenants.api_key_service import (
    SCOPE_PLATFORM,
    PlatformAPIKeyView,
    mask_api_key_token,
    normalize_capabilities,
)
from src.core.tenants.types import TenantAPIKeyInfo
from src.web.deps import (
    AuthDep,
    SystemAdminDep,
    SystemAdminServiceDep,
    TenantAPIKeyServiceDep,
    get_request_user_id,
)
from src.web.deps.system import SystemSettingServiceDep
from src.web.deps.tenants import TenantServiceDep

router = APIRouter(prefix="/system/admin", tags=["system-admin"])


# ── View models (wire shape) ─────────────────────────────────────────


class PlatformAPIKeyResponse(BaseModel):
    """Wire shape for one platform API key (token always masked)."""

    model_config = ConfigDict(frozen=True)

    id: int
    scope_type: str
    name: str
    api_key: str = Field(default="")
    full_access: bool
    knowledge_base_ids: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    last_used_at: datetime | None = None
    expires_at: datetime | None = None
    created_at: datetime


class CreatedPlatformAPIKeyResponse(PlatformAPIKeyResponse):
    """Create response — ``api_key`` is the masked token, ``token`` is plaintext.

    The plaintext token is returned exactly once, by the create call.
    """

    model_config = ConfigDict(frozen=True)

    token: str


class PlatformAPIKeyListEnvelope(BaseModel):
    """``{"success": true, "data": [...]}`` for ``GET /api-keys``."""

    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: list[PlatformAPIKeyResponse]


class PlatformAPIKeyCreateEnvelope(BaseModel):
    """``{"success": true, "data": {..., "token": ...}}`` (HTTP 201)."""

    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: CreatedPlatformAPIKeyResponse


class SuccessEnvelope(BaseModel):
    """``{"success": true}`` for ``DELETE /api-keys/{key_id}``."""

    model_config = ConfigDict(frozen=True)

    success: bool = True


class CreatePlatformAPIKeyRequest(BaseModel):
    """Body for ``POST /system/admin/api-keys``.

    Mirrors the upstream ``platformAPIKeyCreateRequest``: a name, the
    bounded capability grants, and an optional Unix expiry.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    capabilities: list[str]
    expires_at_unix: int | None = None


class PromoteUserRequest(BaseModel):
    """Body for ``POST /system/admin/promote``.

    ``user_id`` takes priority when both identifiers are supplied.
    """

    model_config = ConfigDict(frozen=True)

    user_id: str = ""
    email: str = ""


class RevokeSystemAdminRequest(BaseModel):
    """Body for ``POST /system/admin/revoke``."""

    model_config = ConfigDict(frozen=True)

    user_id: str


class ResetUserPasswordRequest(BaseModel):
    """Body for ``POST /system/admin/users/reset-password``."""

    model_config = ConfigDict(frozen=True)

    email: str
    new_password: str


class ResetPasswordResponse(BaseModel):
    """Wire shape for the successful password-reset response."""

    model_config = ConfigDict(frozen=True)

    message: str


class RuntimeWorkerPool(BaseModel):
    """One worker pool's configured concurrency + live aggregate state."""

    model_config = ConfigDict(frozen=True)

    name: str
    concurrency: int
    queue_count: int
    instances: int = 0
    cluster_capacity: int = 0
    active: int = 0
    utilization: float = 0.0


class RuntimeQueueStat(BaseModel):
    """Read-only depth snapshot of one queue (asynq QueueInfo semantics)."""

    model_config = ConfigDict(frozen=True)

    name: str
    pool: str
    weight: int
    size: int
    pending: int
    active: int
    scheduled: int
    retry: int
    archived: int
    completed: int
    processed: int
    failed: int
    paused: bool
    latency_ms: int
    memory_usage_bytes: int


class RuntimeModelStat(BaseModel):
    """Per-model concurrency limiter state."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    name: str
    active: int
    waiting: int
    limit: int


class RuntimeQueuesResponse(BaseModel):
    """Wire shape for ``GET /system/admin/runtime/queues``."""

    model_config = ConfigDict(frozen=True)

    available: bool
    upstream_concurrency: int
    parse_concurrency: int
    wiki_concurrency: int
    pools: list[RuntimeWorkerPool]
    queues: list[RuntimeQueueStat]
    model_limiter_available: bool
    models: list[RuntimeModelStat]
    timestamp: int


# ── Runtime pool topology (static, mirrors the upstream worker pools) ─

# (pool name, default concurrency, queue count). The queue counts follow
# the upstream queue registry: core=2, postprocess=1, enrichment=4,
# maintenance=2, shared=6 (queues with a shared weight), wiki=1.
_POOL_SPECS: Final[tuple[tuple[str, int, int], ...]] = (
    ("core", 8, 2),
    ("postprocess", 2, 1),
    ("enrichment", 12, 4),
    ("maintenance", 4, 2),
    ("shared", 6, 6),
    ("wiki", 8, 1),
)

# upstream = core + postprocess + enrichment + maintenance + shared.
_UPSTREAM_CONCURRENCY: Final[int] = 8 + 2 + 12 + 4 + 6
_WIKI_CONCURRENCY: Final[int] = 8


# ── Conversion helpers ───────────────────────────────────────────────


def _api_key_to_response(view: PlatformAPIKeyView) -> PlatformAPIKeyResponse:
    """Project a platform key view onto the wire shape (masked token)."""
    key = view.key
    return PlatformAPIKeyResponse(
        id=key.id,
        scope_type=key.scope_type,
        name=key.name,
        api_key=view.api_key_masked,
        full_access=key.full_access,
        knowledge_base_ids=key.knowledge_base_ids,
        capabilities=key.capabilities,
        last_used_at=key.last_used_at,
        expires_at=key.expires_at,
        created_at=key.created_at,
    )


def _runtime_queues_response() -> RuntimeQueuesResponse:
    """Assemble the queue-dashboard payload.

    ``available`` is false: the API process has no live ARQ queue
    state to report (the worker is a separate process), so pools carry
    their configured topology and the live-depth arrays are empty.
    """
    now = int(datetime.now(UTC).timestamp())
    return RuntimeQueuesResponse(
        available=False,
        upstream_concurrency=_UPSTREAM_CONCURRENCY,
        parse_concurrency=_UPSTREAM_CONCURRENCY,
        wiki_concurrency=_WIKI_CONCURRENCY,
        pools=[
            RuntimeWorkerPool(name=name, concurrency=conc, queue_count=count)
            for name, conc, count in _POOL_SPECS
        ],
        queues=[],
        model_limiter_available=False,
        models=[],
        timestamp=now,
    )


# ── Endpoints ────────────────────────────────────────────────────────


@router.get(
    "/api-keys",
    response_model=PlatformAPIKeyListEnvelope,
    response_model_exclude_none=True,
)
async def list_platform_api_keys(
    _auth: AuthDep,
    _admin: SystemAdminDep,
    api_key_service: TenantAPIKeyServiceDep,
) -> PlatformAPIKeyListEnvelope:
    """List live platform API keys, newest first, tokens masked."""
    views = await api_key_service.list_platform_api_keys_for_admin()
    return PlatformAPIKeyListEnvelope(data=[_api_key_to_response(v) for v in views])


@router.post(
    "/api-keys",
    response_model=PlatformAPIKeyCreateEnvelope,
    status_code=201,
    response_model_exclude_none=True,
)
async def create_platform_api_key(
    _auth: AuthDep,
    _admin: SystemAdminDep,
    body: CreatePlatformAPIKeyRequest,
    api_key_service: TenantAPIKeyServiceDep,
) -> PlatformAPIKeyCreateEnvelope:
    """Create a capability-scoped platform key; the token is returned once.

    Mirrors the upstream validation: a non-blank name, a non-empty set
    of valid capabilities (unknown values are rejected, not dropped),
    and an ``expires_at_unix`` in the future when supplied.
    """
    name = body.name.strip()
    if not name:
        raise ValidationError(
            code="system.api_key_name_required",
            message="name is required",
        )
    capabilities = normalize_capabilities(body.capabilities)
    if not capabilities or len(capabilities) != len(body.capabilities):
        raise ValidationError(
            code="system.api_key_invalid_capabilities",
            message="valid capabilities are required",
        )
    expires_at: datetime | None = None
    if body.expires_at_unix is not None:
        value = datetime.fromtimestamp(body.expires_at_unix, tz=UTC)
        if value <= datetime.now(UTC):
            raise ValidationError(
                code="system.api_key_expiry_past",
                message="expires_at_unix must be in the future",
            )
        expires_at = value
    result = await api_key_service.create_api_key(
        name=name,
        scope_type=SCOPE_PLATFORM,
        capabilities=capabilities,
        expires_at=expires_at,
    )
    key: TenantAPIKeyInfo = result.key
    return PlatformAPIKeyCreateEnvelope(
        data=CreatedPlatformAPIKeyResponse(
            id=key.id,
            scope_type=key.scope_type,
            name=key.name,
            api_key=mask_api_key_token(result.token),
            full_access=key.full_access,
            knowledge_base_ids=key.knowledge_base_ids,
            capabilities=key.capabilities,
            last_used_at=key.last_used_at,
            expires_at=key.expires_at,
            created_at=key.created_at,
            token=result.token,
        )
    )


@router.delete("/api-keys/{key_id}", response_model=SuccessEnvelope)
async def delete_platform_api_key(
    _auth: AuthDep,
    _admin: SystemAdminDep,
    key_id: int,
    api_key_service: TenantAPIKeyServiceDep,
) -> SuccessEnvelope:
    """Revoke a platform API key immediately."""
    if key_id <= 0:
        raise ValidationError(
            code="system.invalid_api_key_id",
            message="Invalid API key ID",
        )
    await api_key_service.revoke_platform_api_key(key_id)
    return SuccessEnvelope(success=True)


@router.post("/promote", response_model=UserInfo)
async def promote_user_to_system_admin(
    _auth: AuthDep,
    _admin: SystemAdminDep,
    request: Request,
    body: PromoteUserRequest,
    admin_service: SystemAdminServiceDep,
) -> UserInfo:
    """Promote a user to system admin (idempotent).

    The target is identified by ``user_id`` (priority) or ``email``.
    The response is the updated user profile directly, no envelope.
    """
    user_id = body.user_id.strip()
    email = body.email.strip()
    if not user_id and not email:
        raise ValidationError(
            code="system.promote_target_required",
            message="Either user_id or email is required",
        )
    return await admin_service.promote(
        user_id=user_id,
        email=email,
        actor_id=get_request_user_id(request),
    )


@router.post("/revoke", response_model=UserInfo)
async def revoke_system_admin(
    _auth: AuthDep,
    _admin: SystemAdminDep,
    request: Request,
    body: RevokeSystemAdminRequest,
    admin_service: SystemAdminServiceDep,
) -> UserInfo:
    """Revoke system-admin privileges from a user.

    Guards: cannot revoke the caller's own privileges and cannot revoke
    the last remaining system admin. Revoking a non-admin is an
    idempotent success.
    """
    user_id = body.user_id.strip()
    if not user_id:
        raise ValidationError(
            code="system.revoke_target_required",
            message="user_id is required",
        )
    return await admin_service.revoke(
        user_id=user_id,
        actor_id=get_request_user_id(request),
    )


@router.post("/users/reset-password", response_model=ResetPasswordResponse)
async def reset_user_password(
    _auth: AuthDep,
    _admin: SystemAdminDep,
    request: Request,
    body: ResetUserPasswordRequest,
    admin_service: SystemAdminServiceDep,
) -> ResetPasswordResponse:
    """Replace another user's password and revoke their sessions."""
    email = body.email.strip()
    if not email:
        raise ValidationError(
            code="system.reset_email_required",
            message="email is required",
        )
    await admin_service.reset_password(
        email=email,
        new_password=body.new_password,
        actor_id=get_request_user_id(request),
    )
    return ResetPasswordResponse(message="Password reset successfully")


@router.get("/runtime/queues", response_model=RuntimeQueuesResponse)
async def get_runtime_queues(
    _auth: AuthDep,
    _admin: SystemAdminDep,
) -> RuntimeQueuesResponse:
    """Return the queue-dashboard payload (static topology, no live depth)."""
    return _runtime_queues_response()


@router.post("/tenants/apply-default-storage-quota")
async def apply_default_storage_quota(
    _auth: AuthDep,
    _admin: SystemAdminDep,
    tenant_service: TenantServiceDep,
    system_setting_service: SystemSettingServiceDep,
) -> dict[str, int]:
    """Apply the default storage quota to every existing workspace.

    Reads ``tenant.default_storage_quota_gb`` (DB > default 10) and
    writes that many GiB into ``storage_quota`` for every tenant row.
    Idempotent; SystemAdmin only.
    """
    gb = await system_setting_service.get_int(
        "tenant.default_storage_quota_gb",
        "",
        10,
    )
    if gb <= 0:
        gb = 10
    quota_bytes = gb * 1024 * 1024 * 1024
    affected = await tenant_service.bulk_set_storage_quota(quota_bytes=quota_bytes)
    return {"affected": affected, "quota_bytes": quota_bytes}


@router.get("/list")
async def list_system_admins(
    _auth: AuthDep,
    _admin: SystemAdminDep,
    admin_service: SystemAdminServiceDep,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, object]:
    """List system administrators (paginated). Returns ``{total, admins}``."""
    admins, total = await admin_service.list_system_admins(
        offset=offset,
        limit=limit,
    )
    return {
        "total": total,
        "admins": [admin.model_dump(mode="json") for admin in admins],
    }


@router.get("/runtime/queues/{queue}/tasks")
async def list_runtime_tasks(
    _auth: AuthDep,
    _admin: SystemAdminDep,
    queue: str,
    state: str = Query(default=""),
    cursor: str = Query(default=""),
    page_size: int = Query(default=20, ge=1, le=100),
) -> dict[str, object]:
    """Return one task-state page for the queue dashboard.

    The API process does not hold live ARQ queue state (the worker runs
    as a separate process), so the payload reports ``available: false``
    with an empty task list — matching the runtime-queues endpoint.
    """
    return {
        "available": False,
        "tasks": [],
        "page_size": page_size,
        "has_more": False,
        "next_cursor": None,
    }


__all__ = ["router"]
