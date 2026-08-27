"""System-service HTTP endpoints — info, parser engines, docreader, storage.

Maps the remaining viewer-facing system-service endpoints from the
upstream system handler, complementing the admin endpoints in
``src/web/api/system/router.py``:

==============================================  ====
Route                                           Action
==============================================  ====
``GET    /system/info``                          System version / build / engine config
``GET    /system/parser-engines``                List parser engines (merged local + remote)
``POST   /system/parser-engines/check``          Check engine availability with a config body
``POST   /system/docreader/reconnect``           Reconnect the document reader
``GET    /system/storage-engine-status``         Storage provider availability
``POST   /system/storage-engine-check``          Storage provider connectivity probe
==============================================  ====

The docreader-backed availability probes and live storage
connectivity checks depend on the transport layer (gRPC client, S3
SDKs) that lands with the docreader / storage infrastructure. Until
then the parser-engine registry reports availability from configuration
presence (mirroring the ported ``src.core.system.parser_engine``), and
the reconnect / connectivity endpoints validate the request shape and
return the standard envelope — a deferred seam rather than a
fabricated probe result.

Query-parameter ``description`` strings are intentionally Chinese
(mirrors the upstream swagger annotations).
"""

from __future__ import annotations

import os

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from src.common.exception import ValidationError
from src.common.json import JsonObject
from src.common.oidc_client import validate_ssrf_safe_url
from src.core.contracts.infra import StorageProviderStatus
from src.core.system.parser_engine import (
    ParserEngineInfo,
    list_all_engines,
)
from src.core.system.storage_allowlist import (
    allowed_providers,
    build_storage_provider_statuses,
    supported_providers,
)
from src.web.deps import AuthDep, RoleAdminDep, RoleViewerDep

router = APIRouter(prefix="/system", tags=["system"])

# ── Env-driven engine detection (system.go helpers) ───────────────────


def _minio_enabled() -> bool:
    """Return whether MinIO storage is configured.

    Mirrors ``isMinioEnvAvailable``: presence of the endpoint + access
    key env vars is the environment-level signal (the tenant-scoped
    config is resolved by the storage backend domain).
    """
    return bool(
        os.getenv("MINIO_ENDPOINT", "").strip() and os.getenv("MINIO_ACCESS_KEY_ID", "").strip()
    )


# ── View models (wire shape) ─────────────────────────────────────────


class ParserEnginesResponse(BaseModel):
    """``{"success": true, "data": [...], "docreader_addr": ..., ...}``.

    The parser-engine list response carries the docreader connection
    metadata as envelope siblings, mirroring the upstream handler's
    flat JSON object. Engine entries use the wire shape from
    :class:`ParserEngineInfo` (PascalCase keys).
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    data: list[ParserEngineInfo]
    docreader_addr: str = ""
    docreader_transport: str = ""
    connected: bool = False


class ParserEngineCheckRequest(BaseModel):
    """Body for ``POST /system/parser-engines/check``.

    Mirrors the upstream ``ParserEngineConfig`` shape (the same body
    the save endpoint accepts) so the UI can test a candidate
    configuration without persisting it.
    """

    model_config = ConfigDict(frozen=True)

    addr: str = ""


class ReconnectDocReaderRequest(BaseModel):
    """Body for ``POST /system/docreader/reconnect``.

    ``addr`` is the docreader connection address; the upstream handler
    rejects blank or SSRF-unsafe values with ``400``.
    """

    model_config = ConfigDict(frozen=True)

    addr: str


class StorageEngineStatusData(BaseModel):
    """``data`` payload of ``GET /system/storage-engine-status``."""

    model_config = ConfigDict(frozen=True)

    engines: list[StorageProviderStatus]
    allowed_providers: list[str] = Field(default_factory=list)
    minio_env_available: bool = False


class StorageEngineStatusResponse(BaseModel):
    """``{"success": true, "data": StorageEngineStatusData}``."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: StorageEngineStatusData


class StorageEngineCheckRequest(BaseModel):
    """Body for ``POST /system/storage-engine-check``.

    ``provider`` selects the storage engine; the engine-specific config
    is carried in the ``config`` object (endpoint / credentials /
    bucket per provider shape). Connectivity probing lands with the
    storage-backend transport layer; this endpoint validates the
    request shape and returns the standard envelope.
    """

    model_config = ConfigDict(frozen=True)

    provider: str
    config: JsonObject = Field(default_factory=dict)


class StorageEngineCheckData(BaseModel):
    """``data`` payload of ``POST /system/storage-engine-check``."""

    model_config = ConfigDict(frozen=True)

    ok: bool
    message: str
    bucket_created: bool = False


class StorageEngineCheckResponse(BaseModel):
    """``{"success": true, "data": StorageEngineCheckData}``."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: StorageEngineCheckData


class SuccessResponse(BaseModel):
    """``{"success": true}`` for mutation endpoints."""

    model_config = ConfigDict(frozen=True)

    success: bool


# ── Endpoints ────────────────────────────────────────────────────────


@router.get("/parser-engines", response_model=ParserEnginesResponse)
async def list_parser_engines(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
) -> ParserEnginesResponse:
    """Return the merged parser-engine list.

    Mirrors ``GET /system/parser-engines`` in the upstream Go handler
    — gated by ``g.Viewer()``: any authenticated principal with at
    least Viewer role in a workspace may read it. Local engines come
    from the ported registry; the remote docreader engine discovery
    is a deferred seam (returns the local registry only until the
    transport layer lands).
    """
    engines = list_all_engines(docreader_connected=False)
    return ParserEnginesResponse(
        success=True,
        data=engines,
        docreader_addr="",
        docreader_transport="",
        connected=False,
    )


@router.post("/parser-engines/check", response_model=ParserEnginesResponse)
async def check_parser_engines(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    body: ParserEngineCheckRequest,
) -> ParserEnginesResponse:
    """Check parser-engine availability under a candidate config.

    Mirrors ``POST /system/parser-engines/check`` in the upstream Go
    handler — gated by ``g.Admin()``. The body mirrors the
    parser-engine configuration shape; availability is resolved from
    configuration presence until the live probing transport lands.
    """
    engines = list_all_engines(docreader_connected=False)
    return ParserEnginesResponse(
        success=True,
        data=engines,
        docreader_addr=body.addr,
        docreader_transport="",
        connected=False,
    )


@router.post("/docreader/reconnect", response_model=SuccessResponse)
async def reconnect_docreader(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    body: ReconnectDocReaderRequest,
) -> SuccessResponse:
    """Reconnect the document reader to ``addr``.

    Mirrors ``POST /system/docreader/reconnect`` in the upstream Go
    handler — gated by ``g.Admin()``. ``addr`` must be non-blank and
    SSRF-safe (mirrors the upstream validation). The live reconnection
    is a deferred seam — the transport layer is not yet wired — so a
    valid request returns the success envelope without a probe.
    """
    addr = body.addr.strip()
    if not addr:
        raise ValidationError(
            code="system.docreader_addr_empty",
            message="addr 不能为空",
        )
    try:
        await validate_ssrf_safe_url(addr)
    except ValidationError:
        raise ValidationError(
            code="system.docreader_ssrf_blocked",
            message="DocReader 地址未通过安全校验",
        ) from None
    return SuccessResponse(success=True)


@router.get("/storage-engine-status", response_model=StorageEngineStatusResponse)
async def get_storage_engine_status(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
) -> StorageEngineStatusResponse:
    """Return storage provider availability.

    Mirrors ``GET /system/storage-engine-status`` in the upstream Go
    handler — gated by ``g.Viewer()``. ``local`` is unconditionally
    available; the object-storage providers report availability from
    the environment signal (the tenant-scoped storage-backend config
    is resolved by the storage domain). ``allowed`` reflects the
    ``STORAGE_ALLOW_LIST`` gate.
    """
    engines = build_storage_provider_statuses()
    return StorageEngineStatusResponse(
        success=True,
        data=StorageEngineStatusData(
            engines=engines,
            allowed_providers=allowed_providers(),
            minio_env_available=_minio_enabled(),
        ),
    )


@router.post("/storage-engine-check", response_model=StorageEngineCheckResponse)
async def check_storage_engine(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    body: StorageEngineCheckRequest,
) -> StorageEngineCheckResponse:
    """Probe one storage provider's connectivity.

    Mirrors ``POST /system/storage-engine-check`` in the upstream Go
    handler — gated by ``g.Admin()``. ``provider`` must name a
    supported storage engine; the live connectivity probe is a
    deferred seam that lands with the storage-backend transport layer.
    A supported provider yields the success envelope with an explicit
    ``ok=false`` so the UI does not mistake the deferred probe for a
    confirmed connection.
    """
    provider = body.provider.strip().lower()
    if provider not in set(supported_providers()):
        raise ValidationError(
            code="system.unknown_storage_provider",
            message=f"unsupported storage provider {body.provider!r}",
        )
    return StorageEngineCheckResponse(
        success=True,
        data=StorageEngineCheckData(
            ok=False,
            message="连接检查暂未启用",
        ),
    )


__all__ = ["router"]
