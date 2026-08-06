"""Model HTTP endpoints - provider catalog, CRUD, debug probe.

The seven endpoints here cover the basic CRUD + provider-list +
debug-probe surface; the per-field credentials subresource is not yet
implemented.
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile

from src.common.exception import ValidationError
from src.common.json import JsonValue
from src.core.contracts.infra import (
    CreateModelRequest,
    ProviderTypeMeta,
    UpdateModelRequest,
)
from src.core.infra.models.catalog import PROVIDER_CATALOG, filter_providers
from src.web.api.infra.models.views import (
    DeleteModelResponse,
    ModelDebugEnvelope,
    ModelEnvelope,
    ModelListEnvelope,
    ProviderListEnvelope,
    model_envelope,
    model_list_envelope,
    provider_list_envelope,
)
from src.web.deps import (
    AuthDep,
    RoleAdminDep,
    RoleViewerDep,
)
from src.web.deps.context import get_is_system_admin_dep, get_tenant_id_dep
from src.web.deps.infra_models import ModelServiceDep

# Function-arg-style principal dep aliases.
_PrincipalTenant = Annotated[int, Depends(get_tenant_id_dep)]
_PrincipalSystemAdmin = Annotated[bool, Depends(get_is_system_admin_dep)]


router = APIRouter(prefix="/models", tags=["models"])

# The ``modelDebugMaxInputBytes`` hard cap
# protects the debug probe from runaway input sizes.
_DEBUG_MAX_INPUT_BYTES = 64 * 1024


def _require_tenant(tenant_id: int) -> int:
    """Return the resolved workspace id, or raise when missing."""
    if tenant_id == 0:
        raise ValidationError(
            code="model.context_missing",
            message="No active workspace in request context",
        )
    return tenant_id


# ── Provider catalog ────────────────────────────────────────────────


@router.get("/providers", response_model=ProviderListEnvelope)
async def list_model_providers(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    model_type: str | None = Query(default=None),
) -> ProviderListEnvelope:
    """Return the provider catalog filtered by model type.

    The upstream handler returns a list of provider descriptors built
    from ``provider.List`` / ``provider.ListByModelType``. This build
    ships a static catalog (no inference-provider land yet) behind
    the same wire shape so the implementation can be swapped later
    without breaking callers.
    """
    providers = _static_providers(model_type)
    return provider_list_envelope(providers)


def _static_providers(model_type: str | None) -> list[ProviderTypeMeta]:
    """Build the static provider catalog used by ``GET /models/providers``.

    Returns the full ``PROVIDER_CATALOG`` (or the subset whose
    ``model_types`` includes the requested frontend alias) so the wire
    shape mirrors ``docs/api/model.md``. The implementation can later
    swap in inference-provider metadata without changing the response
    model.
    """
    return filter_providers(PROVIDER_CATALOG, model_type=model_type)


# ── CRUD ─────────────────────────────────────────────────────────────


@router.post("", response_model=ModelEnvelope, status_code=201)
async def create_model(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    body: CreateModelRequest,
    model_service: ModelServiceDep,
    tenant_id: _PrincipalTenant,
) -> ModelEnvelope:
    """Create a model; status defaults to ``active``."""
    tenant_id = _require_tenant(tenant_id)
    info = await model_service.create_model(tenant_id=tenant_id, body=body)
    return model_envelope(info)


@router.get("", response_model=ModelListEnvelope)
async def list_models(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    model_service: ModelServiceDep,
    tenant_id: _PrincipalTenant,
    type: str | None = Query(default=None),
    source: str | None = Query(default=None),
    include_builtin: bool = Query(default=True),
) -> ModelListEnvelope:
    """List every model visible to the active workspace."""
    tenant_id = _require_tenant(tenant_id)
    infos = await model_service.list_models(
        tenant_id=tenant_id,
        model_type=type,
        source=source,
        include_builtin=include_builtin,
    )
    return model_list_envelope(infos)


@router.get("/{model_id}", response_model=ModelEnvelope)
async def get_model(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    model_id: str,
    model_service: ModelServiceDep,
    tenant_id: _PrincipalTenant,
) -> ModelEnvelope:
    """Return one model; ``model.not_found`` when absent."""
    tenant_id = _require_tenant(tenant_id)
    info = await model_service.get_model(tenant_id=tenant_id, model_id=model_id)
    return model_envelope(info)


@router.put("/{model_id}", response_model=ModelEnvelope)
async def update_model(
    _auth: AuthDep,
    _admin_or_system: RoleAdminDep,
    model_id: str,
    body: UpdateModelRequest,
    model_service: ModelServiceDep,
    tenant_id: _PrincipalTenant,
    is_system_admin: _PrincipalSystemAdmin,
) -> ModelEnvelope:
    """Update a model; built-ins require a system admin (enforced in service)."""
    tenant_id = _require_tenant(tenant_id)
    info = await model_service.update_model(
        tenant_id=tenant_id,
        model_id=model_id,
        body=body,
        is_system_admin=is_system_admin,
    )
    return model_envelope(info)


@router.delete("/{model_id}", response_model=DeleteModelResponse)
async def delete_model(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    model_id: str,
    model_service: ModelServiceDep,
    tenant_id: _PrincipalTenant,
) -> DeleteModelResponse:
    """Delete a model; idempotent for unknown ids, refuses built-ins."""
    tenant_id = _require_tenant(tenant_id)
    await model_service.delete_model(tenant_id=tenant_id, model_id=model_id)
    return DeleteModelResponse(success=True, message="Model deleted")


# ── Debug probe ──────────────────────────────────────────────────────


@router.post("/{model_id}/debug", response_model=ModelDebugEnvelope)
async def debug_model(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    model_id: str,
    model_service: ModelServiceDep,
    tenant_id: _PrincipalTenant,
    input: str = Form(default=""),
    options: str = Form(default=""),
    documents: str = Form(default=""),
    file: Annotated[UploadFile | None, File()] = None,
) -> ModelDebugEnvelope:
    """Probe a saved model end-to-end.

    Mirrors the upstream ``DebugModel`` handler.
    The real implementation dispatches by ``model.type`` to the right
    inference client (chat / embedding / rerank / vllm / asr); the
    inference providers are not implemented yet. For now this endpoint
    returns a static response describing the request envelope so
    callers can wire their UI against the same shape.
    """
    tenant_id = _require_tenant(tenant_id)
    if len(input.encode("utf-8")) > _DEBUG_MAX_INPUT_BYTES:
        raise ValidationError(
            code="model.debug_input_too_long",
            message="input is too long",
        )
    # Resolve the model so a missing id surfaces as 404 (mirrors the
    # upstream behaviour) instead of an opaque debug probe failure.
    info = await model_service.get_model(tenant_id=tenant_id, model_id=model_id)
    started = time.monotonic()
    data: dict[str, JsonValue] = {
        "ok": False,
        "elapsed_ms": int((time.monotonic() - started) * 1000),
        "request": {
            "model_id": info.id,
            "model_name": info.name,
            "model_type": info.type,
            "source": info.source,
            "input": input,
            "options": options,
            "documents": documents,
            "file": ({"name": file.filename, "size": file.size} if file is not None else None),
        },
        "raw_response": None,
        "observations": {
            "probe": "stub",
            "reason": (
                "debug probe is wired but the inference-provider dispatch is not yet implemented"
            ),
        },
        "error": "debug probe is a no-op until inference providers land",
    }
    return ModelDebugEnvelope(success=True, data=data)


__all__ = ["router"]
