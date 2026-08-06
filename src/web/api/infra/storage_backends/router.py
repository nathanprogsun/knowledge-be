"""Storage-backend HTTP endpoints - registry CRUD, probes, default binding.

Reads are Viewer+, everything that mutates the registry or dials an
external service is Admin+. Every endpoint carries the global ``AuthDep`` in addition to its
role gate.

Route order matters - ``/types`` and ``/test`` are declared before
``/{id}`` so the literal segments are not captured as an id.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from src.common.exception import ValidationError
from src.core.contracts.infra import (
    CreateStorageBackendRequest,
    StorageBackendListResponse,
    TestStorageBackendRequest,
    UpdateStorageBackendRequest,
)
from src.web.api.infra.storage_backends.views import (
    StorageBackendAckEnvelope,
    StorageBackendEnvelope,
    StorageConnectivityEnvelope,
    StorageProviderTypesEnvelope,
    backend_envelope,
    backend_list_response,
    config_from_contract,
    connectivity_envelope,
)
from src.web.deps import AuthDep, RoleAdminDep, RoleViewerDep
from src.web.deps.context import get_tenant_id_dep
from src.web.deps.infra_storage_backends import StorageBackendServiceDep

# Function-arg-style principal dep alias.
_PrincipalTenant = Annotated[int, Depends(get_tenant_id_dep)]


router = APIRouter(prefix="/storage-backends", tags=["storage-backends"])


def _require_tenant(tenant_id: int) -> int:
    """Return the active workspace id, or raise when the context is empty."""
    if tenant_id == 0:
        raise ValidationError(
            code="auth.tenant_context_missing",
            message="Workspace context missing",
        )
    return tenant_id


# ── Provider catalogue (Viewer) ───────────────────────────────────────


@router.get("/types", response_model=StorageProviderTypesEnvelope)
async def list_storage_provider_types(
    _auth: AuthDep,
    _role: RoleViewerDep,
    service: StorageBackendServiceDep,
) -> StorageProviderTypesEnvelope:
    """List the provider types allowed by ``STORAGE_ALLOW_LIST``."""
    return StorageProviderTypesEnvelope(success=True, data=service.list_provider_types())


# ── Connectivity probe on an unsaved config (Admin) ───────────────────


@router.post("/test", response_model=StorageConnectivityEnvelope)
async def test_storage_backend_config(
    _auth: AuthDep,
    _role: RoleAdminDep,
    body: TestStorageBackendRequest,
    service: StorageBackendServiceDep,
    tenant_id: _PrincipalTenant,
) -> StorageConnectivityEnvelope:
    """Probe a configuration without persisting it.

    A connectivity failure answers 200 with ``success=false`` and a
    sanitized message; a validation failure (SSRF block, missing field)
    answers the same way - Go's ``TestRaw`` keeps the HTTP status at
    200 and reports the error in the body.
    """
    try:
        result = await service.test_config(
            tenant_id=_require_tenant(tenant_id),
            name=body.name,
            provider=body.provider,
            config=config_from_contract(body.config),
        )
    except ValidationError as exc:
        return StorageConnectivityEnvelope(success=False, error=exc.message)
    return connectivity_envelope(result)


# ── Registry CRUD ─────────────────────────────────────────────────────


@router.post("", response_model=StorageBackendEnvelope, status_code=201)
async def create_storage_backend(
    _auth: AuthDep,
    _role: RoleAdminDep,
    body: CreateStorageBackendRequest,
    service: StorageBackendServiceDep,
    tenant_id: _PrincipalTenant,
) -> StorageBackendEnvelope:
    """Register a storage instance; validated and probed before insert."""
    info = await service.create(
        tenant_id=_require_tenant(tenant_id),
        name=body.name,
        provider=body.provider,
        config=config_from_contract(body.config),
        status=body.status or "",
    )
    return backend_envelope(info)


@router.get("", response_model=StorageBackendListResponse)
async def list_storage_backends(
    _auth: AuthDep,
    _role: RoleViewerDep,
    service: StorageBackendServiceDep,
    tenant_id: _PrincipalTenant,
) -> StorageBackendListResponse:
    """List the workspace's backends with credentials masked."""
    result = await service.list_backends(_require_tenant(tenant_id))
    return backend_list_response(result)


@router.get("/{id}", response_model=StorageBackendEnvelope)
async def get_storage_backend(
    _auth: AuthDep,
    _role: RoleViewerDep,
    id: str,
    service: StorageBackendServiceDep,
    tenant_id: _PrincipalTenant,
) -> StorageBackendEnvelope:
    """Return one backend of the workspace with credentials masked."""
    info = await service.get_backend(tenant_id=_require_tenant(tenant_id), id=id)
    return backend_envelope(info)


@router.put("/{id}", response_model=StorageBackendEnvelope)
async def update_storage_backend(
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    body: UpdateStorageBackendRequest,
    service: StorageBackendServiceDep,
    tenant_id: _PrincipalTenant,
) -> StorageBackendEnvelope:
    """Update a backend's name, credentials or status.

    ``provider`` in the body is ignored - it is immutable, as is the
    physical location. Redacted secret placeholders keep the stored
    credentials.
    """
    info = await service.update(
        tenant_id=_require_tenant(tenant_id),
        id=id,
        name=body.name,
        config=config_from_contract(body.config) if body.config is not None else None,
        status=body.status,
    )
    return backend_envelope(info)


@router.delete("/{id}", response_model=StorageBackendAckEnvelope)
async def delete_storage_backend(
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    service: StorageBackendServiceDep,
    tenant_id: _PrincipalTenant,
) -> StorageBackendAckEnvelope:
    """Soft-delete a backend that nothing depends on."""
    await service.delete(tenant_id=_require_tenant(tenant_id), id=id)
    return StorageBackendAckEnvelope(success=True)


# ── Probe / default binding on a saved backend (Admin) ────────────────


@router.post("/{id}/test", response_model=StorageConnectivityEnvelope)
async def test_storage_backend_by_id(
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    service: StorageBackendServiceDep,
    tenant_id: _PrincipalTenant,
) -> StorageConnectivityEnvelope:
    """Probe a saved backend using its stored credentials.

    A validation failure answers 200 with ``success=false`` (Go
    ``TestByID`` keeps the HTTP status at 200).
    """
    try:
        result = await service.test_backend(tenant_id=_require_tenant(tenant_id), id=id)
    except ValidationError as exc:
        return StorageConnectivityEnvelope(success=False, error=exc.message)
    return connectivity_envelope(result)


@router.put("/{id}/default", response_model=StorageBackendAckEnvelope)
async def set_default_storage_backend(
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    service: StorageBackendServiceDep,
    tenant_id: _PrincipalTenant,
) -> StorageBackendAckEnvelope:
    """Make an active backend the workspace default."""
    await service.set_default(tenant_id=_require_tenant(tenant_id), id=id)
    return StorageBackendAckEnvelope(success=True)


__all__ = ["router"]
