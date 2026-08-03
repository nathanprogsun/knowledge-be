"""Tenant HTTP endpoints - workspace CRUD, listing and search."""

from __future__ import annotations

from fastapi import APIRouter, Query

from src.core.contracts.tenants import CreateTenantRequest, UpdateTenantRequest
from src.web.api.tenants.views import (
    DeleteTenantResponse,
    TenantEnvelope,
    TenantListEnvelope,
    tenant_envelope,
    tenant_list_envelope,
)
from src.web.deps import TenantServiceDep

router = APIRouter(prefix="/tenants", tags=["tenants"])

# Paging values are clamped rather than rejected: a page below 1 becomes
# 1, a page size outside [1, 100] becomes the default (20) or the cap (100).
_DEFAULT_PAGE = 1
_DEFAULT_PAGE_SIZE = 20
_MAX_PAGE_SIZE = 100


@router.post("", response_model=TenantEnvelope, status_code=201)
async def create_tenant(
    body: CreateTenantRequest,
    tenant_service: TenantServiceDep,
) -> TenantEnvelope:
    """Create a workspace; the status is assigned server-side."""
    info = await tenant_service.create_tenant(
        name=body.name,
        description=body.description,
        business=body.business or "",
        retriever_engines=_engines_payload(body),
        storage_quota=body.storage_quota,
    )
    return tenant_envelope(info)


@router.get("/all", response_model=TenantListEnvelope)
async def list_all_tenants(tenant_service: TenantServiceDep) -> TenantListEnvelope:
    """List every workspace, newest first."""
    return tenant_list_envelope(await tenant_service.list_tenants())


@router.get("/search", response_model=TenantListEnvelope)
async def search_tenants(
    tenant_service: TenantServiceDep,
    keyword: str | None = Query(default=None),
    tenant_id: int | None = Query(default=None),
    page: int = Query(default=_DEFAULT_PAGE),
    page_size: int = Query(default=_DEFAULT_PAGE_SIZE),
) -> TenantListEnvelope:
    """Search workspaces by id and/or keyword, with pagination."""
    page = page if page >= 1 else _DEFAULT_PAGE
    page_size = page_size if page_size >= 1 else _DEFAULT_PAGE_SIZE
    page_size = min(page_size, _MAX_PAGE_SIZE)
    infos, total = await tenant_service.search_tenants(
        keyword=keyword,
        tenant_id=tenant_id,
        page=page,
        page_size=page_size,
    )
    return tenant_list_envelope(infos, total=total, page=page, page_size=page_size)


@router.get("/{tenant_id}", response_model=TenantEnvelope)
async def get_tenant(tenant_id: int, tenant_service: TenantServiceDep) -> TenantEnvelope:
    """Return one workspace; an invalid id yields a path validation error."""
    return tenant_envelope(await tenant_service.get_tenant(tenant_id))


@router.put("/{tenant_id}", response_model=TenantEnvelope)
async def update_tenant(
    tenant_id: int,
    body: UpdateTenantRequest,
    tenant_service: TenantServiceDep,
) -> TenantEnvelope:
    """Update a workspace's name and/or description.

    Only those two columns are mutable through this endpoint, so a
    caller cannot escalate by sending a larger body.
    """
    info = await tenant_service.update_tenant(
        tenant_id,
        name=body.name,
        description=body.description.strip() if body.description is not None else None,
    )
    return tenant_envelope(info)


@router.delete("/{tenant_id}", response_model=DeleteTenantResponse)
async def delete_tenant(
    tenant_id: int,
    tenant_service: TenantServiceDep,
) -> DeleteTenantResponse:
    """Soft-delete a workspace; idempotent for unknown ids."""
    await tenant_service.delete_tenant(tenant_id)
    return DeleteTenantResponse(success=True, message="Workspace deleted successfully")


def _engines_payload(body: CreateTenantRequest) -> dict[str, object] | None:
    """Render the request's retriever engines as the stored JSON shape."""
    if body.retriever_engines is None:
        return None
    return body.retriever_engines.model_dump(mode="json")


__all__ = ["router"]
