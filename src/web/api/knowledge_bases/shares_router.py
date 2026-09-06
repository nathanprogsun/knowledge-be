"""HTTP endpoints for knowledge-base share management.

Registered next to the knowledge-base router. Mutations require
Contributor+ at the HTTP floor; the service still rejects a contributor
who is not the creator, a tenant admin, or the original sharer.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from src.common.exception import NotFoundError, UnauthorizedError
from src.core.contracts.organizations import (
    CreateKnowledgeBaseShareRequest,
    UpdateKnowledgeBaseShareRequest,
)
from src.core.organizations.service.organization_service import OrganizationService
from src.core.sharing.types import KnowledgeBaseShareInfo
from src.web.api.organizations.shared_views import (
    KnowledgeBaseShareEnvelope,
    KnowledgeBaseShareListEnvelope,
    kb_share_envelope,
    kb_share_list_envelope,
)
from src.web.deps import AuthDep, RoleContributorDep, RoleViewerDep
from src.web.deps.context import get_tenant_id_dep, get_tenant_role_dep, get_user_id_dep
from src.web.deps.organizations import OrganizationServiceDep
from src.web.deps.sharing import KBShareServiceDep

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])

_PrincipalTenant = Annotated[int, Depends(get_tenant_id_dep)]
_PrincipalRole = Annotated[str, Depends(get_tenant_role_dep)]
_PrincipalUser = Annotated[str | None, Depends(get_user_id_dep)]


def _require_tenant(tenant_id: int) -> int:
    if tenant_id == 0:
        raise UnauthorizedError(
            code="knowledge_base.tenant_context_missing",
            message="unauthorized: workspace context missing",
        )
    return tenant_id


@router.post(
    "/{id}/shares",
    response_model=KnowledgeBaseShareEnvelope,
    status_code=201,
)
async def share_knowledge_base(
    _auth: AuthDep,
    _role: RoleContributorDep,
    id: str,
    body: CreateKnowledgeBaseShareRequest,
    service: KBShareServiceDep,
    org_service: OrganizationServiceDep,
    tenant_id: _PrincipalTenant,
    tenant_role: _PrincipalRole,
    user_id: _PrincipalUser,
) -> KnowledgeBaseShareEnvelope:
    """Share an owned knowledge base into an organization."""
    tenant_id = _require_tenant(tenant_id)
    share = await service.share_knowledge_base(
        knowledge_base_id=id,
        organization_id=body.organization_id,
        user_id=user_id or "",
        tenant_id=tenant_id,
        tenant_role=tenant_role,
        permission=body.permission,
    )
    return kb_share_envelope(share, org_name=await _org_name(org_service, share.organization_id))


@router.get(
    "/{id}/shares",
    response_model=KnowledgeBaseShareListEnvelope,
)
async def list_knowledge_base_shares(
    _auth: AuthDep,
    _role: RoleViewerDep,
    id: str,
    service: KBShareServiceDep,
    org_service: OrganizationServiceDep,
    tenant_id: _PrincipalTenant,
    tenant_role: _PrincipalRole,
) -> KnowledgeBaseShareListEnvelope:
    """List live grants of a knowledge base owned by the caller."""
    tenant_id = _require_tenant(tenant_id)
    shares = await service.list_shares_by_knowledge_base(
        knowledge_base_id=id,
        tenant_id=tenant_id,
        tenant_role=tenant_role,
    )
    return kb_share_list_envelope(shares, org_names=await _org_names(org_service, shares))


@router.put(
    "/{id}/shares/{share_id}",
    response_model=KnowledgeBaseShareEnvelope,
)
async def update_knowledge_base_share(
    _auth: AuthDep,
    _role: RoleContributorDep,
    id: str,
    share_id: str,
    body: UpdateKnowledgeBaseShareRequest,
    service: KBShareServiceDep,
    org_service: OrganizationServiceDep,
    tenant_id: _PrincipalTenant,
    tenant_role: _PrincipalRole,
    user_id: _PrincipalUser,
) -> KnowledgeBaseShareEnvelope:
    """Update a stored knowledge-base share grant."""
    tenant_id = _require_tenant(tenant_id)
    await service.update_share_permission(
        knowledge_base_id=id,
        share_id=share_id,
        permission=body.permission,
        user_id=user_id or "",
        tenant_id=tenant_id,
        tenant_role=tenant_role,
    )
    share = await service.get_share(share_id=share_id)
    return kb_share_envelope(share, org_name=await _org_name(org_service, share.organization_id))


@router.delete(
    "/{id}/shares/{share_id}",
    response_model=KnowledgeBaseShareEnvelope,
)
async def remove_knowledge_base_share(
    _auth: AuthDep,
    _role: RoleContributorDep,
    id: str,
    share_id: str,
    service: KBShareServiceDep,
    org_service: OrganizationServiceDep,
    tenant_id: _PrincipalTenant,
    tenant_role: _PrincipalRole,
    user_id: _PrincipalUser,
) -> KnowledgeBaseShareEnvelope:
    """Revoke a knowledge-base share grant."""
    tenant_id = _require_tenant(tenant_id)
    share = await service.get_share(share_id=share_id)
    await service.remove_share(
        knowledge_base_id=id,
        share_id=share_id,
        user_id=user_id or "",
        tenant_id=tenant_id,
        tenant_role=tenant_role,
    )
    return kb_share_envelope(share, org_name=await _org_name(org_service, share.organization_id))


async def _org_name(org_service: OrganizationService, organization_id: str) -> str | None:
    try:
        org = await org_service.get_organization(id=organization_id)
    except NotFoundError:
        return None
    return org.name


async def _org_names(
    org_service: OrganizationService,
    shares: list[KnowledgeBaseShareInfo],
) -> dict[str, str]:
    names: dict[str, str] = {}
    for share in shares:
        if share.organization_id in names:
            continue
        name = await _org_name(org_service, share.organization_id)
        if name is not None:
            names[share.organization_id] = name
    return names


__all__ = ["router"]
