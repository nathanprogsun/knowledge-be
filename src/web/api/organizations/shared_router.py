"""Shared-resource HTTP endpoints — cross-tenant shared KBs and agents.

Registered by the app factory at the top level (no ``/organizations``
prefix): the upstream contract mounts these under ``/shared-*``, not
under the organization router. Role floors mirror the upstream route
guard — Viewer for the two read-only lists, Admin for the tenant-wide
hide-preference toggle.

Every endpoint reads the caller's workspace id from the request context;
a missing context fails closed with 401.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from src.common.exception import NotFoundError, UnauthorizedError
from src.core.contracts.organizations import (
    CreateAgentShareRequest,
    SetSharedAgentDisabledRequest,
    UpdateSharePermissionRequest,
)
from src.core.organizations.service.organization_service import OrganizationService
from src.web.api.organizations.shared_views import (
    AgentShareEnvelope,
    AgentShareListEnvelope,
    SharedAgentDisabledEnvelope,
    SharedAgentListEnvelope,
    SharedKnowledgeBaseListEnvelope,
    agent_share_envelope,
    agent_share_list_envelope,
    shared_agent_disabled_envelope,
    shared_agent_list_envelope,
    shared_knowledge_base_list_envelope,
)
from src.web.deps import AuthDep, RoleAdminDep, RoleViewerDep
from src.web.deps.context import get_tenant_id_dep, get_tenant_role_dep, get_user_id_dep
from src.web.deps.organizations import (
    AgentShareServiceDep,
    OrganizationServiceDep,
    SharedResourceServiceDep,
)

router = APIRouter(prefix="", tags=["organizations"])

# Function-arg-style principal deps.
_PrincipalTenant = Annotated[int, Depends(get_tenant_id_dep)]
_PrincipalRole = Annotated[str, Depends(get_tenant_role_dep)]
_PrincipalUser = Annotated[str | None, Depends(get_user_id_dep)]


def _require_tenant(tenant_id: int) -> int:
    """Return the active workspace id, or fail closed.

    Shared-resource visibility is workspace-scoped; without a workspace
    context there is no safe default, so this rejects rather than
    guessing.
    """
    if tenant_id == 0:
        raise UnauthorizedError(
            code="organization.tenant_context_missing",
            message="unauthorized: workspace context missing",
        )
    return tenant_id


@router.get(
    "/shared-knowledge-bases",
    response_model=SharedKnowledgeBaseListEnvelope,
    response_model_exclude_none=True,
)
async def list_shared_knowledge_bases(
    _auth: AuthDep,
    _role: RoleViewerDep,
    service: SharedResourceServiceDep,
    tenant_id: _PrincipalTenant,
    tenant_role: _PrincipalRole,
) -> SharedKnowledgeBaseListEnvelope:
    """List knowledge bases shared into the caller's workspace."""
    tenant_id = _require_tenant(tenant_id)
    items = await service.list_shared_knowledge_bases(
        tenant_id=tenant_id,
        caller_tenant_role=tenant_role,
    )
    return shared_knowledge_base_list_envelope(items)


@router.get(
    "/shared-agents",
    response_model=SharedAgentListEnvelope,
    response_model_exclude_none=True,
)
async def list_shared_agents(
    _auth: AuthDep,
    _role: RoleViewerDep,
    service: SharedResourceServiceDep,
    tenant_id: _PrincipalTenant,
    tenant_role: _PrincipalRole,
) -> SharedAgentListEnvelope:
    """List agents shared into the caller's workspace."""
    tenant_id = _require_tenant(tenant_id)
    items = await service.list_shared_agents(
        tenant_id=tenant_id,
        caller_tenant_role=tenant_role,
    )
    return shared_agent_list_envelope(items)


@router.post(
    "/shared-agents/disabled",
    response_model=SharedAgentDisabledEnvelope,
)
async def set_shared_agent_disabled_by_me(
    _auth: AuthDep,
    _role: RoleAdminDep,
    body: SetSharedAgentDisabledRequest,
    service: SharedResourceServiceDep,
    tenant_id: _PrincipalTenant,
) -> SharedAgentDisabledEnvelope:
    """Record or clear the workspace's hide preference for a shared agent."""
    tenant_id = _require_tenant(tenant_id)
    await service.set_shared_agent_disabled_by_me(
        tenant_id=tenant_id,
        agent_id=body.agent_id,
        disabled=body.disabled,
    )
    return shared_agent_disabled_envelope()


# ── Agent share management ─────────────────────────────────────────


@router.post(
    "/agents/{agent_id}/shares",
    response_model=AgentShareEnvelope,
    status_code=201,
)
async def share_agent(
    _auth: AuthDep,
    _role: RoleViewerDep,
    agent_id: str,
    body: CreateAgentShareRequest,
    service: AgentShareServiceDep,
    org_service: OrganizationServiceDep,
    tenant_id: _PrincipalTenant,
    user_id: _PrincipalUser,
) -> AgentShareEnvelope:
    """Share an owned agent into an organization (read-only grant)."""
    tenant_id = _require_tenant(tenant_id)
    share = await service.share_agent(
        agent_id=agent_id,
        organization_id=body.organization_id,
        user_id=user_id,
        tenant_id=tenant_id,
        permission=body.permission,
    )
    return agent_share_envelope(share, org_name=await _org_name(org_service, share.organization_id))


@router.get(
    "/agents/{agent_id}/shares",
    response_model=AgentShareListEnvelope,
)
async def list_agent_shares(
    _auth: AuthDep,
    _role: RoleViewerDep,
    agent_id: str,
    service: AgentShareServiceDep,
    org_service: OrganizationServiceDep,
    tenant_id: _PrincipalTenant,
) -> AgentShareListEnvelope:
    """List every share of an agent owned by the caller's workspace."""
    tenant_id = _require_tenant(tenant_id)
    shares = await service.list_shares_by_agent(agent_id=agent_id)
    org_names: dict[str, str] = {}
    for share in shares:
        if share.organization_id in org_names:
            continue
        name = await _org_name(org_service, share.organization_id)
        if name is not None:
            org_names[share.organization_id] = name
    return agent_share_list_envelope(shares, org_names=org_names)


@router.put(
    "/agents/{agent_id}/shares/{share_id}",
    response_model=AgentShareEnvelope,
)
async def update_agent_share_permission(
    _auth: AuthDep,
    _role: RoleViewerDep,
    agent_id: str,
    share_id: str,
    body: UpdateSharePermissionRequest,
    service: AgentShareServiceDep,
    org_service: OrganizationServiceDep,
    tenant_id: _PrincipalTenant,
    tenant_role: _PrincipalRole,
    user_id: _PrincipalUser,
) -> AgentShareEnvelope:
    """Update a share's permission (agent shares stay read-only)."""
    tenant_id = _require_tenant(tenant_id)
    await service.update_share_permission(
        share_id=share_id,
        permission=body.permission,
        user_id=user_id,
        tenant_id=tenant_id,
        tenant_role=tenant_role,
    )
    share = await service.get_share(share_id=share_id)
    return agent_share_envelope(share, org_name=await _org_name(org_service, share.organization_id))


@router.delete(
    "/agents/{agent_id}/shares/{share_id}",
    response_model=AgentShareEnvelope,
)
async def remove_agent_share(
    _auth: AuthDep,
    _role: RoleViewerDep,
    agent_id: str,
    share_id: str,
    service: AgentShareServiceDep,
    org_service: OrganizationServiceDep,
    tenant_id: _PrincipalTenant,
    tenant_role: _PrincipalRole,
    user_id: _PrincipalUser,
) -> AgentShareEnvelope:
    """Revoke an agent share."""
    tenant_id = _require_tenant(tenant_id)
    share = await service.get_share(share_id=share_id)
    await service.remove_share(
        share_id=share_id,
        user_id=user_id,
        tenant_id=tenant_id,
        tenant_role=tenant_role,
    )
    return agent_share_envelope(share, org_name=await _org_name(org_service, share.organization_id))


async def _org_name(org_service: OrganizationService, organization_id: str) -> str | None:
    """Resolve an organization's display name, tolerating a missing row."""
    try:
        org = await org_service.get_organization(id=organization_id)
    except NotFoundError:
        return None
    return org.name


__all__ = ["router"]
