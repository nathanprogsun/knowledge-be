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

from src.common.exception import UnauthorizedError
from src.core.contracts.organizations import SetSharedAgentDisabledRequest
from src.web.deps import AuthDep, RoleAdminDep, RoleViewerDep
from src.web.deps.context import get_tenant_id_dep, get_tenant_role_dep
from src.web.deps.organizations import SharedResourceServiceDep
from src.web.api.organizations.shared_views import (
    SharedAgentDisabledEnvelope,
    SharedAgentListEnvelope,
    SharedKnowledgeBaseListEnvelope,
    shared_agent_disabled_envelope,
    shared_agent_list_envelope,
    shared_knowledge_base_list_envelope,
)

router = APIRouter(prefix="", tags=["organizations"])

# Function-arg-style principal deps.
_PrincipalTenant = Annotated[int, Depends(get_tenant_id_dep)]
_PrincipalRole = Annotated[str, Depends(get_tenant_role_dep)]


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


__all__ = ["router"]
