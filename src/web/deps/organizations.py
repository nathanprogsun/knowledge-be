"""Organization-domain FastAPI dependency factory.

One-line forwarder to ``src.core.organizations.factory``: repositories are
assembled in ``core`` on the request-scoped ``AsyncSession`` so the
request's reads and writes share one transactional unit of work.
``web`` never imports ``db``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.core.organizations.factory import (
    build_agent_share_service,
    build_organization_service,
    build_shared_resource_service,
)
from src.core.organizations.service.organization_service import OrganizationService
from src.core.organizations.service.shared_resource_service import (
    SharedResourceService,
)
from src.core.sharing.agent_share_service import AgentShareServiceImpl
from src.web.deps.session import SessionDep


def get_organization_service(session: SessionDep) -> OrganizationService:
    """Build a per-request ``OrganizationService`` on the shared session."""
    return build_organization_service(session)


def get_shared_resource_service(session: SessionDep) -> SharedResourceService:
    """Build a per-request ``SharedResourceService`` on the shared session."""
    return build_shared_resource_service(session)


def get_agent_share_service(session: SessionDep) -> AgentShareServiceImpl:
    """Build a per-request ``AgentShareServiceImpl`` on the shared session."""
    return build_agent_share_service(session)


OrganizationServiceDep = Annotated[
    OrganizationService, Depends(get_organization_service)
]

SharedResourceServiceDep = Annotated[
    SharedResourceService, Depends(get_shared_resource_service)
]

AgentShareServiceDep = Annotated[
    AgentShareServiceImpl, Depends(get_agent_share_service)
]


__all__ = [
    "AgentShareServiceDep",
    "OrganizationServiceDep",
    "SharedResourceServiceDep",
    "get_agent_share_service",
    "get_organization_service",
    "get_shared_resource_service",
]
