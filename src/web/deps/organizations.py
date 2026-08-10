"""Organization-domain FastAPI dependency factory.

One-line forwarder to ``src.core.organizations.factory``: repositories are
assembled in ``core`` on the request-scoped ``AsyncSession`` so the
request's reads and writes share one transactional unit of work.
``web`` never imports ``db``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.core.organizations.factory import build_organization_service
from src.core.organizations.service.organization_service import OrganizationService
from src.web.deps.session import SessionDep


def get_organization_service(session: SessionDep) -> OrganizationService:
    """Build a per-request ``OrganizationService`` on the shared session."""
    return build_organization_service(session)


OrganizationServiceDep = Annotated[
    OrganizationService, Depends(get_organization_service)
]


__all__ = [
    "OrganizationServiceDep",
    "get_organization_service",
]
