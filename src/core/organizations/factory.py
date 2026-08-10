"""Organization-domain request-scoped service factory.

See ``src.core.knowledge.knowledge_bases.factory`` for the pattern: the
three repositories are built per request on the shared ``AsyncSession``;
``web`` never imports ``db``. A single ``OrganizationService`` instance
owns all three repos so a request's reads and writes share one
transactional unit of work.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.organizations.service.organization_service import OrganizationService
from src.db.dao.organization_repository import (
    OrganizationJoinRequestRepository,
    OrganizationMemberRepository,
    OrganizationRepository,
)


def build_organization_service(session: AsyncSession) -> OrganizationService:
    """Per-request ``OrganizationService`` with fresh repositories."""
    return OrganizationService(
        org_repo=OrganizationRepository(session),
        member_repo=OrganizationMemberRepository(session),
        join_request_repo=OrganizationJoinRequestRepository(session),
    )


__all__ = ["build_organization_service"]
