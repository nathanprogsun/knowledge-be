"""Organization-domain request-scoped service factory.

See ``src.core.knowledge.knowledge_bases.factory`` for the pattern: the
repositories are built per request on the shared ``AsyncSession``;
``web`` never imports ``db``. A single service instance owns its
repositories so a request's reads and writes share one transactional
unit of work.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.organizations.service.organization_service import OrganizationService
from src.core.organizations.service.shared_resource_service import (
    SharedResourceService,
)
from src.db.dao.agent_share_repository import AgentShareRepository
from src.db.dao.custom_agent_repository import CustomAgentRepository
from src.db.dao.kb_share_repository import KBShareRepository
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.dao.organization_repository import (
    OrganizationJoinRequestRepository,
    OrganizationMemberRepository,
    OrganizationRepository,
)
from src.db.dao.tenant_disabled_shared_agent_repository import (
    TenantDisabledSharedAgentRepository,
)
from src.db.dao.users_repository import UserRepository
from src.db.dao.web_search_provider_repository import WebSearchProviderRepository


def build_organization_service(session: AsyncSession) -> OrganizationService:
    """Per-request ``OrganizationService`` with fresh repositories."""
    return OrganizationService(
        org_repo=OrganizationRepository(session),
        member_repo=OrganizationMemberRepository(session),
        join_request_repo=OrganizationJoinRequestRepository(session),
    )


def build_shared_resource_service(session: AsyncSession) -> SharedResourceService:
    """Per-request ``SharedResourceService`` with fresh repositories.

    Cross-tenant shared KB / agent reads and the per-tenant hide
    preference; shares one request session with the organization service.
    """
    return SharedResourceService(
        org_repo=OrganizationRepository(session),
        member_repo=OrganizationMemberRepository(session),
        kb_share_repo=KBShareRepository(session),
        agent_share_repo=AgentShareRepository(session),
        disabled_repo=TenantDisabledSharedAgentRepository(session),
        kb_repo=KnowledgeBaseRepository(session),
        agent_repo=CustomAgentRepository(session),
        user_repo=UserRepository(session),
        web_search_provider_repo=WebSearchProviderRepository(session),
    )


__all__ = [
    "build_organization_service",
    "build_shared_resource_service",
]
