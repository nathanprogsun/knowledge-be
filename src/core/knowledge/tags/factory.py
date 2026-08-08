"""Tags-domain request-scoped service factory.

See ``src.core.tenants.factory`` for the pattern: the repositories are
built per request on the shared ``AsyncSession``; ``web`` never
imports ``db``.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.knowledge.tags.service.tag_service import TagService
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.dao.knowledge_tag_repository import TagRepository


def build_tag_service(session: AsyncSession) -> TagService:
    """Per-request ``TagService`` with fresh repositories."""
    return TagService(
        tag_repo=TagRepository(session),
        kb_repo=KnowledgeBaseRepository(session),
    )


__all__ = ["build_tag_service"]
