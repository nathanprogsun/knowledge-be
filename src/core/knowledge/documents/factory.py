"""Documents-domain request-scoped service factory.

See ``src.core.auth.factory`` for the pattern: repos are built per
request on the shared ``AsyncSession``; ``web`` never imports ``db``.
The web layer consumes this via a ``Depends`` forwarder.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.knowledge.documents.service.knowledge_service import KnowledgeService
from src.db.dao.knowledge_repository import KnowledgeRepository


def build_knowledge_service(session: AsyncSession) -> KnowledgeService:
    """Per-request ``KnowledgeService`` with a fresh repository."""
    return KnowledgeService(
        knowledge_repo=KnowledgeRepository(session),
    )


__all__ = ["build_knowledge_service"]
