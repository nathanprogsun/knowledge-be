"""Knowledge-base-domain request-scoped service factories.

See ``src.core.auth.factory`` for the pattern: repos are built per
request on the shared ``AsyncSession``; ``web`` never imports ``db``.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.dao.user_kb_pin_repository import UserKBPinRepository


def build_kb_service(session: AsyncSession) -> KBService:
    """Per-request ``KBService`` with a fresh repository."""
    return KBService(
        kb_repo=KnowledgeBaseRepository(session),
        pin_repo=UserKBPinRepository(session),
    )


__all__ = ["build_kb_service"]
