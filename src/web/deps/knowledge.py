"""Knowledge-document domain FastAPI dependency factory.

One-line forwarder to ``src.core.knowledge.documents.factory`` so the
FAQ views can resolve a knowledge base's FAQ container without importing
``db``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.core.knowledge.documents.factory import build_knowledge_service
from src.core.knowledge.documents.service.knowledge_service import KnowledgeService
from src.web.deps.session import SessionDep


def get_knowledge_service(session: SessionDep) -> KnowledgeService:
    """Build a per-request ``KnowledgeService`` on the shared session."""
    return build_knowledge_service(session)


KnowledgeServiceDep = Annotated[KnowledgeService, Depends(get_knowledge_service)]


__all__ = ["KnowledgeServiceDep", "get_knowledge_service"]
