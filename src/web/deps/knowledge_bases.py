"""Knowledge-base domain FastAPI dependency factory.

One-line forwarder to ``src.core.knowledge.knowledge_bases.factory``:
repositories are assembled in ``core`` on the request-scoped
``AsyncSession`` so a request's reads and writes share one
transactional unit of work. ``web`` never imports ``db``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.core.knowledge.knowledge_bases.factory import build_kb_service
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.web.deps.session import SessionDep


def get_kb_service(session: SessionDep) -> KBService:
    """Build a per-request ``KBService`` on the shared session."""
    return build_kb_service(session)


KBServiceDep = Annotated[KBService, Depends(get_kb_service)]

__all__ = ["KBServiceDep", "get_kb_service"]
