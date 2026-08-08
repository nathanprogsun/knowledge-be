"""Knowledge-tag domain FastAPI dependency factories.

One-line forwarders to ``src.core.knowledge.tags.factory``: the
service is assembled in ``core`` on the request-scoped ``AsyncSession``
so the request's reads and writes share one transactional unit of work.
``web`` never imports ``db``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.core.knowledge.tags.factory import build_tag_service
from src.core.knowledge.tags.service.tag_service import TagService
from src.web.deps.session import SessionDep


def get_tag_service(session: SessionDep) -> TagService:
    """Build a per-request ``TagService`` on the shared session."""
    return build_tag_service(session)


TagServiceDep = Annotated[TagService, Depends(get_tag_service)]


__all__ = ["TagServiceDep", "get_tag_service"]
