"""VectorStore-domain FastAPI dependency factories.

One-line forwarders to ``src.core.infra.vector_stores.factory``:
repositories are assembled in ``core`` on the request-scoped
``AsyncSession`` so the request's reads and writes share one
transactional unit of work. ``web`` never imports ``db``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.core.infra.vector_stores.factory import build_vector_store_service
from src.core.infra.vector_stores.service.vector_store_service import VectorStoreService
from src.web.deps.session import SessionDep


def get_vector_store_service(session: SessionDep) -> VectorStoreService:
    """Build a per-request ``VectorStoreService`` on the shared session."""
    return build_vector_store_service(session)


VectorStoreServiceDep = Annotated[
    VectorStoreService,
    Depends(get_vector_store_service),
]


__all__ = [
    "VectorStoreServiceDep",
    "get_vector_store_service",
]
