"""VectorStore-domain request-scoped service factories.

See ``src.core.auth.factory`` for the pattern: repos are built per
request on the shared ``AsyncSession``; ``web`` never imports ``db``.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.infra.vector_stores.service.vector_store_service import VectorStoreService
from src.db.dao.vector_store_repository import VectorStoreRepository


def build_vector_store_service(session: AsyncSession) -> VectorStoreService:
    """Per-request ``VectorStoreService`` with a fresh repo."""
    return VectorStoreService(
        vector_store_repo=VectorStoreRepository(session),
    )


__all__ = ["build_vector_store_service"]
