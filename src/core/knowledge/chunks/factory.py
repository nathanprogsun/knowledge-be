"""Chunk-domain request-scoped service factory.

See ``src.core.tenants.factory`` for the pattern: the repository is built
per request on the shared ``AsyncSession``; ``web`` never imports ``db``.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.knowledge.chunks.revisions import ChunkRevisionService
from src.core.knowledge.chunks.service.chunk_service import (
    ChunkIndexSyncer,
    ChunkService,
)
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.chunk_revision_repository import ChunkRevisionRepository


def build_chunk_service(
    session: AsyncSession,
    *,
    index_syncer: ChunkIndexSyncer | None = None,
) -> ChunkService:
    """Per-request ``ChunkService`` with a fresh chunk repository.

    ``index_syncer`` is the retrieval-index re-embedding hook; pass it once
    the retrieval engine is wired. Until then the service settles
    ``index_status`` without it.
    """
    return ChunkService(
        chunk_repo=ChunkRepository(session),
        index_syncer=index_syncer,
    )


def build_chunk_revision_service(session: AsyncSession) -> ChunkRevisionService:
    """Per-request revision-history service with a fresh revision repository."""
    return ChunkRevisionService(ChunkRevisionRepository(session))


__all__ = ["build_chunk_revision_service", "build_chunk_service"]
