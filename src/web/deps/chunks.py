"""Chunk-domain FastAPI dependency factories.

One-line forwarders to ``src.core.knowledge.chunks.factory``:
repositories are assembled in ``core`` on the request-scoped
``AsyncSession`` so the request's reads and writes share one
transactional unit of work. ``web`` never imports ``db``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.core.knowledge.chunks.factory import (
    build_chunk_revision_service,
    build_chunk_service,
)
from src.core.knowledge.chunks.revisions import ChunkRevisionService
from src.core.knowledge.chunks.service.chunk_service import ChunkService
from src.web.deps.session import SessionDep


def get_chunk_service(session: SessionDep) -> ChunkService:
    """Build a per-request ``ChunkService`` on the shared session."""
    return build_chunk_service(session)


def get_chunk_revision_service(session: SessionDep) -> ChunkRevisionService:
    """Build a per-request ``ChunkRevisionService`` on the shared session."""
    return build_chunk_revision_service(session)


ChunkServiceDep = Annotated[ChunkService, Depends(get_chunk_service)]
ChunkRevisionServiceDep = Annotated[ChunkRevisionService, Depends(get_chunk_revision_service)]


__all__ = [
    "ChunkRevisionServiceDep",
    "ChunkServiceDep",
    "get_chunk_revision_service",
    "get_chunk_service",
]
