"""Chunk revision history — domain types and read-side queries.

Maps the revision-history half of the upstream chunk service
(``ListChunkRevisions`` / ``GetChunkRevision``). ``ChunkRevisionInfo``
is the service-side projection of a ``chunk_revisions`` row; reads are
scoped by ``tenant_id`` so a caller can never see another workspace's
history.

Reverting a chunk replays a historical snapshot through the chunk edit
pipeline, which needs the current-chunk repository and update service
that land in an earlier wave; it is deferred here until those
dependencies are merged.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict

from src.common.exception import NotFoundError
from src.db.dao.chunk_revision_repository import ChunkRevisionRepository
from src.db.models.chunk_revision import ChunkRevision


class ChunkRevisionInfo(BaseModel):
    """Service-side projection of a ``chunk_revisions`` row."""

    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: int
    knowledge_base_id: str
    knowledge_id: str
    chunk_id: str
    revision: int
    content: str
    is_enabled: bool
    editor_id: str
    edit_source: str
    edited_at: datetime
    created_at: datetime

    @classmethod
    def map_from_db(cls, db: ChunkRevision) -> Self:
        """Project a storage ``ChunkRevision`` row to the service DTO."""
        return cls.model_validate(db.model_dump())


async def list_chunk_revisions(
    repo: ChunkRevisionRepository,
    *,
    tenant_id: int,
    chunk_id: str,
) -> list[ChunkRevisionInfo]:
    """Return the chunk's revision history, newest revision first."""
    rows = await repo.list_chunk_revisions(tenant_id=tenant_id, chunk_id=chunk_id)
    return [ChunkRevisionInfo.map_from_db(row) for row in rows]


async def get_chunk_revision(
    repo: ChunkRevisionRepository,
    *,
    tenant_id: int,
    chunk_id: str,
    revision: int,
) -> ChunkRevisionInfo:
    """Return one historical snapshot, raising ``NotFoundError`` when absent."""
    row = await repo.get_chunk_revision(
        tenant_id=tenant_id,
        chunk_id=chunk_id,
        revision=revision,
    )
    if row is None:
        raise NotFoundError(
            code="chunk.revision_not_found",
            message=f"chunk revision {revision} not found",
        )
    return ChunkRevisionInfo.map_from_db(row)


class ChunkRevisionService:
    """Request-scoped chunk-revision history queries.

    Wraps the module-level read functions over a per-request
    ``ChunkRevisionRepository`` so the web layer can depend on a built
    service (via ``build_chunk_revision_service``) instead of importing
    the repository itself.
    """

    def __init__(self, repo: ChunkRevisionRepository) -> None:
        self._repo = repo

    async def list_revisions(self, *, tenant_id: int, chunk_id: str) -> list[ChunkRevisionInfo]:
        """Return the chunk's revision history, newest revision first."""
        return await list_chunk_revisions(self._repo, tenant_id=tenant_id, chunk_id=chunk_id)

    async def get_revision(
        self,
        *,
        tenant_id: int,
        chunk_id: str,
        revision: int,
    ) -> ChunkRevisionInfo:
        """Return one historical snapshot, raising ``NotFoundError`` when absent."""
        return await get_chunk_revision(
            self._repo,
            tenant_id=tenant_id,
            chunk_id=chunk_id,
            revision=revision,
        )


__all__ = [
    "ChunkRevisionInfo",
    "ChunkRevisionService",
    "get_chunk_revision",
    "list_chunk_revisions",
]
