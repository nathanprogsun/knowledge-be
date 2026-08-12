"""Chunk revision history — domain types, reads, and the revert.

Maps the revision-history half of the upstream chunk service
(``ListChunkRevisions`` / ``GetChunkRevision`` / ``RevertDocumentChunk``).
``ChunkRevisionInfo`` is the service-side projection of a
``chunk_revisions`` row; reads are scoped by ``tenant_id`` so a caller
can never see another workspace's history.

Reverting a chunk (``revert_document_chunk``) replays a historical
snapshot through the guarded chunk-edit pipeline: it loads the snapshot
and re-applies its content and enabled state via the chunk service, so
the write stays revision-guarded and the retrieval index is settled by
the edit path's sync hook.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict

from src.common.exception import NotFoundError
from src.core.knowledge.chunks.service.chunk_service import ChunkService
from src.db.dao.chunk_revision_repository import ChunkRevisionRepository
from src.db.models.chunk import Chunk
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


async def revert_document_chunk(
    *,
    revision_repo: ChunkRevisionRepository,
    chunk_service: ChunkService,
    tenant_id: int,
    chunk_id: str,
    revision: int,
    expected_revision: int | None = None,
    last_editor_id: str,
) -> Chunk:
    """Replay a historical snapshot through the guarded edit pipeline.

    Loads the requested snapshot (raising ``NotFoundError`` when absent),
    then re-applies its content and enabled state via
    :meth:`ChunkService.update_document_chunk` so the write is
    revision-guarded and the retrieval index is settled by the edit
    path's sync hook. ``expected_revision`` is optional: ``None`` lets
    the edit path resolve the current revision, while an explicit value
    rejects a revert that has been overtaken by a concurrent edit.
    """
    snapshot = await get_chunk_revision(
        revision_repo,
        tenant_id=tenant_id,
        chunk_id=chunk_id,
        revision=revision,
    )
    return await chunk_service.update_document_chunk(
        tenant_id=tenant_id,
        chunk_id=chunk_id,
        content=snapshot.content,
        is_enabled=snapshot.is_enabled,
        expected_revision=expected_revision,
        last_editor_id=last_editor_id,
    )


__all__ = [
    "ChunkRevisionInfo",
    "ChunkRevisionService",
    "get_chunk_revision",
    "list_chunk_revisions",
    "revert_document_chunk",
]
