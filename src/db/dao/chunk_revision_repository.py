"""Chunk-revision persistence — raw SQL only, no ORM.

Maps the revision-history methods declared in the upstream
``ChunkRevisionRepository`` interface:

``CreateChunkRevision``
    append one immutable snapshot.
``ListChunkRevisions``
    snapshots of a chunk, newest first (``revision DESC``).
``GetChunkRevision``
    one historical snapshot, keyed by tenant + chunk + revision.

Every query is ``sqlalchemy.text()`` with named ``bindparams`` and
scopes by ``tenant_id`` so a caller can never read another workspace's
rows.

The atomic ``SaveChunkRevision`` (UPDATE the current ``chunks`` row +
INSERT a snapshot in one transaction, with optimistic locking on
``content_revision``) is deferred until the current-chunk model lands in
an earlier wave — it touches the ``chunks`` row, which this module does
not model.
"""

from __future__ import annotations

from sqlalchemy import text

from src.db.dao.generic_repository import GenericRepository
from src.db.models.chunk_revision import ChunkRevision


class ChunkRevisionRepository(GenericRepository[ChunkRevision]):
    """`chunk_revisions`-table SQL — immutable snapshots of chunk edits."""

    model_class = ChunkRevision

    async def create(self, revision: ChunkRevision) -> ChunkRevision:
        """Append one immutable snapshot and return the persisted row."""
        return await self.insert(revision)

    async def list_chunk_revisions(
        self,
        *,
        tenant_id: int,
        chunk_id: str,
    ) -> list[ChunkRevision]:
        """Return every snapshot of a chunk, newest revision first.

        Mirrors ``ListChunkRevisions`` (``ORDER BY revision DESC``).
        """
        stmt = text(
            "select * from chunk_revisions "
            "where tenant_id = :tenant_id and chunk_id = :chunk_id "
            "order by revision desc"
        ).bindparams(tenant_id=tenant_id, chunk_id=chunk_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def get_chunk_revision(
        self,
        *,
        tenant_id: int,
        chunk_id: str,
        revision: int,
    ) -> ChunkRevision | None:
        """Return one historical snapshot, or ``None`` when absent."""
        return await self.find_unique_by_column_values(
            {
                "tenant_id": tenant_id,
                "chunk_id": chunk_id,
                "revision": revision,
            },
        )


__all__ = ["ChunkRevisionRepository"]
