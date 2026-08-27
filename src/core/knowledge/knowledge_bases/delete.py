"""Knowledge-base cascade delete — standalone sub-service module.

Faithful port of the upstream async knowledge-base-delete worker
semantics: given a tenant and a knowledge-base id, soft-delete every
chunk and document row beneath it and offer a hook for the deferred
heavy cleanup (vector store, physical files, graph data). The web layer
composes this module after the knowledge-base row has been soft-deleted;
it passes the repositories built on the shared ``AsyncSession``.

Scope and deferred seams
------------------------

- The knowledge-base row itself is not touched here: the caller
  soft-deletes it first (mirroring the upstream synchronous delete
  path) and snapshots the vector-store binding before the row is
  hidden from reads. That snapshot is passed back in as
  ``vector_store_id`` so the index cleanup can still resolve the store.
- Chunk and document rows are soft-deleted tenant-scoped. Chunk deletes
  are best-effort (a failing sibling query is logged and the cascade
  continues, matching the upstream worker); the document batch delete
  is the durable step and its failure aborts the cascade for a
  caller-side retry.
- Vector-store / file / graph cleanup runs through the optional
  ``index_cleanup`` hook BEFORE any row is written, so an aborted
  cleanup leaves the relational tables untouched and a retry resumes
  cleanly. Without a hook the module only cascades the relational
  soft delete.
- Enqueue / retry orchestration of the async delete task belongs to the
  worker layer; this module performs the synchronous cascade within the
  caller's async context.

Errors use the sanctioned domain hierarchy: a blank knowledge-base id
or a non-positive tenant raises ``ValidationError`` with the
knowledge-base domain codes; repository and hook failures propagate
unchanged for the caller to classify.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from src.common.exception import ValidationError
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.knowledge import Document

logger = logging.getLogger(__name__)

_TENANT_REQUIRED_CODE = "knowledge_base.tenant_required"
_KB_ID_REQUIRED_CODE = "knowledge_base.id_required"


def _require_tenant_id(tenant_id: int) -> None:
    """Reject a non-positive tenant id at the service boundary."""
    if not isinstance(tenant_id, int) or tenant_id <= 0:
        raise ValidationError(
            code=_TENANT_REQUIRED_CODE,
            message="tenant ID is required",
        )


def _require_knowledge_base_id(knowledge_base_id: str) -> None:
    """Reject a blank knowledge-base id at the service boundary."""
    if not knowledge_base_id.strip():
        raise ValidationError(
            code=_KB_ID_REQUIRED_CODE,
            message="knowledge base ID cannot be empty",
        )


class KBIndexCleanup(Protocol):
    """Deferred heavy cleanup (vector store, files, graph) for a deleted KB.

    A plain async callable satisfies the protocol: the storage / worker
    domain wires a real implementation later. Implementations receive the
    full set of documents being removed plus the vector-store snapshot
    taken before the knowledge-base row was soft-deleted. They must be
    idempotent so a caller-side retry after a partial failure is safe.
    """

    async def __call__(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        knowledge: Sequence[Document],
        vector_store_id: str | None,
    ) -> None: ...


@dataclass(frozen=True)
class KBDeleteResult:
    """Observability summary of one cascade delete pass."""

    knowledge_ids: tuple[str, ...]
    deleted_chunks: int
    deleted_knowledge: int
    vector_store_id: str | None = None


async def process_kb_delete(
    *,
    tenant_id: int,
    knowledge_base_id: str,
    knowledge_repo: KnowledgeRepository,
    chunk_repo: ChunkRepository,
    vector_store_id: str | None = None,
    index_cleanup: KBIndexCleanup | None = None,
    now: datetime | None = None,
) -> KBDeleteResult:
    """Cascade soft-delete a knowledge base's documents and chunks.

    Ordering mirrors the upstream worker: the index-cleanup hook (when
    supplied) runs first while the relational rows are still live, then
    chunks are soft-deleted per document (best-effort), then the
    document rows themselves (the durable step). A knowledge base with
    no live documents is a noop. A raising hook aborts before any row
    is written; a failing document batch delete aborts at the durable
    step — in both cases the caller can retry the whole pass.
    """
    _require_tenant_id(tenant_id)
    _require_knowledge_base_id(knowledge_base_id)
    stamp = now if now is not None else datetime.now(UTC)

    knowledge = await knowledge_repo.list_by_knowledge_base(tenant_id, knowledge_base_id)
    knowledge_ids = tuple(row.id for row in knowledge)

    if not knowledge_ids:
        return KBDeleteResult(
            knowledge_ids=(),
            deleted_chunks=0,
            deleted_knowledge=0,
            vector_store_id=vector_store_id,
        )

    if index_cleanup is not None:
        await index_cleanup(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            knowledge=knowledge,
            vector_store_id=vector_store_id,
        )

    deleted_chunks = 0
    for row in knowledge:
        try:
            deleted_chunks += await chunk_repo.delete_by_knowledge_id(
                tenant_id=tenant_id,
                knowledge_id=row.id,
                now=stamp,
            )
        except Exception:
            logger.warning(
                "kb delete: chunk sweep for document %s (kb=%s) failed; continuing",
                row.id,
                knowledge_base_id,
            )

    deleted_knowledge = await knowledge_repo.soft_delete_list(
        tenant_id=tenant_id,
        ids=list(knowledge_ids),
        now=stamp,
    )

    return KBDeleteResult(
        knowledge_ids=knowledge_ids,
        deleted_chunks=deleted_chunks,
        deleted_knowledge=deleted_knowledge,
        vector_store_id=vector_store_id,
    )


__all__ = ["KBDeleteResult", "KBIndexCleanup", "process_kb_delete"]
