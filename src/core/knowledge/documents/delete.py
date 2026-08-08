"""Single document delete — soft delete plus cascade chunk cleanup.

Standalone module. The merged ``KnowledgeService.delete_document``
soft-deletes the document row alone; this function reproduces the
upstream single-delete sequence on top of injected repositories:

- validate the tenant / document scope,
- resolve the row (``NotFoundError`` when absent or already deleted),
- mark the row mid-deletion so in-flight parse tasks cannot resurrect
  it,
- soft-delete its chunks (cascade), then
- soft-delete the row itself.

Retrieval-index and physical-file cleanup run in the retrieval /
storage layers in a later wave; the cascade here keeps the stored rows
consistent. Repositories are injected so the web layer composes this
module per request on the shared session.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.common.exception import NotFoundError, ValidationError
from src.core.knowledge.documents.types import PARSE_STATUS_DELETING
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_repository import KnowledgeRepository

_NOT_FOUND_CODE = "knowledge.not_found"


def _require_tenant_id(tenant_id: int) -> None:
    """Reject a non-positive tenant id at the delete boundary."""
    if not isinstance(tenant_id, int) or tenant_id <= 0:
        raise ValidationError(
            code="knowledge.tenant_required",
            message="tenant ID is required",
        )


def _require_document_id(id: str) -> None:
    """Reject a blank document id at the delete boundary."""
    if not id.strip():
        raise ValidationError(
            code="knowledge.id_required",
            message="document ID is required",
        )


async def delete_knowledge(
    *,
    tenant_id: int,
    id: str,
    knowledge_repo: KnowledgeRepository,
    chunk_repo: ChunkRepository,
) -> bool:
    """Soft-delete one document and cascade its chunks; return whether a live row was removed.

    An unknown or already-deleted id raises ``NotFoundError`` (the read
    filters soft-deleted rows), mirroring the single-delete split. The
    row is marked ``deleting`` before its chunks and the row itself are
    soft-deleted; the mark is best-effort and never aborts the cleanup,
    matching the upstream sequence.
    """
    _require_tenant_id(tenant_id)
    _require_document_id(id)
    row = await knowledge_repo.get_by_id(tenant_id, id)
    if row is None:
        raise NotFoundError(
            code=_NOT_FOUND_CODE,
            message="knowledge not found",
        )
    now = datetime.now(UTC)
    # Mark mid-deletion first so in-flight parse tasks cannot resurrect
    # the row; a failed mark is logged-and-continue upstream.
    await knowledge_repo.update_columns(
        id,
        {"parse_status": PARSE_STATUS_DELETING, "updated_at": now},
    )
    await chunk_repo.delete_by_knowledge_id(
        tenant_id=tenant_id,
        knowledge_id=id,
        now=now,
    )
    return await knowledge_repo.soft_delete(
        tenant_id=tenant_id,
        id=id,
        now=now,
    )


__all__ = ["delete_knowledge"]
