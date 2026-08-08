"""Batch document delete — soft delete plus cascade chunk cleanup.

Standalone module mirroring the upstream batch-delete sequence on top
of injected repositories: validate the tenant scope, resolve the live
rows (absent / already-deleted ids are dropped silently, matching the
batch split from the single delete which raises ``NotFoundError``), mark
each row mid-deletion, soft-delete their chunks, then soft-delete the
rows themselves.

Retrieval-index and physical-file cleanup run in the retrieval / storage
layers in a later wave; the cascade here keeps the stored rows
consistent. Repositories are injected so the web layer composes this
module per request on the shared session.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.core.knowledge.documents.delete import _require_tenant_id
from src.core.knowledge.documents.types import PARSE_STATUS_DELETING
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_repository import KnowledgeRepository


async def delete_knowledge_list(
    *,
    tenant_id: int,
    ids: list[str],
    knowledge_repo: KnowledgeRepository,
    chunk_repo: ChunkRepository,
) -> int:
    """Soft-delete a batch of documents and cascade their chunks.

    Returns the number of document rows removed. Blank ids are dropped
    before the query; an empty ``ids`` (or one resolving no live rows)
    returns zero without touching the database.
    """
    _require_tenant_id(tenant_id)
    clean_ids = [value for value in ids if value.strip()]
    if not clean_ids:
        return 0
    rows = await knowledge_repo.get_batch(tenant_id, clean_ids)
    if not rows:
        return 0
    now = datetime.now(UTC)
    for row in rows:
        await knowledge_repo.update_columns(
            row.id,
            {"parse_status": PARSE_STATUS_DELETING, "updated_at": now},
        )
        await chunk_repo.delete_by_knowledge_id(
            tenant_id=tenant_id,
            knowledge_id=row.id,
            now=now,
        )
    return await knowledge_repo.soft_delete_list(
        tenant_id=tenant_id,
        ids=[row.id for row in rows],
        now=now,
    )


__all__ = ["delete_knowledge_list"]
