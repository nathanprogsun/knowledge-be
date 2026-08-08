"""Create a knowledge entry from text passages.

Standalone create-variant: validates each passage, persists a
``passage`` knowledge row through the merged repositories, and either
leaves it ``pending`` for the async processing seam (default) or, in
``sync`` mode, materialises one chunk per passage and settles the row to
``completed``.

Behaviour mirrors the upstream create-from-passage service:

- every passage passes the same input validation (control characters,
  XSS patterns) and is stored trimmed;
- the knowledge row is stamped ``type=passage`` / ``pending`` with the
  knowledge base's embedding model;
- ``sync`` mode skips the queue and writes the chunks directly, marking
  the row processed when the write succeeds.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from src.common.exception import ValidationError
from src.core.contracts.knowledge import Knowledge
from src.core.knowledge.documents.create_common import (
    ENABLE_STATUS_DISABLED,
    ENABLE_STATUS_ENABLED,
    build_document_row,
    default_channel,
    require_knowledge_base_id,
    require_tenant_id,
    to_knowledge,
    validate_input,
)
from src.core.knowledge.documents.types import (
    PARSE_STATUS_COMPLETED,
    PARSE_STATUS_PENDING,
)
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.chunk import Chunk
from src.db.models.knowledge import Document


def _sanitize_passages(passages: list[str]) -> list[str]:
    """Validate and trim every passage, in input order."""
    safe: list[str] = []
    for index, passage in enumerate(passages):
        cleaned, ok = validate_input(passage)
        if not ok:
            raise ValidationError(
                code="knowledge.passage_invalid",
                message=f"段落 {index + 1} 包含非法内容",
            )
        safe.append(cleaned)
    return safe


def _build_passage_chunks(
    *,
    row: Document,
    passages: list[str],
    now: datetime,
) -> list[Chunk]:
    """Build one chunk per non-empty passage with rune offsets.

    Empty passages are skipped; the offsets accumulate over the trimmed
    passages so each chunk's ``start_at`` / ``end_at`` cover the content
    stored on the preceding chunks.
    """
    chunks: list[Chunk] = []
    start = 0
    for index, passage in enumerate(passages):
        if not passage:
            continue
        end = start + len(passage)
        chunks.append(
            Chunk(
                id=str(uuid.uuid4()),
                tenant_id=row.tenant_id,
                knowledge_base_id=row.knowledge_base_id,
                knowledge_id=row.id,
                content=passage,
                chunk_index=index,
                is_enabled=True,
                start_at=start,
                end_at=end,
                chunk_type="text",
                flags=1,
                source_content="",
                content_revision=0,
                index_status="ready",
                last_editor_id="",
                context_header="",
                created_at=now,
                updated_at=now,
            )
        )
        start = end
    return chunks


async def _persist_passage_chunks(
    *,
    row: Document,
    passages: list[str],
    chunk_repo: ChunkRepository,
    knowledge_repo: KnowledgeRepository,
    now: datetime,
) -> Document:
    """Write the passage chunks and settle the row to ``completed``.

    The retrieval index is not written here (no indexing domain yet), so
    chunks are stamped ``index_status=ready`` and the row is settled
    straight to ``completed`` / ``enabled`` with a processed timestamp.
    """
    chunks = _build_passage_chunks(row=row, passages=passages, now=now)
    if chunks:
        await chunk_repo.create_many(chunks)
    settled = row.model_copy(
        update={
            "parse_status": PARSE_STATUS_COMPLETED,
            "enable_status": ENABLE_STATUS_ENABLED,
            "processed_at": now,
            "updated_at": now,
        }
    )
    return await knowledge_repo.update(settled)


async def create_knowledge_from_passage(
    *,
    tenant_id: int,
    kb_id: str,
    passages: list[str],
    channel: str | None = None,
    sync: bool = False,
    knowledge_repo: KnowledgeRepository,
    kb_service: KBService,
    chunk_repo: ChunkRepository | None = None,
    now: datetime | None = None,
) -> Knowledge:
    """Create a knowledge entry from text passages.

    ``sync=False`` (default) leaves the row ``pending`` for the deferred
    processing enqueue; ``sync=True`` requires ``chunk_repo`` and writes
    one chunk per passage before settling the row.
    """
    require_tenant_id(tenant_id)
    require_knowledge_base_id(kb_id)
    if not passages:
        raise ValidationError(
            code="knowledge.passage_required",
            message="段落不能为空",
        )
    safe_passages = _sanitize_passages(passages)
    kb = await kb_service.get_knowledge_base_by_id(knowledge_base_id=kb_id)
    stamp = now or datetime.now(UTC)
    row = build_document_row(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        type="passage",
        title="",
        source="",
        channel=default_channel(channel),
        parse_status=PARSE_STATUS_PENDING,
        enable_status=ENABLE_STATUS_DISABLED,
        embedding_model_id=kb.embedding_model_id,
        now=stamp,
    )
    persisted = await knowledge_repo.create(row)
    if sync:
        if chunk_repo is None:
            raise ValueError("sync mode requires a chunk repository")
        persisted = await _persist_passage_chunks(
            row=persisted,
            passages=safe_passages,
            chunk_repo=chunk_repo,
            knowledge_repo=knowledge_repo,
            now=stamp,
        )
    return to_knowledge(persisted)


__all__ = ["create_knowledge_from_passage"]
