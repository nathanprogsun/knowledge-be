"""Cross-KB knowledge move — standalone module.

``move_knowledge`` migrates one completed knowledge item from a source
knowledge base to a target knowledge base, mirroring the upstream move
semantics and its two modes:

- ``reuse_vectors`` keeps the existing chunks: chunk rows are re-pointed
  at the target knowledge base and the retrieval indices are copied
  through the source store (an injected ``VectorIndexReplicator``).
- ``reparse`` re-ingests from scratch: existing chunks are soft-deleted,
  the row is reset to the target knowledge base's configuration
  (``pending`` / ``disabled``, description cleared) and a reparse task
  is triggered through the injected ``ReparseTrigger`` hook.

Compatibility gates mirror the upstream contract: source and target
knowledge bases must share a type and an embedding model, and a
``reuse_vectors`` move requires both knowledge bases to share a vector
store. Tags are knowledge-base-scoped, so the item's tag relations are
cleared on every move.

Retrieval and re-ingest seams
-----------------------------

The vector-index copy (``VectorIndexReplicator``) and the reparse
trigger (``ReparseTrigger``) are injected hooks that land with the
retrieval / worker domains:

- A ``reuse_vectors`` move without a replicator still re-homes the DB
  rows; the destination chunks land without retrieval indices and the
  source-store entries are left orphaned (recoverable by re-indexing).
- A ``reparse`` move without a trigger leaves the item ``pending`` /
  ``disabled`` until a later reparse picks it up.

A failed move restores the item's pre-move status (``completed``) so it
is never left stranded in ``processing``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from src.common.exception import DataError, NotFoundError, ValidationError
from src.common.json import BindParams
from src.core.contracts.knowledge import Knowledge
from src.core.knowledge.documents.types import (
    PARSE_STATUS_COMPLETED,
    PARSE_STATUS_PENDING,
    PARSE_STATUS_PROCESSING,
)
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.core.knowledge.tags.service.tag_service import TagService
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.knowledge import Document

_NOT_FOUND_CODE = "knowledge.not_found"

# Move modes, matching the upstream payload contract.
MOVE_MODE_REUSE_VECTORS = "reuse_vectors"
MOVE_MODE_REPARSE = "reparse"
MOVE_MODES: frozenset[str] = frozenset({MOVE_MODE_REUSE_VECTORS, MOVE_MODE_REPARSE})

# Disabled during a reparse move; the re-ingest pipeline re-enables it.
_ENABLE_STATUS_DISABLED = "disabled"


class VectorIndexReplicator(Protocol):
    """Copy retrieval-index entries from one knowledge base to another."""

    async def copy_indices(
        self,
        *,
        source_kb_id: str,
        target_kb_id: str,
        knowledge_id_map: dict[str, str],
        chunk_id_map: dict[str, str],
    ) -> None:
        """Copy index entries so the moved chunks surface in the target."""


class ReparseTrigger(Protocol):
    """Re-ingest hook for a ``reparse`` move."""

    async def trigger_reparse(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        knowledge_base_id: str,
    ) -> None:
        """Enqueue the document re-processing task for the item."""


def _require_tenant_id(tenant_id: int) -> None:
    """Reject a non-positive tenant id at the service boundary."""
    if not isinstance(tenant_id, int) or tenant_id <= 0:
        raise ValidationError(
            code="knowledge.tenant_required",
            message="tenant ID is required",
        )


def _require_non_empty(value: str, *, code: str, message: str) -> None:
    """Reject a blank string at the service boundary."""
    if not value.strip():
        raise ValidationError(code=code, message=message)


def _require_knowledge_id(id: str) -> None:
    """Reject a blank knowledge id at the service boundary."""
    _require_non_empty(id, code="knowledge.id_required", message="knowledge ID is required")


def _to_knowledge(row: Document) -> Knowledge:
    """Project a persisted ``documents`` row onto the wire shape."""
    return Knowledge(
        id=row.id,
        tenant_id=row.tenant_id,
        knowledge_base_id=row.knowledge_base_id,
        type=row.type,
        title=row.title,
        description=row.description,
        source=row.source,
        channel=row.channel,
        summary_status=row.summary_status,
        parse_status=row.parse_status,
        enable_status=row.enable_status,
        embedding_model_id=row.embedding_model_id,
        file_name=row.file_name,
        file_type=row.file_type,
        file_size=row.file_size,
        file_hash=row.file_hash,
        file_path=row.file_path,
        storage_size=row.storage_size,
        metadata=row.metadata,
        created_at=row.created_at,
        updated_at=row.updated_at,
        processed_at=row.processed_at,
        error_message=row.error_message,
        deleted_at=row.deleted_at,
    )


async def _rehome_knowledge(
    *,
    knowledge_repo: KnowledgeRepository,
    knowledge_id: str,
    values: BindParams,
) -> Document:
    """Write the re-homing columns of a knowledge row, returning the row.

    ``knowledge_base_id`` is immutable in the full-row update path (it is
    owned by the create-time scope), so a move writes it — together with
    the mode's status columns — through the explicit column-update path.
    """
    updated = await knowledge_repo.update_columns(knowledge_id, values)
    if updated is None:
        raise DataError(
            code="document.update_no_row",
            message=f"document {knowledge_id} not found for update",
        )
    return updated


def _shares_store(source: KnowledgeBaseInfo, target: KnowledgeBaseInfo) -> bool:
    """Return whether both knowledge bases resolve to one vector store.

    An empty-string binding is treated as unbound (``None``); two
    unbound knowledge bases share the tenant's default engine, and two
    bound knowledge bases share a store only when their bindings are
    identical.
    """
    a = source.vector_store_id or None
    b = target.vector_store_id or None
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a == b


async def _move_reuse_vectors(
    *,
    tenant_id: int,
    knowledge: Document,
    source_kb: KnowledgeBaseInfo,
    target_kb: KnowledgeBaseInfo,
    knowledge_repo: KnowledgeRepository,
    chunk_repo: ChunkRepository,
    tag_service: TagService,
    index_replicator: VectorIndexReplicator | None,
) -> Knowledge:
    """Move a knowledge item keeping its existing chunks and indices."""
    now = datetime.now(UTC)

    # 1. Identity chunk map for the vector-index copy: the same chunk ids,
    #    just re-homed under the target knowledge base.
    chunks = await chunk_repo.find_all_by_column_values(
        {"tenant_id": tenant_id, "knowledge_id": knowledge.id},
    )
    chunk_id_map = {chunk.id: chunk.id for chunk in chunks}

    # 2. Copy the retrieval indices through the source store (optional
    #    hook; without it the destination chunks land without indices).
    replicator = index_replicator
    if replicator is not None and chunk_id_map and knowledge.embedding_model_id:
        await replicator.copy_indices(
            source_kb_id=source_kb.id,
            target_kb_id=target_kb.id,
            knowledge_id_map={knowledge.id: knowledge.id},
            chunk_id_map=chunk_id_map,
        )

    # 3. Re-point every chunk at the target knowledge base.
    await chunk_repo.move_by_knowledge_id(
        tenant_id=tenant_id,
        knowledge_id=knowledge.id,
        target_kb_id=target_kb.id,
    )

    # 4. Tags are knowledge-base-scoped; clear the item's relations before
    #    re-homing so it cannot leak the source scope.
    await tag_service.delete_knowledge_tag_relations(knowledge.id)

    updated = await _rehome_knowledge(
        knowledge_repo=knowledge_repo,
        knowledge_id=knowledge.id,
        values={
            "knowledge_base_id": target_kb.id,
            "parse_status": PARSE_STATUS_COMPLETED,
            "updated_at": now,
        },
    )
    return _to_knowledge(updated)


async def _move_reparse(
    *,
    tenant_id: int,
    knowledge: Document,
    target_kb: KnowledgeBaseInfo,
    knowledge_repo: KnowledgeRepository,
    chunk_repo: ChunkRepository,
    tag_service: TagService,
    reparse_trigger: ReparseTrigger | None,
) -> Knowledge:
    """Move a knowledge item and reset it for re-ingestion in the target."""
    now = datetime.now(UTC)

    # 1. Drop the existing chunks; reparse re-ingests from the source file.
    await chunk_repo.delete_by_knowledge_id(
        tenant_id=tenant_id,
        knowledge_id=knowledge.id,
        now=now,
    )

    # 2. Tags are knowledge-base-scoped; clear relations and re-home the
    #    row with the target knowledge base's configuration.
    await tag_service.delete_knowledge_tag_relations(knowledge.id)
    updated = await _rehome_knowledge(
        knowledge_repo=knowledge_repo,
        knowledge_id=knowledge.id,
        values={
            "knowledge_base_id": target_kb.id,
            "embedding_model_id": target_kb.embedding_model_id,
            "parse_status": PARSE_STATUS_PENDING,
            "enable_status": _ENABLE_STATUS_DISABLED,
            "description": "",
            "processed_at": None,
            "updated_at": now,
        },
    )

    # 3. Trigger re-ingestion (optional hook; without it the item stays
    #    pending / disabled until a later reparse).
    trigger = reparse_trigger
    if trigger is not None:
        await trigger.trigger_reparse(
            tenant_id=tenant_id,
            knowledge_id=knowledge.id,
            knowledge_base_id=target_kb.id,
        )
    return _to_knowledge(updated)


async def move_knowledge(
    *,
    tenant_id: int,
    knowledge_id: str,
    source_kb_id: str,
    target_kb_id: str,
    mode: str,
    knowledge_repo: KnowledgeRepository,
    chunk_repo: ChunkRepository,
    tag_service: TagService,
    kb_service: KBService,
    index_replicator: VectorIndexReplicator | None = None,
    reparse_trigger: ReparseTrigger | None = None,
) -> Knowledge:
    """Move one completed knowledge item into ``target_kb_id``.

    Compatibility gates (same type, same embedding model, and a shared
    vector store for ``reuse_vectors``) are enforced before any status
    change, so a rejected move leaves the item untouched. A failed move
    restores the item's pre-move status and re-raises.
    """
    _require_tenant_id(tenant_id)
    _require_knowledge_id(knowledge_id)
    _require_non_empty(
        source_kb_id,
        code="knowledge.kb_required",
        message="source knowledge base ID is required",
    )
    _require_non_empty(
        target_kb_id,
        code="knowledge.kb_required",
        message="target knowledge base ID is required",
    )
    if mode not in MOVE_MODES:
        raise ValidationError(
            code="knowledge.move_mode_invalid",
            message=f"unknown move mode: {mode}",
        )

    knowledge = await knowledge_repo.get_by_id(tenant_id, knowledge_id)
    if knowledge is None:
        raise NotFoundError(code=_NOT_FOUND_CODE, message="knowledge not found")

    if knowledge.parse_status != PARSE_STATUS_COMPLETED:
        raise ValidationError(
            code="knowledge.move_not_completed",
            message=(
                f"knowledge {knowledge_id} is not in completed status "
                f"(current: {knowledge.parse_status})"
            ),
        )

    source_kb = await kb_service.get_knowledge_base_by_id(knowledge_base_id=source_kb_id)
    target_kb = await kb_service.get_knowledge_base_by_id(knowledge_base_id=target_kb_id)

    if source_kb.type != target_kb.type:
        raise ValidationError(
            code="knowledge.move_type_mismatch",
            message=f"type mismatch: source={source_kb.type}, target={target_kb.type}",
        )
    if source_kb.embedding_model_id != target_kb.embedding_model_id:
        raise ValidationError(
            code="knowledge.move_embedding_mismatch",
            message=(
                f"embedding model mismatch: source={source_kb.embedding_model_id}, "
                f"target={target_kb.embedding_model_id}"
            ),
        )

    # A reuse_vectors move only works when both knowledge bases resolve to
    # the same vector store. The guard runs before any status change so a
    # rejected move leaves the item untouched (completed).
    if mode == MOVE_MODE_REUSE_VECTORS and not _shares_store(source_kb, target_kb):
        raise ValidationError(
            code="knowledge.move_cross_store_not_supported",
            message=(
                "reuse_vectors move across different vector stores is not "
                "supported; use reparse mode"
            ),
        )

    # Mark as processing during the move; a failure restores the item's
    # pre-move status below.
    processing = knowledge.model_copy(
        update={
            "parse_status": PARSE_STATUS_PROCESSING,
            "updated_at": datetime.now(UTC),
        }
    )
    await knowledge_repo.update(processing)

    try:
        if mode == MOVE_MODE_REUSE_VECTORS:
            return await _move_reuse_vectors(
                tenant_id=tenant_id,
                knowledge=knowledge,
                source_kb=source_kb,
                target_kb=target_kb,
                knowledge_repo=knowledge_repo,
                chunk_repo=chunk_repo,
                tag_service=tag_service,
                index_replicator=index_replicator,
            )
        return await _move_reparse(
            tenant_id=tenant_id,
            knowledge=knowledge,
            target_kb=target_kb,
            knowledge_repo=knowledge_repo,
            chunk_repo=chunk_repo,
            tag_service=tag_service,
            reparse_trigger=reparse_trigger,
        )
    except Exception:
        current = await knowledge_repo.get_by_id(tenant_id, knowledge_id)
        if current is not None and current.parse_status == PARSE_STATUS_PROCESSING:
            await knowledge_repo.update(
                current.model_copy(update={"parse_status": PARSE_STATUS_COMPLETED})
            )
        raise


__all__ = [
    "MOVE_MODES",
    "MOVE_MODE_REPARSE",
    "MOVE_MODE_REUSE_VECTORS",
    "ReparseTrigger",
    "VectorIndexReplicator",
    "move_knowledge",
]
