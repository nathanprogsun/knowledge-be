"""Standalone document parse cancellation.

Mirrors the cancel semantics for an in-flight parse: a row in
``pending`` / ``processing`` / ``finalizing`` is flipped to ``cancelled``
with the retrieval counter zeroed and a user-facing error message
recorded. Already-``cancelled`` rows are idempotent (the pending-task
dequeue is still retried); finished (``completed`` / ``failed``) or
mid-deletion rows are rejected. Persisted partial chunks are left in
place so a later reparse can reuse them.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from typing import Protocol

from src.common.exception import NotFoundError, ValidationError
from src.core.contracts.knowledge import Knowledge
from src.core.knowledge.documents.types import (
    PARSE_STATUS_CANCELLED,
    PARSE_STATUS_COMPLETED,
    PARSE_STATUS_DELETING,
    PARSE_STATUS_FAILED,
)
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.knowledge import Document

_NOT_FOUND_CODE = "knowledge.not_found"

_CANCELLED_MESSAGE = "用户已取消解析"


class ParseTaskInspector(Protocol):
    """Best-effort removal of queued / in-flight parse tasks.

    Implemented by the web layer over the task broker; a no-op inspector
    keeps the cancellation path functional without one.
    """

    async def cancel_tasks_for_knowledge(self, *, knowledge_id: str) -> None:
        """Ask the task broker to dequeue pending work for a knowledge item."""


def _require_tenant_id(tenant_id: int) -> None:
    """Reject a non-positive tenant id at the service boundary."""
    if not isinstance(tenant_id, int) or tenant_id <= 0:
        raise ValidationError(
            code="knowledge.tenant_required",
            message="tenant ID is required",
        )


def _require_document_id(knowledge_id: str) -> None:
    """Reject a blank document id at the service boundary."""
    if not knowledge_id.strip():
        raise ValidationError(
            code="knowledge.id_required",
            message="document ID is required",
        )


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


async def _dequeue_best_effort(
    task_inspector: ParseTaskInspector | None,
    knowledge_id: str,
) -> None:
    """Ask the broker to drop pending tasks; failures never block the cancel."""
    if task_inspector is None:
        return
    with contextlib.suppress(Exception):
        await task_inspector.cancel_tasks_for_knowledge(knowledge_id=knowledge_id)


async def cancel_knowledge_parse(
    *,
    tenant_id: int,
    knowledge_id: str,
    knowledge_repo: KnowledgeRepository,
    task_inspector: ParseTaskInspector | None = None,
) -> Knowledge:
    """Cancel an in-progress parse, returning the updated document shape.

    Raises ``NotFoundError`` for an absent or out-of-scope row and
    ``ValidationError`` when the parse has already finished (``completed``
    / ``failed``) or the row is mid-deletion.
    """
    _require_tenant_id(tenant_id)
    _require_document_id(knowledge_id)

    existing = await knowledge_repo.get_by_id(tenant_id, knowledge_id)
    if existing is None:
        raise NotFoundError(code=_NOT_FOUND_CODE, message="knowledge not found")

    if existing.parse_status == PARSE_STATUS_CANCELLED:
        # Idempotent: re-attempt the dequeue, skip the row update.
        await _dequeue_best_effort(task_inspector, knowledge_id)
        return _to_knowledge(existing)

    if existing.parse_status in (PARSE_STATUS_COMPLETED, PARSE_STATUS_FAILED):
        raise ValidationError(
            code="knowledge.parse_not_cancellable",
            message="解析已结束，无法取消",
        )
    if existing.parse_status == PARSE_STATUS_DELETING:
        raise ValidationError(
            code="knowledge.parse_deleting",
            message="知识正在删除中，无法取消解析",
        )

    now = datetime.now(UTC)
    updated = existing.model_copy(
        update={
            "parse_status": PARSE_STATUS_CANCELLED,
            "error_message": _CANCELLED_MESSAGE,
            "pending_subtasks_count": 0,
            "updated_at": now,
        }
    )
    # Flip the row and zero the enrichment counter in one write so a late
    # subtask finalize can not race-promote the row back to ``completed``.
    await knowledge_repo.update_columns(
        updated.id,
        {
            "parse_status": updated.parse_status,
            "error_message": updated.error_message,
            "pending_subtasks_count": 0,
            "updated_at": now,
        },
    )
    await _dequeue_best_effort(task_inspector, knowledge_id)
    return _to_knowledge(updated)


__all__ = [
    "ParseTaskInspector",
    "cancel_knowledge_parse",
]
