"""Standalone document reparse — reset a document and re-submit it for parsing.

Implements the reparse lifecycle for a single document row: load it
tenant-scoped, accept an optional parse-config override (validated and
persisted under ``metadata.process_overrides``), reset the row to a fresh
``pending`` attempt with the retrieval counter zeroed, clear the existing
chunk set for non-manual sources, and submit the async parse task through
an injected enqueuer. The worker-side parse and the retrieval-index
cleanup land with the task infrastructure in a later wave; this module
only owns the orchestration up to submission.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from src.common.exception import NotFoundError, ValidationError
from src.common.json import JsonObject
from src.core.contracts.knowledge import Knowledge
from src.core.knowledge.chunks.service.chunk_service import ChunkService
from src.core.knowledge.documents.types import (
    KNOWLEDGE_TYPE_MANUAL,
    PARSE_STATUS_FAILED,
    PARSE_STATUS_PENDING,
)
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.knowledge import Document

_NOT_FOUND_CODE = "knowledge.not_found"

_ENABLE_STATUS_DISABLED = "disabled"
_ENQUEUE_FAILED_MESSAGE = "Failed to enqueue processing task"

_METADATA_KEY_PROCESS_OVERRIDES = "process_overrides"
_METADATA_KEY_MANUAL_CONTENT = "content"

_KNOWLEDGE_TYPE_FILE_URL = "file_url"
_KNOWLEDGE_TYPE_URL = "url"
_UNKNOWN_FILE_TYPE = "unknown"
_DEFAULT_QUESTION_COUNT = 3


class DocumentProcessPayload(BaseModel):
    """Async document-parse task payload for file / file-URL / URL sources."""

    model_config = ConfigDict(frozen=True)

    tenant_id: int
    knowledge_id: str
    knowledge_base_id: str
    file_path: str | None = None
    file_url: str | None = None
    url: str | None = None
    file_name: str | None = None
    file_type: str | None = None
    enable_multimodel: bool = False
    enable_question_generation: bool = False
    question_count: int = _DEFAULT_QUESTION_COUNT
    language: str | None = None


class ReparseEnqueuer(Protocol):
    """Async parse-submission hook implemented by the web layer.

    ``enqueue_manual_process`` covers manual Markdown knowledge (the
    content travels in the payload); ``enqueue_document_process`` covers
    file / file-URL / URL knowledge. Implementations raise
    ``ApplicationError`` subclasses when the task could not be queued.
    """

    async def enqueue_manual_process(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        content: str,
    ) -> None:
        """Submit a manual-content parse task."""

    async def enqueue_document_process(
        self,
        *,
        tenant_id: int,
        payload: DocumentProcessPayload,
    ) -> None:
        """Submit a document parse task for a file / URL source."""


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
    """Project a persisted ``documents`` row onto the wire shape.

    Storage-only columns (``pending_subtasks_count``, ``custom_metadata``)
    are deliberately absent from the wire projection.
    """
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


def _store_process_overrides(row: Document, overrides: JsonObject) -> JsonObject:
    """Validate and merge a reparse config override into ``metadata``.

    The override is a plain JSON object persisted under the
    ``process_overrides`` metadata key so the worker re-reads the same
    config the submission used.
    """
    if not isinstance(overrides, dict):
        raise ValidationError(
            code="knowledge.invalid_process_overrides",
            message="process_overrides must be a JSON object",
        )
    metadata = dict(row.metadata or {})
    metadata[_METADATA_KEY_PROCESS_OVERRIDES] = overrides
    return metadata


def _effective_flags(
    overrides: JsonObject | None,
    kb: KnowledgeBaseInfo,
) -> tuple[bool, bool, int]:
    """Resolve the multimodal / question-generation flags for the payload.

    Per-override values win; otherwise the knowledge base's stored config
    is the default.
    """
    overrides = overrides or {}
    enable_multimodel = bool(overrides.get("enable_multimodel"))
    if not enable_multimodel:
        chunking = kb.chunking_config or {}
        enable_multimodel = bool(chunking.get("enable_multimodal", False))
    qg_raw = overrides.get("question_generation_config")
    if not isinstance(qg_raw, dict):
        qg_raw = kb.question_generation_config or {}
    qg_enabled = bool(qg_raw.get("enabled", False))
    question_count = qg_raw.get("question_count")
    if not isinstance(question_count, int) or question_count <= 0:
        question_count = _DEFAULT_QUESTION_COUNT
    return enable_multimodel, qg_enabled, question_count


def _reset_for_reparse(row: Document, embedding_model_id: str) -> Document:
    """Return the row reset to a fresh parse attempt.

    ``pending`` parse status, disabled retrieval, cleared description and
    error, and the knowledge base's embedding model. The
    ``pending_subtasks_count`` counter is zeroed separately because the
    full-row update deliberately omits it.
    """
    return row.model_copy(
        update={
            "parse_status": PARSE_STATUS_PENDING,
            "enable_status": _ENABLE_STATUS_DISABLED,
            "description": "",
            "processed_at": None,
            "error_message": "",
            "embedding_model_id": embedding_model_id,
            "pending_subtasks_count": 0,
            "updated_at": datetime.now(UTC),
        }
    )


def _manual_content(row: Document) -> str | None:
    """Return the manual knowledge content stored in ``metadata``."""
    metadata = row.metadata or {}
    content = metadata.get(_METADATA_KEY_MANUAL_CONTENT)
    if not isinstance(content, str):
        return None
    return content


def _file_type(file_name: str | None) -> str:
    """Derive the extension of a file name, or ``unknown`` when absent."""
    if not file_name:
        return _UNKNOWN_FILE_TYPE
    parts = file_name.split(".")
    if len(parts) < 2:
        return _UNKNOWN_FILE_TYPE
    return parts[-1]


def _build_payload(
    *,
    row: Document,
    enable_multimodel: bool,
    qg_enabled: bool,
    question_count: int,
) -> DocumentProcessPayload:
    """Build the document-process payload for a file / URL source.

    ``file_url`` / ``url`` route onto the document's ``source``; a plain
    file routes onto its stored ``file_path`` with the extension derived
    from the file name.
    """
    tenant_id = row.tenant_id
    knowledge_id = row.id
    kb_id = row.knowledge_base_id
    file_name = row.file_name
    if row.type == _KNOWLEDGE_TYPE_FILE_URL:
        return DocumentProcessPayload(
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
            knowledge_base_id=kb_id,
            file_url=row.source,
            file_name=file_name,
            file_type=row.file_type,
            enable_multimodel=enable_multimodel,
            enable_question_generation=qg_enabled,
            question_count=question_count,
        )
    if row.type == _KNOWLEDGE_TYPE_URL:
        return DocumentProcessPayload(
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
            knowledge_base_id=kb_id,
            url=row.source,
            file_name=file_name,
            enable_multimodel=enable_multimodel,
            enable_question_generation=qg_enabled,
            question_count=question_count,
        )
    return DocumentProcessPayload(
        tenant_id=tenant_id,
        knowledge_id=knowledge_id,
        knowledge_base_id=kb_id,
        file_path=row.file_path,
        file_name=file_name,
        file_type=_file_type(file_name),
        enable_multimodel=enable_multimodel,
        enable_question_generation=qg_enabled,
        question_count=question_count,
    )


async def _mark_enqueue_failed(
    knowledge_repo: KnowledgeRepository,
    row: Document,
) -> None:
    """Best-effort flip to ``failed`` after a task-submission failure."""
    failed = row.model_copy(
        update={
            "parse_status": PARSE_STATUS_FAILED,
            "error_message": _ENQUEUE_FAILED_MESSAGE,
            "updated_at": datetime.now(UTC),
        }
    )
    with contextlib.suppress(Exception):
        await knowledge_repo.update(failed)


async def reparse_knowledge(
    *,
    tenant_id: int,
    knowledge_id: str,
    knowledge_repo: KnowledgeRepository,
    kb_service: KBService,
    chunk_service: ChunkService,
    enqueuer: ReparseEnqueuer,
    process_overrides: JsonObject | None = None,
) -> Knowledge:
    """Reset the document to a fresh parse attempt and submit the parse task.

    Raises ``NotFoundError`` when the document is absent or out of the
    tenant scope, ``ValidationError`` on invalid input, and re-raises the
    enqueuer's error after marking the document ``failed`` when the task
    could not be queued.
    """
    _require_tenant_id(tenant_id)
    _require_document_id(knowledge_id)

    existing = await knowledge_repo.get_by_id(tenant_id, knowledge_id)
    if existing is None:
        raise NotFoundError(code=_NOT_FOUND_CODE, message="knowledge not found")

    kb = await kb_service.get_knowledge_base_by_id_and_tenant(
        tenant_id=tenant_id,
        knowledge_base_id=existing.knowledge_base_id,
    )

    if process_overrides is not None:
        existing = existing.model_copy(
            update={"metadata": _store_process_overrides(existing, process_overrides)}
        )
        await knowledge_repo.update_columns(existing.id, {"metadata": existing.metadata})

    reset = _reset_for_reparse(existing, kb.embedding_model_id)
    await knowledge_repo.update(reset)
    await knowledge_repo.update_columns(reset.id, {"pending_subtasks_count": 0})

    if existing.type == KNOWLEDGE_TYPE_MANUAL:
        content = _manual_content(existing)
        if content is None:
            raise ValidationError(
                code="knowledge.manual_content_missing",
                message="无法获取手工知识内容",
            )
        try:
            await enqueuer.enqueue_manual_process(
                tenant_id=tenant_id,
                knowledge_id=existing.id,
                content=content,
            )
        except Exception:
            await _mark_enqueue_failed(knowledge_repo, reset)
            raise
        return _to_knowledge(reset)

    await chunk_service.delete_chunks_by_knowledge_id(
        tenant_id=tenant_id,
        knowledge_id=existing.id,
    )

    enable_multimodel, qg_enabled, question_count = _effective_flags(
        process_overrides, kb
    )
    payload = _build_payload(
        row=existing,
        enable_multimodel=enable_multimodel,
        qg_enabled=qg_enabled,
        question_count=question_count,
    )
    if not payload.file_path and not payload.file_url and not payload.url:
        # No file, URL, or manual content to parse — surface the state.
        failed = reset.model_copy(
            update={
                "parse_status": PARSE_STATUS_FAILED,
                "error_message": "Knowledge has no parseable content",
            }
        )
        await knowledge_repo.update(failed)
        raise ValidationError(
            code="knowledge.not_parseable",
            message="Knowledge has no parseable content",
        )
    try:
        await enqueuer.enqueue_document_process(tenant_id=tenant_id, payload=payload)
    except Exception:
        await _mark_enqueue_failed(knowledge_repo, reset)
        raise
    return _to_knowledge(reset)


__all__ = [
    "DocumentProcessPayload",
    "ReparseEnqueuer",
    "reparse_knowledge",
]
