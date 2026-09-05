"""Read-side document handlers shared by the document router.

``GET /knowledge/batch``, ``GET /knowledge/search``,
``GET /knowledge/{id}/spans``, and
``POST /knowledge/{id}/regenerate-summary`` live here so ``router.py``
stays under the file-size cap. Route decorators stay on ``router.py``
so the feature-map and endpoint-coverage scanners see them.
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import quote

from fastapi import Query
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.exception import NotFoundError, ValidationError
from src.core.contracts.knowledge import Knowledge, UpdateKnowledgeRequest
from src.core.knowledge.documents.documents_orchestrator import (
    KnowledgeDocumentsOrchestrator,
)
from src.core.knowledge.documents.service.knowledge_service import KnowledgeService
from src.core.knowledge.documents.span_tracker import SpanProgress, SpanTracker
from src.core.knowledge.documents.span_tree import SpansRead, spans_read_payload
from src.core.knowledge.documents.types import KNOWLEDGE_TYPE_MANUAL
from src.web.api.files.router import content_type_for_storage_path, resolve_file_service_for_path
from src.web.api.knowledge.documents.views import (
    DeleteEnvelope,
    DeleteResult,
    KnowledgeBatchEnvelope,
    KnowledgeSearchEnvelope,
    KnowledgeSpansEnvelope,
    KnowledgeTaskEnvelope,
    KnowledgeUpdatedEnvelope,
    ReparseRequest,
)

_SUMMARY_QUEUED_MESSAGE = "Summary refresh queued"


class SearchQuery:
    """Query params for ``GET /knowledge/search``. Agent ids are unused."""

    def __init__(
        self,
        keyword: str = Query(default=""),
        query: str = Query(default=""),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=20, ge=1, le=100),
        file_types: str = Query(default=""),
        recent: bool = Query(default=False),
        agent_id: str | None = Query(default=None),
        agent_source_tenant_id: str | None = Query(default=None),
    ) -> None:
        del agent_id, agent_source_tenant_id
        self.keyword: str = keyword.strip() or query.strip()
        self.offset: int = offset
        self.limit: int = limit
        self.file_types: list[str] = _split_file_types(file_types)
        self.recent: bool = recent


def _split_file_types(raw: str) -> list[str]:
    """Split a comma-separated file-type list, dropping blanks and dots."""
    parts: list[str] = []
    seen: set[str] = set()
    for item in raw.split(","):
        value = item.strip().lstrip(".").lower()
        if value and value not in seen:
            seen.add(value)
            parts.append(value)
    return parts


def _dedupe_ids(raw_ids: list[str]) -> list[str]:
    """Trim, deduplicate, and drop blank ids, preserving first-seen order."""
    seen: set[str] = set()
    clean: list[str] = []
    for raw in raw_ids:
        value = raw.strip()
        if value and value not in seen:
            seen.add(value)
            clean.append(value)
    return clean


async def batch_query(
    *,
    service: KnowledgeService,
    tenant_id: int,
    ids: list[str],
    kb_id: str | None,
    max_batch: int,
) -> KnowledgeBatchEnvelope:
    """Return the live documents matching ``ids`` in the caller's workspace.

    Missing ids are dropped (same as the service batch read). An optional
    ``kb_id`` further restricts the set so a shared-KB poll cannot leak
    rows from another knowledge base in the same workspace.
    """
    clean_ids = _dedupe_ids(ids)
    if len(clean_ids) > max_batch:
        raise ValidationError(
            code="knowledge.batch_too_many",
            message=f"too many ids (max {max_batch} per batch)",
        )
    documents = await service.get_documents(tenant_id=tenant_id, ids=clean_ids)
    if kb_id:
        documents = [row for row in documents if row.knowledge_base_id == kb_id]
    return KnowledgeBatchEnvelope(success=True, data=documents)


async def search_knowledge_documents(
    *,
    service: KnowledgeService,
    tenant_id: int,
    params: SearchQuery,
) -> KnowledgeSearchEnvelope:
    """Return a tenant-scoped file page for the chat @ picker."""
    items, total = await service.search_documents(
        tenant_id=tenant_id,
        keyword=params.keyword,
        offset=params.offset,
        limit=params.limit,
        file_types=params.file_types,
        recent=params.recent,
    )
    return KnowledgeSearchEnvelope(
        success=True,
        data=items,
        has_more=params.offset + len(items) < total,
        total=total,
    )


async def read_spans(
    *,
    service: KnowledgeService,
    tracker: SpanTracker,
    tenant_id: int,
    knowledge_id: str,
    attempt: int | None,
) -> KnowledgeSpansEnvelope:
    """Return the processing-span tree, or an empty root if never tracked.

    The document is loaded first so a cross-workspace id is 404. A
    document that exists but has no span rows still answers 200 with an
    empty tree so the timeline can render instead of treating a missing
    tracker as a hard error.
    """
    knowledge = await service.get_document(tenant_id=tenant_id, id=knowledge_id)
    progress = await _progress_or_none(tracker, knowledge_id, attempt)
    payload: SpansRead = spans_read_payload(
        knowledge_id=knowledge.id,
        parse_status=knowledge.parse_status,
        progress=progress,
    )
    return KnowledgeSpansEnvelope(success=True, data=payload)


async def update_document(
    *,
    service: KnowledgeService,
    tenant_id: int,
    knowledge_id: str,
    body: UpdateKnowledgeRequest,
) -> KnowledgeUpdatedEnvelope:
    """Patch mutable document fields, including manual content."""
    knowledge = await service.update_document(
        tenant_id=tenant_id,
        id=knowledge_id,
        title=body.title,
        description=body.description,
        content=body.content,
        status=body.status,
        process_config=body.process_config,
    )
    return KnowledgeUpdatedEnvelope(
        success=True,
        message="Knowledge updated successfully",
        data=knowledge,
    )


async def stream_document(
    *,
    service: KnowledgeService,
    session: AsyncSession,
    tenant_id: int,
    knowledge_id: str,
    disposition: Literal["attachment", "inline"],
) -> Response:
    """Stream stored bytes, or the manual markdown body."""
    knowledge = await service.get_document(tenant_id=tenant_id, id=knowledge_id)
    if knowledge.type == KNOWLEDGE_TYPE_MANUAL:
        return _manual_file_response(knowledge, disposition)
    path = (knowledge.file_path or "").strip()
    if not path:
        raise NotFoundError(
            code="knowledge.file_unavailable",
            message="file is not stored for this document",
        )
    file_service = await resolve_file_service_for_path(session, tenant_id, path)
    if file_service is None:
        raise NotFoundError(
            code="knowledge.file_unavailable",
            message="file service unavailable",
        )
    try:
        stream = await file_service.get_file(path)
    except Exception as exc:
        raise NotFoundError(
            code="knowledge.file_not_found",
            message="file not found",
        ) from exc
    filename = knowledge.file_name or knowledge.title or knowledge.id
    return StreamingResponse(
        stream,
        media_type=_media_type_for_disposition(disposition, path),
        headers=_stream_headers(disposition, filename),
    )


def _manual_file_response(knowledge: Knowledge, disposition: str) -> Response:
    """Serve the markdown stored on a manual document."""
    metadata = knowledge.metadata or {}
    raw = metadata.get("content")
    content = raw if isinstance(raw, str) else ""
    filename = knowledge.file_name or f"{knowledge.title or knowledge.id}.md"
    if not filename.lower().endswith(".md"):
        filename = f"{filename}.md"
    return Response(
        content=content.encode("utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers=_stream_headers(disposition, filename),
    )


def _media_type_for_disposition(disposition: str, path: str) -> str:
    if disposition == "attachment":
        return "application/octet-stream"
    return content_type_for_storage_path(path)


def _stream_headers(disposition: str, filename: str) -> dict[str, str]:
    headers = {
        "Content-Disposition": _content_disposition(disposition, filename),
        "X-Content-Type-Options": "nosniff",
    }
    if disposition == "attachment":
        headers["Cache-Control"] = "must-revalidate"
    else:
        headers["Cache-Control"] = "private, max-age=3600"
    return headers


def _content_disposition(disposition: str, filename: str) -> str:
    """RFC 5987 filename so non-ASCII titles survive the download attribute."""
    safe = filename.replace('"', "")
    return f"{disposition}; filename=\"{safe}\"; filename*=UTF-8''{quote(filename)}"


async def delete_document(
    *,
    service: KnowledgeDocumentsOrchestrator,
    tenant_id: int,
    knowledge_id: str,
) -> DeleteEnvelope:
    """Soft-delete one document and cascade its chunks."""
    deleted = await service.delete(tenant_id=tenant_id, id=knowledge_id)
    return DeleteEnvelope(
        success=True,
        message="Knowledge deleted",
        data=DeleteResult(deleted=deleted),
    )


async def reparse_document(
    *,
    service: KnowledgeDocumentsOrchestrator,
    tenant_id: int,
    knowledge_id: str,
    body: ReparseRequest | None,
) -> KnowledgeTaskEnvelope:
    """Reset a document for a fresh parse attempt."""
    knowledge = await service.reparse(
        tenant_id=tenant_id,
        knowledge_id=knowledge_id,
        process_overrides=body.process_config if body is not None else None,
    )
    return KnowledgeTaskEnvelope(
        success=True,
        message="Knowledge reparse task submitted",
        data=knowledge,
    )


async def cancel_document_parse(
    *,
    service: KnowledgeDocumentsOrchestrator,
    tenant_id: int,
    knowledge_id: str,
) -> KnowledgeTaskEnvelope:
    """Cancel an in-flight document parse."""
    knowledge = await service.cancel_parse(tenant_id=tenant_id, knowledge_id=knowledge_id)
    return KnowledgeTaskEnvelope(
        success=True,
        message="Knowledge parse cancelled",
        data=knowledge,
    )


def move_progress_missing() -> None:
    """Move progress storage has not landed."""
    raise NotFoundError(code="task_progress.not_found", message="task not found")


async def regenerate_summary(
    *,
    service: KnowledgeService,
    tenant_id: int,
    knowledge_id: str,
) -> KnowledgeTaskEnvelope:
    """Generate the document summary and return the updated row."""
    knowledge: Knowledge = await service.request_summary_refresh(
        tenant_id=tenant_id,
        id=knowledge_id,
    )
    return KnowledgeTaskEnvelope(
        success=True,
        message=_SUMMARY_QUEUED_MESSAGE,
        data=knowledge,
    )


async def _progress_or_none(
    tracker: SpanTracker,
    knowledge_id: str,
    attempt: int | None,
) -> SpanProgress | None:
    """Return the attempt's spans, or ``None`` when none were recorded."""
    try:
        return await tracker.get_progress(knowledge_id=knowledge_id, attempt=attempt)
    except NotFoundError as exc:
        if exc.code != "span.not_found":
            raise
        return None


__all__ = [
    "SearchQuery",
    "batch_query",
    "cancel_document_parse",
    "delete_document",
    "move_progress_missing",
    "read_spans",
    "regenerate_summary",
    "reparse_document",
    "search_knowledge_documents",
    "stream_document",
    "update_document",
]
