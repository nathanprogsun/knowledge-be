"""Knowledge document HTTP endpoints.

KB-scoped uploads live under ``/knowledge-bases/{id}/knowledge``.
Per-document routes address ``/knowledge/{id}``. Literal collection
paths sit outside ``{id}``. Reads are Viewer+; mutations Contributor+.
Query-parameter descriptions stay Chinese (upstream swagger).
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import Response

from src.common.exception import (
    PermissionDeniedError,
    UnauthorizedError,
    ValidationError,
)
from src.common.json import JsonObject
from src.core.contracts.knowledge import (
    CreateKnowledgeFromURLRequest,
    CreateManualKnowledgeRequest,
    KnowledgeMoveRequest,
    KnowledgeMoveResponse,
    UpdateKnowledgeRequest,
)
from src.core.knowledge.documents.types import DocumentListFilter
from src.web.api.knowledge.documents import document_reads
from src.web.api.knowledge.documents.views import (
    BatchDeleteData,
    BatchDeleteEnvelope,
    BatchDeleteRequest,
    BatchReparseData,
    BatchReparseEnvelope,
    BatchReparseRequest,
    CloneKnowledgeRequest,
    CreatePassageKnowledgeRequest,
    DeleteEnvelope,
    KnowledgeBatchEnvelope,
    KnowledgeEnvelope,
    KnowledgeListEnvelope,
    KnowledgeSearchEnvelope,
    KnowledgeSpansEnvelope,
    KnowledgeTagBatchEnvelope,
    KnowledgeTagBatchUpdateRequest,
    KnowledgeTaskEnvelope,
    KnowledgeUpdatedEnvelope,
    MoveEnvelope,
    ReparseRequest,
)
from src.web.deps import AuthDep, RoleContributorDep, RoleViewerDep
from src.web.deps.context import get_tenant_id_dep
from src.web.deps.knowledge_bases import KBServiceDep
from src.web.deps.knowledge_documents import (
    KnowledgeDocumentsDep,
    KnowledgeServiceDep,
    SpanTrackerDep,
)
from src.web.deps.knowledge_tags import TagServiceDep
from src.web.deps.session import SessionDep
from src.web.deps.task_progress import TaskProgressTenantDep

# Function-arg-style principal dep aliases.
_PrincipalTenant = Annotated[int, Depends(get_tenant_id_dep)]


kb_documents_router = APIRouter(
    prefix="/knowledge-bases/{id}/knowledge",
    tags=["knowledge-documents"],
)

documents_router = APIRouter(prefix="/knowledge", tags=["knowledge-documents"])

_MOVE_TASK_TYPE = "kg_move"
_MOVE_STARTED_MESSAGE = "Knowledge move task started"

# Batch cross-document writes mirror the single-document lifecycle
# routes: Contributor+, tenant-scoped, and a synthetic task id so the
# wire contract (task_id + count) matches the async upstream while the
# work itself runs synchronously until the task infrastructure lands.
_MAX_BATCH = 200
_BATCH_DELETE_TASK_TYPE = "kg_batch_delete"
_BATCH_REPARSE_TASK_TYPE = "kg_batch_reparse"
_BATCH_DELETE_MESSAGE = "Batch delete task submitted"
_BATCH_REPARSE_MESSAGE = "Batch reparse task submitted"


def _require_tenant(tenant_id: int) -> int:
    """Return the active workspace id, or fail.

    A document is always workspace-scoped; without a tenant context
    there is no safe default, so this rejects rather than guessing.
    """
    if tenant_id == 0:
        raise UnauthorizedError(
            code="auth.tenant_context_missing",
            message="unauthorized: workspace context missing",
        )
    return tenant_id


def _split_tag_ids(raw: str) -> list[str]:
    """Split a comma-separated tag id list, dropping blanks and the sentinel."""
    if not raw:
        return []
    return [
        part.strip() for part in raw.split(",") if part.strip() and part.strip() != "__untagged__"
    ]


def _parse_filter_time(raw: str | None, label: str) -> datetime | None:
    """Parse a query timestamp into a datetime, or fail with a clear error.

    Accepts the ISO-8601 forms the list filter expects; an unparseable
    value is rejected rather than silently ignored.
    """
    if not raw or not raw.strip():
        return None
    try:
        return datetime.fromisoformat(raw.strip())
    except ValueError as exc:
        raise ValidationError(
            code="knowledge.invalid_filter_time",
            message=f"invalid {label}: {raw}",
        ) from exc


def _parse_json_object(raw: str, label: str) -> JsonObject | None:
    """Parse a JSON form field into an object, rejecting non-objects."""
    if not raw or not raw.strip():
        return None
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise ValidationError(
            code="knowledge.invalid_form_json",
            message=f"Invalid {label} format",
        ) from exc
    if not isinstance(value, dict):
        raise ValidationError(
            code="knowledge.invalid_form_json",
            message=f"Invalid {label} format",
        )
    return value


def _parse_optional_bool(raw: str, label: str) -> bool | None:
    """Parse a boolean form field, treating blanks as ``None``."""
    if not raw or not raw.strip():
        return None
    normalized = raw.strip().lower()
    if normalized in ("true", "1"):
        return True
    if normalized in ("false", "0"):
        return False
    raise ValidationError(
        code="knowledge.invalid_form_bool",
        message=f"Invalid {label} format",
    )


def _move_task_id(tenant_id: int, kb_id: str) -> str:
    """Generate a tenant-embedded move task id (``<type>_<tenant>_<ms>_<uuid>_<biz>``).

    The workspace id is embedded so the move-progress guard can verify
    ownership before serving the record.
    """
    millis = int(datetime.now(UTC).timestamp() * 1000)
    business = kb_id.replace("-", "").replace("_", "")[:12]
    return f"{_MOVE_TASK_TYPE}_{tenant_id}_{millis}_{uuid.uuid4().hex[:8]}_{business}"


def _batch_task_id(tenant_id: int, kb_id: str, task_type: str) -> str:
    """Generate a tenant-embedded batch task id mirroring the move shape."""
    millis = int(datetime.now(UTC).timestamp() * 1000)
    business = kb_id.replace("-", "").replace("_", "")[:12]
    return f"{task_type}_{tenant_id}_{millis}_{uuid.uuid4().hex[:8]}_{business}"


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


# ── KB-scoped creates + list ──────────────────────────────────────────


@kb_documents_router.post("/file", response_model=KnowledgeEnvelope, status_code=200)
async def create_file_document(
    _auth: AuthDep,
    _contributor: RoleContributorDep,
    id: str,
    file: Annotated[UploadFile, File()],
    service: KnowledgeDocumentsDep,
    tenant_id: _PrincipalTenant,
    file_name: str = Form(default="", description="自定义文件名"),
    metadata: str = Form(default="", description="元数据JSON"),
    enable_multimodel: str = Form(default="", description="启用多模态处理"),
    tag_ids: str = Form(default="", description="分类ID列表，逗号分隔"),
    process_config: str = Form(default="", description="处理配置JSON"),
    channel: str = Form(default="", description="来源渠道"),
) -> KnowledgeEnvelope:
    """Upload a file and create a document entry from it."""
    tenant_id = _require_tenant(tenant_id)
    knowledge = await service.create_from_file(
        tenant_id=tenant_id,
        knowledge_base_id=id,
        file=file,  # type: ignore[arg-type]  # UploadFile satisfies the FileUpload protocol at runtime
        metadata=_parse_json_object(metadata, "metadata"),
        enable_multimodel=_parse_optional_bool(enable_multimodel, "enable_multimodel"),
        custom_file_name=file_name or None,
        tag_ids=_split_tag_ids(tag_ids),
        channel=channel,
        process_overrides=_parse_json_object(process_config, "process_config"),
    )
    return KnowledgeEnvelope(success=True, data=knowledge)


@kb_documents_router.post("/url", response_model=KnowledgeEnvelope, status_code=201)
async def create_url_document(
    _auth: AuthDep,
    _contributor: RoleContributorDep,
    id: str,
    body: CreateKnowledgeFromURLRequest,
    service: KnowledgeDocumentsDep,
    tenant_id: _PrincipalTenant,
) -> KnowledgeEnvelope:
    """Create a document from a URL (web page or downloadable file)."""
    tenant_id = _require_tenant(tenant_id)
    knowledge = await service.create_from_url(
        tenant_id=tenant_id,
        kb_id=id,
        url=body.url,
        file_name=body.file_name,
        file_type=body.file_type,
        enable_multimodel=body.enable_multimodel,
        title=body.title,
        tag_ids=[body.tag_id] if body.tag_id else None,
        channel=body.channel,
    )
    return KnowledgeEnvelope(success=True, data=knowledge)


@kb_documents_router.post("/passage", response_model=KnowledgeEnvelope, status_code=201)
async def create_passage_document(
    _auth: AuthDep,
    _contributor: RoleContributorDep,
    id: str,
    body: CreatePassageKnowledgeRequest,
    service: KnowledgeDocumentsDep,
    tenant_id: _PrincipalTenant,
) -> KnowledgeEnvelope:
    """Create a document from text passages."""
    tenant_id = _require_tenant(tenant_id)
    knowledge = await service.create_from_passage(
        tenant_id=tenant_id,
        kb_id=id,
        passages=body.passages,
        channel=body.channel,
        sync=body.sync,
    )
    return KnowledgeEnvelope(success=True, data=knowledge)


@kb_documents_router.post("/manual", response_model=KnowledgeEnvelope, status_code=200)
async def create_manual_document(
    _auth: AuthDep,
    _contributor: RoleContributorDep,
    id: str,
    body: CreateManualKnowledgeRequest,
    service: KnowledgeDocumentsDep,
    tenant_id: _PrincipalTenant,
) -> KnowledgeEnvelope:
    """Create a manual Markdown document."""
    tenant_id = _require_tenant(tenant_id)
    knowledge = await service.create_from_manual(
        tenant_id=tenant_id,
        kb_id=id,
        title=body.title,
        content=body.content,
        status=body.status,
        tag_ids=[body.tag_id] if body.tag_id else None,
        channel=body.channel,
    )
    return KnowledgeEnvelope(success=True, data=knowledge)


@kb_documents_router.get("", response_model=KnowledgeListEnvelope)
async def list_documents(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    id: str,
    service: KnowledgeServiceDep,
    tenant_id: _PrincipalTenant,
    page: int = Query(default=1, ge=1, description="页码"),
    page_size: int = Query(default=20, ge=1, le=100, description="每页数量"),
    tag_ids: str = Query(default="", description="标签ID筛选，逗号分隔"),
    keyword: str | None = Query(default=None, description="关键词搜索"),
    file_type: str | None = Query(default=None, description="文件类型筛选"),
    parse_status: str | None = Query(default=None, description="解析状态筛选"),
    source: str | None = Query(default=None, description="来源/渠道筛选"),
    start_time: str | None = Query(default=None, description="更新时间起点，RFC3339 格式"),
    end_time: str | None = Query(default=None, description="更新时间终点，RFC3339 格式"),
) -> KnowledgeListEnvelope:
    """List the knowledge base's documents, paged and filterable."""
    tenant_id = _require_tenant(tenant_id)
    list_filter = DocumentListFilter(
        tag_ids=_split_tag_ids(tag_ids),
        keyword=keyword,
        file_type=file_type,
        parse_status=parse_status,
        source=source,
        updated_from=_parse_filter_time(start_time, "start_time"),
        updated_to=_parse_filter_time(end_time, "end_time"),
    )
    result = await service.list_documents_paged(
        tenant_id=tenant_id,
        knowledge_base_id=id,
        page=page,
        page_size=page_size,
        list_filter=list_filter,
    )
    return KnowledgeListEnvelope(
        success=True,
        total=result.total,
        page=result.page,
        page_size=result.page_size,
        data=result.data,
    )


# ── Cross-document static routes (declared before /{id}) ──────────────


@documents_router.get("/batch", response_model=KnowledgeBatchEnvelope)
async def batch_query_documents(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    service: KnowledgeServiceDep,
    tenant_id: _PrincipalTenant,
    ids: Annotated[list[str] | None, Query()] = None,
    kb_id: str | None = Query(default=None),
    agent_id: str | None = Query(default=None),
    agent_source_tenant_id: str | None = Query(default=None),
) -> KnowledgeBatchEnvelope:
    """Return documents for ``ids``. Agent query params are accepted unused."""
    del agent_id, agent_source_tenant_id
    return await document_reads.batch_query(
        service=service,
        tenant_id=_require_tenant(tenant_id),
        ids=ids or [],
        kb_id=kb_id,
        max_batch=_MAX_BATCH,
    )


@documents_router.get("/search", response_model=KnowledgeSearchEnvelope)
async def search_knowledge_documents(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    service: KnowledgeServiceDep,
    tenant_id: _PrincipalTenant,
    params: Annotated[document_reads.SearchQuery, Depends()],
) -> KnowledgeSearchEnvelope:
    """Keyword file search for the chat @ picker. Agent params are unused."""
    return await document_reads.search_knowledge_documents(
        service=service,
        tenant_id=_require_tenant(tenant_id),
        params=params,
    )


@documents_router.post("/batch-delete", response_model=BatchDeleteEnvelope)
async def batch_delete_documents(
    _auth: AuthDep,
    _contributor: RoleContributorDep,
    body: BatchDeleteRequest,
    service: KnowledgeServiceDep,
    orchestrator: KnowledgeDocumentsDep,
    kb_service: KBServiceDep,
    tenant_id: _PrincipalTenant,
) -> BatchDeleteEnvelope:
    """Soft-delete a batch of documents under one knowledge base."""
    tenant_id = _require_tenant(tenant_id)
    ids = _dedupe_ids(body.ids)
    if not ids:
        raise ValidationError(
            code="knowledge.batch_ids_required",
            message="ids cannot be empty",
        )
    if len(ids) > _MAX_BATCH:
        raise ValidationError(
            code="knowledge.batch_too_many",
            message=f"too many ids (max {_MAX_BATCH} per batch)",
        )
    await kb_service.get_knowledge_base_by_id_and_tenant(
        tenant_id=tenant_id,
        knowledge_base_id=body.kb_id,
    )
    documents = await service.get_documents(tenant_id=tenant_id, ids=ids)
    if len(documents) != len(ids):
        raise ValidationError(
            code="knowledge.batch_not_found",
            message="One or more knowledge entries not found",
        )
    for document in documents:
        if document.knowledge_base_id != body.kb_id:
            raise ValidationError(
                code="knowledge.batch_cross_kb",
                message=(f"Knowledge {document.id} does not belong to knowledge base {body.kb_id}"),
            )
    await orchestrator.delete_documents(tenant_id=tenant_id, ids=ids)
    task_id = _batch_task_id(tenant_id, body.kb_id, _BATCH_DELETE_TASK_TYPE)
    return BatchDeleteEnvelope(
        success=True,
        message=_BATCH_DELETE_MESSAGE,
        data=BatchDeleteData(task_id=task_id, deleted_count=len(ids)),
    )


@documents_router.post("/batch-reparse", response_model=BatchReparseEnvelope)
async def batch_reparse_documents(
    _auth: AuthDep,
    _contributor: RoleContributorDep,
    body: BatchReparseRequest,
    service: KnowledgeServiceDep,
    orchestrator: KnowledgeDocumentsDep,
    kb_service: KBServiceDep,
    tenant_id: _PrincipalTenant,
) -> BatchReparseEnvelope:
    """Reset and re-submit a batch of documents under one knowledge base."""
    tenant_id = _require_tenant(tenant_id)
    ids = _dedupe_ids(body.ids)
    if not ids:
        raise ValidationError(
            code="knowledge.batch_ids_required",
            message="no knowledge IDs provided for batch reparse",
        )
    if len(ids) > _MAX_BATCH:
        raise ValidationError(
            code="knowledge.batch_too_many",
            message=f"too many ids (max {_MAX_BATCH} per batch)",
        )
    await kb_service.get_knowledge_base_by_id_and_tenant(
        tenant_id=tenant_id,
        knowledge_base_id=body.kb_id,
    )
    documents = await service.get_documents(tenant_id=tenant_id, ids=ids)
    if len(documents) != len(ids):
        raise ValidationError(
            code="knowledge.batch_not_found",
            message="some knowledge entries were not found",
        )
    for document in documents:
        if document.knowledge_base_id != body.kb_id:
            raise ValidationError(
                code="knowledge.batch_cross_kb",
                message=(f"Knowledge {document.id} does not belong to knowledge base {body.kb_id}"),
            )
    for document in documents:
        await orchestrator.reparse(
            tenant_id=tenant_id,
            knowledge_id=document.id,
            process_overrides=body.process_config,
        )
    task_id = _batch_task_id(tenant_id, body.kb_id, _BATCH_REPARSE_TASK_TYPE)
    return BatchReparseEnvelope(
        success=True,
        message=_BATCH_REPARSE_MESSAGE,
        data=BatchReparseData(task_id=task_id, reparse_count=len(ids)),
    )


@documents_router.put("/tags", response_model=KnowledgeTagBatchEnvelope)
async def update_knowledge_tag_batch(
    _auth: AuthDep,
    _contributor: RoleContributorDep,
    body: KnowledgeTagBatchUpdateRequest,
    service: KnowledgeServiceDep,
    tag_service: TagServiceDep,
    kb_service: KBServiceDep,
    tenant_id: _PrincipalTenant,
) -> KnowledgeTagBatchEnvelope:
    """Replace the tag bindings of many documents in one request."""
    tenant_id = _require_tenant(tenant_id)
    updates = body.updates
    if not updates:
        raise ValidationError(
            code="knowledge.tags_updates_required",
            message="请求参数不合法",
        )
    documents = await service.get_documents(tenant_id=tenant_id, ids=list(updates))
    if len(documents) != len(updates):
        raise PermissionDeniedError(
            code="knowledge.tags_knowledge_not_found",
            message="some knowledge IDs are not accessible in the authorized scope",
        )
    by_id = {document.id: document for document in documents}

    if body.kb_id:
        await kb_service.get_knowledge_base_by_id_and_tenant(
            tenant_id=tenant_id,
            knowledge_base_id=body.kb_id,
        )
        for document in documents:
            if document.knowledge_base_id != body.kb_id:
                raise PermissionDeniedError(
                    code="knowledge.tags_cross_kb",
                    message=(
                        f"knowledge {document.id} does not belong to authorized knowledge base"
                    ),
                )

    tag_ids = {tag_id for ids in updates.values() for tag_id in ids if tag_id}
    tags = await tag_service.get_tags_by_ids(
        tenant_id=tenant_id,
        ids=sorted(tag_ids),
    )
    for knowledge_id, tag_ids_for_knowledge in updates.items():
        document = by_id[knowledge_id]
        for tag_id in tag_ids_for_knowledge:
            if not tag_id:
                continue
            tag = tags.get(tag_id)
            if tag is None:
                raise ValidationError(
                    code="knowledge.tags_not_found",
                    message=f"标签 {tag_id} 不存在",
                )
            if tag.knowledge_base_id != document.knowledge_base_id:
                raise ValidationError(
                    code="knowledge.tags_cross_kb",
                    message=(f"标签 {tag_id} 不属于知识库 {document.knowledge_base_id}"),
                )

    for knowledge_id, tag_ids_for_knowledge in updates.items():
        await tag_service.set_knowledge_tags(
            knowledge_id=knowledge_id,
            tag_ids=tag_ids_for_knowledge,
        )
    return KnowledgeTagBatchEnvelope(success=True)


# ── Per-document routes (declared after the static /move group) ──────


@documents_router.get("/{id}", response_model=KnowledgeEnvelope)
async def get_document(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    id: str,
    service: KnowledgeServiceDep,
    tenant_id: _PrincipalTenant,
) -> KnowledgeEnvelope:
    """Return one document within the caller's workspace scope."""
    tenant_id = _require_tenant(tenant_id)
    knowledge = await service.get_document(tenant_id=tenant_id, id=id)
    return KnowledgeEnvelope(success=True, data=knowledge)


@documents_router.get("/{id}/spans", response_model=KnowledgeSpansEnvelope)
async def get_document_spans(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    id: str,
    service: KnowledgeServiceDep,
    tracker: SpanTrackerDep,
    tenant_id: _PrincipalTenant,
    attempt: int | None = Query(default=None, ge=1),
) -> KnowledgeSpansEnvelope:
    """Return the processing-span tree for one document."""
    return await document_reads.read_spans(
        service=service,
        tracker=tracker,
        tenant_id=_require_tenant(tenant_id),
        knowledge_id=id,
        attempt=attempt,
    )


@documents_router.post("/{id}/regenerate-summary", response_model=KnowledgeTaskEnvelope)
async def regenerate_document_summary(
    _auth: AuthDep,
    _contributor: RoleContributorDep,
    id: str,
    service: KnowledgeServiceDep,
    tenant_id: _PrincipalTenant,
) -> KnowledgeTaskEnvelope:
    """Queue a summary refresh for one document."""
    return await document_reads.regenerate_summary(
        service=service,
        tenant_id=_require_tenant(tenant_id),
        knowledge_id=id,
    )


@documents_router.put("/{id}", response_model=KnowledgeUpdatedEnvelope)
async def update_document(
    _auth: AuthDep,
    _contributor: RoleContributorDep,
    id: str,
    body: UpdateKnowledgeRequest,
    service: KnowledgeServiceDep,
    tenant_id: _PrincipalTenant,
) -> KnowledgeUpdatedEnvelope:
    """Update a document's mutable fields, including manual content."""
    return await document_reads.update_document(
        service=service,
        tenant_id=_require_tenant(tenant_id),
        knowledge_id=id,
        body=body,
    )


async def _stream_knowledge_file(
    service: KnowledgeServiceDep,
    session: SessionDep,
    tenant_id: int,
    knowledge_id: str,
    disposition: Literal["attachment", "inline"],
) -> Response:
    return await document_reads.stream_document(
        service=service,
        session=session,
        tenant_id=_require_tenant(tenant_id),
        knowledge_id=knowledge_id,
        disposition=disposition,
    )


@documents_router.get("/{id}/download")
async def download_document(
    _auth: AuthDep,
    _contributor: RoleContributorDep,
    id: str,
    service: KnowledgeServiceDep,
    session: SessionDep,
    tenant_id: _PrincipalTenant,
) -> Response:
    return await _stream_knowledge_file(service, session, tenant_id, id, "attachment")


@documents_router.get("/{id}/preview")
async def preview_document(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    id: str,
    service: KnowledgeServiceDep,
    session: SessionDep,
    tenant_id: _PrincipalTenant,
) -> Response:
    return await _stream_knowledge_file(service, session, tenant_id, id, "inline")


@documents_router.delete("/{id}", response_model=DeleteEnvelope)
async def delete_document(
    _auth: AuthDep,
    _contributor: RoleContributorDep,
    id: str,
    service: KnowledgeDocumentsDep,
    tenant_id: _PrincipalTenant,
) -> DeleteEnvelope:
    return await document_reads.delete_document(
        service=service,
        tenant_id=_require_tenant(tenant_id),
        knowledge_id=id,
    )


@documents_router.post("/{id}/reparse", response_model=KnowledgeTaskEnvelope)
async def reparse_document(
    _auth: AuthDep,
    _contributor: RoleContributorDep,
    id: str,
    service: KnowledgeDocumentsDep,
    tenant_id: _PrincipalTenant,
    body: ReparseRequest | None = None,
) -> KnowledgeTaskEnvelope:
    return await document_reads.reparse_document(
        service=service,
        tenant_id=_require_tenant(tenant_id),
        knowledge_id=id,
        body=body,
    )


@documents_router.post("/{id}/cancel-parse", response_model=KnowledgeTaskEnvelope)
async def cancel_document_parse(
    _auth: AuthDep,
    _contributor: RoleContributorDep,
    id: str,
    service: KnowledgeDocumentsDep,
    tenant_id: _PrincipalTenant,
) -> KnowledgeTaskEnvelope:
    return await document_reads.cancel_document_parse(
        service=service,
        tenant_id=_require_tenant(tenant_id),
        knowledge_id=id,
    )


@documents_router.post("/{id}/clone", response_model=KnowledgeEnvelope)
async def clone_document(
    _auth: AuthDep,
    _contributor: RoleContributorDep,
    id: str,
    body: CloneKnowledgeRequest,
    service: KnowledgeDocumentsDep,
    tenant_id: _PrincipalTenant,
) -> KnowledgeEnvelope:
    """Clone a completed document into another knowledge base."""
    tenant_id = _require_tenant(tenant_id)
    knowledge = await service.clone(
        tenant_id=tenant_id,
        knowledge_id=id,
        target_kb_id=body.target_kb_id,
    )
    if knowledge is None:
        raise ValidationError(
            code="knowledge.clone_not_completed",
            message="source knowledge is not in completed status",
        )
    return KnowledgeEnvelope(success=True, data=knowledge)


# ── Cross-KB move (static paths declared before /{id}) ────────────────


@documents_router.post("/move", response_model=MoveEnvelope)
async def move_documents(
    _auth: AuthDep,
    _contributor: RoleContributorDep,
    body: KnowledgeMoveRequest,
    service: KnowledgeDocumentsDep,
    tenant_id: _PrincipalTenant,
) -> MoveEnvelope:
    """Move documents into another knowledge base (runs synchronously)."""
    tenant_id = _require_tenant(tenant_id)
    if body.source_kb_id == body.target_kb_id:
        raise ValidationError(
            code="knowledge.move_same_kb",
            message="Source and target knowledge base cannot be the same",
        )
    if not body.knowledge_ids:
        raise ValidationError(
            code="knowledge.move_ids_required",
            message="knowledge_ids cannot be empty",
        )
    for knowledge_id in body.knowledge_ids:
        await service.move(
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
            source_kb_id=body.source_kb_id,
            target_kb_id=body.target_kb_id,
            mode=body.mode,
        )
    task_id = _move_task_id(tenant_id, body.source_kb_id)
    return MoveEnvelope(
        success=True,
        data=KnowledgeMoveResponse(
            task_id=task_id,
            source_kb_id=body.source_kb_id,
            target_kb_id=body.target_kb_id,
            knowledge_count=len(body.knowledge_ids),
            message=_MOVE_STARTED_MESSAGE,
        ),
    )


@documents_router.get("/move/progress/{task_id}")
async def get_move_progress(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    _guard: TaskProgressTenantDep,
    task_id: str,
) -> None:
    return document_reads.move_progress_missing()


__all__ = ["documents_router", "kb_documents_router"]
