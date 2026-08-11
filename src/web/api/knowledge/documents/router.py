"""Knowledge document HTTP endpoints.

Document (knowledge) lifecycle: upload creates (file / URL / passage /
manual), CRUD on ``/knowledge``, and the lifecycle actions (reparse,
cancel-parse, clone, move) plus the move-progress read. Route and role
mapping follows the upstream knowledge handler:

- KB-scoped uploads and list live under ``/knowledge-bases/{id}/knowledge``;
- per-document routes address ``/knowledge/{id}``;
- ``/move`` and ``/move/progress/{task_id}`` sit outside the ``{id}``
  group so a literal ``move`` segment is never captured as an id;
- reads are Viewer+, every mutation Contributor+.

Cross-workspace ids read as 404 rather than 403 so the id space is not
enumerable; the move-progress task id carries the owning workspace and
is guarded by ``require_task_progress_tenant`` before any read.

Query-parameter descriptions are intentionally Chinese (mirrors the
upstream swagger annotations). RUF001 flags the full-width punctuation;
suppressed file-wide for the same reason as ``src/web/api/system/router.py``.
"""
# ruff: noqa: RUF001

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile

from src.common.exception import NotFoundError, UnauthorizedError, ValidationError
from src.common.json import JsonObject
from src.core.contracts.knowledge import (
    CreateKnowledgeFromURLRequest,
    CreateManualKnowledgeRequest,
    KnowledgeMoveRequest,
    KnowledgeMoveResponse,
    UpdateKnowledgeRequest,
)
from src.core.knowledge.documents.types import DocumentListFilter
from src.web.api.knowledge.documents.views import (
    CloneKnowledgeRequest,
    CreatePassageKnowledgeRequest,
    DeleteEnvelope,
    DeleteResult,
    KnowledgeEnvelope,
    KnowledgeListEnvelope,
    KnowledgeTaskEnvelope,
    KnowledgeUpdatedEnvelope,
    MoveEnvelope,
    ReparseRequest,
)
from src.web.deps import AuthDep, RoleContributorDep, RoleViewerDep
from src.web.deps.context import get_tenant_id_dep
from src.web.deps.knowledge_documents import (
    KnowledgeDocumentsDep,
    KnowledgeServiceDep,
)
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
        part.strip()
        for part in raw.split(",")
        if part.strip() and part.strip() != "__untagged__"
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


@documents_router.put("/{id}", response_model=KnowledgeUpdatedEnvelope)
async def update_document(
    _auth: AuthDep,
    _contributor: RoleContributorDep,
    id: str,
    body: UpdateKnowledgeRequest,
    service: KnowledgeServiceDep,
    tenant_id: _PrincipalTenant,
) -> KnowledgeUpdatedEnvelope:
    """Update a document's mutable fields (title / description)."""
    tenant_id = _require_tenant(tenant_id)
    knowledge = await service.update_document(
        tenant_id=tenant_id,
        id=id,
        title=body.title,
        description=body.description,
    )
    return KnowledgeUpdatedEnvelope(
        success=True,
        message="Knowledge updated successfully",
        data=knowledge,
    )


@documents_router.delete("/{id}", response_model=DeleteEnvelope)
async def delete_document(
    _auth: AuthDep,
    _contributor: RoleContributorDep,
    id: str,
    service: KnowledgeDocumentsDep,
    tenant_id: _PrincipalTenant,
) -> DeleteEnvelope:
    """Soft-delete a document and cascade its chunks."""
    tenant_id = _require_tenant(tenant_id)
    deleted = await service.delete(tenant_id=tenant_id, id=id)
    return DeleteEnvelope(
        success=True,
        message="Knowledge deleted",
        data=DeleteResult(deleted=deleted),
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
    """Reset a document for a fresh parse attempt."""
    tenant_id = _require_tenant(tenant_id)
    knowledge = await service.reparse(
        tenant_id=tenant_id,
        knowledge_id=id,
        process_overrides=body.process_config if body is not None else None,
    )
    return KnowledgeTaskEnvelope(
        success=True,
        message="Knowledge reparse task submitted",
        data=knowledge,
    )


@documents_router.post("/{id}/cancel-parse", response_model=KnowledgeTaskEnvelope)
async def cancel_document_parse(
    _auth: AuthDep,
    _contributor: RoleContributorDep,
    id: str,
    service: KnowledgeDocumentsDep,
    tenant_id: _PrincipalTenant,
) -> KnowledgeTaskEnvelope:
    """Cancel an in-flight document parse."""
    tenant_id = _require_tenant(tenant_id)
    knowledge = await service.cancel_parse(tenant_id=tenant_id, knowledge_id=id)
    return KnowledgeTaskEnvelope(
        success=True,
        message="Knowledge parse cancelled",
        data=knowledge,
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
    """Move documents into another knowledge base (runs synchronously).

    The response carries a workspace-embedded task id so the progress
    endpoint keeps the upstream wire contract; the async task record and
    broker land with the task infrastructure.
    """
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
    """Return a move task's progress, scoped to the caller's workspace.

    The tenant guard rejects malformed task ids and hides cross-workspace
    tasks as not-found. Progress records land with the async task
    infrastructure; until then a well-formed in-workspace task resolves
    to no record and answers not-found.
    """
    raise NotFoundError(code="task_progress.not_found", message="task not found")


__all__ = ["documents_router", "kb_documents_router"]
