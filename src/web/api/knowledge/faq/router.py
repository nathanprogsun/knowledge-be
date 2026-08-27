"""FAQ HTTP endpoints — knowledge-base-scoped entries plus import progress.

Maps the FAQ endpoints from the upstream handler. Entries are knowledge-base
content: reads are Viewer+, every mutation (create / update / delete /
batch import) is Contributor+ — mirroring the upstream route guards.

Route order matters: the static ``/entries/export`` path is declared before
the ``/entries/{entry_id}``-shaped routes so a literal segment is never
captured as an entry id.

Scope notes (this build):

- ``POST /entries`` is the FAQ batch import. The merged import pipeline
  consumes a CSV / Excel file, so this route binds a file upload plus the
  ``mode`` / ``dry_run`` switches; the JSON batch-upsert body of the
  upstream handler is not yet wired.
- Search, similar-question append, batch field / tag updates, and the
  import-result display switch need not-yet-merged services and are absent.
- The FAQ container knowledge id for entry writes is resolved from the
  knowledge base's documents (the FAQ knowledge). Without a container the
  write fails with ``faq.knowledge_container_missing``.

Swagger ``description`` strings are Chinese, mirroring the upstream
annotations; RUF001 is suppressed file-wide for the same reason as
``src/web/api/system/router.py``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Query, Response, UploadFile

from src.common.exception import NotFoundError, UnauthorizedError, ValidationError
from src.core.contracts.knowledge import FAQBatchDeleteRequest, FAQEntryPayload
from src.core.knowledge.documents.faq_import import (
    FAQ_BATCH_MODE_APPEND,
    FAQ_BATCH_MODE_REPLACE,
)
from src.core.knowledge.faq.task_ids import task_tenant_id
from src.web.api.knowledge.faq.views import (
    DeleteFAQEntriesResponse,
    FAQEntryEnvelope,
    FAQEntryListEnvelope,
    FAQImportProgressEnvelope,
    build_export_csv,
    build_export_json,
    collect_all_entries,
    delete_ack,
    entry_envelope,
    import_progress_envelope,
    list_envelope,
    resolve_faq_knowledge_id,
)
from src.web.deps import AuthDep, RoleContributorDep, RoleViewerDep
from src.web.deps.context import get_tenant_id_dep
from src.web.deps.knowledge import KnowledgeServiceDep
from src.web.deps.knowledge_faq import FAQImportRunnerDep, FAQServiceDep

# Function-arg-style principal dependency aliases.
_PrincipalTenant = Annotated[int, Depends(get_tenant_id_dep)]

# Multipart parameter aliases (module-level markers for the B008 rule).
ImportFile = Annotated[UploadFile, File(description="FAQ 导入文件（CSV / Excel）")]
ModeField = Annotated[str, Form(description="导入模式：append 或 replace")]
DryRunField = Annotated[bool, Form(description="仅验证，不实际导入")]

# UTF-8 BOM prepended to the CSV export for Excel compatibility.
_CSV_BOM = b"\xef\xbb\xbf"

#: Page-size cap shared by every FAQ list. Mirrors the upstream handler's
#: ``maxListPageSize = 100`` constant; values above 100 are rejected by the
#: Pagination contract.
_MAX_PAGE_SIZE = 100


router = APIRouter(prefix="/knowledge-bases/{id}/faq", tags=["knowledge.faq"])
import_progress_router = APIRouter(prefix="/faq/import/progress", tags=["knowledge.faq"])


def _require_tenant(tenant_id: int) -> int:
    """Return the active workspace id, or fail.

    A FAQ entry is always workspace-scoped; without a tenant context there
    is no safe default (tenant 0 is the system scope), so this rejects
    rather than guessing.
    """
    if tenant_id == 0:
        raise UnauthorizedError(
            code="auth.tenant_context_missing",
            message="unauthorized: workspace context missing",
        )
    return tenant_id


def _require_task_progress_tenant(task_id: str, tenant_id: int) -> None:
    """Gate: the import task must belong to the caller's workspace.

    Mirrors the upstream ``requireTaskProgressTenant`` guard: the tenant is
    recovered from the task id, so a cross-workspace probe reads as
    not-found rather than confirming the task exists. A malformed id is a
    client error; a missing caller workspace is unauthorized.
    """
    task_tenant = task_tenant_id(task_id)
    if task_tenant is None:
        raise ValidationError(
            code="faq.invalid_task_id",
            message="invalid task ID",
        )
    if tenant_id == 0:
        raise UnauthorizedError(
            code="auth.tenant_context_missing",
            message="unauthorized: workspace context missing",
        )
    if task_tenant != tenant_id:
        raise NotFoundError(
            code="faq.task_not_found",
            message="task not found",
        )


def _reject_unsupported_filters(
    *,
    tag_id: int | None,
    tag_ids: str,
    search_field: str | None,
    sort_order: str | None,
) -> None:
    """Reject list filters the merged service cannot honour.

    Tag filtering belongs to the tag-relation layer and field-specific /
    custom sorting are not applied by the FAQ list query yet. Failing
    loudly mirrors the knowledge list behaviour rather than silently
    returning an unfiltered page.
    """
    if tag_id not in (None, 0) or tag_ids.strip():
        raise ValidationError(
            code="faq.tag_filter_unsupported",
            message="标签过滤由标签域服务处理，暂未开放",
        )
    if search_field:
        raise ValidationError(
            code="faq.search_field_unsupported",
            message="分字段搜索暂未开放，仅支持标准问关键词过滤",
        )
    if sort_order:
        raise ValidationError(
            code="faq.sort_order_unsupported",
            message="自定义排序暂未开放",
        )


# ── Read ────────────────────────────────────────────────────────────


@router.get("/entries", response_model=FAQEntryListEnvelope)
async def list_faq_entries(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    id: str,
    faq_service: FAQServiceDep,
    tenant_id: _PrincipalTenant,
    page: int = Query(default=1, ge=1, description="页码"),
    page_size: int = Query(default=20, ge=1, le=_MAX_PAGE_SIZE, description="每页数量"),
    tag_id: int | None = Query(default=None, description="标签ID筛选(seq_id)，兼容旧版单标签"),
    tag_ids: str = Query(default="", description="标签UUID筛选，逗号分隔（OR语义）"),
    keyword: str | None = Query(default=None, description="关键词搜索"),
    search_field: str | None = Query(
        default=None,
        description="搜索字段: standard_question(标准问题), similar_questions(相似问法), "
        "answers(答案), 默认搜索全部",
    ),
    sort_order: str | None = Query(
        default=None,
        description="排序方式: asc(按更新时间正序), 默认按更新时间倒序",
    ),
) -> FAQEntryListEnvelope:
    """List one page of the knowledge base's FAQ entries."""
    tenant_id = _require_tenant(tenant_id)
    _reject_unsupported_filters(
        tag_id=tag_id,
        tag_ids=tag_ids,
        search_field=search_field,
        sort_order=sort_order,
    )
    response = await faq_service.list_entries(
        tenant_id=tenant_id,
        knowledge_base_id=id,
        keyword=keyword or None,
        limit=page_size,
        offset=(page - 1) * page_size,
    )
    return list_envelope(response)


@router.get("/entries/export")
async def export_faq_entries(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    id: str,
    faq_service: FAQServiceDep,
    tenant_id: _PrincipalTenant,
    format: str = Query(default="csv", description="导出格式：csv（默认）或 json"),
) -> Response:
    """Export every FAQ entry as CSV (default) or JSON.

    CSV rows follow the import template so the file can be edited and
    re-imported; JSON is an array of entry-payload-compatible objects.
    """
    tenant_id = _require_tenant(tenant_id)
    entries = await collect_all_entries(
        faq_service,
        tenant_id=tenant_id,
        knowledge_base_id=id,
    )
    export_format = (format or "csv").strip().lower()
    if export_format == "json":
        return Response(
            content=build_export_json(entries),
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=faq_export.json"},
        )
    if export_format != "csv":
        raise ValidationError(
            code="faq.invalid_export_format",
            message="导出格式仅支持 csv 或 json",
        )
    return Response(
        content=_CSV_BOM + build_export_csv(entries),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=faq_export.csv"},
    )


@router.get("/entries/{entry_id}", response_model=FAQEntryEnvelope)
async def get_faq_entry(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    id: str,
    entry_id: int,
    faq_service: FAQServiceDep,
    tenant_id: _PrincipalTenant,
) -> FAQEntryEnvelope:
    """Return one FAQ entry; unknown or foreign entries read as 404."""
    tenant_id = _require_tenant(tenant_id)
    entry = await faq_service.get_entry(
        tenant_id=tenant_id,
        knowledge_base_id=id,
        entry_id=entry_id,
    )
    return entry_envelope(entry)


# ── Mutations ───────────────────────────────────────────────────────


@router.post("/entry", response_model=FAQEntryEnvelope)
async def create_faq_entry(
    _auth: AuthDep,
    _contributor: RoleContributorDep,
    id: str,
    body: FAQEntryPayload,
    faq_service: FAQServiceDep,
    knowledge_service: KnowledgeServiceDep,
    tenant_id: _PrincipalTenant,
) -> FAQEntryEnvelope:
    """Create one FAQ entry under the knowledge base's FAQ container."""
    tenant_id = _require_tenant(tenant_id)
    knowledge_id = await resolve_faq_knowledge_id(
        knowledge_service,
        tenant_id=tenant_id,
        knowledge_base_id=id,
    )
    entry = await faq_service.create_entry(
        tenant_id=tenant_id,
        knowledge_base_id=id,
        knowledge_id=knowledge_id,
        payload=body,
    )
    return entry_envelope(entry)


@router.put("/entries/{entry_id}", response_model=FAQEntryEnvelope)
async def update_faq_entry(
    _auth: AuthDep,
    _contributor: RoleContributorDep,
    id: str,
    entry_id: int,
    body: FAQEntryPayload,
    faq_service: FAQServiceDep,
    tenant_id: _PrincipalTenant,
) -> FAQEntryEnvelope:
    """Update one FAQ entry's content and flags."""
    tenant_id = _require_tenant(tenant_id)
    entry = await faq_service.update_entry(
        tenant_id=tenant_id,
        knowledge_base_id=id,
        entry_id=entry_id,
        payload=body,
    )
    return entry_envelope(entry)


@router.delete("/entries", response_model=DeleteFAQEntriesResponse)
async def delete_faq_entries(
    _auth: AuthDep,
    _contributor: RoleContributorDep,
    id: str,
    body: FAQBatchDeleteRequest,
    faq_service: FAQServiceDep,
    tenant_id: _PrincipalTenant,
) -> DeleteFAQEntriesResponse:
    """Batch-delete FAQ entries; any foreign or unknown id fails the batch."""
    tenant_id = _require_tenant(tenant_id)
    await faq_service.delete_entries(
        tenant_id=tenant_id,
        knowledge_base_id=id,
        entry_ids=body.ids,
    )
    return delete_ack()


@router.post("/entries", response_model=FAQImportProgressEnvelope)
async def import_faq_entries(
    _auth: AuthDep,
    _contributor: RoleContributorDep,
    id: str,
    runner: FAQImportRunnerDep,
    knowledge_service: KnowledgeServiceDep,
    tenant_id: _PrincipalTenant,
    file: ImportFile,
    mode: ModeField = FAQ_BATCH_MODE_APPEND,
    dry_run: DryRunField = False,
) -> FAQImportProgressEnvelope:
    """Import FAQ entries from a CSV / Excel file into the knowledge base.

    The import runs synchronously; the returned progress describes the
    completed task, which the progress endpoint reads back by ``task_id``.
    ``dry_run`` validates without persisting.
    """
    tenant_id = _require_tenant(tenant_id)
    if mode not in (FAQ_BATCH_MODE_APPEND, FAQ_BATCH_MODE_REPLACE):
        raise ValidationError(
            code="faq.invalid_import_mode",
            message="模式仅支持 append 或 replace",
        )
    knowledge_id = await resolve_faq_knowledge_id(
        knowledge_service,
        tenant_id=tenant_id,
        knowledge_base_id=id,
    )
    file_data = await file.read()
    progress = await runner.run(
        file_data=file_data,
        filename=file.filename or "",
        tenant_id=tenant_id,
        knowledge_base_id=id,
        knowledge_id=knowledge_id,
        mode=mode,
        dry_run=dry_run,
    )
    return import_progress_envelope(progress)


# ── Import progress (outside the knowledge-base scope) ─────────────


@import_progress_router.get("/{task_id}", response_model=FAQImportProgressEnvelope)
async def get_faq_import_progress(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    task_id: str,
    runner: FAQImportRunnerDep,
    tenant_id: _PrincipalTenant,
) -> FAQImportProgressEnvelope:
    """Return one FAQ import task's progress (Viewer+).

    The task is tenant-scoped by its embedded tenant id; a cross-workspace
    or unknown task reads as 404 so task existence is not enumerable.
    """
    _require_task_progress_tenant(task_id, tenant_id)
    progress = runner.get_progress(task_id)
    if progress is None:
        raise NotFoundError(
            code="faq.import_task_not_found",
            message="FAQ 导入任务不存在",
        )
    return import_progress_envelope(progress)


__all__ = ["import_progress_router", "router"]
