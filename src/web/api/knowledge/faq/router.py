"""FAQ HTTP endpoints — knowledge-base-scoped entries plus import progress.

Entries are knowledge-base content: reads are Viewer+, every mutation
(create / update / delete / batch import / batch patch) is Contributor+.

Route order matters: static paths such as ``/entries/export``,
``/entries/tags``, and ``/entries/fields`` are declared before
``/entries/{entry_id}`` so a literal segment is never captured as an
entry id.

``POST /entries`` branches on ``Content-Type``. ``application/json`` is
the SPA ``{entries, mode}`` upsert. Multipart is the CSV / Excel file
runner. The two bodies are not interchangeable.

The FAQ container knowledge id for entry writes is resolved from the
knowledge base's documents. A first write creates that container when
the knowledge base has none.

Swagger ``description`` strings are Chinese, mirroring the existing
annotations; RUF001 is suppressed file-wide for the same reason as
``src/web/api/system/router.py``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, Response

from src.common.exception import NotFoundError, UnauthorizedError, ValidationError
from src.common.json import JsonObject
from src.core.contracts.knowledge import (
    FAQBatchDeleteRequest,
    FAQBatchUpsertPayload,
    FAQEntryFieldsBatchUpdate,
    FAQEntryPayload,
    FAQEntryTagsBatchUpdate,
    FAQImportDisplayStatusRequest,
    FAQSearchRequest,
)
from src.core.knowledge.documents.faq_import import (
    FAQ_BATCH_MODE_APPEND,
    FAQ_BATCH_MODE_REPLACE,
)
from src.core.knowledge.documents.service.knowledge_service import KnowledgeService
from src.core.knowledge.faq.import_runner import FAQImportRunner
from src.core.knowledge.faq.service.faq_service import FAQService
from src.core.knowledge.faq.task_ids import task_tenant_id
from src.web.api.knowledge.faq.views import (
    DeleteFAQEntriesResponse,
    FAQAckResponse,
    FAQEntryEnvelope,
    FAQEntryListEnvelope,
    FAQImportProgressEnvelope,
    FAQSearchEnvelope,
    build_export_csv,
    build_export_json,
    collect_all_entries,
    delete_ack,
    entry_envelope,
    import_progress_envelope,
    inlined_json_schema,
    is_json_content_type,
    list_envelope,
    mutation_ack,
    read_json_upsert_payload,
    read_multipart_import,
    resolve_faq_knowledge_id,
    search_envelope,
)
from src.web.deps import AuthDep, RoleContributorDep, RoleViewerDep
from src.web.deps.context import get_tenant_id_dep
from src.web.deps.knowledge import KnowledgeServiceDep
from src.web.deps.knowledge_faq import FAQImportRunnerDep, FAQServiceDep

# Function-arg-style principal dependency aliases.
_PrincipalTenant = Annotated[int, Depends(get_tenant_id_dep)]

_DISPLAY_STATUSES: frozenset[str] = frozenset({"open", "close"})

_ENTRIES_POST_OPENAPI: JsonObject = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {"schema": inlined_json_schema(FAQBatchUpsertPayload)},
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                        "file": {
                            "type": "string",
                            "format": "binary",
                            "description": "FAQ 导入文件（CSV / Excel）",
                        },
                        "mode": {
                            "type": "string",
                            "description": "导入模式：append 或 replace",
                        },
                        "dry_run": {
                            "type": "boolean",
                            "description": "仅验证，不实际导入",
                        },
                    },
                }
            },
        },
    }
}

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


@router.put("/entries/tags", response_model=FAQAckResponse)
async def update_faq_entry_tags(
    _auth: AuthDep,
    _contributor: RoleContributorDep,
    id: str,
    body: FAQEntryTagsBatchUpdate,
    faq_service: FAQServiceDep,
    tenant_id: _PrincipalTenant,
) -> FAQAckResponse:
    """Set or clear tags for the given entry ids. ``null`` clears a tag."""
    tenant_id = _require_tenant(tenant_id)
    await faq_service.update_entry_tags(
        tenant_id=tenant_id,
        knowledge_base_id=id,
        updates=body.updates,
    )
    return mutation_ack()


@router.put("/entries/fields", response_model=FAQAckResponse)
async def update_faq_entry_fields(
    _auth: AuthDep,
    _contributor: RoleContributorDep,
    id: str,
    body: FAQEntryFieldsBatchUpdate,
    faq_service: FAQServiceDep,
    tenant_id: _PrincipalTenant,
) -> FAQAckResponse:
    """Batch-update enabled / recommended / tag fields by id or tag."""
    tenant_id = _require_tenant(tenant_id)
    await faq_service.update_entry_fields(
        tenant_id=tenant_id,
        knowledge_base_id=id,
        batch=body,
    )
    return mutation_ack()


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


@router.post(
    "/entries",
    response_model=FAQImportProgressEnvelope,
    openapi_extra=_ENTRIES_POST_OPENAPI,
)
async def import_faq_entries(
    request: Request,
    _auth: AuthDep,
    _contributor: RoleContributorDep,
    id: str,
    runner: FAQImportRunnerDep,
    faq_service: FAQServiceDep,
    knowledge_service: KnowledgeServiceDep,
    tenant_id: _PrincipalTenant,
) -> FAQImportProgressEnvelope:
    """Import FAQ entries from JSON upsert or a CSV / Excel file.

    ``application/json`` is the SPA payload. Multipart stays the file
    runner. Both record a completed progress object for later polling.
    """
    tenant_id = _require_tenant(tenant_id)
    if is_json_content_type(request.headers.get("content-type", "")):
        return await _upsert_json_entries(
            request=request,
            faq_service=faq_service,
            runner=runner,
            knowledge_service=knowledge_service,
            tenant_id=tenant_id,
            knowledge_base_id=id,
        )
    return await _import_file_entries(
        request=request,
        runner=runner,
        knowledge_service=knowledge_service,
        tenant_id=tenant_id,
        knowledge_base_id=id,
    )


@router.post("/search", response_model=FAQSearchEnvelope)
async def search_faq_entries(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    id: str,
    body: FAQSearchRequest,
    faq_service: FAQServiceDep,
    tenant_id: _PrincipalTenant,
) -> FAQSearchEnvelope:
    """Keyword-overlap search. ``data`` is a list of scored entries."""
    tenant_id = _require_tenant(tenant_id)
    hits = await faq_service.search_entries(
        tenant_id=tenant_id,
        knowledge_base_id=id,
        request=body,
    )
    return search_envelope(hits)


@router.put("/import/last-result/display", response_model=FAQImportProgressEnvelope)
async def update_faq_import_result_display(
    _auth: AuthDep,
    _contributor: RoleContributorDep,
    id: str,
    body: FAQImportDisplayStatusRequest,
    runner: FAQImportRunnerDep,
    tenant_id: _PrincipalTenant,
) -> FAQImportProgressEnvelope:
    """Persist last-result card visibility on the newest import task."""
    tenant_id = _require_tenant(tenant_id)
    if body.display_status not in _DISPLAY_STATUSES:
        raise ValidationError(
            code="faq.invalid_display_status",
            message="display_status 仅支持 open 或 close",
        )
    progress = runner.set_display_status(
        knowledge_base_id=id,
        display_status=body.display_status,
    )
    if progress is None:
        raise NotFoundError(
            code="faq.import_task_not_found",
            message="FAQ 导入任务不存在",
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


async def _upsert_json_entries(
    *,
    request: Request,
    faq_service: FAQService,
    runner: FAQImportRunner,
    knowledge_service: KnowledgeService,
    tenant_id: int,
    knowledge_base_id: str,
) -> FAQImportProgressEnvelope:
    """Persist a JSON upsert and record completed progress for polling."""
    payload = await read_json_upsert_payload(request)
    _require_import_mode(payload.mode)
    knowledge_id = await resolve_faq_knowledge_id(
        knowledge_service,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
    )
    created = await faq_service.upsert_entries(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        knowledge_id=knowledge_id,
        entries=payload.entries,
        mode=payload.mode,
        dry_run=payload.dry_run,
    )
    success_count = len(created) if not payload.dry_run else len(payload.entries)
    progress = runner.record_completed(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        knowledge_id=knowledge_id,
        mode=payload.mode,
        dry_run=payload.dry_run,
        total=len(payload.entries),
        success_count=success_count,
        task_id=payload.task_id,
    )
    return import_progress_envelope(progress)


async def _import_file_entries(
    *,
    request: Request,
    runner: FAQImportRunner,
    knowledge_service: KnowledgeService,
    tenant_id: int,
    knowledge_base_id: str,
) -> FAQImportProgressEnvelope:
    """Run the CSV / Excel import pipeline and record completed progress."""
    multipart = await read_multipart_import(request)
    _require_import_mode(multipart.mode)
    knowledge_id = await resolve_faq_knowledge_id(
        knowledge_service,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
    )
    progress = await runner.run(
        file_data=multipart.file_data,
        filename=multipart.filename,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        knowledge_id=knowledge_id,
        mode=multipart.mode,
        dry_run=multipart.dry_run,
    )
    return import_progress_envelope(progress)


def _require_import_mode(mode: str) -> None:
    if mode not in (FAQ_BATCH_MODE_APPEND, FAQ_BATCH_MODE_REPLACE):
        raise ValidationError(
            code="faq.invalid_import_mode",
            message="模式仅支持 append 或 replace",
        )


__all__ = ["import_progress_router", "router"]
