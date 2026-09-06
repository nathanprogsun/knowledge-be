"""Wiki HTTP endpoints - page CRUD, folders, graph, lint, search.

Registered by ``application.include_router`` in the app factory.

The route surface mirrors the upstream handler verbatim:

===========================================  ========
Route                                         Role
===========================================  ========
``GET    /knowledgebase/{kb_id}/wiki/pages``  Viewer
``POST   /knowledgebase/{kb_id}/wiki/pages``  Admin
``GET    /knowledgebase/{kb_id}/wiki/pages/{slug}`` Viewer
``PUT    /knowledgebase/{kb_id}/wiki/pages/{slug}`` Admin
``DELETE /knowledgebase/{kb_id}/wiki/pages/{slug}`` Admin
``PUT    /knowledgebase/{kb_id}/wiki/move-page``    Admin
``GET    /knowledgebase/{kb_id}/wiki/folders``  Viewer
``POST   /knowledgebase/{kb_id}/wiki/folders``  Admin
``PUT    /knowledgebase/{kb_id}/wiki/folders/{folder_id}`` Admin
``DELETE /knowledgebase/{kb_id}/wiki/folders/{folder_id}`` Admin
``GET    /knowledgebase/{kb_id}/wiki/index``    Viewer
``GET    /knowledgebase/{kb_id}/wiki/graph``    Viewer
``GET    /knowledgebase/{kb_id}/wiki/stats``    Viewer
``GET    /knowledgebase/{kb_id}/wiki/search``   Viewer
``POST   /knowledgebase/{kb_id}/wiki/rebuild-links`` Admin
``GET    /knowledgebase/{kb_id}/wiki/lint``     Viewer
``POST   /knowledgebase/{kb_id}/wiki/auto-fix`` Admin
===========================================  ========

Wiki pages are KB content: reads are Viewer+, mutations and maintenance
are Admin+. Every endpoint first validates that the knowledge base
exists and has the wiki pipeline enabled.

Slugs are hierarchical (``entity/acme-corp``), so ``{slug:path}`` uses
the path converter — mirroring the upstream catch-all ``*slug``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from src.common.exception import (
    ConflictError,
    UnauthorizedError,
    ValidationError,
)
from src.core.knowledge.wiki.lint_service import WikiLintReport
from src.core.knowledge.wiki.types import (
    WIKI_EDIT_SOURCE_USER,
    WIKI_GRAPH_MODE_EGO,
    WIKI_GRAPH_MODE_OVERVIEW,
    WikiGraphRequest,
    WikiPage,
    WikiPageListFilter,
    clean_category_path,
    is_valid_page_status,
    is_valid_page_type,
    split_page_types,
)
from src.web.api.knowledge.wiki.views import (
    WikiAutoFixData,
    WikiEnvelope,
    WikiFolderCreateRequest,
    WikiFolderListData,
    WikiFolderUpdateRequest,
    WikiFolderView,
    WikiGraphData,
    WikiIndexData,
    WikiMessage,
    WikiPageCreateRequest,
    WikiPageListData,
    WikiPageMoveRequest,
    WikiPageUpdateRequest,
    WikiPageView,
    WikiSearchData,
    WikiStats,
    folder_node_to_view,
    folder_to_view,
    index_to_view,
    page_to_view,
    require_wiki_kb,
)
from src.web.deps import AuthDep, RoleAdminDep, RoleViewerDep
from src.web.deps.context import get_tenant_id_dep, get_user_id_dep
from src.web.deps.knowledge_wiki import (
    KBServiceDep,
    WikiFolderServiceDep,
    WikiLintServiceDep,
    WikiPageServiceDep,
)
from src.web.deps.sharing import KBShareServiceDep

# Shortcut aliases for the function-arg-style principal deps.
_PrincipalTenant = Annotated[int, Depends(get_tenant_id_dep)]
_PrincipalUser = Annotated[str | None, Depends(get_user_id_dep)]


router = APIRouter(prefix="/knowledgebase/{kb_id}/wiki", tags=["wiki"])

# Graph query bounds mirror the upstream defaults and caps.
_GRAPH_DEFAULT_LIMIT = 500
_GRAPH_MAX_LIMIT = 2000
_GRAPH_MAX_DEPTH = 3
_GRAPH_DEFAULT_DEPTH = 1


def _require_tenant(tenant_id: int) -> int:
    """Return the active workspace id, or fail.

    A wiki page is always workspace-scoped; without a tenant context
    there is no safe default, so this rejects rather than guessing.
    """
    if tenant_id == 0:
        raise UnauthorizedError(
            code="auth.tenant_context_missing",
            message="unauthorized: workspace context missing",
        )
    return tenant_id


def _actor(user_id: str | None) -> str:
    """Return the acting user's id for page authorship rows (``""`` when absent)."""
    return user_id or ""


def _parse_category_path(raw: str) -> list[str]:
    """Split a ``category_path`` query value into trimmed segments."""
    return [part.strip() for part in raw.split("/") if part.strip()]


# ── Page CRUD ───────────────────────────────────────────────────────


@router.get("/pages", response_model=WikiEnvelope[WikiPageListData])
async def list_pages(
    _auth: AuthDep,
    _role: RoleViewerDep,
    kb_id: str,
    kb_service: KBServiceDep,
    kb_share_service: KBShareServiceDep,
    service: WikiPageServiceDep,
    page_type: str = Query(default=""),
    status: str = Query(default=""),
    query: str = Query(default=""),
    category_path: str = Query(default=""),
    folder_id: str | None = Query(default=None),
    category_depth: int | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    sort_by: str = Query(default="updated_at"),
    sort_order: str = Query(default="desc"),
) -> WikiEnvelope[WikiPageListData]:
    """List wiki pages with optional filtering and pagination."""
    await require_wiki_kb(
        kb_id=kb_id,
        kb_service=kb_service,
        kb_share_service=kb_share_service,
    )

    # Explicitly-present-but-empty ``folder_id`` means "root" (folder_id
    # = ''); an absent param means "no filter".
    folder = folder_id.strip() if folder_id is not None else None
    depth = category_depth if category_depth is not None and category_depth >= 0 else None

    filters = WikiPageListFilter(
        knowledge_base_id=kb_id,
        page_type=page_type,
        status=status,
        query=query,
        folder_id=folder,
        category_path=clean_category_path(_parse_category_path(category_path)),
        category_depth=depth,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    result = await service.list_pages(filters=filters)
    return WikiEnvelope(
        success=True,
        data=WikiPageListData(
            pages=[page_to_view(p) for p in result.pages],
            total=result.total,
            page=result.page,
            page_size=result.page_size,
            total_pages=result.total_pages,
        ),
    )


@router.post("/pages", response_model=WikiEnvelope[WikiPageView], status_code=201)
async def create_page(
    _auth: AuthDep,
    _role: RoleAdminDep,
    kb_id: str,
    body: WikiPageCreateRequest,
    kb_service: KBServiceDep,
    kb_share_service: KBShareServiceDep,
    service: WikiPageServiceDep,
    tenant_id: _PrincipalTenant,
    user_id: _PrincipalUser,
) -> WikiEnvelope[WikiPageView]:
    """Create a new wiki page in the knowledge base."""
    await require_wiki_kb(
        kb_id=kb_id,
        kb_service=kb_service,
        kb_share_service=kb_share_service,
        write=True,
    )
    tenant_id = _require_tenant(tenant_id)

    page_type = body.page_type.strip()
    status = body.status.strip()
    if page_type and not is_valid_page_type(page_type):
        raise ValidationError(
            code="wiki.page_invalid_type",
            message=f"invalid page_type: {page_type}",
        )
    if status and not is_valid_page_status(status):
        raise ValidationError(
            code="wiki.page_invalid_status",
            message=f"invalid status: {status}",
        )

    now = datetime.now(UTC)
    page = WikiPage(
        id="",
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        slug=body.slug,
        title=body.title,
        page_type=page_type,
        status=status,
        content=body.content,
        summary=body.summary,
        aliases=list(body.aliases),
        parent_slug=body.parent_slug,
        folder_id=body.folder_id,
        sort_order=body.sort_order,
        source_refs=list(body.source_refs),
        chunk_refs=list(body.chunk_refs),
        page_metadata=dict(body.page_metadata),
        created_at=now,
        updated_at=now,
    )
    created = await service.create_page(
        page=page,
        edit_source=WIKI_EDIT_SOURCE_USER,
        editor_id=_actor(user_id),
    )
    return WikiEnvelope(success=True, data=page_to_view(created))


@router.get("/pages/{slug:path}", response_model=WikiEnvelope[WikiPageView])
async def get_page(
    _auth: AuthDep,
    _role: RoleViewerDep,
    kb_id: str,
    slug: str,
    kb_service: KBServiceDep,
    kb_share_service: KBShareServiceDep,
    service: WikiPageServiceDep,
) -> WikiEnvelope[WikiPageView]:
    """Retrieve a wiki page by its (hierarchical) slug."""
    await require_wiki_kb(
        kb_id=kb_id,
        kb_service=kb_service,
        kb_share_service=kb_share_service,
    )
    slug = slug.strip()
    page = await service.get_page_by_slug(knowledge_base_id=kb_id, slug=slug)
    return WikiEnvelope(success=True, data=page_to_view(page))


@router.put("/pages/{slug:path}", response_model=WikiEnvelope[WikiPageView])
async def update_page(
    _auth: AuthDep,
    _role: RoleAdminDep,
    kb_id: str,
    slug: str,
    body: WikiPageUpdateRequest,
    kb_service: KBServiceDep,
    kb_share_service: KBShareServiceDep,
    service: WikiPageServiceDep,
    user_id: _PrincipalUser,
) -> WikiEnvelope[WikiPageView]:
    """Partially update a wiki page; absent fields keep their stored value.

    When ``version`` > 0 it acts as an optimistic-lock guard: a mismatch
    with the stored version returns 409 together with the current version.
    """
    await require_wiki_kb(
        kb_id=kb_id,
        kb_service=kb_service,
        kb_share_service=kb_share_service,
        write=True,
    )
    slug = slug.strip()
    existing = await service.get_page_by_slug(knowledge_base_id=kb_id, slug=slug)
    if body.version > 0 and body.version != existing.version:
        raise ConflictError(
            code="wiki.page_conflict",
            message="wiki page was modified by someone else",
            details={"current_version": existing.version},
        )

    page = existing.model_copy(
        update={
            "title": body.title.strip() if body.title is not None else existing.title,
            "content": body.content if body.content is not None else existing.content,
            "summary": body.summary if body.summary is not None else existing.summary,
            "page_type": body.page_type.strip()
            if body.page_type is not None
            else existing.page_type,
            "status": body.status.strip() if body.status is not None else existing.status,
            "aliases": list(body.aliases) if body.aliases is not None else existing.aliases,
        }
    )
    if body.page_type is not None and not is_valid_page_type(page.page_type):
        raise ValidationError(
            code="wiki.page_invalid_type",
            message=f"invalid page_type: {page.page_type}",
        )
    if body.status is not None and not is_valid_page_status(page.status):
        raise ValidationError(
            code="wiki.page_invalid_status",
            message=f"invalid status: {page.status}",
        )

    updated = await service.update_page(
        page=page,
        edit_source=WIKI_EDIT_SOURCE_USER,
        editor_id=_actor(user_id),
    )
    return WikiEnvelope(success=True, data=page_to_view(updated))


@router.delete("/pages/{slug:path}", status_code=204)
async def delete_page(
    _auth: AuthDep,
    _role: RoleAdminDep,
    kb_id: str,
    slug: str,
    kb_service: KBServiceDep,
    kb_share_service: KBShareServiceDep,
    service: WikiPageServiceDep,
) -> None:
    """Soft-delete a wiki page by slug."""
    await require_wiki_kb(
        kb_id=kb_id,
        kb_service=kb_service,
        kb_share_service=kb_share_service,
        write=True,
    )
    slug = slug.strip()
    await service.delete_page(knowledge_base_id=kb_id, slug=slug)


@router.put("/move-page", response_model=WikiEnvelope[WikiPageView])
async def move_page(
    _auth: AuthDep,
    _role: RoleAdminDep,
    kb_id: str,
    body: WikiPageMoveRequest,
    kb_service: KBServiceDep,
    kb_share_service: KBShareServiceDep,
    service: WikiPageServiceDep,
) -> WikiEnvelope[WikiPageView]:
    """Relocate a page (by slug) into a folder (empty = root)."""
    await require_wiki_kb(
        kb_id=kb_id,
        kb_service=kb_service,
        kb_share_service=kb_share_service,
        write=True,
    )
    slug = body.slug.strip()
    if not slug:
        raise ValidationError(
            code="wiki.page_slug_required",
            message="page slug is required",
        )
    page = await service.move_page(
        knowledge_base_id=kb_id,
        slug=slug,
        folder_id=body.folder_id.strip(),
    )
    return WikiEnvelope(success=True, data=page_to_view(page))


# ── Folder tree ─────────────────────────────────────────────────────


@router.get("/folders", response_model=WikiEnvelope[WikiFolderListData])
async def list_folders(
    _auth: AuthDep,
    _role: RoleViewerDep,
    kb_id: str,
    kb_service: KBServiceDep,
    kb_share_service: KBShareServiceDep,
    service: WikiFolderServiceDep,
    parent_id: str = Query(default=""),
    page_types: str = Query(default=""),
) -> WikiEnvelope[WikiFolderListData]:
    """List the direct child folders of a parent (empty = root)."""
    await require_wiki_kb(
        kb_id=kb_id,
        kb_service=kb_service,
        kb_share_service=kb_share_service,
    )
    types = split_page_types(page_types)
    nodes = await service.list_child_folders(
        knowledge_base_id=kb_id,
        parent_id=parent_id.strip(),
        page_types=types,
    )
    return WikiEnvelope(
        success=True,
        data=WikiFolderListData(
            parent_id=parent_id.strip(),
            folders=[folder_node_to_view(node) for node in nodes],
        ),
    )


@router.post("/folders", response_model=WikiEnvelope[WikiFolderView], status_code=201)
async def create_folder(
    _auth: AuthDep,
    _role: RoleAdminDep,
    kb_id: str,
    body: WikiFolderCreateRequest,
    kb_service: KBServiceDep,
    kb_share_service: KBShareServiceDep,
    service: WikiFolderServiceDep,
    tenant_id: _PrincipalTenant,
) -> WikiEnvelope[WikiFolderView]:
    """Create a new (initially empty) folder under ``parent_id``."""
    await require_wiki_kb(
        kb_id=kb_id,
        kb_service=kb_service,
        kb_share_service=kb_share_service,
        write=True,
    )
    tenant_id = _require_tenant(tenant_id)
    folder = await service.create_folder(
        knowledge_base_id=kb_id,
        tenant_id=tenant_id,
        parent_id=body.parent_id.strip(),
        name=body.name,
    )
    return WikiEnvelope(success=True, data=folder_to_view(folder))


@router.put("/folders/{folder_id}", response_model=WikiEnvelope[WikiFolderView])
async def update_folder(
    _auth: AuthDep,
    _role: RoleAdminDep,
    kb_id: str,
    folder_id: str,
    body: WikiFolderUpdateRequest,
    kb_service: KBServiceDep,
    kb_share_service: KBShareServiceDep,
    service: WikiFolderServiceDep,
) -> WikiEnvelope[WikiFolderView]:
    """Rename and/or reparent a folder; subtree caches are recomputed."""
    await require_wiki_kb(
        kb_id=kb_id,
        kb_service=kb_service,
        kb_share_service=kb_share_service,
        write=True,
    )
    folder = await service.rename_or_move_folder(
        knowledge_base_id=kb_id,
        id=folder_id.strip(),
        new_name=body.name,
        new_parent_id=body.parent_id.strip(),
        move_parent=body.move_parent,
    )
    return WikiEnvelope(success=True, data=folder_to_view(folder))


@router.delete("/folders/{folder_id}", status_code=204)
async def delete_folder(
    _auth: AuthDep,
    _role: RoleAdminDep,
    kb_id: str,
    folder_id: str,
    kb_service: KBServiceDep,
    kb_share_service: KBShareServiceDep,
    service: WikiFolderServiceDep,
) -> None:
    """Delete an empty wiki folder (no pages and no child folders)."""
    await require_wiki_kb(
        kb_id=kb_id,
        kb_service=kb_service,
        kb_share_service=kb_share_service,
        write=True,
    )
    await service.delete_folder(knowledge_base_id=kb_id, id=folder_id.strip())


# ── Index ───────────────────────────────────────────────────────────


@router.get("/index", response_model=WikiEnvelope[WikiIndexData])
async def get_index(
    _auth: AuthDep,
    _role: RoleViewerDep,
    kb_id: str,
    kb_service: KBServiceDep,
    kb_share_service: KBShareServiceDep,
    service: WikiPageServiceDep,
    tenant_id: _PrincipalTenant,
    types: str = Query(default=""),
    limit: int = Query(default=50),
    cursor: str = Query(default=""),
) -> WikiEnvelope[WikiIndexData]:
    """Return the structured wiki index (intro + per-type directory groups)."""
    await require_wiki_kb(
        kb_id=kb_id,
        kb_service=kb_service,
        kb_share_service=kb_share_service,
    )
    tenant_id = _require_tenant(tenant_id)
    response = await service.get_index_view(
        knowledge_base_id=kb_id,
        tenant_id=tenant_id,
        page_types=split_page_types(types),
        limit=limit,
        cursor=cursor,
    )
    return WikiEnvelope(success=True, data=index_to_view(response))


# ── Graph / stats ───────────────────────────────────────────────────


@router.get("/graph", response_model=WikiEnvelope[WikiGraphData])
async def get_graph(
    _auth: AuthDep,
    _role: RoleViewerDep,
    kb_id: str,
    kb_service: KBServiceDep,
    kb_share_service: KBShareServiceDep,
    service: WikiPageServiceDep,
    mode: str = Query(default=""),
    center: str = Query(default=""),
    depth: int = Query(default=_GRAPH_DEFAULT_DEPTH),
    types: str = Query(default=""),
    limit: int = Query(default=_GRAPH_DEFAULT_LIMIT),
) -> WikiEnvelope[WikiGraphData]:
    """Return a slice of the wiki link graph for visualization."""
    await require_wiki_kb(
        kb_id=kb_id,
        kb_service=kb_service,
        kb_share_service=kb_share_service,
    )

    mode = mode.strip() or WIKI_GRAPH_MODE_OVERVIEW
    if mode not in (WIKI_GRAPH_MODE_OVERVIEW, WIKI_GRAPH_MODE_EGO):
        raise ValidationError(
            code="wiki.graph_invalid_mode",
            message="mode must be 'overview' or 'ego'",
        )
    center = center.strip()
    if mode == WIKI_GRAPH_MODE_EGO and not center:
        raise ValidationError(
            code="wiki.graph_center_required",
            message="center is required when mode=ego",
        )
    if depth < 1:
        raise ValidationError(
            code="wiki.graph_invalid_depth",
            message="depth must be a positive integer",
        )
    if limit < 1:
        raise ValidationError(
            code="wiki.graph_invalid_limit",
            message="limit must be a positive integer",
        )
    depth = min(depth, _GRAPH_MAX_DEPTH)
    limit = min(limit, _GRAPH_MAX_LIMIT)

    graph = await service.get_graph(
        request=WikiGraphRequest(
            knowledge_base_id=kb_id,
            mode=mode,
            center=center,
            depth=depth,
            types=split_page_types(types) or [],
            limit=limit,
        )
    )
    return WikiEnvelope(success=True, data=graph)


@router.get("/stats", response_model=WikiEnvelope[WikiStats])
async def get_stats(
    _auth: AuthDep,
    _role: RoleViewerDep,
    kb_id: str,
    kb_service: KBServiceDep,
    kb_share_service: KBShareServiceDep,
    service: WikiPageServiceDep,
) -> WikiEnvelope[WikiStats]:
    """Return aggregate statistics about the KB's wiki."""
    await require_wiki_kb(
        kb_id=kb_id,
        kb_service=kb_service,
        kb_share_service=kb_share_service,
    )
    stats = await service.get_stats(knowledge_base_id=kb_id)
    return WikiEnvelope(success=True, data=stats)


# ── Search / maintenance ────────────────────────────────────────────


@router.get("/search", response_model=WikiEnvelope[WikiSearchData])
async def search_pages(
    _auth: AuthDep,
    _role: RoleViewerDep,
    kb_id: str,
    kb_service: KBServiceDep,
    kb_share_service: KBShareServiceDep,
    service: WikiPageServiceDep,
    q: str = Query(default=""),
    limit: int = Query(default=10),
) -> WikiEnvelope[WikiSearchData]:
    """Full-text search over the KB's wiki pages."""
    await require_wiki_kb(
        kb_id=kb_id,
        kb_service=kb_service,
        kb_share_service=kb_share_service,
    )
    if not q.strip():
        raise ValidationError(
            code="wiki.search_query_required",
            message="search query 'q' is required",
        )
    pages = await service.search_pages(knowledge_base_id=kb_id, query=q, limit=limit)
    return WikiEnvelope(
        success=True,
        data=WikiSearchData(pages=[page_to_view(page) for page in pages]),
    )


@router.post("/rebuild-links", response_model=WikiEnvelope[WikiMessage])
async def rebuild_links(
    _auth: AuthDep,
    _role: RoleAdminDep,
    kb_id: str,
    kb_service: KBServiceDep,
    kb_share_service: KBShareServiceDep,
    service: WikiPageServiceDep,
) -> WikiEnvelope[WikiMessage]:
    """Re-parse all pages and rebuild bidirectional link references."""
    await require_wiki_kb(
        kb_id=kb_id,
        kb_service=kb_service,
        kb_share_service=kb_share_service,
        write=True,
    )
    await service.rebuild_links(knowledge_base_id=kb_id)
    return WikiEnvelope(
        success=True,
        data=WikiMessage(message="Links rebuilt successfully"),
    )


@router.get("/lint", response_model=WikiEnvelope[WikiLintReport])
async def lint(
    _auth: AuthDep,
    _role: RoleViewerDep,
    kb_id: str,
    kb_service: KBServiceDep,
    kb_share_service: KBShareServiceDep,
    service: WikiLintServiceDep,
) -> WikiEnvelope[WikiLintReport]:
    """Run a comprehensive health check over the KB's wiki."""
    await require_wiki_kb(
        kb_id=kb_id,
        kb_service=kb_service,
        kb_share_service=kb_share_service,
    )
    report = await service.run_lint(knowledge_base_id=kb_id)
    return WikiEnvelope(success=True, data=report)


@router.post("/auto-fix", response_model=WikiEnvelope[WikiAutoFixData])
async def auto_fix(
    _auth: AuthDep,
    _role: RoleAdminDep,
    kb_id: str,
    kb_service: KBServiceDep,
    kb_share_service: KBShareServiceDep,
    service: WikiLintServiceDep,
) -> WikiEnvelope[WikiAutoFixData]:
    """Automatically fix machine-safe wiki issues (broken links, etc.)."""
    await require_wiki_kb(
        kb_id=kb_id,
        kb_service=kb_service,
        kb_share_service=kb_share_service,
        write=True,
    )
    fixed = await service.auto_fix(knowledge_base_id=kb_id)
    return WikiEnvelope(
        success=True,
        data=WikiAutoFixData(fixed=fixed, message=f"Auto-fixed {fixed} issues"),
    )


__all__ = ["router"]
