"""Wire-shape conversion for the wiki endpoints.

Projects the wiki domain shapes (``WikiPage`` / ``WikiFolder`` /
``WikiFolderNode`` / ``WikiIndexResponse``) onto frozen wire views plus
the shared ``{"success": true, "data": ...}`` envelope. The domain
models are imported from the core wiki types module — web never imports
``db`` directly.

Every success response uses ``WikiEnvelope``; errors flow through the
registered ``ApplicationError`` handler, which emits the standard
``{"success": false, "error": {...}}`` shape.
"""

from __future__ import annotations

from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from src.app_context import request_context
from src.common.exception import (
    PermissionDeniedError,
    UnauthorizedError,
    ValidationError,
)
from src.common.json import JsonObject
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.wiki.issues import WikiPageIssue
from src.core.knowledge.wiki.lint_service import WikiLintReport
from src.core.knowledge.wiki.revisions import WikiPageRevisionInfo, WikiRevisionList
from src.core.knowledge.wiki.types import (
    WikiFolder,
    WikiFolderNode,
    WikiGraphData,
    WikiIndexEntry,
    WikiIndexResponse,
    WikiPage,
    WikiStats,
)
from src.core.sharing.kb_share_service import KBShareService

T = TypeVar("T")


class WikiEnvelope(BaseModel, Generic[T]):
    """``{"success": true, "data": ...}`` - every wiki success response."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: T


# ── Page wire shape ──────────────────────────────────────────────────


class WikiPageView(BaseModel):
    """One wiki page on the wire — mirrors the upstream ``WikiPage`` JSON."""

    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: int
    knowledge_base_id: str
    slug: str
    title: str = ""
    page_type: str = ""
    status: str = ""
    content: str = ""
    summary: str = ""
    aliases: list[str] = Field(default_factory=list)
    parent_slug: str = ""
    folder_id: str = ""
    category_path: list[str] = Field(default_factory=list)
    wiki_path: str = ""
    depth: int = 0
    sort_order: int = 0
    source_refs: list[str] = Field(default_factory=list)
    chunk_refs: list[str] = Field(default_factory=list)
    in_links: list[str] = Field(default_factory=list)
    out_links: list[str] = Field(default_factory=list)
    page_metadata: JsonObject = Field(default_factory=dict)
    version: int = 1
    last_edit_source: str = ""
    last_editor_id: str = ""
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


class WikiPageListData(BaseModel):
    """Paginated page listing — ``{pages, total, page, page_size, total_pages}``."""

    model_config = ConfigDict(frozen=True)

    pages: list[WikiPageView]
    total: int
    page: int
    page_size: int
    total_pages: int


class WikiSearchData(BaseModel):
    """Search result payload — ``{"pages": [...]}``."""

    model_config = ConfigDict(frozen=True)

    pages: list[WikiPageView]


class WikiPageCreateRequest(BaseModel):
    """Body for ``POST /wiki/pages``.

    ``slug`` is required; every other content-bearing field is optional
    and defaults to the empty value. Placement and directory caches are
    recomputed server-side by the service.
    """

    model_config = ConfigDict(frozen=True)

    slug: str
    title: str = ""
    page_type: str = ""
    status: str = ""
    content: str = ""
    summary: str = ""
    aliases: list[str] = Field(default_factory=list)
    parent_slug: str = ""
    folder_id: str = ""
    sort_order: int = 0
    source_refs: list[str] = Field(default_factory=list)
    chunk_refs: list[str] = Field(default_factory=list)
    page_metadata: JsonObject = Field(default_factory=dict)


class WikiPageUpdateRequest(BaseModel):
    """Body for ``PUT /wiki/pages/{slug}``.

    Absent fields keep their stored value; ``version`` (> 0) is the
    optimistic-lock guard.
    """

    model_config = ConfigDict(frozen=True)

    title: str | None = None
    content: str | None = None
    summary: str | None = None
    page_type: str | None = None
    status: str | None = None
    aliases: list[str] | None = None
    version: int = 0


class WikiPageMoveRequest(BaseModel):
    """Body for ``PUT /wiki/move-page`` — the page moves by slug."""

    model_config = ConfigDict(frozen=True)

    slug: str
    folder_id: str = ""


# ── Folder wire shapes ───────────────────────────────────────────────


class WikiFolderView(BaseModel):
    """One wiki folder on the wire."""

    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: int
    knowledge_base_id: str
    parent_id: str = ""
    name: str
    path: str = ""
    depth: int = 0
    sort_order: int = 0
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


class WikiFolderNodeView(WikiFolderView):
    """A folder node with its live page count and expand affordance.

    The upstream node embeds the folder row, so the wire shape is flat
    (folder fields + ``page_count`` + ``has_children``).
    """

    model_config = ConfigDict(frozen=True)

    page_count: int = 0
    has_children: bool = False


class WikiFolderListData(BaseModel):
    """Folder listing payload — ``{parent_id, folders}``."""

    model_config = ConfigDict(frozen=True)

    parent_id: str
    folders: list[WikiFolderNodeView]


class WikiFolderCreateRequest(BaseModel):
    """Body for ``POST /wiki/folders``."""

    model_config = ConfigDict(frozen=True)

    parent_id: str = ""
    name: str


class WikiFolderUpdateRequest(BaseModel):
    """Body for ``PUT /wiki/folders/{folder_id}``.

    ``parent_id`` is applied only when ``move_parent`` is true, so a pure
    rename does not need to re-send the (possibly root) parent.
    """

    model_config = ConfigDict(frozen=True)

    name: str = ""
    parent_id: str = ""
    move_parent: bool = False


# ── Index wire shapes ────────────────────────────────────────────────


class WikiIndexEntryView(BaseModel):
    """One clickable directory entry in the index response."""

    model_config = ConfigDict(frozen=True)

    slug: str
    title: str = ""
    summary: str = ""
    parent_slug: str = ""
    category_path: list[str] = Field(default_factory=list)
    wiki_path: str = ""
    depth: int = 0
    sort_order: int = 0


class WikiIndexGroupView(BaseModel):
    """One page-type group in the index response."""

    model_config = ConfigDict(frozen=True)

    type: str
    total: int
    items: list[WikiIndexEntryView]
    next_cursor: str = ""


class WikiIndexData(BaseModel):
    """The structured index payload — ``{intro, version, groups}``."""

    model_config = ConfigDict(frozen=True)

    intro: str
    version: int
    groups: list[WikiIndexGroupView]


# ── Simple acks ──────────────────────────────────────────────────────


class WikiMessage(BaseModel):
    """``{"message": "..."}`` — bare acknowledgement payloads."""

    model_config = ConfigDict(frozen=True)

    message: str


class WikiAutoFixData(BaseModel):
    """``{"fixed": N, "message": "..."}`` — the auto-fix acknowledgement."""

    model_config = ConfigDict(frozen=True)

    fixed: int
    message: str


class WikiIssueView(BaseModel):
    """One wiki page issue on the wire."""

    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: int
    knowledge_base_id: str
    slug: str
    issue_type: str
    description: str
    suspected_knowledge_ids: list[str] = Field(default_factory=list)
    status: str
    reported_by: str = ""
    created_at: datetime
    updated_at: datetime


class WikiIssueStatusRequest(BaseModel):
    """Body for ``PUT /wiki/issues/{issue_id}/status``."""

    model_config = ConfigDict(frozen=True)

    status: str


class WikiRevisionView(BaseModel):
    """One page snapshot on the wire. List rows omit ``content``."""

    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: int
    knowledge_base_id: str
    page_id: str
    slug: str
    version: int
    title: str = ""
    page_type: str = ""
    status: str = ""
    content: str | None = None
    summary: str = ""
    aliases: list[str] = Field(default_factory=list)
    edit_source: str = ""
    editor_id: str = ""
    edited_at: datetime
    created_at: datetime


class WikiRevisionListData(BaseModel):
    """Revision listing — ``{revisions, total, current_version}``."""

    model_config = ConfigDict(frozen=True)

    revisions: list[WikiRevisionView]
    total: int
    current_version: int


class WikiRevertRequest(BaseModel):
    """Body for ``POST /wiki/revert``."""

    model_config = ConfigDict(frozen=True)

    slug: str
    version: int


# ── Converters ───────────────────────────────────────────────────────


def page_to_view(page: WikiPage) -> WikiPageView:
    """Project a page row onto the wire shape."""
    return WikiPageView(
        id=page.id,
        tenant_id=page.tenant_id,
        knowledge_base_id=page.knowledge_base_id,
        slug=page.slug,
        title=page.title,
        page_type=page.page_type,
        status=page.status,
        content=page.content,
        summary=page.summary,
        aliases=list(page.aliases),
        parent_slug=page.parent_slug,
        folder_id=page.folder_id,
        category_path=list(page.category_path),
        wiki_path=page.wiki_path,
        depth=page.depth,
        sort_order=page.sort_order,
        source_refs=list(page.source_refs),
        chunk_refs=list(page.chunk_refs),
        in_links=list(page.in_links),
        out_links=list(page.out_links),
        page_metadata=dict(page.page_metadata),
        version=page.version,
        last_edit_source=page.last_edit_source,
        last_editor_id=page.last_editor_id,
        created_at=page.created_at,
        updated_at=page.updated_at,
        deleted_at=page.deleted_at,
    )


def folder_to_view(folder: WikiFolder) -> WikiFolderView:
    """Project a folder row onto the wire shape."""
    return WikiFolderView(
        id=folder.id,
        tenant_id=folder.tenant_id,
        knowledge_base_id=folder.knowledge_base_id,
        parent_id=folder.parent_id,
        name=folder.name,
        path=folder.path,
        depth=folder.depth,
        sort_order=folder.sort_order,
        created_at=folder.created_at,
        updated_at=folder.updated_at,
        deleted_at=folder.deleted_at,
    )


def folder_node_to_view(node: WikiFolderNode) -> WikiFolderNodeView:
    """Project a folder node onto the flattened wire shape."""
    return WikiFolderNodeView(
        **folder_to_view(node.folder).model_dump(),
        page_count=node.page_count,
        has_children=node.has_children,
    )


def index_to_view(response: WikiIndexResponse) -> WikiIndexData:
    """Project the structured index response onto the wire shape."""
    return WikiIndexData(
        intro=response.intro,
        version=response.version,
        groups=[
            WikiIndexGroupView(
                type=group.type,
                total=group.total,
                items=[_index_entry_view(entry) for entry in group.items],
                next_cursor=group.next_cursor,
            )
            for group in response.groups
        ],
    )


def issue_to_view(issue: WikiPageIssue) -> WikiIssueView:
    """Project an issue DTO onto the wire shape."""
    return WikiIssueView(
        id=issue.id,
        tenant_id=issue.tenant_id,
        knowledge_base_id=issue.knowledge_base_id,
        slug=issue.slug,
        issue_type=issue.issue_type,
        description=issue.description,
        suspected_knowledge_ids=list(issue.suspected_knowledge_ids),
        status=issue.status,
        reported_by=issue.reported_by,
        created_at=issue.created_at,
        updated_at=issue.updated_at,
    )


def revision_to_view(revision: WikiPageRevisionInfo) -> WikiRevisionView:
    """Project a revision DTO onto the wire shape."""
    return WikiRevisionView(
        id=revision.id,
        tenant_id=revision.tenant_id,
        knowledge_base_id=revision.knowledge_base_id,
        page_id=revision.page_id,
        slug=revision.slug,
        version=revision.version,
        title=revision.title,
        page_type=revision.page_type,
        status=revision.status,
        content=revision.content,
        summary=revision.summary,
        aliases=list(revision.aliases),
        edit_source=revision.edit_source,
        editor_id=revision.editor_id,
        edited_at=revision.edited_at,
        created_at=revision.created_at,
    )


def revision_list_to_view(result: WikiRevisionList) -> WikiRevisionListData:
    """Project a revision list onto the wire shape."""
    return WikiRevisionListData(
        revisions=[revision_to_view(row) for row in result.revisions],
        total=result.total,
        current_version=result.current_version,
    )


def _index_entry_view(entry: WikiIndexEntry) -> WikiIndexEntryView:
    """Project one light index projection onto the wire shape."""
    return WikiIndexEntryView(
        slug=entry.slug,
        title=entry.title,
        summary=entry.summary,
        parent_slug=entry.parent_slug,
        category_path=list(entry.category_path),
        wiki_path=entry.wiki_path,
        depth=entry.depth,
        sort_order=entry.sort_order,
    )


# ── KB gate ──────────────────────────────────────────────────────────


def _resolve_caller_tenant() -> int:
    """Return the caller's active workspace id from the request store.

    A wiki gate never runs without one: when the store carries no
    tenant (e.g. a mis-configured principal channel), reject rather
    than guess.
    """
    raw = request_context.get_tenant_id()
    try:
        tenant_id = int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        tenant_id = 0
    if tenant_id <= 0:
        raise UnauthorizedError(
            code="auth.tenant_context_missing",
            message="unauthorized: workspace context missing",
        )
    return tenant_id


async def require_wiki_kb(
    *,
    kb_id: str,
    kb_service: KBService,
    kb_share_service: KBShareService,
    write: bool = False,
) -> None:
    """Validate KB access and that the wiki pipeline is enabled for it.

    The gate enforces tenant boundaries on top of existence: reads
    require the caller to own the KB or hold a share grant on it;
    mutations (``write=True``) are restricted to the owner tenant —
    shared access is read-only. The caller's tenant is read from the
    request context store (same source the session deps use). A
    missing KB reads as ``NotFoundError`` (404); a KB without the wiki
    feature enabled is rejected with a ``ValidationError`` (422).
    """
    tenant_id = _resolve_caller_tenant()
    kb = await kb_service.get_knowledge_base_by_id(knowledge_base_id=kb_id)
    if write and kb.tenant_id != tenant_id:
        raise PermissionDeniedError(
            code="wiki.kb_forbidden",
            message=f"forbidden: knowledge base {kb_id} not writable",
        )
    if not write:
        allowed = await kb_share_service.can_access_knowledge_base(
            tenant_id=tenant_id,
            owner_tenant_id=kb.tenant_id,
            knowledge_base_id=kb_id,
        )
        if not allowed:
            raise PermissionDeniedError(
                code="wiki.kb_forbidden",
                message=f"forbidden: knowledge base {kb_id} not accessible",
            )
    strategy = kb.indexing_strategy or {}
    if strategy.get("wiki_enabled") is not True:
        raise ValidationError(
            code="wiki.kb_wiki_not_enabled",
            message=f"wiki feature is not enabled for knowledge base {kb_id}",
        )


__all__ = [
    "WikiAutoFixData",
    "WikiEnvelope",
    "WikiFolderCreateRequest",
    "WikiFolderListData",
    "WikiFolderNodeView",
    "WikiFolderUpdateRequest",
    "WikiFolderView",
    "WikiGraphData",
    "WikiIndexData",
    "WikiIndexEntryView",
    "WikiIndexGroupView",
    "WikiIssueStatusRequest",
    "WikiIssueView",
    "WikiLintReport",
    "WikiMessage",
    "WikiPageCreateRequest",
    "WikiPageListData",
    "WikiPageMoveRequest",
    "WikiPageUpdateRequest",
    "WikiPageView",
    "WikiRevertRequest",
    "WikiRevisionListData",
    "WikiRevisionView",
    "WikiSearchData",
    "WikiStats",
    "folder_node_to_view",
    "folder_to_view",
    "index_to_view",
    "issue_to_view",
    "page_to_view",
    "require_wiki_kb",
    "revision_list_to_view",
    "revision_to_view",
]
