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

from src.common.exception import ValidationError
from src.common.json import JsonObject
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.wiki.lint_service import WikiLintReport
from src.core.knowledge.wiki.types import (
    WikiFolder,
    WikiFolderNode,
    WikiGraphData,
    WikiIndexEntry,
    WikiIndexResponse,
    WikiPage,
    WikiStats,
)

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


async def require_wiki_kb(*, kb_id: str, kb_service: KBService) -> None:
    """Validate the KB exists and the wiki pipeline is enabled for it.

    Mirrors the upstream handler's per-endpoint gate: a missing KB reads
    as ``NotFoundError`` (404); a KB without the wiki feature enabled is
    rejected with a ``ValidationError`` (422).
    """
    kb = await kb_service.get_knowledge_base_by_id(knowledge_base_id=kb_id)
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
    "WikiLintReport",
    "WikiMessage",
    "WikiPageCreateRequest",
    "WikiPageListData",
    "WikiPageMoveRequest",
    "WikiPageUpdateRequest",
    "WikiPageView",
    "WikiSearchData",
    "WikiStats",
    "folder_node_to_view",
    "folder_to_view",
    "index_to_view",
    "page_to_view",
    "require_wiki_kb",
]
