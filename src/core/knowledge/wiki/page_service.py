"""Wiki page service — CRUD, version-tracking, and slug handling.

Request-scoped service over the already-merged wiki repositories: page
create / read / update / delete, the version-bump policy (a user-visible
field change advances ``version``; bookkeeping-only writes do not), the
bidirectional link maintenance, the structured index and link-graph
views, and the link / slug repair passes.

Version-tracking policy (mirrors the upstream contract): ``version`` is a
user-visible edit counter, not a row-rewrite counter. :meth:`update_page`
advances it only when at least one of title / content / summary /
page_type / status / aliases actually changes; every other write path
(:meth:`update_page_meta`, :meth:`move_page`, link maintenance) persists
through ``update_meta`` and leaves it untouched.

The web layer consumes the service through :func:`src.core.knowledge.wiki.factory.build_wiki_page_service`.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import uuid4

from src.common.exception import NotFoundError, ValidationError
from src.core.knowledge.wiki.hierarchy import (
    apply_folder_to_page,
    normalize_wiki_hierarchy,
)
from src.core.knowledge.wiki.link_utils import (
    collect_link_refs,
    linkify_content,
    parse_out_links,
    rewrite_dead_wiki_links,
    slug_namespace,
    strip_page_chunk_citations,
    strip_wiki_inline_chunk_citations,
)
from src.core.knowledge.wiki.slug_utils import resolve_dead_slug
from src.core.knowledge.wiki.types import (
    WIKI_GRAPH_MODE_EGO,
    WIKI_GRAPH_MODE_OVERVIEW,
    WIKI_INDEX_CONTENT_PAGE_TYPES,
    WIKI_INDEX_DEFAULT_LIMIT,
    WIKI_INDEX_MAX_LIMIT,
    WIKI_PAGE_STATUS_PUBLISHED,
    WIKI_PAGE_TYPE_INDEX,
    WikiGraphData,
    WikiGraphEdge,
    WikiGraphMeta,
    WikiGraphNode,
    WikiGraphRequest,
    WikiIndexGroup,
    WikiIndexResponse,
    WikiPageListFilter,
    WikiPageListResponse,
    WikiStats,
    normalize_edit_source,
)
from src.db.dao.wiki_page_repository import WikiFolderRepository, WikiPageRepository
from src.db.models.wiki_page import WikiIndexEntry, WikiPage, WikiPageLite

logger = logging.getLogger(__name__)

# Default index page body created on first open of a KB with no index row.
_DEFAULT_INDEX_CONTENT = (
    "# Wiki Index\n\nThis is the index page. It will be automatically updated as pages are added.\n"
)

# Error codes shared with the persistence layer.
_ERROR_PAGE_NOT_FOUND = "wiki.page_not_found"


class WikiPageService:
    """Page-centric wiki operations; constructed per request."""

    def __init__(
        self,
        *,
        page_repo: WikiPageRepository,
        folder_repo: WikiFolderRepository,
    ) -> None:
        self._page_repo = page_repo
        self._folder_repo = folder_repo

    # ── Create ──────────────────────────────────────────────────────

    async def create_page(
        self,
        *,
        page: WikiPage,
        edit_source: str = "",
        editor_id: str = "",
    ) -> WikiPage:
        """Create a new wiki page and return the persisted row.

        ``slug`` and ``knowledge_base_id`` are required; defaults are
        applied for status (``published``) and version (``1``). Outbound
        links are parsed from the body and the page's folder-derived
        directory cache is computed before insert.
        """
        if not page.slug.strip():
            raise ValidationError(
                code="wiki.page_slug_required",
                message="wiki page slug is required",
            )
        if not page.knowledge_base_id:
            raise ValidationError(
                code="wiki.page_kb_required",
                message="knowledge_base_id is required",
            )

        now = datetime.now(UTC)
        row = page.model_copy(
            update={
                "id": page.id or str(uuid4()),
                "status": page.status or WIKI_PAGE_STATUS_PUBLISHED,
                "version": page.version or 1,
                "last_edit_source": normalize_edit_source(edit_source),
                "last_editor_id": editor_id,
            }
        )
        row = strip_page_chunk_citations(row)
        row = row.model_copy(update={"out_links": parse_out_links(row.content)})
        row = await apply_folder_to_page(row, folder_repo=self._folder_repo)
        row = normalize_wiki_hierarchy(row)
        row = row.model_copy(update={"created_at": now, "updated_at": now})

        persisted = await self._page_repo.create(row)
        await self._update_in_links(
            knowledge_base_id=persisted.knowledge_base_id,
            source_slug=persisted.slug,
            targets=persisted.out_links,
        )
        return persisted

    # ── Update ──────────────────────────────────────────────────────

    async def update_page(
        self,
        *,
        page: WikiPage,
        edit_source: str = "",
        editor_id: str = "",
    ) -> WikiPage:
        """Apply a user-visible edit, advancing ``version`` when content changed.

        The incoming ``page`` is the full desired state. When at least one
        user-visible field differs from the stored row the write goes
        through ``update`` (which bumps ``version`` under an optimistic
        lock); otherwise bookkeeping fields persist through ``update_meta``
        and the version is preserved. Inbound links are refreshed from the
        re-parsed out-links in both cases.
        """
        existing = await self._require_page(page.knowledge_base_id, page.slug)
        row = strip_page_chunk_citations(page)
        old_out_links = existing.out_links

        content_changed = (
            existing.title != row.title
            or existing.content != row.content
            or existing.summary != row.summary
            or existing.page_type != row.page_type
            or existing.status != row.status
            or existing.aliases != row.aliases
        )

        merged = existing.model_copy(
            update={
                "title": row.title,
                "content": row.content,
                "summary": row.summary,
                "page_type": row.page_type,
                "aliases": list(row.aliases),
                "source_refs": list(row.source_refs),
                "chunk_refs": list(row.chunk_refs),
                "page_metadata": dict(row.page_metadata),
                "parent_slug": row.parent_slug,
                "folder_id": row.folder_id,
                "sort_order": row.sort_order,
                "status": row.status,
            }
        )
        merged = await apply_folder_to_page(merged, folder_repo=self._folder_repo)
        merged = merged.model_copy(update={"out_links": parse_out_links(merged.content)})
        merged = normalize_wiki_hierarchy(merged)

        now = datetime.now(UTC)
        if content_changed:
            authored = merged.model_copy(
                update={
                    "version": existing.version,
                    "last_edit_source": normalize_edit_source(edit_source),
                    "last_editor_id": editor_id,
                    "updated_at": now,
                }
            )
            persisted = await self._page_repo.update(row=authored, now=now)
            await self._remove_in_links(
                knowledge_base_id=existing.knowledge_base_id,
                source_slug=existing.slug,
                targets=old_out_links,
            )
            await self._update_in_links(
                knowledge_base_id=existing.knowledge_base_id,
                source_slug=existing.slug,
                targets=persisted.out_links,
            )
            return persisted

        bookkeeping = merged.model_copy(update={"version": existing.version, "updated_at": now})
        persisted = await self._page_repo.update_meta(row=bookkeeping, now=now)
        await self._remove_in_links(
            knowledge_base_id=existing.knowledge_base_id,
            source_slug=existing.slug,
            targets=old_out_links,
        )
        await self._update_in_links(
            knowledge_base_id=existing.knowledge_base_id,
            source_slug=existing.slug,
            targets=persisted.out_links,
        )
        return persisted

    async def update_page_meta(self, *, page: WikiPage) -> WikiPage:
        """Persist metadata (status, source refs, placement) without a version bump.

        No link re-parse: the stored ``out_links`` are trusted as-is.
        """
        row = normalize_wiki_hierarchy(page)
        return await self._page_repo.update_meta(
            row=row.model_copy(update={"updated_at": datetime.now(UTC)}),
            now=datetime.now(UTC),
        )

    async def update_auto_linked_content(self, *, page: WikiPage) -> WikiPage:
        """Persist machine-only link-decorated content without a version bump.

        Out-links are re-parsed from the new body and bidirectional in-link
        references are refreshed so link navigation stays consistent; only
        the user-facing revision counter is preserved.
        """
        existing = await self._require_page(page.knowledge_base_id, page.slug)
        old_out_links = existing.out_links
        body = strip_wiki_inline_chunk_citations(page.content)
        row = existing.model_copy(
            update={
                "content": body,
                "out_links": parse_out_links(body),
                "updated_at": datetime.now(UTC),
            }
        )
        persisted = await self._page_repo.update_auto_linked_content(row=row, now=datetime.now(UTC))
        await self._remove_in_links(
            knowledge_base_id=existing.knowledge_base_id,
            source_slug=existing.slug,
            targets=old_out_links,
        )
        await self._update_in_links(
            knowledge_base_id=existing.knowledge_base_id,
            source_slug=existing.slug,
            targets=persisted.out_links,
        )
        return persisted

    async def move_page(self, *, knowledge_base_id: str, slug: str, folder_id: str) -> WikiPage:
        """Relocate a page into ``folder_id`` (``""`` = root) and refresh its cache.

        Bookkeeping-only write — no version bump.
        """
        page = await self._require_page(knowledge_base_id, slug)
        row = await apply_folder_to_page(
            page.model_copy(update={"folder_id": folder_id.strip()}),
            folder_repo=self._folder_repo,
        )
        now = datetime.now(UTC)
        row = normalize_wiki_hierarchy(row.model_copy(update={"updated_at": now}))
        return await self._page_repo.update_meta(row=row, now=now)

    # ── Read ────────────────────────────────────────────────────────

    async def get_page_by_slug(self, *, knowledge_base_id: str, slug: str) -> WikiPage:
        """Return one page by (KB, slug); raise ``wiki.page_not_found`` when absent."""
        row = await self._page_repo.get_by_slug_or_none(
            knowledge_base_id=knowledge_base_id, slug=slug
        )
        if row is None:
            raise NotFoundError(
                code=_ERROR_PAGE_NOT_FOUND,
                message=f"wiki page {slug} not found in knowledge base {knowledge_base_id}",
            )
        return strip_page_chunk_citations(row)

    async def get_page_by_id(self, *, id: str) -> WikiPage:
        """Return one page by id; raise ``wiki.page_not_found`` when absent."""
        row = await self._page_repo.get_by_id_or_none(id)
        if row is None:
            raise NotFoundError(code=_ERROR_PAGE_NOT_FOUND, message=f"wiki page {id} not found")
        return strip_page_chunk_citations(row)

    async def list_pages(self, *, filters: WikiPageListFilter) -> WikiPageListResponse:
        """Return one page of the KB's wiki pages plus the unpaginated total.

        ``page`` / ``page_size`` default to 1 / 20 when not positive.
        """
        pages, total = await self._page_repo.list_pages(
            knowledge_base_id=filters.knowledge_base_id,
            page_types=[filters.page_type] if filters.page_type else None,
            status=filters.status,
            query=filters.query,
            folder_id=filters.folder_id,
            category_depth=filters.category_depth,
            category_path=filters.category_path or None,
            page=filters.page,
            page_size=filters.page_size,
            sort_by=filters.sort_by,
            sort_order=filters.sort_order,
        )
        cleaned = [normalize_wiki_hierarchy(strip_page_chunk_citations(p)) for p in pages]
        page = max(1, filters.page)
        page_size = filters.page_size if filters.page_size >= 1 else 20
        total_pages = (total + page_size - 1) // page_size
        return WikiPageListResponse(
            pages=cleaned,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    # ── Delete ──────────────────────────────────────────────────────

    async def delete_page(self, *, knowledge_base_id: str, slug: str) -> None:
        """Soft-delete a page and drop its inbound-link references.

        Inbound links are removed from the pages it links to before the
        row is soft-deleted. The snapshot history and synced chunk are
        handled by the ingest wiring in a later change.
        """
        page = await self._require_page(knowledge_base_id, slug)
        await self._remove_in_links(
            knowledge_base_id=knowledge_base_id,
            source_slug=slug,
            targets=page.out_links,
        )
        deleted = await self._page_repo.soft_delete_by_slug(
            knowledge_base_id=knowledge_base_id, slug=slug, now=datetime.now(UTC)
        )
        if not deleted:
            raise NotFoundError(
                code=_ERROR_PAGE_NOT_FOUND,
                message=f"wiki page {slug} not found in knowledge base {knowledge_base_id}",
            )

    # ── Index ───────────────────────────────────────────────────────

    async def get_index(self, *, knowledge_base_id: str, tenant_id: int) -> WikiPage:
        """Return the KB's index page, creating a default one when missing.

        ``tenant_id`` scopes the default row when the KB has no index yet.
        """
        page = await self._page_repo.get_by_slug_or_none(
            knowledge_base_id=knowledge_base_id, slug="index"
        )
        if page is not None:
            return page
        return await self._create_default_page(
            knowledge_base_id=knowledge_base_id,
            tenant_id=tenant_id,
            slug="index",
            title="Index",
            page_type=WIKI_PAGE_TYPE_INDEX,
            content=_DEFAULT_INDEX_CONTENT,
        )

    async def get_index_view(
        self,
        *,
        knowledge_base_id: str,
        tenant_id: int,
        page_types: list[str] | None = None,
        limit: int = 0,
        cursor: str = "",
    ) -> WikiIndexResponse:
        """Build the structured index response without materializing directory markdown.

        ``page_types`` narrows which groups are included (empty = every
        known content type). ``limit`` is the per-group window (default 50,
        capped at 200). ``cursor`` is an opaque offset string; it applies
        uniformly to every group.
        """
        index_page = await self.get_index(knowledge_base_id=knowledge_base_id, tenant_id=tenant_id)

        limit = WIKI_INDEX_DEFAULT_LIMIT if limit <= 0 else min(limit, WIKI_INDEX_MAX_LIMIT)
        offset = 0
        if cursor:
            try:
                offset = int(cursor)
            except ValueError as exc:
                raise ValidationError(
                    code="wiki.index_invalid_cursor",
                    message=f"invalid cursor {cursor}",
                ) from exc
            if offset < 0:
                raise ValidationError(
                    code="wiki.index_invalid_cursor",
                    message=f"invalid cursor {cursor}",
                )

        selected = list(page_types) if page_types else list(WIKI_INDEX_CONTENT_PAGE_TYPES)
        groups: list[WikiIndexGroup] = []
        for page_type in selected:
            entries, total = await self._page_repo.list_by_type_light(
                knowledge_base_id=knowledge_base_id,
                page_type=page_type,
                limit=limit,
                offset=offset,
            )
            next_cursor = ""
            if len(entries) == limit and offset + len(entries) < total:
                next_cursor = str(offset + limit)
            groups.append(
                WikiIndexGroup(
                    type=page_type,
                    total=total,
                    items=entries,
                    next_cursor=next_cursor,
                )
            )

        intro = index_page.content
        if not intro.strip():
            intro = index_page.summary
        return WikiIndexResponse(intro=intro, version=index_page.version, groups=groups)

    # ── Graph ───────────────────────────────────────────────────────

    async def get_graph(self, *, request: WikiGraphRequest) -> WikiGraphData:
        """Return a slice of the wiki link graph for visualization."""
        if not request.knowledge_base_id:
            raise ValidationError(
                code="wiki.graph_kb_required",
                message="wiki graph request requires a knowledge base id",
            )
        pages = await self._page_repo.list_all(knowledge_base_id=request.knowledge_base_id)
        return compute_graph_subset(pages, request)

    # ── Stats / maintenance ─────────────────────────────────────────

    async def get_stats(self, *, knowledge_base_id: str) -> WikiStats:
        """Return aggregate statistics about the KB's wiki.

        The pending-task / pending-issue counts and the in-progress flag
        need wiring that is not yet merged, so they are reported as zero /
        ``False``.
        """
        counts = await self._page_repo.count_by_type(knowledge_base_id=knowledge_base_id)
        total = sum(counts.values())
        orphans = await self._page_repo.count_orphans(knowledge_base_id=knowledge_base_id)
        pages = await self._page_repo.list_all(knowledge_base_id=knowledge_base_id)
        total_links = sum(len(page.out_links) for page in pages)
        recent, _ = await self._page_repo.list_pages(
            knowledge_base_id=knowledge_base_id,
            page=1,
            page_size=10,
            sort_by="updated_at",
            sort_order="desc",
        )
        return WikiStats(
            total_pages=total,
            pages_by_type=counts,
            total_links=total_links,
            orphan_count=orphans,
            recent_updates=[strip_page_chunk_citations(p) for p in recent],
        )

    async def rebuild_links(self, *, knowledge_base_id: str) -> None:
        """Re-parse every page's body and rebuild bidirectional link references.

        Link rebuild is metadata-only — no version bump. A page deleted
        mid-rebuild is skipped rather than aborting the pass.
        """
        pages = await self._page_repo.list_all(knowledge_base_id=knowledge_base_id)
        page_by_slug = {page.slug: page for page in pages}
        out_by_slug = {page.slug: parse_out_links(page.content) for page in pages}
        in_by_slug: dict[str, list[str]] = {}
        for source_slug, targets in out_by_slug.items():
            for target in targets:
                if target not in page_by_slug:
                    continue
                backlinks = in_by_slug.setdefault(target, [])
                if source_slug not in backlinks:
                    backlinks.append(source_slug)

        now = datetime.now(UTC)
        for page in pages:
            row = page.model_copy(
                update={
                    "in_links": list(in_by_slug.get(page.slug, [])),
                    "out_links": list(out_by_slug.get(page.slug, [])),
                    "updated_at": now,
                }
            )
            try:
                await self._page_repo.update_meta(row=row, now=now)
            except NotFoundError:
                logger.warning("wiki: rebuild links: page %s gone mid-rebuild", page.slug)

    async def inject_cross_links(self, *, knowledge_base_id: str, affected_slugs: list[str]) -> int:
        """Inject ``[[wiki-links]]`` for title/alias mentions in affected pages.

        Pure text replacement — no LLM call. Pages that are already fully
        linked, the index page, and pages deleted mid-pass are skipped.
        Returns how many pages were actually updated.
        """
        pages = await self._page_repo.list_all(knowledge_base_id=knowledge_base_id)
        if len(pages) < 2:
            return 0
        refs = collect_link_refs(pages)
        if not refs:
            return 0

        affected = set(affected_slugs)
        updated = 0
        for page in pages:
            if page.slug not in affected or page.page_type == WIKI_PAGE_TYPE_INDEX:
                continue
            new_content, changed = linkify_content(page.content, refs, page.slug)
            if not changed:
                continue
            try:
                await self.update_auto_linked_content(
                    page=page.model_copy(update={"content": new_content})
                )
            except NotFoundError:
                continue
            updated += 1
        return updated

    async def repair_content_links(
        self, *, knowledge_base_id: str, self_slug: str, content: str
    ) -> tuple[str, bool]:
        """Rewrite dead ``[[slug]]`` references whose target is almost certainly mangled.

        Rewrite-only: a dead link is corrected only when a confident live
        candidate exists in the same namespace prefix, otherwise it is left
        untouched. Returns the possibly-updated content and whether any
        rewrite happened.
        """
        if not content.strip():
            return content, False
        out_links = parse_out_links(content)
        if not out_links:
            return content, False

        exist_map = await self._page_repo.exists_slugs(
            knowledge_base_id=knowledge_base_id, slugs=out_links
        )
        dead_prefixes = {
            slug_namespace(link)
            for link in out_links
            if link != self_slug and not exist_map.get(link, False)
        }
        if not dead_prefixes:
            return content, False

        all_slugs = await self._page_repo.list_all_slugs(knowledge_base_id=knowledge_base_id)
        live_by_prefix: dict[str, set[str]] = {}
        candidates: list[str] = []
        for slug in all_slugs:
            namespace = slug_namespace(slug)
            if namespace not in dead_prefixes:
                continue
            live_by_prefix.setdefault(namespace, set()).add(slug)
            candidates.append(slug)
        if not candidates:
            return content, False

        title_to_slug: dict[str, str] = {}
        lites = await self._page_repo.list_by_slugs(
            knowledge_base_id=knowledge_base_id, slugs=candidates
        )
        for lite in lites.values():
            if lite.title:
                title_to_slug[lite.title] = lite.slug

        resolve_cache: dict[str, str] = {}

        def _resolve(norm: str, display: str) -> tuple[str, bool]:
            if norm == self_slug or exist_map.get(norm, False):
                return "", False
            key = norm + "\x00" + display
            if key in resolve_cache:
                cached = resolve_cache[key]
                return cached, cached != ""
            resolved, ok = resolve_dead_slug(
                norm,
                display,
                live_by_prefix.get(slug_namespace(norm), set()),
                title_to_slug,
            )
            if not ok or resolved == norm:
                resolve_cache[key] = ""
                return "", False
            resolve_cache[key] = resolved
            return resolved, True

        return rewrite_dead_wiki_links(content, _resolve)

    # ── Pass-through reads ──────────────────────────────────────────

    async def list_all_pages(self, *, knowledge_base_id: str) -> list[WikiPage]:
        """Return every live page in the KB without pagination."""
        return await self._page_repo.list_all(knowledge_base_id=knowledge_base_id)

    async def list_by_type(self, *, knowledge_base_id: str, page_type: str) -> list[WikiPage]:
        """Return every live page of one type in the KB, newest first."""
        return await self._page_repo.list_by_type(
            knowledge_base_id=knowledge_base_id, page_type=page_type
        )

    async def list_pages_by_source_ref(
        self, *, knowledge_base_id: str, source_knowledge_id: str
    ) -> list[WikiPage]:
        """Return pages whose source refs cite a source knowledge id."""
        return await self._page_repo.list_by_source_ref(
            knowledge_base_id=knowledge_base_id, source_knowledge_id=source_knowledge_id
        )

    async def list_slugs_by_source_ref(
        self, *, knowledge_base_id: str, source_knowledge_id: str
    ) -> list[str]:
        """Return just the slugs of pages citing a source knowledge id."""
        return await self._page_repo.list_slugs_by_source_ref(
            knowledge_base_id=knowledge_base_id, source_knowledge_id=source_knowledge_id
        )

    async def list_by_slugs(
        self, *, knowledge_base_id: str, slugs: list[str]
    ) -> dict[str, WikiPageLite]:
        """Resolve slugs to slim page projections in one IN query."""
        return await self._page_repo.list_by_slugs(knowledge_base_id=knowledge_base_id, slugs=slugs)

    async def list_summaries_by_knowledge_ids(
        self, *, knowledge_base_id: str, knowledge_ids: list[str]
    ) -> dict[str, str]:
        """Return summary content keyed by the knowledge id that authored it."""
        return await self._page_repo.list_summaries_by_knowledge_ids(
            knowledge_base_id=knowledge_base_id, knowledge_ids=knowledge_ids
        )

    async def exists_slugs(self, *, knowledge_base_id: str, slugs: list[str]) -> dict[str, bool]:
        """Report which slugs are live (non-archived, non-deleted) in the KB."""
        return await self._page_repo.exists_slugs(knowledge_base_id=knowledge_base_id, slugs=slugs)

    async def list_all_slugs(self, *, knowledge_base_id: str) -> list[str]:
        """Return every non-archived slug in the KB."""
        return await self._page_repo.list_all_slugs(knowledge_base_id=knowledge_base_id)

    async def list_pages_cursor(
        self, *, knowledge_base_id: str, cursor: str = "", limit: int = 100
    ) -> tuple[list[WikiPage], str]:
        """Return a page of pages ordered by id after ``cursor``."""
        return await self._page_repo.list_pages_cursor(
            knowledge_base_id=knowledge_base_id, cursor=cursor, limit=limit
        )

    async def list_by_type_recent(
        self, *, knowledge_base_id: str, page_type: str, limit: int = 200
    ) -> list[WikiIndexEntry]:
        """Return the most recently updated pages of a type (light projection)."""
        return await self._page_repo.list_by_type_recent(
            knowledge_base_id=knowledge_base_id, page_type=page_type, limit=limit
        )

    async def count_by_type(self, *, knowledge_base_id: str) -> dict[str, int]:
        """Return page counts grouped by type for the KB."""
        return await self._page_repo.count_by_type(knowledge_base_id=knowledge_base_id)

    async def search_pages(
        self, *, knowledge_base_id: str, query: str, limit: int = 10
    ) -> list[WikiPage]:
        """Case-insensitive search over pages, ranked by where the query hit."""
        pages = await self._page_repo.search(
            knowledge_base_id=knowledge_base_id, query=query, limit=limit
        )
        return [strip_page_chunk_citations(page) for page in pages]

    async def rebuild_index_page(self, *, knowledge_base_id: str) -> None:
        """No-op kept for agent-tool call-site compatibility.

        The directory is assembled on demand by :meth:`get_index_view`;
        the intro that still lives on the index row is managed by the
        ingest pipeline on batch completion.
        """

    # ── Internal helpers ────────────────────────────────────────────

    async def _require_page(self, knowledge_base_id: str, slug: str) -> WikiPage:
        """Return the live page for (KB, slug) or raise ``wiki.page_not_found``."""
        row = await self._page_repo.get_by_slug_or_none(
            knowledge_base_id=knowledge_base_id, slug=slug
        )
        if row is None:
            raise NotFoundError(
                code=_ERROR_PAGE_NOT_FOUND,
                message=f"wiki page {slug} not found in knowledge base {knowledge_base_id}",
            )
        return row

    async def _create_default_page(
        self,
        *,
        knowledge_base_id: str,
        tenant_id: int,
        slug: str,
        title: str,
        page_type: str,
        content: str,
    ) -> WikiPage:
        """Create the default page for a KB that has none yet."""
        now = datetime.now(UTC)
        row = WikiPage(
            id=str(uuid4()),
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            slug=slug,
            title=title,
            page_type=page_type,
            status=WIKI_PAGE_STATUS_PUBLISHED,
            content=content,
            summary=title,
            version=1,
            created_at=now,
            updated_at=now,
        )
        row = normalize_wiki_hierarchy(row)
        return await self._page_repo.create(row)

    async def _update_in_links(
        self, *, knowledge_base_id: str, source_slug: str, targets: list[str]
    ) -> None:
        """Add ``source_slug`` to the in-links of each existing target page."""
        for target_slug in targets:
            target = await self._page_repo.get_by_slug_or_none(
                knowledge_base_id=knowledge_base_id, slug=target_slug
            )
            if target is None or source_slug in target.in_links:
                continue
            row = target.model_copy(update={"in_links": [*target.in_links, source_slug]})
            try:
                await self._page_repo.update_meta(row=row, now=datetime.now(UTC))
            except NotFoundError:
                continue

    async def _remove_in_links(
        self, *, knowledge_base_id: str, source_slug: str, targets: list[str]
    ) -> None:
        """Remove ``source_slug`` from the in-links of each existing target page."""
        for target_slug in targets:
            target = await self._page_repo.get_by_slug_or_none(
                knowledge_base_id=knowledge_base_id, slug=target_slug
            )
            if target is None:
                continue
            new_in_links = [link for link in target.in_links if link != source_slug]
            if len(new_in_links) == len(target.in_links):
                continue
            row = target.model_copy(update={"in_links": new_in_links})
            try:
                await self._page_repo.update_meta(row=row, now=datetime.now(UTC))
            except NotFoundError:
                continue


def compute_graph_subset(pages: list[WikiPage], request: WikiGraphRequest) -> WikiGraphData:
    """Compute the requested subgraph from the full page list.

    Pure and I/O-free so the mode / limit / type-filter behavior can be
    tested without a repository.
    """
    mode = request.mode or WIKI_GRAPH_MODE_OVERVIEW
    type_allow = {t for t in request.types if t}
    has_type_filter = bool(type_allow)
    page_by_slug = {page.slug: page for page in pages}
    link_count = {page.slug: len(page.in_links) + len(page.out_links) for page in pages}

    selected: set[str]
    if mode == WIKI_GRAPH_MODE_EGO:
        if not request.center:
            raise ValidationError(
                code="wiki.graph_center_required",
                message="ego graph requires a center slug",
            )
        if request.center not in page_by_slug:
            raise ValidationError(
                code="wiki.graph_center_not_found",
                message=f"ego center slug {request.center} not found",
            )
        depth = request.depth if request.depth >= 1 else 1
        selected = _bfs_ego_slugs(page_by_slug, request.center, depth, type_allow, request.limit)
    else:
        candidates = [page for page in pages if not has_type_filter or page.page_type in type_allow]
        candidates.sort(key=lambda page: (-link_count[page.slug], page.slug))
        if request.limit > 0 and len(candidates) > request.limit:
            candidates = candidates[: request.limit]
        selected = {page.slug for page in candidates}

    nodes = [
        WikiGraphNode(
            slug=page.slug,
            title=page.title,
            page_type=page.page_type,
            link_count=link_count[page.slug],
        )
        for page in page_by_slug.values()
        if page.slug in selected
    ]
    nodes.sort(key=lambda node: (-node.link_count, node.slug))

    edges = [
        WikiGraphEdge(source=page.slug, target=target)
        for page in pages
        if page.slug in selected
        for target in page.out_links
        if target in selected
    ]

    total = len(pages)
    if mode == WIKI_GRAPH_MODE_OVERVIEW and has_type_filter:
        total = sum(1 for page in pages if page.page_type in type_allow)

    meta = WikiGraphMeta(
        mode=mode,
        total=total,
        returned=len(nodes),
        truncated=len(nodes) < total,
    )
    if mode == WIKI_GRAPH_MODE_EGO:
        meta = meta.model_copy(update={"center": request.center, "depth": max(1, request.depth)})
    return WikiGraphData(nodes=nodes, edges=edges, meta=meta)


def _bfs_ego_slugs(
    page_by_slug: dict[str, WikiPage],
    center: str,
    depth: int,
    type_allow: set[str],
    limit: int,
) -> set[str]:
    """Compute the undirected BFS neighborhood of ``center`` up to ``depth`` hops.

    Type-filtered pages are excluded from the result and are also not
    traversed through, so a filter that hides the index page does not leak
    the whole wiki through it.
    """
    has_type_filter = bool(type_allow)
    center_page = page_by_slug.get(center)
    if center_page is None or (has_type_filter and center_page.page_type not in type_allow):
        return set()

    visited: set[str] = {center}
    frontier = [center]

    for _ in range(depth):
        if limit > 0 and len(visited) >= limit:
            break
        next_frontier: list[str] = []
        for slug in frontier:
            page = page_by_slug.get(slug)
            if page is None:
                continue
            for neighbor in [*page.out_links, *page.in_links]:
                if neighbor in visited:
                    continue
                neighbor_page = page_by_slug.get(neighbor)
                if neighbor_page is None:
                    continue
                if has_type_filter and neighbor_page.page_type not in type_allow:
                    continue
                visited.add(neighbor)
                next_frontier.append(neighbor)
                if limit > 0 and len(visited) >= limit:
                    break
            if limit > 0 and len(visited) >= limit:
                break
        frontier = next_frontier
        if not frontier:
            break

    return visited


__all__ = [
    "WikiPageService",
    "compute_graph_subset",
]
