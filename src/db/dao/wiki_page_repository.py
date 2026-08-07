"""Wiki page and folder persistence — raw SQL only, no ORM.

Maps the persistence methods of the wiki page contract that operate on
the ``wiki_pages`` and ``wiki_folders`` tables. Revision *snapshots*
(the ``wiki_page_revisions`` table) are a separate storage contract and
are not created here; the version-tracking writes below are the
``wiki_pages`` half of that contract:

- :meth:`WikiPageRepository.update` is the user-visible edit: it rewrites
  the content-bearing columns and bumps ``version`` under an optimistic
  ``WHERE version = ?`` guard, so a stale edit fails with a conflict
  instead of clobbering a newer revision.
- :meth:`WikiPageRepository.update_meta` refreshes link / reference /
  placement bookkeeping without advancing the version counter.
- :meth:`WikiPageRepository.update_auto_linked_content` rewrites
  link-decorated body text without advancing it either, so the version
  number tracks intentional edits rather than link-maintenance passes.

Folder management lives on :class:`WikiFolderRepository`; the two classes
together cover the single folder-aware repository surface of the wiki
page contract.

Every query is ``sqlalchemy.text()`` with named ``bindparams``; JSON
columns are bound through the dialect-aware JSONB bind type. Reads filter
soft-deleted rows (``deleted_at is null``).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import cast

from sqlalchemy import JSON, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import CursorResult

from src.common.exception import ConflictError, NotFoundError
from src.common.json import BindParams, SqlValue
from src.db.dao.generic_repository import GenericRepository
from src.db.models.wiki_page import WikiFolder, WikiIndexEntry, WikiPage, WikiPageLite

# JSONB on Postgres, JSON on other dialects (e.g. SQLite in tests).
_JSON_BIND_TYPE = JSON().with_variant(JSONB(), "postgresql")

# Page vocabulary. Kept local so the persistence layer stays
# dependency-free; the shared vocabulary lives in the wiki domain types.
_STATUS_ARCHIVED = "archived"
_STATUS_PUBLISHED = "published"
_PAGE_TYPE_INDEX = "index"
_PAGE_TYPE_SUMMARY = "summary"

# Sort keys the paged listing accepts. Any column referenced here is a
# fixed, read-only column name; caller-supplied ``sort_by`` is validated
# against this allow-list before it reaches the SQL string.
_ALLOWED_SORT_COLUMNS: frozenset[str] = frozenset(
    {"title", "created_at", "updated_at", "page_type", "wiki_path", "sort_order", "depth"}
)


def _escape_like_pattern(value: str) -> str:
    """Escape LIKE / ILIKE metacharacters for safe ``%``-wrapped patterns.

    Order matters: the backslash is escaped first so the wildcards it
    would otherwise introduce stay literal.
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class WikiPageRepository(GenericRepository[WikiPage]):
    """`wiki_pages`-table SQL — page CRUD, listing, and aggregates."""

    model_class = WikiPage

    # ── Page writes ──────────────────────────────────────────────────

    async def create(self, row: WikiPage) -> WikiPage:
        """Insert a page; the application supplies the UUID ``id``."""
        return await self.insert(row)

    async def update(self, *, row: WikiPage, now: datetime) -> WikiPage:
        """Apply a user-visible edit with optimistic version locking.

        ``row.version`` is the expected current version. The write bumps
        the stored version and rewrites every content-bearing column.
        Raises ``NotFoundError`` when the page is absent or soft-deleted
        and ``ConflictError`` when the stored version differs.
        """
        data = row.model_dump()
        params: BindParams = {
            key: data[key]
            for key in (
                "title",
                "content",
                "summary",
                "page_type",
                "status",
                "aliases",
                "out_links",
                "source_refs",
                "chunk_refs",
                "page_metadata",
                "parent_slug",
                "folder_id",
                "category_path",
                "wiki_path",
                "depth",
                "sort_order",
                "last_edit_source",
                "last_editor_id",
            )
        }
        params.update(
            {
                "id": row.id,
                "expected_version": row.version,
                "version": row.version + 1,
                "updated_at": now,
            }
        )
        json_bps = self._json_bindparams(
            (
                "aliases",
                "category_path",
                "source_refs",
                "chunk_refs",
                "out_links",
                "page_metadata",
            )
        )
        stmt = text(
            f"update {self._table} set "
            "title = :title, content = :content, summary = :summary, "
            "page_type = :page_type, status = :status, aliases = :aliases, "
            "out_links = :out_links, source_refs = :source_refs, "
            "chunk_refs = :chunk_refs, page_metadata = :page_metadata, "
            "parent_slug = :parent_slug, folder_id = :folder_id, "
            "category_path = :category_path, wiki_path = :wiki_path, "
            "depth = :depth, sort_order = :sort_order, "
            "last_edit_source = :last_edit_source, last_editor_id = :last_editor_id, "
            "version = :version, updated_at = :updated_at "
            "where id = :id and version = :expected_version and deleted_at is null "
            "returning *"
        ).bindparams(*json_bps, **params)
        result = await self._session.execute(stmt)
        mapping = result.mappings().first()
        if mapping is None:
            if not await self._page_exists(row.id):
                raise NotFoundError(
                    code="wiki.page_not_found",
                    message=f"wiki page {row.id} not found",
                )
            raise ConflictError(
                code="wiki.page_conflict",
                message=f"wiki page {row.id} changed since version {row.version}",
            )
        return self._hydrate(mapping)

    async def update_meta(self, *, row: WikiPage, now: datetime) -> WikiPage:
        """Refresh bookkeeping / provenance fields without a version bump.

        The user-visible body (``title`` / ``content`` / ``summary`` /
        ``page_type``) is left untouched; links, refs, aliases, status,
        and placement are refreshed here. Raises ``NotFoundError`` when
        the page is absent.
        """
        data = row.model_dump()
        params: BindParams = {
            key: data[key]
            for key in (
                "in_links",
                "out_links",
                "aliases",
                "status",
                "source_refs",
                "chunk_refs",
                "page_metadata",
                "parent_slug",
                "folder_id",
                "category_path",
                "wiki_path",
                "depth",
                "sort_order",
            )
        }
        params.update({"id": row.id, "updated_at": now})
        json_bps = self._json_bindparams(
            (
                "in_links",
                "out_links",
                "aliases",
                "source_refs",
                "chunk_refs",
                "page_metadata",
                "category_path",
            )
        )
        stmt = text(
            f"update {self._table} set "
            "in_links = :in_links, out_links = :out_links, aliases = :aliases, "
            "status = :status, source_refs = :source_refs, chunk_refs = :chunk_refs, "
            "page_metadata = :page_metadata, parent_slug = :parent_slug, "
            "folder_id = :folder_id, category_path = :category_path, "
            "wiki_path = :wiki_path, depth = :depth, sort_order = :sort_order, "
            "updated_at = :updated_at "
            "where id = :id and deleted_at is null returning *"
        ).bindparams(*json_bps, **params)
        result = await self._session.execute(stmt)
        mapping = result.mappings().first()
        if mapping is None:
            raise NotFoundError(
                code="wiki.page_not_found",
                message=f"wiki page {row.id} not found",
            )
        return self._hydrate(mapping)

    async def update_auto_linked_content(self, *, row: WikiPage, now: datetime) -> WikiPage:
        """Persist link-decorated content without a version bump.

        The automatic link passes rewrite the same revision with
        wiki-link markup added or removed; treating those as real edits
        would make freshly ingested pages appear as v2 on first view.
        Raises ``NotFoundError`` when the page is absent.
        """
        data = row.model_dump()
        params: BindParams = {"content": data["content"], "out_links": data["out_links"]}
        params.update({"id": row.id, "updated_at": now})
        json_bps = self._json_bindparams(("out_links",))
        stmt = text(
            f"update {self._table} set content = :content, out_links = :out_links, "
            "updated_at = :updated_at "
            "where id = :id and deleted_at is null returning *"
        ).bindparams(*json_bps, **params)
        result = await self._session.execute(stmt)
        mapping = result.mappings().first()
        if mapping is None:
            raise NotFoundError(
                code="wiki.page_not_found",
                message=f"wiki page {row.id} not found",
            )
        return self._hydrate(mapping)

    async def soft_delete_by_slug(
        self, *, knowledge_base_id: str, slug: str, now: datetime
    ) -> bool:
        """Soft-delete a live page by (KB, slug). Returns whether a row changed."""
        stmt = text(
            f"update {self._table} set deleted_at = :now, updated_at = :now "
            "where knowledge_base_id = :kb_id and slug = :slug and deleted_at is null"
        ).bindparams(kb_id=knowledge_base_id, slug=slug, now=now)
        result = await self._session.execute(stmt)
        return (cast("CursorResult[SqlValue]", result).rowcount or 0) > 0

    async def soft_delete_by_id(self, *, id: str, now: datetime) -> bool:
        """Soft-delete a live page by id. Returns whether a row changed."""
        stmt = text(
            f"update {self._table} set deleted_at = :now, updated_at = :now "
            "where id = :id and deleted_at is null"
        ).bindparams(id=id, now=now)
        result = await self._session.execute(stmt)
        return (cast("CursorResult[SqlValue]", result).rowcount or 0) > 0

    # ── Page reads ───────────────────────────────────────────────────

    async def get_by_id_or_none(self, id: str) -> WikiPage | None:
        """Return one live page by id, or ``None``."""
        return await self.find_by_primary_key({"id": id})

    async def get_by_slug_or_none(self, *, knowledge_base_id: str, slug: str) -> WikiPage | None:
        """Return one live page by (KB, slug), or ``None``."""
        return await self.find_unique_by_column_values(
            {"knowledge_base_id": knowledge_base_id, "slug": slug}
        )

    async def list_pages(
        self,
        *,
        knowledge_base_id: str,
        page_types: list[str] | None = None,
        status: str = "",
        query: str = "",
        folder_id: str | None = None,
        category_depth: int | None = None,
        category_path: list[str] | None = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "",
        sort_order: str = "desc",
    ) -> tuple[list[WikiPage], int]:
        """Return one page of the KB's wiki pages and the total count.

        Mirrors the page-listing filters: an optional page-type
        allow-list (``page_types``), a status filter, a full-text body /
        alias query, an exact folder placement, an exact category depth,
        and an exact (already-cleaned) category path. ``page_types`` and
        ``category_path`` are expected pre-normalised by the caller.
        """
        where_parts = ["knowledge_base_id = :kb_id", "deleted_at is null"]
        params: BindParams = {"kb_id": knowledge_base_id}

        if page_types is not None and len(page_types) == 1:
            where_parts.append("page_type = :page_type")
            params["page_type"] = page_types[0]
        elif page_types is not None and len(page_types) > 1:
            placeholders = ", ".join(f":pt_{i}" for i in range(len(page_types)))
            where_parts.append(f"page_type in ({placeholders})")
            params.update({f"pt_{i}": t for i, t in enumerate(page_types)})
        if status:
            where_parts.append("status = :status")
            params["status"] = status
        if query:
            where_parts.append(
                "(to_tsvector('simple', coalesce(title, '') || ' ' || coalesce(content, '')) "
                "@@ plainto_tsquery('simple', :query) or aliases::text ilike :alias_like)"
            )
            params["query"] = query
            params["alias_like"] = f"%{query}%"
        if folder_id is not None:
            where_parts.append("folder_id = :folder_id")
            params["folder_id"] = folder_id
        if category_depth is not None:
            where_parts.append("depth = :category_depth")
            params["category_depth"] = category_depth
        if category_path:
            where_parts.append("category_path = :category_path")
            params["category_path"] = cast("SqlValue", category_path)
        where = " and ".join(where_parts)
        json_bps = self._json_bindparams(("category_path",)) if "category_path" in params else []

        count_stmt = text(f"select count(*) from {self._table} where {where}").bindparams(
            *json_bps, **params
        )
        total = int((await self._session.execute(count_stmt)).scalar_one() or 0)

        sort_by = sort_by if sort_by in _ALLOWED_SORT_COLUMNS else "updated_at"
        direction = "asc" if sort_order == "asc" else "desc"
        if sort_by == "wiki_path":
            order_clause = (
                "order by "
                f"{self._category_rank_order()}, "
                f"wiki_path {direction}, sort_order asc, title asc"
            )
        else:
            order_clause = f"order by {sort_by} {direction}"

        page = max(1, page)
        page_size = page_size if page_size >= 1 else 20
        offset = (page - 1) * page_size
        stmt = text(
            f"select * from {self._table} where {where} {order_clause} limit :limit offset :offset"
        ).bindparams(*json_bps, **params, limit=page_size, offset=offset)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()], total

    async def list_by_type(self, *, knowledge_base_id: str, page_type: str) -> list[WikiPage]:
        """Return every live page of a type in the KB, newest first."""
        stmt = text(
            f"select * from {self._table} "
            "where knowledge_base_id = :kb_id and page_type = :page_type "
            "and deleted_at is null order by updated_at desc"
        ).bindparams(kb_id=knowledge_base_id, page_type=page_type)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def list_by_type_light(
        self,
        *,
        knowledge_base_id: str,
        page_type: str,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[WikiIndexEntry], int]:
        """Return the light directory projection for one page type.

        Excludes archived pages; ``limit`` is clamped to [1, 200]. The
        total non-archived count for the type is returned alongside so
        the caller can render "showing N of M".
        """
        limit = max(1, min(limit or 50, 200))
        offset = max(0, offset)
        where = (
            "knowledge_base_id = :kb_id and page_type = :page_type "
            "and status <> :archived and deleted_at is null"
        )
        params: BindParams = {
            "kb_id": knowledge_base_id,
            "page_type": page_type,
            "archived": _STATUS_ARCHIVED,
        }
        count_stmt = text(f"select count(*) from {self._table} where {where}").bindparams(**params)
        total = int((await self._session.execute(count_stmt)).scalar_one() or 0)
        if total == 0:
            return [], 0
        stmt = text(
            "select slug, title, summary, parent_slug, category_path, "
            "wiki_path, depth, sort_order "
            f"from {self._table} where {where} "
            f"order by {self._category_rank_order()}, wiki_path asc, "
            "sort_order asc, title asc limit :limit offset :offset"
        ).bindparams(**params, limit=limit, offset=offset)
        result = await self._session.execute(stmt)
        return [WikiIndexEntry.model_validate(dict(m)) for m in result.mappings().all()], total

    async def list_by_source_ref(
        self, *, knowledge_base_id: str, source_knowledge_id: str
    ) -> list[WikiPage]:
        """Return pages whose source_refs reference a source knowledge id.

        Matches both the bare id form and the legacy "id|title" prefix
        form via a JSONB containment check plus a text LIKE fallback.
        """
        needle = json.dumps([source_knowledge_id])
        prefix = json.dumps(source_knowledge_id + "|")
        prefix_str = prefix[:-1] if prefix.endswith('"') else prefix
        like_pattern = "%" + _escape_like_pattern(prefix_str) + "%"
        stmt = text(
            f"select * from {self._table} "
            "where knowledge_base_id = :kb_id and deleted_at is null "
            "and (source_refs @> cast(:needle as jsonb) or source_refs::text like :pattern)"
        ).bindparams(kb_id=knowledge_base_id, needle=needle, pattern=like_pattern)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def list_slugs_by_source_ref(
        self, *, knowledge_base_id: str, source_knowledge_id: str
    ) -> list[str]:
        """Return just the slugs of pages referencing a source knowledge id.

        Same predicate as :meth:`list_by_source_ref`, projected to a
        single column so the ingest pipeline does not load full rows when
        it only needs a "before" set of slugs.
        """
        needle = json.dumps([source_knowledge_id])
        prefix = json.dumps(source_knowledge_id + "|")
        prefix_str = prefix[:-1] if prefix.endswith('"') else prefix
        like_pattern = "%" + _escape_like_pattern(prefix_str) + "%"
        stmt = text(
            f"select slug from {self._table} "
            "where knowledge_base_id = :kb_id and deleted_at is null "
            "and (source_refs @> cast(:needle as jsonb) or source_refs::text like :pattern)"
        ).bindparams(kb_id=knowledge_base_id, needle=needle, pattern=like_pattern)
        result = await self._session.execute(stmt)
        return [str(s) for s in result.scalars().all() if s is not None]

    async def list_by_slugs(
        self, *, knowledge_base_id: str, slugs: list[str]
    ) -> dict[str, WikiPageLite]:
        """Resolve slugs to slim page projections in one IN query.

        Slugs not present in the KB are silently dropped from the result.
        """
        if not slugs:
            return {}
        placeholders = ", ".join(f":slug_{i}" for i in range(len(slugs)))
        params: BindParams = {f"slug_{i}": s for i, s in enumerate(slugs)}
        params["kb_id"] = knowledge_base_id
        stmt = text(
            "select slug, title, page_type, status, aliases, out_links "
            f"from {self._table} where knowledge_base_id = :kb_id "
            f"and slug in ({placeholders}) and deleted_at is null"
        ).bindparams(**params)
        result = await self._session.execute(stmt)
        out: dict[str, WikiPageLite] = {}
        for mapping in result.mappings().all():
            lite = WikiPageLite.model_validate(dict(mapping))
            out[lite.slug] = lite
        return out

    async def exists_slugs(self, *, knowledge_base_id: str, slugs: list[str]) -> dict[str, bool]:
        """Report which slugs are live (non-archived, non-deleted) in the KB.

        Slugs not present at all map to ``False``.
        """
        if not slugs:
            return {}
        placeholders = ", ".join(f":slug_{i}" for i in range(len(slugs)))
        params: BindParams = {f"slug_{i}": s for i, s in enumerate(slugs)}
        params["kb_id"] = knowledge_base_id
        stmt = text(
            f"select slug from {self._table} where knowledge_base_id = :kb_id "
            f"and slug in ({placeholders}) and status <> :archived and deleted_at is null"
        ).bindparams(**params, archived=_STATUS_ARCHIVED)
        result = await self._session.execute(stmt)
        live = {str(m["slug"]) for m in result.mappings().all()}
        return {s: s in live for s in slugs}

    async def list_all_slugs(self, *, knowledge_base_id: str) -> list[str]:
        """Return every non-archived page slug in the KB."""
        stmt = text(
            f"select slug from {self._table} "
            "where knowledge_base_id = :kb_id and status <> :archived "
            "and deleted_at is null"
        ).bindparams(kb_id=knowledge_base_id, archived=_STATUS_ARCHIVED)
        result = await self._session.execute(stmt)
        return [str(s) for s in result.scalars().all() if s is not None]

    async def list_pages_cursor(
        self, *, knowledge_base_id: str, cursor: str = "", limit: int = 100
    ) -> tuple[list[WikiPage], str]:
        """Return a page of pages ordered by id after ``cursor``.

        ``cursor`` is the id of the last row of the previous page; the
        returned ``next_cursor`` is empty at the end of the stream.
        ``limit`` is clamped to [1, 500].
        """
        limit = max(1, min(limit or 100, 500))
        where = "knowledge_base_id = :kb_id and deleted_at is null"
        params: BindParams = {"kb_id": knowledge_base_id}
        if cursor:
            where += " and id > :cursor"
            params["cursor"] = cursor
        stmt = text(
            f"select * from {self._table} where {where} order by id asc limit :limit"
        ).bindparams(**params, limit=limit)
        result = await self._session.execute(stmt)
        pages = [self._hydrate(m) for m in result.mappings().all()]
        next_cursor = pages[-1].id if len(pages) == limit else ""
        return pages, next_cursor

    async def list_by_type_recent(
        self, *, knowledge_base_id: str, page_type: str, limit: int = 200
    ) -> list[WikiIndexEntry]:
        """Return the most recently updated pages of a type, light projection.

        Excludes archived pages; ``limit`` is clamped to [1, 1000].
        """
        limit = max(1, min(limit or 200, 1000))
        stmt = text(
            "select slug, title, summary, parent_slug, category_path, "
            "wiki_path, depth, sort_order "
            f"from {self._table} where knowledge_base_id = :kb_id "
            "and page_type = :page_type and status <> :archived "
            "and deleted_at is null order by updated_at desc limit :limit"
        ).bindparams(
            kb_id=knowledge_base_id,
            page_type=page_type,
            archived=_STATUS_ARCHIVED,
            limit=limit,
        )
        result = await self._session.execute(stmt)
        return [WikiIndexEntry.model_validate(dict(m)) for m in result.mappings().all()]

    async def list_all(self, *, knowledge_base_id: str) -> list[WikiPage]:
        """Return every live page in the KB, ordered by type then title."""
        stmt = text(
            f"select * from {self._table} "
            "where knowledge_base_id = :kb_id and deleted_at is null "
            "order by page_type asc, title asc"
        ).bindparams(kb_id=knowledge_base_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def list_recent_for_suggestions(
        self, *, tenant_id: int, knowledge_base_ids: list[str], limit: int = 5
    ) -> list[WikiPage]:
        """Return recent user-visible pages across KBs for suggested questions.

        Excludes the index page, archived pages, and empty titles.
        """
        if not knowledge_base_ids or limit <= 0:
            return []
        placeholders = ", ".join(f":kb_{i}" for i in range(len(knowledge_base_ids)))
        params: BindParams = {f"kb_{i}": k for i, k in enumerate(knowledge_base_ids)}
        params["tenant_id"] = tenant_id
        stmt = text(
            f"select * from {self._table} "
            f"where tenant_id = :tenant_id and knowledge_base_id in ({placeholders}) "
            "and page_type <> :index_type and status = :published "
            "and title <> '' and deleted_at is null "
            "order by updated_at desc limit :limit"
        ).bindparams(
            **params,
            index_type=_PAGE_TYPE_INDEX,
            published=_STATUS_PUBLISHED,
            limit=limit,
        )
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def search(
        self, *, knowledge_base_id: str, query: str, limit: int = 10
    ) -> list[WikiPage]:
        """Case-insensitive POSIX-regex search ranked by where the query hit.

        A title hit outranks a slug hit, which outranks a summary hit,
        which outranks a body mention; ``updated_at`` breaks ties.
        """
        limit = max(1, min(limit or 10, 50))
        rank_expr = (
            "case when title ~* :query1 then 4 "
            "when slug ~* :query2 then 3 "
            "when summary ~* :query3 then 2 "
            "when content ~* :query4 then 1 else 0 end as match_rank"
        )
        stmt = text(
            f"select *, {rank_expr} from {self._table} "
            "where knowledge_base_id = :kb_id and deleted_at is null "
            "and (title ~* :query1 or slug ~* :query2 or summary ~* :query3 "
            "or content ~* :query4) and status <> :archived "
            "order by match_rank desc, updated_at desc limit :limit"
        ).bindparams(
            kb_id=knowledge_base_id,
            query1=query,
            query2=query,
            query3=query,
            query4=query,
            archived=_STATUS_ARCHIVED,
            limit=limit,
        )
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def count_by_type(self, *, knowledge_base_id: str) -> dict[str, int]:
        """Return page counts grouped by type for the KB."""
        stmt = text(
            f"select page_type, count(*) as cnt from {self._table} "
            "where knowledge_base_id = :kb_id and deleted_at is null "
            "group by page_type"
        ).bindparams(kb_id=knowledge_base_id)
        result = await self._session.execute(stmt)
        counts: dict[str, int] = {}
        for mapping in result.mappings().all():
            counts[str(mapping["page_type"])] = int(mapping["cnt"])
        return counts

    async def count_orphans(self, *, knowledge_base_id: str) -> int:
        """Count pages with no inbound links, excluding the index page."""
        stmt = text(
            f"select count(*) from {self._table} "
            "where knowledge_base_id = :kb_id and deleted_at is null "
            "and (in_links is null or in_links = '[]'::jsonb) "
            "and page_type <> :index_type"
        ).bindparams(kb_id=knowledge_base_id, index_type=_PAGE_TYPE_INDEX)
        total = (await self._session.execute(stmt)).scalar_one()
        return int(total) if total is not None else 0

    async def list_summaries_by_knowledge_ids(
        self, *, knowledge_base_id: str, knowledge_ids: list[str]
    ) -> dict[str, str]:
        """Return summary content keyed by the knowledge id that authored it.

        Only summary pages are considered; each page's content is mapped
        to every knowledge id present in its ``source_refs`` (either the
        bare id or the legacy "id|title" form). A knowledge id with no
        surviving summary page is silently absent from the result.
        """
        if not knowledge_ids:
            return {}
        clauses: list[str] = []
        params: BindParams = {"kb_id": knowledge_base_id}
        for i, kid in enumerate(knowledge_ids):
            if not kid:
                continue
            needle = json.dumps([kid])
            prefix = json.dumps(kid + "|")
            prefix_str = prefix[:-1] if prefix.endswith('"') else prefix
            clauses.append(f"source_refs @> cast(:needle_{i} as jsonb)")
            params[f"needle_{i}"] = needle
            clauses.append(f"source_refs::text like :pattern_{i}")
            params[f"pattern_{i}"] = "%" + _escape_like_pattern(prefix_str) + "%"
        if not clauses:
            return {}
        where = (
            "knowledge_base_id = :kb_id and page_type = :summary_type "
            "and status <> :archived and deleted_at is null and (" + " or ".join(clauses) + ")"
        )
        params["summary_type"] = _PAGE_TYPE_SUMMARY
        params["archived"] = _STATUS_ARCHIVED
        stmt = text(f"select content, source_refs from {self._table} where {where}").bindparams(
            **params
        )
        result = await self._session.execute(stmt)
        kid_set = {k for k in knowledge_ids if k}
        out: dict[str, str] = {}
        for mapping in result.mappings().all():
            content = mapping["content"]
            refs = mapping["source_refs"]
            if not isinstance(content, str) or not isinstance(refs, list):
                continue
            for ref in refs:
                if not isinstance(ref, str):
                    continue
                ref_kid = ref.split("|", 1)[0] if "|" in ref else ref
                if ref_kid in kid_set and ref_kid not in out:
                    out[ref_kid] = content
        return out

    # ── Folder aggregates (read-side, on wiki_pages) ─────────────────

    async def count_pages_in_folder(self, *, knowledge_base_id: str, folder_id: str) -> int:
        """Count live, non-archived pages placed directly in a folder."""
        stmt = text(
            f"select count(*) from {self._table} "
            "where knowledge_base_id = :kb_id and folder_id = :folder_id "
            "and status <> :archived and deleted_at is null"
        ).bindparams(kb_id=knowledge_base_id, folder_id=folder_id, archived=_STATUS_ARCHIVED)
        total = (await self._session.execute(stmt)).scalar_one()
        return int(total) if total is not None else 0

    async def count_pages_by_folder(
        self, *, knowledge_base_id: str, page_types: list[str] | None = None
    ) -> dict[str, int]:
        """Return live, non-archived page counts grouped by folder_id."""
        where = "knowledge_base_id = :kb_id and status <> :archived and deleted_at is null"
        params: BindParams = {"kb_id": knowledge_base_id, "archived": _STATUS_ARCHIVED}
        if page_types:
            placeholders = ", ".join(f":pt_{i}" for i in range(len(page_types)))
            where += f" and page_type in ({placeholders})"
            params.update({f"pt_{i}": t for i, t in enumerate(page_types)})
        stmt = text(
            f"select folder_id, count(*) as cnt from {self._table} where {where} group by folder_id"
        ).bindparams(**params)
        result = await self._session.execute(stmt)
        out: dict[str, int] = {}
        for mapping in result.mappings().all():
            out[str(mapping["folder_id"])] = int(mapping["cnt"])
        return out

    async def list_pages_by_folder_ids(
        self, *, knowledge_base_id: str, folder_ids: list[str]
    ) -> list[WikiPage]:
        """Return the live pages placed in any of the given folders."""
        if not folder_ids:
            return []
        placeholders = ", ".join(f":folder_{i}" for i in range(len(folder_ids)))
        params: BindParams = {f"folder_{i}": f for i, f in enumerate(folder_ids)}
        params["kb_id"] = knowledge_base_id
        stmt = text(
            f"select * from {self._table} where knowledge_base_id = :kb_id "
            f"and folder_id in ({placeholders}) and deleted_at is null"
        ).bindparams(**params)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    # ── Internal helpers ─────────────────────────────────────────────

    async def _page_exists(self, id: str) -> bool:
        """Return whether a live page row exists for ``id``."""
        stmt = text(
            f"select 1 from {self._table} where id = :id and deleted_at is null limit 1"
        ).bindparams(id=id)
        return (await self._session.execute(stmt)).mappings().first() is not None

    @staticmethod
    def _category_rank_order() -> str:
        """Return the SQL fragment that ranks empty category paths last.

        Postgres-flavoured: ``jsonb_array_length`` on the cached JSON
        column, so pages under a folder sort before wiki-root pages.
        """
        return "case when coalesce(jsonb_array_length(category_path), 0) > 0 then 0 else 1 end asc"


class WikiFolderRepository(GenericRepository[WikiFolder]):
    """`wiki_folders`-table SQL — adjacency-list folder tree management."""

    model_class = WikiFolder

    # ── Folder writes ────────────────────────────────────────────────

    async def create(self, row: WikiFolder) -> WikiFolder:
        """Insert a folder; the application supplies the UUID ``id``."""
        return await self.insert(row)

    async def update(self, *, row: WikiFolder, now: datetime) -> WikiFolder:
        """Overwrite the mutable folder columns, raising on absence."""
        data = row.model_dump()
        params: BindParams = {
            key: data[key] for key in ("parent_id", "name", "path", "depth", "sort_order")
        }
        params.update({"id": row.id, "updated_at": now})
        stmt = text(
            f"update {self._table} set parent_id = :parent_id, name = :name, "
            "path = :path, depth = :depth, sort_order = :sort_order, "
            "updated_at = :updated_at where id = :id and deleted_at is null returning *"
        ).bindparams(**params)
        result = await self._session.execute(stmt)
        mapping = result.mappings().first()
        if mapping is None:
            raise NotFoundError(
                code="wiki.folder_not_found",
                message=f"wiki folder {row.id} not found",
            )
        return self._hydrate(mapping)

    async def delete(self, *, knowledge_base_id: str, id: str, now: datetime) -> None:
        """Atomically soft-delete an empty folder.

        The emptiness test lives in the same SQL statement as the delete,
        so a page move or child-folder create racing the caller's earlier
        checks cannot leave a dangling ``folder_id``. Raises
        ``NotFoundError`` when the folder is absent and ``ConflictError``
        when it still holds a live page or child folder.
        """
        stmt = text(
            f"update {self._table} set deleted_at = :now, updated_at = :now "
            "where knowledge_base_id = :kb_id and id = :id and deleted_at is null "
            "and not exists (select 1 from wiki_pages "
            "  where knowledge_base_id = :kb_id and folder_id = :id and deleted_at is null) "
            "and not exists (select 1 from wiki_folders as child "
            "  where child.knowledge_base_id = :kb_id and child.parent_id = :id "
            "  and child.deleted_at is null)"
        ).bindparams(kb_id=knowledge_base_id, id=id, now=now)
        result = await self._session.execute(stmt)
        if (cast("CursorResult[SqlValue]", result).rowcount or 0) > 0:
            return
        if not await self._folder_exists(knowledge_base_id, id):
            raise NotFoundError(
                code="wiki.folder_not_found",
                message=f"wiki folder {id} not found",
            )
        raise ConflictError(
            code="wiki.folder_not_empty",
            message=f"wiki folder {id} is not empty",
        )

    # ── Folder reads ─────────────────────────────────────────────────

    async def get_by_id_or_none(self, *, knowledge_base_id: str, id: str) -> WikiFolder | None:
        """Return one live folder by (KB, id), or ``None``."""
        return await self.find_unique_by_column_values(
            {"knowledge_base_id": knowledge_base_id, "id": id}
        )

    async def get_child_by_name_or_none(
        self, *, knowledge_base_id: str, parent_id: str, name: str
    ) -> WikiFolder | None:
        """Return one live child folder by name under ``parent_id``, or ``None``."""
        return await self.find_unique_by_column_values(
            {
                "knowledge_base_id": knowledge_base_id,
                "parent_id": parent_id,
                "name": name,
            }
        )

    async def list_children(self, *, knowledge_base_id: str, parent_id: str) -> list[WikiFolder]:
        """Return the live direct children of a folder, sorted."""
        stmt = text(
            f"select * from {self._table} "
            "where knowledge_base_id = :kb_id and parent_id = :parent_id "
            "and deleted_at is null order by sort_order asc, name asc"
        ).bindparams(kb_id=knowledge_base_id, parent_id=parent_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def list_all(self, *, knowledge_base_id: str) -> list[WikiFolder]:
        """Return every live folder of the KB, shallowest first."""
        stmt = text(
            f"select * from {self._table} "
            "where knowledge_base_id = :kb_id and deleted_at is null "
            "order by depth asc, path asc"
        ).bindparams(kb_id=knowledge_base_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def list_distinct_category_paths(
        self, *, knowledge_base_id: str, max_paths: int = 150
    ) -> list[str]:
        """Return the materialized folder paths of the KB, ordered by path.

        Returns the raw "/"-joined path strings; the caller applies the
        domain-level cleaning before use. The folder tree is the single
        source of truth for directory placement.
        """
        max_paths = max(1, max_paths)
        stmt = text(
            f"select path from {self._table} "
            "where knowledge_base_id = :kb_id and path <> '' and deleted_at is null "
            "order by path asc limit :limit"
        ).bindparams(kb_id=knowledge_base_id, limit=max_paths)
        result = await self._session.execute(stmt)
        return [str(p) for p in result.scalars().all() if p]

    # ── Internal helpers ─────────────────────────────────────────────

    async def _folder_exists(self, knowledge_base_id: str, id: str) -> bool:
        """Return whether a live folder row exists for (KB, id)."""
        stmt = text(
            f"select 1 from {self._table} "
            "where knowledge_base_id = :kb_id and id = :id and deleted_at is null limit 1"
        ).bindparams(kb_id=knowledge_base_id, id=id)
        return (await self._session.execute(stmt)).mappings().first() is not None


__all__ = ["WikiFolderRepository", "WikiPageRepository", "_escape_like_pattern"]
