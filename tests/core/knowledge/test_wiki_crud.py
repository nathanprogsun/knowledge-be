"""Unit + integration tests for the wiki page / folder services.

Unit tests drive the services with stateful ``AsyncMock`` repositories
(pytest, AAA) covering validation, the version-bump policy, link
maintenance, index/graph/stats views, and folder-tree semantics. The
pure link / slug / hierarchy helpers are exercised directly.

Integration tests run against the real applied schema (revision 0021+);
isolation is by per-test generated tenant ids and unique entity ids, and
they are skipped when Postgres is not reachable (set
``DATABASE_URL_OVERRIDE``).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool

from src.common.exception import ConflictError, NotFoundError, ValidationError
from src.core.knowledge.wiki.folders import WikiFolderService, validate_folder_name
from src.core.knowledge.wiki.hierarchy import (
    build_wiki_path,
    normalize_wiki_hierarchy,
    wiki_folder_segments,
)
from src.core.knowledge.wiki.link_utils import (
    LinkRef,
    linkify_content,
    parse_out_links,
    rewrite_dead_wiki_links,
    strip_page_chunk_citations,
    strip_wiki_inline_chunk_citations,
)
from src.core.knowledge.wiki.page_service import WikiPageService, compute_graph_subset
from src.core.knowledge.wiki.slug_utils import (
    WikiSlugHandles,
    resolve_dead_slug,
)
from src.core.knowledge.wiki.types import (
    WIKI_GRAPH_MODE_EGO,
    WIKI_GRAPH_MODE_OVERVIEW,
    WikiGraphRequest,
    WikiPageListFilter,
)
from src.db.base import DatabaseEngine
from src.db.dao.wiki_page_repository import WikiFolderRepository, WikiPageRepository
from src.db.models.wiki_page import WikiFolder, WikiIndexEntry, WikiPage
from src.settings import get_settings, reset_settings_cache
from tests.integration.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 1, tzinfo=UTC)

# ── Sample row builders ─────────────────────────────────────────────


def _pid() -> str:
    return f"page-{uuid.uuid4().hex[:12]}"


def _fid() -> str:
    return f"folder-{uuid.uuid4().hex[:12]}"


def _kb() -> str:
    return f"kb-{uuid.uuid4().hex[:8]}"


def _sample_page(
    *,
    tenant_id: int = 1,
    knowledge_base_id: str = "kb-1",
    slug: str = "entity/acme",
    id: str | None = None,
    **overrides: object,
) -> WikiPage:
    values: dict[str, object] = {
        "id": id or _pid(),
        "tenant_id": tenant_id,
        "knowledge_base_id": knowledge_base_id,
        "slug": slug,
        "title": "Acme",
        "page_type": "entity",
        "status": "published",
        "content": "Acme is a fictional company.",
        "summary": "A fictional company.",
        "parent_slug": "",
        "folder_id": "",
        "category_path": [],
        "wiki_path": "entity/Acme",
        "depth": 0,
        "sort_order": 0,
        "source_refs": [],
        "chunk_refs": [],
        "in_links": [],
        "out_links": [],
        "page_metadata": {},
        "aliases": [],
        "version": 1,
        "last_edit_source": "",
        "last_editor_id": "",
        "created_at": _NOW,
        "updated_at": _NOW,
        "deleted_at": None,
    }
    values.update(overrides)
    return WikiPage.model_validate(values)


def _sample_folder(
    *,
    tenant_id: int = 1,
    knowledge_base_id: str = "kb-1",
    id: str = "folder-root",
    parent_id: str = "",
    name: str = "AI",
    path: str = "AI",
    depth: int = 1,
    **overrides: object,
) -> WikiFolder:
    values: dict[str, object] = {
        "id": id,
        "tenant_id": tenant_id,
        "knowledge_base_id": knowledge_base_id,
        "parent_id": parent_id,
        "name": name,
        "path": path,
        "depth": depth,
        "sort_order": 0,
        "created_at": _NOW,
        "updated_at": _NOW,
        "deleted_at": None,
    }
    values.update(overrides)
    return WikiFolder.model_validate(values)


# ── Repository mocks (stateful via side_effect closures) ────────────


def _make_page_repo() -> AsyncMock:
    """``AsyncMock(spec=WikiPageRepository)`` with closure-captured state."""
    repo = AsyncMock(spec=WikiPageRepository)
    rows: dict[str, WikiPage] = {}
    meta_calls: list[WikiPage] = []
    repo.rows = rows  # type: ignore[attr-defined]
    repo.meta_calls = meta_calls  # type: ignore[attr-defined]

    async def _create(row: WikiPage) -> WikiPage:
        rows[row.id] = row
        return row

    async def _get_by_slug_or_none(*, knowledge_base_id: str, slug: str) -> WikiPage | None:
        for row in rows.values():
            if (
                row.knowledge_base_id == knowledge_base_id
                and row.slug == slug
                and row.deleted_at is None
            ):
                return row
        return None

    async def _get_by_id_or_none(id: str) -> WikiPage | None:
        row = rows.get(id)
        if row is None or row.deleted_at is not None:
            return None
        return row

    async def _update(*, row: WikiPage, now: datetime) -> WikiPage:
        existing = rows.get(row.id)
        if existing is None or existing.deleted_at is not None:
            raise NotFoundError(code="wiki.page_not_found", message="missing")
        stored = row.model_copy(update={"version": existing.version + 1, "updated_at": now})
        rows[row.id] = stored
        return stored

    async def _update_meta(*, row: WikiPage, now: datetime) -> WikiPage:
        existing = rows.get(row.id)
        if existing is None or existing.deleted_at is not None:
            raise NotFoundError(code="wiki.page_not_found", message="missing")
        stored = row.model_copy(update={"updated_at": now})
        rows[row.id] = stored
        meta_calls.append(stored)
        return stored

    async def _update_auto_linked_content(*, row: WikiPage, now: datetime) -> WikiPage:
        existing = rows.get(row.id)
        if existing is None or existing.deleted_at is not None:
            raise NotFoundError(code="wiki.page_not_found", message="missing")
        stored = row.model_copy(update={"updated_at": now})
        rows[row.id] = stored
        return stored

    async def _soft_delete_by_slug(*, knowledge_base_id: str, slug: str, now: datetime) -> bool:
        for row in rows.values():
            if (
                row.knowledge_base_id == knowledge_base_id
                and row.slug == slug
                and row.deleted_at is None
            ):
                rows[row.id] = row.model_copy(update={"deleted_at": now})
                return True
        return False

    async def _list_all(*, knowledge_base_id: str) -> list[WikiPage]:
        return [
            row
            for row in rows.values()
            if row.knowledge_base_id == knowledge_base_id and row.deleted_at is None
        ]

    async def _list_pages(**kwargs: object) -> tuple[list[WikiPage], int]:
        kb_id = kwargs["knowledge_base_id"]
        pages = [
            row
            for row in rows.values()
            if row.knowledge_base_id == kb_id and row.deleted_at is None
        ]
        page = max(1, int(kwargs.get("page", 1)))
        page_size = max(1, int(kwargs.get("page_size", 20)))
        start = (page - 1) * page_size
        return pages[start : start + page_size], len(pages)

    async def _list_by_type_light(
        *, knowledge_base_id: str, page_type: str, limit: int = 50, offset: int = 0
    ) -> tuple[list[WikiIndexEntry], int]:
        pages = [
            row
            for row in rows.values()
            if row.knowledge_base_id == knowledge_base_id
            and row.page_type == page_type
            and row.deleted_at is None
        ]
        entries = [
            WikiIndexEntry(
                slug=row.slug,
                title=row.title,
                summary=row.summary,
                parent_slug=row.parent_slug,
                category_path=list(row.category_path),
                wiki_path=row.wiki_path,
                depth=row.depth,
                sort_order=row.sort_order,
            )
            for row in pages
        ]
        return entries[offset : offset + limit], len(entries)

    async def _count_by_type(*, knowledge_base_id: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows.values():
            if row.knowledge_base_id == knowledge_base_id and row.deleted_at is None:
                counts[row.page_type] = counts.get(row.page_type, 0) + 1
        return counts

    async def _count_orphans(*, knowledge_base_id: str) -> int:
        return sum(
            1
            for row in rows.values()
            if row.knowledge_base_id == knowledge_base_id
            and row.deleted_at is None
            and not row.in_links
            and row.page_type != "index"
        )

    async def _list_all_slugs(*, knowledge_base_id: str) -> list[str]:
        return [
            row.slug
            for row in rows.values()
            if row.knowledge_base_id == knowledge_base_id
            and row.deleted_at is None
            and row.status != "archived"
        ]

    async def _exists_slugs(*, knowledge_base_id: str, slugs: list[str]) -> dict[str, bool]:
        live = {
            row.slug
            for row in rows.values()
            if row.knowledge_base_id == knowledge_base_id
            and row.deleted_at is None
            and row.status != "archived"
        }
        return {slug: slug in live for slug in slugs}

    async def _list_by_slugs(*, knowledge_base_id: str, slugs: list[str]) -> dict[str, object]:
        out: dict[str, object] = {}
        for row in rows.values():
            if row.knowledge_base_id == knowledge_base_id and row.slug in slugs:
                out[row.slug] = row
        return out

    async def _search(*, knowledge_base_id: str, query: str, limit: int = 10) -> list[WikiPage]:
        return [
            row
            for row in rows.values()
            if row.knowledge_base_id == knowledge_base_id
            and row.deleted_at is None
            and query in row.title
        ][:limit]

    async def _count_pages_by_folder(
        *, knowledge_base_id: str, page_types: list[str] | None = None
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows.values():
            if (
                row.knowledge_base_id == knowledge_base_id
                and row.deleted_at is None
                and row.status != "archived"
                and (page_types is None or row.page_type in page_types)
            ):
                counts[row.folder_id] = counts.get(row.folder_id, 0) + 1
        return counts

    async def _list_pages_by_folder_ids(
        *, knowledge_base_id: str, folder_ids: list[str]
    ) -> list[WikiPage]:
        return [
            row
            for row in rows.values()
            if row.knowledge_base_id == knowledge_base_id
            and row.folder_id in folder_ids
            and row.deleted_at is None
        ]

    repo.create.side_effect = _create
    repo.get_by_slug_or_none.side_effect = _get_by_slug_or_none
    repo.get_by_id_or_none.side_effect = _get_by_id_or_none
    repo.update.side_effect = _update
    repo.update_meta.side_effect = _update_meta
    repo.update_auto_linked_content.side_effect = _update_auto_linked_content
    repo.soft_delete_by_slug.side_effect = _soft_delete_by_slug
    repo.list_all.side_effect = _list_all
    repo.list_pages.side_effect = _list_pages
    repo.list_by_type_light.side_effect = _list_by_type_light
    repo.count_by_type.side_effect = _count_by_type
    repo.count_orphans.side_effect = _count_orphans
    repo.list_all_slugs.side_effect = _list_all_slugs
    repo.exists_slugs.side_effect = _exists_slugs
    repo.list_by_slugs.side_effect = _list_by_slugs
    repo.search.side_effect = _search
    repo.count_pages_by_folder.side_effect = _count_pages_by_folder
    repo.list_pages_by_folder_ids.side_effect = _list_pages_by_folder_ids
    return repo


def _make_folder_repo() -> AsyncMock:
    """``AsyncMock(spec=WikiFolderRepository)`` with closure-captured state."""
    repo = AsyncMock(spec=WikiFolderRepository)
    rows: dict[str, WikiFolder] = {}
    repo.rows = rows  # type: ignore[attr-defined]

    async def _get_by_id_or_none(*, knowledge_base_id: str, id: str) -> WikiFolder | None:
        row = rows.get(id)
        if row is None or row.deleted_at is not None:
            return None
        if row.knowledge_base_id != knowledge_base_id:
            return None
        return row

    async def _get_child_by_name_or_none(
        *, knowledge_base_id: str, parent_id: str, name: str
    ) -> WikiFolder | None:
        for row in rows.values():
            if (
                row.knowledge_base_id == knowledge_base_id
                and row.parent_id == parent_id
                and row.name == name
                and row.deleted_at is None
            ):
                return row
        return None

    async def _list_all(*, knowledge_base_id: str) -> list[WikiFolder]:
        return [
            row
            for row in rows.values()
            if row.knowledge_base_id == knowledge_base_id and row.deleted_at is None
        ]

    async def _list_children(*, knowledge_base_id: str, parent_id: str) -> list[WikiFolder]:
        return [
            row
            for row in rows.values()
            if row.knowledge_base_id == knowledge_base_id
            and row.parent_id == parent_id
            and row.deleted_at is None
        ]

    async def _create(row: WikiFolder) -> WikiFolder:
        rows[row.id] = row
        return row

    async def _update(*, row: WikiFolder, now: datetime) -> WikiFolder:
        existing = rows.get(row.id)
        if existing is None or existing.deleted_at is not None:
            raise NotFoundError(code="wiki.folder_not_found", message="missing")
        stored = row.model_copy(update={"updated_at": now})
        rows[row.id] = stored
        return stored

    async def _delete(*, knowledge_base_id: str, id: str, now: datetime) -> None:
        row = rows.get(id)
        if row is None or row.deleted_at is not None:
            raise NotFoundError(code="wiki.folder_not_found", message="missing")
        has_children = any(
            other.parent_id == id and other.deleted_at is None for other in rows.values()
        )
        if has_children:
            raise ConflictError(code="wiki.folder_not_empty", message="not empty")
        rows[id] = row.model_copy(update={"deleted_at": now})

    repo.get_by_id_or_none.side_effect = _get_by_id_or_none
    repo.get_child_by_name_or_none.side_effect = _get_child_by_name_or_none
    repo.list_all.side_effect = _list_all
    repo.list_children.side_effect = _list_children
    repo.create.side_effect = _create
    repo.update.side_effect = _update
    repo.delete.side_effect = _delete
    return repo


def _services(
    page_repo: AsyncMock, folder_repo: AsyncMock
) -> tuple[WikiPageService, WikiFolderService]:
    return (
        WikiPageService(page_repo=page_repo, folder_repo=folder_repo),
        WikiFolderService(folder_repo=folder_repo, page_repo=page_repo),
    )


# ── Pure link helpers ───────────────────────────────────────────────


class TestParseOutLinks:
    def test_extracts_and_normalizes_wiki_links(self) -> None:
        content = (
            "See [[Entity/Acme Corp]] and [[Concept/RAG|Retrieval Augmented]] "
            "and [[Entity/Acme Corp]] again."
        )
        assert parse_out_links(content) == ["entity/acme-corp", "concept/rag"]

    def test_empty_content_yields_no_links(self) -> None:
        assert parse_out_links("no links here") == []


class TestStripChunkCitations:
    def test_strips_single_and_multi_handles(self) -> None:
        body = "intro text [c001] and [c002, c003] and a sentence."
        assert strip_wiki_inline_chunk_citations(body) == "intro text and and a sentence."

    def test_strip_page_copy_is_immutable(self) -> None:
        page = _sample_page(content="body [c001] tail", summary="sum [c002]")
        stripped = strip_page_chunk_citations(page)
        assert stripped.content == "body tail"
        assert stripped.summary == "sum"
        assert page.content == "body [c001] tail"


class TestRewriteDeadWikiLinks:
    def test_rewrites_mangled_slug_preserving_display(self) -> None:
        content = "see [[summary/abcdef|The Report]] and [[live-page]]"
        out, changed = rewrite_dead_wiki_links(
            content,
            lambda norm, _disp: (
                ("summary/abcdef12", True) if norm == "summary/abcdef" else ("", False)
            ),
        )
        assert changed is True
        assert out == "see [[summary/abcdef12|The Report]] and [[live-page]]"

    def test_leaves_untouched_when_resolve_refuses(self) -> None:
        content = "see [[dead-page]]"
        out, changed = rewrite_dead_wiki_links(content, lambda _n, _d: ("", False))
        assert out == content
        assert changed is False


class TestLinkifyContent:
    def test_wraps_first_mention_and_skips_already_linked(self) -> None:
        refs = [LinkRef(slug="concept/rag", match_text="RAG")]
        content = "RAG powers RAG everywhere."
        out, changed = linkify_content(content, refs, self_slug="self")
        assert changed is True
        assert out == "[[concept/rag|RAG]] powers RAG everywhere."

    def test_skips_code_blocks_and_inline_code(self) -> None:
        refs = [LinkRef(slug="concept/rag", match_text="RAG")]
        content = "```\nRAG inside fence\n```\nand `RAG` inline, then RAG plain."
        out, _ = linkify_content(content, refs, self_slug="self")
        assert (
            out == "```\nRAG inside fence\n```\nand `RAG` inline, then [[concept/rag|RAG]] plain."
        )

    def test_requires_word_boundary_for_ascii_needles(self) -> None:
        refs = [LinkRef(slug="concept/rag", match_text="RAG")]
        content = "RAGSystem uses RAG."
        out, _ = linkify_content(content, refs, self_slug="self")
        assert out == "RAGSystem uses [[concept/rag|RAG]]."

    def test_skips_self_references(self) -> None:
        refs = [LinkRef(slug="entity/acme", match_text="Acme")]
        content = "Acme is us."
        out, changed = linkify_content(content, refs, self_slug="entity/acme")
        assert out == content
        assert changed is False

    def test_longer_match_text_wins_over_substring(self) -> None:
        refs = [
            LinkRef(slug="entity/acme-corp", match_text="Acme Corporation"),
            LinkRef(slug="entity/acme", match_text="Acme"),
        ]
        content = "Acme Corporation acquired Acme."
        out, _ = linkify_content(content, refs, self_slug="self")
        # the longer mention wins the first (overlapping) occurrence; the
        # standalone "Acme" mention is a distinct safe match and is linked too
        assert out == "[[entity/acme-corp|Acme Corporation]] acquired [[entity/acme|Acme]]."


# ── Pure slug helpers ───────────────────────────────────────────────


class TestResolveDeadSlug:
    def test_already_live_is_noop(self) -> None:
        assert resolve_dead_slug("entity/acme", "", {"entity/acme"}, {}) == (
            "entity/acme",
            True,
        )

    def test_display_text_reverse_lookup(self) -> None:
        live = {"entity/acme-corp"}
        assert resolve_dead_slug(
            "entity/acme-copr", "Acme Corp", live, {"Acme Corp": "entity/acme-corp"}
        ) == (
            "entity/acme-corp",
            True,
        )

    def test_normalized_equality(self) -> None:
        assert resolve_dead_slug("shang-hai-tower", "", {"shanghai-tower"}, {}) == (
            "shanghai-tower",
            True,
        )

    def test_bigram_fallback_and_threshold(self) -> None:
        assert resolve_dead_slug("shang-hai-tower", "", {"shanghai-tower", "other"}, {}) == (
            "shanghai-tower",
            True,
        )
        assert resolve_dead_slug("user-profile", "", {"user-permissions"}, {}) == ("", False)


class TestWikiSlugHandles:
    def test_encode_decode_round_trip(self) -> None:
        handles = WikiSlugHandles()
        encoded = handles.encode_content(
            "see [[summary/abc|The Report]] and [[entity/acme]]",
            known={"summary/abc", "entity/acme"},
        )
        assert encoded == "see [[ref-0|The Report]] and [[ref-1]]"
        assert handles.decode_content(encoded) == (
            "see [[summary/abc|The Report]] and [[entity/acme]]"
        )

    def test_unknown_links_left_untouched(self) -> None:
        handles = WikiSlugHandles()
        assert handles.encode_content("[[entity/other]]", known=set()) == "[[entity/other]]"


# ── Pure hierarchy helpers ──────────────────────────────────────────


class TestHierarchyHelpers:
    def test_build_wiki_path_skips_empty_segments(self) -> None:
        assert build_wiki_path("entity", [], "Acme") == "entity/Acme"
        assert build_wiki_path("entity", ["AI", "RAG"], "Acme") == "entity/AI/RAG/Acme"

    def test_normalize_hierarchy_computes_depth_and_path(self) -> None:
        page = _sample_page(
            title="  Acme  ",
            category_path=["AI", "AI", "摘要"],
            slug="entity/acme",
            parent_slug="  ",
        )
        normalized = normalize_wiki_hierarchy(page)
        assert normalized.parent_slug == ""
        assert normalized.category_path == ["AI"]
        assert normalized.depth == 1
        assert normalized.wiki_path == "entity/AI/Acme"
        assert page.depth == 0  # input untouched

    def test_wiki_folder_segments(self) -> None:
        assert wiki_folder_segments("AI/RAG") == ["AI", "RAG"]
        assert wiki_folder_segments("  ") == []


# ── WikiPageService — create ────────────────────────────────────────


class TestCreatePage:
    async def test_creates_with_defaults_and_out_links(self) -> None:
        page_repo = _make_page_repo()
        folder_repo = _make_folder_repo()
        service, _ = _services(page_repo, folder_repo)

        target = _sample_page(knowledge_base_id="kb-1", slug="entity/acme")
        page_repo.rows[target.id] = target  # type: ignore[attr-defined]

        created = await service.create_page(
            page=_sample_page(
                id="",
                knowledge_base_id="kb-1",
                slug="summary/other",
                content="links to [[entity/acme]] and [[entity/acme|display]]",
                title="Other",
            ),
            edit_source="user",
            editor_id="usr-1",
        )
        assert created.slug == "summary/other"
        assert created.status == "published"
        assert created.version == 1
        assert created.out_links == ["entity/acme"]
        assert created.last_edit_source == "user"
        assert created.last_editor_id == "usr-1"
        assert created.created_at is not None
        # the referenced target page gained a backlink
        assert page_repo.rows[target.id].in_links == ["summary/other"]  # type: ignore[attr-defined]

    async def test_missing_slug_is_rejected(self) -> None:
        service, _ = _services(_make_page_repo(), _make_folder_repo())
        with pytest.raises(ValidationError) as excinfo:
            await service.create_page(page=_sample_page(slug="  "))
        assert excinfo.value.code == "wiki.page_slug_required"

    async def test_missing_kb_is_rejected(self) -> None:
        service, _ = _services(_make_page_repo(), _make_folder_repo())
        with pytest.raises(ValidationError) as excinfo:
            await service.create_page(page=_sample_page(knowledge_base_id=""))
        assert excinfo.value.code == "wiki.page_kb_required"

    async def test_unknown_folder_is_rejected(self) -> None:
        service, _ = _services(_make_page_repo(), _make_folder_repo())
        with pytest.raises(ValidationError) as excinfo:
            await service.create_page(page=_sample_page(folder_id="folder-missing"))
        assert excinfo.value.code == "wiki.page_folder_unknown"

    async def test_folder_application_sets_category_path(self) -> None:
        page_repo = _make_page_repo()
        folder_repo = _make_folder_repo()
        service, _ = _services(page_repo, folder_repo)
        folder_repo.rows["f1"] = _sample_folder(  # type: ignore[attr-defined]
            id="f1", name="RAG", path="AI/RAG", depth=2
        )
        created = await service.create_page(
            page=_sample_page(slug="concept/rag", page_type="concept", folder_id="f1", title="RAG")
        )
        assert created.category_path == ["AI", "RAG"]
        assert created.depth == 2
        assert created.wiki_path == "concept/AI/RAG/RAG"


# ── WikiPageService — update / version policy ───────────────────────


class TestUpdatePage:
    async def test_content_change_bumps_version_and_refreshes_links(self) -> None:
        page_repo = _make_page_repo()
        folder_repo = _make_folder_repo()
        service, _ = _services(page_repo, folder_repo)

        old_target = _sample_page(knowledge_base_id="kb-1", slug="entity/old")
        new_target = _sample_page(knowledge_base_id="kb-1", slug="entity/new")
        page_repo.rows[old_target.id] = old_target  # type: ignore[attr-defined]
        page_repo.rows[new_target.id] = new_target  # type: ignore[attr-defined]

        existing = _sample_page(
            knowledge_base_id="kb-1",
            slug="summary/one",
            content="links [[entity/old]]",
            out_links=["entity/old"],
        )
        page_repo.rows[existing.id] = existing  # type: ignore[attr-defined]

        updated = await service.update_page(
            page=_sample_page(
                id=existing.id,
                knowledge_base_id="kb-1",
                slug="summary/one",
                title="One v2",
                content="links [[entity/new]]",
                out_links=[],
            ),
            edit_source="user",
            editor_id="usr-2",
        )
        assert updated.version == 2
        assert updated.title == "One v2"
        assert updated.out_links == ["entity/new"]
        # old target lost the backlink; new target gained it
        assert page_repo.rows[old_target.id].in_links == []  # type: ignore[attr-defined]
        assert page_repo.rows[new_target.id].in_links == ["summary/one"]  # type: ignore[attr-defined]

    async def test_bookkeeping_only_change_preserves_version(self) -> None:
        page_repo = _make_page_repo()
        folder_repo = _make_folder_repo()
        service, _ = _services(page_repo, folder_repo)

        existing = _sample_page(
            knowledge_base_id="kb-1",
            slug="entity/acme",
            content="same body",
            folder_id="",
        )
        page_repo.rows[existing.id] = existing  # type: ignore[attr-defined]
        folder_repo.rows["f1"] = _sample_folder(id="f1", name="RAG", path="AI/RAG", depth=2)  # type: ignore[attr-defined]

        updated = await service.update_page(
            page=_sample_page(
                id=existing.id,
                knowledge_base_id="kb-1",
                slug="entity/acme",
                title="Acme",
                content="same body",
                folder_id="f1",
                source_refs=["k-9"],
            )
        )
        assert updated.version == 1  # untouched
        page_repo.update.assert_not_called()  # type: ignore[attr-defined]
        page_repo.update_meta.assert_called()  # type: ignore[attr-defined]

    async def test_unknown_page_raises_not_found(self) -> None:
        service, _ = _services(_make_page_repo(), _make_folder_repo())
        with pytest.raises(NotFoundError) as excinfo:
            await service.update_page(page=_sample_page(slug="summary/ghost"))
        assert excinfo.value.code == "wiki.page_not_found"

    async def test_update_meta_never_bumps_version(self) -> None:
        page_repo = _make_page_repo()
        folder_repo = _make_folder_repo()
        service, _ = _services(page_repo, folder_repo)
        existing = _sample_page(knowledge_base_id="kb-1", slug="entity/acme", version=3)
        page_repo.rows[existing.id] = existing  # type: ignore[attr-defined]
        updated = await service.update_page_meta(
            page=existing.model_copy(update={"status": "archived"})
        )
        assert updated.version == 3
        page_repo.update.assert_not_called()  # type: ignore[attr-defined]

    async def test_update_auto_linked_content_keeps_version(self) -> None:
        page_repo = _make_page_repo()
        folder_repo = _make_folder_repo()
        service, _ = _services(page_repo, folder_repo)
        existing = _sample_page(
            knowledge_base_id="kb-1",
            slug="entity/acme",
            content="plain body [c001]",
            version=4,
        )
        page_repo.rows[existing.id] = existing  # type: ignore[attr-defined]
        updated = await service.update_auto_linked_content(
            page=_sample_page(
                id=existing.id,
                knowledge_base_id="kb-1",
                slug="entity/acme",
                content="plain body [[entity/other]] [c001]",
            )
        )
        assert updated.version == 4
        assert updated.content == "plain body [[entity/other]]"
        assert updated.out_links == ["entity/other"]


# ── WikiPageService — read / delete ─────────────────────────────────


class TestGetPage:
    async def test_get_by_slug_strips_chunk_citations(self) -> None:
        page_repo = _make_page_repo()
        folder_repo = _make_folder_repo()
        service, _ = _services(page_repo, folder_repo)
        existing = _sample_page(knowledge_base_id="kb-1", slug="entity/acme", content="body [c003]")
        page_repo.rows[existing.id] = existing  # type: ignore[attr-defined]
        page = await service.get_page_by_slug(knowledge_base_id="kb-1", slug="entity/acme")
        assert page.content == "body"

    async def test_get_by_slug_missing_raises(self) -> None:
        service, _ = _services(_make_page_repo(), _make_folder_repo())
        with pytest.raises(NotFoundError) as excinfo:
            await service.get_page_by_slug(knowledge_base_id="kb-1", slug="entity/ghost")
        assert excinfo.value.code == "wiki.page_not_found"


class TestListPages:
    async def test_pagination_math(self) -> None:
        page_repo = _make_page_repo()
        folder_repo = _make_folder_repo()
        service, _ = _services(page_repo, folder_repo)
        for i in range(5):
            row = _sample_page(knowledge_base_id="kb-1", slug=f"entity/p{i}", title=f"P{i}")
            page_repo.rows[row.id] = row  # type: ignore[attr-defined]

        response = await service.list_pages(
            filters=WikiPageListFilter(knowledge_base_id="kb-1", page=2, page_size=2)
        )
        assert response.total == 5
        assert response.page == 2
        assert response.page_size == 2
        assert response.total_pages == 3
        assert len(response.pages) == 2


class TestDeletePage:
    async def test_soft_deletes_and_removes_backlinks(self) -> None:
        page_repo = _make_page_repo()
        folder_repo = _make_folder_repo()
        service, _ = _services(page_repo, folder_repo)
        target = _sample_page(
            knowledge_base_id="kb-1", slug="entity/acme", in_links=["summary/one"]
        )
        victim = _sample_page(
            knowledge_base_id="kb-1",
            slug="summary/one",
            content="links [[entity/acme]]",
            out_links=["entity/acme"],
        )
        page_repo.rows[target.id] = target  # type: ignore[attr-defined]
        page_repo.rows[victim.id] = victim  # type: ignore[attr-defined]

        await service.delete_page(knowledge_base_id="kb-1", slug="summary/one")
        assert page_repo.rows[victim.id].deleted_at is not None  # type: ignore[attr-defined]
        assert page_repo.rows[target.id].in_links == []  # type: ignore[attr-defined]

    async def test_delete_missing_raises(self) -> None:
        service, _ = _services(_make_page_repo(), _make_folder_repo())
        with pytest.raises(NotFoundError):
            await service.delete_page(knowledge_base_id="kb-1", slug="entity/ghost")


# ── WikiPageService — index / graph / stats ─────────────────────────


class TestGetIndex:
    async def test_returns_existing_index(self) -> None:
        page_repo = _make_page_repo()
        service, _ = _services(page_repo, _make_folder_repo())
        index = _sample_page(
            knowledge_base_id="kb-1", slug="index", title="Index", page_type="index"
        )
        page_repo.rows[index.id] = index  # type: ignore[attr-defined]
        page = await service.get_index(knowledge_base_id="kb-1", tenant_id=1)
        assert page.slug == "index"

    async def test_creates_default_index_when_missing(self) -> None:
        page_repo = _make_page_repo()
        service, _ = _services(page_repo, _make_folder_repo())
        page = await service.get_index(knowledge_base_id="kb-1", tenant_id=7)
        assert page.slug == "index"
        assert page.page_type == "index"
        assert page.tenant_id == 7
        assert page.version == 1
        assert page.content.startswith("# Wiki Index")


class TestGetIndexView:
    async def test_builds_groups_and_cursor(self) -> None:
        page_repo = _make_page_repo()
        service, _ = _services(page_repo, _make_folder_repo())
        index = _sample_page(
            knowledge_base_id="kb-1",
            slug="index",
            title="Index",
            page_type="index",
            content="intro text",
        )
        page_repo.rows[index.id] = index  # type: ignore[attr-defined]
        for i in range(3):
            row = _sample_page(
                knowledge_base_id="kb-1", slug=f"summary/s{i}", title=f"S{i}", page_type="summary"
            )
            page_repo.rows[row.id] = row  # type: ignore[attr-defined]

        response = await service.get_index_view(
            knowledge_base_id="kb-1", tenant_id=1, page_types=["summary"], limit=2
        )
        assert response.intro == "intro text"
        assert len(response.groups) == 1
        group = response.groups[0]
        assert group.type == "summary"
        assert group.total == 3
        assert len(group.items) == 2
        assert group.next_cursor == "2"

    async def test_invalid_cursor_is_rejected(self) -> None:
        service, _ = _services(_make_page_repo(), _make_folder_repo())
        with pytest.raises(ValidationError) as excinfo:
            await service.get_index_view(knowledge_base_id="kb-1", tenant_id=1, cursor="abc")
        assert excinfo.value.code == "wiki.index_invalid_cursor"


class TestGetGraph:
    def _pages(self) -> list[WikiPage]:
        return [
            _sample_page(
                knowledge_base_id="kb-1", slug="entity/a", title="A", out_links=["entity/b"]
            ),
            _sample_page(
                knowledge_base_id="kb-1",
                slug="entity/b",
                title="B",
                in_links=["entity/a"],
                out_links=["entity/c"],
            ),
            _sample_page(
                knowledge_base_id="kb-1", slug="entity/c", title="C", in_links=["entity/b"]
            ),
            _sample_page(
                knowledge_base_id="kb-1",
                slug="concept/x",
                title="X",
                page_type="concept",
                out_links=[],
            ),
        ]

    def test_overview_top_by_link_count(self) -> None:
        data = compute_graph_subset(
            self._pages(),
            WikiGraphRequest(knowledge_base_id="kb-1", mode=WIKI_GRAPH_MODE_OVERVIEW, limit=2),
        )
        assert [n.slug for n in data.nodes] == ["entity/b", "entity/a"]
        assert data.meta.total == 4
        assert data.meta.truncated is True
        assert any(e.source == "entity/a" and e.target == "entity/b" for e in data.edges)

    def test_overview_type_filter_changes_total(self) -> None:
        data = compute_graph_subset(
            self._pages(),
            WikiGraphRequest(knowledge_base_id="kb-1", types=["entity"], limit=10),
        )
        assert data.meta.total == 3
        assert {n.slug for n in data.nodes} == {"entity/a", "entity/b", "entity/c"}

    def test_ego_bfs_neighborhood(self) -> None:
        data = compute_graph_subset(
            self._pages(),
            WikiGraphRequest(
                knowledge_base_id="kb-1", mode=WIKI_GRAPH_MODE_EGO, center="entity/a", depth=2
            ),
        )
        assert {n.slug for n in data.nodes} == {"entity/a", "entity/b", "entity/c"}
        assert data.meta.center == "entity/a"

    def test_ego_requires_center(self) -> None:
        with pytest.raises(ValidationError):
            compute_graph_subset(
                self._pages(),
                WikiGraphRequest(knowledge_base_id="kb-1", mode=WIKI_GRAPH_MODE_EGO, center=""),
            )


class TestGetStats:
    async def test_aggregates(self) -> None:
        page_repo = _make_page_repo()
        service, _ = _services(page_repo, _make_folder_repo())
        for i, (slug, page_type) in enumerate(
            [("entity/a", "entity"), ("entity/b", "entity"), ("concept/x", "concept")]
        ):
            row = _sample_page(
                knowledge_base_id="kb-1",
                slug=slug,
                page_type=page_type,
                out_links=["entity/a"] if i else [],
            )
            page_repo.rows[row.id] = row  # type: ignore[attr-defined]

        stats = await service.get_stats(knowledge_base_id="kb-1")
        assert stats.total_pages == 3
        assert stats.pages_by_type == {"entity": 2, "concept": 1}
        assert stats.total_links == 2
        assert stats.orphan_count == 3  # none have in_links


class TestRebuildLinks:
    async def test_rebuilds_in_and_out_links(self) -> None:
        page_repo = _make_page_repo()
        service, _ = _services(page_repo, _make_folder_repo())
        a = _sample_page(knowledge_base_id="kb-1", slug="entity/a", content="to [[entity/b]]")
        b = _sample_page(knowledge_base_id="kb-1", slug="entity/b", content="to [[entity/a]]")
        page_repo.rows[a.id] = a  # type: ignore[attr-defined]
        page_repo.rows[b.id] = b  # type: ignore[attr-defined]

        await service.rebuild_links(knowledge_base_id="kb-1")
        assert page_repo.rows[a.id].out_links == ["entity/b"]  # type: ignore[attr-defined]
        assert page_repo.rows[a.id].in_links == ["entity/b"]  # type: ignore[attr-defined]
        assert page_repo.rows[b.id].in_links == ["entity/a"]  # type: ignore[attr-defined]


class TestRepairContentLinks:
    async def test_rewrites_mangled_summary_slug(self) -> None:
        page_repo = _make_page_repo()
        service, _ = _services(page_repo, _make_folder_repo())
        real = _sample_page(
            knowledge_base_id="kb-1",
            slug="summary/06fb5d5b5b5e",
            title="Monthly Report",
            page_type="summary",
        )
        page_repo.rows[real.id] = real  # type: ignore[attr-defined]

        content = "read the [[summary/06fb14d5b14b14e|Monthly Report]] now"
        repaired, changed = await service.repair_content_links(
            knowledge_base_id="kb-1", self_slug="entity/me", content=content
        )
        assert changed is True
        assert "summary/06fb5d5b5b5e" in repaired

    async def test_clean_content_is_untouched(self) -> None:
        service, _ = _services(_make_page_repo(), _make_folder_repo())
        content = "no links here"
        repaired, changed = await service.repair_content_links(
            knowledge_base_id="kb-1", self_slug="entity/me", content=content
        )
        assert repaired == content
        assert changed is False


class TestInjectCrossLinks:
    async def test_injects_mentions_across_affected_pages(self) -> None:
        page_repo = _make_page_repo()
        service, _ = _services(page_repo, _make_folder_repo())
        page = _sample_page(
            knowledge_base_id="kb-1", slug="entity/acme", title="Acme", content="Acme rules."
        )
        other = _sample_page(
            knowledge_base_id="kb-1",
            slug="summary/one",
            title="One",
            content="Acme is great, Acme is big.",
        )
        page_repo.rows[page.id] = page  # type: ignore[attr-defined]
        page_repo.rows[other.id] = other  # type: ignore[attr-defined]

        updated = await service.inject_cross_links(
            knowledge_base_id="kb-1", affected_slugs=["summary/one"]
        )
        assert updated == 1
        assert page_repo.rows[other.id].content.startswith("[[entity/acme|Acme]]")  # type: ignore[attr-defined]


# ── WikiFolderService ───────────────────────────────────────────────


class TestCreateFolder:
    async def test_creates_nested_folder(self) -> None:
        page_repo = _make_page_repo()
        folder_repo = _make_folder_repo()
        _, service = _services(page_repo, folder_repo)
        folder_repo.rows["ai"] = _sample_folder(id="ai", name="AI", path="AI", depth=1)  # type: ignore[attr-defined]

        created = await service.create_folder(
            knowledge_base_id="kb-1", tenant_id=1, parent_id="ai", name="  RAG  "
        )
        assert created.name == "RAG"
        assert created.path == "AI/RAG"
        assert created.depth == 2

    async def test_rejects_blank_and_separator_names(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            validate_folder_name("   ")
        assert excinfo.value.code == "wiki.folder_name_required"
        with pytest.raises(ValidationError) as excinfo:
            validate_folder_name("AI/RAG")
        assert excinfo.value.code == "wiki.folder_name_separator"

    async def test_sibling_conflict(self) -> None:
        page_repo = _make_page_repo()
        folder_repo = _make_folder_repo()
        _, service = _services(page_repo, folder_repo)

        await service.create_folder(knowledge_base_id="kb-1", tenant_id=1, parent_id="", name="AI")
        with pytest.raises(ConflictError) as excinfo:
            await service.create_folder(
                knowledge_base_id="kb-1", tenant_id=1, parent_id="", name="AI"
            )
        assert excinfo.value.code == "wiki.folder_conflict"


class TestFindOrCreateFolderPath:
    async def test_creates_chain_and_reuses_existing(self) -> None:
        page_repo = _make_page_repo()
        folder_repo = _make_folder_repo()
        _, service = _services(page_repo, folder_repo)

        folder_id, clean = await service.find_or_create_folder_path(
            knowledge_base_id="kb-1", tenant_id=1, path=["AI", "RAG"]
        )
        assert clean == ["AI", "RAG"]
        assert folder_id != ""

        folder_id2, _ = await service.find_or_create_folder_path(
            knowledge_base_id="kb-1", tenant_id=1, path=["AI", "RAG"]
        )
        assert folder_id2 == folder_id
        assert len([r for r in folder_repo.rows.values() if r.deleted_at is None]) == 2  # type: ignore[attr-defined]

    async def test_empty_path_maps_to_root(self) -> None:
        _, service = _services(_make_page_repo(), _make_folder_repo())
        folder_id, clean = await service.find_or_create_folder_path(
            knowledge_base_id="kb-1", tenant_id=1, path=[]
        )
        assert folder_id == ""
        assert clean == []


class TestRenameOrMoveFolder:
    async def test_rename_recomputes_subtree_paths(self) -> None:
        page_repo = _make_page_repo()
        folder_repo = _make_folder_repo()
        _, service = _services(page_repo, folder_repo)
        folder_repo.rows["ai"] = _sample_folder(id="ai", name="AI", path="AI", depth=1)  # type: ignore[attr-defined]
        folder_repo.rows["rag"] = _sample_folder(  # type: ignore[attr-defined]
            id="rag", parent_id="ai", name="RAG", path="AI/RAG", depth=2
        )
        page_repo.rows["p1"] = _sample_page(  # type: ignore[attr-defined]
            id="p1",
            knowledge_base_id="kb-1",
            slug="entity/e1",
            folder_id="rag",
            category_path=["AI", "RAG"],
            depth=2,
        )

        updated = await service.rename_or_move_folder(
            knowledge_base_id="kb-1",
            id="ai",
            new_name="Intelligence",
            new_parent_id="",
            move_parent=False,
        )
        assert updated.name == "Intelligence"
        assert updated.path == "Intelligence"
        assert folder_repo.rows["rag"].path == "Intelligence/RAG"  # type: ignore[attr-defined]
        assert folder_repo.rows["rag"].depth == 2  # type: ignore[attr-defined]
        # page cache recomputed to the new path
        assert page_repo.rows["p1"].category_path == ["Intelligence", "RAG"]  # type: ignore[attr-defined]

    async def test_move_into_self_rejected(self) -> None:
        page_repo = _make_page_repo()
        folder_repo = _make_folder_repo()
        _, service = _services(page_repo, folder_repo)
        folder_repo.rows["ai"] = _sample_folder(id="ai", name="AI", path="AI", depth=1)  # type: ignore[attr-defined]
        with pytest.raises(ValidationError) as excinfo:
            await service.rename_or_move_folder(
                knowledge_base_id="kb-1", id="ai", new_name="", new_parent_id="ai", move_parent=True
            )
        assert excinfo.value.code == "wiki.folder_move_self"

    async def test_move_into_descendant_rejected(self) -> None:
        page_repo = _make_page_repo()
        folder_repo = _make_folder_repo()
        _, service = _services(page_repo, folder_repo)
        folder_repo.rows["ai"] = _sample_folder(id="ai", name="AI", path="AI", depth=1)  # type: ignore[attr-defined]
        folder_repo.rows["rag"] = _sample_folder(  # type: ignore[attr-defined]
            id="rag", parent_id="ai", name="RAG", path="AI/RAG", depth=2
        )
        with pytest.raises(ValidationError) as excinfo:
            await service.rename_or_move_folder(
                knowledge_base_id="kb-1",
                id="ai",
                new_name="",
                new_parent_id="rag",
                move_parent=True,
            )
        assert excinfo.value.code == "wiki.folder_move_descendant"


class TestDeleteAndPrune:
    async def test_delete_empty_folder(self) -> None:
        page_repo = _make_page_repo()
        folder_repo = _make_folder_repo()
        _, service = _services(page_repo, folder_repo)
        folder_repo.rows["ai"] = _sample_folder(id="ai", name="AI", path="AI", depth=1)  # type: ignore[attr-defined]
        await service.delete_folder(knowledge_base_id="kb-1", id="ai")
        assert folder_repo.rows["ai"].deleted_at is not None  # type: ignore[attr-defined]

    async def test_delete_non_empty_folder_conflicts(self) -> None:
        page_repo = _make_page_repo()
        folder_repo = _make_folder_repo()
        _, service = _services(page_repo, folder_repo)
        folder_repo.rows["ai"] = _sample_folder(id="ai", name="AI", path="AI", depth=1)  # type: ignore[attr-defined]
        page_repo.rows["p1"] = _sample_page(  # type: ignore[attr-defined]
            knowledge_base_id="kb-1", slug="entity/e1", folder_id="ai"
        )

        async def _delete_conflict(*, knowledge_base_id: str, id: str, now: datetime) -> None:
            raise ConflictError(code="wiki.folder_not_empty", message="not empty")

        folder_repo.delete.side_effect = _delete_conflict
        with pytest.raises(ConflictError) as excinfo:
            await service.delete_folder(knowledge_base_id="kb-1", id="ai")
        assert excinfo.value.code == "wiki.folder_not_empty"
        folder_repo.delete.assert_awaited_once()  # type: ignore[attr-defined]
        assert folder_repo.delete.await_args.kwargs["knowledge_base_id"] == "kb-1"  # type: ignore[attr-defined]
        assert folder_repo.delete.await_args.kwargs["id"] == "ai"  # type: ignore[attr-defined]

    async def test_prune_empty_chains_deletes_leaves_and_ancestors(self) -> None:
        page_repo = _make_page_repo()
        folder_repo = _make_folder_repo()
        _, service = _services(page_repo, folder_repo)
        folder_repo.rows["ai"] = _sample_folder(id="ai", name="AI", path="AI", depth=1)  # type: ignore[attr-defined]
        folder_repo.rows["rag"] = _sample_folder(  # type: ignore[attr-defined]
            id="rag", parent_id="ai", name="RAG", path="AI/RAG", depth=2
        )
        folder_repo.rows["keep"] = _sample_folder(id="keep", name="Keep", path="Keep", depth=1)  # type: ignore[attr-defined]

        deleted = await service.prune_empty_folder_chains(
            knowledge_base_id="kb-1", folder_ids=["rag"]
        )
        assert set(deleted) == {"rag", "ai"}
        assert folder_repo.rows["rag"].deleted_at is not None  # type: ignore[attr-defined]
        assert folder_repo.rows["ai"].deleted_at is not None  # type: ignore[attr-defined]
        assert folder_repo.rows["keep"].deleted_at is None  # type: ignore[attr-defined]


class TestListChildFolders:
    async def test_recursive_page_counts(self) -> None:
        page_repo = _make_page_repo()
        folder_repo = _make_folder_repo()
        _, service = _services(page_repo, folder_repo)
        folder_repo.rows["ai"] = _sample_folder(id="ai", name="AI", path="AI", depth=1)  # type: ignore[attr-defined]
        folder_repo.rows["rag"] = _sample_folder(  # type: ignore[attr-defined]
            id="rag", parent_id="ai", name="RAG", path="AI/RAG", depth=2
        )
        page_repo.rows["p1"] = _sample_page(  # type: ignore[attr-defined]
            knowledge_base_id="kb-1", slug="entity/e1", folder_id="rag", page_type="entity"
        )

        nodes = await service.list_child_folders(
            knowledge_base_id="kb-1", parent_id="", page_types=["entity"]
        )
        assert [n.folder.id for n in nodes] == ["ai"]
        assert nodes[0].page_count == 1  # recursive through RAG
        assert nodes[0].has_children is True

    async def test_empty_folder_shown_only_in_merged_view(self) -> None:
        page_repo = _make_page_repo()
        folder_repo = _make_folder_repo()
        _, service = _services(page_repo, folder_repo)
        folder_repo.rows["empty"] = _sample_folder(id="empty", name="Empty", path="Empty", depth=1)  # type: ignore[attr-defined]

        single = await service.list_child_folders(
            knowledge_base_id="kb-1", parent_id="", page_types=["summary"]
        )
        assert single == []
        merged = await service.list_child_folders(
            knowledge_base_id="kb-1", parent_id="", page_types=["summary", "entity"]
        )
        assert [n.folder.id for n in merged] == ["empty"]


# ── Integration (real Postgres, revision 0021+) ─────────────────────


@pytest.fixture(scope="session")
def _db_engine() -> DatabaseEngine:
    """Session-scoped engine against the configured Postgres (NullPool)."""
    reset_settings_cache()
    settings = get_settings()
    return DatabaseEngine(url=settings.database_url, poolclass=NullPool)


@pytest_asyncio.fixture
async def db_session(_db_engine: DatabaseEngine) -> AsyncIterator[AsyncSession]:
    """Per-test session; skips the test when Postgres is unreachable."""
    factory = async_sessionmaker(_db_engine.engine, expire_on_commit=False)
    async with factory() as session:
        try:
            await session.execute(text("select 1"))
        except Exception:
            pytest.skip("integration Postgres is not reachable; set DATABASE_URL_OVERRIDE")
        yield session
        await session.rollback()


async def test_integration_page_crud_round_trip(db_session: AsyncSession) -> None:
    tenant_id = make_test_tenant_id()
    kb_id = _kb()
    page_repo = WikiPageRepository(db_session)
    folder_repo = WikiFolderRepository(db_session)
    service = WikiPageService(page_repo=page_repo, folder_repo=folder_repo)

    created = await service.create_page(
        page=_sample_page(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            slug="entity/acme",
            title="Acme",
            content="Acme links [[entity/partner]].",
        )
    )
    assert created.version == 1
    assert created.out_links == ["entity/partner"]

    fetched = await service.get_page_by_slug(knowledge_base_id=kb_id, slug="entity/acme")
    assert fetched.id == created.id

    updated = await service.update_page(
        page=_sample_page(
            id=created.id,
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            slug="entity/acme",
            title="Acme v2",
            content="Acme links [[entity/partner]] now.",
            status="published",
            version=1,
        ),
        edit_source="user",
        editor_id="usr-int",
    )
    assert updated.version == 2
    assert updated.title == "Acme v2"

    # bookkeeping-only write must NOT bump the version again
    moved = await service.update_page_meta(
        page=updated.model_copy(update={"page_metadata": {"touched": True}})
    )
    assert moved.version == 2


async def test_integration_folder_and_move_page(db_session: AsyncSession) -> None:
    tenant_id = make_test_tenant_id()
    kb_id = _kb()
    page_repo = WikiPageRepository(db_session)
    folder_repo = WikiFolderRepository(db_session)
    page_service = WikiPageService(page_repo=page_repo, folder_repo=folder_repo)
    folder_service = WikiFolderService(folder_repo=folder_repo, page_repo=page_repo)

    folder_id, clean = await folder_service.find_or_create_folder_path(
        knowledge_base_id=kb_id, tenant_id=tenant_id, path=["AI", "RAG"]
    )
    assert clean == ["AI", "RAG"]
    assert folder_id

    await page_service.create_page(
        page=_sample_page(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            slug="concept/rag",
            page_type="concept",
            title="RAG",
        )
    )
    moved = await page_service.move_page(
        knowledge_base_id=kb_id, slug="concept/rag", folder_id=folder_id
    )
    assert moved.category_path == ["AI", "RAG"]
    assert moved.depth == 2
    assert moved.wiki_path == "concept/AI/RAG/RAG"

    folder = await folder_service.get_folder(knowledge_base_id=kb_id, id=folder_id)
    assert folder.path == "AI/RAG"
