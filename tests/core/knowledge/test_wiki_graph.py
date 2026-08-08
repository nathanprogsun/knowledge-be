"""Unit + integration tests for the wiki analysis modules.

Unit tests drive the graph / stats / search / links / issues modules with
stateful mock repositories (pytest, AAA) covering validation, error
classification, deterministic ordering, and the injectable seams.

Integration tests run against the real applied schema (revision 0022+);
isolation is by per-test generated tenant ids and unique entity ids, and
they are skipped when Postgres is not reachable (set
``DATABASE_URL_OVERRIDE``). They seed wiki pages only — no chunk rows, so
no 32-bit tenant-id constraint applies.
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

from src.common.exception import NotFoundError, ValidationError
from src.core.knowledge.wiki import graph as graph_module
from src.core.knowledge.wiki import links as links_module
from src.core.knowledge.wiki import stats as stats_module
from src.core.knowledge.wiki.issues import (
    WIKI_ISSUE_STATUS_IGNORED,
    WIKI_ISSUE_STATUS_PENDING,
    WIKI_ISSUE_STATUS_RESOLVED,
    WikiPageIssue,
    create_issue,
    is_valid_issue_status,
    list_issues,
    pending_issue_count,
    update_issue_status,
)
from src.core.knowledge.wiki.search import search_pages
from src.core.knowledge.wiki.stats import get_stats
from src.core.knowledge.wiki.types import (
    WIKI_GRAPH_MODE_EGO,
    WIKI_GRAPH_MODE_OVERVIEW,
    WikiGraphRequest,
)
from src.db.base import DatabaseEngine
from src.db.dao.wiki_page_repository import WikiPageRepository
from src.db.models.wiki_page import WikiPage
from src.settings import get_settings, reset_settings_cache
from tests.integration.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 1, tzinfo=UTC)

# ── Sample row builders ─────────────────────────────────────────────


def _page_id() -> str:
    return f"page-{uuid.uuid4().hex[:12]}"


def _kb() -> str:
    return f"kb-{uuid.uuid4().hex[:8]}"


def _page(
    *,
    tenant_id: int = 1,
    knowledge_base_id: str = "kb-1",
    slug: str = "entity/acme",
    id: str | None = None,
    **overrides: object,
) -> WikiPage:
    values: dict[str, object] = {
        "id": id or _page_id(),
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


def _issue(
    *,
    id: str = "",
    tenant_id: int = 7,
    knowledge_base_id: str = "kb-1",
    slug: str = "entity/acme",
    **overrides: object,
) -> WikiPageIssue:
    values: dict[str, object] = {
        "id": id,
        "tenant_id": tenant_id,
        "knowledge_base_id": knowledge_base_id,
        "slug": slug,
        "issue_type": "dead_link",
        "description": "references a missing page",
        "suspected_knowledge_ids": [],
        "status": WIKI_ISSUE_STATUS_PENDING,
        "reported_by": "linter",
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    values.update(overrides)
    return WikiPageIssue.model_validate(values)


# ── Repository mocks (stateful via side_effect closures) ────────────


def _make_page_repo() -> AsyncMock:
    """``AsyncMock(spec=WikiPageRepository)`` with closure-captured state.

    Supports exactly the methods the analysis modules reach for:
    ``list_all``, ``list_pages``, ``count_by_type``, ``count_orphans``,
    ``search``, ``exists_slugs``, and ``update_meta``.
    """
    repo = AsyncMock(spec=WikiPageRepository)
    rows: dict[str, WikiPage] = {}
    meta_calls: list[WikiPage] = []
    repo.rows = rows
    repo.meta_calls = meta_calls

    async def _list_all(*, knowledge_base_id: str) -> list[WikiPage]:
        return [
            row
            for row in rows.values()
            if row.knowledge_base_id == knowledge_base_id and row.deleted_at is None
        ]

    async def _list_pages(
        *,
        knowledge_base_id: str,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "",
        sort_order: str = "desc",
        **_extra: object,
    ) -> tuple[list[WikiPage], int]:
        pages = [
            row
            for row in rows.values()
            if row.knowledge_base_id == knowledge_base_id and row.deleted_at is None
        ]
        page = max(1, page)
        page_size = max(1, page_size)
        start = (page - 1) * page_size
        return pages[start : start + page_size], len(pages)

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

    async def _search(*, knowledge_base_id: str, query: str, limit: int = 10) -> list[WikiPage]:
        return [
            row
            for row in rows.values()
            if row.knowledge_base_id == knowledge_base_id
            and row.deleted_at is None
            and query in row.title
        ][:limit]

    async def _exists_slugs(*, knowledge_base_id: str, slugs: list[str]) -> dict[str, bool]:
        live = {
            row.slug
            for row in rows.values()
            if row.knowledge_base_id == knowledge_base_id
            and row.deleted_at is None
            and row.status != "archived"
        }
        return {slug: slug in live for slug in slugs}

    async def _update_meta(*, row: WikiPage, now: datetime) -> WikiPage:
        existing = rows.get(row.id)
        if existing is None or existing.deleted_at is not None:
            raise NotFoundError(code="wiki.page_not_found", message="missing")
        stored = row.model_copy(update={"updated_at": now})
        rows[row.id] = stored
        meta_calls.append(stored)
        return stored

    repo.list_all.side_effect = _list_all
    repo.list_pages.side_effect = _list_pages
    repo.count_by_type.side_effect = _count_by_type
    repo.count_orphans.side_effect = _count_orphans
    repo.search.side_effect = _search
    repo.exists_slugs.side_effect = _exists_slugs
    repo.update_meta.side_effect = _update_meta
    return repo


class _FakeIssueRepo:
    """In-memory issue repository implementing the protocol seam."""

    def __init__(self) -> None:
        self.rows: dict[str, WikiPageIssue] = {}

    async def create(self, issue: WikiPageIssue) -> WikiPageIssue:
        self.rows[issue.id] = issue
        return issue

    async def list(
        self, *, knowledge_base_id: str, slug: str = "", status: str = ""
    ) -> list[WikiPageIssue]:
        out = [
            issue
            for issue in self.rows.values()
            if issue.knowledge_base_id == knowledge_base_id
            and (not slug or issue.slug == slug)
            and (not status or issue.status == status)
        ]
        return sorted(out, key=lambda item: item.created_at, reverse=True)

    async def get_by_id_or_none(self, *, issue_id: str) -> WikiPageIssue | None:
        return self.rows.get(issue_id)

    async def update_status(self, *, issue_id: str, status: str) -> None:
        issue = self.rows[issue_id]
        self.rows[issue_id] = issue.model_copy(update={"status": status})


# ── Graph ───────────────────────────────────────────────────────────


class TestGraph:
    def _graph_pages(self) -> list[WikiPage]:
        return [
            _page(
                knowledge_base_id="kb-1",
                slug="entity/a",
                title="A",
                out_links=["entity/b"],
            ),
            _page(
                knowledge_base_id="kb-1",
                slug="entity/b",
                title="B",
                in_links=["entity/a"],
                out_links=["entity/c"],
            ),
            _page(
                knowledge_base_id="kb-1",
                slug="entity/c",
                title="C",
                in_links=["entity/b"],
            ),
            _page(
                knowledge_base_id="kb-1",
                slug="concept/x",
                title="X",
                page_type="concept",
            ),
        ]

    def _seeded_repo(self, pages: list[WikiPage]) -> AsyncMock:
        repo = _make_page_repo()
        for page in pages:
            repo.rows[page.id] = page
        return repo

    async def test_overview_returns_top_by_link_count_with_edges(self) -> None:
        repo = self._seeded_repo(self._graph_pages())
        data = await graph_module.get_graph(
            page_repo=repo,
            request=WikiGraphRequest(knowledge_base_id="kb-1", limit=2),
        )
        assert [node.slug for node in data.nodes] == ["entity/b", "entity/a"]
        assert data.edges == [graph_module.WikiGraphEdge(source="entity/a", target="entity/b")]
        assert data.meta.total == 4
        assert data.meta.returned == 2
        assert data.meta.truncated is True
        assert data.meta.mode == WIKI_GRAPH_MODE_OVERVIEW

    async def test_overview_type_filter_narrows_total(self) -> None:
        repo = self._seeded_repo(self._graph_pages())
        data = await graph_module.get_graph(
            page_repo=repo,
            request=WikiGraphRequest(knowledge_base_id="kb-1", types=["entity"], limit=10),
        )
        assert data.meta.total == 3
        assert {node.slug for node in data.nodes} == {"entity/a", "entity/b", "entity/c"}

    async def test_overview_zero_limit_is_uncapped(self) -> None:
        repo = self._seeded_repo(self._graph_pages())
        data = await graph_module.get_graph(
            page_repo=repo,
            request=WikiGraphRequest(knowledge_base_id="kb-1", limit=0),
        )
        assert data.meta.returned == 4
        assert data.meta.truncated is False

    async def test_ego_bfs_neighborhood_and_meta(self) -> None:
        repo = self._seeded_repo(self._graph_pages())
        data = await graph_module.get_graph(
            page_repo=repo,
            request=WikiGraphRequest(
                knowledge_base_id="kb-1",
                mode=WIKI_GRAPH_MODE_EGO,
                center="entity/a",
                depth=2,
            ),
        )
        assert {node.slug for node in data.nodes} == {"entity/a", "entity/b", "entity/c"}
        assert data.meta.mode == WIKI_GRAPH_MODE_EGO
        assert data.meta.center == "entity/a"
        assert data.meta.depth == 2

    async def test_ego_type_filter_excludes_and_does_not_traverse(self) -> None:
        repo = self._seeded_repo(self._graph_pages())
        # The concept page is not an entity; asking for an entity center is
        # fine, but a concept center fails the allow-list and yields nothing.
        data = await graph_module.get_graph(
            page_repo=repo,
            request=WikiGraphRequest(
                knowledge_base_id="kb-1",
                mode=WIKI_GRAPH_MODE_EGO,
                center="concept/x",
                depth=2,
                types=["entity"],
            ),
        )
        assert data.meta.returned == 0

    async def test_missing_kb_is_rejected(self) -> None:
        repo = self._seeded_repo([])
        with pytest.raises(ValidationError) as excinfo:
            await graph_module.get_graph(
                page_repo=repo,
                request=WikiGraphRequest(knowledge_base_id="   "),
            )
        assert excinfo.value.code == "wiki.graph_kb_required"

    async def test_ego_requires_center(self) -> None:
        repo = self._seeded_repo(self._graph_pages())
        with pytest.raises(ValidationError) as excinfo:
            await graph_module.get_graph(
                page_repo=repo,
                request=WikiGraphRequest(
                    knowledge_base_id="kb-1", mode=WIKI_GRAPH_MODE_EGO, center=""
                ),
            )
        assert excinfo.value.code == "wiki.graph_center_required"

    async def test_ego_unknown_center(self) -> None:
        repo = self._seeded_repo(self._graph_pages())
        with pytest.raises(ValidationError) as excinfo:
            await graph_module.get_graph(
                page_repo=repo,
                request=WikiGraphRequest(
                    knowledge_base_id="kb-1",
                    mode=WIKI_GRAPH_MODE_EGO,
                    center="entity/ghost",
                    depth=1,
                ),
            )
        assert excinfo.value.code == "wiki.graph_center_not_found"


# ── Stats ───────────────────────────────────────────────────────────


class TestStats:
    def _seeded_repo(self) -> AsyncMock:
        repo = _make_page_repo()
        rows = [
            _page(
                knowledge_base_id="kb-1",
                slug="entity/a",
                page_type="entity",
                out_links=["entity/b"],
            ),
            _page(knowledge_base_id="kb-1", slug="entity/b", page_type="entity"),
            _page(knowledge_base_id="kb-1", slug="concept/x", page_type="concept"),
        ]
        for row in rows:
            repo.rows[row.id] = row
        return repo

    async def test_aggregates_with_neutral_seams(self) -> None:
        repo = self._seeded_repo()
        stats = await stats_module.get_stats(page_repo=repo, knowledge_base_id="kb-1")
        assert stats.total_pages == 3
        assert stats.pages_by_type == {"entity": 2, "concept": 1}
        assert stats.total_links == 1
        assert stats.orphan_count == 3
        assert stats.pending_tasks == 0
        assert stats.pending_issues == 0
        assert stats.is_active is False
        assert len(stats.recent_updates) == 3

    async def test_injectable_seams_are_used(self) -> None:
        repo = self._seeded_repo()

        async def _pending_tasks() -> int:
            return 5

        async def _pending_issues() -> int:
            return 3

        async def _active() -> bool:
            return True

        stats = await get_stats(
            page_repo=repo,
            knowledge_base_id="kb-1",
            pending_task_count=_pending_tasks,
            pending_issue_count=_pending_issues,
            is_active=_active,
        )
        assert stats.pending_tasks == 5
        assert stats.pending_issues == 3
        assert stats.is_active is True

    async def test_recent_updates_strip_chunk_citations(self) -> None:
        repo = _make_page_repo()
        row = _page(
            knowledge_base_id="kb-1",
            slug="entity/a",
            content="body [c001]",
            summary="sum [c002]",
        )
        repo.rows[row.id] = row
        stats = await get_stats(page_repo=repo, knowledge_base_id="kb-1")
        assert stats.recent_updates[0].content == "body"
        assert stats.recent_updates[0].summary == "sum"


# ── Search ──────────────────────────────────────────────────────────


class TestSearch:
    async def test_returns_hits_with_citations_stripped(self) -> None:
        repo = _make_page_repo()
        row = _page(
            knowledge_base_id="kb-1",
            slug="entity/acme",
            title="Acme Corp",
            content="We sell widgets [c001].",
        )
        repo.rows[row.id] = row
        results = await search_pages(page_repo=repo, knowledge_base_id="kb-1", query="Acme")
        assert [page.slug for page in results] == ["entity/acme"]
        assert results[0].content == "We sell widgets."

    async def test_empty_query_is_rejected(self) -> None:
        repo = _make_page_repo()
        with pytest.raises(ValidationError) as excinfo:
            await search_pages(page_repo=repo, knowledge_base_id="kb-1", query="  ")
        assert excinfo.value.code == "wiki.search_query_required"

    async def test_limit_is_clamped(self) -> None:
        repo = _make_page_repo()
        await search_pages(page_repo=repo, knowledge_base_id="kb-1", query="x", limit=0)
        assert repo.search.await_args.kwargs["limit"] == 10
        await search_pages(page_repo=repo, knowledge_base_id="kb-1", query="x", limit=200)
        assert repo.search.await_args.kwargs["limit"] == 50


# ── Links ───────────────────────────────────────────────────────────


class TestLinks:
    def _seeded_repo(self) -> AsyncMock:
        repo = _make_page_repo()
        rows = [
            _page(
                knowledge_base_id="kb-1",
                slug="entity/a",
                content="to [[entity/b]]",
                out_links=["entity/b"],
            ),
            _page(
                knowledge_base_id="kb-1",
                slug="entity/b",
                content="to [[entity/a]]",
                out_links=["entity/a"],
            ),
            _page(
                knowledge_base_id="kb-1",
                slug="entity/c",
                content="to [[entity/ghost]]",
                out_links=["entity/ghost"],
            ),
        ]
        for row in rows:
            repo.rows[row.id] = row
        return repo

    async def test_rebuild_links_persists_bidirectional_refs(self) -> None:
        repo = self._seeded_repo()
        await links_module.rebuild_links(page_repo=repo, knowledge_base_id="kb-1")

        by_slug = {row.slug: row for row in repo.rows.values()}
        a = by_slug["entity/a"]
        b = by_slug["entity/b"]
        c = by_slug["entity/c"]
        assert a.out_links == ["entity/b"]
        assert a.in_links == ["entity/b"]
        assert b.out_links == ["entity/a"]
        assert b.in_links == ["entity/a"]
        # dead target never became an in-link and version is untouched
        assert c.out_links == ["entity/ghost"]
        assert c.in_links == []
        assert c.version == 1
        assert len(repo.meta_calls) == 3

    async def test_count_total_links(self) -> None:
        repo = self._seeded_repo()
        total = await links_module.count_total_links(page_repo=repo, knowledge_base_id="kb-1")
        assert total == 3

    async def test_count_orphans_delegates(self) -> None:
        repo = self._seeded_repo()
        orphans = await links_module.count_orphans(page_repo=repo, knowledge_base_id="kb-1")
        assert orphans == 3  # all three pages have empty in_links

    async def test_broken_link_report_lists_dead_targets(self) -> None:
        repo = self._seeded_repo()
        report = await links_module.broken_link_report(page_repo=repo, knowledge_base_id="kb-1")
        assert report == [links_module.BrokenLink(source="entity/c", target="entity/ghost")]


# ── Issues ──────────────────────────────────────────────────────────


class TestIssues:
    async def test_create_applies_defaults_and_persists(self) -> None:
        repo = _FakeIssueRepo()
        created = await create_issue(issue_repo=repo, issue=_issue(id=""), now=_NOW)
        assert created.id != ""
        assert created.status == WIKI_ISSUE_STATUS_PENDING
        assert created.created_at == _NOW
        assert repo.rows[created.id].status == WIKI_ISSUE_STATUS_PENDING

    async def test_create_validation(self) -> None:
        repo = _FakeIssueRepo()
        with pytest.raises(ValidationError) as excinfo:
            await create_issue(issue_repo=repo, issue=_issue(slug="  "))
        assert excinfo.value.code == "wiki.issue_slug_required"
        with pytest.raises(ValidationError) as excinfo:
            await create_issue(issue_repo=repo, issue=_issue(tenant_id=0))
        assert excinfo.value.code == "wiki.issue_tenant_required"
        with pytest.raises(ValidationError) as excinfo:
            await create_issue(issue_repo=repo, issue=_issue(status="bogus"))
        assert excinfo.value.code == "wiki.issue_invalid_status"

    async def test_list_filters_and_validates_status(self) -> None:
        repo = _FakeIssueRepo()
        await create_issue(
            issue_repo=repo,
            issue=_issue(id="i-1", slug="entity/a", status=WIKI_ISSUE_STATUS_PENDING),
        )
        await create_issue(
            issue_repo=repo,
            issue=_issue(id="i-2", slug="entity/b", status=WIKI_ISSUE_STATUS_RESOLVED),
        )

        all_issues = await list_issues(issue_repo=repo, knowledge_base_id="kb-1")
        assert {item.id for item in all_issues} == {"i-1", "i-2"}

        pending = await list_issues(
            issue_repo=repo,
            knowledge_base_id="kb-1",
            status=WIKI_ISSUE_STATUS_PENDING,
        )
        assert [item.id for item in pending] == ["i-1"]

        scoped = await list_issues(issue_repo=repo, knowledge_base_id="kb-1", slug="entity/b")
        assert [item.id for item in scoped] == ["i-2"]

        with pytest.raises(ValidationError) as excinfo:
            await list_issues(issue_repo=repo, knowledge_base_id="kb-1", status="bogus")
        assert excinfo.value.code == "wiki.issue_invalid_status"
        with pytest.raises(ValidationError) as excinfo:
            await list_issues(issue_repo=repo, knowledge_base_id="  ")
        assert excinfo.value.code == "wiki.issue_kb_required"

    async def test_update_status_returns_updated_record(self) -> None:
        repo = _FakeIssueRepo()
        await create_issue(issue_repo=repo, issue=_issue(id="i-1"))
        updated = await update_issue_status(
            issue_repo=repo,
            issue_id="i-1",
            status=WIKI_ISSUE_STATUS_IGNORED,
        )
        assert updated.status == WIKI_ISSUE_STATUS_IGNORED
        assert repo.rows["i-1"].status == WIKI_ISSUE_STATUS_IGNORED

    async def test_update_status_validation_and_missing(self) -> None:
        repo = _FakeIssueRepo()
        with pytest.raises(ValidationError) as excinfo:
            await update_issue_status(
                issue_repo=repo, issue_id="  ", status=WIKI_ISSUE_STATUS_PENDING
            )
        assert excinfo.value.code == "wiki.issue_id_required"
        with pytest.raises(ValidationError) as bad_status:
            await update_issue_status(issue_repo=repo, issue_id="i-1", status="bogus")
        assert bad_status.value.code == "wiki.issue_invalid_status"
        with pytest.raises(NotFoundError) as missing:
            await update_issue_status(
                issue_repo=repo,
                issue_id="i-missing",
                status=WIKI_ISSUE_STATUS_RESOLVED,
            )
        assert missing.value.code == "wiki.issue_not_found"

    async def test_pending_issue_count(self) -> None:
        repo = _FakeIssueRepo()
        await create_issue(
            issue_repo=repo, issue=_issue(id="i-1", status=WIKI_ISSUE_STATUS_PENDING)
        )
        await create_issue(
            issue_repo=repo, issue=_issue(id="i-2", status=WIKI_ISSUE_STATUS_RESOLVED)
        )
        assert await pending_issue_count(issue_repo=repo, knowledge_base_id="kb-1") == 1

    async def test_is_valid_issue_status(self) -> None:
        assert is_valid_issue_status(WIKI_ISSUE_STATUS_PENDING) is True
        assert is_valid_issue_status(WIKI_ISSUE_STATUS_IGNORED) is True
        assert is_valid_issue_status(WIKI_ISSUE_STATUS_RESOLVED) is True
        assert is_valid_issue_status("bogus") is False


# ── Integration (real Postgres, revision 0022+) ─────────────────────


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


async def test_integration_graph_overview_and_ego(db_session: AsyncSession) -> None:
    tenant_id = make_test_tenant_id()
    kb_id = _kb()
    page_repo = WikiPageRepository(db_session)
    for row in [
        _page(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            slug="entity/a",
            title="A",
            out_links=["entity/b"],
        ),
        _page(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            slug="entity/b",
            title="B",
            in_links=["entity/a"],
            out_links=["entity/c"],
        ),
        _page(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            slug="entity/c",
            title="C",
            in_links=["entity/b"],
        ),
        _page(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            slug="concept/x",
            title="X",
            page_type="concept",
        ),
    ]:
        await page_repo.create(row)

    overview = await graph_module.get_graph(
        page_repo=page_repo,
        request=WikiGraphRequest(knowledge_base_id=kb_id, limit=2),
    )
    assert [node.slug for node in overview.nodes] == ["entity/b", "entity/a"]
    assert overview.meta.total == 4
    assert any(edge.source == "entity/a" and edge.target == "entity/b" for edge in overview.edges)

    ego = await graph_module.get_graph(
        page_repo=page_repo,
        request=WikiGraphRequest(
            knowledge_base_id=kb_id,
            mode=WIKI_GRAPH_MODE_EGO,
            center="entity/a",
            depth=2,
        ),
    )
    assert {node.slug for node in ego.nodes} == {"entity/a", "entity/b", "entity/c"}


async def test_integration_stats_search_links(db_session: AsyncSession) -> None:
    tenant_id = make_test_tenant_id()
    kb_id = _kb()
    page_repo = WikiPageRepository(db_session)
    for row in [
        _page(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            slug="entity/acme",
            title="Acme Corp",
            content="Acme acquisition details [c001].",
            out_links=["entity/partner"],
        ),
        _page(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            slug="entity/partner",
            title="Partner",
            content="partners with [[entity/acme]]",
            in_links=["entity/acme"],
        ),
        _page(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            slug="entity/widgets",
            title="Widgets",
            content="Acme buys widgets [c002].",
            page_type="entity",
        ),
    ]:
        await page_repo.create(row)

    stats = await stats_module.get_stats(page_repo=page_repo, knowledge_base_id=kb_id)
    assert stats.total_pages == 3
    assert stats.pages_by_type == {"entity": 3}
    assert stats.total_links == 1  # only acme carries a stored out-link
    assert stats.orphan_count == 2  # acme + widgets have no in-links; index excluded

    results = await search_pages(page_repo=page_repo, knowledge_base_id=kb_id, query="Widgets")
    # a title hit outranks every body mention; citations stripped
    assert [page.slug for page in results] == ["entity/widgets"]
    assert results[0].content == "Acme buys widgets."

    await links_module.rebuild_links(page_repo=page_repo, knowledge_base_id=kb_id)
    acme = await page_repo.get_by_slug_or_none(knowledge_base_id=kb_id, slug="entity/acme")
    assert acme is not None
    # out-links are re-derived from the body (the stored array is replaced)
    assert acme.out_links == []
    # the inbound reference from partner's body survived the rebuild
    assert acme.in_links == ["entity/partner"]
    assert acme.version == 1  # link rebuild never bumps version

    report = await links_module.broken_link_report(page_repo=page_repo, knowledge_base_id=kb_id)
    assert report == []
