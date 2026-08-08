"""Unit + integration tests for the wiki lint service.

Unit tests drive the lint service with ``AsyncMock`` services (pytest,
AAA) covering validation, every issue type, the health score, the
summary, and the auto-fix pass. The pure helpers
(``compute_health_score`` / ``build_summary`` / ``remove_source_ref``)
are exercised directly.

Integration tests run against the real applied schema; isolation is by
per-test generated tenant ids and unique entity ids, and they are
skipped when Postgres is not reachable (set ``DATABASE_URL_OVERRIDE``).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from random import randint
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from faker import Faker
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool

from src.common.exception import NotFoundError, ValidationError
from src.common.json import JsonObject
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.core.knowledge.wiki.lint_service import (
    LINT_ISSUE_BROKEN_LINK,
    LINT_ISSUE_EMPTY_CONTENT,
    LINT_ISSUE_MISSING_CROSS_REF,
    LINT_ISSUE_ORPHAN_PAGE,
    LINT_ISSUE_STALE_REF,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    KnowledgeResolver,
    WikiLintIssue,
    WikiLintReport,
    WikiLintService,
    build_summary,
    compute_health_score,
    remove_source_ref,
)
from src.core.knowledge.wiki.page_service import WikiPageService
from src.core.knowledge.wiki.types import (
    WikiStats,
)
from src.db.base import DatabaseEngine
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.dao.wiki_page_repository import WikiFolderRepository, WikiPageRepository
from src.db.models.knowledge import Document
from src.db.models.wiki_page import WikiPage
from src.settings import get_settings, reset_settings_cache
from tests.integration.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 1, tzinfo=UTC)

_LONG_BODY = (
    "This page has enough words to comfortably clear the minimum content "
    "threshold that the lint pass uses to flag near-empty pages."
)


@pytest.fixture(autouse=True)
def _reseed_faker() -> None:
    """Re-seed Faker per test for varied-but-reproducible generation."""
    Faker.seed(randint(1, 100_000_000))


# ── Sample row builders ─────────────────────────────────────────────


def _pid() -> str:
    return f"page-{uuid.uuid4().hex[:12]}"


def _kb_id() -> str:
    return f"kb-{uuid.uuid4().hex[:8]}"


def _sample_page(
    *,
    knowledge_base_id: str = "kb-1",
    slug: str = "entity/acme",
    id: str | None = None,
    **overrides: object,
) -> WikiPage:
    values: dict[str, object] = {
        "id": id or _pid(),
        "tenant_id": 1,
        "knowledge_base_id": knowledge_base_id,
        "slug": slug,
        "title": "Acme",
        "page_type": "entity",
        "status": "published",
        "content": _LONG_BODY,
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


def _wiki_kb(*, kb_id: str = "kb-1", enabled: bool = True) -> KnowledgeBaseInfo:
    """A wiki-enabled (or not) ``KnowledgeBaseInfo`` for the KB mock."""
    strategy: JsonObject = (
        {
            "vector_enabled": False,
            "keyword_enabled": False,
            "wiki_enabled": True,
            "graph_enabled": False,
        }
        if enabled
        else {
            "vector_enabled": True,
            "keyword_enabled": True,
            "wiki_enabled": False,
            "graph_enabled": False,
        }
    )
    return KnowledgeBaseInfo(
        id=kb_id,
        name="Test Wiki",
        tenant_id=1,
        indexing_strategy=strategy,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _stats(
    *,
    total_pages: int = 0,
    total_links: int = 0,
    orphan_count: int = 0,
    pages_by_type: dict[str, int] | None = None,
) -> WikiStats:
    return WikiStats(
        total_pages=total_pages,
        pages_by_type=pages_by_type or {},
        total_links=total_links,
        orphan_count=orphan_count,
        recent_updates=[],
    )


# ── Service mocks (stateful via side_effect closures) ───────────────


def _make_wiki_service(pages: list[WikiPage]) -> AsyncMock:
    """``AsyncMock(spec=WikiPageService)`` with closure-captured page state.

    ``list_pages_cursor`` treats ``cursor`` as an integer offset into the
    live page list so multi-batch streaming is exercised naturally.
    """
    svc = AsyncMock(spec=WikiPageService)
    store: dict[str, WikiPage] = {p.id: p for p in pages}
    svc.rows = store

    def _live() -> list[WikiPage]:
        return [p for p in store.values() if p.deleted_at is None]

    async def _list_pages_cursor(
        *, knowledge_base_id: str, cursor: str = "", limit: int = 100
    ) -> tuple[list[WikiPage], str]:
        del knowledge_base_id
        live = _live()
        idx = int(cursor) if cursor else 0
        window = live[idx : idx + limit]
        next_cursor = str(idx + limit) if len(window) == limit and idx + limit < len(live) else ""
        return window, next_cursor

    async def _list_all_slugs(*, knowledge_base_id: str) -> list[str]:
        del knowledge_base_id
        return [p.slug for p in _live() if p.status != "archived"]

    async def _get_stats(*, knowledge_base_id: str) -> WikiStats:
        del knowledge_base_id
        live = _live()
        counts: dict[str, int] = {}
        for p in live:
            counts[p.page_type] = counts.get(p.page_type, 0) + 1
        return _stats(
            total_pages=len(live),
            total_links=sum(len(p.out_links) for p in live),
            orphan_count=sum(1 for p in live if not p.in_links and p.page_type != "index"),
            pages_by_type=counts,
        )

    async def _get_page_by_slug(*, knowledge_base_id: str, slug: str) -> WikiPage:
        del knowledge_base_id
        for p in _live():
            if p.slug == slug:
                return p
        raise NotFoundError(code="wiki.page_not_found", message=f"page {slug} not found")

    async def _update_auto_linked_content(*, page: WikiPage) -> WikiPage:
        existing = store.get(page.id)
        if existing is None:
            raise NotFoundError(code="wiki.page_not_found", message="missing")
        updated = existing.model_copy(update={"content": page.content})
        store[page.id] = updated
        return updated

    async def _update_page(
        *, page: WikiPage, edit_source: str = "", editor_id: str = ""
    ) -> WikiPage:
        existing = store.get(page.id)
        if existing is None:
            raise NotFoundError(code="wiki.page_not_found", message="missing")
        updated = existing.model_copy(
            update={"status": page.status, "version": existing.version + 1}
        )
        store[page.id] = updated
        return updated

    async def _update_page_meta(*, page: WikiPage) -> WikiPage:
        existing = store.get(page.id)
        if existing is None:
            raise NotFoundError(code="wiki.page_not_found", message="missing")
        updated = existing.model_copy(update={"source_refs": list(page.source_refs)})
        store[page.id] = updated
        return updated

    async def _delete_page(*, knowledge_base_id: str, slug: str) -> None:
        del knowledge_base_id
        for p in list(store.values()):
            if p.slug == slug:
                store[p.id] = p.model_copy(update={"deleted_at": _NOW})
                return
        raise NotFoundError(code="wiki.page_not_found", message=f"page {slug} not found")

    svc.list_pages_cursor.side_effect = _list_pages_cursor
    svc.list_all_slugs.side_effect = _list_all_slugs
    svc.get_stats.side_effect = _get_stats
    svc.get_page_by_slug.side_effect = _get_page_by_slug
    svc.update_auto_linked_content.side_effect = _update_auto_linked_content
    svc.update_page.side_effect = _update_page
    svc.update_page_meta.side_effect = _update_page_meta
    svc.delete_page.side_effect = _delete_page
    return svc


def _make_kb_service(*, kb: KnowledgeBaseInfo | None = None) -> AsyncMock:
    """``AsyncMock(spec=KBService)`` returning ``kb`` for the wiki KB."""
    svc = AsyncMock(spec=KBService)
    svc.get_knowledge_base_by_id_only.return_value = kb or _wiki_kb()
    return svc


def _lint_service(
    pages: list[WikiPage],
    *,
    enabled: bool = True,
    resolver: KnowledgeResolver | None = None,
) -> tuple[WikiLintService, AsyncMock, AsyncMock]:
    wiki = _make_wiki_service(pages)
    kb = _make_kb_service(kb=_wiki_kb(enabled=enabled))
    service = WikiLintService(wiki_service=wiki, kb_service=kb, knowledge_resolver=resolver)
    return service, wiki, kb


# ── Pure helpers ────────────────────────────────────────────────────


class TestRemoveSourceRef:
    def test_removes_bare_and_prefixed_forms(self) -> None:
        refs = ["k-1", "k-2|Some Title", "k-3", "k-2"]
        assert remove_source_ref(refs, "k-2") == ["k-1", "k-3"]

    def test_input_not_mutated(self) -> None:
        refs = ["k-1"]
        remove_source_ref(refs, "k-1")
        assert refs == ["k-1"]


class TestComputeHealthScore:
    def test_healthy_wiki_scores_full(self) -> None:
        assert compute_health_score(stats=_stats(total_pages=3, total_links=1), issues=[]) == 100

    def test_empty_wiki_scores_full(self) -> None:
        assert compute_health_score(stats=_stats(total_pages=0), issues=[]) == 100

    def test_orphan_heaviness_penalizes_by_bucket(self) -> None:
        heavy = _stats(total_pages=4, total_links=1, orphan_count=3)  # 75% > 50
        assert compute_health_score(stats=heavy, issues=[]) == 75
        moderate = _stats(total_pages=4, total_links=1, orphan_count=2)  # 50% > 25
        assert compute_health_score(stats=moderate, issues=[]) == 90

    def test_broken_links_penalize_per_link(self) -> None:
        issues = [
            WikiLintIssue(
                type=LINT_ISSUE_BROKEN_LINK, severity=SEVERITY_ERROR, page_slug="a", description="x"
            )
        ] * 2
        assert compute_health_score(stats=_stats(total_pages=2), issues=issues) == 90

    def test_linkless_wiki_penalized_when_big_enough(self) -> None:
        stats = _stats(total_pages=3, total_links=0)
        assert compute_health_score(stats=stats, issues=[]) == 85

    def test_empty_pages_penalize_per_page(self) -> None:
        issues = [
            WikiLintIssue(
                type=LINT_ISSUE_EMPTY_CONTENT,
                severity=SEVERITY_WARNING,
                page_slug="a",
                description="x",
            )
        ] * 3
        assert compute_health_score(stats=_stats(total_pages=3, total_links=1), issues=issues) == 91

    def test_score_clamps_at_zero(self) -> None:
        issues = [
            WikiLintIssue(
                type=LINT_ISSUE_BROKEN_LINK, severity=SEVERITY_ERROR, page_slug="a", description="x"
            )
        ] * 25
        assert compute_health_score(stats=_stats(total_pages=1), issues=issues) == 0


class TestBuildSummary:
    def test_no_issues_is_healthy(self) -> None:
        assert build_summary([]) == "Wiki is healthy! No issues found."

    def test_counts_by_severity(self) -> None:
        issues = [
            WikiLintIssue(type="x", severity=SEVERITY_ERROR, page_slug="a", description="d"),
            WikiLintIssue(type="x", severity=SEVERITY_ERROR, page_slug="b", description="d"),
            WikiLintIssue(type="x", severity=SEVERITY_WARNING, page_slug="c", description="d"),
            WikiLintIssue(type="x", severity=SEVERITY_INFO, page_slug="d", description="d"),
        ]
        assert build_summary(issues) == "Found 4 issues: 2 errors, 1 warnings, 1 suggestions."


# ── WikiLintService.run_lint ────────────────────────────────────────


class TestRunLint:
    async def test_rejects_non_wiki_kb(self) -> None:
        service, _, _ = _lint_service([], enabled=False)
        with pytest.raises(ValidationError) as excinfo:
            await service.run_lint(knowledge_base_id="kb-1")
        assert excinfo.value.code == "wiki.lint_kb_not_wiki"

    async def test_missing_kb_propagates_not_found(self) -> None:
        service, _, kb = _lint_service([])
        kb.get_knowledge_base_by_id_only.side_effect = NotFoundError(
            code="knowledge_base.not_found", message="missing"
        )
        with pytest.raises(NotFoundError):
            await service.run_lint(knowledge_base_id="kb-1")

    async def test_healthy_wiki_reports_no_issues(self) -> None:
        pages = [
            _sample_page(slug="index", page_type="index", title="Index", in_links=["entity/acme"]),
            _sample_page(slug="entity/acme", in_links=["summary/one"]),
            _sample_page(
                slug="summary/one",
                page_type="summary",
                title="One",
                in_links=["entity/acme"],
                out_links=["entity/acme"],
            ),
        ]
        service, _, _ = _lint_service(pages)
        report = await service.run_lint(knowledge_base_id="kb-1")
        assert report.health_score == 100
        assert report.issues == []
        assert report.summary == "Wiki is healthy! No issues found."

    async def test_report_shape(self) -> None:
        page = _sample_page(slug="entity/acme", in_links=["summary/one"])
        service, _, _ = _lint_service([page])
        report = await service.run_lint(knowledge_base_id="kb-1")
        assert isinstance(report, WikiLintReport)
        assert report.knowledge_base_id == "kb-1"
        assert report.stats.total_pages == 1
        assert 0 <= report.health_score <= 100

    async def test_detects_orphan_page(self) -> None:
        page = _sample_page(slug="entity/acme", in_links=[])
        service, _, _ = _lint_service([page])
        report = await service.run_lint(knowledge_base_id="kb-1")
        orphan = [i for i in report.issues if i.type == LINT_ISSUE_ORPHAN_PAGE]
        assert len(orphan) == 1
        assert orphan[0].page_slug == "entity/acme"
        assert orphan[0].severity == SEVERITY_WARNING
        assert orphan[0].auto_fixable is False

    async def test_index_page_is_not_orphan(self) -> None:
        page = _sample_page(slug="index", page_type="index", title="Index", in_links=[])
        service, _, _ = _lint_service([page])
        report = await service.run_lint(knowledge_base_id="kb-1")
        assert all(i.type != LINT_ISSUE_ORPHAN_PAGE for i in report.issues)

    async def test_detects_broken_link(self) -> None:
        page = _sample_page(
            slug="summary/one", page_type="summary", title="One", out_links=["entity/ghost"]
        )
        service, _, _ = _lint_service([page])
        report = await service.run_lint(knowledge_base_id="kb-1")
        broken = [i for i in report.issues if i.type == LINT_ISSUE_BROKEN_LINK]
        assert len(broken) == 1
        assert broken[0].target_slug == "entity/ghost"
        assert broken[0].severity == SEVERITY_ERROR
        assert broken[0].auto_fixable is True

    async def test_detects_empty_content(self) -> None:
        page = _sample_page(
            slug="summary/tiny", page_type="summary", title="Tiny", content="  few words  "
        )
        service, _, _ = _lint_service([page])
        report = await service.run_lint(knowledge_base_id="kb-1")
        empty = [i for i in report.issues if i.type == LINT_ISSUE_EMPTY_CONTENT]
        assert len(empty) == 1
        assert "(9 chars)" in empty[0].description
        assert empty[0].auto_fixable is True

    async def test_detects_stale_ref(self) -> None:
        async def resolver(kid: str) -> bool:
            return kid == "kid-live"

        page = _sample_page(
            slug="summary/one",
            page_type="summary",
            title="One",
            source_refs=["kid-live", "kid-dead"],
        )
        service, _, _ = _lint_service([page], resolver=resolver)
        report = await service.run_lint(knowledge_base_id="kb-1")
        stale = [i for i in report.issues if i.type == LINT_ISSUE_STALE_REF]
        assert len(stale) == 1
        assert stale[0].target_slug == "kid-dead"
        assert stale[0].severity == SEVERITY_ERROR
        assert stale[0].auto_fixable is True

    async def test_stale_ref_handles_prefixed_form(self) -> None:
        async def resolver(kid: str) -> bool:
            return False

        page = _sample_page(
            slug="summary/one", page_type="summary", title="One", source_refs=["kid-dead|A Title"]
        )
        service, _, _ = _lint_service([page], resolver=resolver)
        report = await service.run_lint(knowledge_base_id="kb-1")
        stale = [i for i in report.issues if i.type == LINT_ISSUE_STALE_REF]
        assert [i.target_slug for i in stale] == ["kid-dead"]

    async def test_stale_ref_skipped_without_resolver(self) -> None:
        page = _sample_page(
            slug="summary/one", page_type="summary", title="One", source_refs=["kid-dead"]
        )
        service, _, _ = _lint_service([page])
        report = await service.run_lint(knowledge_base_id="kb-1")
        assert all(i.type != LINT_ISSUE_STALE_REF for i in report.issues)

    async def test_stale_ref_skips_index_page(self) -> None:
        async def resolver(kid: str) -> bool:
            return False

        page = _sample_page(
            slug="index", page_type="index", title="Index", source_refs=["kid-dead"]
        )
        service, _, _ = _lint_service([page], resolver=resolver)
        report = await service.run_lint(knowledge_base_id="kb-1")
        assert all(i.type != LINT_ISSUE_STALE_REF for i in report.issues)

    async def test_knowledge_liveness_is_cached_per_id(self) -> None:
        calls: list[str] = []

        async def resolver(kid: str) -> bool:
            calls.append(kid)
            return False

        pages = [
            _sample_page(
                id="p1",
                slug="summary/one",
                page_type="summary",
                title="One",
                source_refs=["kid-dead"],
            ),
            _sample_page(
                id="p2",
                slug="summary/two",
                page_type="summary",
                title="Two",
                source_refs=["kid-dead"],
            ),
        ]
        service, _, _ = _lint_service(pages, resolver=resolver)
        report = await service.run_lint(knowledge_base_id="kb-1")
        assert len(calls) == 1
        assert len([i for i in report.issues if i.type == LINT_ISSUE_STALE_REF]) == 2

    async def test_resolver_failure_reads_as_dead(self) -> None:
        async def resolver(kid: str) -> bool:
            raise RuntimeError("lookup failed")

        page = _sample_page(
            slug="summary/one", page_type="summary", title="One", source_refs=["kid-x"]
        )
        service, _, _ = _lint_service([page], resolver=resolver)
        report = await service.run_lint(knowledge_base_id="kb-1")
        assert len([i for i in report.issues if i.type == LINT_ISSUE_STALE_REF]) == 1

    async def test_detects_missing_cross_ref(self) -> None:
        entity = _sample_page(slug="entity/acme", in_links=["summary/one"])
        mention = _sample_page(
            slug="summary/one",
            page_type="summary",
            title="One",
            content="Acme dominates the market here.",
        )
        service, _, _ = _lint_service([entity, mention])
        report = await service.run_lint(knowledge_base_id="kb-1")
        cross = [i for i in report.issues if i.type == LINT_ISSUE_MISSING_CROSS_REF]
        assert len(cross) == 1
        assert cross[0].page_slug == "summary/one"
        assert cross[0].target_slug == "entity/acme"
        assert cross[0].severity == SEVERITY_INFO

    async def test_cross_ref_skips_already_linked_and_self(self) -> None:
        entity = _sample_page(slug="entity/acme", in_links=["summary/one"])
        linked = _sample_page(
            slug="summary/one",
            page_type="summary",
            title="One",
            content="Acme is linked via [[entity/acme]].",
            out_links=["entity/acme"],
        )
        self_mention = _sample_page(
            slug="entity/other",
            page_type="entity",
            title="Other",
            content="Other is self-referential.",
        )
        service, _, _ = _lint_service([entity, linked, self_mention])
        report = await service.run_lint(knowledge_base_id="kb-1")
        assert all(i.type != LINT_ISSUE_MISSING_CROSS_REF for i in report.issues)

    async def test_cross_ref_matches_case_insensitively(self) -> None:
        entity = _sample_page(slug="entity/acme", in_links=["summary/one"])
        mention = _sample_page(
            slug="summary/one",
            page_type="summary",
            title="One",
            content="Mentions ACME in uppercase.",
        )
        service, _, _ = _lint_service([entity, mention])
        report = await service.run_lint(knowledge_base_id="kb-1")
        assert len([i for i in report.issues if i.type == LINT_ISSUE_MISSING_CROSS_REF]) == 1

    async def test_streams_over_multiple_batches_and_crosses_them(self) -> None:
        # Batch 1 carries the entity title; batch 2 carries a page that
        # mentions it. The cross-ref check needs the full entity set, so
        # this only passes when pass 1 ran to completion first.
        entity = _sample_page(slug="entity/acme", in_links=["summary/one"])
        mention = _sample_page(
            slug="summary/one",
            page_type="summary",
            title="One",
            content="Acme mention without a link.",
        )
        service, wiki, _ = _lint_service([entity, mention])

        async def _cursor(
            *, knowledge_base_id: str, cursor: str = "", limit: int = 100
        ) -> tuple[list[WikiPage], str]:
            del knowledge_base_id, limit
            live = [entity, mention]
            idx = int(cursor) if cursor else 0
            window = live[idx : idx + 1]
            next_cursor = str(idx + 1) if idx + 1 < len(live) else ""
            return window, next_cursor

        wiki.list_pages_cursor.side_effect = _cursor
        report = await service.run_lint(knowledge_base_id="kb-1")
        assert len([i for i in report.issues if i.type == LINT_ISSUE_MISSING_CROSS_REF]) == 1
        cursors = [c.kwargs["cursor"] for c in wiki.list_pages_cursor.await_args_list]
        assert cursors == ["", "1", "", "1"]

    async def test_health_score_penalizes_orphan_heavy_wiki(self) -> None:
        pages = [
            _sample_page(
                slug=f"entity/p{i}", in_links=[], out_links=["entity/p1"] if i == 0 else []
            )
            for i in range(4)
        ]
        service, _, _ = _lint_service(pages)
        report = await service.run_lint(knowledge_base_id="kb-1")
        assert report.health_score == 75  # 100% orphans -> -25


# ── WikiLintService.auto_fix ────────────────────────────────────────


class TestAutoFix:
    async def test_fixes_broken_link_and_rebuilds(self) -> None:
        page = _sample_page(
            slug="summary/one",
            page_type="summary",
            title="One",
            content=f"{_LONG_BODY} See [[entity/ghost]] for details.",
            out_links=["entity/ghost"],
        )
        service, wiki, _ = _lint_service([page])
        fixed = await service.auto_fix(knowledge_base_id="kb-1")
        assert fixed == 1
        assert "[[entity/ghost]]" not in wiki.rows[page.id].content
        assert "entity/ghost" in wiki.rows[page.id].content
        wiki.rebuild_links.assert_awaited_once()

    async def test_fixes_broken_link_with_display_text_preserved(self) -> None:
        page = _sample_page(
            slug="summary/one",
            page_type="summary",
            title="One",
            content=f"{_LONG_BODY} See [[entity/ghost|Ghost Co]] for details.",
            out_links=["entity/ghost"],
        )
        service, wiki, _ = _lint_service([page])
        await service.auto_fix(knowledge_base_id="kb-1")
        # only the bare [[target]] form is degraded; display form is untouched
        assert "[[entity/ghost|Ghost Co]]" in wiki.rows[page.id].content

    async def test_empty_content_page_is_archived(self) -> None:
        page = _sample_page(slug="summary/tiny", page_type="summary", title="Tiny", content="short")
        service, wiki, _ = _lint_service([page])
        fixed = await service.auto_fix(knowledge_base_id="kb-1")
        assert fixed == 1
        assert wiki.rows[page.id].status == "archived"

    async def test_index_page_is_not_archived(self) -> None:
        page = _sample_page(slug="index", page_type="index", title="Index", content="short")
        service, wiki, _ = _lint_service([page])
        await service.auto_fix(knowledge_base_id="kb-1")
        assert wiki.rows[page.id].status == "published"
        wiki.rebuild_links.assert_not_awaited()

    async def test_stale_ref_keeps_remaining_sources(self) -> None:
        async def resolver(kid: str) -> bool:
            return kid == "kid-keep"

        page = _sample_page(
            slug="summary/one",
            page_type="summary",
            title="One",
            source_refs=["kid-dead", "kid-keep"],
        )
        service, wiki, _ = _lint_service([page], resolver=resolver)
        fixed = await service.auto_fix(knowledge_base_id="kb-1")
        assert fixed == 1
        assert wiki.rows[page.id].source_refs == ["kid-keep"]

    async def test_stale_ref_deletes_page_without_survivors(self) -> None:
        async def resolver(kid: str) -> bool:
            return False

        page = _sample_page(
            slug="summary/one", page_type="summary", title="One", source_refs=["kid-dead"]
        )
        service, wiki, _ = _lint_service([page], resolver=resolver)
        fixed = await service.auto_fix(knowledge_base_id="kb-1")
        assert fixed == 1
        assert wiki.rows[page.id].deleted_at is not None

    async def test_no_fixable_issues_skips_rebuild(self) -> None:
        page = _sample_page(slug="entity/acme", in_links=["summary/one"])
        service, wiki, _ = _lint_service([page])
        await service.auto_fix(knowledge_base_id="kb-1")
        wiki.rebuild_links.assert_not_awaited()

    async def test_page_deleted_mid_fix_is_skipped(self) -> None:
        page = _sample_page(
            slug="summary/one",
            page_type="summary",
            title="One",
            content=f"{_LONG_BODY} See [[entity/ghost]].",
            out_links=["entity/ghost"],
        )
        service, wiki, _ = _lint_service([page])

        async def _gone(*, knowledge_base_id: str, slug: str) -> WikiPage:
            del knowledge_base_id, slug
            raise NotFoundError(code="wiki.page_not_found", message="gone")

        wiki.get_page_by_slug.side_effect = _gone
        fixed = await service.auto_fix(knowledge_base_id="kb-1")
        assert fixed == 0
        wiki.rebuild_links.assert_not_awaited()


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


async def _seed_wiki_kb(db_session: AsyncSession, tenant_id: int) -> tuple[str, KBService]:
    kb_service = KBService(kb_repo=KnowledgeBaseRepository(db_session))
    created = await kb_service.create_knowledge_base(
        tenant_id=tenant_id,
        name="Lint Wiki",
        indexing_strategy={
            "vector_enabled": False,
            "keyword_enabled": False,
            "wiki_enabled": True,
            "graph_enabled": False,
        },
    )
    return created.id, kb_service


async def test_integration_run_lint_health_report(db_session: AsyncSession) -> None:
    tenant_id = make_test_tenant_id()
    kb_id, kb_service = await _seed_wiki_kb(db_session, tenant_id)
    page_service = WikiPageService(
        page_repo=WikiPageRepository(db_session),
        folder_repo=WikiFolderRepository(db_session),
    )
    await page_service.create_page(
        page=_sample_page(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            slug="index",
            page_type="index",
            title="Index",
            content=_LONG_BODY,
        )
    )
    await page_service.create_page(
        page=_sample_page(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            slug="summary/one",
            page_type="summary",
            title="One",
            content=f"{_LONG_BODY} Links to [[entity/ghost]] nowhere.",
        )
    )

    service = WikiLintService(wiki_service=page_service, kb_service=kb_service)
    report = await service.run_lint(knowledge_base_id=kb_id)

    assert isinstance(report, WikiLintReport)
    assert report.knowledge_base_id == kb_id
    assert 0 <= report.health_score <= 100
    assert report.stats.total_pages == 2
    assert any(i.type == LINT_ISSUE_BROKEN_LINK for i in report.issues)
    assert report.summary


async def test_integration_auto_fix_broken_link(db_session: AsyncSession) -> None:
    tenant_id = make_test_tenant_id()
    kb_id, kb_service = await _seed_wiki_kb(db_session, tenant_id)
    page_service = WikiPageService(
        page_repo=WikiPageRepository(db_session),
        folder_repo=WikiFolderRepository(db_session),
    )
    created = await page_service.create_page(
        page=_sample_page(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            slug="summary/one",
            page_type="summary",
            title="One",
            content=f"{_LONG_BODY} See [[entity/ghost]] for details.",
        )
    )

    service = WikiLintService(wiki_service=page_service, kb_service=kb_service)
    fixed = await service.auto_fix(knowledge_base_id=kb_id)
    assert fixed >= 1

    page = await page_service.get_page_by_slug(knowledge_base_id=kb_id, slug="summary/one")
    assert "[[entity/ghost]]" not in page.content
    assert page.id == created.id


async def test_integration_non_wiki_rejected(db_session: AsyncSession) -> None:
    tenant_id = make_test_tenant_id()
    kb_service = KBService(kb_repo=KnowledgeBaseRepository(db_session))
    created = await kb_service.create_knowledge_base(
        tenant_id=tenant_id,
        name="Plain Docs",
        indexing_strategy={
            "vector_enabled": True,
            "keyword_enabled": True,
            "wiki_enabled": False,
            "graph_enabled": False,
        },
    )
    page_service = WikiPageService(
        page_repo=WikiPageRepository(db_session),
        folder_repo=WikiFolderRepository(db_session),
    )
    service = WikiLintService(wiki_service=page_service, kb_service=kb_service)
    with pytest.raises(ValidationError) as excinfo:
        await service.run_lint(knowledge_base_id=created.id)
    assert excinfo.value.code == "wiki.lint_kb_not_wiki"


async def test_integration_stale_ref_against_real_documents(db_session: AsyncSession) -> None:
    tenant_id = make_test_tenant_id()
    kb_id, kb_service = await _seed_wiki_kb(db_session, tenant_id)
    knowledge_repo = KnowledgeRepository(db_session)

    live_doc_id = f"doc-{uuid.uuid4().hex[:12]}"
    await knowledge_repo.insert(
        Document(
            id=live_doc_id,
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            type="knowledge",
            title="Source doc",
            source="manual",
            channel="web",
            created_at=_NOW,
            updated_at=_NOW,
        )
    )

    async def resolver(kid: str) -> bool:
        return await knowledge_repo.get_by_id_only(kid) is not None

    page_service = WikiPageService(
        page_repo=WikiPageRepository(db_session),
        folder_repo=WikiFolderRepository(db_session),
    )
    await page_service.create_page(
        page=_sample_page(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            slug="summary/one",
            page_type="summary",
            title="One",
            source_refs=[live_doc_id, "missing-doc"],
            content=_LONG_BODY,
        )
    )

    service = WikiLintService(
        wiki_service=page_service, kb_service=kb_service, knowledge_resolver=resolver
    )
    report = await service.run_lint(knowledge_base_id=kb_id)
    stale = [i for i in report.issues if i.type == LINT_ISSUE_STALE_REF]
    assert [i.target_slug for i in stale] == ["missing-doc"]
