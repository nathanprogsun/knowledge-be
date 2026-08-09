"""Unit + integration tests for the wiki agent tools.

Unit tests drive each tool through injected seams — an in-memory wiki
page service, a fake tag fetcher, a fake knowledge lookup, a fake chunk
store, a fake issue repository, a fake KB loader, and a fake search
runner — so no test touches storage or an LLM. They cover validation,
routing / scope enforcement, budgeted rendering, the write mutation
cascade (delete / rename link rewrites and rollback), graph aggregation,
and the issue lifecycle.

Integration tests run against the real applied schema. The ``chunks``
table carries an INTEGER 32-bit ``tenant_id``, so integration tests mint
ids from a local counter; ``wiki_pages.tenant_id`` is BIGINT and accepts
the same values. They seed a knowledge base, wiki pages, and chunks,
then execute the read / search / write / source-doc tools through the
real service layers. Requires a reachable database — run with
``DATABASE_URL_OVERRIDE``.
"""

from __future__ import annotations

import itertools
import json
import re
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from random import randint

import pytest
import pytest_asyncio
from faker import Faker
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.ai.embedding.base import Context
from src.ai.retrieval.types import MatchType
from src.common.exception import NotFoundError, ValidationError
from src.common.json import JsonObject
from src.core.agents.tools.chunk_store import PagedChunkStore, SqlPagedChunkStore
from src.core.agents.tools.kb_tool import SearchCall
from src.core.agents.tools.scope_auth import KnowledgeTagsFetcher
from src.core.agents.tools.search_target import SearchTarget, SearchTargets, SearchTargetType
from src.core.agents.tools.wiki_graph import (
    GRAPH_MAX_KB_IDS,
    QueryKnowledgeGraphTool,
    aggregate_graph_config,
    build_graph_visualization_data,
    build_wiki_graph_definition,
    graph_configs_to_data,
    summarize_graph_config,
)
from src.core.agents.tools.wiki_issue import (
    WikiFlagIssueTool,
    WikiReadIssueTool,
    WikiUpdateIssueTool,
    build_wiki_flag_issue_definition,
    build_wiki_read_issue_definition,
    build_wiki_update_issue_definition,
)
from src.core.agents.tools.wiki_read import (
    WikiReadPageTool,
    WikiReadSourceDocTool,
    build_wiki_read_page_definition,
    build_wiki_read_source_doc_definition,
    render_index_overview_for_agent,
    render_wiki_pages_within_budget,
)
from src.core.agents.tools.wiki_route import (
    WikiPageAmbiguousError,
    WikiPageNotFoundInScopeError,
    WikiRouteResolver,
    WikiScope,
    apply_incoming_wiki_content_rewrite,
    new_wiki_scopes_from_kb_ids,
    new_wiki_scopes_from_search_targets,
    normalize_and_validate_wiki_slug,
    page_passes_wiki_scope,
    resolve_unique_wiki_page,
    rollback_wiki_content_changes,
)
from src.core.agents.tools.wiki_search import (
    WikiSearchTool,
    build_wiki_search_definition,
    extract_snippet,
)
from src.core.agents.tools.wiki_write import (
    WikiDeletePageTool,
    WikiRenamePageTool,
    WikiReplaceTextTool,
    WikiWritePageTool,
    build_wiki_delete_page_definition,
    build_wiki_rename_page_definition,
    build_wiki_replace_text_definition,
    build_wiki_write_page_definition,
)
from src.core.contracts.knowledge import Knowledge
from src.core.knowledge.documents.service.knowledge_service import KnowledgeService
from src.core.knowledge.documents.types import CHANNEL_WEB, PARSE_STATUS_COMPLETED
from src.core.knowledge.knowledge_bases.hybrid_search import SearchResult
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import (
    KNOWLEDGE_BASE_TYPE_DOCUMENT,
    KnowledgeBaseInfo,
)
from src.core.knowledge.tags.types import TagInfo
from src.core.knowledge.wiki.issues import (
    WIKI_ISSUE_STATUS_PENDING,
    WIKI_ISSUE_STATUS_RESOLVED,
    WikiPageIssue,
    WikiPageIssueRepository,
)
from src.core.knowledge.wiki.page_service import WikiPageService
from src.core.knowledge.wiki.types import (
    WIKI_PAGE_TYPE_ENTITY,
    WIKI_PAGE_TYPE_INDEX,
    WIKI_PAGE_TYPE_SUMMARY,
    WikiIndexGroup,
    WikiIndexResponse,
)
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.dao.wiki_page_repository import WikiFolderRepository, WikiPageRepository
from src.db.models.chunk import Chunk
from src.db.models.knowledge import Document
from src.db.models.wiki_page import WikiIndexEntry, WikiPage
from src.settings import get_settings, reset_settings_cache

_NOW = datetime(2026, 2, 1, tzinfo=UTC)
_FAKER_SEED_MAX = 100_000_000

#: ``chunks.tenant_id`` is INTEGER (32-bit); integration tests mint ids
#: from this counter so they stay inside the range.
_INT32_TENANT_BASE = 6_000_000
_INT32_TENANT_SEQ = itertools.count(start=1)


def _int32_tenant_id() -> int:
    """Return a tenant id unique within the session, safe for INTEGER."""
    return _INT32_TENANT_BASE + next(_INT32_TENANT_SEQ)


@pytest.fixture(autouse=True)
def faker_seed() -> None:
    """Re-seed Faker per test for varied-but-reproducible generation."""
    Faker.seed(randint(1, _FAKER_SEED_MAX))


# ── Shared test doubles ───────────────────────────────────────────────


class _Context:
    """Minimal task context satisfying the ``Context`` protocol."""

    is_background_task: bool = False


def _page(
    slug: str,
    *,
    knowledge_base_id: str = "kb-1",
    tenant_id: int = 7,
    title: str = "",
    summary: str = "",
    content: str = "",
    page_type: str = WIKI_PAGE_TYPE_ENTITY,
    aliases: list[str] | None = None,
    source_refs: list[str] | None = None,
    in_links: list[str] | None = None,
    out_links: list[str] | None = None,
) -> WikiPage:
    """Build one wiki page row for unit tests."""
    return WikiPage(
        id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        slug=slug,
        title=title or slug.split("/")[-1].replace("-", " ").title(),
        summary=summary,
        content=content,
        page_type=page_type,
        status="published",
        source_refs=list(source_refs or []),
        in_links=list(in_links or []),
        out_links=list(out_links or []),
        aliases=list(aliases or []),
        created_at=_NOW,
        updated_at=_NOW,
    )


class _FakeWikiService:
    """In-memory wiki page service satisfying the tool protocol."""

    def __init__(self, pages: list[WikiPage] | None = None) -> None:
        self._pages: list[WikiPage] = list(pages or [])
        self.index_views: dict[str, WikiIndexResponse] = {}
        self.created: list[tuple[WikiPage, str]] = []
        self.updated: list[tuple[WikiPage, str]] = []
        self.auto_updated: list[WikiPage] = []
        self.deleted: list[tuple[str, str]] = []
        self.injected: list[tuple[str, list[str]]] = []
        self.rebuilt: list[str] = []
        self.repair_results: tuple[str, bool] | None = None

    def stored(self) -> list[WikiPage]:
        return list(self._pages)

    async def get_page_by_slug(self, *, knowledge_base_id: str, slug: str) -> WikiPage:
        for page in self._pages:
            if page.knowledge_base_id == knowledge_base_id and page.slug == slug:
                return page
        raise NotFoundError(
            code="wiki.page_not_found",
            message=f"wiki page {slug} not found in knowledge base {knowledge_base_id}",
        )

    async def create_page(
        self, *, page: WikiPage, edit_source: str = "", editor_id: str = ""
    ) -> WikiPage:
        self._pages.append(page)
        self.created.append((page, edit_source))
        return page

    async def update_page(
        self, *, page: WikiPage, edit_source: str = "", editor_id: str = ""
    ) -> WikiPage:
        for i, stored in enumerate(self._pages):
            if stored.knowledge_base_id == page.knowledge_base_id and stored.slug == page.slug:
                self._pages[i] = page
                self.updated.append((page, edit_source))
                return page
        self._pages.append(page)
        self.updated.append((page, edit_source))
        return page

    async def update_auto_linked_content(self, *, page: WikiPage) -> WikiPage:
        for i, stored in enumerate(self._pages):
            if stored.knowledge_base_id == page.knowledge_base_id and stored.slug == page.slug:
                self._pages[i] = page
                self.auto_updated.append(page)
                return page
        self._pages.append(page)
        self.auto_updated.append(page)
        return page

    async def delete_page(self, *, knowledge_base_id: str, slug: str) -> None:
        self.deleted.append((knowledge_base_id, slug))
        self._pages = [
            page
            for page in self._pages
            if not (page.knowledge_base_id == knowledge_base_id and page.slug == slug)
        ]

    async def search_pages(
        self, *, knowledge_base_id: str, query: str, limit: int = 10
    ) -> list[WikiPage]:
        pattern = re.compile(query, re.IGNORECASE)
        hits: list[WikiPage] = []
        for page in self._pages:
            if page.knowledge_base_id != knowledge_base_id:
                continue
            haystack = f"{page.title} {page.summary} {page.content} {page.slug}"
            if pattern.search(haystack):
                hits.append(page)
        return hits[:limit]

    async def get_index_view(
        self,
        *,
        knowledge_base_id: str,
        tenant_id: int,
        page_types: list[str] | None = None,
        limit: int = 0,
        cursor: str = "",
    ) -> WikiIndexResponse:
        return self.index_views[knowledge_base_id]

    async def inject_cross_links(
        self, *, knowledge_base_id: str, affected_slugs: list[str]
    ) -> int:
        self.injected.append((knowledge_base_id, affected_slugs))
        return 0

    async def rebuild_index_page(self, *, knowledge_base_id: str) -> None:
        self.rebuilt.append(knowledge_base_id)

    async def repair_content_links(
        self, *, knowledge_base_id: str, self_slug: str, content: str
    ) -> tuple[str, bool]:
        if self.repair_results is not None:
            return self.repair_results
        return content, False


class _FakeTagFetcher(KnowledgeTagsFetcher):
    """Fake tag fetcher returning canned knowledge → tag bindings."""

    def __init__(self, bindings: dict[str, list[TagInfo]]) -> None:
        self._bindings = bindings
        self.calls: list[list[str]] = []

    async def get_knowledge_tags(self, knowledge_ids: list[str]) -> dict[str, list[TagInfo]]:
        self.calls.append(list(knowledge_ids))
        return {kid: list(self._bindings.get(kid, [])) for kid in knowledge_ids}


def _tag(tag_id: str, name: str = "tag") -> TagInfo:
    return TagInfo(
        id=tag_id,
        seq_id=0,
        tenant_id=7,
        knowledge_base_id="kb-1",
        name=name,
        created_at=_NOW,
        updated_at=_NOW,
    )


class _FakeKnowledgeLookup:
    """Fake document lookup seam returning canned ``Knowledge`` records."""

    def __init__(self, docs: list[Knowledge] | None = None) -> None:
        self._docs = list(docs or [])
        self.calls: list[str] = []

    async def get_document_by_id_only(self, *, id: str) -> Knowledge | None:
        self.calls.append(id)
        for doc in self._docs:
            if doc.id == id:
                return doc
        return None


def _knowledge(
    id: str = "d1",
    knowledge_base_id: str = "kb-1",
    tenant_id: int = 7,
    title: str = "Doc",
    file_name: str = "doc.pdf",
) -> Knowledge:
    """Build one document contract record."""
    return Knowledge(
        id=id,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        type="file",
        title=title,
        source="doc.pdf",
        channel=CHANNEL_WEB,
        parse_status=PARSE_STATUS_COMPLETED,
        enable_status="enabled",
        file_name=file_name,
        created_at=_NOW,
        updated_at=_NOW,
    )


class _FakeChunkStore(PagedChunkStore):
    """In-memory paged chunk store for the source-doc reader."""

    def __init__(self, chunks: list[Chunk] | None = None, total: int = 0) -> None:
        self._chunks = list(chunks or [])
        self._total = total if total > 0 else len(self._chunks)
        self.calls: list[tuple[int, str, int, int]] = []

    async def list_paged_chunks(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        page: int,
        page_size: int,
        enabled_only: bool = True,
    ) -> tuple[list[Chunk], int]:
        self.calls.append((tenant_id, knowledge_id, page, page_size))
        offset = (page - 1) * page_size
        return self._chunks[offset : offset + page_size], self._total


def _chunk(
    chunk_index: int,
    content: str,
    *,
    id: str | None = None,
    knowledge_id: str = "d1",
    knowledge_base_id: str = "kb-1",
    tenant_id: int = 7,
    image_info: str = "",
) -> Chunk:
    """Build one chunk row for unit tests."""
    return Chunk(
        id=id or f"c-{chunk_index}",
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        knowledge_id=knowledge_id,
        content=content,
        chunk_index=chunk_index,
        is_enabled=True,
        start_at=0,
        end_at=len(content),
        pre_chunk_id=None,
        next_chunk_id=None,
        chunk_type="text",
        parent_chunk_id=None,
        image_info=image_info or None,
        relation_chunks=None,
        indirect_relation_chunks=None,
        metadata={"source": "manual"},
        tag_id=None,
        status=1,
        content_hash=None,
        flags=1,
        seq_id=0,
        source_content="",
        content_revision=0,
        index_status="ready",
        last_editor_id="",
        context_header="",
        created_at=_NOW,
        updated_at=_NOW,
        deleted_at=None,
    )


class _FakeKbLoader:
    """Fake KB loader returning canned ``KnowledgeBaseInfo`` records."""

    def __init__(self, kbs: list[KnowledgeBaseInfo] | None = None) -> None:
        self._kbs = list(kbs or [])
        self.calls: list[str] = []

    async def load(self, *, knowledge_base_id: str) -> KnowledgeBaseInfo | None:
        self.calls.append(knowledge_base_id)
        for kb in self._kbs:
            if kb.id == knowledge_base_id:
                return kb
        return None


def _kb_info(
    id: str = "kb-1",
    *,
    tenant_id: int = 7,
    extract_config: JsonObject | None = None,
) -> KnowledgeBaseInfo:
    """Build one knowledge-base info record for the graph tool."""
    return KnowledgeBaseInfo(
        id=id,
        name="Test KB",
        type=KNOWLEDGE_BASE_TYPE_DOCUMENT,
        tenant_id=tenant_id,
        extract_config=extract_config,
        created_at=_NOW,
        updated_at=_NOW,
    )


class _FakeSearchRunner:
    """Search-runner seam returning canned search results.

    ``per_kb`` maps a knowledge-base id to that KB's hits so the tool's
    per-KB query loop yields distinct result sets.
    """

    def __init__(
        self,
        results: list[SearchResult] | None = None,
        per_kb: dict[str, list[SearchResult]] | None = None,
    ) -> None:
        self._results = list(results or [])
        self._per_kb = dict(per_kb or {})
        self.calls: list[SearchCall] = []

    async def search(self, ctx: Context, call: SearchCall) -> list[SearchResult]:
        self.calls.append(call)
        if call.kb_id in self._per_kb:
            return list(self._per_kb[call.kb_id])
        return list(self._results)


def _result(
    id: str,
    *,
    content: str = "alpha beta gamma",
    knowledge_id: str = "d1",
    chunk_index: int = 1,
    knowledge_title: str = "Doc",
    score: float = 0.9,
    knowledge_base_id: str = "kb-1",
    match_type: MatchType = MatchType.EMBEDDING,
) -> SearchResult:
    """Build one hydrated search hit for the graph tool."""
    return SearchResult(
        id=id,
        content=content,
        knowledge_id=knowledge_id,
        chunk_index=chunk_index,
        knowledge_title=knowledge_title,
        score=score,
        match_type=match_type,
        knowledge_base_id=knowledge_base_id,
    )


class _FakeIssueRepo(WikiPageIssueRepository):
    """In-memory issue repository."""

    def __init__(self, issues: list[WikiPageIssue] | None = None) -> None:
        self._issues = list(issues or [])
        self.created: list[WikiPageIssue] = []
        self.status_updates: list[tuple[str, str]] = []

    async def create(self, issue: WikiPageIssue) -> WikiPageIssue:
        self._issues.append(issue)
        self.created.append(issue)
        return issue

    async def list(
        self, *, knowledge_base_id: str, slug: str = "", status: str = ""
    ) -> list[WikiPageIssue]:
        return [
            issue
            for issue in self._issues
            if issue.knowledge_base_id == knowledge_base_id
            and (not slug or issue.slug == slug)
            and (not status or issue.status == status)
        ]

    async def get_by_id_or_none(self, *, issue_id: str) -> WikiPageIssue | None:
        for issue in self._issues:
            if issue.id == issue_id:
                return issue
        return None

    async def update_status(self, *, issue_id: str, status: str) -> None:
        self.status_updates.append((issue_id, status))


def _issue(
    issue_id: str,
    *,
    knowledge_base_id: str = "kb-1",
    slug: str = "entity/acme",
    status: str = WIKI_ISSUE_STATUS_PENDING,
) -> WikiPageIssue:
    """Build one issue record."""
    return WikiPageIssue(
        id=issue_id,
        tenant_id=7,
        knowledge_base_id=knowledge_base_id,
        slug=slug,
        issue_type="contradictory_facts",
        description="The facts conflict.",
        status=status,
        reported_by="wiki-researcher-agent",
        created_at=_NOW,
        updated_at=_NOW,
    )


def _scope(kb_id: str, *, knowledge_ids: list[str] | None = None, tag_ids: list[str] | None = None) -> WikiScope:
    """Build one wiki scope."""
    return WikiScope(
        knowledge_base_id=kb_id,
        knowledge_ids=list(knowledge_ids or []),
        tag_ids=list(tag_ids or []),
    )


def _kb_target(knowledge_base_id: str, tenant_id: int = 7) -> SearchTarget:
    """Build one whole-KB search target."""
    return SearchTarget(
        type=SearchTargetType.KNOWLEDGE_BASE,
        knowledge_base_id=knowledge_base_id,
        tenant_id=tenant_id,
    )


def _targets(*targets: SearchTarget) -> SearchTargets:
    return SearchTargets(targets=tuple(targets))


# ── wiki_route ────────────────────────────────────────────────────────


class TestWikiRouteResolver:
    def test_remember_and_scopes_for_slug_preserve_scope_order(self) -> None:
        resolver = WikiRouteResolver()
        scopes = [_scope("kb-1"), _scope("kb-2"), _scope("kb-3")]
        resolver.remember("entity/acme", "kb-2")

        assert resolver.scopes_for_slug("entity/acme", scopes) == [_scope("kb-2")]

    def test_forget_drops_slug_when_no_owners_left(self) -> None:
        resolver = WikiRouteResolver()
        resolver.remember("entity/acme", "kb-1")
        resolver.forget("entity/acme", "kb-1")
        assert resolver.scopes_for_slug("entity/acme", [_scope("kb-1")]) == []

    def test_remember_page_registers_neighbours(self) -> None:
        resolver = WikiRouteResolver()
        page = _page("entity/acme", out_links=["concept/rag"], in_links=["index"])
        resolver.remember_page(page, "kb-1")
        assert resolver.scopes_for_slug("concept/rag", [_scope("kb-1")]) == [_scope("kb-1")]
        assert resolver.scopes_for_slug("index", [_scope("kb-1")]) == [_scope("kb-1")]


class TestSlugNormalization:
    def test_normalizes_whitespace_and_case(self) -> None:
        normalized, error = normalize_and_validate_wiki_slug("  Entity/ACME Corp  ")
        assert error == ""
        assert normalized == "entity/acme-corp"

    def test_keeps_cjk_characters(self) -> None:
        normalized, error = normalize_and_validate_wiki_slug("实体/大模型")
        assert error == ""
        assert normalized == "实体/大模型"

    def test_rejects_duplicate_slashes(self) -> None:
        normalized, error = normalize_and_validate_wiki_slug("entity//acme")
        assert normalized == ""
        assert "malformed" in error

    def test_rejects_stray_characters(self) -> None:
        normalized, error = normalize_and_validate_wiki_slug("entity/acme!")
        assert normalized == ""
        assert "not allowed" in error


class TestUniquePageResolution:
    async def test_resolves_unique_slug(self) -> None:
        service = _FakeWikiService([_page("entity/acme")])
        page, kb_id = await resolve_unique_wiki_page(_Context(), service, "entity/acme", ["kb-1"], WikiRouteResolver())
        assert page.slug == "entity/acme"
        assert kb_id == "kb-1"

    async def test_refuses_ambiguous_slug(self) -> None:
        service = _FakeWikiService(
            [
                _page("entity/acme", knowledge_base_id="kb-1"),
                _page("entity/acme", knowledge_base_id="kb-2"),
            ]
        )
        with pytest.raises(WikiPageAmbiguousError):
            await resolve_unique_wiki_page(
                _Context(), service, "entity/acme", ["kb-1", "kb-2"], WikiRouteResolver()
            )

    async def test_not_found_in_scope(self) -> None:
        service = _FakeWikiService([_page("entity/other")])
        with pytest.raises(WikiPageNotFoundInScopeError):
            await resolve_unique_wiki_page(_Context(), service, "entity/acme", ["kb-1"], WikiRouteResolver())

    async def test_empty_slug_is_rejected(self) -> None:
        service = _FakeWikiService()
        with pytest.raises(ValidationError):
            await resolve_unique_wiki_page(_Context(), service, "  ", ["kb-1"], WikiRouteResolver())


class TestScopeComposition:
    def test_kb_ids_deduplicated(self) -> None:
        scopes = new_wiki_scopes_from_kb_ids(["kb-1", "kb-1", "kb-2"])
        assert [scope.knowledge_base_id for scope in scopes] == ["kb-1", "kb-2"]

    def test_search_targets_union_and_whole_kb_supersede(self) -> None:
        targets = _targets(
            SearchTarget(
                type=SearchTargetType.KNOWLEDGE,
                knowledge_base_id="kb-1",
                knowledge_ids=("d1",),
            ),
            SearchTarget(
                type=SearchTargetType.KNOWLEDGE,
                knowledge_base_id="kb-1",
                knowledge_ids=("d2",),
            ),
            _kb_target("kb-2"),
        )
        scopes = new_wiki_scopes_from_search_targets(targets, ["kb-1", "kb-2"])
        by_kb = {scope.knowledge_base_id: scope for scope in scopes}
        assert sorted(by_kb["kb-1"].knowledge_ids) == ["d1", "d2"]
        assert by_kb["kb-2"].knowledge_ids == []
        assert by_kb["kb-2"].tag_ids == []


class TestPageScope:
    async def test_unconstrained_scope_passes(self) -> None:
        page = _page("entity/acme", source_refs=["d1"])
        assert await page_passes_wiki_scope(_Context(), page, _scope("kb-1"), None)

    async def test_knowledge_whitelist_filters(self) -> None:
        page = _page("entity/acme", source_refs=["d1"])
        assert await page_passes_wiki_scope(_Context(), page, _scope("kb-1", knowledge_ids=["d1"]), None)
        assert not await page_passes_wiki_scope(
            _Context(), page, _scope("kb-1", knowledge_ids=["d2"]), None
        )

    async def test_tag_scope_filters_by_tag(self) -> None:
        page = _page("entity/acme", source_refs=["d1"])
        fetcher = _FakeTagFetcher({"d1": [_tag("t1")], "d2": []})
        assert await page_passes_wiki_scope(_Context(), page, _scope("kb-1", tag_ids=["t1"]), fetcher)
        assert not await page_passes_wiki_scope(
            _Context(), page, _scope("kb-1", tag_ids=["t2"]), fetcher
        )

    async def test_uncited_page_never_passes_constrained_scope(self) -> None:
        page = _page("entity/acme")
        assert not await page_passes_wiki_scope(
            _Context(), page, _scope("kb-1", knowledge_ids=["d1"]), None
        )


class TestIncomingLinkRewrite:
    async def test_applies_and_tracks_changes(self) -> None:
        service = _FakeWikiService(
            [
                _page("entity/other", content="See [[entity/acme|Acme]]."),
                _page("entity/unrelated", content="No links."),
            ]
        )
        changes, updated, error = await apply_incoming_wiki_content_rewrite(
            _Context(),
            service,
            "kb-1",
            ["entity/other", "entity/unrelated"],
            lambda content: (content.replace("acme", "corp"), "acme" in content),
        )
        assert error == ""
        assert updated == ["entity/other"]
        assert len(changes) == 1
        assert service.stored()[0].content == "See [[entity/corp|Acme]]."

    async def test_rollback_restores_original_content(self) -> None:
        service = _FakeWikiService([_page("entity/other", content="See [[entity/acme]].")])
        changes, _, error = await apply_incoming_wiki_content_rewrite(
            _Context(),
            service,
            "kb-1",
            ["entity/other"],
            lambda content: (content.replace("[[entity/acme]]", "[[entity/corp]]"), True),
        )
        assert error == ""
        assert service.stored()[0].content == "See [[entity/corp]]."
        rollback_error = await rollback_wiki_content_changes(_Context(), service, changes)
        assert rollback_error == ""
        assert service.stored()[0].content == "See [[entity/acme]]."


# ── wiki_search ───────────────────────────────────────────────────────


class TestWikiSearchTool:
    def _tool(
        self,
        service: _FakeWikiService,
        scopes: list[WikiScope],
        *,
        tag_fetcher: _FakeTagFetcher | None = None,
    ) -> WikiSearchTool:
        return WikiSearchTool(
            definition=build_wiki_search_definition(),
            wiki_service=service,
            scopes=scopes,
            tag_fetcher=tag_fetcher,
        )

    async def test_missing_queries_fails(self) -> None:
        tool = self._tool(_FakeWikiService(), [_scope("kb-1")])
        result = await tool.execute(_Context(), "{}")
        assert not result.success
        assert "Missing 'queries'" in result.error

    async def test_knowledge_base_id_outside_scope_fails(self) -> None:
        tool = self._tool(_FakeWikiService(), [_scope("kb-1")])
        result = await tool.execute(_Context(), '{"queries": ["acme"], "knowledge_base_id": "kb-9"}')
        assert not result.success
        assert "not within the current wiki scope" in result.error

    async def test_renders_matching_pages(self) -> None:
        service = _FakeWikiService(
            [
                _page(
                    "entity/acme",
                    content="stardust engine powers the starship",
                    summary="Acme corp",
                )
            ]
        )
        tool = self._tool(service, [_scope("kb-1")])
        result = await tool.execute(_Context(), '{"queries": ["stardust|skyvault"]}')
        assert result.success
        assert 'query="stardust|skyvault"' in result.output
        assert "[[entity/acme|Acme]]" in result.output
        assert result.data["found_kbs"] == {"entity/acme": ["kb-1"]}

    async def test_no_hits_renders_empty_block(self) -> None:
        tool = self._tool(_FakeWikiService(), [_scope("kb-1")])
        result = await tool.execute(_Context(), '{"queries": ["nothing-matches"]}')
        assert result.success
        assert '<search_results count="0"' in result.output

    async def test_seen_slug_omits_summary(self) -> None:
        service = _FakeWikiService([_page("entity/acme", summary="summary text")])
        tool = self._tool(service, [_scope("kb-1")])
        await tool.execute(_Context(), '{"queries": ["acme"]}')
        result = await tool.execute(_Context(), '{"queries": ["acme"]}')
        assert result.success
        assert "(summary omitted, already seen in previous search)" in result.output

    async def test_tag_scope_filters_results(self) -> None:
        service = _FakeWikiService(
            [
                _page("entity/acme", source_refs=["d1"], content="acme"),
                _page("entity/other", source_refs=["d2"], content="acme"),
            ]
        )
        fetcher = _FakeTagFetcher({"d1": [_tag("t1")], "d2": []})
        tool = self._tool(service, [_scope("kb-1", tag_ids=["t1"])], tag_fetcher=fetcher)
        result = await tool.execute(_Context(), '{"queries": ["acme"]}')
        assert result.success
        assert "[[entity/acme|Acme]]" in result.output
        assert "[[entity/other|Other]]" not in result.output


class TestSnippetExtraction:
    def test_surrounds_match_with_context(self) -> None:
        content = "aaa " * 30 + "needle" + " bbb " * 30
        snippet = extract_snippet(content, "needle")
        assert "needle" in snippet
        assert snippet.startswith("... ")
        assert snippet.endswith(" ...")

    def test_invalid_regex_returns_empty(self) -> None:
        assert extract_snippet("content", "(") == ""

    def test_no_match_returns_empty(self) -> None:
        assert extract_snippet("content", "missing") == ""


# ── wiki_read_page ────────────────────────────────────────────────────


class TestRenderIndexOverview:
    def _index_response(self) -> WikiIndexResponse:
        return WikiIndexResponse(
            intro="# Wiki\n\nIntro text.\n\n## Legacy directory\n",
            version=1,
            groups=[
                WikiIndexGroup(
                    type=WIKI_PAGE_TYPE_ENTITY,
                    total=3,
                    items=[
                        WikiIndexEntry(
                            slug="entity/acme", title="Acme Corp", summary="A fictional company."
                        )
                    ],
                )
            ],
        )

    def test_clips_legacy_directory_and_renders_top_k(self) -> None:
        rendered = render_index_overview_for_agent(self._index_response())
        assert "## Legacy directory" not in rendered
        assert "## Entity (3 total, showing top 1)" in rendered
        assert "[[entity/acme|Acme Corp]] — A fictional company." in rendered


class TestBudgetRendering:
    def test_fits_budget_untouched(self) -> None:
        pages = [
            _pending("entity/acme", body="x" * 100),
            _pending("entity/corp", body="y" * 100),
        ]
        output, truncated, omitted = render_wiki_pages_within_budget(pages, 100_000)
        assert truncated == []
        assert omitted == []
        assert "<wiki_page>" in output

    def test_trims_bodies_under_tight_budget(self) -> None:
        pages = [_pending("entity/acme", body="x" * 5000)]
        output, truncated, omitted = render_wiki_pages_within_budget(pages, 3000)
        assert truncated == ["entity/acme"]
        assert omitted == []
        assert "(body omitted: output budget exhausted)" in output or "..." in output

    def test_drops_trailing_pages_when_min_body_does_not_fit(self) -> None:
        pages = [_pending("entity/acme", body="x" * 5000), _pending("entity/corp", body="y" * 5000)]
        output, _truncated, omitted = render_wiki_pages_within_budget(pages, 2000)
        assert omitted
        assert "entity/acme" in output


def _pending(slug: str, *, body: str) -> object:
    """Build a pending page carrier for budget rendering."""
    from src.core.agents.tools.wiki_read import _PendingWikiPage

    return _PendingWikiPage(
        page=_page(slug, content=body),
        kb_id="kb-1",
        out_links=[],
        in_links=[],
        sources=[],
        body=body,
    )


class TestWikiReadPageTool:
    def _tool(
        self,
        service: _FakeWikiService,
        scopes: list[WikiScope],
        *,
        routes: WikiRouteResolver | None = None,
        tag_fetcher: _FakeTagFetcher | None = None,
    ) -> WikiReadPageTool:
        return WikiReadPageTool(
            definition=build_wiki_read_page_definition(),
            wiki_service=service,
            scopes=scopes,
            routes=routes,
            tag_fetcher=tag_fetcher,
        )

    async def test_missing_slugs_fails(self) -> None:
        tool = self._tool(_FakeWikiService(), [_scope("kb-1")])
        result = await tool.execute(_Context(), "{}")
        assert not result.success
        assert "Missing 'slugs'" in result.error

    async def test_reads_single_page(self) -> None:
        service = _FakeWikiService(
            [
                _page(
                    "entity/acme",
                    content="# Acme\n\nBody text.",
                    summary="A company.",
                    source_refs=["d1|Doc"],
                    out_links=["concept/rag"],
                )
            ]
        )
        tool = self._tool(service, [_scope("kb-1")])
        result = await tool.execute(_Context(), '{"slugs": ["entity/acme"]}')
        assert result.success
        assert "<wiki_page>" in result.output
        assert "<link>[[entity/acme|Acme]]</link>" in result.output
        assert '<source knowledge_id="d1">Doc</source>' in result.output
        assert result.data["found_kbs"] == {
            "entity/acme": ["kb-1"],
            "concept/rag": ["kb-1"],
        }
        assert result.data["ambiguous_slugs"] == {}

    async def test_renders_ambiguous_slug_from_both_kbs(self) -> None:
        service = _FakeWikiService(
            [
                _page("entity/acme", knowledge_base_id="kb-1", title="Acme One"),
                _page("entity/acme", knowledge_base_id="kb-2", title="Acme Two"),
            ]
        )
        tool = self._tool(service, [_scope("kb-1"), _scope("kb-2")])
        result = await tool.execute(_Context(), '{"slug": "entity/acme"}')
        assert result.success
        assert "Acme One" in result.output
        assert "Acme Two" in result.output
        assert result.data["ambiguous_slugs"] == {"entity/acme": ["kb-1", "kb-2"]}

    async def test_unknown_slug_reports_error(self) -> None:
        tool = self._tool(_FakeWikiService(), [_scope("kb-1")])
        result = await tool.execute(_Context(), '{"slugs": ["entity/ghost"]}')
        assert not result.success
        assert "entity/ghost" in result.error

    async def test_scope_filtered_slug_reports_clear_error(self) -> None:
        service = _FakeWikiService([_page("entity/acme", source_refs=["d1"])])
        tool = self._tool(service, [_scope("kb-1", knowledge_ids=["d2"])])
        result = await tool.execute(_Context(), '{"slugs": ["entity/acme"]}')
        assert not result.success
        assert "source documents are within the scope" in result.error

    async def test_index_page_surfaces_overview(self) -> None:
        service = _FakeWikiService(
            [_page("index", page_type=WIKI_PAGE_TYPE_INDEX, content="# Intro")]
        )
        service.index_views["kb-1"] = WikiIndexResponse(
            intro="# Intro",
            version=1,
            groups=[
                WikiIndexGroup(
                    type=WIKI_PAGE_TYPE_ENTITY,
                    total=1,
                    items=[WikiIndexEntry(slug="entity/acme", title="Acme", summary="Corp")],
                )
            ],
        )
        tool = self._tool(service, [_scope("kb-1")])
        result = await tool.execute(_Context(), '{"slugs": ["index"]}')
        assert result.success
        assert "## Entity (1)" in result.output
        assert "[[entity/acme|Acme]] — Corp" in result.output


# ── wiki_read_source_doc ──────────────────────────────────────────────


class TestWikiReadSourceDocTool:
    def _tool(
        self,
        store: _FakeChunkStore,
        docs: list[Knowledge] | None = None,
        *,
        search_targets: SearchTargets | None = None,
        knowledge_service: _FakeKnowledgeLookup | None = None,
    ) -> WikiReadSourceDocTool:
        return WikiReadSourceDocTool(
            definition=build_wiki_read_source_doc_definition(),
            chunk_store=store,
            knowledge_service=knowledge_service or _FakeKnowledgeLookup(docs),
            search_targets=search_targets,
        )

    async def test_missing_knowledge_id_fails(self) -> None:
        tool = self._tool(_FakeChunkStore(), [_knowledge()])
        result = await tool.execute(_Context(), "{}")
        assert not result.success
        assert "knowledge_id is required" in result.error

    async def test_document_not_found_fails(self) -> None:
        tool = self._tool(_FakeChunkStore(), [_knowledge(id="d1")])
        result = await tool.execute(_Context(), '{"knowledge_id": "d-unknown"}')
        assert not result.success
        assert "Document not found" in result.error

    async def test_reads_beginning_as_preview(self) -> None:
        store = _FakeChunkStore(
            [_chunk(0, "chunk zero"), _chunk(1, "chunk one"), _chunk(2, "chunk two")]
        )
        tool = self._tool(store, [_knowledge()])
        result = await tool.execute(_Context(), '{"knowledge_id": "d1"}')
        assert result.success
        assert "<total_chunks>3</total_chunks>" in result.output
        assert 'Showing the first 10 chunks as a preview' in result.output
        assert result.data["fetched_chunks"] == 3
        assert result.data["total_chunks"] == 3

    async def test_range_reader_clamps_window(self) -> None:
        store = _FakeChunkStore([_chunk(i, f"chunk {i}") for i in range(30)])
        tool = self._tool(store, [_knowledge()])
        result = await tool.execute(
            _Context(), '{"knowledge_id": "d1", "start_chunk_index": 3, "end_chunk_index": 4}'
        )
        assert result.success
        assert '<chunk_range start="3" end="4"/>' in result.output
        assert '<chunk index="3" type="range">' in result.output
        assert '<chunk index="5"' not in result.output

    async def test_regex_query_includes_context_before_and_after(self) -> None:
        store = _FakeChunkStore(
            [
                _chunk(0, "alpha before"),
                _chunk(1, "the target phrase here"),
                _chunk(2, "beta after"),
            ]
        )
        tool = self._tool(store, [_knowledge()])
        result = await tool.execute(_Context(), '{"knowledge_id": "d1", "query": "target"}')
        assert result.success
        assert 'type="context_before"' in result.output
        assert 'type="match"' in result.output
        assert 'type="context_after"' in result.output

    async def test_invalid_regex_fails(self) -> None:
        tool = self._tool(_FakeChunkStore(), [_knowledge()])
        result = await tool.execute(_Context(), '{"knowledge_id": "d1", "query": "("}')
        assert not result.success
        assert "Invalid regex query" in result.error

    async def test_out_of_scope_document_rejected_when_enforced(self) -> None:
        store = _FakeChunkStore()
        targets = _targets(_kb_target("kb-1"))
        docs = [_knowledge(id="d1", knowledge_base_id="kb-2")]
        tool = self._tool(store, docs, search_targets=targets, knowledge_service=_FakeKnowledgeLookup(docs))
        result = await tool.execute(_Context(), '{"knowledge_id": "d1"}')
        assert not result.success
        assert "Document not found" in result.error


# ── wiki_write_page ───────────────────────────────────────────────────


class TestWikiWritePageTool:
    def _tool(
        self,
        service: _FakeWikiService,
        kb_ids: list[str] | None = None,
        *,
        knowledge_service: _FakeKnowledgeLookup | None = None,
        search_targets: SearchTargets | None = None,
        kb_loader: _FakeKbLoader | None = None,
    ) -> WikiWritePageTool:
        return WikiWritePageTool(
            definition=build_wiki_write_page_definition(),
            wiki_service=service,
            kb_ids=kb_ids if kb_ids is not None else ["kb-1"],
            knowledge_service=knowledge_service,
            search_targets=search_targets,
            kb_loader=kb_loader,
        )

    def _args(self, slug: str = "entity/acme", **overrides: object) -> str:
        payload: dict[str, object] = {
            "slug": slug,
            "title": "Acme",
            "summary": "A company.",
            "content": "# Acme\n\nBody.",
            "page_type": WIKI_PAGE_TYPE_ENTITY,
        }
        payload.update(overrides)
        return json.dumps(payload)

    async def test_no_kbs_available_fails(self) -> None:
        tool = self._tool(_FakeWikiService(), kb_ids=[])
        result = await tool.execute(_Context(), self._args())
        assert not result.success
        assert "No knowledge bases available" in result.error

    async def test_missing_required_fields_fails(self) -> None:
        tool = self._tool(_FakeWikiService())
        result = await tool.execute(_Context(), self._args(content=""))
        assert not result.success
        assert "required for write action" in result.error

    async def test_invalid_slug_fails(self) -> None:
        tool = self._tool(_FakeWikiService())
        result = await tool.execute(_Context(), self._args(slug="entity/acme!"))
        assert not result.success
        assert "not allowed" in result.error

    async def test_creates_new_page(self) -> None:
        service = _FakeWikiService()
        loader = _FakeKbLoader([_kb_info("kb-1", tenant_id=7)])
        tool = self._tool(service, kb_loader=loader)
        result = await tool.execute(_Context(), self._args())
        assert result.success
        assert "Successfully created page [[entity/acme]]." in result.output
        assert service.created[0][1] == "agent"
        assert service.created[0][0].tenant_id == 7
        assert service.injected == [("kb-1", ["entity/acme"])]
        assert service.rebuilt == ["kb-1"]

    async def test_updates_existing_page(self) -> None:
        service = _FakeWikiService([_page("entity/acme", title="Old")])
        tool = self._tool(service)
        result = await tool.execute(_Context(), self._args(title="New Title"))
        assert result.success
        assert "Successfully updated page [[entity/acme]]." in result.output
        assert service.updated[0][1] == "agent"
        assert service.updated[0][0].title == "New Title"

    async def test_summary_namespace_create_is_rejected(self) -> None:
        service = _FakeWikiService()
        tool = self._tool(service)
        result = await tool.execute(_Context(), self._args(slug=f"{WIKI_PAGE_TYPE_SUMMARY}/abc"))
        assert not result.success
        assert "generated automatically" in result.error

    async def test_scope_enforced_source_refs_rejected(self) -> None:
        service = _FakeWikiService()
        docs = [_knowledge(id="d1", knowledge_base_id="kb-2")]
        tool = self._tool(
            service,
            knowledge_service=_FakeKnowledgeLookup(docs),
            search_targets=_targets(_kb_target("kb-1")),
        )
        result = await tool.execute(_Context(), self._args(source_refs=["d1"]))
        assert not result.success
        assert "Invalid source_refs" in result.error


# ── wiki_delete_page ──────────────────────────────────────────────────


class TestWikiDeletePageTool:
    def _tool(self, service: _FakeWikiService) -> WikiDeletePageTool:
        return WikiDeletePageTool(
            definition=build_wiki_delete_page_definition(),
            wiki_service=service,
            kb_ids=["kb-1"],
        )

    async def test_deletes_and_cleans_incoming_links(self) -> None:
        service = _FakeWikiService(
            [
                _page("entity/acme", title="Acme", in_links=["entity/other"]),
                _page("entity/other", content="See [[entity/acme|Acme Corp]]."),
            ]
        )
        tool = self._tool(service)
        result = await tool.execute(_Context(), '{"slug": "entity/acme"}')
        assert result.success
        assert "Successfully deleted page [[entity/acme]] and cleaned up 1 incoming links." in result.output
        assert service.deleted == [("kb-1", "entity/acme")]
        assert service.stored()[0].content == "See Acme Corp."

    async def test_rolls_back_when_cleanup_fails(self) -> None:
        service = _FakeWikiService([_page("entity/acme", in_links=["entity/other"])])

        async def fail_auto(*, page: WikiPage) -> WikiPage:  # pragma: no cover
            raise RuntimeError("boom")

        service.update_auto_linked_content = fail_auto  # type: ignore[method-assign]
        tool = self._tool(service)
        result = await tool.execute(_Context(), '{"slug": "entity/acme"}')
        assert not result.success
        assert "Delete aborted while cleaning incoming links" in result.error


# ── wiki_rename_page ──────────────────────────────────────────────────


class TestWikiRenamePageTool:
    def _tool(self, service: _FakeWikiService) -> WikiRenamePageTool:
        return WikiRenamePageTool(
            definition=build_wiki_rename_page_definition(),
            wiki_service=service,
            kb_ids=["kb-1"],
        )

    async def test_renames_and_cascades_links(self) -> None:
        service = _FakeWikiService(
            [
                _page("entity/acme", title="Acme", in_links=["entity/other"]),
                _page("entity/other", content="See [[entity/acme|Acme]]."),
            ]
        )
        tool = self._tool(service)
        result = await tool.execute(_Context(), '{"slug": "entity/acme", "new_slug": "entity/acme-corp"}')
        assert result.success
        assert "Successfully renamed page [[entity/acme]] → [[entity/acme-corp]]" in result.output
        slugs = {page.slug for page in service.stored()}
        assert slugs == {"entity/acme-corp", "entity/other"}
        assert service.stored()[0].content == "See [[entity/acme-corp|Acme]]."
        assert service.deleted == [("kb-1", "entity/acme")]
        assert service.injected == [("kb-1", ["entity/acme-corp"])]

    async def test_same_slug_rejected(self) -> None:
        tool = self._tool(_FakeWikiService())
        result = await tool.execute(_Context(), '{"slug": "entity/acme", "new_slug": "entity/acme"}')
        assert not result.success
        assert "new_slug must be different" in result.error


# ── wiki_replace_text ─────────────────────────────────────────────────


class TestWikiReplaceTextTool:
    def _tool(self, service: _FakeWikiService) -> WikiReplaceTextTool:
        return WikiReplaceTextTool(
            definition=build_wiki_replace_text_definition(),
            wiki_service=service,
            kb_ids=["kb-1"],
        )

    async def test_replaces_first_occurrence(self) -> None:
        service = _FakeWikiService([_page("entity/acme", content="old text then old text")])
        tool = self._tool(service)
        result = await tool.execute(
            _Context(), '{"slug": "entity/acme", "old_text": "old", "new_text": "new"}'
        )
        assert result.success
        assert "Successfully replaced text on page [[entity/acme]]." in result.output
        assert service.updated[0][0].content == "new text then old text"

    async def test_old_text_missing_fails(self) -> None:
        service = _FakeWikiService([_page("entity/acme", content="hello world")])
        tool = self._tool(service)
        result = await tool.execute(
            _Context(), '{"slug": "entity/acme", "old_text": "missing", "new_text": "x"}'
        )
        assert not result.success
        assert "old_text not found" in result.error


# ── query_knowledge_graph ─────────────────────────────────────────────


class TestGraphConfigHelpers:
    def test_summarize_dedupes_and_sorts(self) -> None:
        summary = summarize_graph_config(
            {
                "nodes": [{"name": "Tech"}, {"name": "Tech"}, {"name": "Tool"}],
                "relations": [{"type": "uses"}, {"type": "depends_on"}],
            }
        )
        assert summary.nodes == ["Tech", "Tool"]
        assert summary.relations == ["depends_on", "uses"]

    def test_aggregate_merges_across_kbs(self) -> None:
        configs = {
            "kb-1": summarize_graph_config({"nodes": ["Tech"], "relations": ["uses"]}),
            "kb-2": summarize_graph_config({"nodes": ["Tech", "Tool"], "relations": ["uses"]}),
        }
        merged = aggregate_graph_config(configs)
        assert merged["nodes"] == ["Tech", "Tool"]
        assert merged["relations"] == ["uses"]

    def test_graph_configs_to_data(self) -> None:
        configs = {"kb-1": summarize_graph_config({"nodes": ["Tech"], "relations": []})}
        data = graph_configs_to_data(configs)
        assert data == {"kb-1": {"nodes": ["Tech"], "relations": []}}

    def test_build_visualization_dedupes_by_id(self) -> None:
        results = [_result("c1"), _result("c1", score=0.5)]
        payload = build_graph_visualization_data(results)
        assert payload["total_nodes"] == 1
        assert payload["total_edges"] == 0


class TestQueryKnowledgeGraphTool:
    def _tool(
        self,
        loader: _FakeKbLoader,
        runner: _FakeSearchRunner,
        *,
        search_targets: SearchTargets | None = None,
    ) -> QueryKnowledgeGraphTool:
        return QueryKnowledgeGraphTool(
            definition=build_wiki_graph_definition(),
            kb_loader=loader,
            search_runner=runner,
            search_targets=search_targets,
        )

    async def test_missing_kb_ids_fails(self) -> None:
        tool = self._tool(_FakeKbLoader(), _FakeSearchRunner())
        result = await tool.execute(_Context(), '{"query": "rag"}')
        assert not result.success
        assert "knowledge_base_ids is required" in result.error

    async def test_too_many_kb_ids_fails(self) -> None:
        kb_ids = [f"kb-{i}" for i in range(GRAPH_MAX_KB_IDS + 1)]
        tool = self._tool(_FakeKbLoader(), _FakeSearchRunner())
        result = await tool.execute(_Context(), json.dumps({"query": "rag", "knowledge_base_ids": kb_ids}))
        assert not result.success
        assert "at most 10" in result.error

    async def test_missing_query_fails(self) -> None:
        tool = self._tool(_FakeKbLoader(), _FakeSearchRunner())
        result = await tool.execute(_Context(), '{"knowledge_base_ids": ["kb-1"]}')
        assert not result.success
        assert "query is required" in result.error

    async def test_deduplicates_and_sorts_by_score(self) -> None:
        loader = _FakeKbLoader(
            [
                _kb_info("kb-1", extract_config={"nodes": ["Tech"], "relations": ["uses"]}),
                _kb_info("kb-2", extract_config={"nodes": ["Tech"], "relations": ["uses"]}),
            ]
        )
        runner = _FakeSearchRunner(
            per_kb={
                "kb-1": [_result("c1", score=0.5, knowledge_base_id="kb-1")],
                "kb-2": [
                    _result("c1", score=0.9, knowledge_base_id="kb-2"),
                    _result("c2", score=0.7, knowledge_base_id="kb-2"),
                ],
            }
        )
        tool = self._tool(loader, runner)
        result = await tool.execute(_Context(), '{"knowledge_base_ids": ["kb-1", "kb-2"], "query": "rag"}')
        assert result.success
        assert result.data["count"] == 2
        assert result.data["kb_counts"] == {"kb-1": 1, "kb-2": 2}
        ordered = [item["chunk_id"] for item in result.data["results"]]  # type: ignore[index]
        assert ordered == ["c2", "c1"]

    async def test_kb_without_graph_config_reports_error(self) -> None:
        loader = _FakeKbLoader([_kb_info("kb-1", extract_config=None)])
        tool = self._tool(loader, _FakeSearchRunner())
        result = await tool.execute(_Context(), '{"knowledge_base_ids": ["kb-1"], "query": "rag"}')
        assert result.success
        assert result.output == "No relevant graph information found."
        assert result.data["errors"] == ["KB kb-1: graph extraction not configured"]

    async def test_scope_enforced_kb_outside_targets_rejected(self) -> None:
        loader = _FakeKbLoader([_kb_info("kb-9", extract_config={"nodes": ["Tech"], "relations": []})])
        tool = self._tool(loader, _FakeSearchRunner(), search_targets=_targets(_kb_target("kb-1")))
        result = await tool.execute(_Context(), '{"knowledge_base_ids": ["kb-9"], "query": "rag"}')
        assert not result.success
        assert "not within the current Agent scope" in result.error


# ── wiki issue tools ──────────────────────────────────────────────────


class TestWikiFlagIssueTool:
    def _tool(
        self,
        service: _FakeWikiService,
        repo: _FakeIssueRepo,
        *,
        knowledge_service: _FakeKnowledgeLookup | None = None,
        search_targets: SearchTargets | None = None,
    ) -> WikiFlagIssueTool:
        return WikiFlagIssueTool(
            definition=build_wiki_flag_issue_definition(),
            wiki_service=service,
            kb_ids=["kb-1"],
            issue_repo=repo,
            knowledge_service=knowledge_service,
            search_targets=search_targets,
        )

    async def test_flags_issue_for_resolved_page(self) -> None:
        service = _FakeWikiService([_page("entity/acme", tenant_id=7)])
        repo = _FakeIssueRepo()
        tool = self._tool(service, repo)
        result = await tool.execute(
            _Context(),
            '{"slug": "entity/acme", "issue_type": "contradictory_facts", "description": "conflict"}',
        )
        assert result.success
        assert "Successfully flagged issue" in result.output
        assert repo.created[0].slug == "entity/acme"
        assert repo.created[0].status == WIKI_ISSUE_STATUS_PENDING
        assert repo.created[0].tenant_id == 7

    async def test_unknown_page_fails(self) -> None:
        tool = self._tool(_FakeWikiService(), _FakeIssueRepo())
        result = await tool.execute(
            _Context(),
            '{"slug": "entity/ghost", "issue_type": "other", "description": "d"}',
        )
        assert not result.success


class TestWikiReadIssueTool:
    def _tool(self, repo: _FakeIssueRepo, kb_ids: list[str] | None = None) -> WikiReadIssueTool:
        return WikiReadIssueTool(
            definition=build_wiki_read_issue_definition(),
            issue_repo=repo,
            kb_ids=kb_ids if kb_ids is not None else ["kb-1"],
        )

    async def test_neither_issue_id_nor_slug_fails(self) -> None:
        tool = self._tool(_FakeIssueRepo())
        result = await tool.execute(_Context(), "{}")
        assert not result.success
        assert "Either issue_id or slug is required" in result.error

    async def test_reads_issue_by_id(self) -> None:
        repo = _FakeIssueRepo([_issue("i1")])
        tool = self._tool(repo)
        result = await tool.execute(_Context(), '{"issue_id": "i1"}')
        assert result.success
        assert '"issue_id": "i1"' in result.output or '"id": "i1"' in result.output

    async def test_lists_pending_issues_for_slug(self) -> None:
        repo = _FakeIssueRepo(
            [
                _issue("i1", slug="entity/acme"),
                _issue("i2", slug="entity/acme", status="resolved"),
            ]
        )
        tool = self._tool(repo)
        result = await tool.execute(_Context(), '{"slug": "entity/acme"}')
        assert result.success
        assert "i1" in result.output
        assert "i2" not in result.output

    async def test_no_pending_issues_message(self) -> None:
        tool = self._tool(_FakeIssueRepo())
        result = await tool.execute(_Context(), '{"slug": "entity/acme"}')
        assert result.success
        assert "No pending issues found" in result.output


class TestWikiUpdateIssueTool:
    def _tool(self, repo: _FakeIssueRepo, kb_ids: list[str] | None = None) -> WikiUpdateIssueTool:
        return WikiUpdateIssueTool(
            definition=build_wiki_update_issue_definition(),
            issue_repo=repo,
            kb_ids=kb_ids if kb_ids is not None else ["kb-1"],
        )

    async def test_updates_status_after_resolution(self) -> None:
        repo = _FakeIssueRepo([_issue("i1")])
        tool = self._tool(repo)
        result = await tool.execute(_Context(), '{"issue_id": "i1", "status": "resolved"}')
        assert result.success
        assert "Successfully updated issue i1 to status 'resolved'" in result.output
        assert repo.status_updates == [("i1", WIKI_ISSUE_STATUS_RESOLVED)]

    async def test_out_of_scope_issue_rejected(self) -> None:
        repo = _FakeIssueRepo([_issue("i1", knowledge_base_id="kb-2")])
        tool = self._tool(repo)
        result = await tool.execute(_Context(), '{"issue_id": "i1", "status": "resolved"}')
        assert not result.success
        assert "not within the current Wiki scope" in result.error


# ── Integration tests (real applied schema) ───────────────────────────


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Per-test session against the real applied schema (no cleanup)."""
    reset_settings_cache()
    settings = get_settings()
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as s:
            yield s
    finally:
        await engine.dispose()


def _integration_doc(
    *, id: str, tenant_id: int, knowledge_base_id: str
) -> Document:
    """Build one document row for the integration tests."""
    return Document(
        id=id,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        type="file",
        title="Q3 budget",
        description="the budget",
        source="budget-2026.pdf",
        channel=CHANNEL_WEB,
        parse_status=PARSE_STATUS_COMPLETED,
        summary_status="none",
        enable_status="enabled",
        embedding_model_id="em-1",
        file_name="budget-2026.pdf",
        file_type="pdf",
        file_size=1024,
        storage_size=2048,
        metadata={"owner": "finance"},
        custom_metadata={"scope": "2026"},
        created_at=_NOW,
        updated_at=_NOW,
    )


def _integration_chunk(
    *,
    id: str,
    tenant_id: int,
    knowledge_base_id: str,
    knowledge_id: str,
    chunk_index: int,
    content: str,
) -> Chunk:
    """Build one chunk row for the integration tests."""
    return Chunk(
        id=id,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        knowledge_id=knowledge_id,
        content=content,
        chunk_index=chunk_index,
        is_enabled=True,
        start_at=0,
        end_at=len(content),
        pre_chunk_id=None,
        next_chunk_id=None,
        chunk_type="text",
        parent_chunk_id=None,
        image_info=None,
        relation_chunks=None,
        indirect_relation_chunks=None,
        metadata={"source": "manual"},
        tag_id=None,
        status=1,
        content_hash=None,
        flags=1,
        seq_id=0,
        source_content="",
        content_revision=0,
        index_status="ready",
        last_editor_id="",
        context_header="",
        created_at=_NOW,
        updated_at=_NOW,
        deleted_at=None,
    )


def _integration_page(
    *,
    tenant_id: int,
    knowledge_base_id: str,
    slug: str,
    title: str = "Acme Corp",
    content: str = "Acme is a fictional company.",
) -> WikiPage:
    """Build one wiki page row for the integration tests."""
    return WikiPage(
        id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        slug=slug,
        title=title,
        page_type=WIKI_PAGE_TYPE_ENTITY,
        status="published",
        content=content,
        summary="Fictional company.",
        parent_slug="",
        folder_id="",
        category_path=[],
        wiki_path=slug,
        depth=1,
        sort_order=0,
        source_refs=[],
        chunk_refs=[],
        in_links=[],
        out_links=[],
        page_metadata={},
        aliases=[],
        version=1,
        last_edit_source="pipeline",
        last_editor_id="",
        created_at=_NOW,
        updated_at=_NOW,
        deleted_at=None,
    )


async def _seed_kb(session: AsyncSession, tenant_id: int, name: str = "agent-wiki-kb") -> KnowledgeBaseInfo:
    """Create one knowledge base and return its info record."""
    kb_service = KBService(kb_repo=KnowledgeBaseRepository(session))
    return await kb_service.create_knowledge_base(
        tenant_id=tenant_id, name=name, kb_type=KNOWLEDGE_BASE_TYPE_DOCUMENT
    )


def _real_wiki_service(session: AsyncSession) -> WikiPageService:
    return WikiPageService(
        page_repo=WikiPageRepository(session),
        folder_repo=WikiFolderRepository(session),
    )


async def test_integration_wiki_search_and_read_real_rows(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    kb = await _seed_kb(session, tenant_id)
    wiki = _real_wiki_service(session)
    await wiki.create_page(
        page=_integration_page(
            tenant_id=tenant_id,
            knowledge_base_id=kb.id,
            slug="entity/acme",
            content="stardust engine powers the starship",
        )
    )

    search_tool = WikiSearchTool(
        definition=build_wiki_search_definition(),
        wiki_service=wiki,
        scopes=new_wiki_scopes_from_kb_ids([kb.id]),
    )
    search_result = await search_tool.execute(_Context(), '{"queries": ["stardust|skyvault"]}')
    assert search_result.success
    assert "[[entity/acme|Acme Corp]]" in search_result.output

    read_tool = WikiReadPageTool(
        definition=build_wiki_read_page_definition(),
        wiki_service=wiki,
        scopes=new_wiki_scopes_from_kb_ids([kb.id]),
    )
    read_result = await read_tool.execute(_Context(), '{"slugs": ["entity/acme"]}')
    assert read_result.success
    assert "<wiki_page>" in read_result.output
    assert "stardust engine powers the starship" in read_result.output


async def test_integration_wiki_write_update_delete_cycle(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    kb = await _seed_kb(session, tenant_id)
    wiki = _real_wiki_service(session)
    kb_loader = _FakeKbLoader([_kb_info(kb.id, tenant_id=tenant_id)])

    write_tool = WikiWritePageTool(
        definition=build_wiki_write_page_definition(),
        wiki_service=wiki,
        kb_ids=[kb.id],
        kb_loader=kb_loader,
    )
    created = await write_tool.execute(
        _Context(),
        json.dumps(
            {
                "slug": "entity/acme",
                "title": "Acme",
                "summary": "A company.",
                "content": "# Acme\n\nOriginal body.",
                "page_type": WIKI_PAGE_TYPE_ENTITY,
            }
        ),
    )
    assert created.success
    assert created.data["action"] == "created"

    replace_tool = WikiReplaceTextTool(
        definition=build_wiki_replace_text_definition(),
        wiki_service=wiki,
        kb_ids=[kb.id],
    )
    replaced = await replace_tool.execute(
        _Context(),
        '{"slug": "entity/acme", "old_text": "Original", "new_text": "Updated"}',
    )
    assert replaced.success
    page = await wiki.get_page_by_slug(knowledge_base_id=kb.id, slug="entity/acme")
    assert "Updated body" in page.content

    delete_tool = WikiDeletePageTool(
        definition=build_wiki_delete_page_definition(),
        wiki_service=wiki,
        kb_ids=[kb.id],
    )
    deleted = await delete_tool.execute(_Context(), '{"slug": "entity/acme"}')
    assert deleted.success
    with pytest.raises(NotFoundError):
        await wiki.get_page_by_slug(knowledge_base_id=kb.id, slug="entity/acme")


async def test_integration_wiki_read_source_doc_real_chunks(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    kb = await _seed_kb(session, tenant_id)
    doc = await KnowledgeRepository(session).create(
        _integration_doc(id=str(uuid.uuid4()), tenant_id=tenant_id, knowledge_base_id=kb.id)
    )
    await ChunkRepository(session).create_many(
        [
            _integration_chunk(
                id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                knowledge_base_id=kb.id,
                knowledge_id=doc.id,
                chunk_index=index,
                content=f"chunk body {index}",
            )
            for index in range(3)
        ]
    )

    knowledge_service = KnowledgeService(knowledge_repo=KnowledgeRepository(session))
    tool = WikiReadSourceDocTool(
        definition=build_wiki_read_source_doc_definition(),
        chunk_store=SqlPagedChunkStore(session),
        knowledge_service=knowledge_service,
    )
    result = await tool.execute(_Context(), f'{{"knowledge_id": "{doc.id}"}}')
    assert result.success
    assert result.data["total_chunks"] == 3
    assert result.data["fetched_chunks"] == 3
    assert {chunk["content"] for chunk in result.data["chunks"]} == {  # type: ignore[index]
        "chunk body 0",
        "chunk body 1",
        "chunk body 2",
    }


async def test_integration_wiki_issue_lifecycle(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    kb = await _seed_kb(session, tenant_id)
    wiki = _real_wiki_service(session)
    await wiki.create_page(
        page=_integration_page(tenant_id=tenant_id, knowledge_base_id=kb.id, slug="entity/acme")
    )

    repo = _FakeIssueRepo()
    flag_tool = WikiFlagIssueTool(
        definition=build_wiki_flag_issue_definition(),
        wiki_service=wiki,
        kb_ids=[kb.id],
        issue_repo=repo,
    )
    flagged = await flag_tool.execute(
        _Context(),
        '{"slug": "entity/acme", "issue_type": "contradictory_facts", "description": "conflict"}',
    )
    assert flagged.success
    assert repo.created[0].knowledge_base_id == kb.id
    assert repo.created[0].tenant_id == tenant_id

    read_tool = WikiReadIssueTool(
        definition=build_wiki_read_issue_definition(),
        issue_repo=repo,
        kb_ids=[kb.id],
    )
    listed = await read_tool.execute(_Context(), '{"slug": "entity/acme"}')
    assert listed.success
    assert repo.created[0].id in listed.output

    update_tool = WikiUpdateIssueTool(
        definition=build_wiki_update_issue_definition(),
        issue_repo=repo,
        kb_ids=[kb.id],
    )
    updated = await update_tool.execute(
        _Context(), f'{{"issue_id": "{repo.created[0].id}", "status": "resolved"}}'
    )
    assert updated.success
    assert repo.status_updates == [(repo.created[0].id, WIKI_ISSUE_STATUS_RESOLVED)]
