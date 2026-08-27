"""Unit + integration tests for the chat-pipeline search steps.

Unit tests drive ``SearchStep``, ``ExtractEntityStep`` and the query-expansion
helpers through injected fakes — a scripted search runner, an in-memory
knowledge-base loader, a scripted chat client, and a fake web-search service —
so no test touches a vector store, a model API, or the network.

Integration tests run against the real applied schema (``chunks.tenant_id``
is INTEGER 32-bit, so they mint int32-safe ids from a local counter) and
drive one real chain: the ``SearchStep`` registered on an ``EventManager``,
running the real ``HybridSearchRunner`` over a real knowledge base + document
+ chunk, with the retrieval engine faked at the registry boundary. Requires a
reachable database — run with ``DATABASE_URL_OVERRIDE``.
"""

from __future__ import annotations

import itertools
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from random import randint
from typing import cast

import pytest
import pytest_asyncio
from faker import Faker
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.ai.embedding import Embedder
from src.ai.llm.types import ChatOptions, ChatResponse, Message
from src.ai.retrieval.base import RetrieveEngineService
from src.ai.retrieval.registry import new_retrieve_engine_registry
from src.ai.retrieval.types import (
    IndexWithScore,
    MatchType,
    RetrieveParams,
    RetrieverEngineParams,
    RetrieverEngineType,
    RetrieveResult,
    RetrieverType,
)
from src.common.json import JsonObject
from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.engine import ERR_SEARCH_NOTHING, EventManager, PluginError
from src.core.chat.pipeline.steps.extract_entity import ExtractEntityStep
from src.core.chat.pipeline.steps.query_expansion import (
    expand_queries,
    extract_keywords,
    extract_phrases,
    remove_question_words,
    split_by_delimiters,
    tokenize,
)
from src.core.chat.pipeline.steps.search import (
    HybridSearchRunner,
    KBServiceKbLoader,
    SearchCall,
    SearchStep,
    WebSearchHit,
    convert_web_search_results,
    effective_web_search_config,
    has_knowledge_retrieval_scope,
    recall_thresholds,
)
from src.core.chat.pipeline.types import (
    Context,
    EventType,
    SearchResult,
    SearchTarget,
    SearchTargetType,
)
from src.core.contracts.knowledge import Knowledge
from src.core.knowledge.chunks.types import (
    CHUNK_FLAG_RECOMMENDED,
    CHUNK_STATUS_STORED,
    CHUNK_TYPE_TEXT,
)
from src.core.knowledge.documents.types import CHANNEL_WEB, PARSE_STATUS_COMPLETED
from src.core.knowledge.knowledge_bases.hybrid_search import (
    ChunkRepositoryLoader,
    KBServiceKnowledgeBaseLoader,
    KnowledgeRepositoryLoader,
    RetrievalConfig,
    SearchDependencies,
)
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import (
    KNOWLEDGE_BASE_TYPE_DOCUMENT,
    KnowledgeBaseInfo,
)
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.chunk import Chunk
from src.db.models.knowledge import Document
from src.settings import get_settings, reset_settings_cache

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_FAKER_SEED_MAX = 100_000_000

#: ``chunks.tenant_id`` is INTEGER (32-bit); integration tests mint ids
#: from this counter so seeded rows never overflow.
_INT32_TENANT_BASE = 4_000_000
_INT32_TENANT_SEQ = itertools.count(start=1)

_ES = RetrieverEngineType.ELASTICSEARCH


def _int32_tenant_id() -> int:
    """Return a tenant id unique within the session, safe for INTEGER."""
    return _INT32_TENANT_BASE + next(_INT32_TENANT_SEQ)


@pytest.fixture(autouse=True)
def faker_seed() -> None:
    """Re-seed Faker per test for varied-but-reproducible generation."""
    Faker.seed(randint(1, _FAKER_SEED_MAX))


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


# ── Test doubles ───────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _FakeContext:
    """Opaque execution context satisfying the pipeline Context protocol."""

    is_background_task: bool = False


class _FakeRunner:
    """Scripted ``SearchRunner`` recording calls and returning canned hits."""

    def __init__(
        self,
        responses: dict[str, list[SearchResult]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._responses = responses or {}
        self._error = error
        self.calls: list[SearchCall] = []

    async def search(self, ctx: Context, call: SearchCall) -> list[SearchResult]:
        self.calls.append(call)
        if self._error is not None:
            raise self._error
        return self._responses.get(call.query_text, [])


class _FakeKbLoader:
    """In-memory ``KbLoader`` / ``KnowledgeBaseLoader``."""

    def __init__(
        self,
        kbs: list[KnowledgeBaseInfo] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._by_id = {kb.id: kb for kb in (kbs or [])}
        self._error = error

    async def load_by_ids(self, ids: list[str]) -> list[KnowledgeBaseInfo]:
        if self._error is not None:
            raise self._error
        return [self._by_id[id] for id in ids if id in self._by_id]


class _FakeEmbeddingProvider:
    """Scripted ``QueryEmbeddingProvider``."""

    def __init__(self, vector: tuple[float, ...] = (), error: Exception | None = None) -> None:
        self._vector = vector
        self._error = error
        self.calls: list[tuple[str, str]] = []

    async def get_query_embedding(self, ctx: Context, kb_id: str, query_text: str) -> list[float]:
        self.calls.append((kb_id, query_text))
        if self._error is not None:
            raise self._error
        return list(self._vector)


class _FakeWebSearch:
    """Scripted ``WebSearchService``."""

    def __init__(
        self,
        hits: list[WebSearchHit] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._hits = hits or []
        self._error = error
        self.calls: list[dict[str, object]] = []

    async def search(
        self,
        ctx: Context,
        *,
        tenant_id: int,
        provider_id: str,
        query: str,
        max_results: int,
        include_date: bool,
        blacklist: list[str],
        proxy_url: str,
    ) -> list[WebSearchHit]:
        self.calls.append(
            {
                "tenant_id": tenant_id,
                "provider_id": provider_id,
                "query": query,
                "max_results": max_results,
                "include_date": include_date,
                "blacklist": blacklist,
                "proxy_url": proxy_url,
            }
        )
        if self._error is not None:
            raise self._error
        return self._hits


class _FakeWebSearchConfig:
    """Scripted ``WebSearchConfigProvider``."""

    def __init__(self, config: JsonObject | None = None, error: Exception | None = None) -> None:
        self._config = config
        self._error = error
        self.calls: list[int] = []

    async def load(self, ctx: Context, tenant_id: int) -> JsonObject | None:
        self.calls.append(tenant_id)
        if self._error is not None:
            raise self._error
        return self._config


class _FakeChat:
    """Scripted chat seam: records calls and returns canned output."""

    def __init__(self, *, content: str = "", error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls: list[tuple[list[Message], ChatOptions | None]] = []

    async def chat(
        self,
        messages: list[Message],
        opts: ChatOptions | None = None,
    ) -> ChatResponse:
        if self.error is not None:
            raise self.error
        self.calls.append((messages, opts))
        return ChatResponse(content=self.content)

    def get_model_name(self) -> str:
        return "fake-chat"

    def get_model_id(self) -> str:
        return "fake-chat"


class _FakeModelProvider:
    """Scripted ``ChatModelProvider``."""

    def __init__(self, model: _FakeChat | None = None, error: Exception | None = None) -> None:
        self._model = model or _FakeChat()
        self._error = error
        self.calls: list[str] = []

    async def get_chat_model(self, ctx: Context, model_id: str) -> _FakeChat:
        self.calls.append(model_id)
        if self._error is not None:
            raise self._error
        return self._model


class _FakeKnowledgeLoader:
    """Scripted ``KnowledgeLoader`` for the extract step."""

    def __init__(
        self,
        documents: list[Knowledge] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._documents = documents or []
        self._error = error
        self.calls: list[tuple[int, list[str]]] = []

    async def load_documents(self, tenant_id: int, knowledge_ids: list[str]) -> list[Knowledge]:
        self.calls.append((tenant_id, knowledge_ids))
        if self._error is not None:
            raise self._error
        return self._documents


class _RecordingNext:
    """Records how many times the chain's ``next`` was resumed."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self) -> PluginError | None:
        self.calls += 1
        return None


def _kb(
    *,
    kb_id: str,
    tenant_id: int = 7,
    extract_enabled: bool | None = None,
    embedding_model_id: str = "em-1",
) -> KnowledgeBaseInfo:
    """Build an in-memory ``KnowledgeBaseInfo`` for seam fakes."""
    extract_config: JsonObject | None = None
    if extract_enabled is not None:
        extract_config = {"enabled": extract_enabled}
    return KnowledgeBaseInfo(
        id=kb_id,
        name=kb_id,
        tenant_id=tenant_id,
        type=KNOWLEDGE_BASE_TYPE_DOCUMENT,
        embedding_model_id=embedding_model_id,
        extract_config=extract_config,
        indexing_strategy={
            "vector_enabled": True,
            "keyword_enabled": True,
            "wiki_enabled": False,
            "graph_enabled": False,
        },
        created_at=_NOW,
        updated_at=_NOW,
    )


def _knowledge(*, knowledge_id: str, knowledge_base_id: str, tenant_id: int) -> Knowledge:
    """Build an in-memory ``Knowledge`` contract for the extract seam."""
    return Knowledge(
        id=knowledge_id,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        type="file",
        parse_status=PARSE_STATUS_COMPLETED,
        enable_status="enabled",
        created_at=_NOW,
        updated_at=_NOW,
    )


def _hit(*, chunk_id: str, knowledge_id: str = "kn-1", score: float = 0.9) -> SearchResult:
    """Build one pipeline search hit."""
    return SearchResult(
        id=chunk_id,
        content="payload",
        knowledge_id=knowledge_id,
        knowledge_base_id="kb-1",
        score=score,
    )


def _step(
    *,
    runner: _FakeRunner,
    kb_loader: _FakeKbLoader | None = None,
    embedding: _FakeEmbeddingProvider | None = None,
    web: _FakeWebSearch | None = None,
    web_config: _FakeWebSearchConfig | None = None,
) -> SearchStep:
    """Assemble a ``SearchStep`` over the fake seams."""
    return SearchStep(
        runner=runner,
        kb_loader=kb_loader or _FakeKbLoader(),
        query_embedding_provider=embedding,
        web_search=web,
        web_search_config_provider=web_config,
    )


def _extract_step(
    *,
    model_provider: _FakeModelProvider | None = None,
    kb_loader: _FakeKbLoader | None = None,
    knowledge_loader: _FakeKnowledgeLoader | None = None,
    graph_enabled: bool | None = None,
) -> ExtractEntityStep:
    """Assemble an ``ExtractEntityStep`` over the fake seams."""
    return ExtractEntityStep(
        model_provider=model_provider or _FakeModelProvider(),
        kb_loader=kb_loader or _FakeKbLoader(),
        knowledge_loader=knowledge_loader or _FakeKnowledgeLoader(),
        graph_enabled=graph_enabled,
    )


# ── SearchStep: orchestration ──────────────────────────────────────────


async def test_search_step_skips_when_no_scope_and_web_disabled() -> None:
    runner = _FakeRunner()
    step = _step(runner=runner)
    pipeline_ctx = PipelineContext(rewrite_query="budget")

    result = await step.on_event(
        _FakeContext(), EventType.CHUNK_SEARCH, pipeline_ctx, _RecordingNext()
    )

    assert result is None
    assert runner.calls == []
    assert pipeline_ctx.search_result == []


async def test_search_step_runs_kb_and_web_concurrently() -> None:
    runner = _FakeRunner(responses={"budget": [_hit(chunk_id="chunk-1")]})
    web = _FakeWebSearch(hits=[WebSearchHit(title="Web hit", url="https://x.dev/1", snippet="s")])
    step = _step(
        runner=runner,
        kb_loader=_FakeKbLoader([_kb(kb_id="kb-1")]),
        web=web,
        web_config=_FakeWebSearchConfig({"max_results": 5}),
    )
    pipeline_ctx = PipelineContext(
        search_targets=[SearchTarget(knowledge_base_id="kb-1")],
        web_search_enabled=True,
        web_search_provider_id="prov-1",
        rewrite_query="budget",
    )
    next_fn = _RecordingNext()

    result = await step.on_event(_FakeContext(), EventType.CHUNK_SEARCH, pipeline_ctx, next_fn)

    assert result is None
    assert next_fn.calls == 1
    assert [hit.id for hit in pipeline_ctx.search_result] == ["chunk-1", "https://x.dev/1"]
    assert len(runner.calls) == 1
    assert len(web.calls) == 1


async def test_search_step_returns_search_nothing_when_no_results() -> None:
    runner = _FakeRunner()
    step = _step(runner=runner, kb_loader=_FakeKbLoader([_kb(kb_id="kb-1")]))
    pipeline_ctx = PipelineContext(search_targets=[SearchTarget(knowledge_base_id="kb-1")])
    next_fn = _RecordingNext()

    result = await step.on_event(_FakeContext(), EventType.CHUNK_SEARCH, pipeline_ctx, next_fn)

    assert result == ERR_SEARCH_NOTHING
    assert next_fn.calls == 0
    assert pipeline_ctx.search_result == []


async def test_search_step_expands_query_when_recall_low() -> None:
    runner = _FakeRunner(
        responses={
            "the budget for 2026": [_hit(chunk_id="chunk-1")],
            "budget 2026": [_hit(chunk_id="chunk-2")],
        }
    )
    step = _step(runner=runner, kb_loader=_FakeKbLoader([_kb(kb_id="kb-1")]))
    pipeline_ctx = PipelineContext(
        search_targets=[SearchTarget(knowledge_base_id="kb-1")],
        rewrite_query="the budget for 2026",
        query="the budget for 2026",
        enable_query_expansion=True,
        embedding_top_k=10,
        keyword_threshold=0.5,
        rerank_top_k=0,
    )

    await step.on_event(_FakeContext(), EventType.CHUNK_SEARCH, pipeline_ctx, _RecordingNext())

    expansion_queries = [
        call.query_text for call in runner.calls if call.query_text != "the budget for 2026"
    ]
    assert "budget 2026" in expansion_queries
    assert {hit.id for hit in pipeline_ctx.search_result} >= {"chunk-1", "chunk-2"}


async def test_search_step_skips_expansion_when_recall_sufficient() -> None:
    runner = _FakeRunner(responses={"budget": [_hit(chunk_id="chunk-1")]})
    step = _step(runner=runner, kb_loader=_FakeKbLoader([_kb(kb_id="kb-1")]))
    pipeline_ctx = PipelineContext(
        search_targets=[SearchTarget(knowledge_base_id="kb-1")],
        rewrite_query="budget",
        query="budget",
        enable_query_expansion=True,
        embedding_top_k=1,
        keyword_threshold=0.5,
        rerank_top_k=0,
    )

    await step.on_event(_FakeContext(), EventType.CHUNK_SEARCH, pipeline_ctx, _RecordingNext())

    assert [call.query_text for call in runner.calls] == ["budget"]


# ── SearchStep: target grouping ────────────────────────────────────────


async def test_search_by_targets_groups_by_embedding_model() -> None:
    runner = _FakeRunner(responses={"budget": [_hit(chunk_id="chunk-1")]})
    embedding = _FakeEmbeddingProvider(vector=(0.1, 0.2))
    step = _step(
        runner=runner,
        kb_loader=_FakeKbLoader([_kb(kb_id="kb-1"), _kb(kb_id="kb-2")]),
        embedding=embedding,
    )
    pipeline_ctx = PipelineContext(
        search_targets=[
            SearchTarget(knowledge_base_id="kb-1"),
            SearchTarget(knowledge_base_id="kb-2"),
        ],
        rewrite_query="budget",
        embedding_top_k=10,
    )

    results = await step.search_by_targets(_FakeContext(), pipeline_ctx)

    assert len(results) == 1
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call.kb_id == "kb-1"
    assert call.knowledge_base_ids == ("kb-1", "kb-2")
    assert call.query_embedding == (0.1, 0.2)
    assert call.top_k == 10
    # One embedding computation per model group.
    assert embedding.calls == [("kb-1", "budget")]


async def test_search_by_targets_individual_knowledge_target() -> None:
    runner = _FakeRunner(responses={"budget": [_hit(chunk_id="chunk-1")]})
    step = _step(
        runner=runner,
        kb_loader=_FakeKbLoader([_kb(kb_id="kb-1")]),
    )
    pipeline_ctx = PipelineContext(
        search_targets=[
            SearchTarget(
                type=SearchTargetType.KNOWLEDGE,
                knowledge_base_id="kb-1",
                knowledge_ids=["kn-1", "kn-2"],
            )
        ],
        rewrite_query="budget",
        embedding_top_k=10,
        vector_threshold=0.5,
        keyword_threshold=0.4,
    )

    await step.search_by_targets(_FakeContext(), pipeline_ctx)

    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call.knowledge_ids == ("kn-1", "kn-2")
    assert call.knowledge_base_ids == ()
    assert call.vector_threshold == 0.5
    assert call.keyword_threshold == 0.4


async def test_search_by_targets_skips_empty_knowledge_target() -> None:
    runner = _FakeRunner()
    step = _step(runner=runner, kb_loader=_FakeKbLoader([_kb(kb_id="kb-1")]))
    pipeline_ctx = PipelineContext(
        search_targets=[
            SearchTarget(
                type=SearchTargetType.KNOWLEDGE,
                knowledge_base_id="kb-1",
                knowledge_ids=[],
            )
        ],
        rewrite_query="budget",
    )

    results = await step.search_by_targets(_FakeContext(), pipeline_ctx)

    assert results == []
    assert runner.calls == []


async def test_search_by_targets_tag_scope_uses_individual_call() -> None:
    runner = _FakeRunner(responses={"budget": [_hit(chunk_id="chunk-1")]})
    step = _step(runner=runner, kb_loader=_FakeKbLoader([_kb(kb_id="kb-1")]))
    pipeline_ctx = PipelineContext(
        search_targets=[
            SearchTarget(knowledge_base_id="kb-1", tag_ids=["tag-1"], scope_tag_ids=["tag-2"])
        ],
        rewrite_query="budget",
    )

    await step.search_by_targets(_FakeContext(), pipeline_ctx)

    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call.knowledge_base_ids == ()
    assert call.tag_ids == ("tag-1",)
    assert call.scope_tag_ids == ("tag-2",)


async def test_search_by_targets_disabled_recall_thresholds_zero_gates() -> None:
    runner = _FakeRunner(responses={"budget": [_hit(chunk_id="chunk-1")]})
    step = _step(runner=runner, kb_loader=_FakeKbLoader([_kb(kb_id="kb-1")]))
    pipeline_ctx = PipelineContext(
        search_targets=[
            SearchTarget(
                knowledge_base_id="kb-1",
                tag_ids=["tag-1"],
                disable_recall_thresholds=True,
            )
        ],
        rewrite_query="budget",
        vector_threshold=0.5,
        keyword_threshold=0.4,
    )

    await step.search_by_targets(_FakeContext(), pipeline_ctx)

    call = runner.calls[0]
    assert call.vector_threshold == 0.0
    assert call.keyword_threshold == 0.0


async def test_search_by_targets_embedding_failure_degrades_to_empty() -> None:
    runner = _FakeRunner(responses={"budget": [_hit(chunk_id="chunk-1")]})
    embedding = _FakeEmbeddingProvider(error=RuntimeError("embedder down"))
    step = _step(
        runner=runner,
        kb_loader=_FakeKbLoader([_kb(kb_id="kb-1")]),
        embedding=embedding,
    )
    pipeline_ctx = PipelineContext(
        search_targets=[SearchTarget(knowledge_base_id="kb-1")],
        rewrite_query="budget",
    )

    results = await step.search_by_targets(_FakeContext(), pipeline_ctx)

    assert len(results) == 1
    assert runner.calls[0].query_embedding == ()


async def test_recall_thresholds_disabled_zeroes_both() -> None:
    target = SearchTarget(disable_recall_thresholds=True)
    assert recall_thresholds(target, 0.5, 0.4) == (0.0, 0.0)
    assert recall_thresholds(SearchTarget(), 0.5, 0.4) == (0.5, 0.4)


# ── SearchStep: web search ─────────────────────────────────────────────


async def test_web_search_skipped_when_disabled() -> None:
    web = _FakeWebSearch()
    step = _step(runner=_FakeRunner(), web=web, web_config=_FakeWebSearchConfig())
    pipeline_ctx = PipelineContext(
        web_search_enabled=False,
        web_search_provider_id="prov-1",
        rewrite_query="budget",
    )

    results = await step.search_web_if_enabled(_FakeContext(), pipeline_ctx)

    assert results == []
    assert web.calls == []


async def test_web_search_skipped_without_provider() -> None:
    web = _FakeWebSearch()
    step = _step(runner=_FakeRunner(), web=web, web_config=_FakeWebSearchConfig())
    pipeline_ctx = PipelineContext(
        web_search_enabled=True,
        web_search_provider_id="",
        rewrite_query="budget",
    )

    results = await step.search_web_if_enabled(_FakeContext(), pipeline_ctx)

    assert results == []
    assert web.calls == []


async def test_web_search_applies_agent_max_results_override() -> None:
    web = _FakeWebSearch(hits=[WebSearchHit(url="https://x.dev/1")])
    step = _step(
        runner=_FakeRunner(),
        web=web,
        web_config=_FakeWebSearchConfig({"max_results": 5, "include_date": True}),
    )
    pipeline_ctx = PipelineContext(
        web_search_enabled=True,
        web_search_provider_id="prov-1",
        web_search_max_results=3,
        rewrite_query="budget",
    )

    results = await step.search_web_if_enabled(_FakeContext(), pipeline_ctx)

    assert len(results) == 1
    assert web.calls[-1]["max_results"] == 3
    assert web.calls[-1]["include_date"] is True


async def test_web_search_uses_defaults_when_config_missing() -> None:
    web = _FakeWebSearch(hits=[WebSearchHit(url="https://x.dev/1")])
    step = _step(
        runner=_FakeRunner(),
        web=web,
        web_config=_FakeWebSearchConfig(config=None, error=RuntimeError("tenant missing")),
    )
    pipeline_ctx = PipelineContext(
        web_search_enabled=True,
        web_search_provider_id="prov-1",
        rewrite_query="budget",
    )

    results = await step.search_web_if_enabled(_FakeContext(), pipeline_ctx)

    assert len(results) == 1
    assert web.calls[-1]["max_results"] == 10


async def test_web_search_error_degrades_to_no_hits() -> None:
    web = _FakeWebSearch(error=RuntimeError("provider down"))
    step = _step(
        runner=_FakeRunner(),
        web=web,
        web_config=_FakeWebSearchConfig(),
    )
    pipeline_ctx = PipelineContext(
        web_search_enabled=True,
        web_search_provider_id="prov-1",
        rewrite_query="budget",
    )

    results = await step.search_web_if_enabled(_FakeContext(), pipeline_ctx)

    assert results == []
    assert len(web.calls) == 1


# ── Web conversion / config / scope helpers ────────────────────────────


def test_convert_web_search_results_maps_hits() -> None:
    hits = [
        WebSearchHit(
            title="Title",
            url="https://x.dev/1",
            snippet="Snippet",
            content="Body",
            source="duckduckgo",
            published_at=datetime(2026, 1, 2, tzinfo=UTC),
        )
    ]

    results = convert_web_search_results(hits)

    assert len(results) == 1
    result = results[0]
    assert result.id == "https://x.dev/1"
    assert result.knowledge_id == "https://x.dev/1"
    assert result.content == "Title\n\nSnippet\n\nBody"
    assert result.knowledge_title == "Title"
    assert result.end_at == len(result.content)
    assert result.seq == 1
    assert result.score == 0.6
    assert result.match_type == MatchType.WEB_SEARCH
    assert result.chunk_type == "web_search"
    assert result.knowledge_source == "web_search"
    assert result.metadata == {
        "url": "https://x.dev/1",
        "source": "duckduckgo",
        "title": "Title",
        "snippet": "Snippet",
        "published_at": "2026-01-02T00:00:00+00:00",
    }


def test_convert_web_search_results_url_fallback_id() -> None:
    results = convert_web_search_results([WebSearchHit(title="No URL")])
    assert results[0].id == "web_search_0"
    assert results[0].knowledge_id == "web_search_0"


def test_convert_web_search_results_skips_none() -> None:
    # ``None`` entries are tolerated defensively.
    results = convert_web_search_results([WebSearchHit(url="https://x.dev/1"), None])  # type: ignore[list-item]
    assert len(results) == 1


def test_effective_web_search_config_defaults() -> None:
    config = effective_web_search_config(None)
    assert config.max_results == 10
    assert config.compression_method == "none"
    assert config.blacklist == []
    assert config.include_date is False


def test_effective_web_search_config_overrides() -> None:
    config = effective_web_search_config(
        {"max_results": 3, "include_date": True, "blacklist": ["/ads/"], "proxy_url": "http://p"}
    )
    assert config.max_results == 3
    assert config.include_date is True
    assert config.blacklist == ["/ads/"]
    assert config.proxy_url == "http://p"


def test_effective_web_search_config_ignores_non_positive_results() -> None:
    config = effective_web_search_config({"max_results": -5})
    assert config.max_results == 10


def test_has_knowledge_retrieval_scope_cases() -> None:
    assert has_knowledge_retrieval_scope([], ["kb-1"], []) is True
    assert has_knowledge_retrieval_scope([], [], ["kn-1"]) is True
    assert has_knowledge_retrieval_scope([SearchTarget(knowledge_base_id="kb-1")], [], []) is True
    assert (
        has_knowledge_retrieval_scope(
            [
                SearchTarget(
                    type=SearchTargetType.KNOWLEDGE,
                    knowledge_base_id="kb-1",
                    knowledge_ids=["kn-1"],
                )
            ],
            [],
            [],
        )
        is True
    )
    assert has_knowledge_retrieval_scope([SearchTarget(knowledge_base_id="")], [], []) is False
    assert has_knowledge_retrieval_scope([], [], []) is False


# ── Query expansion helpers ────────────────────────────────────────────


def test_expand_queries_empty_rewrite() -> None:
    assert expand_queries(PipelineContext()) == []


def test_expand_queries_joins_keywords_and_excludes_original() -> None:
    pipeline_ctx = PipelineContext(rewrite_query="the budget for 2026", query="the budget for 2026")

    expansions = expand_queries(pipeline_ctx)

    assert "budget 2026" in expansions
    assert "the budget for 2026" not in expansions


def test_expand_queries_extracts_quoted_phrases() -> None:
    pipeline_ctx = PipelineContext(rewrite_query='search for "open source" tools')

    expansions = expand_queries(pipeline_ctx)

    assert "open source" in expansions


def test_expand_queries_strips_question_words() -> None:
    pipeline_ctx = PipelineContext(rewrite_query="什么是知识图谱", query="什么是知识图谱")

    expansions = expand_queries(pipeline_ctx)

    assert "知识图谱" in expansions


def test_expand_queries_limits_to_five() -> None:
    pipeline_ctx = PipelineContext(rewrite_query="a b c d e f g h i j")

    expansions = expand_queries(pipeline_ctx)

    assert len(expansions) <= 5


def test_expand_queries_dedupes_case_insensitively() -> None:
    pipeline_ctx = PipelineContext(rewrite_query="Budget BUDGET for 2026")

    expansions = expand_queries(pipeline_ctx)

    assert len(expansions) == len({expansion.lower() for expansion in expansions})


def test_tokenize_han_runs_through_jieba() -> None:
    tokens = tokenize("检索增强生成")
    assert tokens
    assert all(token in "检索增强生成" for token in tokens)


def test_tokenize_latin_run_passes_whole() -> None:
    assert tokenize("RAG2026") == ["RAG2026"]


def test_tokenize_mixed_han_and_latin() -> None:
    tokens = tokenize("RAG检索增强生成2026")
    assert "RAG" in tokens
    assert "2026" in tokens


def test_extract_keywords_filters_stopwords_and_single_runes() -> None:
    assert extract_keywords("what is the budget") == ["budget"]
    assert extract_keywords("a") == []


def test_split_by_delimiters_splits_cjk_and_latin() -> None:
    assert split_by_delimiters("a，b; c。d") == ["a", "b", "c", "d"]


def test_extract_phrases_handles_ascii_and_cjk_quotes() -> None:
    assert extract_phrases('"hello" and 「世界」') == ["hello", "世界"]


def test_remove_question_words_strips_leading_word() -> None:
    assert remove_question_words("什么是知识图谱") == "知识图谱"
    assert remove_question_words("what is this") == "what is this"


# ── ExtractEntityStep ──────────────────────────────────────────────────


async def test_extract_entity_skips_when_graph_disabled() -> None:
    model_provider = _FakeModelProvider()
    step = _extract_step(model_provider=model_provider, graph_enabled=False)
    pipeline_ctx = PipelineContext(query="Who are the lovers?")
    next_fn = _RecordingNext()

    result = await step.on_event(_FakeContext(), EventType.QUERY_UNDERSTAND, pipeline_ctx, next_fn)

    assert result is None
    assert next_fn.calls == 1
    assert model_provider.calls == []
    assert pipeline_ctx.entity == []


async def test_extract_entity_skips_when_model_resolution_fails() -> None:
    model_provider = _FakeModelProvider(error=RuntimeError("no model"))
    step = _extract_step(model_provider=model_provider, graph_enabled=True)
    pipeline_ctx = PipelineContext(query="Who are the lovers?")
    next_fn = _RecordingNext()

    result = await step.on_event(_FakeContext(), EventType.QUERY_UNDERSTAND, pipeline_ctx, next_fn)

    assert result is None
    assert next_fn.calls == 1
    assert pipeline_ctx.entity == []


async def test_extract_entity_skips_when_knowledge_load_fails() -> None:
    knowledge_loader = _FakeKnowledgeLoader(error=RuntimeError("doc down"))
    step = _extract_step(graph_enabled=True, knowledge_loader=knowledge_loader)
    pipeline_ctx = PipelineContext(query="q", knowledge_ids=["kn-1"])
    next_fn = _RecordingNext()

    result = await step.on_event(_FakeContext(), EventType.QUERY_UNDERSTAND, pipeline_ctx, next_fn)

    assert result is None
    assert next_fn.calls == 1
    assert pipeline_ctx.entity == []


async def test_extract_entity_skips_when_kb_load_fails() -> None:
    kb_loader = _FakeKbLoader(error=RuntimeError("kb down"))
    step = _extract_step(graph_enabled=True, kb_loader=kb_loader)
    pipeline_ctx = PipelineContext(query="q", knowledge_base_ids=["kb-1"])
    next_fn = _RecordingNext()

    result = await step.on_event(_FakeContext(), EventType.QUERY_UNDERSTAND, pipeline_ctx, next_fn)

    assert result is None
    assert next_fn.calls == 1


async def test_extract_entity_skips_when_no_enabled_kb() -> None:
    step = _extract_step(
        graph_enabled=True,
        kb_loader=_FakeKbLoader([_kb(kb_id="kb-1", extract_enabled=False)]),
    )
    pipeline_ctx = PipelineContext(query="q", knowledge_base_ids=["kb-1"])
    next_fn = _RecordingNext()

    result = await step.on_event(_FakeContext(), EventType.QUERY_UNDERSTAND, pipeline_ctx, next_fn)

    assert result is None
    assert next_fn.calls == 1
    assert pipeline_ctx.entity_kb_ids == []
    assert pipeline_ctx.entity == []


async def test_extract_entity_sets_entities_and_scope() -> None:
    chat = _FakeChat(content='```json\n[{"entity": "Romeo"}, {"entity": "Juliet"}]\n```')
    step = _extract_step(
        graph_enabled=True,
        model_provider=_FakeModelProvider(model=chat),
        kb_loader=_FakeKbLoader([_kb(kb_id="kb-1", extract_enabled=True)]),
        knowledge_loader=_FakeKnowledgeLoader(
            [_knowledge(knowledge_id="kn-1", knowledge_base_id="kb-1", tenant_id=7)]
        ),
    )
    pipeline_ctx = PipelineContext(
        query="Who are the lovers?",
        tenant_id=7,
        knowledge_base_ids=["kb-1"],
        knowledge_ids=["kn-1"],
        chat_model_id="chat-1",
    )
    next_fn = _RecordingNext()

    result = await step.on_event(_FakeContext(), EventType.QUERY_UNDERSTAND, pipeline_ctx, next_fn)

    assert result is None
    assert next_fn.calls == 1
    assert pipeline_ctx.entity == ["Romeo", "Juliet"]
    assert pipeline_ctx.entity_kb_ids == ["kb-1"]
    assert pipeline_ctx.entity_knowledge == {"kn-1": "kb-1"}
    assert chat.calls[0][1] is not None
    assert chat.calls[0][1].temperature == 0.3
    assert chat.calls[0][1].max_tokens == 4096


async def test_extract_entity_filters_knowledge_to_enabled_kbs() -> None:
    chat = _FakeChat(content='```json\n[{"entity": "Romeo"}]\n```')
    step = _extract_step(
        graph_enabled=True,
        model_provider=_FakeModelProvider(model=chat),
        kb_loader=_FakeKbLoader(
            [
                _kb(kb_id="kb-1", extract_enabled=True),
                _kb(kb_id="kb-2", extract_enabled=False),
            ]
        ),
        knowledge_loader=_FakeKnowledgeLoader(
            [
                _knowledge(knowledge_id="kn-1", knowledge_base_id="kb-1", tenant_id=7),
                _knowledge(knowledge_id="kn-2", knowledge_base_id="kb-2", tenant_id=7),
            ]
        ),
    )
    pipeline_ctx = PipelineContext(
        query="q",
        tenant_id=7,
        knowledge_base_ids=[],
        knowledge_ids=["kn-1", "kn-2"],
    )

    await step.on_event(_FakeContext(), EventType.QUERY_UNDERSTAND, pipeline_ctx, _RecordingNext())

    assert pipeline_ctx.entity_kb_ids == ["kb-1"]
    assert pipeline_ctx.entity_knowledge == {"kn-1": "kb-1"}


async def test_extract_entity_continues_on_parse_failure() -> None:
    chat = _FakeChat(content="this is not json")
    step = _extract_step(
        graph_enabled=True,
        model_provider=_FakeModelProvider(model=chat),
        kb_loader=_FakeKbLoader([_kb(kb_id="kb-1", extract_enabled=True)]),
    )
    pipeline_ctx = PipelineContext(query="q", knowledge_base_ids=["kb-1"])
    next_fn = _RecordingNext()

    result = await step.on_event(_FakeContext(), EventType.QUERY_UNDERSTAND, pipeline_ctx, next_fn)

    assert result is None
    assert next_fn.calls == 1
    assert pipeline_ctx.entity == []


async def test_extract_entity_env_gate_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_ENABLE", "true")
    model_provider = _FakeModelProvider()
    step = ExtractEntityStep(
        model_provider=model_provider,
        kb_loader=_FakeKbLoader([_kb(kb_id="kb-1", extract_enabled=True)]),
        knowledge_loader=_FakeKnowledgeLoader(),
        graph_enabled=None,
    )
    pipeline_ctx = PipelineContext(query="q", knowledge_base_ids=["kb-1"])

    await step.on_event(_FakeContext(), EventType.QUERY_UNDERSTAND, pipeline_ctx, _RecordingNext())

    assert model_provider.calls == [""]


async def test_extract_entity_env_gate_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEO4J_ENABLE", raising=False)
    model_provider = _FakeModelProvider()
    step = ExtractEntityStep(
        model_provider=model_provider,
        kb_loader=_FakeKbLoader(),
        knowledge_loader=_FakeKnowledgeLoader(),
        graph_enabled=None,
    )
    pipeline_ctx = PipelineContext(query="q")

    await step.on_event(_FakeContext(), EventType.QUERY_UNDERSTAND, pipeline_ctx, _RecordingNext())

    assert model_provider.calls == []


# ── Integration: real DB chain ─────────────────────────────────────────


def _integration_doc(
    *,
    id: str,
    tenant_id: int,
    knowledge_base_id: str,
) -> Document:
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
    content: str = "chunk text",
) -> Chunk:
    return Chunk(
        id=id,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        knowledge_id=knowledge_id,
        content=content,
        chunk_index=0,
        is_enabled=True,
        start_at=0,
        end_at=len(content),
        pre_chunk_id=None,
        next_chunk_id=None,
        chunk_type=CHUNK_TYPE_TEXT,
        parent_chunk_id=None,
        image_info=None,
        relation_chunks=None,
        indirect_relation_chunks=None,
        metadata={"source": "manual"},
        tag_id=None,
        status=CHUNK_STATUS_STORED,
        content_hash=None,
        flags=CHUNK_FLAG_RECOMMENDED,
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


class _FakeEngine:
    """Engine service exposing only what retrieval consumes."""

    def __init__(
        self,
        engine_type: RetrieverEngineType,
        supported: list[RetrieverType],
        *,
        factory: Callable[[RetrieveParams], list[RetrieveResult]],
    ) -> None:
        self._engine_type = engine_type
        self._supported = supported
        self._factory = factory

    def engine_type(self) -> RetrieverEngineType:
        return self._engine_type

    def support(self) -> list[RetrieverType]:
        return list(self._supported)

    async def retrieve(self, _ctx: Context, params: RetrieveParams) -> list[RetrieveResult]:
        return self._factory(params)


def _fake_engine(
    engine_type: RetrieverEngineType,
    supported: list[RetrieverType],
    factory: Callable[[RetrieveParams], list[RetrieveResult]],
) -> RetrieveEngineService:
    return cast("RetrieveEngineService", _FakeEngine(engine_type, supported, factory=factory))


class _Tenant:
    """Tenant carrier exposing effective retriever engines."""

    def __init__(self, engines: list[RetrieverEngineParams]) -> None:
        self._engines = engines

    def get_effective_engines(self) -> list[RetrieverEngineParams]:
        return list(self._engines)


@dataclass(frozen=True, slots=True)
class _TenantCtx:
    """Context carrying a tenant carrier for the unbound engine path."""

    is_background_task: bool = False
    tenant_info: _Tenant | None = None


class _FakeOwnership:
    """In-memory tenant ownership: store_id -> owning tenant_id."""

    async def store_owned_by(self, _ctx: Context, store_id: str, tenant_id: int) -> bool:
        return False


class _FakeEmbedder:
    """Embedder stand-in returning a fixed vector."""

    def __init__(self, vector: tuple[float, ...] = (0.1, 0.2, 0.3)) -> None:
        self._vector = vector

    async def embed(self, ctx: Context, text: str) -> list[float]:
        return list(self._vector)


def _params(
    engine_type: RetrieverEngineType, retriever_type: RetrieverType
) -> RetrieverEngineParams:
    return RetrieverEngineParams(retriever_engine_type=engine_type, retriever_type=retriever_type)


async def test_integration_search_step_hydrates_real_rows(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    kb_service = KBService(kb_repo=KnowledgeBaseRepository(session))
    kb = await kb_service.create_knowledge_base(
        tenant_id=tenant_id, name="search-kb", kb_type=KNOWLEDGE_BASE_TYPE_DOCUMENT
    )
    doc_id = str(uuid.uuid4())
    doc = await KnowledgeRepository(session).create(
        _integration_doc(id=doc_id, tenant_id=tenant_id, knowledge_base_id=kb.id)
    )
    chunk_id = str(uuid.uuid4())
    await ChunkRepository(session).create(
        _integration_chunk(
            id=chunk_id,
            tenant_id=tenant_id,
            knowledge_base_id=kb.id,
            knowledge_id=doc.id,
        )
    )

    def factory(params: RetrieveParams) -> list[RetrieveResult]:
        if params.retriever_type == RetrieverType.VECTOR:
            return [
                RetrieveResult(
                    results=[_index_hit(chunk_id, score=0.9, knowledge_id=doc.id)],
                    retriever_engine_type=_ES,
                    retriever_type=RetrieverType.VECTOR,
                )
            ]
        return [
            RetrieveResult(
                results=[_index_hit(chunk_id, score=0.5, knowledge_id=doc.id)],
                retriever_engine_type=_ES,
                retriever_type=RetrieverType.KEYWORDS,
            )
        ]

    engine = _fake_engine(_ES, [RetrieverType.VECTOR, RetrieverType.KEYWORDS], factory=factory)
    registry = new_retrieve_engine_registry(None, None)
    registry.register(engine)

    deps = SearchDependencies(
        kb_loader=KBServiceKnowledgeBaseLoader(kb_service),
        engine_registry=registry,
        ownership=_FakeOwnership(),
        embedder=cast("Embedder", _FakeEmbedder()),
        chunk_loader=ChunkRepositoryLoader(ChunkRepository(session)),
        knowledge_loader=KnowledgeRepositoryLoader(KnowledgeRepository(session)),
        retrieval_config=RetrievalConfig(),
    )
    runner = HybridSearchRunner(deps)
    step = SearchStep(runner=runner, kb_loader=KBServiceKbLoader(kb_service))

    manager = EventManager()
    manager.register(step)

    pipeline_ctx = PipelineContext(
        search_targets=[SearchTarget(knowledge_base_id=kb.id, tenant_id=tenant_id)],
        rewrite_query="budget",
        embedding_top_k=10,
    )
    ctx = _TenantCtx(
        tenant_info=_Tenant(
            [_params(_ES, RetrieverType.VECTOR), _params(_ES, RetrieverType.KEYWORDS)]
        )
    )

    result = await manager.trigger(ctx, EventType.CHUNK_SEARCH, pipeline_ctx)

    assert result is None
    assert len(pipeline_ctx.search_result) == 1
    hit = pipeline_ctx.search_result[0]
    assert hit.id == chunk_id
    assert hit.content == "chunk text"
    assert hit.knowledge_id == doc.id
    assert hit.knowledge_base_id == kb.id
    assert hit.knowledge_title == "Q3 budget"


def _index_hit(
    chunk_id: str,
    *,
    score: float,
    knowledge_id: str,
) -> IndexWithScore:
    return IndexWithScore(
        id=chunk_id,
        chunk_id=chunk_id,
        score=score,
        content="chunk text",
        knowledge_id=knowledge_id,
        is_enabled=True,
    )
