"""Unit + integration tests for the knowledge-base hybrid search.

Unit tests drive the six search sub-modules and the ``hybrid_search``
orchestrator with faked engines (registered in a real registry), faked
embedders / LLM seams / FAQ metadata loaders, and closure-captured KB
loaders — no vector database or chat API is contacted. They cover query
preparation and rewriting, scope filtering, RRF fusion, FAQ
post-processing, score normalization across mixed engine types,
multi-store fan-out failure collapse, and the empty/error contracts.

Integration tests run against the real applied schema (the ``chunks``
table carries an INTEGER 32-bit ``tenant_id``, so they mint an int32-safe
id from a local counter). One test hydrates a real KB + document + chunk
rows through the repository loaders and asserts the assembled result.
"""

from __future__ import annotations

import itertools
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from random import randint
from typing import TypeAlias, cast

import pytest
import pytest_asyncio
from faker import Faker
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.ai.embedding import Context, Embedder
from src.ai.retrieval.base import RetrieveEngineService
from src.ai.retrieval.registry import (
    VectorStoreUnavailableError,
    new_retrieve_engine_registry,
)
from src.ai.retrieval.types import (
    IndexWithScore,
    RetrieveParams,
    RetrieverEngineParams,
    RetrieverEngineType,
    RetrieveResult,
    RetrieverType,
)
from src.common.exception import NotFoundError, ValidationError
from src.core.knowledge.chunks.types import (
    CHUNK_FLAG_RECOMMENDED,
    CHUNK_STATUS_STORED,
    CHUNK_TYPE_TEXT,
)
from src.core.knowledge.documents.types import CHANNEL_WEB, PARSE_STATUS_COMPLETED
from src.core.knowledge.knowledge_bases.hybrid_search import (
    ChunkRepositoryLoader,
    HybridSearchParams,
    KBServiceKnowledgeBaseLoader,
    KnowledgeRepositoryLoader,
    RetrievalConfig,
    SearchDependencies,
    SearchResult,
    hybrid_search,
)
from src.core.knowledge.knowledge_bases.search_faq import (
    apply_faq_post_processing,
    filter_by_negative_questions,
    iterative_retrieve_with_deduplication,
    matches_negative_questions,
)
from src.core.knowledge.knowledge_bases.search_filter import (
    filter_index_scores,
    scope_retrieve_params,
)
from src.core.knowledge.knowledge_bases.search_mixed import (
    classify_retrieval_results,
    deduplicate_by_score,
    fuse_or_deduplicate,
    fuse_with_rrf,
)
from src.core.knowledge.knowledge_bases.search_query import (
    QueryRewriter,
    prepare_query,
)
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import (
    KNOWLEDGE_BASE_TYPE_DOCUMENT,
    KNOWLEDGE_BASE_TYPE_FAQ,
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
#: from this counter so they stay inside the range.
_INT32_TENANT_BASE = 4_000_000
_INT32_TENANT_SEQ = itertools.count(start=1)

_ES = RetrieverEngineType.ELASTICSEARCH
_MILVUS = RetrieverEngineType.MILVUS


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

#: Produces the engine's canned results for a given retrieval param.
EngineResultFactory: TypeAlias = Callable[[RetrieveParams], list[RetrieveResult]]


class _FakeEngine:
    """Engine service exposing only what retrieval consumes."""

    def __init__(
        self,
        engine_type: RetrieverEngineType,
        supported: list[RetrieverType],
        *,
        factory: EngineResultFactory,
    ) -> None:
        self._engine_type = engine_type
        self._supported = supported
        self._factory = factory
        self.calls: list[RetrieveParams] = []

    def engine_type(self) -> RetrieverEngineType:
        return self._engine_type

    def support(self) -> list[RetrieverType]:
        return list(self._supported)

    async def retrieve(self, _ctx: Context, params: RetrieveParams) -> list[RetrieveResult]:
        self.calls.append(params)
        return self._factory(params)


def _engine(
    engine_type: RetrieverEngineType,
    supported: list[RetrieverType],
    factory: EngineResultFactory,
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
    """Context carrying a tenant carrier for the unbound path."""

    is_background_task: bool = False
    tenant_info: _Tenant | None = None


class _FakeOwnership:
    """In-memory tenant ownership: store_id -> owning tenant_id."""

    def __init__(self, owned: dict[str, int] | None = None) -> None:
        self._owned = owned or {}
        self.calls: list[tuple[str, int]] = []

    async def store_owned_by(self, _ctx: Context, store_id: str, tenant_id: int) -> bool:
        self.calls.append((store_id, tenant_id))
        return self._owned.get(store_id) == tenant_id


class _FakeEmbedder:
    """Embedder stand-in recording the texts it embeds."""

    def __init__(self, vector: tuple[float, ...] = (0.1, 0.2, 0.3)) -> None:
        self._vector = vector
        self.calls: list[str] = []

    async def embed(self, _ctx: Context, text: str) -> list[float]:
        self.calls.append(text)
        return list(self._vector)


class _RecordingRewriter:
    """Query-rewrite seam returning an uppercased query."""

    async def rewrite(self, _ctx: Context, query: str) -> str:
        return query.upper()


class _FaqLoader:
    """Negative-question seam over an in-memory map."""

    def __init__(self, negative: Mapping[str, Sequence[str]] | None = None) -> None:
        self._negative = (
            {chunk_id: tuple(items) for chunk_id, items in negative.items()} if negative else {}
        )
        self.calls: list[list[str]] = []

    async def load_negative_questions(
        self, _ctx: Context, chunk_ids: list[str]
    ) -> dict[str, tuple[str, ...]]:
        self.calls.append(chunk_ids)
        return {
            chunk_id: self._negative[chunk_id]
            for chunk_id in chunk_ids
            if chunk_id in self._negative
        }


class _KbLoader:
    """Knowledge-base loader over an in-memory map."""

    def __init__(self, kbs: list[KnowledgeBaseInfo]) -> None:
        self._by_id = {kb.id: kb for kb in kbs}

    async def load_by_ids(self, ids: list[str]) -> list[KnowledgeBaseInfo]:
        return [self._by_id[id] for id in ids if id in self._by_id]


# ── Builders ───────────────────────────────────────────────────────────


def _kb(
    *,
    kb_id: str,
    tenant_id: int = 7,
    kb_type: str = KNOWLEDGE_BASE_TYPE_DOCUMENT,
    vector_store_id: str | None = None,
    embedding_model_id: str = "em-1",
    vector_enabled: bool = True,
    keyword_enabled: bool = True,
) -> KnowledgeBaseInfo:
    return KnowledgeBaseInfo(
        id=kb_id,
        name=kb_id,
        tenant_id=tenant_id,
        type=kb_type,
        embedding_model_id=embedding_model_id,
        vector_store_id=vector_store_id,
        indexing_strategy={
            "vector_enabled": vector_enabled,
            "keyword_enabled": keyword_enabled,
            "wiki_enabled": False,
            "graph_enabled": False,
        },
        created_at=_NOW,
        updated_at=_NOW,
    )


def _vector_result(*hits: IndexWithScore, engine: RetrieverEngineType = _ES) -> RetrieveResult:
    return RetrieveResult(
        results=list(hits),
        retriever_engine_type=engine,
        retriever_type=RetrieverType.VECTOR,
    )


def _keyword_result(*hits: IndexWithScore, engine: RetrieverEngineType = _ES) -> RetrieveResult:
    return RetrieveResult(
        results=list(hits),
        retriever_engine_type=engine,
        retriever_type=RetrieverType.KEYWORDS,
    )


def _hit(
    chunk_id: str,
    *,
    score: float = 1.0,
    content: str = "content",
    knowledge_id: str = "kn-1",
    is_enabled: bool = True,
) -> IndexWithScore:
    return IndexWithScore(
        id=chunk_id,
        chunk_id=chunk_id,
        score=score,
        content=content,
        knowledge_id=knowledge_id,
        is_enabled=is_enabled,
    )


def _params(
    engine_type: RetrieverEngineType, retriever_type: RetrieverType
) -> RetrieverEngineParams:
    return RetrieverEngineParams(retriever_engine_type=engine_type, retriever_type=retriever_type)


def _env_ctx() -> Context:
    """Context carrying a tenant carrier over the env-store ES engine."""
    return _TenantCtx(
        tenant_info=_Tenant(
            [_params(_ES, RetrieverType.VECTOR), _params(_ES, RetrieverType.KEYWORDS)]
        )
    )


def _hybrid(
    engine: RetrieveEngineService,
    *,
    kbs: list[KnowledgeBaseInfo],
    embedder: _FakeEmbedder | None = None,
    faq_loader: _FaqLoader | None = None,
    rewriter: QueryRewriter | None = None,
    retrieval_config: RetrievalConfig | None = None,
    ownership: _FakeOwnership | None = None,
) -> tuple[SearchDependencies, _FakeEngine]:
    """Build search deps over one env-store engine (no hydration)."""
    registry = new_retrieve_engine_registry(None, None)
    registry.register(engine)
    fake = cast("_FakeEngine", cast("object", engine))
    deps = SearchDependencies(
        kb_loader=_KbLoader(kbs),
        engine_registry=registry,
        ownership=ownership or _FakeOwnership(),
        embedder=cast("Embedder", embedder) if embedder is not None else None,
        query_rewriter=rewriter,
        faq_loader=faq_loader,
        chunk_loader=None,
        knowledge_loader=None,
        retrieval_config=retrieval_config,
    )
    return deps, fake


# ── RetrievalConfig ────────────────────────────────────────────────────


def test_retrieval_config_from_json_keeps_defaults_on_zero() -> None:
    config = RetrievalConfig.from_json(
        {"rrf_k": 0, "vector_threshold": 0, "embedding_top_k": 90, "rrf_vector_weight": 0.8}
    )
    assert config.embedding_top_k == 90
    assert config.rrf_k == 60
    assert config.vector_threshold == 0.15
    assert config.rrf_vector_weight == 0.8


def test_retrieval_config_from_json_none_uses_defaults() -> None:
    assert RetrievalConfig.from_json(None) == RetrievalConfig()


def test_retrieval_config_effective_rrf_weights_default_when_unset() -> None:
    assert RetrievalConfig().effective_rrf_weights() == (0.7, 0.3)


def test_retrieval_config_effective_rrf_weights_fills_single_zero() -> None:
    config = RetrievalConfig(rrf_vector_weight=0.9, rrf_keyword_weight=0.0)
    assert config.effective_rrf_weights() == (0.9, 0.3)


# ── prepare_query ──────────────────────────────────────────────────────


async def test_prepare_query_passthrough_without_rewriter() -> None:
    query = await prepare_query(_TenantCtx(), query_text="  hello world  ", needs_embedding=False)
    assert query.text == "hello world"
    assert query.rewritten == "hello world"
    assert query.embedding == ()


async def test_prepare_query_applies_rewrite_seam() -> None:
    query = await prepare_query(
        _TenantCtx(),
        query_text="hello world",
        needs_embedding=False,
        rewriter=_RecordingRewriter(),
    )
    assert query.rewritten == "HELLO WORLD"


async def test_prepare_query_empty_rewrite_falls_back_to_original() -> None:
    class _EmptyRewriter:
        async def rewrite(self, _ctx: Context, _query: str) -> str:
            return "   "

    query = await prepare_query(
        _TenantCtx(), query_text="hello", needs_embedding=False, rewriter=_EmptyRewriter()
    )
    assert query.rewritten == "hello"


async def test_prepare_query_embeds_only_when_needed() -> None:
    embedder = _FakeEmbedder()
    prepared = await prepare_query(
        _TenantCtx(),
        query_text="hello",
        needs_embedding=True,
        embedder=cast("Embedder", embedder),
    )
    assert prepared.embedding == (0.1, 0.2, 0.3)
    assert embedder.calls == ["hello"]

    not_embedded = await prepare_query(
        _TenantCtx(),
        query_text="hello",
        needs_embedding=False,
        embedder=cast("Embedder", embedder),
    )
    assert not_embedded.embedding == ()
    assert embedder.calls == ["hello"]


# ── search_filter ──────────────────────────────────────────────────────


async def test_scope_retrieve_params_applies_filters() -> None:
    base = RetrieveParams(query="q", retriever_type=RetrieverType.VECTOR)
    scoped = scope_retrieve_params(
        base,
        knowledge_ids=["k1", "k2", "k1"],
        tag_ids=["t1"],
        exclude_knowledge_ids=["k9"],
        exclude_chunk_ids=["c9"],
    )
    assert scoped is not base
    assert scoped.knowledge_ids == ["k1", "k2"]
    assert scoped.tag_ids == ["t1"]
    assert scoped.exclude_knowledge_ids == ["k9"]
    assert scoped.exclude_chunk_ids == ["c9"]
    # The base object is untouched.
    assert base.knowledge_ids == []


async def test_scope_retrieve_params_noop_without_filters() -> None:
    base = RetrieveParams(query="q", retriever_type=RetrieverType.VECTOR)
    assert scope_retrieve_params(base) is base


async def test_filter_index_scores_drops_disabled_and_threshold() -> None:
    results = [
        _hit("c1", score=0.9),
        _hit("c2", score=0.8),
        _hit("c3", score=0.5),
        _hit("c4", score=0.2),
    ]
    disabled = _hit("c3", score=0.7)
    results[2] = disabled.model_copy(update={"is_enabled": False})

    kept = filter_index_scores(results, threshold=0.5)
    # c3 is disabled and c4 (0.2) sits below the threshold.
    assert [hit.chunk_id for hit in kept] == ["c1", "c2"]

    below = filter_index_scores(results, threshold=0.85)
    assert [hit.chunk_id for hit in below] == ["c1"]


async def test_filter_index_scores_drops_excluded_chunk_ids() -> None:
    results = [_hit("c1", score=0.9), _hit("c2", score=0.8)]
    kept = filter_index_scores(results, excluded_chunk_ids=["c2"], enabled_only=False)
    assert [hit.chunk_id for hit in kept] == ["c1"]


# ── search_mixed ───────────────────────────────────────────────────────


async def test_classify_splits_by_retriever_type() -> None:
    results = [
        _vector_result(_hit("c1")),
        _keyword_result(_hit("c2")),
        _vector_result(_hit("c3")),
    ]
    vector, keyword = classify_retrieval_results(results)
    assert [hit.chunk_id for hit in vector] == ["c1", "c3"]
    assert [hit.chunk_id for hit in keyword] == ["c2"]


async def test_deduplicate_keeps_highest_score() -> None:
    results = [
        _hit("c1", score=0.5),
        _hit("c2", score=0.9),
        _hit("c1", score=0.8),
        _hit("c3", score=0.7),
    ]
    deduped = deduplicate_by_score(results)
    assert [hit.chunk_id for hit in deduped] == ["c2", "c1", "c3"]
    assert deduped[1].score == 0.8


async def test_fuse_with_rrf_merges_weighted_ranks() -> None:
    vector = [_hit("c1", score=0.9), _hit("c2", score=0.8)]
    keyword = [_hit("c2", score=0.7), _hit("c3", score=0.6)]
    fused = fuse_with_rrf(vector, keyword, rrf_k=60, vector_weight=0.7, keyword_weight=0.3)

    assert [hit.chunk_id for hit in fused] == ["c2", "c1", "c3"]
    assert fused[0].score == pytest.approx(0.7 / 62 + 0.3 / 61)
    assert fused[1].score == pytest.approx(0.7 / 61)
    assert fused[2].score == pytest.approx(0.3 / 62)


async def test_fuse_or_deduplicate_single_retriever_preserves_scores() -> None:
    vector = [_hit("c1", score=0.9), _hit("c1", score=0.7), _hit("c2", score=0.8)]
    deduped = fuse_or_deduplicate(vector, [])
    assert [hit.chunk_id for hit in deduped] == ["c1", "c2"]
    assert deduped[0].score == 0.9


# ── search_faq ─────────────────────────────────────────────────────────


async def test_matches_negative_questions_exact_trimmed_case_insensitive() -> None:
    assert matches_negative_questions("how to reset", ["  HOW TO RESET  "]) is True
    assert matches_negative_questions("how to reset", ["other", ""]) is False
    assert matches_negative_questions("how to reset", []) is False


async def test_filter_negative_questions_drops_matches() -> None:
    chunks = [_hit("c1"), _hit("c2"), _hit("c3")]
    loader = _FaqLoader({"c2": ("how to reset", "other")})
    kept = await filter_by_negative_questions(_TenantCtx(), chunks, "how to reset", loader)
    assert [hit.chunk_id for hit in kept] == ["c1", "c3"]


async def test_filter_negative_questions_missing_loader_keeps_all() -> None:
    chunks = [_hit("c1")]
    kept = await filter_by_negative_questions(_TenantCtx(), chunks, "how to reset")
    assert kept == chunks


async def test_iterative_retrieve_grows_topk_until_match_count() -> None:
    requested: list[int] = []

    async def retrieve(top_k: int) -> list[IndexWithScore]:
        requested.append(top_k)
        if top_k <= 30:
            return [_hit(f"c{i % 3}") for i in range(top_k)]
        if top_k <= 60:
            return [_hit(f"c{i % 6}") for i in range(top_k)]
        return [_hit(f"c{i % 10}") for i in range(top_k)]

    result = await iterative_retrieve_with_deduplication(
        _TenantCtx(), match_count=10, query_text="q", retrieve=retrieve
    )
    # Rounds start at 10*3=30 and double: 30, 60, 120.
    assert requested == [30, 60, 120]
    assert len(result) == 10
    assert requested[-1] == 120


async def test_iterative_retrieve_stops_when_corpus_exhausted() -> None:
    async def retrieve(top_k: int) -> list[IndexWithScore]:
        return [_hit("c1")]  # far fewer than top_k: no more results exist

    result = await iterative_retrieve_with_deduplication(
        _TenantCtx(), match_count=10, query_text="q", retrieve=retrieve
    )
    assert [hit.chunk_id for hit in result] == ["c1"]


async def test_iterative_retrieve_filters_negative_questions_across_rounds() -> None:
    negatives = {"c1": ("q",)}

    async def retrieve(top_k: int) -> list[IndexWithScore]:
        return [_hit(f"c{i % 3}") for i in range(top_k)]

    result = await iterative_retrieve_with_deduplication(
        _TenantCtx(),
        match_count=5,
        query_text="q",
        retrieve=retrieve,
        faq_loader=_FaqLoader(negatives),
    )
    assert "c1" not in {hit.chunk_id for hit in result}


async def test_apply_faq_passthrough_for_non_faq() -> None:
    chunks = [_hit("c1")]
    result = await apply_faq_post_processing(
        _TenantCtx(),
        kb_type=KNOWLEDGE_BASE_TYPE_DOCUMENT,
        chunks=chunks,
        vector_result_count=0,
        requested_count=1,
        over_retrieve_count=50,
        query_text="q",
        retrieve=_unused_retrieve,
    )
    assert result == chunks


async def test_apply_faq_negative_filter_when_not_capped() -> None:
    chunks = [_hit("c1"), _hit("c2")]
    loader = _FaqLoader({"c2": ("q",)})
    result = await apply_faq_post_processing(
        _TenantCtx(),
        kb_type=KNOWLEDGE_BASE_TYPE_FAQ,
        chunks=chunks,
        vector_result_count=1,  # != over_retrieve_count -> not iterative
        requested_count=10,
        over_retrieve_count=50,
        query_text="q",
        retrieve=_unused_retrieve,
        faq_loader=loader,
    )
    assert [hit.chunk_id for hit in result] == ["c1"]


async def _unused_retrieve(_top_k: int) -> list[IndexWithScore]:
    raise AssertionError("retrieve should not be called")


# ── hybrid_search orchestrator (no hydration) ──────────────────────────


async def test_hybrid_search_fuses_vector_and_keyword() -> None:
    kbs = [_kb(kb_id="kb-1")]
    factory = lambda params: (  # noqa: E731
        [_vector_result(_hit("c1", score=0.9), _hit("c2", score=0.8))]
        if params.retriever_type == RetrieverType.VECTOR
        else [_keyword_result(_hit("c2", score=0.7), _hit("c3", score=0.6))]
    )
    engine = _engine(_ES, [RetrieverType.VECTOR, RetrieverType.KEYWORDS], factory)
    deps, fake = _hybrid(engine, kbs=kbs, embedder=_FakeEmbedder())

    results = await hybrid_search(
        _env_ctx(),
        kb_id="kb-1",
        params=HybridSearchParams(query_text="hello", match_count=10),
        deps=deps,
    )

    assert results is not None
    assert [r.id for r in results] == ["c2", "c1", "c3"]
    assert results[0].score == pytest.approx(0.7 / 62 + 0.3 / 61)
    # The engine saw one vector + one keyword param over the over-retrieve cap.
    assert [p.retriever_type for p in fake.calls] == [
        RetrieverType.VECTOR,
        RetrieverType.KEYWORDS,
    ]
    assert all(p.top_k == 50 for p in fake.calls)
    assert fake.calls[0].embedding == [0.1, 0.2, 0.3]
    assert fake.calls[0].knowledge_base_ids == ["kb-1"]


async def test_hybrid_search_precomputed_embedding_skips_embedder() -> None:
    kbs = [_kb(kb_id="kb-1")]
    factory = lambda params: (  # noqa: E731
        [_vector_result(_hit("c1", score=0.9))]
        if params.retriever_type == RetrieverType.VECTOR
        else []
    )
    engine = _engine(_ES, [RetrieverType.VECTOR, RetrieverType.KEYWORDS], factory)
    embedder = _FakeEmbedder()
    deps, fake = _hybrid(engine, kbs=kbs, embedder=embedder)

    results = await hybrid_search(
        _env_ctx(),
        kb_id="kb-1",
        params=HybridSearchParams(
            query_text="hello", query_embedding=(0.5, 0.5, 0.5), match_count=10
        ),
        deps=deps,
    )

    assert results is not None
    assert results[0].id == "c1"
    assert fake.calls[0].embedding == [0.5, 0.5, 0.5]
    assert embedder.calls == []


async def test_hybrid_search_applies_query_rewrite_seam() -> None:
    kbs = [_kb(kb_id="kb-1")]
    factory = lambda params: (  # noqa: E731
        [_vector_result(_hit("c1", score=0.9))]
        if params.retriever_type == RetrieverType.VECTOR
        else []
    )
    engine = _engine(_ES, [RetrieverType.VECTOR, RetrieverType.KEYWORDS], factory)
    embedder = _FakeEmbedder()
    deps, fake = _hybrid(engine, kbs=kbs, embedder=embedder, rewriter=_RecordingRewriter())

    await hybrid_search(
        _env_ctx(),
        kb_id="kb-1",
        params=HybridSearchParams(query_text="hello", match_count=10),
        deps=deps,
    )

    assert fake.calls[0].query == "HELLO"
    assert embedder.calls == ["HELLO"]


async def test_hybrid_search_truncates_to_match_count() -> None:
    kbs = [_kb(kb_id="kb-1")]
    factory = lambda params: (  # noqa: E731
        [_vector_result(*[_hit(f"c{i}", score=0.9 - i * 0.01) for i in range(5)])]
        if params.retriever_type == RetrieverType.VECTOR
        else []
    )
    engine = _engine(_ES, [RetrieverType.VECTOR, RetrieverType.KEYWORDS], factory)
    deps, _fake = _hybrid(engine, kbs=kbs, embedder=_FakeEmbedder())

    results = await hybrid_search(
        _env_ctx(),
        kb_id="kb-1",
        params=HybridSearchParams(query_text="hello", match_count=2),
        deps=deps,
    )
    assert results is not None
    assert [r.id for r in results] == ["c0", "c1"]


async def test_hybrid_search_missing_query_raises() -> None:
    deps, _fake = _hybrid(_engine(_ES, [RetrieverType.VECTOR], lambda params: []), kbs=[])
    with pytest.raises(ValidationError):
        await hybrid_search(
            _TenantCtx(),
            kb_id="kb-1",
            params=HybridSearchParams(),
            deps=deps,
        )


async def test_hybrid_search_missing_primary_kb_raises() -> None:
    deps, _fake = _hybrid(
        _engine(_ES, [RetrieverType.VECTOR], lambda params: []),
        kbs=[_kb(kb_id="kb-other")],
    )
    with pytest.raises(NotFoundError):
        await hybrid_search(
            _env_ctx(),
            kb_id="kb-1",
            params=HybridSearchParams(query_text="hello"),
            deps=deps,
        )


async def test_hybrid_search_non_retrievable_kb_returns_none() -> None:
    kbs = [
        _kb(
            kb_id="kb-1",
            vector_enabled=False,
            keyword_enabled=False,
            embedding_model_id="",
        )
    ]
    deps, _fake = _hybrid(
        _engine(_ES, [RetrieverType.VECTOR, RetrieverType.KEYWORDS], lambda params: []),
        kbs=kbs,
    )
    result = await hybrid_search(
        _env_ctx(),
        kb_id="kb-1",
        params=HybridSearchParams(query_text="hello"),
        deps=deps,
    )
    assert result is None


async def test_hybrid_search_normalizes_mixed_engine_scores() -> None:
    """Two bound stores with different engine types rescale vector scores."""
    kb_a = _kb(kb_id="kb-a", vector_store_id="store-A", tenant_id=7)
    kb_b = _kb(kb_id="kb-b", vector_store_id="store-B", tenant_id=7)

    milvus = _engine(
        _MILVUS,
        [RetrieverType.VECTOR],
        lambda params: [_vector_result(_hit("cA", score=0.6), engine=_MILVUS)],
    )
    es = _engine(
        _ES,
        [RetrieverType.VECTOR, RetrieverType.KEYWORDS],
        lambda params: [_vector_result(_hit("cB", score=0.9))],
    )
    registry = new_retrieve_engine_registry(None, None)
    registry.register_with_store_id("store-A", milvus)
    registry.register_with_store_id("store-B", es)
    ownership = _FakeOwnership({"store-A": 7, "store-B": 7})

    deps = SearchDependencies(
        kb_loader=_KbLoader([kb_a, kb_b]),
        engine_registry=registry,
        ownership=ownership,
        embedder=cast("Embedder", _FakeEmbedder()),
        chunk_loader=None,
        knowledge_loader=None,
        retrieval_config=RetrievalConfig(),
    )
    results = await hybrid_search(
        _TenantCtx(),
        kb_id="kb-a",
        params=HybridSearchParams(
            query_text="hello",
            match_count=10,
            knowledge_base_ids=("kb-a", "kb-b"),
            disable_keywords_match=True,
        ),
        deps=deps,
    )
    assert results is not None
    # Milvus raw cosine 0.6 rescales to (0.6+1)/2 = 0.8; ES 0.9 passes through.
    by_id = {r.id: r.score for r in results}
    assert by_id["cA"] == pytest.approx(0.8)
    assert by_id["cB"] == pytest.approx(0.9)


async def test_hybrid_search_single_group_fast_path_keeps_native_score() -> None:
    """A single store group skips normalization even for Milvus raw scores."""
    kbs = [_kb(kb_id="kb-1", vector_store_id="store-A", tenant_id=7)]
    milvus = _engine(
        _MILVUS,
        [RetrieverType.VECTOR],
        lambda params: [_vector_result(_hit("cA", score=0.6), engine=_MILVUS)],
    )
    registry = new_retrieve_engine_registry(None, None)
    registry.register_with_store_id("store-A", milvus)
    ownership = _FakeOwnership({"store-A": 7})

    deps = SearchDependencies(
        kb_loader=_KbLoader(kbs),
        engine_registry=registry,
        ownership=ownership,
        embedder=cast("Embedder", _FakeEmbedder()),
        chunk_loader=None,
        knowledge_loader=None,
        retrieval_config=RetrievalConfig(),
    )
    results = await hybrid_search(
        _TenantCtx(),
        kb_id="kb-1",
        params=HybridSearchParams(query_text="hello", match_count=10, disable_keywords_match=True),
        deps=deps,
    )
    assert results is not None
    assert results[0].score == pytest.approx(0.6)


async def test_hybrid_search_multi_store_failure_collapses_to_unavailable() -> None:
    kb_a = _kb(kb_id="kb-a", vector_store_id="store-A", tenant_id=7)
    kb_b = _kb(kb_id="kb-b", vector_store_id="store-B", tenant_id=7)

    class _BoomEngine(_FakeEngine):
        async def retrieve(self, _ctx: Context, params: RetrieveParams) -> list[RetrieveResult]:
            self.calls.append(params)
            raise RuntimeError("backend down")

    engine_a = cast(
        "RetrieveEngineService", _BoomEngine(_ES, [RetrieverType.VECTOR], factory=lambda p: [])
    )
    engine_b = _engine(
        _MILVUS,
        [RetrieverType.VECTOR],
        lambda params: [_vector_result(_hit("cB", score=0.9), engine=_MILVUS)],
    )
    registry = new_retrieve_engine_registry(None, None)
    registry.register_with_store_id("store-A", engine_a)
    registry.register_with_store_id("store-B", engine_b)
    ownership = _FakeOwnership({"store-A": 7, "store-B": 7})

    deps = SearchDependencies(
        kb_loader=_KbLoader([kb_a, kb_b]),
        engine_registry=registry,
        ownership=ownership,
        embedder=cast("Embedder", _FakeEmbedder()),
        chunk_loader=None,
        knowledge_loader=None,
        retrieval_config=RetrievalConfig(),
    )
    with pytest.raises(VectorStoreUnavailableError):
        await hybrid_search(
            _TenantCtx(),
            kb_id="kb-a",
            params=HybridSearchParams(
                query_text="hello",
                match_count=10,
                knowledge_base_ids=("kb-a", "kb-b"),
            ),
            deps=deps,
        )


async def test_hybrid_search_faq_negative_filtering() -> None:
    kbs = [_kb(kb_id="kb-1", kb_type=KNOWLEDGE_BASE_TYPE_FAQ)]
    factory = lambda params: (  # noqa: E731
        [_vector_result(_hit("c1", score=0.9))]
        if params.retriever_type == RetrieverType.VECTOR
        else []
    )
    engine = _engine(_ES, [RetrieverType.VECTOR, RetrieverType.KEYWORDS], factory)
    loader = _FaqLoader({"c1": ("how to reset",)})
    deps, _fake = _hybrid(engine, kbs=kbs, embedder=_FakeEmbedder(), faq_loader=loader)

    result = await hybrid_search(
        _env_ctx(),
        kb_id="kb-1",
        params=HybridSearchParams(query_text="how to reset", match_count=10),
        deps=deps,
    )
    # The FAQ hit is dropped by negative-question filtering -> empty result.
    assert result is None


async def test_hybrid_search_faq_iterative_retrieval() -> None:
    kbs = [_kb(kb_id="kb-1", kb_type=KNOWLEDGE_BASE_TYPE_FAQ)]

    def factory(params: RetrieveParams) -> list[RetrieveResult]:
        top = params.top_k
        # Initial pass (over-retrieve cap 50) yields 3 unique chunks; the
        # iterative rounds (30/60/120) surface progressively more unique ids.
        unique = {50: 3, 30: 4, 60: 7}.get(top, 10)
        hits = [_hit(f"c{i % unique}", score=0.9, content="q") for i in range(top)]
        return [_vector_result(*hits)]

    engine = _engine(_ES, [RetrieverType.VECTOR, RetrieverType.KEYWORDS], factory)
    deps, _fake = _hybrid(engine, kbs=kbs, embedder=_FakeEmbedder())

    result = await hybrid_search(
        _env_ctx(),
        kb_id="kb-1",
        params=HybridSearchParams(query_text="q", match_count=10),
        deps=deps,
    )
    assert result is not None
    assert len(result) == 10
    assert {r.id for r in result} == {f"c{i}" for i in range(10)}


# ── hybrid_search hydration (real DB) ──────────────────────────────────


def _integration_doc(
    *,
    id: str,
    tenant_id: int,
    knowledge_base_id: str,
    title: str = "Q3 budget",
) -> Document:
    return Document(
        id=id,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        type="file",
        title=title,
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
    content: str = "chunk text",
) -> Chunk:
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


async def test_integration_hybrid_search_hydrates_results(
    session: AsyncSession,
) -> None:
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
    await ChunkRepository(session).create_many(
        [
            _integration_chunk(
                id=chunk_id,
                tenant_id=tenant_id,
                knowledge_base_id=kb.id,
                knowledge_id=doc.id,
                chunk_index=0,
                content="chunk text",
            )
        ]
    )

    engine = _engine(
        _ES,
        [RetrieverType.VECTOR, RetrieverType.KEYWORDS],
        factory=lambda params: (
            [_vector_result(_hit(chunk_id, score=0.9, content="chunk text", knowledge_id=doc.id))]
            if params.retriever_type == RetrieverType.VECTOR
            else [
                _keyword_result(
                    _hit(chunk_id, score=0.5, content="chunk text", knowledge_id=doc.id)
                )
            ]
        ),
    )
    registry = new_retrieve_engine_registry(None, None)
    registry.register(engine)
    tenant = _Tenant([_params(_ES, RetrieverType.VECTOR), _params(_ES, RetrieverType.KEYWORDS)])
    deps = SearchDependencies(
        kb_loader=KBServiceKnowledgeBaseLoader(kb_service),
        engine_registry=registry,
        ownership=_FakeOwnership(),
        embedder=cast("Embedder", _FakeEmbedder()),
        chunk_loader=ChunkRepositoryLoader(ChunkRepository(session)),
        knowledge_loader=KnowledgeRepositoryLoader(KnowledgeRepository(session)),
        retrieval_config=RetrievalConfig(),
    )

    results = await hybrid_search(
        _TenantCtx(tenant_info=tenant),
        kb_id=kb.id,
        params=HybridSearchParams(query_text="budget", match_count=10),
        deps=deps,
    )

    assert results is not None
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, SearchResult)
    assert result.id == chunk_id
    assert result.content == "chunk text"
    assert result.knowledge_id == doc.id
    assert result.knowledge_base_id == kb.id
    assert result.knowledge_title == "Q3 budget"
    assert result.knowledge_filename == "budget-2026.pdf"
    assert result.knowledge_source == "budget-2026.pdf"
    assert result.metadata == {"owner": "finance"}
    assert "scope: 2026" in result.knowledge_custom_metadata
    assert result.chunk_type == CHUNK_TYPE_TEXT
    assert result.chunk_index == 0
