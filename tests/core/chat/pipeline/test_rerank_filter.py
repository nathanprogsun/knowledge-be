"""Unit + integration tests for the rerank, filter-top-k and web-fetch steps.

Unit tests drive the pure helpers (passage cleaning, enrichment, composite
scoring, deterministic sort, MMR) and each step's ``on_event`` behaviour
with in-memory fakes for the model-service and fetcher seams — no
database, no network.

Integration tests run the ``CHUNK_RERANK`` + ``FILTER_TOP_K`` chain over
real ``chunks`` rows (seeded with an int32-safe tenant id) through the
pipeline engine.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from random import randint

import pytest
import pytest_asyncio
from faker import Faker
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.ai.rerank.base import Reranker
from src.ai.rerank.remote_api import RankResult
from src.common.json import JsonObject
from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.engine import (
    ERR_GET_RERANK_MODEL,
    ERR_SEARCH_NOTHING,
    EventManager,
    PluginError,
)
from src.core.chat.pipeline.steps.filter_topk import (
    FilterTopKPlugin,
    sort_search_results_deterministically,
)
from src.core.chat.pipeline.steps.rerank import (
    RerankPlugin,
    apply_mmr,
    clean_passage_for_rerank,
    composite_score,
    get_enriched_passage,
    rerank_fallback_min_score,
)
from src.core.chat.pipeline.steps.web_fetch import WebFetchPlugin
from src.core.chat.pipeline.types import (
    Context,
    EventType,
    QueryIntent,
    SearchResult,
    SearchTarget,
    SearchTargetType,
)
from src.db.dao.chunk_repository import ChunkRepository
from src.db.models.chunk import Chunk
from src.settings import get_settings, reset_settings_cache
from tests.integration.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_FAKER_SEED_MAX = 100_000_000

# ``chunks.tenant_id`` is an INTEGER (32-bit) column; integration ids are
# minted from this counter so seeded rows never overflow.
_INT32_TENANT_BASE = 8_000_000
_INT32_TENANT_SEQ = itertools.count(start=1)


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


class _FakeContext:
    """Opaque execution context (empty structural protocol)."""


_CTX: Context = _FakeContext()


async def _noop_next() -> PluginError | None:
    return None


def _result(
    *,
    result_id: str,
    content: str = "",
    score: float = 0.0,
    knowledge_id: str = "",
    knowledge_source: str = "",
    chunk_type: str = "text",
    chunk_index: int = 0,
    metadata: dict[str, str] | None = None,
    image_info: str = "",
    chunk_metadata: JsonObject | None = None,
) -> SearchResult:
    return SearchResult(
        id=result_id,
        content=content,
        score=score,
        knowledge_id=knowledge_id,
        knowledge_source=knowledge_source,
        chunk_type=chunk_type,
        chunk_index=chunk_index,
        metadata=dict(metadata or {}),
        image_info=image_info,
        chunk_metadata=chunk_metadata,
    )


class _FakeRerankModel:
    """Model stub whose ``rerank`` returns a fixed per-call result list.

    ``responses`` is a list of one result list per call; an empty sequence
    makes every call return no results.
    """

    def __init__(self, responses: Sequence[list[RankResult]]) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.received: list[tuple[str, list[str]]] = []

    async def rerank(self, query: str, documents: list[str]) -> list[RankResult]:
        self.calls += 1
        self.received.append((query, documents))
        if not self._responses:
            return []
        response = self._responses[min(self.calls - 1, len(self._responses) - 1)]
        return list(response)

    def get_model_name(self) -> str:
        return "fake-rerank"

    def get_model_id(self) -> str:
        return "model-fake"


class _FakeRerankService:
    """Model-service stub resolving to one fake model (or an error)."""

    def __init__(
        self,
        model: Reranker,
        *,
        error: Exception | None = None,
    ) -> None:
        self._model = model
        self._error = error
        self.resolved: list[tuple[int, str]] = []

    async def get_rerank_model(self, *, tenant_id: int, model_id: str) -> Reranker:
        self.resolved.append((tenant_id, model_id))
        if self._error is not None:
            raise self._error
        return self._model


class _FakeFetcher:
    """Fetcher stub: per-URL content, error, or empty response."""

    def __init__(self, contents: dict[str, str], errors: dict[str, Exception]) -> None:
        self._contents = contents
        self._errors = errors
        self.requested: list[str] = []

    async def fetch(self, url: str) -> str:
        self.requested.append(url)
        error = self._errors.get(url)
        if error is not None:
            raise error
        return self._contents.get(url, "")


# ── clean_passage_for_rerank ───────────────────────────────────────────


@pytest.mark.parametrize(
    ("input_text", "expected"),
    [
        ("这是一段普通的文本内容", "这是一段普通的文本内容"),
        ("前文 ![图片说明](https://example.com/img.png) 后文", "前文  后文"),
        ("请参考 [官方文档](https://docs.example.com) 了解详情", "请参考 官方文档 了解详情"),
        ("访问 https://example.com/path?q=1&b=2 获取更多信息", "访问  获取更多信息"),
        ("示例代码：\n```python\nprint('hello')\n```\n以上是示例", "示例代码：\n\n以上是示例"),  # noqa: RUF001
        ("公式如下 $$E=mc^2$$ 其中E是能量", "公式如下  其中E是能量"),
        ("| 名称 | 值 |\n| --- | --- |\n| A | 1 |", "名称, 值\n\nA, 1"),
        ("## 第二章 概述\n### 2.1 背景", "第二章 概述\n2.1 背景"),
        ("> 这是一段引用\n> 第二行引用", "这是一段引用\n第二行引用"),
        ("这是 **加粗** 和 *斜体* 以及 ***粗斜体*** 文本", "这是 加粗 和 斜体 以及 粗斜体 文本"),
        ("- 项目一\n- 项目二\n1. 有序一\n2. 有序二", "项目一\n项目二\n有序一\n有序二"),
        ("文本<br>换行<div class=\"test\">内容</div>结尾", "文本换行内容结尾"),
        ("段落一\n\n\n\n\n段落二", "段落一\n\n段落二"),
        ("| col1 | col2 | col3 |", "col1, col2, col3"),
        (
            "| Header1 | Header2 |\n| --- | --- |\n| data1 | data2 |\n| data3 | data4 |",
            "Header1, Header2\n\ndata1, data2\ndata3, data4",
        ),
        ("| --- | --- |", ""),
        ("   \n\n   ", ""),
    ],
)
def test_clean_passage_for_rerank(input_text: str, expected: str) -> None:
    assert clean_passage_for_rerank(input_text) == expected


def test_clean_passage_for_rerank_combined_real_world() -> None:
    passage = (
        "## 产品介绍\n\n"
        "这是一个 **重要的** 产品。详见 [产品页面](https://example.com/product)。\n\n"
        "![产品截图](images/product.png)\n\n"
        "> 用户评价：非常好用\n\n"  # noqa: RUF001
        "- 功能一\n- 功能二\n\n"
        "```json\n{\"key\": \"value\"}\n```\n"
    )
    expected = (
        "产品介绍\n\n"
        "这是一个 重要的 产品。详见 产品页面。\n\n"
        "用户评价：非常好用\n\n"  # noqa: RUF001
        "功能一\n功能二"
    )
    assert clean_passage_for_rerank(passage) == expected


# ── get_enriched_passage ───────────────────────────────────────────────


def test_get_enriched_passage_plain_content() -> None:
    result = _result(result_id="chunk-1", content="正文内容")
    assert get_enriched_passage(result) == "正文内容"


def test_get_enriched_passage_appends_image_captions_and_ocr() -> None:
    image_info = json.dumps(
        [
            {"url": "https://x/a.png", "caption": "架构图", "ocr_text": ""},
            {"url": "https://x/b.png", "caption": "", "ocr_text": "表格数据"},
        ]
    )
    result = _result(result_id="chunk-1", content="图表说明", image_info=image_info)
    passage = get_enriched_passage(result)
    assert "图表说明" in passage
    assert "架构图" in passage
    assert "表格数据" in passage
    assert passage.endswith("架构图\n表格数据")


def test_get_enriched_passage_appends_generated_questions() -> None:
    result = _result(
        result_id="chunk-1",
        content="edited chunk body",
        chunk_metadata={
            "generated_questions_revision": 1,
            "generated_questions": [
                {"id": "old", "question": "question generated before the edit"},
                {"id": "second", "question": "另一条生成问题"},
            ],
        },
    )
    passage = get_enriched_passage(result)
    assert "question generated before the edit" in passage
    assert "另一条生成问题" in passage
    assert "question generated before the edit; 另一条生成问题" in passage


def test_get_enriched_passage_ignores_malformed_image_info() -> None:
    result = _result(result_id="chunk-1", content="正文", image_info="{not json")
    assert get_enriched_passage(result) == "正文"


def test_get_enriched_passage_cleans_content_before_appending() -> None:
    result = _result(
        result_id="chunk-1",
        content="标题\n\n![](https://x/img.png)\n\n正文",
        chunk_metadata={"generated_questions": [{"id": "q1", "question": "生成问题"}]},
    )
    passage = get_enriched_passage(result)
    assert "![](https://x/img.png)" not in passage
    assert "标题" in passage
    assert "正文" in passage
    assert passage.endswith("生成问题")


# ── Scoring helpers ────────────────────────────────────────────────────


def test_composite_score_web_search_source_weighted_lower() -> None:
    web = _result(result_id="w", knowledge_source="web_search")
    kb = _result(result_id="k")
    assert composite_score(web, 1.0, 1.0) < composite_score(kb, 1.0, 1.0)


def test_composite_score_formula_and_clamp() -> None:
    result = _result(result_id="k")
    assert composite_score(result, 0.0, 0.0) == pytest.approx(0.1)
    assert composite_score(result, 1.0, 1.0) == pytest.approx(1.0)
    assert composite_score(result, 5.0, 5.0) == 1.0  # clamped
    assert composite_score(result, -1.0, -1.0) == 0.0  # clamped


def test_rerank_fallback_min_score_default_and_explicit_scope() -> None:
    assert rerank_fallback_min_score([]) == 0.15
    explicit = [
        SearchTarget(
            type=SearchTargetType.KNOWLEDGE_BASE,
            knowledge_base_id="kb",
            disable_recall_thresholds=True,
        )
    ]
    assert rerank_fallback_min_score(explicit) == 0.0
    assert rerank_fallback_min_score(
        [SearchTarget(type=SearchTargetType.KNOWLEDGE_BASE, knowledge_base_id="kb")]
    ) == 0.15


# ── Deterministic sort ─────────────────────────────────────────────────


def test_sort_search_results_deterministically_by_score() -> None:
    results = [
        _result(result_id="low", knowledge_id="doc-c", score=0.2),
        _result(result_id="high", knowledge_id="doc-a", score=0.9),
        _result(result_id="medium", knowledge_id="doc-b", score=0.5),
        _result(result_id="second", knowledge_id="doc-d", score=0.8),
    ]
    ordered = sort_search_results_deterministically(results)
    assert [item.id for item in ordered] == ["high", "second", "medium", "low"]


def test_sort_search_results_deterministic_tie_breakers() -> None:
    results = [
        _result(result_id="chunk-b", knowledge_id="doc-b", chunk_type="text", chunk_index=10, score=0.8),
        _result(result_id="chunk-c", knowledge_id="doc-a", chunk_type="summary", chunk_index=0, score=0.8),
        _result(result_id="chunk-a", knowledge_id="doc-a", chunk_type="text", chunk_index=0, score=0.8),
    ]
    ordered = sort_search_results_deterministically(results)
    assert [item.id for item in ordered] == ["chunk-c", "chunk-a", "chunk-b"]


# ── MMR ────────────────────────────────────────────────────────────────


def test_apply_mmr_empty_or_zero_k() -> None:
    assert apply_mmr([], 3, 0.7) == []
    assert apply_mmr([_result(result_id="a", score=0.9)], 0, 0.7) == []


def test_apply_mmr_selects_top_k_by_relevance() -> None:
    results = [
        _result(result_id="a", content="alpha beta gamma", score=0.9),
        _result(result_id="b", content="delta epsilon zeta", score=0.8),
        _result(result_id="c", content="eta theta iota", score=0.7),
    ]
    selected = apply_mmr(results, 2, 0.7)
    assert len(selected) == 2
    assert {item.id for item in selected} == {"a", "b"}


def test_apply_mmr_trades_redundancy_for_relevance() -> None:
    first = _result(result_id="a", content="python programming language guide", score=0.9)
    near_duplicate = _result(result_id="b", content="python programming language tutorial", score=0.8)
    distinct = _result(result_id="c", content="cooking recipes for pasta dinner", score=0.7)
    # k=2: by relevance the pair would be a+b, but b is a near-duplicate of
    # a, so MMR trades it for the distinct c.
    selected = apply_mmr([first, near_duplicate, distinct], 2, 0.7)
    assert [item.id for item in selected] == ["a", "c"]


# ── RerankPlugin.on_event ──────────────────────────────────────────────


async def test_rerank_skips_when_retrieval_not_needed() -> None:
    plugin = RerankPlugin(_FakeRerankService(_FakeRerankModel([])))
    pipeline_ctx = PipelineContext(intent=QueryIntent.GREETING)
    error = await plugin.on_event(_CTX, EventType.CHUNK_RERANK, pipeline_ctx, _noop_next)
    assert error is None


async def test_rerank_skips_when_no_search_results() -> None:
    service = _FakeRerankService(_FakeRerankModel([]))
    plugin = RerankPlugin(service)
    pipeline_ctx = PipelineContext(rerank_model_id="model-1", rewrite_query="q")
    error = await plugin.on_event(_CTX, EventType.CHUNK_RERANK, pipeline_ctx, _noop_next)
    assert error is None
    assert service.resolved == []


async def test_rerank_skips_when_no_model_id() -> None:
    service = _FakeRerankService(_FakeRerankModel([]))
    plugin = RerankPlugin(service)
    pipeline_ctx = PipelineContext(search_result=[_result(result_id="a", content="x")])
    error = await plugin.on_event(_CTX, EventType.CHUNK_RERANK, pipeline_ctx, _noop_next)
    assert error is None
    assert service.resolved == []


async def test_rerank_model_resolution_error_returns_plugin_error() -> None:
    service = _FakeRerankService(
        _FakeRerankModel([]),
        error=RuntimeError("boom"),
    )
    plugin = RerankPlugin(service)
    pipeline_ctx = PipelineContext(
        tenant_id=make_test_tenant_id(),
        rerank_model_id="model-1",
        search_result=[_result(result_id="a", content="x")],
    )
    error = await plugin.on_event(_CTX, EventType.CHUNK_RERANK, pipeline_ctx, _noop_next)
    assert error is not None
    assert error.error_type == ERR_GET_RERANK_MODEL.error_type
    assert error.err is not None
    assert service.resolved == [(pipeline_ctx.tenant_id, "model-1")]


async def test_rerank_builds_composite_scores_and_publishes_result() -> None:
    model = _FakeRerankModel(
        [
            [
                RankResult(index=1, relevance_score=0.9),
                RankResult(index=0, relevance_score=0.8),
            ]
        ]
    )
    plugin = RerankPlugin(_FakeRerankService(model))
    pipeline_ctx = PipelineContext(
        tenant_id=make_test_tenant_id(),
        rerank_model_id="model-1",
        rerank_threshold=0.3,
        rerank_top_k=5,
        rewrite_query="query",
        search_result=[
            _result(result_id="a", content="first passage", score=0.6),
            _result(result_id="b", content="second passage", score=0.7),
        ],
    )

    error = await plugin.on_event(_CTX, EventType.CHUNK_RERANK, pipeline_ctx, _noop_next)

    assert error is None
    assert model.received == [("query", ["first passage", "second passage"])]
    assert len(pipeline_ctx.rerank_result) == 2
    by_id = {item.id: item for item in pipeline_ctx.rerank_result}
    assert by_id["a"].score == pytest.approx(
        composite_score(_result(result_id="a", content="first passage", score=0.6), 0.8, 0.6)
    )
    assert by_id["b"].score == pytest.approx(
        composite_score(_result(result_id="b", content="second passage", score=0.7), 0.9, 0.7)
    )
    assert by_id["a"].metadata["model_score"] == "0.8000"
    assert by_id["b"].metadata["base_score"] == "0.7000"
    # The shared search_result reflects the updated composite scores too.
    assert pipeline_ctx.search_result[0].metadata.get("model_score") == "0.8000"
    assert pipeline_ctx.search_result[1].metadata.get("model_score") == "0.9000"


async def test_rerank_applies_faq_boost() -> None:
    model = _FakeRerankModel([[RankResult(index=0, relevance_score=0.5)]])
    plugin = RerankPlugin(_FakeRerankService(model))
    pipeline_ctx = PipelineContext(
        tenant_id=make_test_tenant_id(),
        rerank_model_id="model-1",
        rerank_threshold=0.1,
        rerank_top_k=5,
        rewrite_query="q",
        faq_priority_enabled=True,
        faq_score_boost=1.5,
        search_result=[_result(result_id="faq-1", content="faq body", score=0.4, chunk_type="faq")],
    )

    error = await plugin.on_event(_CTX, EventType.CHUNK_RERANK, pipeline_ctx, _noop_next)

    assert error is None
    item = pipeline_ctx.rerank_result[0]
    assert item.metadata["faq_boosted"] == "true"
    assert item.metadata["faq_original_score"] == f"{0.6 * 0.5 + 0.3 * 0.4 + 0.1:.4f}"
    assert item.score == pytest.approx(min(composite_score(
        _result(result_id="faq-1", content="faq body", score=0.4, chunk_type="faq"),
        0.5,
        0.4,
    ) * 1.5, 1.0))


async def test_rerank_skips_faq_boost_when_disabled_or_not_faq() -> None:
    model = _FakeRerankModel([[RankResult(index=0, relevance_score=0.5)]])
    plugin = RerankPlugin(_FakeRerankService(model))
    pipeline_ctx = PipelineContext(
        tenant_id=make_test_tenant_id(),
        rerank_model_id="model-1",
        rerank_threshold=0.1,
        rerank_top_k=5,
        rewrite_query="q",
        faq_priority_enabled=False,
        faq_score_boost=1.5,
        search_result=[_result(result_id="faq-1", content="faq body", score=0.4, chunk_type="faq")],
    )
    await plugin.on_event(_CTX, EventType.CHUNK_RERANK, pipeline_ctx, _noop_next)
    assert "faq_boosted" not in pipeline_ctx.rerank_result[0].metadata


async def test_rerank_api_error_falls_back_to_original_results() -> None:
    class _ExplodingModel:
        async def rerank(self, query: str, documents: list[str]) -> list[RankResult]:
            raise RuntimeError("api down")

        def get_model_name(self) -> str:
            return "explode"

        def get_model_id(self) -> str:
            return "model-x"

    plugin = RerankPlugin(_FakeRerankService(_ExplodingModel()))
    candidates = [
        _result(result_id="a", content="first", score=0.5),
        _result(result_id="b", content="second", score=0.4),
    ]
    pipeline_ctx = PipelineContext(
        tenant_id=make_test_tenant_id(),
        rerank_model_id="model-1",
        rerank_threshold=0.3,
        rewrite_query="q",
        search_result=list(candidates),
    )

    error = await plugin.on_event(_CTX, EventType.CHUNK_RERANK, pipeline_ctx, _noop_next)

    assert error is None
    assert [item.id for item in pipeline_ctx.search_result] == ["a", "b"]
    assert pipeline_ctx.rerank_result == []


async def test_rerank_threshold_degradation_retries_then_restores() -> None:
    model = _FakeRerankModel(
        [
            [],
            [RankResult(index=0, relevance_score=0.5)],
        ]
    )
    plugin = RerankPlugin(_FakeRerankService(model))
    pipeline_ctx = PipelineContext(
        tenant_id=make_test_tenant_id(),
        rerank_model_id="model-1",
        rerank_threshold=0.8,
        rerank_top_k=5,
        rewrite_query="q",
        search_result=[_result(result_id="a", content="passage", score=0.6)],
    )

    error = await plugin.on_event(_CTX, EventType.CHUNK_RERANK, pipeline_ctx, _noop_next)

    assert error is None
    assert model.calls == 2
    assert pipeline_ctx.rerank_threshold == 0.8  # restored
    assert len(pipeline_ctx.rerank_result) == 1


async def test_rerank_degrades_only_when_threshold_is_high() -> None:
    model = _FakeRerankModel([])
    plugin = RerankPlugin(_FakeRerankService(model))
    pipeline_ctx = PipelineContext(
        tenant_id=make_test_tenant_id(),
        rerank_model_id="model-1",
        rerank_threshold=0.2,  # below the degrade floor: no retry
        rerank_top_k=5,
        rewrite_query="q",
        search_result=[_result(result_id="a", content="passage", score=0.6)],
    )

    error = await plugin.on_event(_CTX, EventType.CHUNK_RERANK, pipeline_ctx, _noop_next)

    assert error is not None
    assert error.error_type == ERR_SEARCH_NOTHING.error_type
    assert model.calls == 1


async def test_rerank_empty_passages_skipped_before_model_call() -> None:
    model = _FakeRerankModel([[RankResult(index=0, relevance_score=0.9)]])
    plugin = RerankPlugin(_FakeRerankService(model))
    pipeline_ctx = PipelineContext(
        tenant_id=make_test_tenant_id(),
        rerank_model_id="model-1",
        rerank_threshold=0.3,
        rerank_top_k=5,
        rewrite_query="q",
        search_result=[
            _result(result_id="noise", content="![x](https://a/b.png)", score=0.9),
            _result(result_id="text", content="real passage", score=0.5),
        ],
    )

    error = await plugin.on_event(_CTX, EventType.CHUNK_RERANK, pipeline_ctx, _noop_next)

    assert error is None
    assert model.received[0][1] == ["real passage"]
    assert len(pipeline_ctx.rerank_result) == 1


async def test_rerank_fallback_top1_preserves_best_candidate() -> None:
    model = _FakeRerankModel([[RankResult(index=0, relevance_score=0.2)]])
    plugin = RerankPlugin(_FakeRerankService(model))
    pipeline_ctx = PipelineContext(
        tenant_id=make_test_tenant_id(),
        rerank_model_id="model-1",
        rerank_threshold=0.8,
        rerank_top_k=5,
        rewrite_query="q",
        search_result=[_result(result_id="a", content="passage", score=0.6)],
    )

    error = await plugin.on_event(_CTX, EventType.CHUNK_RERANK, pipeline_ctx, _noop_next)

    # 0.2 >= fallback min (0.15) so the top candidate survives the filter.
    assert error is None
    assert len(pipeline_ctx.rerank_result) == 1


async def test_rerank_empty_result_returns_search_nothing() -> None:
    model = _FakeRerankModel([[RankResult(index=0, relevance_score=0.05)]])
    plugin = RerankPlugin(_FakeRerankService(model))
    pipeline_ctx = PipelineContext(
        tenant_id=make_test_tenant_id(),
        rerank_model_id="model-1",
        rerank_threshold=0.8,
        rerank_top_k=5,
        rewrite_query="q",
        search_result=[_result(result_id="a", content="passage", score=0.6)],
    )

    error = await plugin.on_event(_CTX, EventType.CHUNK_RERANK, pipeline_ctx, _noop_next)

    # 0.05 < 0.15 fallback min: nothing survives and the caller handles it.
    assert error is not None
    assert error.error_type == ERR_SEARCH_NOTHING.error_type
    assert pipeline_ctx.rerank_result == []


# ── FilterTopKPlugin.on_event ──────────────────────────────────────────


async def test_filter_topk_sorts_and_truncates_merge_result() -> None:
    plugin = FilterTopKPlugin()
    pipeline_ctx = PipelineContext(
        rerank_top_k=3,
        merge_result=[
            _result(result_id="low", knowledge_id="doc-c", score=0.2),
            _result(result_id="high", knowledge_id="doc-a", score=0.9),
            _result(result_id="medium", knowledge_id="doc-b", score=0.5),
            _result(result_id="second", knowledge_id="doc-d", score=0.8),
        ],
    )

    error = await plugin.on_event(_CTX, EventType.FILTER_TOP_K, pipeline_ctx, _noop_next)

    assert error is None
    assert [item.id for item in pipeline_ctx.merge_result] == ["high", "second", "medium"]


async def test_filter_topk_uses_rerank_result_when_no_merge() -> None:
    plugin = FilterTopKPlugin()
    pipeline_ctx = PipelineContext(
        rerank_top_k=2,
        rerank_result=[
            _result(result_id="a", score=0.9),
            _result(result_id="b", score=0.8),
            _result(result_id="c", score=0.7),
        ],
    )

    await plugin.on_event(_CTX, EventType.FILTER_TOP_K, pipeline_ctx, _noop_next)

    assert [item.id for item in pipeline_ctx.rerank_result] == ["a", "b"]
    assert pipeline_ctx.search_result == []


async def test_filter_topk_uses_search_result_when_no_merge_or_rerank() -> None:
    plugin = FilterTopKPlugin()
    pipeline_ctx = PipelineContext(
        rerank_top_k=1,
        search_result=[
            _result(result_id="a", score=0.6),
            _result(result_id="b", score=0.9),
        ],
    )

    await plugin.on_event(_CTX, EventType.FILTER_TOP_K, pipeline_ctx, _noop_next)

    assert [item.id for item in pipeline_ctx.search_result] == ["b"]


async def test_filter_topk_keeps_results_when_within_top_k() -> None:
    plugin = FilterTopKPlugin()
    pipeline_ctx = PipelineContext(
        rerank_top_k=10,
        rerank_result=[
            _result(result_id="a", score=0.3),
            _result(result_id="b", score=0.2),
        ],
    )

    await plugin.on_event(_CTX, EventType.FILTER_TOP_K, pipeline_ctx, _noop_next)

    # Still deterministically sorted even without truncation.
    assert [item.id for item in pipeline_ctx.rerank_result] == ["a", "b"]


async def test_filter_topk_skips_when_no_results() -> None:
    plugin = FilterTopKPlugin()
    pipeline_ctx = PipelineContext(rerank_top_k=5)

    error = await plugin.on_event(_CTX, EventType.FILTER_TOP_K, pipeline_ctx, _noop_next)

    assert error is None


# ── WebFetchPlugin.on_event ────────────────────────────────────────────


async def test_web_fetch_skips_when_disabled() -> None:
    plugin = WebFetchPlugin(_FakeFetcher({}, {}))
    pipeline_ctx = PipelineContext(rerank_result=[_result(result_id="http://x", knowledge_source="web_search")])
    error = await plugin.on_event(_CTX, EventType.WEB_FETCH, pipeline_ctx, _noop_next)
    assert error is None
    assert pipeline_ctx.rerank_result[0].content == ""


async def test_web_fetch_skips_when_no_web_results() -> None:
    plugin = WebFetchPlugin(_FakeFetcher({}, {}))
    pipeline_ctx = PipelineContext(
        web_fetch_enabled=True,
        web_search_enabled=True,
        rerank_result=[_result(result_id="doc-1", knowledge_source="knowledge", content="snippet")],
    )
    error = await plugin.on_event(_CTX, EventType.WEB_FETCH, pipeline_ctx, _noop_next)
    assert error is None
    assert pipeline_ctx.rerank_result[0].content == "snippet"


async def test_web_fetch_replaces_web_snippets_with_full_content() -> None:
    fetcher = _FakeFetcher(
        {
            "https://a.example": "full page A text",
            "https://b.example": "full page B text",
        },
        {},
    )
    plugin = WebFetchPlugin(fetcher)
    pipeline_ctx = PipelineContext(
        web_fetch_enabled=True,
        web_search_enabled=True,
        web_fetch_top_n=5,
        rerank_result=[
            _result(result_id="doc-1", knowledge_source="knowledge", content="kb snippet"),
            _result(result_id="https://a.example", knowledge_source="web_search", content="web snippet A"),
            _result(result_id="https://b.example", knowledge_source="web_search", content="web snippet B"),
        ],
    )

    error = await plugin.on_event(_CTX, EventType.WEB_FETCH, pipeline_ctx, _noop_next)

    assert error is None
    assert fetcher.requested == ["https://a.example", "https://b.example"]
    assert pipeline_ctx.rerank_result[0].content == "kb snippet"  # untouched
    assert pipeline_ctx.rerank_result[1].content == "full page A text"
    assert pipeline_ctx.rerank_result[2].content == "full page B text"


async def test_web_fetch_truncates_long_content() -> None:
    long_text = "x" * 9000
    fetcher = _FakeFetcher({"https://a.example": long_text}, {})
    plugin = WebFetchPlugin(fetcher)
    pipeline_ctx = PipelineContext(
        web_fetch_enabled=True,
        web_search_enabled=True,
        rerank_result=[
            _result(result_id="https://a.example", knowledge_source="web_search", content="snippet")
        ],
    )

    await plugin.on_event(_CTX, EventType.WEB_FETCH, pipeline_ctx, _noop_next)

    content = pipeline_ctx.rerank_result[0].content
    assert len(content) == 8000 + len("\n...(truncated)")
    assert content.endswith("\n...(truncated)")


async def test_web_fetch_keeps_snippet_on_fetch_error() -> None:
    fetcher = _FakeFetcher({}, {"https://a.example": RuntimeError("dns failed")})
    plugin = WebFetchPlugin(fetcher)
    pipeline_ctx = PipelineContext(
        web_fetch_enabled=True,
        web_search_enabled=True,
        rerank_result=[
            _result(result_id="https://a.example", knowledge_source="web_search", content="snippet")
        ],
    )

    error = await plugin.on_event(_CTX, EventType.WEB_FETCH, pipeline_ctx, _noop_next)

    assert error is None
    assert pipeline_ctx.rerank_result[0].content == "snippet"


async def test_web_fetch_honours_top_n_and_skips_empty_url() -> None:
    fetcher = _FakeFetcher(
        {
            "https://a.example": "page A",
            "https://b.example": "page B",
            "https://c.example": "page C",
        },
        {},
    )
    plugin = WebFetchPlugin(fetcher)
    pipeline_ctx = PipelineContext(
        web_fetch_enabled=True,
        web_search_enabled=True,
        web_fetch_top_n=2,
        rerank_result=[
            _result(result_id="", knowledge_source="web_search", content="no url"),
            _result(result_id="https://a.example", knowledge_source="web_search", content="a"),
            _result(result_id="https://b.example", knowledge_source="web_search", content="b"),
            _result(result_id="https://c.example", knowledge_source="web_search", content="c"),
        ],
    )

    await plugin.on_event(_CTX, EventType.WEB_FETCH, pipeline_ctx, _noop_next)

    # The empty-URL result consumes a topN slot (faithful to the upstream
    # scan); the real URLs within the cap are fetched in order.
    assert fetcher.requested == ["https://a.example"]
    assert pipeline_ctx.rerank_result[0].content == "no url"
    assert pipeline_ctx.rerank_result[1].content == "page A"
    assert pipeline_ctx.rerank_result[3].content == "c"  # beyond topN untouched


# ── Integration: real DB chain ─────────────────────────────────────────


async def _chunk_rows(session: AsyncSession, tenant_id: int) -> list[SearchResult]:
    """Seed four real chunk rows and return them as search results."""
    rows = [
        ("c-low", "low relevance content", 0.2, 0),
        ("c-high", "high relevance content", 0.9, 1),
        ("c-medium", "medium relevance content", 0.5, 2),
        ("c-second", "second relevance content", 0.8, 3),
    ]
    repo = ChunkRepository(session)
    for chunk_id, content, _score, index in rows:
        await repo.create(
            Chunk(
                id=chunk_id,
                tenant_id=tenant_id,
                knowledge_base_id="kb-rerank-integration",
                knowledge_id="doc-rerank-integration",
                content=content,
                chunk_index=index,
                start_at=0,
                end_at=len(content),
                created_at=_NOW,
                updated_at=_NOW,
            )
        )
    return [
        _result(
            result_id=chunk_id,
            content=content,
            score=score,
            knowledge_id="doc-rerank-integration",
            chunk_index=index,
        )
        for chunk_id, content, score, index in rows
    ]


async def test_integration_rerank_then_filter_topk_chain(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    search_result = await _chunk_rows(session, tenant_id)

    # Rerank model hands back a fixed ranking over the seeded candidates.
    model = _FakeRerankModel(
        [
            [
                RankResult(index=2, relevance_score=0.9),
                RankResult(index=0, relevance_score=0.8),
                RankResult(index=3, relevance_score=0.7),
                RankResult(index=1, relevance_score=0.6),
            ]
        ]
    )
    manager = EventManager()
    manager.register(RerankPlugin(_FakeRerankService(model)))
    manager.register(FilterTopKPlugin())

    pipeline_ctx = PipelineContext(
        tenant_id=tenant_id,
        rerank_model_id="model-1",
        rerank_threshold=0.3,
        rerank_top_k=2,
        rewrite_query="integration query",
        search_result=search_result,
    )

    rerank_error = await manager.trigger(_CTX, EventType.CHUNK_RERANK, pipeline_ctx)
    assert rerank_error is None
    assert len(pipeline_ctx.rerank_result) == 2

    filter_error = await manager.trigger(_CTX, EventType.FILTER_TOP_K, pipeline_ctx)
    assert filter_error is None
    assert [item.id for item in pipeline_ctx.rerank_result] == ["c-medium", "c-second"]

    # The composite scores are visible on the shared search_result too.
    by_id = {item.id: item for item in pipeline_ctx.search_result}
    assert by_id["c-medium"].metadata["model_score"] == "0.9000"
    assert by_id["c-low"].metadata["model_score"] == "0.8000"
