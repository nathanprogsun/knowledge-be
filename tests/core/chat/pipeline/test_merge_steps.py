"""Unit + integration tests for the chat-pipeline chunk-merge steps.

Unit tests drive the merge helpers and the ``PluginMerge`` step with fake
chunk sources (no database). Integration tests run the merge step end to
end against the real applied schema, seeding ``chunks`` rows with an
int32-safe tenant id.
"""

from __future__ import annotations

import itertools
import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from random import randint

import pytest
import pytest_asyncio
from faker import Faker
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.ai.retrieval.types import MatchType
from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.engine import EventManager
from src.core.chat.pipeline.steps.merge import (
    PluginMerge,
    assign_scoped_image_info,
    collect_scoped_text_child_ids,
    group_and_merge_current_content,
    resolve_parent_chunks,
)
from src.core.chat.pipeline.steps.merge_expand import expand_short_context_with_neighbors
from src.core.chat.pipeline.steps.merge_faq import build_faq_answer_content
from src.core.chat.pipeline.steps.merge_history import filter_history_results
from src.core.chat.pipeline.steps.merge_overlap import (
    merge_image_info_json,
    merge_sequential_chunks,
)
from src.core.chat.pipeline.steps.merge_utils import (
    SqlChunkSource,
    contains_chunk_content,
    content_overlap_ratio,
    filter_image_info_by_content_urls,
    join_chunk_content,
    normalize_content,
    prune_markdown_images_by_image_info,
    remove_duplicate_results,
    remove_partial_overlaps,
)
from src.core.chat.pipeline.types import EventType, History, QueryIntent, SearchResult
from src.db.dao.chunk_repository import ChunkRepository
from src.db.models.chunk import Chunk
from src.settings import get_settings, reset_settings_cache
from tests.integration.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_FAKER_SEED_MAX = 100_000_000

# ``chunks.tenant_id`` is an INTEGER (32-bit) column; integration ids are
# minted from this counter so seeded rows never overflow.
_INT32_TENANT_BASE = 4_100_000
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


@dataclass(frozen=True, slots=True)
class _FakeContext:
    """Opaque execution context (empty structural protocol)."""

    is_background_task: bool = False


class _FakeChunkSource:
    """In-memory chunk source for unit tests (mirrors the Go test fake)."""

    def __init__(
        self,
        chunks: dict[str, Chunk] | None = None,
        children: dict[str, list[Chunk]] | None = None,
    ) -> None:
        self._chunks = chunks or {}
        self._children = children or {}

    async def list_chunks_by_ids(
        self,
        _ctx: _FakeContext,
        _tenant_id: int,
        ids: list[str],
    ) -> list[Chunk]:
        return [self._chunks[chunk_id] for chunk_id in ids if chunk_id in self._chunks]

    async def list_chunks_by_parent_ids(
        self,
        _ctx: _FakeContext,
        _tenant_id: int,
        parent_ids: list[str],
    ) -> list[Chunk]:
        out: list[Chunk] = []
        for parent_id in parent_ids:
            out.extend(self._children.get(parent_id, []))
        return out


def _make_chunk(
    *,
    chunk_id: str,
    content: str = "",
    chunk_index: int = 0,
    chunk_type: str = "text",
    parent_chunk_id: str | None = None,
    pre_chunk_id: str | None = None,
    next_chunk_id: str | None = None,
    image_info: str | None = None,
    metadata: dict[str, object] | None = None,
    knowledge_id: str = "doc",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        tenant_id=1,
        knowledge_base_id="kb",
        knowledge_id=knowledge_id,
        content=content,
        chunk_index=chunk_index,
        start_at=0,
        end_at=len(content),
        pre_chunk_id=pre_chunk_id,
        next_chunk_id=next_chunk_id,
        chunk_type=chunk_type,
        parent_chunk_id=parent_chunk_id,
        image_info=image_info,
        metadata=metadata,
        created_at=_NOW,
        updated_at=_NOW,
    )


async def _noop_next() -> None:
    return None


# ── join / containment helpers ─────────────────────────────────────────


def test_join_chunk_content_removes_suffix_prefix_overlap() -> None:
    left = "the quick brown fox jumps over the lazy dog and the"
    right = "the lazy dog and the clever cat"
    joined = join_chunk_content(left, right, "\n\n")
    assert joined == left + right[len("the lazy dog and the") :]
    assert right[len("the lazy dog and the") :] in joined


def test_join_chunk_content_collapses_exact_containment() -> None:
    long_text = "a fairly long body of text with meaningful length for overlap"
    assert join_chunk_content(long_text, long_text, "\n\n") == long_text
    assert join_chunk_content(long_text, "", "\n\n") == long_text
    assert join_chunk_content("", long_text, "\n\n") == long_text


def test_join_chunk_content_falls_back_to_separator() -> None:
    left = "first independent body"
    right = "second unrelated body"
    assert join_chunk_content(left, right, "\n\n") == left + "\n\n" + right


def test_contains_chunk_content_requires_min_length() -> None:
    container = "the quick brown fox jumps over the lazy dog"
    assert contains_chunk_content(container, "brown fox jumps") is True
    # Short substrings are not treated as containment.
    assert contains_chunk_content(container, "fox") is False


def test_normalize_content_and_containment() -> None:
    assert normalize_content("  Hello   World\n") == "hello world"
    short = normalize_content("hello world")
    long_ = normalize_content("Hello world, welcome back")
    assert short in long_


def test_content_overlap_ratio() -> None:
    assert content_overlap_ratio("alpha beta gamma", "alpha beta gamma") == 1.0
    assert content_overlap_ratio("alpha beta gamma", "alpha beta omega") < 1.0
    assert content_overlap_ratio("", "alpha beta gamma") == 0.0


# ── deduplication ──────────────────────────────────────────────────────


def test_remove_duplicate_results_by_id_and_signature() -> None:
    results = [
        SearchResult(id="a", content="same content body"),
        SearchResult(id="b", content="same content body"),
        SearchResult(id="a", content="different body"),
    ]
    out = remove_duplicate_results(results)
    assert [result.id for result in out] == ["a"]


def test_remove_partial_overlaps_drops_contained_and_high_overlap() -> None:
    long_result = SearchResult(id="long", content="alpha beta gamma delta epsilon", score=0.9)
    short_result = SearchResult(id="short", content="alpha beta gamma", score=0.2)
    unrelated = SearchResult(id="unrelated", content="totally different topic here", score=0.5)
    out = remove_partial_overlaps([long_result, short_result, unrelated])
    assert [result.id for result in out] == ["long", "unrelated"]


def test_remove_partial_overlaps_keeps_both_when_below_threshold() -> None:
    left = SearchResult(id="left", content="alpha beta gamma", score=0.8)
    right = SearchResult(id="right", content="alpha beta omega", score=0.7)
    out = remove_partial_overlaps([left, right])
    assert [result.id for result in out] == ["left", "right"]


# ── image_info helpers ─────────────────────────────────────────────────


def test_filter_image_info_by_content_urls() -> None:
    infos = json.dumps([{"url": "u1", "ocr_text": "one"}, {"url": "u2", "ocr_text": "two"}])
    content = "![p2](u2)"
    assert filter_image_info_by_content_urls(content, infos) == json.dumps(
        [{"url": "u2", "ocr_text": "two"}], ensure_ascii=False
    )


def test_prune_markdown_images_by_image_info() -> None:
    content = "![a](u1) keep\n\n![b](u2) drop"
    infos = json.dumps([{"url": "u1"}])
    pruned = prune_markdown_images_by_image_info(content, infos)
    assert "u1" in pruned
    assert "u2" not in pruned


def test_merge_image_info_json_dedup_by_url() -> None:
    target = json.dumps([{"url": "u1", "caption": "first"}])
    source = json.dumps([{"url": "u1", "caption": "updated"}, {"url": "u2", "ocr_text": "two"}])
    merged, error = merge_image_info_json(target, source)
    assert error is False
    decoded = json.loads(merged)
    assert [entry["url"] for entry in decoded] == ["u1", "u2"]
    assert decoded[0]["caption"] == "first"


# ── sequential merge ───────────────────────────────────────────────────


def test_merge_sequential_chunks_merges_contained_or_sequential() -> None:
    chunks = [
        SearchResult(
            id="outer",
            knowledge_id="doc",
            chunk_type="text",
            chunk_index=1,
            content="completely rewritten outer content",
            score=0.4,
        ),
        SearchResult(
            id="inner",
            knowledge_id="doc",
            chunk_type="text",
            chunk_index=2,
            content="independent current inner content",
            score=0.9,
        ),
    ]
    merged = merge_sequential_chunks(_FakeContext(), "doc", chunks)
    assert len(merged) == 1
    assert "completely rewritten outer content" in merged[0].content
    assert "independent current inner content" in merged[0].content
    assert merged[0].score == 0.9
    assert merged[0].sub_chunk_id == ["inner"]


def test_merge_sequential_chunks_does_not_merge_gapped_chunks() -> None:
    chunks = [
        SearchResult(id="one", knowledge_id="doc", chunk_type="text", chunk_index=1, content="one"),
        SearchResult(
            id="three", knowledge_id="doc", chunk_type="text", chunk_index=3, content="three"
        ),
    ]
    merged = merge_sequential_chunks(_FakeContext(), "doc", chunks)
    assert len(merged) == 2


def test_merge_sequential_chunks_sorts_by_score_descending() -> None:
    chunks = [
        SearchResult(
            id="low",
            knowledge_id="doc",
            chunk_type="text",
            chunk_index=0,
            content="low body",
            score=0.4,
        ),
        SearchResult(
            id="high",
            knowledge_id="doc",
            chunk_type="text",
            chunk_index=2,
            content="high body",
            score=0.9,
        ),
    ]
    merged = merge_sequential_chunks(_FakeContext(), "doc", chunks)
    assert [result.id for result in merged] == ["high", "low"]


# ── grouping + deterministic order ─────────────────────────────────────


async def test_group_and_merge_deterministic_order() -> None:
    chunks = [
        SearchResult(id="low", knowledge_id="kb-001", chunk_type="text", content="low", score=0.4),
        SearchResult(
            id="high", knowledge_id="kb-001", chunk_type="summary", content="high", score=0.9
        ),
        SearchResult(
            id="mid", knowledge_id="kb-001", chunk_type="parent_text", content="mid", score=0.6
        ),
    ]
    results = await group_and_merge_current_content(_FakeContext(), chunks)
    assert [result.id for result in results] == ["high", "mid", "low"]


async def test_group_and_merge_uses_knowledge_id_as_tie_breaker() -> None:
    chunks = [
        SearchResult(id="ab", knowledge_id="kb-ab", chunk_type="text", content="ab", score=0.8),
        SearchResult(id="aa", knowledge_id="kb-aa", chunk_type="text", content="aa", score=0.8),
    ]
    results = await group_and_merge_current_content(_FakeContext(), chunks)
    assert [result.id for result in results] == ["aa", "ab"]


async def test_group_and_merge_does_not_merge_across_knowledge() -> None:
    chunks = [
        SearchResult(id="a-1", knowledge_id="kb-a", chunk_type="text", content="a", score=0.9),
        SearchResult(id="b-1", knowledge_id="kb-b", chunk_type="text", content="b", score=0.5),
    ]
    results = await group_and_merge_current_content(_FakeContext(), chunks)
    assert [result.id for result in results] == ["a-1", "b-1"]


# ── history injection ──────────────────────────────────────────────────


def test_filter_history_results_marks_and_discounts() -> None:
    pipeline_ctx = PipelineContext(
        query="how to reset my password",
        history=[
            History(
                query="old",
                answer="old",
                references=[SearchResult(id="h1", content="reset password steps here", score=1.0)],
            )
        ],
    )
    out = filter_history_results(pipeline_ctx, [])
    assert len(out) == 1
    assert out[0].id == "h1"
    assert out[0].match_type == MatchType.HISTORY
    assert out[0].score == pytest.approx(0.6)
    assert out[0].metadata["history_similarity"] == "0.2857"


def test_filter_history_results_excludes_existing_and_caps() -> None:
    pipeline_ctx = PipelineContext(
        query="how to reset my password",
        history=[
            History(
                query="old",
                answer="old",
                references=[
                    SearchResult(id="h1", content="reset password steps here", score=1.0),
                    SearchResult(id="h2", content="reset password steps here", score=1.0),
                    SearchResult(id="h3", content="reset password steps here", score=1.0),
                    SearchResult(id="h4", content="reset password steps here", score=1.0),
                    SearchResult(id="h5", content="reset password steps here", score=1.0),
                ],
            )
        ],
    )
    current = [SearchResult(id="h1", content="reset password steps here", score=1.0)]
    out = filter_history_results(pipeline_ctx, current)
    assert [result.id for result in out] == ["h2", "h3", "h4"]


def test_filter_history_results_drops_unrelated_references() -> None:
    pipeline_ctx = PipelineContext(
        query="how to reset my password",
        history=[
            History(
                query="old",
                answer="old",
                references=[SearchResult(id="h1", content="completely unrelated topic", score=1.0)],
            )
        ],
    )
    assert filter_history_results(pipeline_ctx, []) == []


# ── FAQ enrichment ─────────────────────────────────────────────────────


def test_build_faq_answer_content() -> None:
    content = build_faq_answer_content(
        {
            "standard_question": "How to reset?",
            "answers": ["  Go to settings  ", "", "Press restart"],
        }
    )
    assert content == "Q: How to reset?\nAnswer:\n- Go to settings\n- Press restart"


def test_build_faq_answer_content_empty() -> None:
    assert build_faq_answer_content(None) == ""
    assert build_faq_answer_content({"standard_question": "", "answers": []}) == ""


# ── short-context expansion ────────────────────────────────────────────


async def test_expand_short_context_keeps_source_coordinates() -> None:
    fake = _FakeChunkSource(
        chunks={
            "prev": _make_chunk(
                chunk_id="prev",
                content="edited previous body",
                next_chunk_id="base",
            ),
            "base": _make_chunk(
                chunk_id="base",
                content="edited base body",
                pre_chunk_id="prev",
                next_chunk_id="next",
            ),
            "next": _make_chunk(
                chunk_id="next",
                content="edited next body",
                pre_chunk_id="base",
            ),
        }
    )
    result = SearchResult(
        id="base",
        knowledge_id="doc",
        chunk_type="text",
        content="edited base body",
        start_at=100,
        end_at=120,
    )
    got = await expand_short_context_with_neighbors(_FakeContext(), fake, 1, [result])
    assert len(got) == 1
    assert got[0].start_at == 100
    assert got[0].end_at == 120
    for body in ("edited previous body", "edited base body", "edited next body"):
        assert body in got[0].content


async def test_expand_short_context_skips_non_text() -> None:
    fake = _FakeChunkSource(chunks={})
    result = SearchResult(id="f1", knowledge_id="doc", chunk_type="faq", content="short")
    got = await expand_short_context_with_neighbors(_FakeContext(), fake, 1, [result])
    assert got == [result]


# ── parent-child resolution ────────────────────────────────────────────


async def test_resolve_parent_chunks_uses_current_content_and_image_urls() -> None:
    parent = _make_chunk(
        chunk_id="parent",
        content="manually inserted prefix\n\n![one](u1)\n\nparent body\n\n![two](u2)",
        chunk_type="parent_text",
        chunk_index=7,
    )
    fake = _FakeChunkSource(
        chunks={"parent": parent},
        children={
            "child": [
                _make_chunk(
                    chunk_id="image",
                    chunk_type="image_ocr",
                    parent_chunk_id="child",
                    image_info=json.dumps([{"url": "u2", "ocr_text": "two"}]),
                )
            ]
        },
    )
    result = SearchResult(
        id="child",
        knowledge_id="doc",
        chunk_type="text",
        parent_chunk_id="parent",
        content="current edited child body",
        start_at=999,
        end_at=1001,
    )
    got = await resolve_parent_chunks(_FakeContext(), fake, 1, [result])
    assert len(got) == 1
    assert "current edited child body" in got[0].content
    assert "u2" in got[0].content
    assert "u1" not in got[0].content
    assert got[0].start_at == 999
    assert got[0].end_at == 1001


async def test_resolve_image_chunk_keeps_grandparent_context() -> None:
    image_info = json.dumps([{"url": "u1", "ocr_text": "matched image"}])
    fake = _FakeChunkSource(
        chunks={
            "text": _make_chunk(
                chunk_id="text",
                content="current edited text child\n\n![matched](u1)",
                chunk_type="text",
                parent_chunk_id="parent",
                chunk_index=4,
            ),
            "parent": _make_chunk(
                chunk_id="parent",
                content=(
                    "grandparent context before\n\n![matched](u1)\n\n"
                    "grandparent context after\n\n![sibling](u2)"
                ),
                chunk_type="parent_text",
            ),
        },
        children={
            "text": [
                _make_chunk(
                    chunk_id="image",
                    chunk_type="image_ocr",
                    parent_chunk_id="text",
                    image_info=image_info,
                )
            ]
        },
    )
    result = SearchResult(
        id="image",
        knowledge_id="doc",
        chunk_type="image_ocr",
        parent_chunk_id="text",
        content="matched image",
        image_info=image_info,
        start_at=500,
        end_at=510,
    )
    got = await resolve_parent_chunks(_FakeContext(), fake, 1, [result])
    assert len(got) == 1
    for wanted in (
        "grandparent context before",
        "grandparent context after",
        "current edited text child",
        "u1",
    ):
        assert wanted in got[0].content
    assert "u2" not in got[0].content
    assert got[0].start_at == 500
    assert got[0].end_at == 510


async def test_collect_scoped_text_child_ids() -> None:
    parent_map = {
        "parent-1": _make_chunk(chunk_id="parent-1", content="x", chunk_type="parent_text"),
        "text-x": _make_chunk(chunk_id="text-x", content="x", chunk_type="text"),
    }
    results = [
        SearchResult(id="text-1", chunk_type="text", parent_chunk_id="parent-1"),
        SearchResult(id="img-1", chunk_type="image_ocr", parent_chunk_id="text-2"),
        SearchResult(id="text-3", chunk_type="text", parent_chunk_id="text-x"),
    ]
    ids = collect_scoped_text_child_ids(results, parent_map)
    assert sorted(ids) == ["text-1", "text-2"]


async def test_assign_scoped_image_info_filters_to_content_urls() -> None:
    all_infos = json.dumps([{"url": "u1", "ocr_text": "one"}, {"url": "u2", "ocr_text": "two"}])
    result = SearchResult(content="![p2](u2)", image_info=all_infos)
    scoped = assign_scoped_image_info(result, None, "missing-child")
    decoded = json.loads(scoped.image_info)
    assert [entry["url"] for entry in decoded] == ["u2"]


# ── PluginMerge orchestration ──────────────────────────────────────────


async def test_plugin_merge_skips_when_no_retrieval() -> None:
    called: list[bool] = []
    plugin = PluginMerge()
    pipeline_ctx = PipelineContext(intent=QueryIntent.CHITCHAT)

    async def _next() -> None:
        called.append(True)

    result = await plugin.on_event(_FakeContext(), EventType.CHUNK_MERGE, pipeline_ctx, _next)
    assert result is None
    assert called == [True]
    assert pipeline_ctx.merge_result == []


async def test_plugin_merge_no_candidates_proceeds() -> None:
    called: list[bool] = []
    plugin = PluginMerge()
    pipeline_ctx = PipelineContext(tenant_id=1)

    async def _next() -> None:
        called.append(True)

    result = await plugin.on_event(_FakeContext(), EventType.CHUNK_MERGE, pipeline_ctx, _next)
    assert result is None
    assert called == [True]
    assert pipeline_ctx.merge_result == []


async def test_plugin_merge_full_orchestration() -> None:
    parent_content = "parent body " * 60  # long enough to skip short-context expansion
    fake = _FakeChunkSource(
        chunks={
            "parent": _make_chunk(
                chunk_id="parent", content=parent_content, chunk_type="parent_text"
            )
        },
        children={},
    )
    plugin = PluginMerge(fake)
    pipeline_ctx = PipelineContext(
        tenant_id=1,
        rerank_result=[
            SearchResult(
                id="child",
                knowledge_id="doc",
                chunk_type="text",
                parent_chunk_id="parent",
                content="current child body",
                score=0.9,
            )
        ],
    )
    result = await plugin.on_event(_FakeContext(), EventType.CHUNK_MERGE, pipeline_ctx, _noop_next)
    assert result is None
    assert len(pipeline_ctx.merge_result) == 1
    merged = pipeline_ctx.merge_result[0]
    assert parent_content.strip() in merged.content
    assert "current child body" in merged.content
    assert merged.sub_chunk_id == ["child"]


def test_plugin_merge_activation_events() -> None:
    assert PluginMerge().activation_events() == [EventType.CHUNK_MERGE]


# ── Integration: real DB ───────────────────────────────────────────────


async def test_integration_merge_resolves_parent_and_enriches(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    parent_id = str(uuid.uuid4())
    child_id = str(uuid.uuid4())
    repo = ChunkRepository(session)
    parent_content = "parent context body " * 60
    await repo.create_many(
        [
            Chunk(
                id=parent_id,
                tenant_id=tenant_id,
                knowledge_base_id="kb-merge",
                knowledge_id="doc-merge",
                content=parent_content,
                chunk_index=0,
                start_at=0,
                end_at=len(parent_content),
                chunk_type="parent_text",
                created_at=_NOW,
                updated_at=_NOW,
            ),
            Chunk(
                id=child_id,
                tenant_id=tenant_id,
                knowledge_base_id="kb-merge",
                knowledge_id="doc-merge",
                content="child body",
                chunk_index=1,
                start_at=0,
                end_at=10,
                chunk_type="text",
                parent_chunk_id=parent_id,
                created_at=_NOW,
                updated_at=_NOW,
            ),
        ]
    )
    plugin = PluginMerge(SqlChunkSource(repo))
    pipeline_ctx = PipelineContext(
        tenant_id=tenant_id,
        knowledge_base_ids=["kb-merge"],
        rerank_result=[
            SearchResult(
                id=child_id,
                knowledge_id="doc-merge",
                chunk_type="text",
                parent_chunk_id=parent_id,
                content="child body",
                score=0.9,
            )
        ],
    )
    result = await plugin.on_event(_FakeContext(), EventType.CHUNK_MERGE, pipeline_ctx, _noop_next)
    assert result is None
    assert len(pipeline_ctx.merge_result) == 1
    merged = pipeline_ctx.merge_result[0]
    assert "child body" in merged.content
    assert parent_content.strip() in merged.content


async def test_integration_merge_enriches_faq_content(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    faq_id = str(uuid.uuid4())
    repo = ChunkRepository(session)
    faq_content = "faq row body"
    await repo.create(
        Chunk(
            id=faq_id,
            tenant_id=tenant_id,
            knowledge_base_id="kb-faq",
            knowledge_id="faq-doc",
            content=faq_content,
            chunk_index=0,
            start_at=0,
            end_at=len(faq_content),
            chunk_type="faq",
            metadata={
                "standard_question": "How to reset?",
                "answers": ["Go to settings", "Restart the device"],
            },
            created_at=_NOW,
            updated_at=_NOW,
        )
    )
    plugin = PluginMerge(SqlChunkSource(repo))
    pipeline_ctx = PipelineContext(
        tenant_id=tenant_id,
        knowledge_base_ids=["kb-faq"],
        rerank_result=[
            SearchResult(
                id=faq_id,
                knowledge_id="faq-doc",
                chunk_type="faq",
                content=faq_content,
                score=0.9,
            )
        ],
    )
    result = await plugin.on_event(_FakeContext(), EventType.CHUNK_MERGE, pipeline_ctx, _noop_next)
    assert result is None
    assert len(pipeline_ctx.merge_result) == 1
    merged = pipeline_ctx.merge_result[0]
    assert merged.content == "Q: How to reset?\nAnswer:\n- Go to settings\n- Restart the device"


async def test_integration_merge_wires_into_event_manager(session: AsyncSession) -> None:
    """The merge plugin registers and runs through the engine chain."""
    tenant_id = _int32_tenant_id()
    chunk_id = str(uuid.uuid4())
    repo = ChunkRepository(session)
    body = "standalone merge body"
    await repo.create(
        Chunk(
            id=chunk_id,
            tenant_id=tenant_id,
            knowledge_base_id="kb-chain",
            knowledge_id="chain-doc",
            content=body,
            chunk_index=0,
            start_at=0,
            end_at=len(body),
            chunk_type="text",
            created_at=_NOW,
            updated_at=_NOW,
        )
    )
    manager = EventManager()
    plugin = PluginMerge(SqlChunkSource(repo))
    manager.register(plugin)
    pipeline_ctx = PipelineContext(
        tenant_id=tenant_id,
        rerank_result=[
            SearchResult(
                id=chunk_id,
                knowledge_id="chain-doc",
                chunk_type="text",
                content=body,
                score=0.9,
            )
        ],
    )
    result = await manager.trigger(_FakeContext(), EventType.CHUNK_MERGE, pipeline_ctx)
    assert result is None
    assert pipeline_ctx.merge_result[0].content == body


def test_make_test_tenant_id_is_positive() -> None:
    assert make_test_tenant_id() > 0
