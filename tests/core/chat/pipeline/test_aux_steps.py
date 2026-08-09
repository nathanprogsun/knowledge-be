"""Unit + integration tests for the auxiliary pipeline steps.

Unit tests cover the references, progress, history-loading and wiki-boost
helpers / plugins with recording doubles (no database, no async services).
Integration tests run the wiki-boost plugin end to end against the real
applied schema: a ``knowledge_bases`` row (wiki-enabled strategy) and real
``chunks`` rows (seeded with an int32-safe tenant id) drive the plugin
through the ``EventManager`` chain.
"""

from __future__ import annotations

import itertools
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from random import randint

import pytest
import pytest_asyncio
from faker import Faker
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.ai.llm.types import Message
from src.common.exception import NotFoundError
from src.common.json import JsonObject
from src.core.agents.engine.modelcontext.registry import Registry
from src.core.chat.bus import Event
from src.core.chat.pipeline.common import ChatMessage, StoredMessage
from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.engine import (
    ERR_SEARCH_NOTHING,
    EventManager,
    Next,
    PluginError,
)
from src.core.chat.pipeline.steps import (
    WIKI_BOOST_FACTOR,
    LoadHistoryPlugin,
    WikiBoostPlugin,
    begin_query_understand_progress,
    begin_retrieval_progress,
    end_query_understand_progress,
    end_retrieval_progress,
    enrich_content_with_image_info_for_chat,
    first_pipeline_title,
    get_enriched_passage_for_chat,
    is_consolidated_retrieval_stage,
    is_pipeline_web_reference,
    last_consolidated_retrieval_stage,
    ordered_pipeline_references,
    prepare_messages_with_model_context,
    should_close_retrieval_progress,
    should_emit_query_understand_progress,
)
from src.core.chat.pipeline.types import (
    Context,
    EventType,
    History,
    MessageAttachment,
    MessageImage,
    SearchResult,
    SearchTarget,
    SummaryConfig,
)
from src.core.chat.types import EventType as ChatEventType
from src.core.knowledge.chunks.types import CHUNK_TYPE_WIKI_PAGE
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.models.chunk import Chunk
from src.db.models.knowledge_base import KnowledgeBase
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


@dataclass(frozen=True, slots=True)
class _FakeContext:
    """Opaque execution context (empty structural protocol)."""

    is_background_task: bool = False


class _RecordingEventBus:
    """Event bus that records every emitted event."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)


class _FailingEventBus(_RecordingEventBus):
    """Event bus whose ``emit`` always raises (progress must not fail)."""

    async def emit(self, event: Event) -> None:
        self.events.append(event)
        raise RuntimeError("bus down")


@dataclass
class _StoredMessage:
    """Minimal structural ``StoredMessage`` for history-loading tests."""

    request_id: str
    role: str
    content: str
    created_at: datetime | None = None
    images: Sequence[MessageImage] = ()
    attachments: Sequence[MessageAttachment] = ()
    knowledge_references: Sequence[SearchResult] = ()


class _FakeMessageService:
    """Message service returning fixed stored messages or an error."""

    def __init__(
        self,
        stored: Sequence[StoredMessage],
        error: Exception | None = None,
    ) -> None:
        self._stored = stored
        self._error = error
        self.fetch_count: int | None = None

    async def get_recent_messages_by_session(
        self,
        ctx: object,
        session_id: str,
        count: int,
    ) -> Sequence[StoredMessage]:
        self.fetch_count = count
        if self._error is not None:
            raise self._error
        return self._stored


class _FakeKbService:
    """Knowledge-base service returning wiki-enabled KBs for known ids."""

    def __init__(self, wiki_kb_ids: set[str], missing: bool = False) -> None:
        self._wiki_kb_ids = wiki_kb_ids
        self._missing = missing

    async def get_knowledge_base_by_id_only(self, *, knowledge_base_id: str) -> KnowledgeBaseInfo:
        if self._missing:
            raise NotFoundError(
                code="knowledge_base.not_found",
                message=f"knowledge base {knowledge_base_id} not found",
            )
        wiki_enabled = knowledge_base_id in self._wiki_kb_ids
        strategy: JsonObject = {"vector_enabled": True}
        if wiki_enabled:
            strategy["wiki_enabled"] = True
        return KnowledgeBaseInfo(
            id=knowledge_base_id,
            name=f"KB {knowledge_base_id}",
            tenant_id=1,
            created_at=_NOW,
            updated_at=_NOW,
            indexing_strategy=strategy,
        )


class _RecordingNext:
    """Records invocations of the ``next()`` chain callback."""

    def __init__(self) -> None:
        self.invocations = 0
        self.error: PluginError | None = None

    async def run(self) -> PluginError | None:
        self.invocations += 1
        return self.error


def _to_llm_message(message: ChatMessage) -> Message:
    """Project a pipeline chat message onto the registry's message shape."""
    return Message(role=message.role, content=message.content, images=list(message.images))


def _event_data(event: Event) -> JsonObject:
    """Return the event payload narrowed to a JSON object (empty when absent)."""
    data = event.data
    if not isinstance(data, dict):
        return {}
    return data


# ── References: pure helpers ───────────────────────────────────────────


def test_first_pipeline_title_prefers_knowledge_title() -> None:
    result = SearchResult(knowledge_title="Title", knowledge_filename="file.md")
    assert first_pipeline_title(result) == "Title"
    assert first_pipeline_title(SearchResult(knowledge_filename="file.md")) == "file.md"
    assert first_pipeline_title(SearchResult()) == ""


def test_is_pipeline_web_reference_matches_chunk_type_or_source() -> None:
    assert is_pipeline_web_reference(SearchResult(chunk_type="WEB_SEARCH"))
    assert is_pipeline_web_reference(SearchResult(knowledge_source="Web_Search"))
    assert not is_pipeline_web_reference(SearchResult(chunk_type="text"))
    assert not is_pipeline_web_reference(SearchResult())


def test_ordered_pipeline_references_faq_priority() -> None:
    faq = SearchResult(id="faq-1", chunk_type="faq")
    text = SearchResult(id="text-1", chunk_type="text")
    wiki = SearchResult(id="wiki-1", chunk_type="wiki_page")
    ctx = PipelineContext(faq_priority_enabled=True, merge_result=[text, faq, wiki])
    assert ordered_pipeline_references(ctx) == [faq, text, wiki]


def test_ordered_pipeline_references_no_faq_priority_keeps_order() -> None:
    first = SearchResult(id="a")
    second = SearchResult(id="b")
    ctx = PipelineContext(faq_priority_enabled=False, merge_result=[first, second])
    assert ordered_pipeline_references(ctx) == [first, second]


def test_get_enriched_passage_for_chat_passthrough_cases() -> None:
    assert get_enriched_passage_for_chat("", "") == ""
    assert get_enriched_passage_for_chat("plain text", "") == "plain text"


def test_enrich_content_with_image_info_for_chat_inline_caption() -> None:
    content = "Before ![alt](https://img.example/a.png) after"
    image_info = '[{"url": "https://img.example/a.png", "caption": "A chart"}]'
    enriched = enrich_content_with_image_info_for_chat(content, image_info)
    assert "**Image caption:** A chart" in enriched
    # The image itself stays Markdown for chat context.
    assert "![alt](https://img.example/a.png)" in enriched
    assert enriched.index("**Image caption:** A chart") > enriched.index("https://img.example/a.png")


def test_enrich_content_with_image_info_for_chat_unmatched_image_unchanged() -> None:
    content = "![alt](https://img.example/other.png)"
    image_info = '[{"url": "https://img.example/a.png", "caption": "A chart"}]'
    assert enrich_content_with_image_info_for_chat(content, image_info) == content


def test_enrich_content_with_image_info_for_chat_html_src() -> None:
    content = 'Before <img src="https://img.example/x.png" alt="x"> after'
    image_info = '[{"url": "https://img.example/x.png", "ocr_text": "OCR line"}]'
    enriched = enrich_content_with_image_info_for_chat(content, image_info)
    assert "**Image text (OCR):** OCR line" in enriched


def test_enrich_content_with_image_info_for_chat_invalid_json_unchanged() -> None:
    content = "![alt](https://img.example/a.png)"
    assert enrich_content_with_image_info_for_chat(content, "not-json") == content


# ── References: prepare_messages_with_model_context ────────────────────


def _chunk_centric_context() -> PipelineContext:
    rendered = '<context id="1">first content</context><context id="2">second content</context>'
    return PipelineContext(
        query="question",
        summary_config=SummaryConfig(prompt="system"),
        rendered_contexts=rendered,
        user_content="References:\n" + rendered + "\nQuestion: question",
        merge_result=[
            SearchResult(
                id="chunk-1",
                knowledge_id="doc-1",
                knowledge_base_id="kb-1",
                knowledge_title="Doc",
                chunk_index=1,
                content="first content",
            ),
            SearchResult(
                id="chunk-2",
                knowledge_id="doc-1",
                knowledge_base_id="kb-1",
                knowledge_title="Doc",
                chunk_index=2,
                content="second content",
            ),
        ],
    )


def test_prepare_messages_with_model_context_uses_chunk_centric_context() -> None:
    pipeline_ctx = _chunk_centric_context()
    messages, registry = prepare_messages_with_model_context(pipeline_ctx)

    assert len(messages) == 2
    assert "Source handling protocol" in messages[0].content
    assert '<document id="d1" kb="b1" title="Doc">' in messages[1].content
    assert '<chunk id="c1" index="1" view="full">' in messages[1].content
    assert '<chunk id="c2" index="2" view="full">' in messages[1].content
    assert "chunk-1" not in messages[1].content
    assert (
        registry.decode_output_text('<ref id="c1"/>')
        == '<kb doc="Doc" chunk_id="chunk-1" kb_id="kb-1" />'
    )


def test_prepare_messages_with_model_context_replaces_placeholder_and_history() -> None:
    rendered = '<context id="1">first content</context>'
    pipeline_ctx = PipelineContext(
        query="question",
        summary_config=SummaryConfig(prompt="System references: {{contexts}}"),
        rendered_contexts=rendered,
        user_content="Question: question",
        history=[
            History(
                query="previous",
                answer='Previous <kb doc="Old" chunk_id="old-chunk" kb_id="old-kb" />',
            )
        ],
        merge_result=[
            SearchResult(
                id="current-chunk",
                knowledge_id="current-doc",
                knowledge_base_id="current-kb",
                knowledge_title="Current",
                content="first content",
            )
        ],
    )

    messages, registry = prepare_messages_with_model_context(pipeline_ctx)
    encoded = registry.encode_messages([_to_llm_message(m) for m in messages])

    assert rendered not in encoded[0].content
    assert '<chunk id="c1"' in encoded[0].content
    assert '<ref id="c2"/>' in encoded[2].content
    assert "old-chunk" not in encoded[2].content


def test_prepare_messages_with_model_context_keeps_web_separate() -> None:
    pipeline_ctx = PipelineContext(
        query="question",
        summary_config=SummaryConfig(prompt="system"),
        user_content="question",
        merge_result=[
            SearchResult(
                id="https://example.com/page",
                knowledge_title="Example",
                content="web content",
                chunk_type="web_search",
                knowledge_source="web_search",
            )
        ],
    )

    messages, registry = prepare_messages_with_model_context(pipeline_ctx)

    assert '<retrieval type="web" mode="search">' in messages[1].content
    assert '<page id="w1" title="Example">' in messages[1].content
    assert "<chunk id=" not in messages[1].content
    assert (
        registry.decode_output_text('<ref id="w1"/>')
        == '<web url="https://example.com/page" title="Example" />'
    )


def test_prepare_messages_with_model_context_compacts_history_without_retrieval() -> None:
    pipeline_ctx = PipelineContext(
        query="follow-up",
        summary_config=SummaryConfig(prompt="system"),
        user_content="follow-up",
        history=[
            History(
                query="previous",
                answer='Previous <web url="https://example.com/old" title="Old" />',
            )
        ],
    )

    messages, registry = prepare_messages_with_model_context(pipeline_ctx)
    encoded = registry.encode_messages([_to_llm_message(m) for m in messages])

    assert "Source handling protocol" in encoded[0].content
    assert '<ref id="w1"/>' in encoded[2].content
    assert "https://example.com/old" not in encoded[2].content


def test_prepare_messages_with_model_context_suppresses_citations_when_disabled() -> None:
    pipeline_ctx = PipelineContext(
        query="question",
        citation_enabled=False,
        summary_config=SummaryConfig(prompt="custom system prompt"),
        user_content="question",
        merge_result=[
            SearchResult(
                id="chunk-1",
                knowledge_id="doc-1",
                knowledge_base_id="kb-1",
                knowledge_title="Doc",
                content="evidence",
            )
        ],
    )

    messages, registry = prepare_messages_with_model_context(pipeline_ctx)

    assert "Source citations are disabled" in messages[0].content
    assert '<chunk id="c1"' in messages[1].content
    assert "chunk-1" not in messages[1].content
    assert registry.decode_output_text('answer <ref id="c1"/>') == "answer "


def test_prepare_messages_with_model_context_empty_merge_returns_messages() -> None:
    pipeline_ctx = PipelineContext(
        query="hi",
        summary_config=SummaryConfig(prompt="system"),
        user_content="hi",
    )
    messages, registry = prepare_messages_with_model_context(pipeline_ctx)
    assert len(messages) == 2
    assert isinstance(registry, Registry)


# ── Progress: stage classification ─────────────────────────────────────


def test_is_consolidated_retrieval_stage() -> None:
    pipeline_ctx = PipelineContext()
    assert is_consolidated_retrieval_stage(EventType.CHUNK_SEARCH_PARALLEL, pipeline_ctx)
    assert not is_consolidated_retrieval_stage(EventType.QUERY_UNDERSTAND, pipeline_ctx)
    assert not is_consolidated_retrieval_stage(EventType.LOAD_HISTORY, pipeline_ctx)


def test_last_consolidated_retrieval_stage() -> None:
    pipeline_ctx = PipelineContext()
    pipeline = [
        EventType.LOAD_HISTORY,
        EventType.QUERY_UNDERSTAND,
        EventType.CHUNK_SEARCH_PARALLEL,
        EventType.CHUNK_RERANK,
        EventType.CHUNK_MERGE,
        EventType.FILTER_TOP_K,
        EventType.INTO_CHAT_MESSAGE,
        EventType.CHAT_COMPLETION_STREAM,
    ]
    assert last_consolidated_retrieval_stage(pipeline, pipeline_ctx) == EventType.FILTER_TOP_K


def test_should_close_retrieval_progress() -> None:
    last = EventType.FILTER_TOP_K
    assert should_close_retrieval_progress(EventType.FILTER_TOP_K, last, None)
    assert not should_close_retrieval_progress(EventType.CHUNK_SEARCH_PARALLEL, last, None)
    assert should_close_retrieval_progress(
        EventType.CHUNK_SEARCH_PARALLEL, last, ERR_SEARCH_NOTHING
    )
    assert should_close_retrieval_progress(
        EventType.CHUNK_RERANK, last, PluginError(description="hard failure")
    )


def test_should_emit_query_understand_progress() -> None:
    assert should_emit_query_understand_progress(PipelineContext(enable_rewrite=True))
    assert should_emit_query_understand_progress(
        PipelineContext(images=["data:image/png;base64,abc"])
    )
    assert not should_emit_query_understand_progress(PipelineContext())


# ── Progress: event emission ───────────────────────────────────────────


async def test_query_understand_progress_emits_tool_call_and_result() -> None:
    bus = _RecordingEventBus()
    pipeline_ctx = PipelineContext(session_id="sess-1", query="rewrite me", enable_rewrite=True)

    progress = await begin_query_understand_progress(pipeline_ctx, bus)
    assert progress is not None
    await end_query_understand_progress(pipeline_ctx, progress, 0.0, None, bus)

    assert len(bus.events) == 2
    assert bus.events[0].type == ChatEventType.AGENT_TOOL_CALL
    assert bus.events[0].session_id == "sess-1"
    call_data = _event_data(bus.events[0])
    assert call_data["tool_name"] == "query_understand"
    assert call_data["tool_call_id"] == progress.tool_call_id
    assert call_data["arguments"] == {"query": "rewrite me"}

    assert bus.events[1].type == ChatEventType.AGENT_TOOL_RESULT
    result_data = _event_data(bus.events[1])
    assert result_data["tool_name"] == "query_understand"
    assert result_data["success"] is True
    assert result_data["output"] == "已完成问题理解"
    assert result_data["error"] == ""


async def test_retrieval_progress_emits_single_tool_call_and_result() -> None:
    bus = _RecordingEventBus()
    pipeline_ctx = PipelineContext(
        session_id="sess-1",
        merge_result=[SearchResult(id="r1"), SearchResult(id="r2"), SearchResult(id="r3")],
    )

    progress = await begin_retrieval_progress(pipeline_ctx, bus)
    assert progress is not None
    await end_retrieval_progress(pipeline_ctx, progress, 0.0, None, bus)

    assert len(bus.events) == 2
    assert bus.events[0].type == ChatEventType.AGENT_TOOL_CALL
    assert bus.events[1].type == ChatEventType.AGENT_TOOL_RESULT

    call_data = _event_data(bus.events[0])
    assert call_data["tool_name"] == "knowledge_search"
    arguments = call_data["arguments"]
    assert isinstance(arguments, dict)
    assert arguments["search_source"] == "knowledge"

    result_data = _event_data(bus.events[1])
    assert result_data["success"] is True
    data_block = result_data["data"]
    assert isinstance(data_block, dict)
    assert data_block["count"] == 3
    assert data_block["doc_count"] == 3
    assert data_block["web_count"] == 0
    assert data_block["search_source"] == "knowledge"


async def test_retrieval_progress_web_only_search_source() -> None:
    bus = _RecordingEventBus()
    pipeline_ctx = PipelineContext(
        session_id="sess-web",
        web_search_enabled=True,
        merge_result=[
            SearchResult(id="w1", chunk_type="web_search"),
            SearchResult(id="w2", knowledge_source="web_search"),
        ],
    )

    progress = await begin_retrieval_progress(pipeline_ctx, bus)
    assert progress is not None
    await end_retrieval_progress(pipeline_ctx, progress, 0.0, None, bus)

    call_data = _event_data(bus.events[0])
    arguments = call_data["arguments"]
    assert isinstance(arguments, dict)
    assert arguments["search_source"] == "web"

    result_data = _event_data(bus.events[1])
    data_block = result_data["data"]
    assert isinstance(data_block, dict)
    assert data_block["count"] == 2
    assert data_block["doc_count"] == 0
    assert data_block["web_count"] == 2
    assert data_block["search_source"] == "web"


async def test_retrieval_progress_search_nothing_is_success() -> None:
    bus = _RecordingEventBus()
    pipeline_ctx = PipelineContext(session_id="sess-empty")

    progress = await begin_retrieval_progress(pipeline_ctx, bus)
    await end_retrieval_progress(pipeline_ctx, progress, 0.0, ERR_SEARCH_NOTHING, bus)

    result_data = _event_data(bus.events[1])
    assert result_data["success"] is True
    assert result_data["output"] == "未检索到相关内容"
    data_block = result_data["data"]
    assert isinstance(data_block, dict)
    assert data_block["count"] == 0


async def test_retrieval_progress_hard_error_reports_failure() -> None:
    bus = _RecordingEventBus()
    pipeline_ctx = PipelineContext(session_id="sess-err")
    cause = RuntimeError("rerank failed")
    stage_err = PluginError(description="Reranking failed", error_type="rerank_failed")

    progress = await begin_retrieval_progress(pipeline_ctx, bus)
    await end_retrieval_progress(pipeline_ctx, progress, 0.0, stage_err.with_error(cause), bus)

    result_data = _event_data(bus.events[1])
    assert result_data["success"] is False
    error_text = result_data["error"]
    assert isinstance(error_text, str)
    assert "rerank failed" in error_text


async def test_progress_without_event_bus_is_noop() -> None:
    pipeline_ctx = PipelineContext(session_id="sess-1")
    assert await begin_retrieval_progress(pipeline_ctx, None) is None
    assert await begin_query_understand_progress(pipeline_ctx, None) is None
    await end_retrieval_progress(pipeline_ctx, None, 0.0, None, None)


async def test_progress_survives_event_bus_failure() -> None:
    bus = _FailingEventBus()
    pipeline_ctx = PipelineContext(session_id="sess-1", merge_result=[SearchResult(id="r1")])

    progress = await begin_retrieval_progress(pipeline_ctx, bus)
    assert progress is not None
    # The failed emit must not raise into the pipeline.
    await end_retrieval_progress(pipeline_ctx, progress, 0.0, None, bus)


# ── LoadHistory plugin ─────────────────────────────────────────────────


async def test_load_history_skips_when_multi_turn_disabled() -> None:
    service = _FakeMessageService([])
    plugin = LoadHistoryPlugin(service)
    pipeline_ctx = PipelineContext(session_id="sess-1", max_rounds=0)

    result = await plugin.on_event(
        _FakeContext(),
        EventType.LOAD_HISTORY,
        pipeline_ctx,
        _RecordingNext().run,
    )

    assert result is None
    assert service.fetch_count is None
    assert pipeline_ctx.history == []


async def test_load_history_sets_grouped_history() -> None:
    later = _NOW
    earlier = _NOW - timedelta(days=2)
    stored = [
        _StoredMessage(request_id="r1", role="user", content="first q", created_at=earlier),
        _StoredMessage(request_id="r1", role="assistant", content="first a", created_at=earlier),
        _StoredMessage(request_id="r2", role="user", content="second q", created_at=later),
        _StoredMessage(request_id="r2", role="assistant", content="second a", created_at=later),
    ]
    service = _FakeMessageService(stored)
    plugin = LoadHistoryPlugin(service)
    pipeline_ctx = PipelineContext(session_id="sess-1", max_rounds=4)
    next_cb = _RecordingNext()

    result = await plugin.on_event(
        _FakeContext(), EventType.LOAD_HISTORY, pipeline_ctx, next_cb.run
    )

    assert result is None
    assert service.fetch_count == 4 * 2 + 10
    assert next_cb.invocations == 1
    assert [(entry.query, entry.answer) for entry in pipeline_ctx.history] == [
        ("first q", "first a"),
        ("second q", "second a"),
    ]


async def test_load_history_continues_on_fetch_error() -> None:
    service = _FakeMessageService([], error=RuntimeError("db down"))
    plugin = LoadHistoryPlugin(service)
    pipeline_ctx = PipelineContext(session_id="sess-1", max_rounds=2)
    next_cb = _RecordingNext()

    result = await plugin.on_event(
        _FakeContext(), EventType.LOAD_HISTORY, pipeline_ctx, next_cb.run
    )

    assert result is None
    assert pipeline_ctx.history == []
    assert next_cb.invocations == 1


async def test_load_history_truncates_to_max_rounds() -> None:
    stored: list[_StoredMessage] = []
    for index in range(6):
        stored.append(_StoredMessage(request_id=f"r{index}", role="user", content=f"q{index}"))
        stored.append(
            _StoredMessage(request_id=f"r{index}", role="assistant", content=f"a{index}")
        )
    service = _FakeMessageService(stored)
    plugin = LoadHistoryPlugin(service)
    pipeline_ctx = PipelineContext(session_id="sess-1", max_rounds=2)

    result = await plugin.on_event(
        _FakeContext(), EventType.LOAD_HISTORY, pipeline_ctx, _RecordingNext().run
    )

    assert result is None
    assert len(pipeline_ctx.history) == 2


# ── WikiBoost plugin ───────────────────────────────────────────────────


def _wiki_pipeline_ctx() -> PipelineContext:
    return PipelineContext(
        tenant_id=make_test_tenant_id(),
        search_targets=[SearchTarget(knowledge_base_id="kb-1")],
        rerank_result=[
            SearchResult(id="wiki-1", chunk_type=CHUNK_TYPE_WIKI_PAGE, score=0.4),
            SearchResult(id="text-1", chunk_type="text", score=0.5),
        ],
    )


async def test_wiki_boost_skips_when_no_wiki_chunks() -> None:
    service = _FakeKbService({"kb-1"})
    plugin = WikiBoostPlugin(service)
    pipeline_ctx = PipelineContext(
        search_targets=[SearchTarget(knowledge_base_id="kb-1")],
        rerank_result=[SearchResult(id="text-1", score=0.5)],
    )

    result = await plugin.on_event(
        _FakeContext(), EventType.CHUNK_RERANK, pipeline_ctx, _RecordingNext().run
    )

    assert result is None
    assert pipeline_ctx.rerank_result[0].score == 0.5


async def test_wiki_boost_skips_when_kb_not_wiki_enabled() -> None:
    service = _FakeKbService(set())  # kb-1 is not wiki-enabled
    plugin = WikiBoostPlugin(service)
    pipeline_ctx = _wiki_pipeline_ctx()

    result = await plugin.on_event(
        _FakeContext(), EventType.CHUNK_RERANK, pipeline_ctx, _RecordingNext().run
    )

    assert result is None
    assert [hit.score for hit in pipeline_ctx.rerank_result] == [0.4, 0.5]


async def test_wiki_boost_skips_when_kb_lookup_fails() -> None:
    service = _FakeKbService({"kb-1"}, missing=True)
    plugin = WikiBoostPlugin(service)
    pipeline_ctx = _wiki_pipeline_ctx()

    result = await plugin.on_event(
        _FakeContext(), EventType.CHUNK_RERANK, pipeline_ctx, _RecordingNext().run
    )

    assert result is None
    assert [hit.score for hit in pipeline_ctx.rerank_result] == [0.4, 0.5]


async def test_wiki_boost_boosts_and_resorts() -> None:
    service = _FakeKbService({"kb-1"})
    plugin = WikiBoostPlugin(service)
    pipeline_ctx = _wiki_pipeline_ctx()

    result = await plugin.on_event(
        _FakeContext(), EventType.CHUNK_RERANK, pipeline_ctx, _RecordingNext().run
    )

    assert result is None
    # Wiki chunk: 0.4 * 1.3 = 0.52, now ranking above the 0.5 text chunk.
    assert [hit.id for hit in pipeline_ctx.rerank_result] == ["wiki-1", "text-1"]
    assert pipeline_ctx.rerank_result[0].score == 0.4 * WIKI_BOOST_FACTOR
    assert pipeline_ctx.rerank_result[1].score == 0.5


async def test_wiki_boost_propagates_downstream_rerank_error() -> None:
    service = _FakeKbService({"kb-1"})
    plugin = WikiBoostPlugin(service)
    pipeline_ctx = _wiki_pipeline_ctx()
    next_cb = _RecordingNext()
    next_cb.error = PluginError(description="rerank failed")

    result = await plugin.on_event(
        _FakeContext(), EventType.CHUNK_RERANK, pipeline_ctx, next_cb.run
    )

    assert result is next_cb.error
    assert pipeline_ctx.rerank_result[0].score == 0.4


# ── Integration: wiki boost against the real schema ────────────────────


class _WikiRerankLoader:
    """Loads real wiki chunks into ``rerank_result`` when triggered."""

    def __init__(self, session: AsyncSession, tenant_id: int, chunk_ids: list[str]) -> None:
        self._session = session
        self._tenant_id = tenant_id
        self._chunk_ids = chunk_ids

    async def on_event(
        self,
        _ctx: Context,
        _event_type: EventType | str,
        pipeline_ctx: PipelineContext,
        next: Next,
    ) -> PluginError | None:
        repository = ChunkRepository(self._session)
        for chunk_id in self._chunk_ids:
            chunk = await repository.get_by_id_or_none(self._tenant_id, chunk_id)
            if chunk is None:
                return PluginError(description=f"chunk {chunk_id} not found")
            pipeline_ctx.rerank_result.append(
                SearchResult(
                    id=chunk.id,
                    content=chunk.content,
                    knowledge_id=chunk.knowledge_id,
                    knowledge_base_id=chunk.knowledge_base_id,
                    chunk_index=chunk.chunk_index,
                    chunk_type=chunk.chunk_type,
                    score=0.5,
                )
            )
        return await next()

    def activation_events(self) -> list[EventType]:
        return [EventType.CHUNK_RERANK]


async def test_integration_wiki_boost_chain_with_real_schema(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    kb_id = str(uuid.uuid4())
    await KnowledgeBaseRepository(session).create(
        KnowledgeBase(
            id=kb_id,
            name="Wiki KB",
            tenant_id=tenant_id,
            indexing_strategy={"vector_enabled": True, "wiki_enabled": True},
            created_at=_NOW,
            updated_at=_NOW,
        )
    )
    wiki_chunk_id = str(uuid.uuid4())
    text_chunk_id = str(uuid.uuid4())
    repository = ChunkRepository(session)
    await repository.create(
        Chunk(
            id=wiki_chunk_id,
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            knowledge_id="doc-integration",
            content="wiki payload",
            chunk_index=0,
            start_at=0,
            end_at=12,
            chunk_type=CHUNK_TYPE_WIKI_PAGE,
            created_at=_NOW,
            updated_at=_NOW,
        )
    )
    await repository.create(
        Chunk(
            id=text_chunk_id,
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            knowledge_id="doc-integration",
            content="text payload",
            chunk_index=1,
            start_at=0,
            end_at=11,
            chunk_type="text",
            created_at=_NOW,
            updated_at=_NOW,
        )
    )

    kb_service = KBService(kb_repo=KnowledgeBaseRepository(session))
    manager = EventManager()
    manager.register(_WikiRerankLoader(session, tenant_id, [wiki_chunk_id, text_chunk_id]))
    manager.register(WikiBoostPlugin(kb_service))

    pipeline_ctx = PipelineContext(
        tenant_id=tenant_id,
        search_targets=[SearchTarget(knowledge_base_id=kb_id)],
    )
    result = await manager.trigger(_FakeContext(), EventType.CHUNK_RERANK, pipeline_ctx)

    assert result is None
    assert pipeline_ctx.rerank_result[0].id == wiki_chunk_id
    assert pipeline_ctx.rerank_result[0].score == 0.5 * WIKI_BOOST_FACTOR
    assert pipeline_ctx.rerank_result[1].id == text_chunk_id
    assert pipeline_ctx.rerank_result[1].score == 0.5
