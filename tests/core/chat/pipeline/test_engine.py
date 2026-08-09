"""Unit + integration tests for the chat pipeline engine.

Unit tests drive ``EventManager`` registration / dispatch / chain
semantics with recording plugins, plus the pipeline contract types
(``EventType``, ``QueryIntent``, ``PipelineBuilder``, ``PipelineContext``)
— no database, no async services.

Integration tests run one chain against the real applied schema: a plugin
loads a real ``chunks`` row (seeded with an int32-safe tenant id) into a
``PipelineContext`` and the engine drives the chain end to end.
"""

from __future__ import annotations

import itertools
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from random import randint

import pytest
import pytest_asyncio
from faker import Faker
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.common.exception import NotFoundError
from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.engine import (
    ERR_GET_CHAT_MODEL,
    ERR_SEARCH,
    ERR_SEARCH_NOTHING,
    EventManager,
    Next,
    PluginError,
)
from src.core.chat.pipeline.types import (
    PIPELINE_MODES,
    EventType,
    FallbackStrategy,
    PipelineBuilder,
    QueryIntent,
    SearchResult,
    SummaryConfig,
)
from src.db.dao.chunk_repository import ChunkRepository
from src.db.models.chunk import Chunk
from src.settings import get_settings, reset_settings_cache
from tests.integration.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_FAKER_SEED_MAX = 100_000_000

# ``chunks.tenant_id`` is an INTEGER (32-bit) column; integration ids are
# minted from this counter so seeded rows never overflow.
_INT32_TENANT_BASE = 4_000_000
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


@dataclass(frozen=True, slots=True)
class _Call:
    """One recorded plugin invocation."""

    event_type: EventType | str
    pipeline_ctx: PipelineContext


class _RecordingPlugin:
    """Async plugin that records its invocations."""

    def __init__(
        self,
        events: list[EventType],
        *,
        error: PluginError | None = None,
        call_next: bool = True,
    ) -> None:
        self._events = list(events)
        self._error = error
        self._call_next = call_next
        self.calls: list[_Call] = []
        self.next_invocations = 0

    def activation_events(self) -> list[EventType]:
        return list(self._events)

    async def on_event(
        self,
        _ctx: _FakeContext,
        event_type: EventType | str,
        pipeline_ctx: PipelineContext,
        next: Next,
    ) -> PluginError | None:
        self.calls.append(_Call(event_type=event_type, pipeline_ctx=pipeline_ctx))
        if not self._call_next:
            return self._error
        self.next_invocations += 1
        downstream = await next()
        return self._error if self._error is not None else downstream


# ── Engine: registration ───────────────────────────────────────────────


async def test_trigger_with_no_plugins_is_noop() -> None:
    manager = EventManager()
    pipeline_ctx = PipelineContext()
    result = await manager.trigger(_FakeContext(), "test_event", pipeline_ctx)
    assert result is None


async def test_trigger_for_unregistered_event_returns_none() -> None:
    manager = EventManager()
    manager.register(_RecordingPlugin([EventType.CHUNK_SEARCH]))
    result = await manager.trigger(_FakeContext(), EventType.CHAT_COMPLETION, PipelineContext())
    assert result is None


async def test_trigger_accepts_arbitrary_string_event_without_error() -> None:
    manager = EventManager()
    manager.register(_RecordingPlugin([EventType.CHUNK_SEARCH]))
    result = await manager.trigger(_FakeContext(), "custom_event", PipelineContext())
    assert result is None


# ── Engine: dispatch ───────────────────────────────────────────────────


async def test_single_plugin_success() -> None:
    manager = EventManager()
    plugin = _RecordingPlugin([EventType.CHUNK_SEARCH])
    manager.register(plugin)

    pipeline_ctx = PipelineContext()
    result = await manager.trigger(_FakeContext(), EventType.CHUNK_SEARCH, pipeline_ctx)

    assert result is None
    assert len(plugin.calls) == 1
    assert plugin.calls[0].event_type == EventType.CHUNK_SEARCH
    # The chain hands the very same carrier to the plugin.
    assert plugin.calls[0].pipeline_ctx is pipeline_ctx


async def test_plugin_chain_runs_in_registration_order() -> None:
    manager = EventManager()
    first = _RecordingPlugin([EventType.CHUNK_SEARCH])
    second = _RecordingPlugin([EventType.CHUNK_SEARCH])
    manager.register(first)
    manager.register(second)

    result = await manager.trigger(_FakeContext(), EventType.CHUNK_SEARCH, PipelineContext())

    assert result is None
    assert first.next_invocations == 1
    assert second.next_invocations == 1
    assert [call.event_type for call in first.calls] == [EventType.CHUNK_SEARCH]
    assert [call.event_type for call in second.calls] == [EventType.CHUNK_SEARCH]
    # Both plugins share the same carrier instance.
    assert first.calls[0].pipeline_ctx is second.calls[0].pipeline_ctx


async def test_plugin_chain_observes_state_written_by_previous_plugin() -> None:
    """Earlier plugins' writes are visible to later plugins in the chain."""

    class _Writer:
        async def on_event(
            self,
            _ctx: _FakeContext,
            _event_type: EventType | str,
            pipeline_ctx: PipelineContext,
            next: Next,
        ) -> PluginError | None:
            pipeline_ctx.user_content = "written upstream"
            return await next()

        def activation_events(self) -> list[EventType]:
            return [EventType.INTO_CHAT_MESSAGE]

    manager = EventManager()
    writer = _Writer()
    reader = _RecordingPlugin([EventType.INTO_CHAT_MESSAGE])
    manager.register(writer)
    manager.register(reader)

    pipeline_ctx = PipelineContext()
    result = await manager.trigger(_FakeContext(), EventType.INTO_CHAT_MESSAGE, pipeline_ctx)

    assert result is None
    assert pipeline_ctx.user_content == "written upstream"


# ── Engine: error propagation ──────────────────────────────────────────


async def test_plugin_returning_error_short_circuits_chain() -> None:
    manager = EventManager()
    expected = PluginError(description="test error", error_type="test_type")
    failing = _RecordingPlugin([EventType.CHUNK_SEARCH], error=expected, call_next=False)
    manager.register(failing)

    result = await manager.trigger(_FakeContext(), EventType.CHUNK_SEARCH, PipelineContext())

    assert result is expected
    assert failing.next_invocations == 0


async def test_error_from_mid_chain_propagates_to_trigger() -> None:
    manager = EventManager()
    expected = PluginError(description="deep failure", error_type="deep")
    first = _RecordingPlugin([EventType.CHUNK_SEARCH])
    second = _RecordingPlugin([EventType.CHUNK_SEARCH])
    third = _RecordingPlugin([EventType.CHUNK_SEARCH], error=expected, call_next=False)
    for plugin in (first, second, third):
        manager.register(plugin)

    result = await manager.trigger(_FakeContext(), EventType.CHUNK_SEARCH, PipelineContext())

    assert result is expected
    assert first.next_invocations == 1
    assert second.next_invocations == 1
    assert third.next_invocations == 0


async def test_plugin_that_skips_next_stops_the_chain_without_error() -> None:
    manager = EventManager()
    blocker = _RecordingPlugin([EventType.CHUNK_SEARCH], call_next=False)
    tail = _RecordingPlugin([EventType.CHUNK_SEARCH])
    manager.register(blocker)
    manager.register(tail)

    result = await manager.trigger(_FakeContext(), EventType.CHUNK_SEARCH, PipelineContext())

    assert result is None
    assert len(tail.calls) == 0  # the tail plugin never ran


async def test_plugin_registered_for_multiple_event_types() -> None:
    manager = EventManager()
    plugin = _RecordingPlugin([EventType.LOAD_HISTORY, EventType.CHAT_COMPLETION])
    manager.register(plugin)

    assert await manager.trigger(_FakeContext(), EventType.LOAD_HISTORY, PipelineContext()) is None
    assert (
        await manager.trigger(_FakeContext(), EventType.CHAT_COMPLETION, PipelineContext()) is None
    )
    assert len(plugin.calls) == 2


async def test_same_event_chain_rebuilt_after_further_registrations() -> None:
    manager = EventManager()
    early = _RecordingPlugin([EventType.CHUNK_SEARCH])
    late = _RecordingPlugin([EventType.CHUNK_SEARCH])
    manager.register(early)
    manager.register(late)

    pipeline_ctx = PipelineContext()
    await manager.trigger(_FakeContext(), EventType.CHUNK_SEARCH, pipeline_ctx)

    assert len(early.calls) == 1
    assert len(late.calls) == 1


# ── PluginError ────────────────────────────────────────────────────────


def test_plugin_error_with_error_returns_new_instance() -> None:
    error = PluginError(description="desc", error_type="type")
    cause = ValueError("boom")

    enriched = error.with_error(cause)

    assert enriched is not error
    assert enriched.err is cause
    assert enriched.description == "desc"
    assert error.err is None  # the original stays untouched


def test_predefined_plugin_errors_are_stable() -> None:
    assert ERR_SEARCH_NOTHING.error_type == "search_nothing"
    assert ERR_SEARCH.error_type == "search_failed"
    assert ERR_GET_CHAT_MODEL.error_type == "get_chat_model_failed"
    assert ERR_GET_CHAT_MODEL.with_error(RuntimeError("x")).err is not None
    assert ERR_SEARCH_NOTHING.err is None


# ── Event / intent types ───────────────────────────────────────────────


def test_event_type_values_match_contract() -> None:
    assert EventType.LOAD_HISTORY == "load_history"
    assert EventType.QUERY_UNDERSTAND == "query_understand"
    assert EventType.CHUNK_SEARCH == "chunk_search"
    assert EventType.CHUNK_SEARCH_PARALLEL == "chunk_search_parallel"
    assert EventType.ENTITY_SEARCH == "entity_search"
    assert EventType.CHUNK_RERANK == "chunk_rerank"
    assert EventType.WEB_FETCH == "web_fetch"
    assert EventType.CHUNK_MERGE == "chunk_merge"
    assert EventType.DATA_ANALYSIS == "data_analysis"
    assert EventType.INTO_CHAT_MESSAGE == "into_chat_message"
    assert EventType.CHAT_COMPLETION == "chat_completion"
    assert EventType.CHAT_COMPLETION_STREAM == "chat_completion_stream"
    assert EventType.FILTER_TOP_K == "filter_top_k"
    assert str(EventType.CHUNK_SEARCH) == "chunk_search"


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        (QueryIntent.KB_SEARCH, True),
        (QueryIntent.CLARIFICATION, True),
        (QueryIntent.SUMMARIZE, True),
        (QueryIntent.WEB_SEARCH, False),
        (QueryIntent.GREETING, False),
        (QueryIntent.CHITCHAT, False),
        (QueryIntent.FOLLOW_UP, False),
        (QueryIntent.IMAGE_ONLY, False),
        (QueryIntent.DOC_ONLY, False),
    ],
)
def test_query_intent_needs_kb_retrieval(intent: QueryIntent, expected: bool) -> None:
    assert intent.needs_kb_retrieval() is expected


def test_fallback_strategy_values() -> None:
    assert FallbackStrategy.FIXED == "fixed"
    assert FallbackStrategy.MODEL == "model"


# ── PipelineBuilder ────────────────────────────────────────────────────


def test_pipeline_builder_builds_ordered_stages() -> None:
    pipeline = (
        PipelineBuilder()
        .add(EventType.LOAD_HISTORY)
        .add(EventType.QUERY_UNDERSTAND)
        .add(EventType.CHAT_COMPLETION_STREAM)
        .build()
    )
    assert pipeline == [
        EventType.LOAD_HISTORY,
        EventType.QUERY_UNDERSTAND,
        EventType.CHAT_COMPLETION_STREAM,
    ]


def test_pipeline_builder_add_if_skips_false_conditions() -> None:
    pipeline = (
        PipelineBuilder()
        .add(EventType.LOAD_HISTORY)
        .add_if(False, EventType.QUERY_UNDERSTAND)
        .add_if(True, EventType.CHAT_COMPLETION_STREAM)
        .build()
    )
    assert pipeline == [EventType.LOAD_HISTORY, EventType.CHAT_COMPLETION_STREAM]


def test_pipeline_builder_empty() -> None:
    assert PipelineBuilder().build() == []


def test_pipeline_builder_build_returns_fresh_copy() -> None:
    builder = PipelineBuilder().add(EventType.LOAD_HISTORY)
    first = builder.build()
    first.append(EventType.CHAT_COMPLETION)
    assert builder.build() == [EventType.LOAD_HISTORY]


def test_pipeline_modes_match_contract() -> None:
    assert PIPELINE_MODES["rag"] == (
        EventType.CHUNK_SEARCH,
        EventType.CHUNK_RERANK,
        EventType.CHUNK_MERGE,
        EventType.INTO_CHAT_MESSAGE,
        EventType.CHAT_COMPLETION,
    )
    assert PIPELINE_MODES["rag_stream"][0] == EventType.LOAD_HISTORY
    assert PIPELINE_MODES["rag_stream"][-1] == EventType.CHAT_COMPLETION_STREAM
    assert PIPELINE_MODES["chat"] == (EventType.CHAT_COMPLETION,)


# ── PipelineContext behaviour ──────────────────────────────────────────


def test_context_default_needs_retrieval() -> None:
    # The unclassified intent defaults to retrieval for safety.
    assert PipelineContext().needs_retrieval() is True


def test_context_web_search_intent_requires_web_search_enabled() -> None:
    assert (
        PipelineContext(intent=QueryIntent.WEB_SEARCH, web_search_enabled=False).needs_retrieval()
        is False
    )
    assert (
        PipelineContext(intent=QueryIntent.WEB_SEARCH, web_search_enabled=True).needs_retrieval()
        is True
    )


def test_context_intent_drives_needs_retrieval() -> None:
    assert PipelineContext(intent=QueryIntent.CHITCHAT).needs_retrieval() is False
    assert PipelineContext(intent=QueryIntent.KB_SEARCH).needs_retrieval() is True


def test_context_citations_enabled_defaults_to_true() -> None:
    assert PipelineContext().citations_enabled() is True
    assert PipelineContext(citation_enabled=True).citations_enabled() is True
    assert PipelineContext(citation_enabled=False).citations_enabled() is False


def test_context_references_prefer_merge_result() -> None:
    search_hit = SearchResult(content="search")
    merge_hit = SearchResult(content="merge")
    ctx = PipelineContext(search_result=[search_hit], merge_result=[merge_hit])
    assert ctx.references() == [merge_hit]
    assert PipelineContext(search_result=[search_hit]).references() == [search_hit]
    assert PipelineContext().references() == []


def test_context_clone_is_a_deep_copy() -> None:
    tenant_id = make_test_tenant_id()
    ctx = PipelineContext(tenant_id=tenant_id, search_result=[SearchResult(content="a")])
    clone = ctx.clone()
    assert clone is not ctx
    assert clone.tenant_id == tenant_id
    clone.search_result = [SearchResult(content="b")]
    assert ctx.search_result[0].content == "a"


def test_summary_config_is_frozen() -> None:
    config = SummaryConfig(temperature=0.7)
    assert config.temperature == 0.7
    with pytest.raises(ValidationError):
        config.temperature = 0.9  # type: ignore[misc]


def test_search_result_is_frozen() -> None:
    hit = SearchResult(content="payload")
    with pytest.raises(ValidationError):
        hit.content = "mutated"  # type: ignore[misc]


# ── Integration: real DB chain ─────────────────────────────────────────


class _ChunkLoadPlugin:
    """Loads one real chunk row into ``search_result`` when triggered."""

    def __init__(self, session: AsyncSession, tenant_id: int, chunk_id: str) -> None:
        self._session = session
        self._tenant_id = tenant_id
        self._chunk_id = chunk_id

    async def on_event(
        self,
        _ctx: _FakeContext,
        _event_type: EventType | str,
        pipeline_ctx: PipelineContext,
        next: Next,
    ) -> PluginError | None:
        chunk = await ChunkRepository(self._session).get_by_id_or_none(
            self._tenant_id, self._chunk_id
        )
        if chunk is None:
            return ERR_SEARCH.with_error(NotFoundError(message=f"chunk {self._chunk_id} not found"))
        pipeline_ctx.search_result = [
            SearchResult(
                id=chunk.id,
                content=chunk.content,
                knowledge_id=chunk.knowledge_id,
                knowledge_base_id=chunk.knowledge_base_id,
                chunk_index=chunk.chunk_index,
                chunk_type=chunk.chunk_type,
            )
        ]
        return await next()

    def activation_events(self) -> list[EventType]:
        return [EventType.CHUNK_SEARCH]


async def test_integration_chain_loads_real_chunk_into_context(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    chunk_id = str(uuid.uuid4())
    await ChunkRepository(session).create(
        Chunk(
            id=chunk_id,
            tenant_id=tenant_id,
            knowledge_base_id="kb-integration",
            knowledge_id="doc-integration",
            content="pipeline integration payload",
            chunk_index=0,
            start_at=0,
            end_at=28,
            created_at=_NOW,
            updated_at=_NOW,
        )
    )

    manager = EventManager()
    loader = _ChunkLoadPlugin(session, tenant_id, chunk_id)
    manager.register(loader)

    pipeline_ctx = PipelineContext(tenant_id=tenant_id, knowledge_base_ids=["kb-integration"])
    result = await manager.trigger(_FakeContext(), EventType.CHUNK_SEARCH, pipeline_ctx)

    assert result is None
    assert len(pipeline_ctx.search_result) == 1
    hit = pipeline_ctx.search_result[0]
    assert hit.id == chunk_id
    assert hit.content == "pipeline integration payload"
    assert hit.knowledge_id == "doc-integration"
    assert hit.knowledge_base_id == "kb-integration"
