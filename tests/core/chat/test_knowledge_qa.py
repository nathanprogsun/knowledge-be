"""Tests for the knowledge-QA pipeline orchestration.

Exercises pipeline assembly, execution, reference emission, fallback
handling, and the high-level ``run_knowledge_qa`` entry point. All
heavy seams (event manager, event bus) are replaced with tiny stubs.
"""

from __future__ import annotations

import pytest

from src.core.chat.bus import Event
from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.engine import (
    ERR_SEARCH,
    ERR_SEARCH_NOTHING,
    PluginError,
)
from src.core.chat.pipeline.types import (
    Context,
    EventType,
    FallbackStrategy,
    MessageAttachment,
    SearchResult,
    SearchTarget,
)
from src.core.chat.sessions.knowledge_qa import (
    FallbackHandler,
    build_pipeline_stages,
    build_pure_chat_user_content,
    emit_references,
    execute_knowledge_qa,
    has_knowledge_retrieval_scope,
    run_knowledge_qa,
)
from src.core.chat.types import EventType as ChatEventType

# ── Stubs ─────────────────────────────────────────────────────────────


class _RecordingBus:
    """Minimal event bus that records every emitted event."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)


class _StubEventManager:
    """Event manager stub with controllable per-stage results."""

    def __init__(
        self,
        results: dict[EventType, PluginError | None] | None = None,
    ) -> None:
        self._results = results or {}
        self.triggered: list[EventType] = []

    async def trigger(
        self,
        ctx: Context,
        event_type: EventType | str,
        pipeline_ctx: PipelineContext,
    ) -> PluginError | None:
        et = event_type if isinstance(event_type, EventType) else EventType(str(event_type))
        self.triggered.append(et)
        return self._results.get(et)


class _Ctx:
    """Opaque execution context satisfying the ``Context`` protocol."""

    tenant_id = 1
    user_id = "test-user"
    request_id = "req-1"


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_ctx(**overrides: object) -> PipelineContext:
    """Build a ``PipelineContext`` with sensible defaults."""
    defaults: dict[str, object] = {
        "session_id": "sess-1",
        "user_id": "user-1",
        "query": "What is RAG?",
        "max_rounds": 4,
    }
    defaults.update(overrides)
    return PipelineContext(**defaults)  # type: ignore[arg-type]


# ── has_knowledge_retrieval_scope ─────────────────────────────────────


class TestHasKnowledgeRetrievalScope:
    """Tests for KB retrieval scope detection."""

    def test_returns_false_when_all_empty(self) -> None:
        # Arrange / Act / Assert
        assert not has_knowledge_retrieval_scope(
            search_targets=[],
            knowledge_base_ids=[],
            knowledge_ids=[],
        )

    def test_returns_true_when_kb_ids_present(self) -> None:
        assert has_knowledge_retrieval_scope(
            search_targets=[],
            knowledge_base_ids=["kb-1"],
            knowledge_ids=[],
        )

    def test_returns_true_when_knowledge_ids_present(self) -> None:
        assert has_knowledge_retrieval_scope(
            search_targets=[],
            knowledge_base_ids=[],
            knowledge_ids=["k-1"],
        )

    def test_returns_true_when_search_targets_present(self) -> None:
        target = SearchTarget(knowledge_base_id="kb-1")
        assert has_knowledge_retrieval_scope(
            search_targets=[target],
            knowledge_base_ids=[],
            knowledge_ids=[],
        )

    def test_returns_true_when_all_present(self) -> None:
        target = SearchTarget(knowledge_base_id="kb-1")
        assert has_knowledge_retrieval_scope(
            search_targets=[target],
            knowledge_base_ids=["kb-2"],
            knowledge_ids=["k-1"],
        )


# ── build_pipeline_stages ────────────────────────────────────────────


class TestBuildPipelineStages:
    """Tests for pipeline stage assembly."""

    def test_pure_chat_no_history(self) -> None:
        # Arrange
        ctx = _make_ctx(knowledge_base_ids=[], max_rounds=0)

        # Act
        stages = build_pipeline_stages(pipeline_ctx=ctx)

        # Assert
        assert stages == [EventType.CHAT_COMPLETION_STREAM]

    def test_pure_chat_with_history(self) -> None:
        ctx = _make_ctx(knowledge_base_ids=[], max_rounds=4)

        stages = build_pipeline_stages(pipeline_ctx=ctx)

        assert stages == [
            EventType.LOAD_HISTORY,
            EventType.CHAT_COMPLETION_STREAM,
        ]

    def test_rag_with_kb_no_history(self) -> None:
        ctx = _make_ctx(knowledge_base_ids=["kb-1"], max_rounds=0)

        stages = build_pipeline_stages(pipeline_ctx=ctx)

        assert stages == [
            EventType.QUERY_UNDERSTAND,
            EventType.CHUNK_SEARCH_PARALLEL,
            EventType.CHUNK_RERANK,
            EventType.CHUNK_MERGE,
            EventType.FILTER_TOP_K,
            EventType.INTO_CHAT_MESSAGE,
            EventType.CHAT_COMPLETION_STREAM,
        ]

    def test_rag_with_kb_and_history(self) -> None:
        ctx = _make_ctx(knowledge_base_ids=["kb-1"], max_rounds=4)

        stages = build_pipeline_stages(pipeline_ctx=ctx)

        assert stages == [
            EventType.LOAD_HISTORY,
            EventType.QUERY_UNDERSTAND,
            EventType.CHUNK_SEARCH_PARALLEL,
            EventType.CHUNK_RERANK,
            EventType.CHUNK_MERGE,
            EventType.FILTER_TOP_K,
            EventType.INTO_CHAT_MESSAGE,
            EventType.CHAT_COMPLETION_STREAM,
        ]

    def test_rag_with_web_search_includes_web_fetch(self) -> None:
        ctx = _make_ctx(
            knowledge_base_ids=["kb-1"],
            web_search_enabled=True,
        )

        stages = build_pipeline_stages(pipeline_ctx=ctx)

        assert EventType.WEB_FETCH in stages
        assert stages.index(EventType.WEB_FETCH) < stages.index(EventType.CHUNK_MERGE)

    def test_rag_without_web_search_excludes_web_fetch(self) -> None:
        ctx = _make_ctx(
            knowledge_base_ids=["kb-1"],
            web_search_enabled=False,
        )

        stages = build_pipeline_stages(pipeline_ctx=ctx)

        assert EventType.WEB_FETCH not in stages

    def test_rag_with_data_analysis_includes_stage(self) -> None:
        ctx = _make_ctx(
            knowledge_base_ids=["kb-1"],
            data_analysis_enabled=True,
        )

        stages = build_pipeline_stages(pipeline_ctx=ctx)

        assert EventType.DATA_ANALYSIS in stages
        assert stages.index(EventType.DATA_ANALYSIS) < stages.index(EventType.INTO_CHAT_MESSAGE)

    def test_rag_without_data_analysis_excludes_stage(self) -> None:
        ctx = _make_ctx(
            knowledge_base_ids=["kb-1"],
            data_analysis_enabled=False,
        )

        stages = build_pipeline_stages(pipeline_ctx=ctx)

        assert EventType.DATA_ANALYSIS not in stages

    def test_web_search_only_triggers_rag_path(self) -> None:
        """Web search without KB IDs still needs RAG."""
        ctx = _make_ctx(
            knowledge_base_ids=[],
            web_search_enabled=True,
        )

        stages = build_pipeline_stages(pipeline_ctx=ctx)

        assert EventType.QUERY_UNDERSTAND in stages
        assert EventType.WEB_FETCH in stages


# ── build_pure_chat_user_content ──────────────────────────────────────


class TestBuildPureChatUserContent:
    """Tests for pure-chat user content assembly."""

    def test_query_only(self) -> None:
        ctx = _make_ctx()

        content = build_pure_chat_user_content(ctx)

        assert content == "What is RAG?"

    def test_includes_image_description_when_no_vision(self) -> None:
        ctx = _make_ctx(
            image_description="A diagram of RAG architecture",
            chat_model_supports_vision=False,
        )

        content = build_pure_chat_user_content(ctx)

        assert "[用户上传图片内容]" in content
        assert "A diagram of RAG architecture" in content

    def test_excludes_image_description_when_vision_supported(self) -> None:
        ctx = _make_ctx(
            image_description="A diagram of RAG architecture",
            chat_model_supports_vision=True,
        )

        content = build_pure_chat_user_content(ctx)

        assert "[用户上传图片内容]" not in content
        assert "A diagram of RAG architecture" not in content

    def test_includes_quoted_context(self) -> None:
        ctx = _make_ctx(quoted_context="> Previous message")

        content = build_pure_chat_user_content(ctx)

        assert "> Previous message" in content

    def test_includes_attachments_prompt(self) -> None:
        attachment = MessageAttachment(
            id="att-1",
            file_name="report.pdf",
            file_type="pdf",
            file_size=1024,
            content="PDF content here",
        )
        ctx = _make_ctx(attachments=[attachment])

        content = build_pure_chat_user_content(ctx)

        assert "<attachments>" in content
        assert "report.pdf" in content

    def test_empty_query_returns_empty_string(self) -> None:
        ctx = _make_ctx(query="")

        content = build_pure_chat_user_content(ctx)

        assert content == ""


# ── emit_references ───────────────────────────────────────────────────


class TestEmitReferences:
    """Tests for reference emission before streaming."""

    @pytest.mark.asyncio
    async def test_no_merge_result_emits_nothing(self) -> None:
        ctx = _make_ctx()
        bus = _RecordingBus()

        await emit_references(ctx, bus)

        assert bus.events == []

    @pytest.mark.asyncio
    async def test_emits_references_event_with_results(self) -> None:
        result = SearchResult(
            id="chunk-1",
            content="RAG stands for Retrieval-Augmented Generation",
            knowledge_base_id="kb-1",
        )
        ctx = _make_ctx(merge_result=[result])
        bus = _RecordingBus()

        await emit_references(ctx, bus)

        assert len(bus.events) == 1
        event = bus.events[0]
        assert event.type == ChatEventType.AGENT_REFERENCES
        assert event.session_id == "sess-1"
        references = event.data["references"]
        assert isinstance(references, list)
        assert len(references) == 1
        assert references[0]["id"] == "chunk-1"


# ── FallbackHandler ───────────────────────────────────────────────────


class TestFallbackHandler:
    """Tests for fallback response handling."""

    @pytest.mark.asyncio
    async def test_fixed_strategy_emits_fallback_answer(self) -> None:
        ctx = _make_ctx(
            fallback_strategy=FallbackStrategy.FIXED,
            fallback_response="Sorry, I cannot answer this.",
        )
        bus = _RecordingBus()
        handler = FallbackHandler()

        await handler.handle(pipeline_ctx=ctx, event_bus=bus)

        assert len(bus.events) == 1
        event = bus.events[0]
        assert event.type == ChatEventType.AGENT_FINAL_ANSWER
        assert event.data["content"] == "Sorry, I cannot answer this."
        assert event.data["done"] is True
        assert event.data["is_fallback"] is True
        assert ctx.chat_response is not None
        assert ctx.chat_response.content == "Sorry, I cannot answer this."

    @pytest.mark.asyncio
    async def test_model_strategy_with_empty_prompt_falls_to_fixed(
        self,
    ) -> None:
        ctx = _make_ctx(
            fallback_strategy=FallbackStrategy.MODEL,
            fallback_prompt="",
            fallback_response="Default fallback",
        )
        bus = _RecordingBus()
        handler = FallbackHandler()

        await handler.handle(pipeline_ctx=ctx, event_bus=bus)

        assert len(bus.events) == 1
        assert bus.events[0].data["content"] == "Default fallback"
        assert bus.events[0].data["is_fallback"] is True

    @pytest.mark.asyncio
    async def test_model_strategy_with_prompt_falls_to_fixed(self) -> None:
        """Model fallback without a wired chat model delegates to fixed."""
        ctx = _make_ctx(
            fallback_strategy=FallbackStrategy.MODEL,
            fallback_prompt="You are a helpful assistant.",
            fallback_response="Fixed response",
        )
        bus = _RecordingBus()
        handler = FallbackHandler()

        await handler.handle(pipeline_ctx=ctx, event_bus=bus)

        assert len(bus.events) == 1
        assert bus.events[0].data["content"] == "Fixed response"


# ── execute_knowledge_qa ─────────────────────────────────────────────


class TestExecuteKnowledgeQa:
    """Tests for pipeline stage execution."""

    @pytest.mark.asyncio
    async def test_all_stages_succeed(self) -> None:
        ctx = _make_ctx(knowledge_base_ids=["kb-1"])
        stages = [
            EventType.QUERY_UNDERSTAND,
            EventType.CHAT_COMPLETION_STREAM,
        ]
        bus = _RecordingBus()
        manager = _StubEventManager()

        await execute_knowledge_qa(
            ctx=_Ctx(),
            event_manager=manager,
            pipeline_ctx=ctx,
            stages=stages,
            event_bus=bus,
        )

        assert manager.triggered == stages

    @pytest.mark.asyncio
    async def test_emits_references_before_stream(self) -> None:
        result = SearchResult(id="chunk-1", content="content")
        ctx = _make_ctx(merge_result=[result])
        stages = [EventType.CHAT_COMPLETION_STREAM]
        bus = _RecordingBus()
        manager = _StubEventManager()

        await execute_knowledge_qa(
            ctx=_Ctx(),
            event_manager=manager,
            pipeline_ctx=ctx,
            stages=stages,
            event_bus=bus,
        )

        ref_events = [e for e in bus.events if e.type == ChatEventType.AGENT_REFERENCES]
        assert len(ref_events) == 1

    @pytest.mark.asyncio
    async def test_no_references_when_merge_result_empty(self) -> None:
        ctx = _make_ctx(merge_result=[])
        stages = [EventType.CHAT_COMPLETION_STREAM]
        bus = _RecordingBus()
        manager = _StubEventManager()

        await execute_knowledge_qa(
            ctx=_Ctx(),
            event_manager=manager,
            pipeline_ctx=ctx,
            stages=stages,
            event_bus=bus,
        )

        ref_events = [e for e in bus.events if e.type == ChatEventType.AGENT_REFERENCES]
        assert len(ref_events) == 0

    @pytest.mark.asyncio
    async def test_search_nothing_triggers_fallback(self) -> None:
        ctx = _make_ctx(
            knowledge_base_ids=["kb-1"],
            fallback_strategy=FallbackStrategy.FIXED,
            fallback_response="No results found",
        )
        stages = [EventType.CHUNK_SEARCH_PARALLEL]
        bus = _RecordingBus()
        manager = _StubEventManager(results={EventType.CHUNK_SEARCH_PARALLEL: ERR_SEARCH_NOTHING})

        await execute_knowledge_qa(
            ctx=_Ctx(),
            event_manager=manager,
            pipeline_ctx=ctx,
            stages=stages,
            event_bus=bus,
        )

        # Only the search stage was triggered, then the pipeline stops.
        assert manager.triggered == [EventType.CHUNK_SEARCH_PARALLEL]
        fallback_events = [e for e in bus.events if e.type == ChatEventType.AGENT_FINAL_ANSWER]
        assert len(fallback_events) == 1
        assert fallback_events[0].data["is_fallback"] is True
        assert fallback_events[0].data["content"] == "No results found"

    @pytest.mark.asyncio
    async def test_search_nothing_stops_remaining_stages(self) -> None:
        ctx = _make_ctx(
            knowledge_base_ids=["kb-1"],
            fallback_strategy=FallbackStrategy.FIXED,
            fallback_response="Fallback",
        )
        stages = [
            EventType.QUERY_UNDERSTAND,
            EventType.CHUNK_SEARCH_PARALLEL,
            EventType.CHAT_COMPLETION_STREAM,
        ]
        bus = _RecordingBus()
        manager = _StubEventManager(results={EventType.CHUNK_SEARCH_PARALLEL: ERR_SEARCH_NOTHING})

        await execute_knowledge_qa(
            ctx=_Ctx(),
            event_manager=manager,
            pipeline_ctx=ctx,
            stages=stages,
            event_bus=bus,
        )

        # Pipeline stops at the search stage.
        assert manager.triggered == [
            EventType.QUERY_UNDERSTAND,
            EventType.CHUNK_SEARCH_PARALLEL,
        ]

    @pytest.mark.asyncio
    async def test_plugin_error_with_exception_is_raised(self) -> None:
        ctx = _make_ctx(knowledge_base_ids=["kb-1"])
        stages = [EventType.CHUNK_RERANK]
        bus = _RecordingBus()
        original_exc = RuntimeError("rerank service unavailable")
        manager = _StubEventManager(
            results={EventType.CHUNK_RERANK: ERR_SEARCH.with_error(original_exc)}
        )

        with pytest.raises(RuntimeError, match="rerank service unavailable"):
            await execute_knowledge_qa(
                ctx=_Ctx(),
                event_manager=manager,
                pipeline_ctx=ctx,
                stages=stages,
                event_bus=bus,
            )

    @pytest.mark.asyncio
    async def test_plugin_error_without_exception_stops_pipeline(
        self,
    ) -> None:
        ctx = _make_ctx(knowledge_base_ids=["kb-1"])
        stages = [
            EventType.QUERY_UNDERSTAND,
            EventType.CHAT_COMPLETION_STREAM,
        ]
        bus = _RecordingBus()
        manager = _StubEventManager(results={EventType.QUERY_UNDERSTAND: ERR_SEARCH})

        # Should not raise; pipeline stops silently.
        await execute_knowledge_qa(
            ctx=_Ctx(),
            event_manager=manager,
            pipeline_ctx=ctx,
            stages=stages,
            event_bus=bus,
        )

        # Only the first stage was triggered.
        assert manager.triggered == [EventType.QUERY_UNDERSTAND]

    @pytest.mark.asyncio
    async def test_empty_stages_completes_without_error(self) -> None:
        ctx = _make_ctx()
        bus = _RecordingBus()
        manager = _StubEventManager()

        await execute_knowledge_qa(
            ctx=_Ctx(),
            event_manager=manager,
            pipeline_ctx=ctx,
            stages=[],
            event_bus=bus,
        )

        assert manager.triggered == []


# ── run_knowledge_qa ─────────────────────────────────────────────────


class TestRunKnowledgeQa:
    """Tests for the high-level entry point."""

    @pytest.mark.asyncio
    async def test_pure_chat_sets_user_content(self) -> None:
        ctx = _make_ctx(
            knowledge_base_ids=[],
            max_rounds=0,
            query="Hello",
        )
        bus = _RecordingBus()
        manager = _StubEventManager()

        await run_knowledge_qa(
            ctx=_Ctx(),
            event_manager=manager,
            pipeline_ctx=ctx,
            event_bus=bus,
        )

        assert ctx.user_content == "Hello"
        assert manager.triggered == [EventType.CHAT_COMPLETION_STREAM]

    @pytest.mark.asyncio
    async def test_pure_chat_assembles_image_content(self) -> None:
        ctx = _make_ctx(
            knowledge_base_ids=[],
            max_rounds=0,
            image_description="A chart",
            chat_model_supports_vision=False,
        )
        bus = _RecordingBus()
        manager = _StubEventManager()

        await run_knowledge_qa(
            ctx=_Ctx(),
            event_manager=manager,
            pipeline_ctx=ctx,
            event_bus=bus,
        )

        assert "[用户上传图片内容]" in ctx.user_content
        assert "A chart" in ctx.user_content

    @pytest.mark.asyncio
    async def test_rag_path_does_not_override_user_content(self) -> None:
        ctx = _make_ctx(
            knowledge_base_ids=["kb-1"],
            user_content="pre-set content",
        )
        bus = _RecordingBus()
        manager = _StubEventManager()

        await run_knowledge_qa(
            ctx=_Ctx(),
            event_manager=manager,
            pipeline_ctx=ctx,
            event_bus=bus,
        )

        # RAG path leaves user_content alone; INTO_CHAT_MESSAGE sets it.
        assert ctx.user_content == "pre-set content"
        assert EventType.QUERY_UNDERSTAND in manager.triggered

    @pytest.mark.asyncio
    async def test_rag_path_executes_full_pipeline(self) -> None:
        ctx = _make_ctx(knowledge_base_ids=["kb-1"])
        bus = _RecordingBus()
        manager = _StubEventManager()

        await run_knowledge_qa(
            ctx=_Ctx(),
            event_manager=manager,
            pipeline_ctx=ctx,
            event_bus=bus,
        )

        expected = [
            EventType.LOAD_HISTORY,
            EventType.QUERY_UNDERSTAND,
            EventType.CHUNK_SEARCH_PARALLEL,
            EventType.CHUNK_RERANK,
            EventType.CHUNK_MERGE,
            EventType.FILTER_TOP_K,
            EventType.INTO_CHAT_MESSAGE,
            EventType.CHAT_COMPLETION_STREAM,
        ]
        assert manager.triggered == expected

    @pytest.mark.asyncio
    async def test_fallback_handler_injection(self) -> None:
        """A custom fallback handler is used instead of the default."""

        class _RecordingFallback:
            def __init__(self) -> None:
                self.called = False

            async def handle(
                self,
                *,
                pipeline_ctx: PipelineContext,
                event_bus: object,
            ) -> None:
                self.called = True

        ctx = _make_ctx(knowledge_base_ids=["kb-1"])
        bus = _RecordingBus()
        manager = _StubEventManager(results={EventType.CHUNK_SEARCH_PARALLEL: ERR_SEARCH_NOTHING})
        handler = _RecordingFallback()

        await run_knowledge_qa(
            ctx=_Ctx(),
            event_manager=manager,
            pipeline_ctx=ctx,
            event_bus=bus,
            fallback_handler=handler,  # type: ignore[arg-type]
        )

        assert handler.called is True
