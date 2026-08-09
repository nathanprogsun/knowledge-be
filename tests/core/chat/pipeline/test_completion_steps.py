"""Unit tests for the pipeline completion steps.

Covers the four completion stages — data analysis, into-chat-message
context rendering, chat completion, and streaming completion — plus their
shared message-preparation and passage helpers. Service seams (model,
knowledge, message, event bus) are driven with in-memory fakes following
the AAA pattern; no database is required.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest

from src.ai.llm.types import (
    ChatOptions as LLMChatOptions,
)
from src.ai.llm.types import (
    ChatResponse as LLMChatResponse,
)
from src.ai.llm.types import (
    Message,
    ResponseType,
    StreamResponse,
)
from src.ai.llm.types import (
    TokenUsage as LLMTokenUsage,
)
from src.ai.retrieval.types import MatchType
from src.core.agents.tools.data_analysis import (
    ColumnInfo,
    TableSchema,
    data_analysis_input_schema,
    format_query_results,
    sql_single_quote_escape,
)
from src.core.chat.bus import Event
from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.engine import (
    ERR_GET_CHAT_MODEL,
    ERR_MODEL_CALL,
    ERR_TEMPLATE_EXECUTE,
)
from src.core.chat.pipeline.steps.chat_completion import (
    ChatCompletionStep,
    to_pipeline_chat_response,
)
from src.core.chat.pipeline.steps.data_analysis import (
    DataAnalysisStep,
    filter_out_table_chunks,
    is_data_file,
)
from src.core.chat.pipeline.steps.into_chat_message import IntoChatMessageStep
from src.core.chat.pipeline.steps.model_context import (
    chat_message_to_llm,
    ordered_pipeline_references,
)
from src.core.chat.pipeline.steps.passage import (
    CHUNK_TYPE_FAQ,
    build_document_header,
    enrich_content_with_image_info_for_chat,
    get_enriched_passage_for_chat,
)
from src.core.chat.pipeline.steps.stream import ChatCompletionStreamStep
from src.core.chat.pipeline.types import (
    ChatResponse,
    History,
    SearchResult,
    SummaryConfig,
    TokenUsage,
)
from src.core.chat.types import EventType as ChatEventType
from src.core.contracts.knowledge import Knowledge

#: Opaque task context (empty structural protocol) for seam fakes.
_CTX = object()

_NOW = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)


async def _noop_next() -> None:
    return None


def _is_error(result, expected) -> bool:
    """Return whether ``result`` is a plugin error of ``expected``'s type."""
    return result is not None and result.error_type == expected.error_type


def _knowledge(id: str = "k-1") -> Knowledge:
    return Knowledge(
        id=id,
        tenant_id=1,
        knowledge_base_id="kb-1",
        type="file",
        title="sales",
        description="monthly sales",
        file_name="sales.csv",
        file_type="csv",
        file_path="storage://local/sales.csv",
        parse_status="done",
        enable_status="enabled",
        created_at=_NOW,
        updated_at=_NOW,
    )


def _search_result(**overrides: object) -> SearchResult:
    base: dict[str, object] = {
        "id": "c-1",
        "content": "chunk content",
        "knowledge_id": "k-1",
        "knowledge_title": "title",
        "knowledge_filename": "doc.pdf",
    }
    base.update(overrides)
    return SearchResult(**base)


# ── Shared fakes ────────────────────────────────────────────────────────


class _FakeChatModel:
    """Minimal chat client recording calls and returning scripted output."""

    def __init__(
        self,
        response: LLMChatResponse | Exception | None = None,
        stream: AsyncIterator[StreamResponse] | None = None,
    ) -> None:
        self._response = response
        self._stream = stream
        self.chat_calls: list[tuple[list[Message], LLMChatOptions]] = []
        self.stream_calls: list[tuple[list[Message], LLMChatOptions]] = []

    async def chat(
        self, messages: list[Message], opts: LLMChatOptions | None = None
    ) -> LLMChatResponse:
        self.chat_calls.append((messages, opts or LLMChatOptions()))
        if isinstance(self._response, Exception):
            raise self._response
        if self._response is not None:
            return self._response
        return LLMChatResponse(content="default answer")

    def chat_stream(
        self, messages: list[Message], opts: LLMChatOptions | None = None
    ) -> AsyncIterator[StreamResponse] | None:
        self.stream_calls.append((messages, opts or LLMChatOptions()))
        return self._stream

    def get_model_name(self) -> str:
        return "fake-model"

    def get_model_id(self) -> str:
        return "model-1"


class _FakeModelService:
    def __init__(self, model: _FakeChatModel | None = None) -> None:
        self._model = model or _FakeChatModel()
        self.model_ids: list[str] = []

    async def get_chat_model(self, _ctx: object, model_id: str) -> _FakeChatModel:
        self.model_ids.append(model_id)
        return self._model


class _FakeKnowledgeService:
    def __init__(self, knowledge: Knowledge | None = None) -> None:
        self._knowledge = knowledge
        self.ids: list[str] = []

    async def get_knowledge_by_id(
        self, _ctx: object, knowledge_id: str
    ) -> Knowledge | None:
        self.ids.append(knowledge_id)
        return self._knowledge


class _FakeDataAnalysisTool:
    """Scripted tool recording every call for assertions."""

    def __init__(
        self,
        schema: TableSchema | Exception,
        result=None,
        session_id: str = "s-1",
    ) -> None:
        self._schema = schema
        self._result = result
        self.session_id = session_id
        self.loaded: list[str] = []
        self.executed: list[str] = []
        self.cleaned_up = 0

    async def load_from_knowledge(self, _ctx: object, knowledge: Knowledge) -> TableSchema:
        self.loaded.append(knowledge.id)
        if isinstance(self._schema, Exception):
            raise self._schema
        return self._schema

    async def execute(self, _ctx: object, args_json: str):
        self.executed.append(args_json)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result

    async def cleanup(self, _ctx: object) -> None:
        self.cleaned_up += 1


def _fake_tool_factory(tool: _FakeDataAnalysisTool):
    created: list[str] = []

    def factory(session_id: str) -> _FakeDataAnalysisTool:
        created.append(session_id)
        return tool

    return factory, created


class _FakeMessageService:
    def __init__(self) -> None:
        self.updates: list[tuple[str, str, str]] = []

    async def update_message_rendered_content(
        self, _ctx: object, session_id: str, user_message_id: str, content: str
    ) -> None:
        self.updates.append((session_id, user_message_id, content))


class _FakeEventBus:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)


def _stream(*events: StreamResponse) -> AsyncIterator[StreamResponse]:
    async def _gen() -> AsyncIterator[StreamResponse]:
        for event in events:
            yield event

    return _gen()


# ── Data-analysis helpers ───────────────────────────────────────────────


def test_is_data_file() -> None:
    assert is_data_file("report.csv")
    assert is_data_file("BOOK.XLSX")
    assert is_data_file("data.xls")
    assert not is_data_file("notes.txt")
    assert not is_data_file("")
    assert not is_data_file("report.csv.bak")


def test_filter_out_table_chunks_drops_table_probes() -> None:
    results = [
        _search_result(id="a", chunk_type="table_column"),
        _search_result(id="b", chunk_type="table_summary"),
        _search_result(id="c", chunk_type="faq"),
        _search_result(id="d", chunk_type=""),
    ]
    filtered = filter_out_table_chunks(results)
    assert [r.id for r in filtered] == ["c", "d"]


# ── Data-analysis step ──────────────────────────────────────────────────


async def test_data_analysis_skips_when_retrieval_not_needed() -> None:
    tool = _FakeDataAnalysisTool(TableSchema(table_name="t", columns=[]))
    factory, created = _fake_tool_factory(tool)
    step = DataAnalysisStep(_FakeModelService(), _FakeKnowledgeService(), factory)
    pipeline_ctx = PipelineContext(intent="greeting")

    result = await step.on_event(_CTX, "data_analysis", pipeline_ctx, _noop_next)

    assert result is None
    assert created == []
    assert tool.cleaned_up == 0


async def test_data_analysis_skips_without_data_files_and_filters_table_chunks() -> None:
    tool = _FakeDataAnalysisTool(TableSchema(table_name="t", columns=[]))
    factory, created = _fake_tool_factory(tool)
    step = DataAnalysisStep(_FakeModelService(), _FakeKnowledgeService(), factory)
    pipeline_ctx = PipelineContext(
        merge_result=[
            _search_result(id="a", knowledge_filename="doc.pdf"),
            _search_result(id="b", knowledge_filename="doc.pdf", chunk_type="table_summary"),
        ]
    )

    result = await step.on_event(_CTX, "data_analysis", pipeline_ctx, _noop_next)

    assert result is None
    assert created == []
    assert [r.id for r in pipeline_ctx.merge_result] == ["a"]


async def test_data_analysis_skips_when_knowledge_not_found() -> None:
    tool = _FakeDataAnalysisTool(TableSchema(table_name="t", columns=[]))
    factory, created = _fake_tool_factory(tool)
    knowledge_service = _FakeKnowledgeService(knowledge=None)
    step = DataAnalysisStep(_FakeModelService(), knowledge_service, factory)
    pipeline_ctx = PipelineContext(
        merge_result=[_search_result(knowledge_filename="sales.csv")]
    )

    result = await step.on_event(_CTX, "data_analysis", pipeline_ctx, _noop_next)

    assert result is None
    assert created == []
    assert knowledge_service.ids == ["k-1"]


async def test_data_analysis_skips_when_knowledge_lookup_fails() -> None:
    tool = _FakeDataAnalysisTool(TableSchema(table_name="t", columns=[]))
    factory, created = _fake_tool_factory(tool)

    class _RaisingKnowledgeService:
        async def get_knowledge_by_id(self, _ctx: object, knowledge_id: str):
            raise RuntimeError("db down")

    step = DataAnalysisStep(_FakeModelService(), _RaisingKnowledgeService(), factory)
    pipeline_ctx = PipelineContext(
        session_id="s-1",
        merge_result=[_search_result(knowledge_filename="sales.csv")],
    )

    result = await step.on_event(_CTX, "data_analysis", pipeline_ctx, _noop_next)

    assert result is None
    assert created == []


async def test_data_analysis_cleanup_failure_does_not_mask_outcome() -> None:
    from src.core.agents.tools.base import ToolResult

    schema = TableSchema(table_name="k_1", columns=[], row_count=0)
    tool = _FakeDataAnalysisTool(
        schema, result=ToolResult(success=True, output="rows")
    )
    factory, _ = _fake_tool_factory(tool)
    model = _FakeChatModel(
        response=LLMChatResponse(content='{"knowledge_id": "k-1", "sql": "SELECT 1"}')
    )
    step = DataAnalysisStep(
        _FakeModelService(model), _FakeKnowledgeService(_knowledge()), factory
    )
    pipeline_ctx = PipelineContext(
        session_id="s-1",
        chat_model_id="cm-1",
        query="total?",
        merge_result=[_search_result(knowledge_filename="sales.csv")],
    )

    async def _failing_cleanup(_ctx: object) -> None:
        raise RuntimeError("drop failed")

    tool.cleanup = _failing_cleanup

    result = await step.on_event(_CTX, "data_analysis", pipeline_ctx, _noop_next)

    assert result is None
    assert pipeline_ctx.merge_result[-1].match_type == MatchType.DATA_ANALYSIS


async def test_data_analysis_load_failure_continues_chain() -> None:
    tool = _FakeDataAnalysisTool(RuntimeError("cannot load"))
    factory, created = _fake_tool_factory(tool)
    step = DataAnalysisStep(
        _FakeModelService(), _FakeKnowledgeService(_knowledge()), factory
    )
    pipeline_ctx = PipelineContext(
        session_id="s-1",
        merge_result=[_search_result(knowledge_filename="sales.csv")],
    )

    result = await step.on_event(_CTX, "data_analysis", pipeline_ctx, _noop_next)

    assert result is None
    assert tool.loaded == ["k-1"]
    assert tool.cleaned_up == 1
    assert created == ["s-1"]


async def test_data_analysis_returns_get_chat_model_error() -> None:
    schema = TableSchema(
        table_name="k_1",
        columns=[ColumnInfo(name="revenue", type="DOUBLE")],
        row_count=10,
    )
    tool = _FakeDataAnalysisTool(schema)
    factory, _ = _fake_tool_factory(tool)

    class _RaisingModelService:
        async def get_chat_model(self, _ctx: object, model_id: str):
            raise RuntimeError("no model")

    step = DataAnalysisStep(
        _RaisingModelService(), _FakeKnowledgeService(_knowledge()), factory
    )
    pipeline_ctx = PipelineContext(
        chat_model_id="cm-1",
        merge_result=[_search_result(knowledge_filename="sales.csv")],
    )

    result = await step.on_event(_CTX, "data_analysis", pipeline_ctx, _noop_next)

    assert _is_error(result, ERR_GET_CHAT_MODEL)
    assert result.err is not None
    assert tool.cleaned_up == 1


async def test_data_analysis_appends_analysis_result() -> None:
    schema = TableSchema(
        table_name="k_1",
        columns=[ColumnInfo(name="revenue", type="DOUBLE")],
        row_count=10,
    )
    from src.core.agents.tools.base import ToolResult

    tool = _FakeDataAnalysisTool(
        schema,
        result=ToolResult(success=True, output="record 1: {\"revenue\": \"100\"}"),
    )
    factory, _ = _fake_tool_factory(tool)
    model = _FakeChatModel(
        response=LLMChatResponse(content='{"knowledge_id": "k-1", "sql": "SELECT 1"}')
    )
    step = DataAnalysisStep(
        _FakeModelService(model), _FakeKnowledgeService(_knowledge()), factory
    )
    pipeline_ctx = PipelineContext(
        session_id="s-1",
        chat_model_id="cm-1",
        query="total revenue?",
        merge_result=[_search_result(knowledge_filename="sales.csv")],
    )

    result = await step.on_event(_CTX, "data_analysis", pipeline_ctx, _noop_next)

    assert result is None
    assert tool.executed == ['{"knowledge_id": "k-1", "sql": "SELECT 1"}']
    assert tool.cleaned_up == 1
    assert len(pipeline_ctx.merge_result) == 2
    added = pipeline_ctx.merge_result[-1]
    assert added.id == "analysis_k-1"
    assert added.match_type == MatchType.DATA_ANALYSIS
    assert added.score == 1.0
    assert added.content == 'record 1: {"revenue": "100"}'
    assert added.knowledge_title == "sales"
    assert added.knowledge_filename == "sales.csv"


async def test_data_analysis_plans_with_analysis_schema_format() -> None:
    from src.core.agents.tools.base import ToolResult

    schema = TableSchema(
        table_name="k_1",
        columns=[ColumnInfo(name="revenue", type="DOUBLE")],
        row_count=10,
    )
    tool = _FakeDataAnalysisTool(
        schema, result=ToolResult(success=True, output="rows")
    )
    factory, _ = _fake_tool_factory(tool)
    model = _FakeChatModel(
        response=LLMChatResponse(content='{"knowledge_id": "k-1", "sql": ""}')
    )
    step = DataAnalysisStep(
        _FakeModelService(model), _FakeKnowledgeService(_knowledge()), factory
    )
    pipeline_ctx = PipelineContext(
        session_id="s-1",
        chat_model_id="cm-1",
        query="sum by month",
        merge_result=[_search_result(knowledge_filename="sales.csv")],
    )

    await step.on_event(_CTX, "data_analysis", pipeline_ctx, _noop_next)

    messages, opts = model.chat_calls[0]
    assert messages[0].role == "user"
    assert "User Question: sum by month" in messages[0].content
    assert "Table name: k_1" in messages[0].content
    assert opts.temperature == 0.1
    assert opts.format == data_analysis_input_schema()


async def test_data_analysis_skips_when_plan_model_fails() -> None:
    from src.core.agents.tools.base import ToolResult

    schema = TableSchema(table_name="k_1", columns=[], row_count=0)
    tool = _FakeDataAnalysisTool(
        schema, result=ToolResult(success=True, output="")
    )
    factory, _ = _fake_tool_factory(tool)
    model = _FakeChatModel(response=RuntimeError("provider down"))
    step = DataAnalysisStep(
        _FakeModelService(model), _FakeKnowledgeService(_knowledge()), factory
    )
    pipeline_ctx = PipelineContext(
        session_id="s-1",
        chat_model_id="cm-1",
        query="total?",
        merge_result=[_search_result(knowledge_filename="sales.csv")],
    )

    result = await step.on_event(_CTX, "data_analysis", pipeline_ctx, _noop_next)

    assert result is None
    assert tool.executed == []
    assert tool.cleaned_up == 1
    assert len(pipeline_ctx.merge_result) == 1


async def test_data_analysis_skips_when_execute_fails_or_rejected() -> None:
    from src.core.agents.tools.base import ToolResult

    schema = TableSchema(table_name="k_1", columns=[], row_count=0)
    tool = _FakeDataAnalysisTool(
        schema, result=RuntimeError("modification query rejected")
    )
    factory, _ = _fake_tool_factory(tool)
    step = DataAnalysisStep(
        _FakeModelService(),
        _FakeKnowledgeService(_knowledge()),
        factory,
    )
    pipeline_ctx = PipelineContext(
        session_id="s-1",
        chat_model_id="cm-1",
        query="total?",
        merge_result=[_search_result(knowledge_filename="sales.csv")],
    )

    result = await step.on_event(_CTX, "data_analysis", pipeline_ctx, _noop_next)

    assert result is None
    assert tool.cleaned_up == 1
    assert len(pipeline_ctx.merge_result) == 1

    failed_tool = _FakeDataAnalysisTool(
        schema, result=ToolResult(success=False, error="rejected")
    )
    failed_factory, _ = _fake_tool_factory(failed_tool)
    step = DataAnalysisStep(
        _FakeModelService(), _FakeKnowledgeService(_knowledge()), failed_factory
    )
    result = await step.on_event(_CTX, "data_analysis", pipeline_ctx, _noop_next)
    assert result is None
    assert failed_tool.cleaned_up == 1
    assert len(pipeline_ctx.merge_result) == 1


# ── Data-analysis tool contract shapes ──────────────────────────────────


def test_table_schema_describe_renders_columns() -> None:
    schema = TableSchema(
        table_name="k_1",
        columns=[ColumnInfo(name="revenue", type="DOUBLE"), ColumnInfo(name="date", type="VARCHAR")],
        row_count=42,
    )
    assert schema.describe() == (
        "Table name: k_1\n"
        "Columns: 2\n"
        "Rows: 42\n\n"
        "Column info:\n"
        "- revenue (DOUBLE)\n"
        "- date (VARCHAR)\n"
    )


def test_format_query_results_renders_jsonl_records() -> None:
    rendered = format_query_results(
        [{"revenue": "100", "month": "Jan"}], "SELECT * FROM k_1"
    )
    assert rendered.startswith("=== DuckDB Query Results ===\n\nExecuted SQL: SELECT * FROM k_1")
    assert "Returned 1 rows" in rendered
    assert 'record 1: {"month": "Jan", "revenue": "100"}' in rendered


def test_format_query_results_empty() -> None:
    rendered = format_query_results([], "SELECT * FROM k_1")
    assert "No matching records found." in rendered


def test_sql_single_quote_escape() -> None:
    assert sql_single_quote_escape("it's") == "it''s"
    assert sql_single_quote_escape("plain") == "plain"


def test_data_analysis_input_schema_shape() -> None:
    schema = data_analysis_input_schema()
    assert schema["type"] == "object"
    assert schema["required"] == ["knowledge_id", "sql"]
    assert "knowledge_id" in schema["properties"]
    assert "sql" in schema["properties"]


# ── Into-chat-message helpers ───────────────────────────────────────────


def test_build_document_header_escapes_and_dedupes() -> None:
    results = [
        _search_result(id="a", knowledge_id="k-1", knowledge_title="<Sales & Ops>", knowledge_description="d"),
        _search_result(id="b", knowledge_id="k-1", knowledge_title="dup"),
        _search_result(id="c", knowledge_id="k-2", knowledge_title="", knowledge_filename="alt.pdf"),
        _search_result(id="d", knowledge_id=""),
    ]
    header = build_document_header(results)
    assert "<documents>" in header
    assert "<title>&lt;Sales &amp; Ops&gt;</title>" in header
    assert "<description>d</description>" in header
    assert "<title>alt.pdf</title>" in header
    # duplicate knowledge id rendered once
    assert header.count("<document>") == 2


def test_build_document_header_empty_when_no_title() -> None:
    assert build_document_header([_search_result(id="a", knowledge_id="k-1", knowledge_title="", knowledge_filename="")]) == ""
    assert build_document_header([]) == ""


def test_get_enriched_passage_for_chat_with_image_info() -> None:
    result = _search_result(
        content="See ![chart](http://img/a.png) here",
        image_info='[{"url": "http://img/a.png", "caption": "Monthly", "ocr_text": "Jan 100"}]',
    )
    passage = get_enriched_passage_for_chat(result)
    assert "![chart](http://img/a.png)" in passage
    assert "**Image caption:** Monthly" in passage
    assert "**Image text (OCR):** Jan 100" in passage


def test_get_enriched_passage_for_chat_no_image_info() -> None:
    result = _search_result(content="plain")
    assert get_enriched_passage_for_chat(result) == "plain"
    empty = _search_result(content="", image_info="")
    assert get_enriched_passage_for_chat(empty) == ""


def test_enrich_content_with_image_info_ignores_invalid_json() -> None:
    assert enrich_content_with_image_info_for_chat("![a](u)", "{bad") == "![a](u)"
    assert enrich_content_with_image_info_for_chat("![a](u)", "") == "![a](u)"
    assert enrich_content_with_image_info_for_chat("![a](u)", "[]") == "![a](u)"


def test_enrich_content_with_image_info_html_img_src() -> None:
    content = '<img src="http://img/a.png"> trailing'
    enriched = enrich_content_with_image_info_for_chat(
        content, '[{"url": "http://img/a.png", "ocr_text": "t"}]'
    )
    assert "**Image text (OCR):** t" in enriched


# ── Into-chat-message step ──────────────────────────────────────────────


async def test_into_chat_message_rejects_invalid_query() -> None:
    step = IntoChatMessageStep(_FakeMessageService())
    pipeline_ctx = PipelineContext(query="bad\x01query")

    result = await step.on_event(_CTX, "into_chat_message", pipeline_ctx, _noop_next)

    assert _is_error(result, ERR_TEMPLATE_EXECUTE)
    assert result.err is not None


async def test_into_chat_message_no_search_renders_template() -> None:
    step = IntoChatMessageStep(_FakeMessageService())
    pipeline_ctx = PipelineContext(
        session_id="s-1",
        intent="greeting",
        query="hi",
        language="English",
        summary_config=SummaryConfig(context_template="Q: {{query}} C: {{contexts}}"),
    )

    result = await step.on_event(_CTX, "into_chat_message", pipeline_ctx, _noop_next)

    assert result is None
    assert pipeline_ctx.user_content == "Q: hi C: "


async def test_into_chat_message_no_search_uses_rewrite_query() -> None:
    step = IntoChatMessageStep(_FakeMessageService())
    pipeline_ctx = PipelineContext(
        intent="chitchat",
        query="hi",
        rewrite_query="  hello there  ",
        summary_config=SummaryConfig(context_template="{{query}}"),
    )

    result = await step.on_event(_CTX, "into_chat_message", pipeline_ctx, _noop_next)

    assert result is None
    assert pipeline_ctx.user_content == "hello there"


async def test_into_chat_message_appends_image_description_for_non_vision() -> None:
    step = IntoChatMessageStep(_FakeMessageService())
    pipeline_ctx = PipelineContext(
        intent="image_only",
        query="what is this",
        image_description="a cat",
        chat_model_supports_vision=False,
        summary_config=SummaryConfig(context_template=""),
    )

    result = await step.on_event(_CTX, "into_chat_message", pipeline_ctx, _noop_next)

    assert result is None
    assert "[用户上传图片内容]" in pipeline_ctx.user_content
    assert "a cat" in pipeline_ctx.user_content


async def test_into_chat_message_renders_contexts_default_order() -> None:
    step = IntoChatMessageStep(_FakeMessageService())
    pipeline_ctx = PipelineContext(
        session_id="s-1",
        query="question",
        merge_result=[
            _search_result(id="a", content="first", knowledge_id=""),
            _search_result(id="b", content="second", knowledge_id=""),
        ],
        summary_config=SummaryConfig(context_template="{{query}}|{{contexts}}"),
    )

    result = await step.on_event(_CTX, "into_chat_message", pipeline_ctx, _noop_next)

    assert result is None
    rendered = pipeline_ctx.rendered_contexts
    assert rendered == (
        '<context id="1">first</context>\n<context id="2">second</context>'
    )
    assert pipeline_ctx.user_content == "question|" + rendered


async def test_into_chat_message_faq_priority_high_confidence_first() -> None:
    step = IntoChatMessageStep(_FakeMessageService())
    faq_high = _search_result(id="f1", content="faq answer", chunk_type=CHUNK_TYPE_FAQ, score=0.9)
    doc = _search_result(id="d1", content="doc answer", chunk_type="text", score=0.5)
    pipeline_ctx = PipelineContext(
        session_id="s-1",
        query="question",
        faq_priority_enabled=True,
        faq_direct_answer_threshold=0.8,
        merge_result=[doc, faq_high],
        summary_config=SummaryConfig(context_template=""),
    )

    result = await step.on_event(_CTX, "into_chat_message", pipeline_ctx, _noop_next)

    assert result is None
    rendered = pipeline_ctx.rendered_contexts
    assert '<source type="faq" priority="high">' in rendered
    assert '<context id="FAQ-1" match="exact">faq answer</context>' in rendered
    assert '<source type="document" priority="supplementary">' in rendered
    assert '<context id="DOC-1">doc answer</context>' in rendered


async def test_into_chat_message_faq_priority_without_high_confidence() -> None:
    step = IntoChatMessageStep(_FakeMessageService())
    faq = _search_result(id="f1", content="faq", chunk_type=CHUNK_TYPE_FAQ, score=0.3)
    pipeline_ctx = PipelineContext(
        session_id="s-1",
        query="question",
        faq_priority_enabled=True,
        faq_direct_answer_threshold=0.8,
        merge_result=[faq],
        summary_config=SummaryConfig(context_template=""),
    )

    result = await step.on_event(_CTX, "into_chat_message", pipeline_ctx, _noop_next)

    assert result is None
    assert '<context id="FAQ-1">faq</context>' in pipeline_ctx.rendered_contexts
    assert "match=\"exact\"" not in pipeline_ctx.rendered_contexts


async def test_into_chat_message_persists_rendered_content() -> None:
    message_service = _FakeMessageService()
    step = IntoChatMessageStep(message_service)
    pipeline_ctx = PipelineContext(
        session_id="s-1",
        user_message_id="m-1",
        query="question",
        user_content="rendered answer",
    )

    step.persist_rendered_content(_CTX, pipeline_ctx)
    await asyncio.gather(*(list(step._background_tasks)))

    assert message_service.updates == [("s-1", "m-1", "rendered answer")]


async def test_into_chat_message_persist_skips_when_content_matches_query() -> None:
    message_service = _FakeMessageService()
    step = IntoChatMessageStep(message_service)
    pipeline_ctx = PipelineContext(
        session_id="s-1",
        user_message_id="m-1",
        query="same",
        user_content="same",
    )

    result = await step.on_event(_CTX, "into_chat_message", pipeline_ctx, _noop_next)

    assert result is None
    assert message_service.updates == []
    assert step._background_tasks == set()


# ── Shared message-preparation helpers ──────────────────────────────────


def test_ordered_pipeline_references_faq_first() -> None:
    results = [
        _search_result(id="doc", chunk_type="text"),
        _search_result(id="faq", chunk_type=CHUNK_TYPE_FAQ),
    ]
    pipeline_ctx = PipelineContext(faq_priority_enabled=True, merge_result=results)
    assert [r.id for r in ordered_pipeline_references(pipeline_ctx)] == ["faq", "doc"]
    plain = PipelineContext(faq_priority_enabled=False, merge_result=results)
    assert ordered_pipeline_references(plain) == results


def test_chat_message_to_llm_preserves_fields() -> None:
    from src.core.chat.pipeline.common import ChatMessage

    message = chat_message_to_llm(ChatMessage(role="user", content="hi", images=("a.png",)))
    assert message.role == "user"
    assert message.content == "hi"
    assert message.images == ["a.png"]


def test_prepare_messages_with_model_context_appends_protocol_prompt() -> None:
    from src.core.chat.pipeline.steps.model_context import prepare_messages_with_model_context

    pipeline_ctx = PipelineContext(
        query="q",
        user_content="q",
        summary_config=SummaryConfig(prompt="sys"),
    )
    messages, _registry = prepare_messages_with_model_context(pipeline_ctx)
    assert messages[0].role == "system"
    assert messages[0].content.startswith("sys")
    assert "citation" in messages[0].content.lower() or "resource" in messages[0].content.lower()
    assert messages[-1].content == "q"


def test_prepare_messages_with_model_context_replaces_rendered_contexts() -> None:
    from src.core.chat.pipeline.steps.model_context import prepare_messages_with_model_context

    pipeline_ctx = PipelineContext(
        query="q",
        user_content="q",
        rendered_contexts="OLD_CONTEXT",
        summary_config=SummaryConfig(prompt="sys"),
        merge_result=[_search_result(id="c-1", knowledge_id="k-1", content="chunk")],
    )
    messages, _registry = prepare_messages_with_model_context(pipeline_ctx)
    user_content = messages[-1].content
    assert "OLD_CONTEXT" not in user_content
    assert "chunk" in user_content


# ── Chat-completion step ────────────────────────────────────────────────


async def test_chat_completion_get_model_error() -> None:
    class _RaisingModelService:
        async def get_chat_model(self, _ctx: object, model_id: str):
            raise RuntimeError("no model")

    step = ChatCompletionStep(_RaisingModelService())
    pipeline_ctx = PipelineContext(chat_model_id="cm-1")

    result = await step.on_event(_CTX, "chat_completion", pipeline_ctx, _noop_next)

    assert _is_error(result, ERR_GET_CHAT_MODEL)
    assert result.err is not None
    assert pipeline_ctx.chat_response is None


async def test_chat_completion_model_call_error() -> None:
    model = _FakeChatModel(response=RuntimeError("provider down"))
    step = ChatCompletionStep(_FakeModelService(model))
    pipeline_ctx = PipelineContext(chat_model_id="cm-1", query="q", user_content="q")

    result = await step.on_event(_CTX, "chat_completion", pipeline_ctx, _noop_next)

    assert _is_error(result, ERR_MODEL_CALL)
    assert result.err is not None
    assert pipeline_ctx.chat_response is None


async def test_chat_completion_success_sets_response() -> None:
    model = _FakeChatModel(
        response=LLMChatResponse(
            content="answer",
            finish_reason="stop",
            usage=LLMTokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
    )
    step = ChatCompletionStep(_FakeModelService(model))
    pipeline_ctx = PipelineContext(
        session_id="s-1",
        chat_model_id="cm-1",
        query="q",
        user_content="q",
        history=[History(query="hq", answer="ha", created_at=_NOW)],
        summary_config=SummaryConfig(prompt="sys {{query}}"),
    )

    result = await step.on_event(_CTX, "chat_completion", pipeline_ctx, _noop_next)

    assert result is None
    assert pipeline_ctx.chat_response is not None
    assert pipeline_ctx.chat_response.content == "answer"
    assert pipeline_ctx.chat_response.usage.completion_tokens == 5
    # history replays as user/assistant pairs before the current question
    messages, _opts = model.chat_calls[0]
    roles = [m.role for m in messages]
    assert roles == ["system", "user", "assistant", "user"]


def test_to_pipeline_chat_response_projects_usage() -> None:
    response = LLMChatResponse(
        content="a",
        reasoning_content="r",
        finish_reason="length",
        usage=LLMTokenUsage(
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
            cached_tokens=4,
            cache_read_tokens=4,
        ),
    )
    projected = to_pipeline_chat_response(response)
    assert projected == ChatResponse(
        content="a",
        reasoning_content="r",
        finish_reason="length",
        usage=TokenUsage(
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
            cached_tokens=4,
            cache_read_tokens=4,
        ),
    )


# ── Streaming step ──────────────────────────────────────────────────────


async def test_stream_get_model_error() -> None:
    class _RaisingModelService:
        async def get_chat_model(self, _ctx: object, model_id: str):
            raise RuntimeError("no model")

    step = ChatCompletionStreamStep(_RaisingModelService(), _FakeEventBus())
    pipeline_ctx = PipelineContext(chat_model_id="cm-1")

    result = await step.on_event(_CTX, "chat_completion_stream", pipeline_ctx, _noop_next)

    assert _is_error(result, ERR_GET_CHAT_MODEL)
    assert result.err is not None


async def test_stream_nil_stream_returns_model_error() -> None:
    model = _FakeChatModel(stream=None)
    step = ChatCompletionStreamStep(_FakeModelService(model), _FakeEventBus())
    pipeline_ctx = PipelineContext(chat_model_id="cm-1", query="q", user_content="q")

    result = await step.on_event(_CTX, "chat_completion_stream", pipeline_ctx, _noop_next)

    assert _is_error(result, ERR_MODEL_CALL)
    assert result.err is not None


async def test_stream_missing_event_bus_returns_model_error() -> None:
    step = ChatCompletionStreamStep(_FakeModelService(), None)
    pipeline_ctx = PipelineContext(chat_model_id="cm-1", query="q", user_content="q")

    result = await step.on_event(_CTX, "chat_completion_stream", pipeline_ctx, _noop_next)

    assert _is_error(result, ERR_MODEL_CALL)
    assert "EventBus" in str(result.err)


async def test_stream_forwards_thinking_then_answer() -> None:
    stream = _stream(
        StreamResponse(response_type=ResponseType.THINKING, content="reasoning", done=False),
        StreamResponse(response_type=ResponseType.ANSWER, content="Hello", done=True),
    )
    model = _FakeChatModel(stream=stream)
    event_bus = _FakeEventBus()
    step = ChatCompletionStreamStep(_FakeModelService(model), event_bus)
    pipeline_ctx = PipelineContext(
        session_id="s-1", chat_model_id="cm-1", query="q", user_content="q"
    )

    result = await step.on_event(_CTX, "chat_completion_stream", pipeline_ctx, _noop_next)

    assert result is None
    thought_events = [e for e in event_bus.events if e.type == ChatEventType.AGENT_THOUGHT]
    answer_events = [e for e in event_bus.events if e.type == ChatEventType.AGENT_FINAL_ANSWER]
    assert thought_events[0].data == {"content": "reasoning", "done": False}
    # thinking closed before the answer starts
    assert thought_events[-1].data == {"content": "", "done": True}
    assert answer_events[0].data == {"content": "Hello", "done": True}


async def test_stream_suppresses_duplicate_answer_after_done() -> None:
    stream = _stream(
        StreamResponse(response_type=ResponseType.ANSWER, content="done ", done=True),
        StreamResponse(response_type=ResponseType.ANSWER, content="again", done=True),
        StreamResponse(response_type=ResponseType.ANSWER, content="more", done=False),
    )
    model = _FakeChatModel(stream=stream)
    event_bus = _FakeEventBus()
    step = ChatCompletionStreamStep(_FakeModelService(model), event_bus)
    pipeline_ctx = PipelineContext(
        session_id="s-1", chat_model_id="cm-1", query="q", user_content="q"
    )

    await step.on_event(_CTX, "chat_completion_stream", pipeline_ctx, _noop_next)

    answer_events = [e for e in event_bus.events if e.type == ChatEventType.AGENT_FINAL_ANSWER]
    assert len(answer_events) == 1
    assert answer_events[0].data == {"content": "done ", "done": True}


async def test_stream_emits_error_event_and_continues() -> None:
    stream = _stream(
        StreamResponse(response_type=ResponseType.ERROR, content="boom"),
        StreamResponse(response_type=ResponseType.ANSWER, content="ok", done=True),
    )
    model = _FakeChatModel(stream=stream)
    event_bus = _FakeEventBus()
    step = ChatCompletionStreamStep(_FakeModelService(model), event_bus)
    pipeline_ctx = PipelineContext(
        session_id="s-1", chat_model_id="cm-1", query="q", user_content="q"
    )

    result = await step.on_event(_CTX, "chat_completion_stream", pipeline_ctx, _noop_next)

    assert result is None
    error_events = [e for e in event_bus.events if e.type == ChatEventType.ERROR]
    assert error_events[0].data == {
        "error": "boom",
        "stage": "chat_completion_stream",
        "session_id": "s-1",
    }
    answer_events = [e for e in event_bus.events if e.type == ChatEventType.AGENT_FINAL_ANSWER]
    assert len(answer_events) == 1


async def test_stream_decodes_handle_split_across_chunks() -> None:
    from src.core.chat.pipeline.steps.model_context import prepare_messages_with_model_context

    pipeline_ctx = PipelineContext(
        session_id="s-1",
        chat_model_id="cm-1",
        query="q",
        user_content="see local://data/report.csv",
        summary_config=SummaryConfig(prompt="sys"),
    )
    messages, model_context = prepare_messages_with_model_context(pipeline_ctx)
    # Encoding registers the resource handle so the stream decoder can restore it.
    model_context.encode_messages(messages)

    stream = _stream(
        StreamResponse(response_type=ResponseType.ANSWER, content="Report: res://", done=False),
        StreamResponse(response_type=ResponseType.ANSWER, content="0001", done=True),
    )
    model = _FakeChatModel(stream=stream)
    event_bus = _FakeEventBus()
    step = ChatCompletionStreamStep(_FakeModelService(model), event_bus)

    await step._emit_stream_events(
        _CTX, pipeline_ctx, model_context, stream, event_bus
    )

    answer_events = [e for e in event_bus.events if e.type == ChatEventType.AGENT_FINAL_ANSWER]
    assert len(answer_events) == 2
    # first chunk emitted a partial (handle withheld), the final chunk decoded it
    assert answer_events[-1].data == {"content": "local://data/report.csv", "done": True}
    assert "res://" not in answer_events[-1].data["content"]


async def test_stream_flushes_on_cancellation() -> None:
    never = asyncio.Future()

    async def _blocking_stream() -> AsyncIterator[StreamResponse]:
        yield StreamResponse(response_type=ResponseType.THINKING, content="think", done=False)
        await never

    model = _FakeChatModel(stream=_blocking_stream())
    event_bus = _FakeEventBus()
    step = ChatCompletionStreamStep(_FakeModelService(model), event_bus)
    pipeline_ctx = PipelineContext(
        session_id="s-1", chat_model_id="cm-1", query="q", user_content="q"
    )

    task = asyncio.create_task(
        step.on_event(_CTX, "chat_completion_stream", pipeline_ctx, _noop_next)
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    if not never.done():
        never.cancel()

    thought_events = [e for e in event_bus.events if e.type == ChatEventType.AGENT_THOUGHT]
    assert thought_events[-1].data == {"content": "", "done": True}
