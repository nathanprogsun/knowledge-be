"""Unit tests for the agent engine core loop.

Drives ``AgentEngine`` with a scripted fake chat model, a stub tool
registered in a real ``ToolRegistry``, a real request-local model-context
``Registry``, and a recording event bus — no database, no network. Covers
the think / act / observe / finalize cycle: natural stop, tool-call rounds,
empty-content retry, stuck-loop detection, content filter, parallel and
timeout tool execution, unresolved model handles, transient-error retry,
graceful degradation, cancellation salvage, and history redaction.

The loop is deliberately storage-agnostic, so no DB integration test is
required for this deliverable; the tool registries it executes against are
covered by their own integration suites.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from random import randint

import pytest
from faker import Faker

from src.ai.embedding.base import Context
from src.ai.llm.types import (
    ChatOptions,
    ChatResponse,
    FunctionCall,
    LLMToolCall,
    Message,
    ResponseType,
    StreamResponse,
)
from src.common.json import JsonObject
from src.core.agents.engine.loop import AgentEngine
from src.core.agents.engine.modelcontext import Registry
from src.core.agents.engine.types import (
    CONTENT_FILTER_FALLBACK,
    EMPTY_RESPONSE_FALLBACK,
    AgentCancelledError,
    AgentConfig,
    AgentExecutionError,
    AgentState,
    IterOutcome,
    ToolResult,
)
from src.core.agents.tools.base import ToolResult as ToolsToolResult
from src.core.agents.tools.registry import ToolRegistry
from src.core.chat.bus import Event, EventBus
from src.core.chat.types import EventType

_FAKER_SEED_MAX = 100_000_000


@pytest.fixture(autouse=True)
def faker_seed() -> None:
    """Re-seed Faker per test for varied-but-reproducible generation."""
    Faker.seed(randint(1, _FAKER_SEED_MAX))


# ── Test doubles ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _FakeContext:
    """Opaque execution context satisfying the structural protocol."""

    is_background_task: bool = False


async def _iter_stream(chunks: Sequence[StreamResponse]) -> AsyncIterator[StreamResponse]:
    """Yield a scripted stream of response chunks."""
    for chunk in chunks:
        yield chunk


class _FakeChat:
    """Scripted chat client recording every streaming call."""

    def __init__(self) -> None:
        self._scripted: list[AsyncIterator[StreamResponse]] = []
        self.stream_calls: list[tuple[list[Message], ChatOptions]] = []
        self.chat_calls: list[tuple[list[Message], ChatOptions]] = []

    def queue(self, *chunks: StreamResponse) -> None:
        """Queue one scripted stream for the next call."""
        self._scripted.append(_iter_stream(chunks))

    async def chat(self, messages: list[Message], opts: ChatOptions | None = None) -> ChatResponse:
        self.chat_calls.append((list(messages), opts or ChatOptions()))
        return ChatResponse(content="default answer")

    def chat_stream(
        self, messages: list[Message], opts: ChatOptions | None = None
    ) -> AsyncIterator[StreamResponse]:
        self.stream_calls.append((list(messages), opts or ChatOptions()))
        if self._scripted:
            return self._scripted.pop(0)
        return _iter_stream([StreamResponse(content="done", finish_reason="stop")])

    def get_model_name(self) -> str:
        return "fake-model"

    def get_model_id(self) -> str:
        return "model-1"


class _StubTool:
    """Minimal tool that records its executions and returns scripted output."""

    def __init__(
        self,
        name: str,
        *,
        output: str = "tool output",
        parameters: str | None = None,
    ) -> None:
        self._name = name
        self._output = output
        self._parameters = parameters or '{"type": "object", "properties": {}}'
        self.executed = 0
        self.last_args: JsonObject = {}

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return f"stub {self._name}"

    def parameters(self) -> str:
        return self._parameters

    async def execute(self, ctx: Context, args: str) -> ToolsToolResult:
        self.executed += 1
        self.last_args = json.loads(args) if args else {}
        return ToolsToolResult(success=True, output=self._output)


_AGENT_EVENT_TYPES: tuple[EventType, ...] = (
    EventType.AGENT_THOUGHT,
    EventType.AGENT_TOOL_CALL,
    EventType.AGENT_TOOL_RESULT,
    EventType.AGENT_TOOL,
    EventType.AGENT_FINAL_ANSWER,
    EventType.AGENT_COMPLETE,
    EventType.ERROR,
)


def _make_engine(
    fake_chat: _FakeChat,
    registry: ToolRegistry,
    events: list[Event],
    config: AgentConfig,
    *,
    tool_exec_timeout: float = 0.5,
) -> AgentEngine:
    bus = EventBus()

    async def record(event: Event) -> None:
        events.append(event)

    for event_type in _AGENT_EVENT_TYPES:
        bus.on(event_type, record)
    return AgentEngine(
        config=config,
        tool_registry=registry,
        chat_model=fake_chat,
        event_bus=bus,
        model_context=Registry(citations_enabled=True),
        tool_exec_timeout=tool_exec_timeout,
    )


def _tool_call_chunk(
    *,
    content: str = "searching",
    call_id: str = "call-1",
    tool_name: str = "web_search",
    args: str = '{"query": "test"}',
    finish_reason: str = "tool_calls",
) -> StreamResponse:
    return StreamResponse(
        content=content,
        finish_reason=finish_reason,
        tool_calls=[LLMToolCall(id=call_id, function=FunctionCall(name=tool_name, arguments=args))],
    )


# ── Contract types ───────────────────────────────────────────────────


def test_agent_state_serializes_to_contract_shape() -> None:
    state = AgentState(
        current_round=2,
        is_complete=True,
        final_answer="answer",
        round_steps=[],
    )
    payload = state.model_dump()
    assert payload["current_round"] == 2
    assert payload["is_complete"] is True
    assert payload["final_answer"] == "answer"
    assert payload["round_steps"] == []
    assert state.total_tool_calls() == 0


def test_agent_state_is_frozen() -> None:
    state = AgentState()
    with pytest.raises((AttributeError, ValueError)):
        state.final_answer = "mutated"  # type: ignore[misc]


def test_tool_result_from_tools_result() -> None:
    result = ToolResult.from_tools_result(
        ToolsToolResult(success=True, output="out", data={"k": "v"}, error="", images=["i"])
    )
    assert result.success is True
    assert result.output == "out"
    assert result.data == {"k": "v"}
    assert result.images == ["i"]

    failed = ToolResult.from_tools_result(None)
    assert failed.success is False
    assert failed.error == "tool returned no result"


def test_iter_outcome_labels() -> None:
    assert IterOutcome.NEXT.value == "next"
    assert IterOutcome.CONTINUE.value == "continue"
    assert IterOutcome.BREAK.value == "break"


# ── Natural stop ─────────────────────────────────────────────────────


async def test_natural_stop_sets_final_answer() -> None:
    fake = _FakeChat()
    fake.queue(StreamResponse(content="Hello world", finish_reason="stop"))
    registry = ToolRegistry()
    registry.register_tool(_StubTool("web_search"))
    events: list[Event] = []
    engine = _make_engine(fake, registry, events, AgentConfig(max_iterations=5, temperature=0.7))

    state = await engine.execute(_FakeContext(), query="hi", session_id="s1", message_id="m1")

    assert state.is_complete is True
    assert state.final_answer == "Hello world"
    assert len(state.round_steps) == 1
    assert state.round_steps[0].thought == "Hello world"
    assert state.round_steps[0].tool_calls == []
    assert state.current_round == 0

    answer_events = [e for e in events if e.type is EventType.AGENT_FINAL_ANSWER]
    assert len(answer_events) == 2
    assert answer_events[0].data == {"content": "Hello world", "done": False}
    assert answer_events[1].data == {"content": "", "done": True}


async def test_natural_stop_strips_think_blocks() -> None:
    fake = _FakeChat()
    fake.queue(StreamResponse(content="<think>hidden</think> Final answer", finish_reason="stop"))
    engine = _make_engine(fake, ToolRegistry(), [], AgentConfig(max_iterations=5))

    state = await engine.execute(_FakeContext(), query="q", session_id="s", message_id="m")

    assert state.final_answer == "Final answer"


# ── Tool-call round (act + observe) ──────────────────────────────────


async def test_tool_call_round_executes_and_observes() -> None:
    fake = _FakeChat()
    fake.queue(_tool_call_chunk(content="let me search"))
    fake.queue(StreamResponse(content="Here is the answer", finish_reason="stop"))
    stub = _StubTool("web_search", output="search result payload")
    registry = ToolRegistry()
    registry.register_tool(stub)
    events: list[Event] = []
    engine = _make_engine(fake, registry, events, AgentConfig(max_iterations=5))

    state = await engine.execute(_FakeContext(), query="hi", session_id="s1", message_id="m1")

    assert state.is_complete is True
    assert state.final_answer == "Here is the answer"
    assert state.current_round == 1
    assert len(state.round_steps) == 2

    step = state.round_steps[0]
    assert step.thought == "let me search"
    assert len(step.tool_calls) == 1
    tool_call = step.tool_calls[0]
    assert tool_call.name == "web_search"
    assert tool_call.args == {"query": "test"}
    assert tool_call.result is not None
    assert tool_call.result.success is True
    assert tool_call.result.output == "search result payload"
    assert tool_call.duration >= 0
    assert stub.executed == 1
    assert stub.last_args == {"query": "test"}

    # Observe phase: the second LLM call carries the assistant tool-call
    # message plus the rendered tool result.
    second_messages, _ = fake.stream_calls[1]
    roles = [message.role for message in second_messages]
    assert "assistant" in roles
    assert "tool" in roles
    tool_messages = [m for m in second_messages if m.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == tool_call.id
    assert "search result payload" in tool_messages[0].content

    # Completion event carries the steps for downstream persistence.
    complete_events = [e for e in events if e.type is EventType.AGENT_COMPLETE]
    assert len(complete_events) == 1
    complete_data = complete_events[0].data
    assert isinstance(complete_data, dict)
    assert complete_data.get("total_steps") == 2
    assert isinstance(complete_data.get("agent_steps"), list)


async def test_parallel_tool_calls() -> None:
    fake = _FakeChat()
    fake.queue(
        StreamResponse(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                LLMToolCall(
                    id="call-a",
                    function=FunctionCall(name="web_search", arguments='{"query": "a"}'),
                ),
                LLMToolCall(
                    id="call-b",
                    function=FunctionCall(name="web_search", arguments='{"query": "b"}'),
                ),
            ],
        )
    )
    fake.queue(StreamResponse(content="done", finish_reason="stop"))
    stub = _StubTool("web_search")
    registry = ToolRegistry()
    registry.register_tool(stub)
    engine = _make_engine(
        fake, registry, [], AgentConfig(max_iterations=5, parallel_tool_calls=True)
    )

    state = await engine.execute(_FakeContext(), query="q", session_id="s", message_id="m")

    tool_calls = state.round_steps[0].tool_calls
    assert len(tool_calls) == 2
    assert {tc.id for tc in tool_calls} == {"call-a", "call-b"}
    assert all(tc.result is not None and tc.result.success for tc in tool_calls)
    assert stub.executed == 2


async def test_unresolved_model_handles_fail_before_execution() -> None:
    fake = _FakeChat()
    fake.queue(_tool_call_chunk(args='{"query": "res://0001"}'))
    fake.queue(StreamResponse(content="done", finish_reason="stop"))
    stub = _StubTool("web_search")
    registry = ToolRegistry()
    registry.register_tool(stub)
    engine = _make_engine(fake, registry, [], AgentConfig(max_iterations=5))

    state = await engine.execute(_FakeContext(), query="q", session_id="s", message_id="m")

    tool_call = state.round_steps[0].tool_calls[0]
    assert tool_call.result is not None
    assert tool_call.result.success is False
    assert "unresolved model handles" in tool_call.result.error
    assert stub.executed == 0


async def test_malformed_arguments_are_repaired() -> None:
    fake = _FakeChat()
    fake.queue(_tool_call_chunk(args='{"query": "test"'))
    fake.queue(StreamResponse(content="done", finish_reason="stop"))
    stub = _StubTool("web_search")
    registry = ToolRegistry()
    registry.register_tool(stub)
    engine = _make_engine(fake, registry, [], AgentConfig(max_iterations=5))

    state = await engine.execute(_FakeContext(), query="q", session_id="s", message_id="m")

    tool_call = state.round_steps[0].tool_calls[0]
    assert tool_call.result is not None
    assert tool_call.result.success is True
    assert stub.last_args == {"query": "test"}


async def test_unknown_tool_becomes_failed_result() -> None:
    fake = _FakeChat()
    fake.queue(_tool_call_chunk(tool_name="does_not_exist"))
    fake.queue(StreamResponse(content="done", finish_reason="stop"))
    engine = _make_engine(fake, ToolRegistry(), [], AgentConfig(max_iterations=5))

    state = await engine.execute(_FakeContext(), query="q", session_id="s", message_id="m")

    tool_call = state.round_steps[0].tool_calls[0]
    assert tool_call.result is not None
    assert tool_call.result.success is False
    assert "does_not_exist" in tool_call.result.error


async def test_tool_execution_timeout() -> None:
    class _SlowTool(_StubTool):
        async def execute(self, ctx: Context, args: str) -> ToolsToolResult:
            self.executed += 1
            await asyncio.sleep(5)
            return ToolsToolResult(success=True, output="late")

    fake = _FakeChat()
    fake.queue(_tool_call_chunk())
    fake.queue(StreamResponse(content="done", finish_reason="stop"))
    registry = ToolRegistry()
    registry.register_tool(_SlowTool("web_search"))
    engine = _make_engine(fake, registry, [], AgentConfig(max_iterations=5), tool_exec_timeout=0.05)

    state = await engine.execute(_FakeContext(), query="q", session_id="s", message_id="m")

    tool_call = state.round_steps[0].tool_calls[0]
    assert tool_call.result is not None
    assert tool_call.result.success is False
    assert "timed out" in tool_call.result.error


# ── Stop conditions ──────────────────────────────────────────────────


async def test_empty_content_retries_then_falls_back() -> None:
    fake = _FakeChat()
    for _ in range(3):
        fake.queue(StreamResponse(content="", finish_reason="stop"))
    engine = _make_engine(fake, ToolRegistry(), [], AgentConfig(max_iterations=5))

    state = await engine.execute(_FakeContext(), query="q", session_id="s", message_id="m")

    assert state.is_complete is True
    assert state.final_answer == EMPTY_RESPONSE_FALLBACK
    # The two retry calls carry the nudge in their user-turn content.
    assert len(fake.stream_calls) == 3
    assert any("complete answer" in m.content for m in fake.stream_calls[1][0])
    assert any("complete answer" in m.content for m in fake.stream_calls[2][0])


async def test_repeated_content_breaks_stuck_loop() -> None:
    fake = _FakeChat()
    for _ in range(3):
        fake.queue(StreamResponse(content="same text", finish_reason="length"))
    engine = _make_engine(fake, ToolRegistry(), [], AgentConfig(max_iterations=10))

    state = await engine.execute(_FakeContext(), query="q", session_id="s", message_id="m")

    assert state.is_complete is True
    assert state.final_answer == "same text"
    assert state.current_round == 2


async def test_content_filter_stops_with_fallback() -> None:
    fake = _FakeChat()
    fake.queue(StreamResponse(content="", finish_reason="content_filter"))
    events: list[Event] = []
    engine = _make_engine(fake, ToolRegistry(), events, AgentConfig(max_iterations=5))

    state = await engine.execute(_FakeContext(), query="q", session_id="s", message_id="m")

    assert state.is_complete is True
    assert state.final_answer == CONTENT_FILTER_FALLBACK
    final_answers = [e for e in events if e.type is EventType.AGENT_FINAL_ANSWER]
    assert final_answers[0].data == {"content": CONTENT_FILTER_FALLBACK, "done": False}
    assert final_answers[1].data == {"content": "", "done": True}


# ── Retry / degradation ──────────────────────────────────────────────


async def test_transient_error_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop_sleep(_seconds: float) -> None:
        return

    monkeypatch.setattr("asyncio.sleep", _noop_sleep)
    fake = _FakeChat()
    fake.queue(StreamResponse(response_type=ResponseType.ERROR, content="rate limit exceeded"))
    fake.queue(StreamResponse(content="ok", finish_reason="stop"))
    engine = _make_engine(fake, ToolRegistry(), [], AgentConfig(max_iterations=5))

    state = await engine.execute(_FakeContext(), query="q", session_id="s", message_id="m")

    assert state.final_answer == "ok"
    assert len(fake.stream_calls) == 2


async def test_non_transient_error_fails_without_tool_results() -> None:
    fake = _FakeChat()
    fake.queue(StreamResponse(response_type=ResponseType.ERROR, content="provider exploded"))
    events: list[Event] = []
    engine = _make_engine(fake, ToolRegistry(), events, AgentConfig(max_iterations=5))

    with pytest.raises(AgentExecutionError) as exc_info:
        await engine.execute(_FakeContext(), query="q", session_id="s", message_id="m")
    assert "LLM call failed" in str(exc_info.value)
    error_events = [e for e in events if e.type is EventType.ERROR]
    assert len(error_events) == 1


async def test_llm_failure_degrades_to_synthesis_from_prior_tools() -> None:
    fake = _FakeChat()
    fake.queue(_tool_call_chunk())
    fake.queue(StreamResponse(response_type=ResponseType.ERROR, content="provider exploded"))
    fake.queue(StreamResponse(content="degraded answer", finish_reason="stop"))
    stub = _StubTool("web_search")
    registry = ToolRegistry()
    registry.register_tool(stub)
    engine = _make_engine(fake, registry, [], AgentConfig(max_iterations=5))

    state = await engine.execute(_FakeContext(), query="q", session_id="s", message_id="m")

    assert state.is_complete is True
    assert state.final_answer == "degraded answer"
    assert state.total_tool_calls() == 1


# ── Max iterations ──────────────────────────────────────────────────


async def test_max_iterations_synthesizes_final_answer() -> None:
    fake = _FakeChat()
    for _ in range(3):
        fake.queue(_tool_call_chunk())
    fake.queue(StreamResponse(content="Synthesized final answer", finish_reason="stop"))
    stub = _StubTool("web_search")
    registry = ToolRegistry()
    registry.register_tool(stub)
    events: list[Event] = []
    engine = _make_engine(fake, registry, events, AgentConfig(max_iterations=3))

    state = await engine.execute(_FakeContext(), query="q", session_id="s", message_id="m")

    assert state.is_complete is True
    assert state.final_answer == "Synthesized final answer"
    assert state.current_round == 3
    assert len(state.round_steps) == 3
    assert stub.executed == 3


# ── Cancellation ────────────────────────────────────────────────────


async def test_cancellation_before_any_round_salvages_nothing() -> None:
    fake = _FakeChat()
    engine = _make_engine(fake, ToolRegistry(), [], AgentConfig(max_iterations=5))

    with pytest.raises(AgentCancelledError) as exc_info:
        await engine.execute(
            _FakeContext(),
            query="q",
            session_id="s",
            message_id="m",
            is_cancelled=lambda: True,
        )

    salvaged = exc_info.value.state
    assert salvaged.is_complete is False
    assert salvaged.round_steps == []
    assert salvaged.total_tool_calls() == 0


async def test_cancellation_after_tool_round_salvages_synthesis() -> None:
    fake = _FakeChat()
    fake.queue(_tool_call_chunk())
    fake.queue(StreamResponse(content="salvaged answer", finish_reason="stop"))
    stub = _StubTool("web_search")
    registry = ToolRegistry()
    registry.register_tool(stub)

    calls = 0

    def _is_cancelled() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 3  # loop head on the second round

    engine = _make_engine(fake, registry, [], AgentConfig(max_iterations=5))

    with pytest.raises(AgentCancelledError) as exc_info:
        await engine.execute(
            _FakeContext(),
            query="q",
            session_id="s",
            message_id="m",
            is_cancelled=_is_cancelled,
        )

    salvaged = exc_info.value.state
    assert salvaged.is_complete is True
    assert salvaged.total_tool_calls() == 1
    assert salvaged.final_answer == "salvaged answer"


async def test_cancellation_mid_stream_preserves_partial_step() -> None:
    fake = _FakeChat()
    fake.queue(StreamResponse(content="partial thinking", finish_reason="length"))
    calls = 0

    def _is_cancelled() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 2  # after the streaming LLM call in round 0

    engine = _make_engine(fake, ToolRegistry(), [], AgentConfig(max_iterations=5))

    state = await engine.execute(
        _FakeContext(),
        query="q",
        session_id="s",
        message_id="m",
        is_cancelled=_is_cancelled,
    )

    assert state.is_complete is False
    assert len(state.round_steps) == 1
    assert state.round_steps[0].thought == "partial thinking"


# ── History handling ────────────────────────────────────────────────


async def test_history_kb_results_redacted_by_default() -> None:
    fake = _FakeChat()
    fake.queue(StreamResponse(content="answer", finish_reason="stop"))
    history = [
        Message(role="user", content="previous question"),
        Message(role="assistant", content="prev answer"),
        Message(
            role="tool",
            content="KB full result payload",
            tool_call_id="call-9",
            name="knowledge_search",
        ),
    ]
    engine = _make_engine(fake, ToolRegistry(), [], AgentConfig(max_iterations=5))

    await engine.execute(
        _FakeContext(),
        query="q",
        session_id="s",
        message_id="m",
        llm_context=history,
    )

    sent_messages, _ = fake.stream_calls[0]
    system_messages = [m for m in sent_messages if m.role == "system"]
    assert any("Previous retrieval result omitted" in m.content for m in system_messages)


async def test_history_kb_results_retained_when_configured() -> None:
    fake = _FakeChat()
    fake.queue(StreamResponse(content="answer", finish_reason="stop"))
    history = [
        Message(role="user", content="previous question"),
        Message(
            role="tool",
            content="KB full result payload",
            tool_call_id="call-9",
            name="knowledge_search",
        ),
    ]
    engine = _make_engine(
        fake, ToolRegistry(), [], AgentConfig(max_iterations=5, retain_retrieval_history=True)
    )

    await engine.execute(
        _FakeContext(),
        query="q",
        session_id="s",
        message_id="m",
        llm_context=history,
    )

    sent_messages, _ = fake.stream_calls[0]
    system_messages = [m for m in sent_messages if m.role == "system"]
    assert any("KB full result payload" in m.content for m in system_messages)


async def test_tool_calls_are_executed_through_the_registry_and_cleaned_up() -> None:
    class _CleanableTool(_StubTool):
        def __init__(self, name: str, *, cleaned: list[str]) -> None:
            super().__init__(name)
            self._cleaned = cleaned

        async def cleanup(self, ctx: Context) -> None:
            self._cleaned.append(self._name)

    cleaned: list[str] = []
    fake = _FakeChat()
    fake.queue(_tool_call_chunk())
    fake.queue(StreamResponse(content="done", finish_reason="stop"))
    registry = ToolRegistry()
    registry.register_tool(_CleanableTool("web_search", cleaned=cleaned))
    engine = _make_engine(fake, registry, [], AgentConfig(max_iterations=5))

    await engine.execute(_FakeContext(), query="q", session_id="s", message_id="m")

    assert cleaned == ["web_search"]


# ── Event stream behavior ───────────────────────────────────────────


async def test_thinking_chunks_emit_thought_events() -> None:
    fake = _FakeChat()
    fake.queue(
        StreamResponse(response_type=ResponseType.THINKING, content="reasoning..."),
        StreamResponse(content="answer", finish_reason="stop"),
    )
    events: list[Event] = []
    engine = _make_engine(fake, ToolRegistry(), events, AgentConfig(max_iterations=5))

    state = await engine.execute(_FakeContext(), query="q", session_id="s", message_id="m")

    assert state.final_answer == "answer"
    thought_events = [e for e in events if e.type is EventType.AGENT_THOUGHT]
    thought_contents = [
        str(e.data.get("content")) for e in thought_events if isinstance(e.data, dict)
    ]
    assert any("reasoning..." in content for content in thought_contents)
