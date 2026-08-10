"""Unit tests for the agent think / act / observe / finalize phases.

Drives each phase with a scripted fake chat model, a stub tool registered in a
real ``ToolRegistry``, a real request-local model-context ``Registry``, and a
recording event bus — no database, no network. Covers prompt assembly,
streaming event emission, retry and graceful degradation (think); registry
execution, JSON repair, unresolved handles, timeout and parallel calls (act);
message replay (observe); and synthesis, image requirement, fallback and the
completion event (finalize).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

import pytest

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
from src.core.agents.engine.act import ActPhase
from src.core.agents.engine.finalize import FinalizePhase
from src.core.agents.engine.modelcontext import Registry
from src.core.agents.engine.observe import ObservePhase
from src.core.agents.engine.think import ThinkPhase, sanitize_messages
from src.core.agents.engine.types import (
    MAX_ITERATIONS_FALLBACK,
    AgentConfig,
    AgentLLMError,
    AgentState,
    AgentStep,
    ToolCall,
    ToolResult,
)
from src.core.agents.tools.base import ToolResult as ToolsToolResult
from src.core.agents.tools.registry import ToolRegistry
from src.core.chat.bus import Event, EventBus
from src.core.chat.types import EventType

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

    def queue(self, *chunks: StreamResponse) -> None:
        """Queue one scripted stream for the next call."""
        self._scripted.append(_iter_stream(chunks))

    async def chat(self, messages: list[Message], opts: ChatOptions | None = None) -> ChatResponse:
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


def _make_bus(events: list[Event]) -> EventBus:
    """Return a bus recording every agent event into ``events``."""
    bus = EventBus()

    async def record(event: Event) -> None:
        events.append(event)

    for event_type in _AGENT_EVENT_TYPES:
        bus.on(event_type, record)
    return bus


def _make_think(fake: _FakeChat, events: list[Event]) -> ThinkPhase:
    """Build a ThinkPhase wired to a scripted chat and a recording bus."""
    return ThinkPhase(
        config=AgentConfig(temperature=0.7),
        chat_model=fake,
        event_bus=_make_bus(events),
        model_context=Registry(citations_enabled=True),
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


def _state_with_tool_result(output: str) -> AgentState:
    """Build a run state carrying one successful tool result."""
    return AgentState(
        round_steps=[
            AgentStep(
                iteration=0,
                thought="searching",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="web_search",
                        result=ToolResult(success=True, output=output),
                    )
                ],
            )
        ]
    )


# ── ThinkPhase: streaming ────────────────────────────────────────────


async def test_stream_thinking_routes_thinking_and_answer_events() -> None:
    fake = _FakeChat()
    fake.queue(
        StreamResponse(response_type=ResponseType.THINKING, content="reasoning"),
        StreamResponse(content=" answer", finish_reason="stop"),
    )
    events: list[Event] = []
    phase = _make_think(fake, events)

    response = await phase.stream_thinking_to_events(
        _FakeContext(), [Message(role="user", content="q")], [], 0, "s1"
    )

    assert response.content == "answer"
    assert response.reasoning_content == "reasoning"
    assert response.finish_reason == "stop"
    assert response.answer_streamed is True

    thought_contents = [
        str(e.data.get("content")) for e in events if isinstance(e.data, dict)
    ]
    assert "reasoning" in thought_contents
    answer_data = [
        e.data for e in events if e.type is EventType.AGENT_FINAL_ANSWER and isinstance(e.data, dict)
    ]
    assert any(str(d.get("content")) == " answer" for d in answer_data)


async def test_stream_thinking_strips_inline_think_blocks() -> None:
    fake = _FakeChat()
    fake.queue(StreamResponse(content="<think>hidden</think> Final", finish_reason="stop"))
    phase = _make_think(fake, [])

    response = await phase.stream_thinking_to_events(
        _FakeContext(), [Message(role="user", content="q")], [], 0, "s1"
    )

    assert response.content == "Final"


async def test_stream_thinking_emits_tool_call_pending_event() -> None:
    fake = _FakeChat()
    fake.queue(
        StreamResponse(
            response_type=ResponseType.TOOL_CALL,
            content="",
            data={"tool_call_id": "call-1", "tool_name": "web_search"},
        )
    )
    events: list[Event] = []
    phase = _make_think(fake, events)

    await phase.stream_thinking_to_events(_FakeContext(), [Message(role="user", content="q")], [], 0, "s1")

    pending = [e for e in events if e.type is EventType.AGENT_TOOL_CALL]
    assert len(pending) == 1
    assert pending[0].id == "call-1-tool-call-pending"
    assert pending[0].data == {
        "tool_call_id": "call-1",
        "tool_name": "web_search",
        "iteration": 0,
    }


# ── ThinkPhase: retry / graceful degradation ─────────────────────────


async def test_call_with_retry_returns_response_without_state_change() -> None:
    fake = _FakeChat()
    fake.queue(StreamResponse(content="ok", finish_reason="stop"))
    phase = _make_think(fake, [])
    state = AgentState()

    round_result = await phase.call_with_retry(
        _FakeContext(), [Message(role="user", content="q")], [], state, "q", 0, "s1"
    )

    assert round_result.response is not None
    assert round_result.response.content == "ok"
    assert round_result.state is state
    assert state.is_complete is False


async def test_call_with_retry_retries_transient_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop_sleep(_seconds: float) -> None:
        return

    monkeypatch.setattr("asyncio.sleep", _noop_sleep)
    fake = _FakeChat()
    fake.queue(StreamResponse(response_type=ResponseType.ERROR, content="rate limit exceeded"))
    fake.queue(StreamResponse(content="ok", finish_reason="stop"))
    phase = _make_think(fake, [])

    round_result = await phase.call_with_retry(
        _FakeContext(), [Message(role="user", content="q")], [], AgentState(), "q", 0, "s1"
    )

    assert round_result.response is not None
    assert round_result.response.content == "ok"
    assert len(fake.stream_calls) == 2


async def test_call_with_retry_raises_on_non_transient_error() -> None:
    fake = _FakeChat()
    fake.queue(StreamResponse(response_type=ResponseType.ERROR, content="provider exploded"))
    phase = _make_think(fake, [])

    with pytest.raises(AgentLLMError) as exc_info:
        await phase.call_with_retry(
            _FakeContext(), [Message(role="user", content="q")], [], AgentState(), "q", 0, "s1"
        )
    assert "LLM call failed" in str(exc_info.value)


async def test_call_with_retry_degrades_to_synthesis_with_prior_tool_results() -> None:
    fake = _FakeChat()
    fake.queue(StreamResponse(response_type=ResponseType.ERROR, content="provider exploded"))
    phase = _make_think(fake, [])
    state = _state_with_tool_result("found")
    synthesize_calls: list[str] = []

    async def synthesize(
        ctx: Context, query: str, current: AgentState, session_id: str
    ) -> AgentState:
        synthesize_calls.append(query)
        return current.model_copy(update={"final_answer": "degraded answer"})

    round_result = await phase.call_with_retry(
        _FakeContext(),
        [Message(role="user", content="q")],
        [],
        state,
        "q",
        0,
        "s1",
        synthesize=synthesize,
    )

    assert round_result.response is None
    assert round_result.state.is_complete is True
    assert round_result.state.final_answer == "degraded answer"
    assert synthesize_calls == ["q"]


async def test_call_with_retry_raises_when_degradation_also_fails() -> None:
    fake = _FakeChat()
    fake.queue(StreamResponse(response_type=ResponseType.ERROR, content="provider exploded"))
    phase = _make_think(fake, [])
    state = _state_with_tool_result("found")

    async def synthesize(
        ctx: Context, query: str, current: AgentState, session_id: str
    ) -> AgentState:
        raise AgentLLMError("synthesis failed")

    with pytest.raises(AgentLLMError) as exc_info:
        await phase.call_with_retry(
            _FakeContext(),
            [Message(role="user", content="q")],
            [],
            state,
            "q",
            0,
            "s1",
            synthesize=synthesize,
        )
    assert "synthesis also failed" in str(exc_info.value)


# ── ThinkPhase: prompt assembly ──────────────────────────────────────


async def test_build_messages_redacts_kb_results_by_default() -> None:
    fake = _FakeChat()
    fake.queue(StreamResponse(content="answer", finish_reason="stop"))
    phase = _make_think(fake, [])
    llm_context = [
        Message(role="user", content="previous question"),
        Message(
            role="tool",
            content="KB full result payload",
            tool_call_id="call-9",
            name="knowledge_search",
        ),
    ]

    messages = phase.build_messages_with_llm_context("sys", "q", "s1", llm_context, [])

    assert any("Previous retrieval result omitted" in m.content for m in messages)


async def test_build_messages_retains_history_when_configured() -> None:
    fake = _FakeChat()
    fake.queue(StreamResponse(content="answer", finish_reason="stop"))
    phase = ThinkPhase(
        config=AgentConfig(retain_retrieval_history=True),
        chat_model=fake,
        event_bus=_make_bus([]),
        model_context=Registry(citations_enabled=True),
    )
    llm_context = [
        Message(
            role="tool",
            content="KB full result payload",
            tool_call_id="call-9",
            name="knowledge_search",
        )
    ]

    messages = phase.build_messages_with_llm_context("sys", "q", "s1", llm_context, [])

    assert any("KB full result payload" in m.content for m in messages)


def test_render_user_turn_builds_runtime_context() -> None:
    fake = _FakeChat()
    phase = _make_think(fake, [])

    user_turn = phase.render_user_turn("s1", "What is RAG?")

    assert "<runtime_context scope=\"this_turn\">" in user_turn
    assert "<session>s1</session>" in user_turn
    assert user_turn.endswith("What is RAG?")


# ── ThinkPhase: message sanitisation ─────────────────────────────────


def test_sanitize_messages_merges_consecutive_same_role() -> None:
    messages = [Message(role="user", content="a"), Message(role="user", content="b")]

    sanitized = sanitize_messages(messages)

    assert len(sanitized) == 1
    assert sanitized[0].content == "a\n\nb"


def test_sanitize_messages_converts_orphan_tool_result_to_system() -> None:
    messages = [
        Message(
            role="tool",
            content="orphaned result",
            tool_call_id="call-9",
            name="web_search",
        )
    ]

    sanitized = sanitize_messages(messages)

    assert len(sanitized) == 1
    assert sanitized[0].role == "system"
    assert "orphaned result" in sanitized[0].content


def test_sanitize_messages_drops_empty_non_system_messages() -> None:
    messages = [
        Message(role="user", content="q"),
        Message(role="assistant", content=""),
        Message(role="system", content="sys"),
    ]

    sanitized = sanitize_messages(messages)

    assert [m.role for m in sanitized] == ["user", "system"]


# ── ActPhase ─────────────────────────────────────────────────────────


def _make_act(
    registry: ToolRegistry,
    events: list[Event],
    *,
    parallel: bool = False,
    tool_exec_timeout: float = 0.5,
) -> ActPhase:
    return ActPhase(
        config=AgentConfig(parallel_tool_calls=parallel),
        tool_registry=registry,
        event_bus=_make_bus(events),
        model_context=Registry(citations_enabled=True),
        tool_exec_timeout=tool_exec_timeout,
    )


async def test_execute_tool_calls_runs_through_registry() -> None:
    stub = _StubTool("web_search", output="search result payload")
    registry = ToolRegistry()
    registry.register_tool(stub)
    events: list[Event] = []
    phase = _make_act(registry, events)
    response = ChatResponse(
        tool_calls=[LLMToolCall(id="call-1", function=FunctionCall(name="web_search", arguments='{"query": "test"}'))]
    )

    tool_calls = await phase.execute_tool_calls(_FakeContext(), response, 0, "s1", "m1")

    assert len(tool_calls) == 1
    call = tool_calls[0]
    assert call.name == "web_search"
    assert call.args == {"query": "test"}
    assert call.result is not None and call.result.success
    assert call.result.output == "search result payload"
    assert stub.executed == 1
    assert stub.last_args == {"query": "test"}

    assert any(
        e.type is EventType.AGENT_TOOL_CALL and isinstance(e.data, dict) and "hint" in e.data
        for e in events
    )
    result_events = [e for e in events if e.type is EventType.AGENT_TOOL_RESULT]
    assert len(result_events) == 1
    assert result_events[0].data == {
        "tool_call_id": "call-1",
        "tool_name": "web_search",
        "output": "search result payload",
        "error": "",
        "success": True,
        "duration": call.duration,
        "iteration": 0,
        "data": None,
    }
    exec_events = [e for e in events if e.type is EventType.AGENT_TOOL]
    assert len(exec_events) == 1
    assert exec_events[0].data == {
        "iteration": 0,
        "tool_name": "web_search",
        "tool_input": {"query": "test"},
        "tool_output": "search result payload",
        "success": True,
        "error": "",
        "duration": call.duration,
    }


async def test_execute_tool_calls_empty_returns_empty() -> None:
    phase = _make_act(ToolRegistry(), [])

    tool_calls = await phase.execute_tool_calls(_FakeContext(), ChatResponse(), 0, "s1", "m1")

    assert tool_calls == []


async def test_execute_tool_calls_parallel_runs_all_calls() -> None:
    stub = _StubTool("web_search")
    registry = ToolRegistry()
    registry.register_tool(stub)
    phase = _make_act(registry, [], parallel=True)
    response = ChatResponse(
        tool_calls=[
            LLMToolCall(id="call-a", function=FunctionCall(name="web_search", arguments='{"query": "a"}')),
            LLMToolCall(id="call-b", function=FunctionCall(name="web_search", arguments='{"query": "b"}')),
        ]
    )

    tool_calls = await phase.execute_tool_calls(_FakeContext(), response, 0, "s1", "m1")

    assert len(tool_calls) == 2
    assert {tc.id for tc in tool_calls} == {"call-a", "call-b"}
    assert all(tc.result is not None and tc.result.success for tc in tool_calls)
    assert stub.executed == 2


async def test_execute_tool_calls_repairs_malformed_arguments() -> None:
    stub = _StubTool("web_search")
    registry = ToolRegistry()
    registry.register_tool(stub)
    phase = _make_act(registry, [])
    response = ChatResponse(
        tool_calls=[LLMToolCall(id="call-1", function=FunctionCall(name="web_search", arguments='{"query": "test"'))]
    )

    tool_calls = await phase.execute_tool_calls(_FakeContext(), response, 0, "s1", "m1")

    assert tool_calls[0].result is not None and tool_calls[0].result.success
    assert stub.last_args == {"query": "test"}


async def test_execute_tool_calls_fails_unresolved_handles_before_execution() -> None:
    stub = _StubTool("web_search")
    registry = ToolRegistry()
    registry.register_tool(stub)
    phase = _make_act(registry, [])
    response = ChatResponse(
        tool_calls=[
            LLMToolCall(
                id="call-1",
                function=FunctionCall(name="web_search", arguments='{"query": "res://0001"}'),
                unresolved_handles=["res://0001"],
            )
        ]
    )

    tool_calls = await phase.execute_tool_calls(_FakeContext(), response, 0, "s1", "m1")

    assert tool_calls[0].result is not None
    assert tool_calls[0].result.success is False
    assert "unresolved model handles" in tool_calls[0].result.error
    assert stub.executed == 0


async def test_execute_tool_calls_converts_unknown_tool_to_failed_result() -> None:
    phase = _make_act(ToolRegistry(), [])
    response = ChatResponse(
        tool_calls=[
            LLMToolCall(
                id="call-1", function=FunctionCall(name="does_not_exist", arguments="{}")
            )
        ]
    )

    tool_calls = await phase.execute_tool_calls(_FakeContext(), response, 0, "s1", "m1")

    assert tool_calls[0].result is not None
    assert tool_calls[0].result.success is False
    assert "does_not_exist" in tool_calls[0].result.error


async def test_execute_tool_calls_times_out_slow_tools() -> None:
    class _SlowTool(_StubTool):
        async def execute(self, ctx: Context, args: str) -> ToolsToolResult:
            self.executed += 1
            await asyncio.sleep(5)
            return ToolsToolResult(success=True, output="late")

    registry = ToolRegistry()
    registry.register_tool(_SlowTool("web_search"))
    phase = _make_act(registry, [], tool_exec_timeout=0.05)
    response = ChatResponse(
        tool_calls=[LLMToolCall(id="call-1", function=FunctionCall(name="web_search", arguments="{}"))]
    )

    tool_calls = await phase.execute_tool_calls(_FakeContext(), response, 0, "s1", "m1")

    assert tool_calls[0].result is not None
    assert tool_calls[0].result.success is False
    assert "timed out" in tool_calls[0].result.error


# ── ObservePhase ─────────────────────────────────────────────────────


async def test_append_tool_results_adds_assistant_and_tool_messages() -> None:
    phase = ObservePhase(Registry(citations_enabled=True))
    messages = [Message(role="user", content="q")]
    step = AgentStep(
        iteration=0,
        thought="searching",
        reasoning_content="reasoning",
        tool_calls=[
            ToolCall(
                id="call-1",
                name="web_search",
                args={"query": "test"},
                result=ToolResult(success=True, output="payload"),
            )
        ],
    )

    out = phase.append_tool_results(messages, step)

    assert [m.role for m in out] == ["user", "assistant", "tool"]
    assistant = out[1]
    assert assistant.content == "searching"
    assert assistant.reasoning_content == "reasoning"
    assert len(assistant.tool_calls) == 1
    assert assistant.tool_calls[0].id == "call-1"
    assert assistant.tool_calls[0].function.name == "web_search"
    assert assistant.tool_calls[0].function.arguments == '{"query": "test"}'
    tool_msg = out[2]
    assert tool_msg.tool_call_id == "call-1"
    assert tool_msg.name == "web_search"
    assert "payload" in tool_msg.content


async def test_append_tool_results_skips_empty_assistant() -> None:
    phase = ObservePhase(Registry(citations_enabled=True))
    messages = [Message(role="user", content="q")]

    out = phase.append_tool_results(messages, AgentStep(iteration=0))

    assert out == messages
    assert [m.role for m in out] == ["user"]


async def test_append_tool_results_renders_failed_results() -> None:
    phase = ObservePhase(Registry(citations_enabled=True))
    messages: list[Message] = []
    step = AgentStep(
        iteration=0,
        thought="trying",
        tool_calls=[
            ToolCall(
                id="call-1",
                name="web_search",
                args={"query": "test"},
                result=ToolResult(success=False, error="boom"),
            )
        ],
    )

    out = phase.append_tool_results(messages, step)

    assert [m.role for m in out] == ["assistant", "tool"]
    assert out[1].tool_call_id == "call-1"
    assert out[1].content != ""


# ── FinalizePhase ────────────────────────────────────────────────────


def _make_finalize(fake: _FakeChat, events: list[Event]) -> FinalizePhase:
    think = _make_think(fake, events)
    return FinalizePhase(
        config=AgentConfig(temperature=0.7),
        event_bus=_make_bus(events),
        model_context=Registry(citations_enabled=True),
        think=think,
    )


async def test_synthesize_final_answer_sets_final_answer_and_emits() -> None:
    fake = _FakeChat()
    fake.queue(StreamResponse(content="final answer", finish_reason="stop"))
    events: list[Event] = []
    finalize = _make_finalize(fake, events)
    state = _state_with_tool_result("found payload")

    new_state = await finalize.synthesize_final_answer(_FakeContext(), "query", state, "s1")

    assert new_state.final_answer == "final answer"
    assert new_state.is_complete is False
    assert new_state.total_tool_calls() == 1

    sent_messages, _ = fake.stream_calls[0]
    assert any("Tool web_search returned:" in m.content for m in sent_messages)
    assert any("found payload" in m.content for m in sent_messages)
    last_content = sent_messages[-1].content
    assert "User question: query" in last_content

    final_answers = [e for e in events if e.type is EventType.AGENT_FINAL_ANSWER]
    assert final_answers[-1].data == {"content": "", "done": True}
    streamed = "".join(
        str(d.data.get("content")) for d in final_answers[:-1] if isinstance(d.data, dict)
    )
    assert "final answer" in streamed


async def test_synthesize_final_answer_adds_image_requirement_when_present() -> None:
    fake = _FakeChat()
    fake.queue(StreamResponse(content="final", finish_reason="stop"))
    finalize = _make_finalize(fake, [])
    state = _state_with_tool_result("![diagram](https://example.com/d.png)")

    await finalize.synthesize_final_answer(_FakeContext(), "query", state, "s1")

    sent_messages, _ = fake.stream_calls[0]
    assert any("MUST include at least one relevant Markdown image" in m.content for m in sent_messages)


async def test_synthesize_final_answer_omits_image_requirement_when_absent() -> None:
    fake = _FakeChat()
    fake.queue(StreamResponse(content="final", finish_reason="stop"))
    finalize = _make_finalize(fake, [])
    state = _state_with_tool_result("plain text result")

    await finalize.synthesize_final_answer(_FakeContext(), "query", state, "s1")

    sent_messages, _ = fake.stream_calls[0]
    assert not any("MUST include at least one relevant Markdown image" in m.content for m in sent_messages)


async def test_synthesize_final_answer_strips_think_blocks() -> None:
    fake = _FakeChat()
    fake.queue(StreamResponse(content="<think>hidden</think> clean answer", finish_reason="stop"))
    finalize = _make_finalize(fake, [])
    state = _state_with_tool_result("payload")

    new_state = await finalize.synthesize_final_answer(_FakeContext(), "query", state, "s1")

    assert new_state.final_answer == "clean answer"


async def test_handle_max_iterations_sets_complete() -> None:
    fake = _FakeChat()
    fake.queue(StreamResponse(content="synth", finish_reason="stop"))
    finalize = _make_finalize(fake, [])
    state = _state_with_tool_result("payload")

    new_state = await finalize.handle_max_iterations(_FakeContext(), "q", state, "s1")

    assert new_state.is_complete is True
    assert new_state.final_answer == "synth"


async def test_handle_max_iterations_falls_back_when_synthesis_fails() -> None:
    fake = _FakeChat()
    fake.queue(StreamResponse(response_type=ResponseType.ERROR, content="boom"))
    finalize = _make_finalize(fake, [])
    state = _state_with_tool_result("payload")

    new_state = await finalize.handle_max_iterations(_FakeContext(), "q", state, "s1")

    assert new_state.is_complete is True
    assert new_state.final_answer == MAX_ITERATIONS_FALLBACK


async def test_emit_completion_event_emits_summary() -> None:
    events: list[Event] = []
    finalize = _make_finalize(_FakeChat(), events)
    state = _state_with_tool_result("payload")

    await finalize.emit_completion_event(state, "s1", "m1", time.monotonic())

    complete_events = [e for e in events if e.type is EventType.AGENT_COMPLETE]
    assert len(complete_events) == 1
    data = complete_events[0].data
    assert isinstance(data, dict)
    assert data.get("session_id") == "s1"
    assert data.get("message_id") == "m1"
    assert data.get("total_steps") == 1
    assert data.get("final_answer") == ""
    agent_steps = data.get("agent_steps")
    assert isinstance(agent_steps, list)
    assert len(agent_steps) == 1
    assert isinstance(data.get("total_duration_ms"), int)
