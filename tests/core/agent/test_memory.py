"""Unit tests for the agent memory consolidator.

The consolidator reaches the LLM only through an injectable chat seam, so the
tests drive it with stub async callables and a deterministic estimator — no
tokenizer, network, or database required.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest

from src.common.exception import AIProviderError
from src.core.agents.engine.memory import (
    CONSOLIDATION_SYSTEM_PROMPT,
    DEFAULT_CONSOLIDATION_THRESHOLD,
    MAX_CONSOLIDATION_ATTEMPTS,
    MemoryConsolidator,
    _truncate_for_prompt,
)
from src.core.agents.engine.token_est import (
    AgentFunctionCall,
    AgentMessage,
    AgentToolCall,
    TokenEstimator,
)


def _chat_stub(
    *,
    response: str = "",
    error: Exception | None = None,
) -> Callable[[list[AgentMessage]], Awaitable[str]]:
    """Return an async chat seam returning ``response`` or raising ``error``."""

    async def chat(_messages: list[AgentMessage]) -> str:
        if error is not None:
            raise error
        return response

    return chat


def _tool_call(call_id: str, name: str) -> AgentToolCall:
    """Build a minimal tool call for tests."""
    return AgentToolCall(
        id=call_id,
        function=AgentFunctionCall(name=name, arguments="{}"),
    )


def _long_history() -> list[AgentMessage]:
    """A history that clearly exceeds the small budgets used below."""
    long = "data " * 300
    return [
        AgentMessage(role="system", content="sys"),
        AgentMessage(role="user", content=long),
        AgentMessage(role="assistant", content=long),
        AgentMessage(role="user", content=long),
        AgentMessage(role="assistant", content=long),
        AgentMessage(role="user", content="current"),
        AgentMessage(
            role="assistant",
            content="thinking",
            tool_calls=(_tool_call("c1", "t1"),),
        ),
        AgentMessage(role="tool", content="res", name="t1", tool_call_id="c1"),
    ]


# ── should_consolidate ─────────────────────────────────────────────────


def test_should_consolidate_below_threshold_returns_false() -> None:
    consolidator = MemoryConsolidator(_chat_stub(), TokenEstimator(), 100_000)
    messages = [
        AgentMessage(role="system", content="system prompt"),
        AgentMessage(role="user", content="hello"),
        AgentMessage(role="assistant", content="hi"),
        AgentMessage(role="user", content="current"),
    ]
    tokens = TokenEstimator().estimate_messages(messages)
    assert consolidator.should_consolidate(tokens) is False


def test_should_consolidate_over_threshold_returns_true() -> None:
    consolidator = MemoryConsolidator(_chat_stub(), TokenEstimator(), 10)
    assert consolidator.should_consolidate(100) is True


def test_should_consolidate_disabled_returns_false() -> None:
    consolidator = MemoryConsolidator(_chat_stub(), TokenEstimator(), 0)
    assert consolidator.should_consolidate(99_999) is False


def test_threshold_out_of_range_defaults_to_050() -> None:
    consolidator = MemoryConsolidator(_chat_stub(), TokenEstimator(), 100, 1.5)
    assert consolidator.should_consolidate(60) is True
    assert consolidator.should_consolidate(50) is False
    assert DEFAULT_CONSOLIDATION_THRESHOLD == 0.5


# ── prompt / archive helpers ───────────────────────────────────────────


def test_truncate_for_prompt_short_string_unchanged() -> None:
    assert _truncate_for_prompt("hello", 100) == "hello"


def test_truncate_for_prompt_long_string_truncated() -> None:
    result = _truncate_for_prompt("hello world this is long", 10)
    assert result == "hello worl..."


def test_truncate_for_prompt_cjk_truncated_by_rune() -> None:
    result = _truncate_for_prompt("你好世界测试数据中文字符串", 5)
    assert result == "你好世界测..."


def test_raw_archive_formats_roles() -> None:
    consolidator = MemoryConsolidator(_chat_stub(), TokenEstimator(), 100_000)
    messages = [
        AgentMessage(role="user", content="search for X"),
        AgentMessage(
            role="assistant",
            content="let me search",
            tool_calls=(_tool_call("c1", "knowledge_search"),),
        ),
        AgentMessage(
            role="tool", content="result data", name="knowledge_search", tool_call_id="c1"
        ),
    ]
    result = consolidator._raw_archive(messages)
    assert "Raw conversation archive" in result
    assert "User: search for X" in result
    assert "knowledge_search" in result
    assert "Tool[knowledge_search]: result data" in result


def test_build_consolidation_prompt_formats_roles() -> None:
    consolidator = MemoryConsolidator(_chat_stub(), TokenEstimator(), 100_000)
    messages = [
        AgentMessage(role="user", content="find info about AI"),
        AgentMessage(
            role="assistant",
            content="searching...",
            tool_calls=(_tool_call("c1", "web_search"),),
        ),
        AgentMessage(role="tool", content="results here", name="web_search", tool_call_id="c1"),
    ]
    prompt = consolidator._build_consolidation_prompt(messages)
    assert "**User**: find info about AI" in prompt
    assert "web_search" in prompt
    assert "**Tool [web_search]**: results here" in prompt


# ── consolidate ────────────────────────────────────────────────────────


async def test_consolidate_too_few_messages_unchanged() -> None:
    consolidator = MemoryConsolidator(_chat_stub(response="summary"), TokenEstimator(), 100)
    messages = [
        AgentMessage(role="system", content="sys"),
        AgentMessage(role="user", content="hi"),
        AgentMessage(role="assistant", content="hello"),
    ]
    result = await consolidator.consolidate(messages)
    assert result == messages


async def test_consolidate_current_query_at_end_preserves_tail() -> None:
    consolidator = MemoryConsolidator(
        _chat_stub(response="summary of old history"), TokenEstimator(), 200
    )
    long = "old context data " * 200
    messages = [
        AgentMessage(role="system", content="You are a helpful assistant"),
        AgentMessage(role="user", content=long),
        AgentMessage(role="assistant", content=long),
        AgentMessage(role="user", content=long),
        AgentMessage(role="assistant", content=long),
        AgentMessage(role="user", content="current question"),
    ]

    result = await consolidator.consolidate(messages)

    assert result[0].role == "system"
    assert result[0].content == "You are a helpful assistant"
    assert result[-1].role == "user"
    assert result[-1].content == "current question"
    assert result[1].role == "system"
    assert "Memory Summary" in result[1].content
    assert len(result) < len(messages)


async def test_consolidate_round2_preserves_current_turn() -> None:
    consolidator = MemoryConsolidator(
        _chat_stub(response="consolidated history"), TokenEstimator(), 300
    )
    long_content = "verbose content " * 200
    messages = [
        AgentMessage(role="system", content="You are a helpful assistant"),
        AgentMessage(role="user", content=long_content),
        AgentMessage(role="assistant", content=long_content),
        AgentMessage(role="user", content=long_content),
        AgentMessage(
            role="assistant",
            content=long_content,
            tool_calls=(_tool_call("old_call", "search"),),
        ),
        AgentMessage(role="tool", content=long_content, name="search", tool_call_id="old_call"),
        AgentMessage(role="user", content="what is the weather today?"),
        AgentMessage(
            role="assistant",
            content="let me check",
            tool_calls=(_tool_call("call_1", "weather"),),
        ),
        AgentMessage(role="tool", content="sunny, 25°C", name="weather", tool_call_id="call_1"),
    ]

    result = await consolidator.consolidate(messages)

    assert result[0].role == "system"
    user_query_idx = next(
        i
        for i, m in enumerate(result)
        if m.role == "user" and m.content == "what is the weather today?"
    )
    assert len(result) > user_query_idx + 2
    assert result[user_query_idx + 1].role == "assistant"
    assert result[user_query_idx + 1].content == "let me check"
    assert len(result[user_query_idx + 1].tool_calls) == 1
    assert result[user_query_idx + 2].role == "tool"
    assert result[user_query_idx + 2].content == "sunny, 25°C"

    assert any(m.role == "system" and "Memory Summary" in m.content for m in result)
    assert len(result) < len(messages)


async def test_consolidate_parallel_tool_calls_preserved() -> None:
    consolidator = MemoryConsolidator(_chat_stub(response="summary"), TokenEstimator(), 300)
    long_content = "filler " * 300
    messages = [
        AgentMessage(role="system", content="sys"),
        AgentMessage(role="user", content=long_content),
        AgentMessage(role="assistant", content=long_content),
        AgentMessage(role="user", content="do two things"),
        AgentMessage(
            role="assistant",
            content="ok",
            tool_calls=(_tool_call("c1", "toolA"), _tool_call("c2", "toolB")),
        ),
        AgentMessage(role="tool", content="resultA", name="toolA", tool_call_id="c1"),
        AgentMessage(role="tool", content="resultB", name="toolB", tool_call_id="c2"),
    ]

    result = await consolidator.consolidate(messages)

    tool_names = [m.name for m in result if m.role == "tool"]
    assert "toolA" in tool_names
    assert "toolB" in tool_names
    assert any(m.role == "user" and m.content == "do two things" for m in result)


async def test_consolidate_no_user_message_unchanged() -> None:
    consolidator = MemoryConsolidator(_chat_stub(response="summary"), TokenEstimator(), 100)
    messages = [
        AgentMessage(role="system", content="sys"),
        AgentMessage(role="assistant", content="hello"),
        AgentMessage(role="assistant", content="world"),
        AgentMessage(role="assistant", content="more"),
    ]
    result = await consolidator.consolidate(messages)
    assert result == messages


async def test_consolidate_only_current_turn_unchanged() -> None:
    consolidator = MemoryConsolidator(_chat_stub(response="summary"), TokenEstimator(), 100)
    messages = [
        AgentMessage(role="system", content="sys"),
        AgentMessage(role="user", content="hello"),
        AgentMessage(
            role="assistant",
            content="let me help",
            tool_calls=(_tool_call("c1", "t1"),),
        ),
        AgentMessage(role="tool", content="done", name="t1", tool_call_id="c1"),
    ]
    result = await consolidator.consolidate(messages)
    assert result == messages


async def test_consolidate_llm_failure_falls_back_to_raw_archive() -> None:
    consolidator = MemoryConsolidator(
        _chat_stub(error=RuntimeError("provider down")), TokenEstimator(), 200
    )
    messages = _long_history()

    result = await consolidator.consolidate(messages)

    summary_messages = [m for m in result if m.role == "system" and "Memory Summary" in m.content]
    assert len(summary_messages) == 1
    assert "Raw conversation archive" in summary_messages[0].content
    assert result[-1].role == "tool"
    assert any(m.role == "user" and m.content == "current" for m in result)


async def test_consolidate_retries_up_to_max_attempts() -> None:
    calls: list[int] = []

    async def failing_chat(_messages: list[AgentMessage]) -> str:
        calls.append(1)
        raise RuntimeError("provider down")

    consolidator = MemoryConsolidator(failing_chat, TokenEstimator(), 200)
    await consolidator.consolidate(_long_history())
    assert len(calls) == MAX_CONSOLIDATION_ATTEMPTS


async def test_consolidate_empty_summary_retries_then_raises() -> None:
    calls: list[int] = []

    async def empty_chat(_messages: list[AgentMessage]) -> str:
        calls.append(1)
        return ""

    consolidator = MemoryConsolidator(empty_chat, TokenEstimator(), 200)
    with pytest.raises(AIProviderError) as exc_info:
        await consolidator._summarize_with_retry(_long_history()[1:5])
    assert len(calls) == MAX_CONSOLIDATION_ATTEMPTS
    assert "summarization failed" in str(exc_info.value)


async def test_consolidate_summary_uses_prompt_with_system_prompt() -> None:
    received: list[list[AgentMessage]] = []

    async def capturing_chat(messages: list[AgentMessage]) -> str:
        received.append(messages)
        return "summary"

    consolidator = MemoryConsolidator(capturing_chat, TokenEstimator(), 200)
    await consolidator.consolidate(_long_history())

    assert len(received) == 1
    assert received[0][0].role == "system"
    assert received[0][0].content == CONSOLIDATION_SYSTEM_PROMPT
    assert received[0][1].role == "user"
    assert "Conversation to summarize" in received[0][1].content
