"""Unit tests for token estimation and context compression.

The default estimator is a dependency-free heuristic, so the tests are fully
deterministic; an injectable encoder seam is also verified so a real BPE
tokenizer can be plugged in later without changing the message codec.
"""

from __future__ import annotations

from src.core.agents.engine.token_est import (
    DEFAULT_CONTEXT_THRESHOLD_RATIO,
    PER_CONVERSATION_TAIL,
    PER_MESSAGE_OVERHEAD,
    AgentFunctionCall,
    AgentMessage,
    AgentToolCall,
    TokenEstimator,
    compress_context,
    group_tool_messages,
)


def _tool_call(call_id: str, name: str) -> AgentToolCall:
    """Build a minimal tool call for tests."""
    return AgentToolCall(
        id=call_id,
        function=AgentFunctionCall(name=name, arguments='{"q":"test"}'),
    )


# ── estimate_string ────────────────────────────────────────────────────


def test_estimate_string_empty_is_zero() -> None:
    estimator = TokenEstimator()
    assert estimator.estimate_string("") == 0


def test_estimate_string_english_uses_heuristic() -> None:
    estimator = TokenEstimator()
    assert estimator.estimate_string("hello world") == 3


def test_estimate_string_cjk_costs_more_than_latin() -> None:
    estimator = TokenEstimator()
    latin = "a" * 100
    cjk = "中" * 100
    assert estimator.estimate_string(cjk) > estimator.estimate_string(latin)


def test_estimate_string_injected_encoder_is_used() -> None:
    estimator = TokenEstimator(encoder=lambda _text: 42)
    assert estimator.estimate_string("anything") == 42


def test_estimate_string_encoder_failure_falls_back() -> None:
    def _boom(_text: str) -> int:
        raise RuntimeError("tokenizer unavailable")

    estimator = TokenEstimator(encoder=_boom)
    assert estimator.estimate_string("hello world") == 3


def test_estimate_string_negative_encoder_output_clamped() -> None:
    estimator = TokenEstimator(encoder=lambda _text: -5)
    assert estimator.estimate_string("anything") == 0


# ── estimate_message / estimate_messages ───────────────────────────────


def test_estimate_message_includes_overhead() -> None:
    estimator = TokenEstimator()
    message = AgentMessage(role="assistant", content="hello")
    content_tokens = estimator.estimate_string("hello")
    assert estimator.estimate_message(message) > content_tokens


def test_estimate_message_with_tool_calls() -> None:
    estimator = TokenEstimator()
    message = AgentMessage(
        role="assistant",
        content="thinking...",
        tool_calls=(_tool_call("call_1", "knowledge_search"),),
    )
    assert estimator.estimate_message(message) > 10


def test_estimate_messages_adds_conversation_tail() -> None:
    estimator = TokenEstimator()
    messages = [
        AgentMessage(role="system", content="You are a helpful assistant."),
        AgentMessage(role="user", content="Hello"),
        AgentMessage(role="assistant", content="Hi there!"),
    ]
    assert estimator.estimate_messages(messages) > 10
    single = [AgentMessage(role="user", content="Hello")]
    assert estimator.estimate_messages(single) == (
        estimator.estimate_message(single[0]) + PER_CONVERSATION_TAIL
    )


def test_overhead_constants_are_sane() -> None:
    assert PER_MESSAGE_OVERHEAD == 3
    assert PER_CONVERSATION_TAIL == 3
    assert DEFAULT_CONTEXT_THRESHOLD_RATIO == 0.8


# ── group_tool_messages ────────────────────────────────────────────────


def test_group_tool_messages_standalone_messages() -> None:
    messages = [
        AgentMessage(role="user", content="hello"),
        AgentMessage(role="assistant", content="hi"),
    ]
    groups = group_tool_messages(messages)
    assert len(groups) == 2


def test_group_tool_messages_groups_assistant_with_results() -> None:
    messages = [
        AgentMessage(role="user", content="search for X"),
        AgentMessage(
            role="assistant",
            content="thinking",
            tool_calls=(_tool_call("call_1", "search"), _tool_call("call_2", "fetch")),
        ),
        AgentMessage(role="tool", content="result1", name="search", tool_call_id="call_1"),
        AgentMessage(role="tool", content="result2", name="fetch", tool_call_id="call_2"),
        AgentMessage(role="user", content="thanks"),
    ]
    groups = group_tool_messages(messages)
    assert len(groups) == 3
    assert len(groups[1]) == 3


# ── compress_context ───────────────────────────────────────────────────


def test_compress_context_no_compression_needed() -> None:
    messages = [
        AgentMessage(role="system", content="system prompt"),
        AgentMessage(role="user", content="hello"),
    ]
    estimator = TokenEstimator()
    tokens = estimator.estimate_messages(messages)
    result = compress_context(messages, estimator, 100_000, tokens)
    assert result == messages


def test_compress_context_zero_max_tokens_unchanged() -> None:
    messages = [
        AgentMessage(role="system", content="system"),
        AgentMessage(role="user", content="hello"),
    ]
    result = compress_context(messages, TokenEstimator(), 0, 0)
    assert result == messages


def test_compress_context_two_messages_unchanged() -> None:
    messages = [
        AgentMessage(role="system", content="system"),
        AgentMessage(role="user", content="hello"),
    ]
    result = compress_context(messages, TokenEstimator(), 1, 999)
    assert result == messages


def test_compress_context_preserves_system_and_last_message() -> None:
    messages = [
        AgentMessage(role="system", content="system prompt"),
        AgentMessage(role="user", content="old message " * 1000),
        AgentMessage(role="assistant", content="old response " * 1000),
        AgentMessage(role="user", content="current query"),
    ]
    estimator = TokenEstimator()
    tokens = estimator.estimate_messages(messages)
    result = compress_context(messages, estimator, 100, tokens)
    assert len(result) >= 2
    assert result[0].role == "system"
    assert result[-1].content == "current query"


def test_compress_context_keeps_tool_pair_paired() -> None:
    messages = [
        AgentMessage(role="system", content="system prompt"),
        AgentMessage(role="user", content="x" * 500),
        AgentMessage(role="assistant", content="y" * 500),
        AgentMessage(
            role="assistant",
            content="thinking",
            tool_calls=(_tool_call("call_1", "search"),),
        ),
        AgentMessage(role="tool", content="result " * 100, name="search", tool_call_id="call_1"),
        AgentMessage(role="user", content="what did you find?"),
    ]
    estimator = TokenEstimator()
    tokens = estimator.estimate_messages(messages)
    result = compress_context(messages, estimator, 200, tokens)

    for i, msg in enumerate(result):
        if msg.role == "tool":
            assert i > 0
            prev = result[i - 1]
            assert prev.role == "assistant"
            assert len(prev.tool_calls) > 0


def test_compress_context_round2_preserves_current_turn() -> None:
    long_text = "filler content " * 200
    messages = [
        AgentMessage(role="system", content="system prompt"),
        AgentMessage(role="user", content=long_text),
        AgentMessage(role="assistant", content=long_text),
        AgentMessage(role="user", content=long_text),
        AgentMessage(role="assistant", content=long_text),
        AgentMessage(role="user", content="current question"),
        AgentMessage(
            role="assistant",
            content="let me search",
            tool_calls=(_tool_call("c1", "search"),),
        ),
        AgentMessage(role="tool", content="search results", name="search", tool_call_id="c1"),
    ]
    estimator = TokenEstimator()
    tokens = estimator.estimate_messages(messages)
    result = compress_context(messages, estimator, 300, tokens)

    assert result[0].role == "system"
    assert result[0].content == "system prompt"
    user_idx = next(i for i, m in enumerate(result) if m.content == "current question")
    assert len(result) > user_idx + 2
    assert result[user_idx + 1].role == "assistant"
    assert result[user_idx + 1].content == "let me search"
    assert result[user_idx + 2].role == "tool"
    assert result[user_idx + 2].content == "search results"
    assert len(result) < len(messages)


def test_compress_context_parallel_tool_results_preserved() -> None:
    long_text = "data " * 300
    messages = [
        AgentMessage(role="system", content="sys"),
        AgentMessage(role="user", content=long_text),
        AgentMessage(role="assistant", content=long_text),
        AgentMessage(role="user", content="do things"),
        AgentMessage(
            role="assistant",
            content="ok",
            tool_calls=(_tool_call("c1", "t1"), _tool_call("c2", "t2")),
        ),
        AgentMessage(role="tool", content="res1", name="t1", tool_call_id="c1"),
        AgentMessage(role="tool", content="res2", name="t2", tool_call_id="c2"),
    ]
    estimator = TokenEstimator()
    tokens = estimator.estimate_messages(messages)
    result = compress_context(messages, estimator, 200, tokens)

    tool_names = [m.name for m in result if m.role == "tool"]
    assert "t1" in tool_names
    assert "t2" in tool_names
    assert any(m.content == "do things" for m in result)


def test_compress_context_no_history_returns_unchanged() -> None:
    messages = [
        AgentMessage(role="system", content="sys"),
        AgentMessage(role="user", content="hello"),
        AgentMessage(
            role="assistant",
            content="thinking",
            tool_calls=(_tool_call("c1", "t1"),),
        ),
        AgentMessage(role="tool", content="done", name="t1", tool_call_id="c1"),
    ]
    estimator = TokenEstimator()
    tokens = estimator.estimate_messages(messages)
    result = compress_context(messages, estimator, 10, tokens)
    assert result == messages


def test_compress_context_no_user_message_returns_non_empty() -> None:
    messages = [
        AgentMessage(role="system", content="sys"),
        AgentMessage(role="assistant", content="x" * 1000),
        AgentMessage(role="assistant", content="y" * 1000),
    ]
    estimator = TokenEstimator()
    tokens = estimator.estimate_messages(messages)
    result = compress_context(messages, estimator, 10, tokens)
    assert len(result) >= 1


def test_compress_context_tool_pair_in_history_never_split() -> None:
    long_text = "verbose " * 200
    messages = [
        AgentMessage(role="system", content="sys"),
        AgentMessage(role="user", content="old query 1"),
        AgentMessage(
            role="assistant",
            content=long_text,
            tool_calls=(_tool_call("old1", "old_tool"),),
        ),
        AgentMessage(role="tool", content=long_text, name="old_tool", tool_call_id="old1"),
        AgentMessage(role="user", content="old query 2"),
        AgentMessage(role="assistant", content="short reply"),
        AgentMessage(role="user", content="current"),
        AgentMessage(
            role="assistant",
            content="working",
            tool_calls=(_tool_call("new1", "new_tool"),),
        ),
        AgentMessage(role="tool", content="result", name="new_tool", tool_call_id="new1"),
    ]
    estimator = TokenEstimator()
    tokens = estimator.estimate_messages(messages)
    result = compress_context(messages, estimator, 300, tokens)

    for i, msg in enumerate(result):
        if msg.role == "tool":
            assert i > 0, "tool message at index 0 is impossible"
            prev = result[i - 1]
            assert prev.role == "tool" or (prev.role == "assistant" and len(prev.tool_calls) > 0), (
                f"tool message at {i} must be preceded by assistant+tool_calls or another tool"
            )
