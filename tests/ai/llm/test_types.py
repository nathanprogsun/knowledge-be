"""Contract tests for the chat request / response types.

Every assertion here pins a field name or JSON serialization name that is part
of the frozen wire contract (mirrored from the reference implementation), so a
renaming regression fails fast.
"""

from __future__ import annotations

import json

from src.ai.llm.types import (
    ChatOptions,
    ChatResponse,
    FunctionCall,
    FunctionDef,
    ImageURL,
    LLMToolCall,
    Message,
    MessageContentPart,
    PromptCacheStatus,
    ResponseType,
    StreamResponse,
    TokenUsage,
    Tool,
    ToolCall,
)

# ── Request types ────────────────────────────────────────────────────


def test_chat_options_serialization_names() -> None:
    dumped = ChatOptions(
        temperature=0.7,
        top_p=0.9,
        seed=42,
        max_tokens=256,
        max_completion_tokens=128,
        frequency_penalty=0.1,
        presence_penalty=0.2,
        thinking=True,
        tool_choice="auto",
        parallel_tool_calls=True,
    ).model_dump()
    assert set(dumped) == {
        "temperature",
        "top_p",
        "seed",
        "max_tokens",
        "max_completion_tokens",
        "frequency_penalty",
        "presence_penalty",
        "thinking",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "format",
    }


def test_chat_options_tools_round_trip() -> None:
    tool = Tool(
        type="function",
        function=FunctionDef(
            name="lookup", description="looks up", parameters={"type": "object"}
        ),
    )
    opts = ChatOptions(tools=[tool])
    dumped = opts.model_dump()
    assert dumped["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "looks up",
                "parameters": {"type": "object"},
            },
        }
    ]


def test_message_serialization_names() -> None:
    message = Message(role="assistant", content="hi", reasoning_content="think")
    dumped = message.model_dump()
    assert set(dumped) == {
        "role",
        "content",
        "multi_content",
        "name",
        "tool_call_id",
        "tool_calls",
        "images",
        "reasoning_content",
    }


def test_message_content_part_and_image_url() -> None:
    part = MessageContentPart(
        type="image_url",
        image_url=ImageURL(url="http://img", detail="high"),
    )
    dumped = part.model_dump()
    assert dumped == {
        "type": "image_url",
        "text": "",
        "image_url": {"url": "http://img", "detail": "high"},
    }


def test_tool_call_serialization_names() -> None:
    tool_call = ToolCall(
        id="call_1",
        type="function",
        function=FunctionCall(name="lookup", arguments='{"q":"x"}'),
        provider_metadata={"google": {"foo": "bar"}},
    )
    dumped = tool_call.model_dump()
    assert set(dumped) == {"id", "type", "function", "provider_metadata"}
    assert dumped["function"] == {"name": "lookup", "arguments": '{"q":"x"}'}


# ── Response contract ────────────────────────────────────────────────


def test_token_usage_serialization_names() -> None:
    usage = TokenUsage(
        prompt_tokens=1,
        completion_tokens=2,
        total_tokens=3,
        cached_tokens=1,
        cache_read_tokens=1,
        cache_write_tokens=0,
        cache_miss_tokens=2,
        cache_reported=True,
        cache_status=PromptCacheStatus.HIT,
    )
    dumped = usage.model_dump()
    assert set(dumped) == {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "cache_miss_tokens",
        "cache_reported",
        "cache_status",
    }


def test_set_prompt_cache_usage_hit() -> None:
    usage = TokenUsage()
    usage.set_prompt_cache_usage(5, 2, 10, True)
    assert usage.cached_tokens == 5
    assert usage.cache_read_tokens == 5
    assert usage.cache_write_tokens == 2
    assert usage.cache_miss_tokens == 10
    assert usage.cache_reported is True
    assert usage.cache_status == PromptCacheStatus.HIT


def test_set_prompt_cache_usage_miss() -> None:
    usage = TokenUsage()
    usage.set_prompt_cache_usage(0, 0, 5, True)
    assert usage.cache_status == PromptCacheStatus.MISS


def test_set_prompt_cache_usage_unreported() -> None:
    usage = TokenUsage()
    usage.set_prompt_cache_usage(0, 0, 0, False)
    assert usage.cache_status == PromptCacheStatus.UNREPORTED


def test_set_prompt_cache_usage_clamps_negatives() -> None:
    usage = TokenUsage()
    usage.set_prompt_cache_usage(-3, -1, -7, True)
    assert usage.cached_tokens == 0
    assert usage.cache_write_tokens == 0
    assert usage.cache_miss_tokens == 0


def test_mark_prompt_cache_unsupported() -> None:
    usage = TokenUsage()
    usage.mark_prompt_cache_unsupported()
    assert usage.cache_status == PromptCacheStatus.UNSUPPORTED
    assert usage.cache_reported is False


def test_llm_tool_call_excludes_observability_state() -> None:
    tool_call = LLMToolCall(
        id="call_1",
        type="function",
        function=FunctionCall(name="lookup", arguments="{}"),
        model_arguments='{"x":1}',
        argument_resolution="resolved",
        unresolved_handles=["h1"],
    )
    dumped = tool_call.model_dump()
    assert set(dumped) == {"id", "type", "function", "provider_metadata"}
    payload = json.loads(tool_call.model_dump_json())
    assert "model_arguments" not in payload
    assert "unresolved_handles" not in payload


def test_chat_response_excludes_stream_state() -> None:
    response = ChatResponse(
        content="hello",
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        answer_streamed=True,
        answer_event_id="evt-1",
    )
    dumped = response.model_dump()
    assert set(dumped) == {
        "content",
        "reasoning_content",
        "tool_calls",
        "finish_reason",
        "usage",
    }
    payload = json.loads(response.model_dump_json())
    assert "answer_streamed" not in payload
    assert "answer_event_id" not in payload


def test_response_type_values() -> None:
    assert ResponseType.ANSWER.value == "answer"
    assert ResponseType.THINKING.value == "thinking"
    assert ResponseType.TOOL_CALL.value == "tool_call"
    assert ResponseType.ERROR.value == "error"
    assert ResponseType.COMPLETE.value == "complete"


def test_stream_response_serialization_names() -> None:
    response = StreamResponse(
        response_type=ResponseType.ANSWER,
        content="hello",
        done=False,
        session_id="s1",
    )
    dumped = response.model_dump()
    assert set(dumped) == {
        "id",
        "response_type",
        "content",
        "done",
        "knowledge_references",
        "session_id",
        "assistant_message_id",
        "tool_calls",
        "data",
        "usage",
        "finish_reason",
    }
    assert dumped["response_type"] == "answer"
