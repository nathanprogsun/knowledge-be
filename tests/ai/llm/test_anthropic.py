"""Tests for the Anthropic Messages API chat client.

All HTTP is faked through ``httpx.MockTransport`` and the fake endpoint is
whitelisted via ``SSRF_WHITELIST`` so no real DNS or network is involved.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest

from src.ai.llm.anthropic import (
    ANTHROPIC_VERSION,
    AnthropicChat,
    is_anthropic_messages_endpoint,
    is_anthropic_versioned_base_url,
    merge_anthropic_cache_counters,
    merge_anthropic_usage,
    new_anthropic_chat,
    parse_anthropic_sse,
    process_anthropic_stream,
    text_from_multi_content,
    tool_choice_to_anthropic,
    tool_to_anthropic,
    tool_use_from_block,
)
from src.ai.llm.transport import SSRFValidationError
from src.ai.llm.types import (
    ChatConfig,
    ChatOptions,
    FunctionCall,
    FunctionDef,
    ImageURL,
    Message,
    MessageContentPart,
    PromptCacheStatus,
    ResponseType,
    Tool,
    ToolCall,
)
from src.ai.provider.providers.anthropic import ANTHROPIC_BASE_URL
from src.common.exception import ValidationError, AIProviderError

BASE_URL = "http://llm.test/v1"


@pytest.fixture(autouse=True)
def _ssrf_whitelist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SSRF_WHITELIST", "llm.test")


def _config(**overrides: object) -> ChatConfig:
    values: dict[str, object] = {
        "source": "remote",
        "base_url": BASE_URL,
        "model_name": "claude-3-5-sonnet",
        "api_key": "sk-ant-test",
    }
    values.update(overrides)
    return ChatConfig.model_validate(values)


async def _close(chat: AnthropicChat) -> None:
    await chat.aclose()


async def _lines(events: list[str]) -> AsyncIterator[str]:
    for line in events:
        yield line


# ── Constructor ──────────────────────────────────────────────────────


def test_constructor_requires_api_key() -> None:
    with pytest.raises(ValidationError, match="API key"):
        AnthropicChat(_config(api_key=""))


def test_constructor_rejects_restricted_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SSRF_WHITELIST", raising=False)
    with pytest.raises(SSRFValidationError, match="restricted"):
        AnthropicChat(_config(base_url="http://localhost:11434"))


def test_constructor_defaults_to_anthropic_base_url() -> None:
    chat = AnthropicChat(_config(base_url=""))
    assert chat._base_url == ANTHROPIC_BASE_URL


def test_constructor_getters() -> None:
    chat = AnthropicChat(_config(model_id="m-1", custom_headers={"X-Trace": "t"}))
    assert chat.get_model_name() == "claude-3-5-sonnet"
    assert chat.get_model_id() == "m-1"
    assert chat._base_url == BASE_URL.rstrip("/")
    assert chat._api_key == "sk-ant-test"


def test_new_anthropic_chat_factory() -> None:
    chat = new_anthropic_chat(_config())
    assert isinstance(chat, AnthropicChat)
    assert chat.get_model_name() == "claude-3-5-sonnet"


# ── Endpoint derivation ──────────────────────────────────────────────


def test_is_anthropic_messages_endpoint() -> None:
    assert is_anthropic_messages_endpoint("https://api.anthropic.com/v1/messages")
    assert is_anthropic_messages_endpoint("http://llm.test/messages")
    assert not is_anthropic_messages_endpoint("https://api.anthropic.com/v1")
    assert not is_anthropic_messages_endpoint("https://api.anthropic.com")
    assert not is_anthropic_messages_endpoint("https://api.anthropic.com/v1/messages/extra")


def test_is_anthropic_versioned_base_url() -> None:
    assert is_anthropic_versioned_base_url("https://api.anthropic.com/v1")
    assert is_anthropic_versioned_base_url("https://api.anthropic.com/v1beta")
    assert not is_anthropic_versioned_base_url("https://api.anthropic.com")
    assert not is_anthropic_versioned_base_url("https://api.anthropic.com/v1/messages")


def test_endpoint_resolution() -> None:
    assert AnthropicChat(_config(base_url="https://api.anthropic.com/v1")).endpoint() == (
        "https://api.anthropic.com/v1/messages"
    )
    assert AnthropicChat(_config(base_url="http://llm.test/v1beta")).endpoint() == (
        "http://llm.test/v1beta/messages"
    )
    assert AnthropicChat(_config(base_url="http://llm.test/messages")).endpoint() == (
        "http://llm.test/messages"
    )
    assert AnthropicChat(_config(base_url="http://llm.test")).endpoint() == (
        "http://llm.test/v1/messages"
    )


# ── Request building ─────────────────────────────────────────────────


def test_build_request_basics() -> None:
    chat = AnthropicChat(_config())
    req = chat.build_request(
        [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="hello"),
        ],
        None,
    )
    assert req["model"] == "claude-3-5-sonnet"
    assert req["max_tokens"] == 1024
    assert req["system"] == "You are helpful"
    assert req["messages"] == [{"role": "user", "content": "hello"}]


def test_build_request_system_join() -> None:
    chat = AnthropicChat(_config())
    req = chat.build_request(
        [
            Message(role="system", content="first"),
            Message(role="user", content="hi"),
            Message(role="system", content="second"),
        ],
        None,
    )
    assert req["system"] == "first\n\nsecond"


def test_build_request_opts() -> None:
    chat = AnthropicChat(_config())
    req = chat.build_request(
        [Message(role="user", content="hi")],
        ChatOptions(max_tokens=256, temperature=0.5, top_p=0.9),
    )
    assert req["max_tokens"] == 256
    assert req["temperature"] == 0.5
    assert req["top_p"] == 0.9


def test_build_request_max_completion_tokens_fallback() -> None:
    chat = AnthropicChat(_config())
    req = chat.build_request(
        [Message(role="user", content="hi")],
        ChatOptions(max_tokens=0, max_completion_tokens=512),
    )
    assert req["max_tokens"] == 512


def test_build_request_zero_sampling_opts_omitted() -> None:
    chat = AnthropicChat(_config())
    req = chat.build_request(
        [Message(role="user", content="hi")],
        ChatOptions(temperature=0.0, top_p=0.0, max_tokens=0),
    )
    assert "temperature" not in req
    assert "top_p" not in req
    assert req["max_tokens"] == 1024


def test_build_request_tools() -> None:
    chat = AnthropicChat(_config())
    req = chat.build_request(
        [Message(role="user", content="hi")],
        ChatOptions(
            tools=[
                Tool(
                    type="function",
                    function=FunctionDef(
                        name="lookup",
                        description="looks up",
                        parameters={"type": "object"},
                    ),
                )
            ],
            tool_choice="auto",
        ),
    )
    assert req["tools"] == [
        {"name": "lookup", "description": "looks up", "input_schema": {"type": "object"}}
    ]
    assert req["tool_choice"] == {"type": "auto"}


def test_tool_to_anthropic_without_schema() -> None:
    assert tool_to_anthropic(
        Tool(type="function", function=FunctionDef(name="f", description="d"))
    ) == {"name": "f", "description": "d"}


def test_tool_choice_mapping() -> None:
    assert tool_choice_to_anthropic("auto") == {"type": "auto"}
    assert tool_choice_to_anthropic("required") == {"type": "any"}
    assert tool_choice_to_anthropic("none") == {"type": "none"}
    assert tool_choice_to_anthropic("specific_tool") == {
        "type": "tool",
        "name": "specific_tool",
    }


def test_build_request_tool_calls_round_trip() -> None:
    chat = AnthropicChat(_config())
    assistant = Message(
        role="assistant",
        tool_calls=[
            ToolCall(
                id="toolu_1",
                function=FunctionCall(name="lookup", arguments='{"q":"x"}'),
            )
        ],
    )
    tool_result = Message(role="tool", tool_call_id="toolu_1", content="42")
    req = chat.build_request([Message(role="user", content="hi"), assistant, tool_result], None)
    messages = req["messages"]
    assert isinstance(messages, list)
    assert messages[1] == {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}}],
    }
    assert messages[2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "42"}],
    }


def test_build_request_assistant_text_and_tool_calls() -> None:
    chat = AnthropicChat(_config())
    assistant = Message(
        role="assistant",
        content="checking",
        tool_calls=[ToolCall(id="toolu_2", function=FunctionCall(name="f", arguments="{}"))],
    )
    req = chat.build_request([assistant], None)
    assert req["messages"] == [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "toolu_2", "name": "f", "input": {}},
            ],
        }
    ]


def test_build_request_skips_empty_content() -> None:
    chat = AnthropicChat(_config())
    req = chat.build_request([Message(role="user", content="  ")], None)
    assert req["messages"] == []


def test_text_from_multi_content() -> None:
    parts = [
        MessageContentPart(type="text", text="hello"),
        MessageContentPart(type="text", text="  world  "),
        MessageContentPart(
            type="image_url",
            image_url=ImageURL(url="http://img", detail="auto"),
        ),
    ]
    assert text_from_multi_content(parts) == "hello\nworld"
    assert text_from_multi_content([]) == ""


# ── Response parsing ─────────────────────────────────────────────────


def test_parse_response_text_and_tool_use() -> None:
    chat = AnthropicChat(_config())
    result = chat.parse_response(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "lookup",
                    "input": {"q": "x"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_read_input_tokens": 10,
                "cache_creation_input_tokens": 5,
            },
        }
    )
    assert result.content == "Let me check."
    assert result.finish_reason == "tool_use"
    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.id == "toolu_1"
    assert tool_call.function.name == "lookup"
    assert json.loads(tool_call.function.arguments) == {"q": "x"}
    assert tool_call.provider_metadata == {"type": "tool_use"}
    assert result.usage.prompt_tokens == 115
    assert result.usage.completion_tokens == 20
    assert result.usage.total_tokens == 135
    assert result.usage.cache_read_tokens == 10
    assert result.usage.cache_write_tokens == 5
    assert result.usage.cache_status == PromptCacheStatus.HIT


def test_parse_response_without_usage() -> None:
    chat = AnthropicChat(_config())
    result = chat.parse_response(
        {"content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn"}
    )
    assert result.content == "hi"
    assert result.finish_reason == "end_turn"
    assert result.usage.cache_status == PromptCacheStatus.UNREPORTED
    assert result.tool_calls == []


def test_tool_use_from_block() -> None:
    tool_call = tool_use_from_block(
        {"type": "tool_use", "id": "t1", "name": "lookup", "input": {"q": 1}}
    )
    assert tool_call is not None
    assert tool_call.id == "t1"
    assert tool_call.function.name == "lookup"
    assert json.loads(tool_call.function.arguments) == {"q": 1}
    assert tool_use_from_block({"type": "tool_use", "name": "x"}) is None
    assert tool_use_from_block({"type": "tool_use", "id": "t2"}) is None


# ── Usage helpers ────────────────────────────────────────────────────


def test_merge_anthropic_cache_counters() -> None:
    assert merge_anthropic_cache_counters(0, 0, False, 4, None) == (4, 0, True)
    assert merge_anthropic_cache_counters(2, 1, True, 4, 3) == (4, 3, True)
    assert merge_anthropic_cache_counters(0, 0, False, None, None) == (0, 0, False)


def test_merge_anthropic_usage() -> None:
    usage = merge_anthropic_usage(None, 10, 5, 4, None)
    assert usage.prompt_tokens == 14
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 19
    assert usage.cache_read_tokens == 4
    assert usage.cache_status == PromptCacheStatus.HIT

    merged = merge_anthropic_usage(usage, 0, 8, None, None)
    assert merged.completion_tokens == 8
    assert merged.prompt_tokens == 14
    assert merged.total_tokens == 22


# ── Non-streaming path ───────────────────────────────────────────────


async def test_chat_parses_messages_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/messages"
        assert request.headers["x-api-key"] == "sk-ant-test"
        assert request.headers["anthropic-version"] == ANTHROPIC_VERSION
        payload = json.loads(request.content)
        assert payload["model"] == "claude-3-5-sonnet"
        assert payload["max_tokens"] == 1024
        assert payload["messages"] == [{"role": "user", "content": "hi"}]
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 7, "output_tokens": 3},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chat = AnthropicChat(_config(), client=client)
    result = await chat.chat([Message(role="user", content="hi")])
    assert result.content == "hello"
    assert result.finish_reason == "end_turn"
    assert result.usage.prompt_tokens == 7
    assert result.usage.completion_tokens == 3
    await _close(chat)


async def test_chat_applies_custom_headers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Trace"] == "t1"
        # Reserved auth headers may not be overridden.
        assert request.headers["x-api-key"] == "sk-ant-test"
        return httpx.Response(200, json={"content": [], "usage": {}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chat = AnthropicChat(
        _config(custom_headers={"X-Trace": "t1", "x-api-key": "sk-evil"}),
        client=client,
    )
    await chat.chat([Message(role="user", content="hi")])
    await _close(chat)


async def test_chat_error_status_with_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "type": "invalid_request_error",
                    "message": "bad request",
                }
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chat = AnthropicChat(_config(), client=client)
    with pytest.raises(AIProviderError, match="status 400: bad request"):
        await chat.chat([Message(role="user", content="hi")])
    await _close(chat)


async def test_chat_sse_fallback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=(
                'data: {"type":"message_start","message":'
                '{"usage":{"input_tokens":5,"output_tokens":0}}}\n\n'
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"text_delta","text":"hello"}}\n\n'
                'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
                '"usage":{"output_tokens":3}}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chat = AnthropicChat(_config(), client=client)
    result = await chat.chat([Message(role="user", content="hi")])
    assert result.content == "hello"
    assert result.finish_reason == "end_turn"
    assert result.usage.prompt_tokens == 5
    assert result.usage.completion_tokens == 3
    await _close(chat)


# ── parse_anthropic_sse ──────────────────────────────────────────────


async def test_parse_anthropic_sse_aggregates() -> None:
    lines = [
        'data: {"type":"message_start","message":{"usage":'
        '{"input_tokens":10,"output_tokens":0,"cache_read_input_tokens":4}}}',
        'data: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"text_delta","text":"hello "}}',
        'data: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"text_delta","text":"world"}}',
        'data: {"type":"content_block_start","index":1,"content_block":'
        '{"type":"tool_use","id":"toolu_2","name":"lookup"}}',
        'data: {"type":"content_block_delta","index":1,"delta":'
        '{"type":"input_json_delta","partial_json":"{\\"q\\": \\"x\\"}"}}',
        'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},'
        '"usage":{"output_tokens":15}}',
    ]
    result = await parse_anthropic_sse(_lines(lines))
    assert result.content == "hello world"
    assert result.finish_reason == "tool_use"
    assert result.usage.prompt_tokens == 14
    assert result.usage.completion_tokens == 15
    assert result.usage.cache_read_tokens == 4
    assert result.usage.cache_status == PromptCacheStatus.HIT
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "toolu_2"
    assert result.tool_calls[0].function.name == "lookup"
    assert result.tool_calls[0].function.arguments == '{"q": "x"}'


async def test_parse_anthropic_sse_done_sentinel() -> None:
    lines = [
        'data: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"text_delta","text":"partial"}}',
        "data: [DONE]",
        'data: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"text_delta","text":"ignored"}}',
    ]
    result = await parse_anthropic_sse(_lines(lines))
    assert result.content == "partial"


async def test_parse_anthropic_sse_stream_error() -> None:
    lines = ['data: {"type":"error","error":{"type":"overloaded_error","message":"overloaded"}}']
    with pytest.raises(AIProviderError, match="overloaded"):
        await parse_anthropic_sse(_lines(lines))


# ── process_anthropic_stream ─────────────────────────────────────────


async def test_process_anthropic_stream_text() -> None:
    lines = [
        'data: {"type":"message_start","message":{"usage":{"input_tokens":5,"output_tokens":0}}}',
        'data: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"text_delta","text":"Hello"}}',
        'data: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"text_delta","text":" world"}}',
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
        '"usage":{"output_tokens":8}}',
        "data: [DONE]",
    ]
    events = [event async for event in process_anthropic_stream("claude-3-5-sonnet", _lines(lines))]
    assert events[0].response_type == ResponseType.ANSWER
    assert events[0].content == "Hello"
    assert events[0].done is False
    assert events[1].content == " world"
    final = events[-1]
    assert final.done is True
    assert final.finish_reason == "end_turn"
    assert final.usage is not None
    assert final.usage.prompt_tokens == 5
    assert final.usage.completion_tokens == 8


async def test_process_anthropic_stream_tool_call() -> None:
    lines = [
        'data: {"type":"content_block_start","index":0,"content_block":'
        '{"type":"tool_use","id":"toolu_1","name":"lookup"}}',
        'data: {"type":"content_block_delta","index":0,"delta":'
        '{"type":"input_json_delta","partial_json":"{\\"location\\": \\"SF\\"}"}}',
        'data: {"type":"content_block_stop","index":0}',
        'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},'
        '"usage":{"output_tokens":5}}',
        "data: [DONE]",
    ]
    events = [event async for event in process_anthropic_stream("claude-3-5-sonnet", _lines(lines))]
    assert events[0].response_type == ResponseType.TOOL_CALL
    assert events[0].data == {"tool_name": "lookup", "tool_call_id": "toolu_1"}
    final = events[-1]
    assert final.done is True
    assert len(final.tool_calls) == 1
    tool_call = final.tool_calls[0]
    assert tool_call.id == "toolu_1"
    assert tool_call.function.name == "lookup"
    assert tool_call.function.arguments == '{"location": "SF"}'


async def test_process_anthropic_stream_thinking_then_answer() -> None:
    lines = [
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}',
        'data: {"type":"content_block_delta","index":0,"delta":'
        '{"type":"thinking_delta","thinking":"reasoning"}}',
        'data: {"type":"content_block_stop","index":0}',
        'data: {"type":"content_block_delta","index":1,"delta":'
        '{"type":"text_delta","text":"answer"}}',
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
        '"usage":{"output_tokens":3}}',
        "data: [DONE]",
    ]
    events = [event async for event in process_anthropic_stream("claude-3-5-sonnet", _lines(lines))]
    types = [event.response_type for event in events]
    assert types == [
        ResponseType.THINKING,
        ResponseType.THINKING,
        ResponseType.ANSWER,
        ResponseType.ANSWER,
    ]
    assert events[0].content == "reasoning"
    assert events[1].done is True
    assert events[2].content == "answer"


async def test_process_anthropic_stream_error_event() -> None:
    lines = ['data: {"type":"error","error":{"type":"overloaded_error","message":"overloaded"}}']
    events = [event async for event in process_anthropic_stream("claude-3-5-sonnet", _lines(lines))]
    assert len(events) == 1
    assert events[0].response_type == ResponseType.ERROR
    assert events[0].content == "overloaded"
    assert events[0].done is True


# ── Streaming path (end to end) ──────────────────────────────────────


async def test_chat_stream_yields_events() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept"] == "text/event-stream"
        payload = json.loads(request.content)
        assert payload["stream"] is True
        return httpx.Response(
            200,
            text=(
                'data: {"type":"message_start","message":{"usage":'
                '{"input_tokens":3,"output_tokens":0}}}\n\n'
                'data: {"type":"content_block_delta","index":0,"delta":'
                '{"type":"text_delta","text":"hello"}}\n\n'
                'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
                '"usage":{"output_tokens":2}}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chat = AnthropicChat(_config(), client=client)
    events = [event async for event in chat.chat_stream([Message(role="user", content="hi")])]
    assert events[0].content == "hello"
    final = events[-1]
    assert final.done is True
    assert final.finish_reason == "end_turn"
    assert final.usage is not None
    assert final.usage.prompt_tokens == 3
    assert final.usage.completion_tokens == 2
    await _close(chat)


async def test_chat_stream_error_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad request")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chat = AnthropicChat(_config(), client=client)
    with pytest.raises(AIProviderError, match="status 400"):
        async for _event in chat.chat_stream([Message(role="user", content="hi")]):
            pass
    await _close(chat)
