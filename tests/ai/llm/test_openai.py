"""Tests for the OpenAI-compatible request builder and stream parser.

The request builder (``openai``) and the completion / stream parsers
(``openai_stream``) hold the OpenAI wire-format logic that ``RemoteAPIChat``
delegates to. These tests exercise the module functions directly and the
delegating ``RemoteAPIChat`` methods through faked HTTP.

All HTTP is faked through ``httpx.MockTransport`` and the fake endpoint is
whitelisted via ``SSRF_WHITELIST`` so no real DNS or network is involved.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import cast

import httpx
import pytest

from src.ai.llm import openai, openai_stream
from src.ai.llm.openai_stream import StreamChunk, StreamState
from src.ai.llm.providers import GeminiProvider
from src.ai.llm.remote_api import RemoteAPIChat
from src.ai.llm.types import (
    ChatConfig,
    ChatOptions,
    ChatResponse,
    FunctionCall,
    FunctionDef,
    ImageURL,
    LLMToolCall,
    Message,
    MessageContentPart,
    ResponseType,
    StreamResponse,
    Tool,
    ToolCall,
)
from src.common.exception import AIProviderError
from src.common.json import JsonObject, JsonValue

BASE_URL = "http://llm.test/v1"


@pytest.fixture(autouse=True)
def _ssrf_whitelist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SSRF_WHITELIST", "llm.test")


def _config(**overrides: object) -> ChatConfig:
    values: dict[str, object] = {
        "source": "remote",
        "base_url": BASE_URL,
        "model_name": "deepseek-chat",
        "provider": "deepseek",
    }
    values.update(overrides)
    return ChatConfig.model_validate(values)


async def _close(chat: RemoteAPIChat) -> None:
    await chat.aclose()


# ── message conversion ───────────────────────────────────────────────


def test_convert_messages_plain() -> None:
    converted = openai.convert_messages(
        [Message(role="user", content="hello"), Message(role="assistant", content="hi")]
    )
    assert converted == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_convert_messages_multi_content() -> None:
    message = Message(
        role="user",
        multi_content=[
            MessageContentPart(type="text", text="look"),
            MessageContentPart(
                type="image_url",
                image_url=ImageURL(url="http://img", detail="auto"),
            ),
        ],
    )
    converted = openai.convert_messages([message])
    assert converted[0]["multi_content"] == [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "http://img", "detail": "auto"}},
    ]


def test_convert_messages_images() -> None:
    message = Message(role="user", content="describe", images=["http://img/x.png"])
    converted = openai.convert_messages([message])
    assert converted[0]["multi_content"] == [
        {"type": "image_url", "image_url": {"url": "http://img/x.png", "detail": "auto"}},
        {"type": "text", "text": "describe"},
    ]


def test_convert_messages_tool_calls_and_tool_role() -> None:
    assistant = Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(
                id="call-1",
                type="function",
                function=FunctionCall(name="lookup", arguments='{"q":"x"}'),
            )
        ],
        reasoning_content="think",
    )
    tool_result = Message(role="tool", tool_call_id="call-1", name="lookup", content="42")
    converted = openai.convert_messages([assistant, tool_result])
    assert converted[0]["tool_calls"] == [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"q":"x"}'},
        }
    ]
    assert converted[0]["reasoning_content"] == "think"
    assert converted[1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "name": "lookup",
        "content": "42",
    }


# ── request building ─────────────────────────────────────────────────


def test_build_chat_completion_request_basics() -> None:
    chat = RemoteAPIChat(_config())
    req = openai.build_chat_completion_request(
        chat, [Message(role="user", content="hi")], None, True
    )
    assert req["model"] == "deepseek-chat"
    assert req["stream"] is True
    assert req["stream_options"] == {"include_usage": True}


def test_build_chat_completion_request_opts() -> None:
    chat = RemoteAPIChat(_config())
    opts = ChatOptions(
        temperature=0.5,
        top_p=0.9,
        max_tokens=64,
        parallel_tool_calls=False,
        tool_choice="auto",
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
    )
    req = openai.build_chat_completion_request(
        chat, [Message(role="user", content="hi")], opts, False
    )
    assert req["temperature"] == 0.5
    assert req["top_p"] == 0.9
    assert req["max_tokens"] == 64
    assert req["parallel_tool_calls"] is False
    assert req["tool_choice"] == "auto"
    assert req["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "looks up",
                "parameters": {"type": "object"},
            },
        }
    ]
    assert "stream_options" not in req


def test_build_chat_completion_request_specific_tool_choice() -> None:
    chat = RemoteAPIChat(_config())
    req = openai.build_chat_completion_request(
        chat,
        [Message(role="user", content="hi")],
        ChatOptions(tool_choice="specific_tool"),
        False,
    )
    assert req["tool_choice"] == {
        "type": "function",
        "function": {"name": "specific_tool"},
    }


def test_build_chat_completion_request_format_appends_schema() -> None:
    chat = RemoteAPIChat(_config())
    req = openai.build_chat_completion_request(
        chat,
        [Message(role="user", content="list")],
        ChatOptions(format={"type": "object"}),
        False,
    )
    assert req["response_format"] == {"type": "json_object"}
    raw_messages = req["messages"]
    assert isinstance(raw_messages, list)
    last = raw_messages[-1]
    assert isinstance(last, dict)
    content = last.get("content")
    assert isinstance(content, str)
    assert 'Use this JSON schema: {"type": "object"}' in content


def test_build_provider_openai_request_passthrough() -> None:
    chat = RemoteAPIChat(_config())
    messages = [Message(role="user", content="hi")]
    wire = openai.convert_messages(messages)
    body: JsonObject = {"model": "deepseek-chat", "messages": cast(JsonValue, wire)}
    out = openai.build_provider_openai_request(chat, body, wire, messages)
    assert out["model"] == "deepseek-chat"
    out_messages = out["messages"]
    assert isinstance(out_messages, list)
    assert out_messages == [{"role": "user", "content": "hi"}]


def test_build_provider_openai_request_injects_metadata() -> None:
    chat = RemoteAPIChat(_config(provider="gemini", base_url="http://llm.test"))
    messages = [
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    type="function",
                    function=FunctionCall(name="lookup", arguments="{}"),
                    provider_metadata={"google": {"signature": "sig-1"}},
                )
            ],
        )
    ]
    wire = openai.convert_messages(messages)
    body: JsonObject = {"model": "gemini-x", "messages": cast(JsonValue, wire)}
    out = openai.build_provider_openai_request(chat, body, wire, messages)
    out_messages = out["messages"]
    assert isinstance(out_messages, list)
    first = out_messages[0]
    assert isinstance(first, dict)
    tool_calls = first.get("tool_calls")
    assert isinstance(tool_calls, list)
    tc = tool_calls[0]
    assert isinstance(tc, dict)
    assert tc["id"] == "call-1"
    assert tc["extra_content"] == {"google": {"signature": "sig-1"}}


def test_shape_provider_request_passthrough_when_not_forced() -> None:
    chat = RemoteAPIChat(_config(provider="openai", model_name="gpt-4o"))
    body: JsonObject = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
    }
    out = openai.shape_provider_request(chat, body, body, [])
    assert out == body


def test_shape_provider_request_forces_raw_path() -> None:
    chat = RemoteAPIChat(_config(provider="gemini", base_url="http://llm.test"))
    messages = [
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    type="function",
                    function=FunctionCall(name="lookup", arguments="{}"),
                    provider_metadata={"google": {"signature": "sig-1"}},
                )
            ],
        )
    ]
    wire = openai.convert_messages(messages)
    body: JsonObject = {"model": "gemini-x", "messages": cast(JsonValue, wire)}
    out = openai.shape_provider_request(chat, body, body, messages)
    out_messages = out["messages"]
    assert isinstance(out_messages, list)
    first = out_messages[0]
    assert isinstance(first, dict)
    tool_calls = first.get("tool_calls")
    assert isinstance(tool_calls, list)
    tc = tool_calls[0]
    assert isinstance(tc, dict)
    assert tc["extra_content"] == {"google": {"signature": "sig-1"}}


# ── completion response parsing ──────────────────────────────────────


def test_parse_completion_response() -> None:
    chat = RemoteAPIChat(_config())
    resp: JsonObject = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "hello",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    result = openai_stream.parse_completion_response(chat, resp)
    assert result.content == "hello"
    assert result.finish_reason == "tool_calls"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call-1"
    assert result.tool_calls[0].function.name == "lookup"
    assert result.tool_calls[0].function.arguments == "{}"


def test_parse_completion_response_no_choices() -> None:
    chat = RemoteAPIChat(_config())
    with pytest.raises(AIProviderError, match="no response"):
        openai_stream.parse_completion_response(chat, {})


def test_parse_completion_response_keeps_plain_content() -> None:
    chat = RemoteAPIChat(_config())
    result = openai_stream.parse_completion_response(
        chat,
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "plain answer"},
                    "finish_reason": "stop",
                }
            ]
        },
    )
    assert result.content == "plain answer"
    assert result.finish_reason == "stop"


def test_apply_completion_tool_call_metadata() -> None:
    chat = RemoteAPIChat(_config(provider="gemini", base_url="http://llm.test"))
    result = ChatResponse(
        tool_calls=[LLMToolCall(id="call-1", function=FunctionCall(name="lookup"))]
    )
    body = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "index": 0,
                                "extra_content": {"google": {"signature": "s"}},
                            }
                        ]
                    }
                }
            ]
        }
    ).encode("utf-8")
    openai_stream.apply_completion_tool_call_metadata(chat, body, result)
    assert result.tool_calls[0].provider_metadata == {"google": {"signature": "s"}}


# ── stream state ─────────────────────────────────────────────────────


def test_stream_state_ordered_tool_calls() -> None:
    state = StreamState()
    assert state.build_ordered_tool_calls() == []
    state.tool_call_map[1] = LLMToolCall(id="b", function=FunctionCall(name="f2"))
    state.tool_call_map[0] = LLMToolCall(id="a", function=FunctionCall(name="f1"))
    assert [call.id for call in state.build_ordered_tool_calls()] == ["a", "b"]


def test_stream_state_set_tool_call_provider_metadata() -> None:
    state = StreamState()
    state.set_tool_call_provider_metadata(0, {"google": {"signature": "s"}})
    calls = state.build_ordered_tool_calls()
    assert len(calls) == 1
    assert calls[0].provider_metadata == {"google": {"signature": "s"}}
    state.set_tool_call_provider_metadata(0, {})
    assert state.build_ordered_tool_calls()[0].provider_metadata == {"google": {"signature": "s"}}


def test_apply_stream_tool_call_metadata() -> None:
    chat = RemoteAPIChat(_config(provider="gemini", base_url="http://llm.test"))
    state = StreamState()
    stream_resp: JsonObject = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-1",
                            "extra_content": {"google": {"signature": "s"}},
                        }
                    ]
                }
            }
        ]
    }
    openai_stream.apply_stream_tool_call_metadata(chat, stream_resp, state)
    calls = state.build_ordered_tool_calls()
    assert len(calls) == 1
    assert calls[0].provider_metadata == {"google": {"signature": "s"}}


# ── stream delta processing ──────────────────────────────────────────


async def _chunks(*payloads: JsonObject) -> AsyncIterator[StreamChunk]:
    for payload in payloads:
        yield StreamChunk(payload=payload, raw=json.dumps(payload).encode("utf-8"))


async def test_process_stream_yields_events() -> None:
    chat = RemoteAPIChat(_config())
    events = [
        event
        async for event in openai_stream.process_stream(
            chat,
            _chunks(
                {"choices": [{"delta": {"content": "Hello"}}]},
                {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]},
            ),
        )
    ]
    assert events[0].response_type == ResponseType.ANSWER
    assert events[0].content == "Hello"
    assert events[0].done is False
    assert events[1].content == " world"
    assert events[1].done is True
    assert events[1].finish_reason == "stop"
    assert events[-1].done is True
    assert events[-1].response_type == ResponseType.ANSWER
    await _close(chat)


async def test_process_stream_captures_usage() -> None:
    chat = RemoteAPIChat(_config())
    events = [
        event
        async for event in openai_stream.process_stream(
            chat,
            _chunks(
                {
                    "choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            ),
        )
    ]
    final = events[-1]
    assert final.usage is not None
    assert final.usage.prompt_tokens == 10
    assert final.usage.completion_tokens == 5
    assert final.usage.total_tokens == 15
    await _close(chat)


async def test_process_stream_thinking_then_answer() -> None:
    chat = RemoteAPIChat(_config())
    events = [
        event
        async for event in openai_stream.process_stream(
            chat,
            _chunks(
                {"choices": [{"delta": {"reasoning_content": "think "}}]},
                {"choices": [{"delta": {"reasoning_content": "more"}}]},
                {"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]},
            ),
        )
    ]
    types = [event.response_type for event in events]
    assert types == [
        ResponseType.THINKING,
        ResponseType.THINKING,
        ResponseType.THINKING,  # thinking-done marker
        ResponseType.ANSWER,
        ResponseType.ANSWER,  # final done
    ]
    assert events[0].content == "think "
    assert events[1].content == "more"
    assert events[2].done is True
    assert events[3].content == "answer"
    await _close(chat)


async def test_process_stream_tool_calls_delta() -> None:
    chat = RemoteAPIChat(_config())
    events = [
        event
        async for event in openai_stream.process_stream(
            chat,
            _chunks(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {"name": "look", "arguments": ""},
                                    }
                                ]
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": '{"q": "x"}'},
                                    }
                                ]
                            }
                        }
                    ]
                },
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            ),
        )
    ]
    tool_calls = events[-1].tool_calls
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call-1"
    assert tool_calls[0].function.name == "look"
    assert tool_calls[0].function.arguments == '{"q": "x"}'
    await _close(chat)


async def test_process_stream_delta_vllm_reasoning_field() -> None:
    state = StreamState()
    events = [
        event
        async for event in openai_stream.process_stream_delta(
            {"delta": {"reasoning": "why"}, "finish_reason": ""}, state
        )
    ]
    assert events == [StreamResponse(response_type=ResponseType.THINKING, content="why")]


async def test_process_tool_calls_delta_thinking_tool() -> None:
    state = StreamState()
    events = [
        event
        async for event in openai_stream.process_tool_calls_delta(
            [
                {
                    "index": 0,
                    "id": "tc-1",
                    "type": "function",
                    "function": {"name": "thinking", "arguments": ""},
                },
                {"index": 0, "function": {"arguments": '{"thought":"hel'}},
                {"index": 0, "function": {"arguments": "lo"}},
                {"index": 0, "function": {"arguments": "\\n\\u4e16"}},
                {"index": 0, "function": {"arguments": '界"}'}},
            ],
            state,
        )
    ]
    thinking = [event for event in events if event.response_type == ResponseType.THINKING]
    assert thinking
    assert "".join(event.content for event in thinking) == "hello\n世界"
    tool_call_events = [event for event in events if event.response_type == ResponseType.TOOL_CALL]
    assert len(tool_call_events) == 1
    assert tool_call_events[0].data == {"tool_name": "thinking", "tool_call_id": "tc-1"}


# ── raw HTTP stream parsing ──────────────────────────────────────────


async def test_process_raw_http_stream_parses_sse() -> None:
    chat = RemoteAPIChat(_config())
    response = httpx.Response(
        200,
        text=(
            'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
            'data: {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]}\n\n'
            "data: [DONE]\n\n"
        ),
        headers={"content-type": "text/event-stream"},
    )
    events = [event async for event in openai_stream.process_raw_http_stream(chat, response)]
    assert [event.content for event in events if event.content] == ["Hello", " world"]
    assert events[-1].done is True
    assert events[-1].response_type == ResponseType.ANSWER
    await _close(chat)


async def test_process_raw_http_stream_skips_bad_json() -> None:
    chat = RemoteAPIChat(_config())
    response = httpx.Response(
        200,
        text=(
            "data: not-json\n\n"
            'data: {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}\n\n'
            "data: [DONE]\n\n"
        ),
        headers={"content-type": "text/event-stream"},
    )
    events = [event async for event in openai_stream.process_raw_http_stream(chat, response)]
    assert [event.content for event in events if event.content] == ["ok"]
    assert events[-1].done is True
    await _close(chat)


# ── RemoteAPIChat delegation ─────────────────────────────────────────


async def test_remote_chat_convert_matches_module_function() -> None:
    chat = RemoteAPIChat(_config())
    messages = [Message(role="user", content="hi")]
    assert chat.convert_messages(messages) == openai.convert_messages(messages)
    assert chat.build_chat_completion_request(messages, None, False) == (
        openai.build_chat_completion_request(chat, messages, None, False)
    )
    await _close(chat)


async def test_remote_chat_process_stream_delegates() -> None:
    chat = RemoteAPIChat(_config())
    events = [
        event
        async for event in chat.process_stream(
            _chunks({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]})
        )
    ]
    assert events[0].content == "hi"
    assert events[0].done is True
    assert events[-1].done is True
    await _close(chat)


async def test_chat_stream_injects_tool_call_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat/completions"
        return httpx.Response(
            200,
            text=(
                'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call-1", '
                '"type": "function", "extra_content": {"google": {"signature": "s"}}, '
                '"function": {"name": "look", "arguments": "{}"}}]}}]}\n\n'
                'data: {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chat = RemoteAPIChat(_config(provider="gemini", base_url="http://llm.test"), client=client)
    events = [event async for event in chat.chat_stream([Message(role="user", content="hi")])]
    tool_calls = events[-1].tool_calls
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call-1"
    assert tool_calls[0].function.name == "look"
    assert tool_calls[0].provider_metadata == {"google": {"signature": "s"}}
    await _close(chat)


# ── thinking stripping ───────────────────────────────────────────────


def test_remove_thinking_content() -> None:
    assert openai_stream.remove_thinking_content("plain answer") == "plain answer"
    # The strip heuristic trims whitespace before matching, so the leading-space
    # tag never matches real output; mirror the reference behavior exactly.
    assert openai_stream.remove_thinking_content(" think step by step response actual") == (
        " think step by step response actual"
    )
    assert openai_stream.remove_thinking_content("") == ""


def test_gemini_provider_forces_raw_http() -> None:
    assert GeminiProvider().force_raw_http() is True
