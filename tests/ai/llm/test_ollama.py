"""Tests for the local Ollama chat client.

``OllamaChat`` builds ``/api/chat`` request bodies, parses non-streaming and
SSE-streamed responses, and converts tools both directions. All HTTP is faked
through ``httpx.MockTransport`` — no network, no Ollama install.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from src.ai.llm.ollama import (
    OllamaChat,
    new_ollama_chat,
    tooli2s,
    tools2i,
)
from src.ai.llm.types import (
    ChatConfig,
    ChatOptions,
    FunctionCall,
    FunctionDef,
    Message,
    PromptCacheStatus,
    ResponseType,
    Tool,
    ToolCall,
)
from src.ai.utils.ollama_service import OllamaService
from src.common.exception import AIProviderError

_BASE_URL = "http://ollama.test"
_MODEL = "qwen2"


def _service(handler: Callable[[httpx.Request], httpx.Response]) -> OllamaService:
    return OllamaService(base_url=_BASE_URL, transport=httpx.MockTransport(handler))


def _chat(service: OllamaService) -> OllamaChat:
    return new_ollama_chat(
        ChatConfig(source="local", model_name=_MODEL, model_id="m1"),
        service,
    )


def _router(
    chat_handler: Callable[[httpx.Request], httpx.Response],
) -> Callable[[httpx.Request], httpx.Response]:
    """Route the availability probe, model list, and chat call."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD" and request.url.path == "/":
            return httpx.Response(200)
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": f"{_MODEL}:latest"}]},
            )
        if request.url.path == "/api/chat":
            return chat_handler(request)
        return httpx.Response(404)

    return handler


# ── factory ──────────────────────────────────────────────────────────


def test_new_ollama_chat_requires_service() -> None:
    with pytest.raises(AIProviderError, match="Ollama service is required"):
        new_ollama_chat(
            ChatConfig(source="local", model_name=_MODEL, model_id="m1"),
            None,
        )


def test_new_ollama_chat_returns_client() -> None:
    service = _service(lambda request: httpx.Response(404))
    chat = _chat(service)
    assert isinstance(chat, OllamaChat)
    assert chat.get_model_name() == _MODEL
    assert chat.get_model_id() == "m1"


# ── request building ─────────────────────────────────────────────────


def test_build_chat_request_defaults() -> None:
    chat = _chat(_service(lambda request: httpx.Response(404)))
    request = chat.build_chat_request(
        [Message(role="user", content="hi")],
        None,
        False,
    )
    assert request == {
        "model": _MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "options": {},
    }


def test_build_chat_request_with_options() -> None:
    chat = _chat(_service(lambda request: httpx.Response(404)))
    request = chat.build_chat_request(
        [Message(role="user", content="hi")],
        ChatOptions(
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
            thinking=True,
            format={"type": "json"},
        ),
        True,
    )
    assert request == {
        "model": _MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 100},
        "think": True,
        "format": {"type": "json"},
    }


def test_build_chat_request_with_tools() -> None:
    chat = _chat(_service(lambda request: httpx.Response(404)))
    tool = Tool(
        type="function",
        function=FunctionDef(
            name="get_weather",
            description="Get the weather for a city",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        ),
    )
    request = chat.build_chat_request(
        [Message(role="user", content="weather?")],
        ChatOptions(tools=[tool]),
        False,
    )
    assert request["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]


# ── message conversion ───────────────────────────────────────────────


def test_convert_messages_tool_role_sets_tool_name() -> None:
    chat = _chat(_service(lambda request: httpx.Response(404)))
    converted = chat.convert_messages([Message(role="tool", content="42", name="get_weather")])
    assert converted == [{"role": "tool", "content": "42", "tool_name": "get_weather"}]


def test_convert_messages_encodes_user_images() -> None:
    chat = _chat(_service(lambda request: httpx.Response(404)))
    converted = chat.convert_messages(
        [
            Message(
                role="user",
                content="what is this?",
                images=["data:image/png;base64,iVBORw0KGgo="],
            )
        ]
    )
    assert converted == [
        {
            "role": "user",
            "content": "what is this?",
            "images": ["iVBORw0KGgo="],
        }
    ]


def test_convert_messages_round_trips_assistant_tool_calls() -> None:
    chat = _chat(_service(lambda request: httpx.Response(404)))
    converted = chat.convert_messages(
        [
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        id="0",
                        type="function",
                        function=FunctionCall(
                            name="get_weather",
                            arguments='{"city": "Beijing"}',
                        ),
                    )
                ],
            )
        ]
    )
    assert converted == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "index": 0,
                        "name": "get_weather",
                        "arguments": {"city": "Beijing"},
                    }
                }
            ],
        }
    ]


# ── tool conversion ──────────────────────────────────────────────────


def test_tool_call_to_projects_index_into_id() -> None:
    chat = _chat(_service(lambda request: httpx.Response(404)))
    tool_calls = chat.tool_call_to(
        [
            {
                "function": {
                    "index": 2,
                    "name": "get_weather",
                    "arguments": {"city": "Beijing"},
                }
            }
        ]
    )
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "2"
    assert tool_calls[0].type == "function"
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[0].function.arguments == '{"city":"Beijing"}'


def test_tool_call_to_handles_empty_and_missing_arguments() -> None:
    chat = _chat(_service(lambda request: httpx.Response(404)))
    tool_calls = chat.tool_call_to(
        [
            {"function": {"index": 1, "name": "f", "arguments": {}}},
            {"function": {"name": "g"}},
            "not-a-tool",
        ]
    )
    assert len(tool_calls) == 2
    assert tool_calls[0].id == "1"
    assert tool_calls[0].function.arguments == "{}"
    assert tool_calls[1].id == "0"
    assert tool_calls[1].function.arguments == "{}"


def test_tools2i_tooli2s_round_trip() -> None:
    assert tools2i("3") == 3
    assert tools2i("abc") == 0
    assert tools2i("") == 0
    assert tooli2s(3) == "3"


# ── non-streaming chat ───────────────────────────────────────────────


async def test_chat_sends_request_and_parses_response() -> None:
    sent: dict[str, object] = {}

    def chat_handler(request: httpx.Request) -> httpx.Response:
        sent["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": _MODEL,
                "message": {"role": "assistant", "content": "hello"},
                "done": True,
                "prompt_eval_count": 12,
                "eval_count": 20,
            },
        )

    chat = _chat(_service(_router(chat_handler)))
    result = await chat.chat([Message(role="user", content="hi")])
    assert result.content == "hello"
    assert result.tool_calls == []
    assert result.usage.prompt_tokens == 12
    assert result.usage.completion_tokens == 8
    assert result.usage.total_tokens == 20
    assert result.usage.cache_status == PromptCacheStatus.UNSUPPORTED
    body = sent["body"]
    assert isinstance(body, dict)
    assert body["model"] == _MODEL
    assert body["stream"] is False
    assert body["messages"] == [{"role": "user", "content": "hi"}]


async def test_chat_uses_thinking_as_content_fallback() -> None:
    def chat_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "thinking": "reasoned text",
                },
                "done": True,
            },
        )

    chat = _chat(_service(_router(chat_handler)))
    result = await chat.chat([Message(role="user", content="hi")])
    assert result.content == "reasoned text"


async def test_chat_parses_tool_calls_and_zero_usage() -> None:
    def chat_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "index": 0,
                                "name": "get_weather",
                                "arguments": {"city": "Beijing"},
                            }
                        }
                    ],
                },
                "done": True,
            },
        )

    chat = _chat(_service(_router(chat_handler)))
    result = await chat.chat([Message(role="user", content="weather?")])
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "0"
    assert result.tool_calls[0].function.arguments == '{"city":"Beijing"}'
    assert result.usage.prompt_tokens == 0
    assert result.usage.completion_tokens == 0
    assert result.usage.total_tokens == 0


async def test_chat_wraps_service_error() -> None:
    def chat_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    chat = _chat(_service(_router(chat_handler)))
    with pytest.raises(AIProviderError, match="Ollama chat request failed"):
        await chat.chat([Message(role="user", content="hi")])


# ── streaming chat ───────────────────────────────────────────────────


def _sse_response(frames: list[dict[str, object]]) -> httpx.Response:
    lines = ["data: " + json.dumps(frame) for frame in frames]
    lines.append("data: [DONE]")
    return httpx.Response(200, text="\n".join(lines) + "\n")


async def test_chat_stream_yields_answer_chunks_and_done() -> None:
    def chat_handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return _sse_response(
            [
                {"message": {"role": "assistant", "content": "Hel"}, "done": False},
                {"message": {"role": "assistant", "content": "lo"}, "done": False},
                {
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "prompt_eval_count": 5,
                    "eval_count": 9,
                },
            ]
        )

    chat = _chat(_service(_router(chat_handler)))
    events = [event async for event in chat.chat_stream([Message(role="user", content="hi")])]
    assert [event.response_type for event in events] == [
        ResponseType.ANSWER,
        ResponseType.ANSWER,
        ResponseType.ANSWER,
    ]
    assert events[0].content == "Hel"
    assert events[1].content == "lo"
    assert events[0].done is False
    assert events[2].done is True
    assert events[2].usage is not None
    assert events[2].usage.prompt_tokens == 5
    assert events[2].usage.completion_tokens == 9
    assert events[2].usage.total_tokens == 14
    assert events[2].usage.cache_status == PromptCacheStatus.UNSUPPORTED


async def test_chat_stream_forwards_thinking_then_answer() -> None:
    def chat_handler(request: httpx.Request) -> httpx.Response:
        return _sse_response(
            [
                {"message": {"role": "assistant", "thinking": "reasoning..."}, "done": False},
                {"message": {"role": "assistant", "content": "answer"}, "done": False},
                {"message": {"role": "assistant", "content": ""}, "done": True},
            ]
        )

    chat = _chat(_service(_router(chat_handler)))
    events = [event async for event in chat.chat_stream([Message(role="user", content="hi")])]
    assert [event.response_type for event in events] == [
        ResponseType.THINKING,
        ResponseType.THINKING,
        ResponseType.ANSWER,
        ResponseType.ANSWER,
    ]
    assert events[0].content == "reasoning..."
    assert events[1].done is True
    assert events[2].content == "answer"
    assert events[3].done is True
    assert events[3].usage is None


async def test_chat_stream_yields_tool_call_and_thinking_tool() -> None:
    def chat_handler(request: httpx.Request) -> httpx.Response:
        return _sse_response(
            [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "index": 0,
                                    "name": "thinking",
                                    "arguments": {"thought": "let me think"},
                                }
                            },
                            {
                                "function": {
                                    "index": 1,
                                    "name": "get_weather",
                                    "arguments": {"city": "Beijing"},
                                }
                            },
                        ],
                    },
                    "done": False,
                },
                {"message": {"role": "assistant", "content": "done"}, "done": False},
                {"message": {"role": "assistant", "content": ""}, "done": True},
            ]
        )

    chat = _chat(_service(_router(chat_handler)))
    events = [event async for event in chat.chat_stream([Message(role="user", content="hi")])]
    assert [event.response_type for event in events] == [
        ResponseType.TOOL_CALL,
        ResponseType.THINKING,
        ResponseType.ANSWER,
        ResponseType.ANSWER,
    ]
    tool_call_event = events[0]
    assert len(tool_call_event.tool_calls) == 2
    assert tool_call_event.tool_calls[0].id == "0"
    assert tool_call_event.tool_calls[1].function.arguments == '{"city":"Beijing"}'
    thinking_event = events[1]
    assert thinking_event.content == "let me think"
    assert thinking_event.data == {"source": "thinking_tool", "tool_call_id": "0"}
    assert events[2].content == "done"


async def test_chat_stream_emits_error_event_on_service_failure() -> None:
    def chat_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    chat = _chat(_service(_router(chat_handler)))
    events = [event async for event in chat.chat_stream([Message(role="user", content="hi")])]
    assert len(events) == 1
    assert events[0].response_type == ResponseType.ERROR
    assert events[0].done is True
    assert "failed to stream chat response" in events[0].content
