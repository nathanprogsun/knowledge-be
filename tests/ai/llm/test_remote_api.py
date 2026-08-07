"""Tests for the OpenAI-compatible chat backend and its transport helpers.

All HTTP is faked through ``httpx.MockTransport`` and the fake endpoint is
whitelisted via ``SSRF_WHITELIST`` so no real DNS or network is involved.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest

from src.ai.llm.image_resolve import (
    is_application_stored_image,
    resolve_image_url_for_llm,
    resolve_image_url_for_ollama,
    strip_images_from_messages,
)
from src.ai.llm.json_field_extractor import JSONFieldExtractor
from src.ai.llm.prompt_cache import (
    apply_raw_prompt_cache_usage,
    fingerprint_prompt_prefix,
    prompt_prefix_fingerprint,
    token_usage_from_openai,
)
from src.ai.llm.providers import (
    AzureProvider,
    AzureReasoningProvider,
    DeepseekProvider,
    GeminiProvider,
    GenericProvider,
    LkeapProvider,
    MoonshotProvider,
    NvidiaProvider,
    OpenAIReasoningProvider,
    ProviderAdapter,
    QwenThinkingProvider,
    VolcengineProvider,
    WeKnoraCloudProvider,
    resolve_provider,
)
from src.ai.llm.remote_api import (
    RemoteAPIChat,
    remove_thinking_content,
)
from src.ai.llm.sse_reader import SSEReader
from src.ai.llm.transport import (
    SSRFValidationError,
    apply_custom_headers,
    is_reserved_header,
    validate_url_for_ssrf,
)
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
from src.common.exception import ValidationError, AIProviderError
from src.common.json import JsonObject

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


# ── Constructor ──────────────────────────────────────────────────────


def test_constructor_rejects_restricted_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SSRF_WHITELIST", raising=False)
    with pytest.raises(SSRFValidationError, match="localhost"):
        RemoteAPIChat(_config(base_url="http://localhost:11434"))


def test_constructor_requires_managed_cloud_credentials() -> None:
    with pytest.raises(ValidationError, match="AppID"):
        RemoteAPIChat(_config(provider="weknoracloud", app_id="", app_secret=""))
    with pytest.raises(ValidationError, match="AppSecret"):
        RemoteAPIChat(_config(provider="weknoracloud", app_id="app", app_secret=""))


def test_constructor_remote_model_name_override() -> None:
    chat = RemoteAPIChat(
        _config(extra_config={"remote_model_name": "  renamed-model "})
    )
    assert chat.get_model_name() == "renamed-model"


def test_constructor_getters() -> None:
    chat = RemoteAPIChat(
        _config(model_id="m-1", api_key="sk-x", custom_headers={"X-Trace": "t"})
    )
    assert chat.get_model_name() == "deepseek-chat"
    assert chat.get_model_id() == "m-1"
    assert chat.get_provider() == "deepseek"
    assert chat.get_base_url() == BASE_URL.rstrip("/")
    assert chat.get_api_key() == "sk-x"


def test_constructor_detects_provider_from_url() -> None:
    chat = RemoteAPIChat(_config(provider="", base_url="https://llm.test"))
    # Unknown host falls back to the generic provider.
    assert chat.get_provider() == "generic"


def test_constructor_deepseek_default_base_url() -> None:
    chat = RemoteAPIChat(_config(provider="deepseek", base_url=""))
    assert chat.get_base_url() == "https://api.deepseek.com/v1"


def test_constructor_azure_api_version_kept() -> None:
    chat = RemoteAPIChat(
        _config(
            provider="azure_openai",
            base_url="https://llm.test",
            extra_config={"api_version": "2024-06-01"},
        )
    )
    assert chat._api_version == "2024-06-01"
    endpoint = chat._resolve_default_endpoint()
    assert endpoint == (
        "https://llm.test/openai/deployments/deepseek-chat"
        "/chat/completions?api-version=2024-06-01"
    )


# ── provider routing ─────────────────────────────────────────────────


def test_resolve_provider_routing() -> None:
    assert isinstance(resolve_provider("weknoracloud", "m"), WeKnoraCloudProvider)
    assert isinstance(resolve_provider("aliyun", "qwen3-8b"), QwenThinkingProvider)
    assert isinstance(resolve_provider("aliyun", "qwen-max"), QwenThinkingProvider)
    assert isinstance(resolve_provider("lkeap", "deepseek-v3.2"), LkeapProvider)
    assert isinstance(resolve_provider("lkeap", "deepseek-r1"), ProviderAdapter)
    assert isinstance(resolve_provider("deepseek", "deepseek-chat"), DeepseekProvider)
    assert isinstance(resolve_provider("generic", "llama"), GenericProvider)
    assert isinstance(resolve_provider("gemini", "gemini-1.5"), GeminiProvider)
    assert isinstance(resolve_provider("volcengine", "m"), VolcengineProvider)
    assert isinstance(resolve_provider("nvidia", "m"), NvidiaProvider)
    assert isinstance(resolve_provider("azure_openai", "gpt-4o"), AzureProvider)
    assert isinstance(resolve_provider("azure_openai", "o1"), AzureReasoningProvider)
    assert isinstance(resolve_provider("openai", "gpt-4o"), ProviderAdapter)
    assert isinstance(resolve_provider("openai", "o3-mini"), OpenAIReasoningProvider)
    assert isinstance(resolve_provider("moonshot", "moonshot-v1-8k"), MoonshotProvider)
    assert isinstance(resolve_provider("unknown", "m"), ProviderAdapter)


def test_moonshot_shape_request() -> None:
    provider = MoonshotProvider()
    req: JsonObject = {"temperature": 0.5, "top_p": 0.9}
    provider.shape_request(req, None, False)
    assert req == {"temperature": 1}


def test_deepseek_shape_request_removes_tool_choice() -> None:
    provider = DeepseekProvider()
    req: JsonObject = {"tool_choice": "auto"}
    provider.shape_request(req, ChatOptions(tool_choice="auto"), False)
    assert "tool_choice" not in req


def test_openai_reasoning_shape_request_migrates_max_tokens() -> None:
    provider = OpenAIReasoningProvider()
    req: JsonObject = {
        "temperature": 0.7,
        "max_tokens": 100,
        "max_completion_tokens": 0,
    }
    provider.shape_request(req, None, False)
    assert req == {"max_completion_tokens": 100}


# ── message conversion ───────────────────────────────────────────────


def test_convert_messages_plain() -> None:
    chat = RemoteAPIChat(_config())
    converted = chat.convert_messages(
        [Message(role="user", content="hello"), Message(role="assistant", content="hi")]
    )
    assert converted == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_convert_messages_multi_content() -> None:
    chat = RemoteAPIChat(_config())
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
    converted = chat.convert_messages([message])
    assert converted[0]["multi_content"] == [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "http://img", "detail": "auto"}},
    ]


def test_convert_messages_images() -> None:
    chat = RemoteAPIChat(_config())
    message = Message(role="user", content="describe", images=["http://img/x.png"])
    converted = chat.convert_messages([message])
    parts = converted[0]["multi_content"]
    assert parts == [
        {"type": "image_url", "image_url": {"url": "http://img/x.png", "detail": "auto"}},
        {"type": "text", "text": "describe"},
    ]


def test_convert_messages_tool_calls_and_tool_role() -> None:
    chat = RemoteAPIChat(_config())
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
    converted = chat.convert_messages([assistant, tool_result])
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


# ── standard request building ────────────────────────────────────────


def test_build_chat_completion_request_basics() -> None:
    chat = RemoteAPIChat(_config())
    req = chat.build_chat_completion_request(
        [Message(role="user", content="hi")], None, True
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
    req = chat.build_chat_completion_request([Message(role="user", content="hi")], opts, False)
    assert req["temperature"] == 0.5
    assert req["top_p"] == 0.9
    assert req["max_tokens"] == 64
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
    req = chat.build_chat_completion_request(
        [Message(role="user", content="hi")],
        ChatOptions(tool_choice="specific_tool"),
        False,
    )
    assert req["tool_choice"] == {"type": "function", "function": {"name": "specific_tool"}}


def test_build_chat_completion_request_format_appends_schema() -> None:
    chat = RemoteAPIChat(_config())
    req = chat.build_chat_completion_request(
        [Message(role="user", content="list")],
        ChatOptions(format={"type": "object", "properties": {"name": {"type": "string"}}}),
        False,
    )
    assert req["response_format"] == {"type": "json_object"}
    raw_messages = req["messages"]
    assert isinstance(raw_messages, list)
    last = raw_messages[-1]
    assert isinstance(last, dict)
    content = last.get("content")
    assert isinstance(content, str)
    assert 'Use this JSON schema: {"type": "object"' in content


# ── build_outbound (adapter + thinking composition) ──────────────────


def test_build_outbound_enable_thinking_qwen() -> None:
    chat = RemoteAPIChat(
        _config(provider="aliyun", model_name="qwen3-8b", base_url="http://llm.test")
    )
    body, endpoint = chat.build_outbound(
        [Message(role="user", content="hi")], ChatOptions(thinking=True), False
    )
    assert body["enable_thinking"] is False  # forced off for non-stream
    assert endpoint == "http://llm.test/chat/completions"

    stream_body, _ = chat.build_outbound(
        [Message(role="user", content="hi")], ChatOptions(thinking=True), True
    )
    assert stream_body["enable_thinking"] is True


def test_build_outbound_thinking_override_wins() -> None:
    chat = RemoteAPIChat(
        _config(
            provider="deepseek",
            model_name="deepseek-chat",
            extra_config={"thinking_control": "thinking_type"},
        )
    )
    body, _ = chat.build_outbound(
        [Message(role="user", content="hi")], ChatOptions(thinking=True), False
    )
    assert body["thinking"] == {"type": "enabled"}


def test_build_outbound_managed_cloud_endpoint_and_signing() -> None:
    chat = RemoteAPIChat(
        _config(
            provider="weknoracloud",
            model_name="m",
            app_id="app",
            app_secret="secret",
            base_url="http://llm.test",
        )
    )
    _body, endpoint = chat.build_outbound([Message(role="user", content="hi")], None, False)
    assert endpoint == "http://llm.test/api/v1/chat/completions"
    creds = chat.auth_creds()
    headers = chat._adapter.auth_headers(creds, b"{}")
    assert headers["X-APPID"] == "app"
    assert "X-Signature" in headers


# ── non-streaming path ───────────────────────────────────────────────


async def test_chat_with_raw_http_parses_completion() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        payload = json.loads(request.content)
        assert payload["model"] == "deepseek-chat"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chat = RemoteAPIChat(_config(), client=client)
    result = await chat.chat([Message(role="user", content="hi")])
    assert result.content == "hello"
    assert result.finish_reason == "stop"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.usage.total_tokens == 15
    await _close(chat)


async def test_chat_with_raw_http_error_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad request")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chat = RemoteAPIChat(_config(), client=client)
    with pytest.raises(AIProviderError, match="status 400"):
        await chat.chat([Message(role="user", content="hi")])
    await _close(chat)


async def test_chat_with_raw_http_tool_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
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
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chat = RemoteAPIChat(_config(provider="gemini", base_url="http://llm.test"), client=client)
    result = await chat.chat([Message(role="user", content="hi")])
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call-1"
    assert result.tool_calls[0].function.name == "lookup"
    assert result.finish_reason == "tool_calls"
    await _close(chat)


# ── streaming path ───────────────────────────────────────────────────


async def _sse_response(data_lines: list[str]) -> httpx.Response:
    return httpx.Response(
        200,
        text="\n\n".join(data_lines) + "\n\n",
        headers={"content-type": "text/event-stream"},
    )


async def test_chat_stream_yields_events() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=(
                'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
                'data: {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chat = RemoteAPIChat(_config(), client=client)
    events = [event async for event in chat.chat_stream([Message(role="user", content="hi")])]
    assert events[0].response_type == ResponseType.ANSWER
    assert events[0].content == "Hello"
    assert events[0].done is False
    assert events[1].content == " world"
    assert events[1].done is True
    assert events[1].finish_reason == "stop"
    assert events[-1].done is True
    assert events[-1].response_type == ResponseType.ANSWER
    await _close(chat)


async def test_chat_stream_thinking_then_answer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=(
                'data: {"choices": [{"delta": {"reasoning_content": "think "}}]}\n\n'
                'data: {"choices": [{"delta": {"reasoning_content": "more"}}]}\n\n'
                'data: {"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chat = RemoteAPIChat(_config(), client=client)
    events = [event async for event in chat.chat_stream([Message(role="user", content="hi")])]
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


async def test_chat_stream_tool_calls_delta() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=(
                'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call-1", '
                '"type": "function", "function": {"name": "look", "arguments": ""}}]}}]}\n\n'
                'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": '
                '{"arguments": "{\\"q\\": \\"x\\"}"}}]}}]}\n\n'
                'data: {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chat = RemoteAPIChat(_config(), client=client)
    events = [event async for event in chat.chat_stream([Message(role="user", content="hi")])]
    tool_calls = events[-1].tool_calls
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call-1"
    assert tool_calls[0].function.name == "look"
    assert tool_calls[0].function.arguments == '{"q": "x"}'
    await _close(chat)


# ── SSRF validation ──────────────────────────────────────────────────


def test_validate_url_for_ssrf_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SSRF_WHITELIST", "trusted.test")
    validate_url_for_ssrf("http://trusted.test/v1")
    validate_url_for_ssrf("")


def test_validate_url_for_ssrf_rejects_restricted_hostname() -> None:
    with pytest.raises(SSRFValidationError, match="restricted"):
        validate_url_for_ssrf("http://localhost:11434")
    with pytest.raises(SSRFValidationError, match="restricted"):
        validate_url_for_ssrf("http://metadata.google.internal")


def test_validate_url_for_ssrf_rejects_loopback() -> None:
    # 127.0.0.1 is a restricted hostname before the direct-IP rule.
    with pytest.raises(SSRFValidationError, match="restricted"):
        validate_url_for_ssrf("http://127.0.0.1")


def test_validate_url_for_ssrf_rejects_direct_ip() -> None:
    with pytest.raises(SSRFValidationError, match="direct IP"):
        validate_url_for_ssrf("http://10.0.0.1")
    with pytest.raises(SSRFValidationError, match="direct IP"):
        validate_url_for_ssrf("http://[2001:db8::1]")


def test_validate_url_for_ssrf_rejects_bad_scheme() -> None:
    with pytest.raises(SSRFValidationError, match="invalid scheme"):
        validate_url_for_ssrf("ftp://example.com")


def test_validate_url_for_ssrf_rejects_ip_like_hostname() -> None:
    with pytest.raises(SSRFValidationError, match="IP-like"):
        validate_url_for_ssrf("http://2130706433")
    with pytest.raises(SSRFValidationError, match="IP-like"):
        validate_url_for_ssrf("http://0x7f.0.0.1")


def test_validate_url_for_ssrf_rejects_blocked_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stub DNS so a public-looking hostname resolves without real lookups.
    import socket

    def fake_getaddrinfo(host: str, port: object, **kwargs: object) -> list[object]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]

    monkeypatch.delenv("SSRF_WHITELIST", raising=False)
    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(SSRFValidationError, match="port 6379 is blocked"):
        validate_url_for_ssrf("http://example.com:6379")


def test_custom_headers_skip_reserved() -> None:
    assert is_reserved_header("Authorization")
    assert is_reserved_header("Content-Type")
    assert not is_reserved_header("X-Custom")
    merged = apply_custom_headers(
        {"Authorization": "Bearer x", "Accept": "application/json"},
        {"Authorization": "Bearer evil", "X-Trace": "t1"},
    )
    assert merged == {
        "Authorization": "Bearer x",
        "Accept": "application/json",
        "X-Trace": "t1",
    }


# ── SSE reader ───────────────────────────────────────────────────────


async def _async_lines(lines: list[str]) -> AsyncIterator[str]:
    for line in lines:
        yield line


async def test_sse_reader_events() -> None:
    reader = SSEReader(
        _async_lines(
            [
                "event: message",
                "data: {\"a\":1}",
                "",
                "data: [DONE]",
                "data: ignored",
            ]
        )
    )
    events = [event async for event in reader]
    assert events[0].data == '{"a":1}'
    assert events[0].done is False
    assert events[1].done is True


# ── response parsing helpers ─────────────────────────────────────────


def test_remove_thinking_content() -> None:
    assert remove_thinking_content("plain answer") == "plain answer"
    # The strip heuristic trims whitespace before matching, so the leading-space
    # tag never matches real output; mirror the reference behavior exactly.
    assert remove_thinking_content(" think step by step response actual") == (
        " think step by step response actual"
    )
    assert remove_thinking_content("") == ""


def test_token_usage_from_openai_details() -> None:
    usage = token_usage_from_openai(
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
         "prompt_tokens_details": {"cached_tokens": 4}},
        "deepseek",
    )
    assert usage.cache_read_tokens == 4
    assert usage.cache_status == PromptCacheStatus.HIT
    assert usage.cache_reported is True


def test_token_usage_from_openai_unsupported_provider() -> None:
    usage = token_usage_from_openai(
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "moonshot",
    )
    assert usage.cache_status == PromptCacheStatus.UNSUPPORTED


def test_apply_raw_prompt_cache_usage() -> None:
    usage = token_usage_from_openai(
        {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}, "deepseek"
    )
    apply_raw_prompt_cache_usage(
        b'{"usage": {"prompt_cache_hit_tokens": 7, "prompt_cache_miss_tokens": 3}}',
        usage,
    )
    assert usage.cache_read_tokens == 7
    assert usage.cache_miss_tokens == 3
    assert usage.cache_status == PromptCacheStatus.HIT


def test_fingerprint_prompt_prefix_stable() -> None:
    fp = prompt_prefix_fingerprint(
        [Message(role="system", content="sys"), Message(role="user", content="hi")],
        ChatOptions(
            tools=[Tool(type="function", function=FunctionDef(name="t", description=""))]
        ),
    )
    assert len(fp) == 16
    assert fingerprint_prompt_prefix("a", "b") == fingerprint_prompt_prefix("a", "b")
    assert fingerprint_prompt_prefix("a", "b") != fingerprint_prompt_prefix("a", "c")


# ── image resolution ─────────────────────────────────────────────────


def test_resolve_image_url_for_llm_passthrough() -> None:
    assert resolve_image_url_for_llm("http://img/x.png") == "http://img/x.png"
    assert resolve_image_url_for_llm("https://img/x.png") == "https://img/x.png"
    assert resolve_image_url_for_llm("data:image/png;base64,AAAA") == "data:image/png;base64,AAAA"


def test_resolve_image_url_for_llm_local_storage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    import pathlib

    tmp = pathlib.Path(str(tmp_path))
    (tmp / "a.png").write_bytes(b"\x89PNG\r\n\x1a\nbytes")
    monkeypatch.setenv("LOCAL_STORAGE_BASE_DIR", str(tmp))
    resolved = resolve_image_url_for_llm("local://a.png")
    assert resolved.startswith("data:image/png;base64,")
    assert is_application_stored_image("local://a.png")
    assert not is_application_stored_image("http://x")


def test_resolve_image_url_for_ollama() -> None:
    import base64

    raw = b"\x00\x01\x02"
    uri = f"data:image/png;base64,{base64.b64encode(raw).decode()}"
    assert resolve_image_url_for_ollama(uri) == raw
    assert resolve_image_url_for_ollama("http://img/x.png") is None


def test_strip_images_from_messages() -> None:
    messages = [Message(role="user", content="x", images=["http://img"])]
    cleaned = strip_images_from_messages(messages)
    assert cleaned[0].images == []
    assert messages[0].images == ["http://img"]


# ── json field extractor ─────────────────────────────────────────────


def test_json_field_extractor_streams_field() -> None:
    extractor = JSONFieldExtractor("thought")
    assert extractor.feed('{"thought":"hel') == "hel"
    assert extractor.feed("lo") == "lo"
    assert extractor.feed("\\n\\u4e16") == "\n世"
    assert extractor.feed('界"}') == "界"
    assert extractor.is_done()
    assert extractor.feed("ignored") == ""


def test_json_field_extractor_unescape_and_skip() -> None:
    extractor = JSONFieldExtractor("thought")
    assert extractor.feed('{"unrelated":1, "thought":"a\\"b"}') == 'a"b'
    assert extractor.is_done()
