"""Anthropic Messages API chat client.

``AnthropicChat`` talks the native Anthropic Messages protocol for both the
non-streaming and SSE streaming paths, mirroring the upstream contract. The
``/v1/messages`` endpoint is derived from the configured base URL the same way
the reference implementation derives it. Tool calls round-trip between the
internal ``ToolCall`` / ``LLMToolCall`` shape and Anthropic's ``tool_use`` /
``tool_result`` content blocks.

Stream parsing reuses the shared ``SSEReader``; the thinking/answer hand-off
goes through the shared ``ThinkingEmitter`` so every streaming path emits the
same single ``thinking`` done marker before the first answer token.
"""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import AsyncIterable, AsyncIterator

import httpx

from src.ai.llm.sse_reader import SSEReader
from src.ai.llm.stream_emit import ThinkingEmitter
from src.ai.llm.transport import (
    DEFAULT_CHAT_TIMEOUT_SECONDS,
    DEFAULT_STREAM_TIMEOUT_SECONDS,
    apply_custom_headers,
    build_ssrf_safe_client,
    validate_url_for_ssrf,
    with_llm_timeout,
)
from src.ai.llm.types import (
    ChatConfig,
    ChatOptions,
    ChatResponse,
    FunctionCall,
    LLMToolCall,
    Message,
    MessageContentPart,
    ResponseType,
    StreamResponse,
    TokenUsage,
    Tool,
    ToolCall,
)
from src.ai.llm.usage import log_usage
from src.ai.provider.providers.anthropic import ANTHROPIC_BASE_URL
from src.common.exception import ValidationError, AIProviderError
from src.common.json import JsonObject, JsonValue

#: ``anthropic-version`` header value sent on every request.
ANTHROPIC_VERSION = "2023-06-01"

#: Completion budget used when the caller sets no token limit.
_DEFAULT_MAX_TOKENS = 1024


# ── Endpoint derivation ───────────────────────────────────────────────


def is_anthropic_messages_endpoint(base_url: str) -> bool:
    """True when ``base_url`` already points at the ``/messages`` endpoint."""
    parsed = urllib.parse.urlsplit(base_url)
    path = parsed.path.rstrip("/")
    return path.endswith("/messages")


def is_anthropic_versioned_base_url(base_url: str) -> bool:
    """True when ``base_url`` carries a versioned prefix (``/v1``, ``/v1beta``)."""
    parsed = urllib.parse.urlsplit(base_url)
    path = parsed.path.rstrip("/")
    return path.endswith(("/v1", "/v1beta"))


# ── Usage helpers ─────────────────────────────────────────────────────


def _usage_int(raw: JsonObject, key: str) -> int:
    value = raw.get(key)
    return value if isinstance(value, int) else 0


def _usage_optional_int(raw: JsonObject, key: str) -> int | None:
    value = raw.get(key)
    return value if isinstance(value, int) else None


def anthropic_usage_from(raw: JsonObject) -> TokenUsage:
    """Build a normalized :class:`TokenUsage` from an Anthropic usage block."""
    input_tokens = _usage_int(raw, "input_tokens")
    output_tokens = _usage_int(raw, "output_tokens")
    cache_read = _usage_optional_int(raw, "cache_read_input_tokens")
    cache_write = _usage_optional_int(raw, "cache_creation_input_tokens")
    read = cache_read or 0
    write = cache_write or 0
    prompt_tokens = input_tokens + read + write
    usage = TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=output_tokens,
        total_tokens=prompt_tokens + output_tokens,
    )
    usage.set_prompt_cache_usage(
        read,
        write,
        max(0, prompt_tokens - read),
        cache_read is not None or cache_write is not None,
    )
    return usage


def merge_anthropic_cache_counters(
    current_read: int,
    current_write: int,
    current_reported: bool,
    cache_read: int | None,
    cache_write: int | None,
) -> tuple[int, int, bool]:
    """Merge one usage block's cache counters into the running totals."""
    read = current_read
    write = current_write
    reported = current_reported or cache_read is not None or cache_write is not None
    if cache_read is not None:
        read = max(read, cache_read)
    if cache_write is not None:
        write = max(write, cache_write)
    return read, write, reported


def merge_anthropic_usage(
    current: TokenUsage | None,
    input_tokens: int,
    output_tokens: int,
    cache_read: int | None,
    cache_write: int | None,
) -> TokenUsage:
    """Merge one stream event's usage block into the running usage."""
    if current is None:
        current = TokenUsage()
    read, write, reported = merge_anthropic_cache_counters(
        current.cache_read_tokens,
        current.cache_write_tokens,
        current.cache_reported,
        cache_read,
        cache_write,
    )
    uncached_input = max(
        0, current.prompt_tokens - current.cache_read_tokens - current.cache_write_tokens
    )
    uncached_input = max(uncached_input, input_tokens)
    current.prompt_tokens = uncached_input + read + write
    current.completion_tokens = max(current.completion_tokens, output_tokens)
    current.total_tokens = current.prompt_tokens + current.completion_tokens
    current.set_prompt_cache_usage(read, write, max(0, current.prompt_tokens - read), reported)
    return current


# ── Tool-call conversion ──────────────────────────────────────────────


def _as_str(value: JsonValue | None) -> str:
    return value if isinstance(value, str) else ""


def _as_object(value: JsonValue | None) -> JsonObject:
    return value if isinstance(value, dict) else {}


def _parse_json_object(raw: str) -> JsonValue:
    """Parse ``raw`` as a JSON object, falling back to ``{}``."""
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def text_from_multi_content(parts: list[MessageContentPart]) -> str:
    """Join the trimmed text parts of a multi-content message."""
    if not parts:
        return ""
    text_parts = [part.text.strip() for part in parts if part.type == "text" and part.text.strip()]
    return "\n".join(text_parts)


def tool_call_to_anthropic(tool_call: ToolCall) -> JsonObject:
    """Convert an internal assistant tool call into a ``tool_use`` block."""
    return {
        "type": "tool_use",
        "id": tool_call.id,
        "name": tool_call.function.name,
        "input": _parse_json_object(tool_call.function.arguments),
    }


def tool_result_to_anthropic(message: Message) -> JsonObject:
    """Convert a ``tool`` role message into a ``tool_result`` block."""
    return {
        "type": "tool_result",
        "tool_use_id": message.tool_call_id,
        "content": message.content,
    }


def tool_to_anthropic(tool: Tool) -> JsonObject:
    """Convert an internal tool definition into the Anthropic tools shape."""
    result: JsonObject = {
        "name": tool.function.name,
        "description": tool.function.description,
    }
    if tool.function.parameters is not None:
        result["input_schema"] = tool.function.parameters
    return result


def tool_choice_to_anthropic(tool_choice: str) -> JsonObject:
    """Map the internal ``tool_choice`` string onto an Anthropic object."""
    if tool_choice == "auto":
        return {"type": "auto"}
    if tool_choice == "required":
        return {"type": "any"}
    if tool_choice == "none":
        return {"type": "none"}
    return {"type": "tool", "name": tool_choice}


def tool_use_from_block(block: JsonObject) -> LLMToolCall | None:
    """Convert a response ``tool_use`` block into an internal tool call."""
    tool_id = block.get("id")
    name = block.get("name")
    if not isinstance(tool_id, str) or not isinstance(name, str) or not name:
        return None
    raw_input = block.get("input")
    if isinstance(raw_input, dict):
        arguments = json.dumps(raw_input, ensure_ascii=False)
    elif isinstance(raw_input, str):
        arguments = raw_input
    else:
        arguments = ""
    return LLMToolCall(
        id=tool_id,
        function=FunctionCall(name=name, arguments=arguments),
        provider_metadata={"type": "tool_use"},
    )


def _ordered_tool_calls(tool_call_map: dict[int, LLMToolCall]) -> list[LLMToolCall]:
    """Return accumulated tool calls ordered by their stream block index."""
    if not tool_call_map:
        return []
    return [tool_call_map[index] for index in sorted(tool_call_map)]


# ── Client ────────────────────────────────────────────────────────────


class AnthropicChat:
    """Native Anthropic Messages API chat client.

    ``config`` supplies the endpoint, credentials and model names; ``client``
    lets tests inject an ``httpx.AsyncClient`` with a mock transport.
    """

    def __init__(
        self,
        config: ChatConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if config.base_url:
            validate_url_for_ssrf(config.base_url)
        if not config.api_key.strip():
            raise ValidationError(code="anthropic.api_key_required", message="Anthropic provider: API key is required")

        base_url = config.base_url.rstrip("/")
        if not base_url:
            base_url = ANTHROPIC_BASE_URL

        self._model_name = config.model_name
        self._model_id = config.model_id
        self._base_url = base_url
        self._api_key = config.api_key
        self._custom_headers = dict(config.custom_headers or {})
        self._client = client or build_ssrf_safe_client()

    # ── Getters ─────────────────────────────────────────────────────

    def get_model_name(self) -> str:
        return self._model_name

    def get_model_id(self) -> str:
        return self._model_id

    def endpoint(self) -> str:
        """Resolve the Messages endpoint from the configured base URL."""
        base_url = self._base_url.rstrip("/")
        if is_anthropic_messages_endpoint(base_url):
            return base_url
        if is_anthropic_versioned_base_url(base_url):
            return base_url + "/messages"
        return base_url + "/v1/messages"

    async def aclose(self) -> None:
        """Release the underlying HTTP client."""
        await self._client.aclose()

    # ── Request building ─────────────────────────────────────────────

    def build_request(
        self,
        messages: list[Message],
        opts: ChatOptions | None,
    ) -> JsonObject:
        """Build the Anthropic Messages request body."""
        req: JsonObject = {
            "model": self._model_name,
            "max_tokens": _DEFAULT_MAX_TOKENS,
            "messages": [],
        }
        if opts is not None:
            if opts.max_tokens > 0:
                req["max_tokens"] = opts.max_tokens
            elif opts.max_completion_tokens > 0:
                req["max_tokens"] = opts.max_completion_tokens
            if opts.temperature > 0:
                req["temperature"] = opts.temperature
            if opts.top_p > 0:
                req["top_p"] = opts.top_p
            if opts.tools:
                req["tools"] = [tool_to_anthropic(tool) for tool in opts.tools]
            if opts.tool_choice:
                req["tool_choice"] = tool_choice_to_anthropic(opts.tool_choice)

        system_parts: list[str] = []
        wire_messages: list[JsonValue] = []
        for msg in messages:
            content = msg.content.strip()
            if content == "":
                content = text_from_multi_content(msg.multi_content)
            if msg.role == "system":
                if content:
                    system_parts.append(content)
                continue
            if msg.tool_calls:
                blocks: list[JsonValue] = []
                if content:
                    blocks.append({"type": "text", "text": content})
                for tool_call in msg.tool_calls:
                    blocks.append(tool_call_to_anthropic(tool_call))
                wire_messages.append({"role": "assistant", "content": blocks})
                continue
            if msg.role == "tool":
                wire_messages.append({"role": "user", "content": [tool_result_to_anthropic(msg)]})
                continue
            if content == "":
                continue
            role = "assistant" if msg.role == "assistant" else "user"
            wire_messages.append({"role": role, "content": content})
        req["messages"] = wire_messages
        if system_parts:
            req["system"] = "\n\n".join(system_parts)
        return req

    # ── Response parsing ─────────────────────────────────────────────

    def parse_response(self, resp: JsonObject) -> ChatResponse:
        """Parse a non-streaming Messages response into ``ChatResponse``."""
        content_parts: list[str] = []
        tool_calls: list[LLMToolCall] = []
        raw_content = resp.get("content")
        if isinstance(raw_content, list):
            for block in raw_content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text = block.get("text")
                    if isinstance(text, str) and text:
                        content_parts.append(text)
                elif block_type == "tool_use":
                    tool_call = tool_use_from_block(block)
                    if tool_call is not None:
                        tool_calls.append(tool_call)
        result = ChatResponse(
            content="".join(content_parts),
            finish_reason=_as_str(resp.get("stop_reason")),
            usage=anthropic_usage_from(_as_object(resp.get("usage"))),
        )
        result.tool_calls = tool_calls
        return result

    # ── Non-streaming path ───────────────────────────────────────────

    async def chat(self, messages: list[Message], opts: ChatOptions | None = None) -> ChatResponse:
        """Perform a non-streaming Messages round-trip."""
        async with with_llm_timeout(DEFAULT_CHAT_TIMEOUT_SECONDS):
            body = self.build_request(messages, opts)
            json_data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            endpoint_url = self.endpoint()
            validate_url_for_ssrf(endpoint_url)

            headers = {
                "Content-Type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": ANTHROPIC_VERSION,
            }
            headers = apply_custom_headers(headers, self._custom_headers)

            response = await self._client.post(endpoint_url, content=json_data, headers=headers)
            content_type = response.headers.get("content-type", "").lower()
            if "text/event-stream" in content_type:
                result = await parse_anthropic_sse(response.aiter_lines())
                if response.status_code != 200:
                    raise AIProviderError(
                        f"API request failed with status {response.status_code}: {result.content}"
                    )
                log_usage(self._model_name, result.usage)
                return result

            body_bytes = response.content
            try:
                payload = json.loads(body_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise AIProviderError("decode response: failed to parse response JSON") from exc
            if not isinstance(payload, dict):
                raise AIProviderError("decode response: unexpected response shape")
            if response.status_code != 200:
                error = payload.get("error")
                if isinstance(error, dict):
                    message = error.get("message")
                    if isinstance(message, str) and message:
                        raise AIProviderError(
                            f"API request failed with status {response.status_code}: {message}"
                        )
                raise AIProviderError(
                    f"API request failed with status {response.status_code}: {response.text}"
                )
            result = self.parse_response(payload)
            log_usage(self._model_name, result.usage)
            return result

    # ── Streaming path ───────────────────────────────────────────────

    async def chat_stream(
        self, messages: list[Message], opts: ChatOptions | None = None
    ) -> AsyncIterator[StreamResponse]:
        """Stream a Messages round-trip, yielding one event per SSE event."""
        async with with_llm_timeout(DEFAULT_STREAM_TIMEOUT_SECONDS):
            body = self.build_request(messages, opts)
            body["stream"] = True
            json_data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            endpoint_url = self.endpoint()
            validate_url_for_ssrf(endpoint_url)

            headers = {
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "x-api-key": self._api_key,
                "anthropic-version": ANTHROPIC_VERSION,
            }
            headers = apply_custom_headers(headers, self._custom_headers)

            async with self._client.stream(
                "POST", endpoint_url, content=json_data, headers=headers
            ) as response:
                if response.status_code != 200:
                    error_body = (await response.aread()).decode("utf-8", errors="replace")
                    raise AIProviderError(
                        f"API request failed with status {response.status_code}: {error_body}"
                    )
                async for event in process_anthropic_stream(
                    self._model_name, response.aiter_lines()
                ):
                    yield event


# ── Stream parsing ────────────────────────────────────────────────────


async def parse_anthropic_sse(lines: AsyncIterable[str]) -> ChatResponse:
    """Aggregate a full SSE body into a :class:`ChatResponse`.

    Used when a non-streaming request gets an SSE payload back (gateway
    fallback).
    """
    reader = SSEReader(lines)
    content_parts: list[str] = []
    finish_reason = ""
    input_tokens = 0
    output_tokens = 0
    cache_read_tokens = 0
    cache_write_tokens = 0
    cache_reported = False
    tool_call_map: dict[int, LLMToolCall] = {}

    async for event in reader:
        if event.done:
            break
        if event.data is None:
            continue
        try:
            stream_event = json.loads(event.data)
        except json.JSONDecodeError as exc:
            raise AIProviderError("decode SSE response: failed to parse response JSON") from exc
        if not isinstance(stream_event, dict):
            continue

        error = stream_event.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                raise AIProviderError(f"API stream error: {message}")

        message = stream_event.get("message")
        if isinstance(message, dict):
            raw_usage = message.get("usage")
            if isinstance(raw_usage, dict):
                input_tokens = max(input_tokens, _usage_int(raw_usage, "input_tokens"))
                output_tokens = max(output_tokens, _usage_int(raw_usage, "output_tokens"))
                cache_read_tokens, cache_write_tokens, cache_reported = (
                    merge_anthropic_cache_counters(
                        cache_read_tokens,
                        cache_write_tokens,
                        cache_reported,
                        _usage_optional_int(raw_usage, "cache_read_input_tokens"),
                        _usage_optional_int(raw_usage, "cache_creation_input_tokens"),
                    )
                )

        block = stream_event.get("content_block")
        if isinstance(block, dict) and block.get("type") == "tool_use":
            index = stream_event.get("index")
            if isinstance(index, int):
                tool_call = tool_use_from_block(block)
                if tool_call is not None:
                    tool_call_map[index] = tool_call

        delta = stream_event.get("delta")
        if isinstance(delta, dict):
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                text = delta.get("text")
                if isinstance(text, str) and text:
                    content_parts.append(text)
            elif delta_type == "input_json_delta":
                partial = delta.get("partial_json")
                index = stream_event.get("index")
                if isinstance(partial, str) and isinstance(index, int):
                    entry = tool_call_map.get(index)
                    if entry is not None:
                        entry.function.arguments += partial
            stop_reason = delta.get("stop_reason")
            if isinstance(stop_reason, str) and stop_reason:
                finish_reason = stop_reason

        raw_usage = stream_event.get("usage")
        if isinstance(raw_usage, dict):
            input_tokens = max(input_tokens, _usage_int(raw_usage, "input_tokens"))
            output_tokens = max(output_tokens, _usage_int(raw_usage, "output_tokens"))
            cache_read_tokens, cache_write_tokens, cache_reported = merge_anthropic_cache_counters(
                cache_read_tokens,
                cache_write_tokens,
                cache_reported,
                _usage_optional_int(raw_usage, "cache_read_input_tokens"),
                _usage_optional_int(raw_usage, "cache_creation_input_tokens"),
            )

    prompt_tokens = input_tokens + cache_read_tokens + cache_write_tokens
    usage = TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=output_tokens,
        total_tokens=prompt_tokens + output_tokens,
    )
    usage.set_prompt_cache_usage(
        cache_read_tokens,
        cache_write_tokens,
        max(0, prompt_tokens - cache_read_tokens),
        cache_reported,
    )
    return ChatResponse(
        content="".join(content_parts),
        finish_reason=finish_reason,
        usage=usage,
        tool_calls=_ordered_tool_calls(tool_call_map),
    )


async def process_anthropic_stream(
    model: str,
    lines: AsyncIterable[str],
) -> AsyncIterator[StreamResponse]:
    """Parse an Anthropic SSE stream into ``StreamResponse`` events."""
    reader = SSEReader(lines)
    usage: TokenUsage | None = None
    finish_reason = ""
    emitter = ThinkingEmitter()
    tool_call_map: dict[int, LLMToolCall] = {}

    async for event in reader:
        if event.done:
            log_usage(model, usage)
            done_event = emitter.finish()
            if done_event is not None:
                yield done_event
            yield StreamResponse(
                response_type=ResponseType.ANSWER,
                done=True,
                tool_calls=_ordered_tool_calls(tool_call_map),
                usage=usage,
                finish_reason=finish_reason,
            )
            return
        if event.data is None:
            continue
        try:
            stream_event = json.loads(event.data)
        except json.JSONDecodeError:
            yield StreamResponse(
                response_type=ResponseType.ERROR,
                content="decode SSE response: failed to parse response JSON",
                done=True,
            )
            return
        if not isinstance(stream_event, dict):
            continue

        error = stream_event.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                yield StreamResponse(
                    response_type=ResponseType.ERROR,
                    content=message,
                    done=True,
                )
                return

        message = stream_event.get("message")
        if isinstance(message, dict):
            raw_usage = message.get("usage")
            if isinstance(raw_usage, dict):
                usage = merge_anthropic_usage(
                    usage,
                    _usage_int(raw_usage, "input_tokens"),
                    _usage_int(raw_usage, "output_tokens"),
                    _usage_optional_int(raw_usage, "cache_read_input_tokens"),
                    _usage_optional_int(raw_usage, "cache_creation_input_tokens"),
                )

        block = stream_event.get("content_block")
        if isinstance(block, dict) and block.get("type") == "tool_use":
            index = stream_event.get("index")
            if isinstance(index, int):
                tool_call = tool_use_from_block(block)
                if tool_call is not None:
                    tool_call_map[index] = tool_call

        delta = stream_event.get("delta")
        if isinstance(delta, dict):
            delta_type = delta.get("type")
            index = stream_event.get("index")
            if delta_type == "text_delta":
                text = delta.get("text")
                if isinstance(text, str) and text:
                    done_event = emitter.finish()
                    if done_event is not None:
                        yield done_event
                    yield StreamResponse(
                        response_type=ResponseType.ANSWER,
                        content=text,
                        done=False,
                    )
            elif delta_type == "thinking_delta":
                thinking = delta.get("thinking")
                if isinstance(thinking, str) and thinking:
                    yield emitter.emit(thinking)
            elif delta_type == "input_json_delta" and isinstance(index, int):
                partial = delta.get("partial_json")
                if isinstance(partial, str) and partial:
                    entry = tool_call_map.get(index)
                    if entry is not None:
                        entry.function.arguments += partial
            stop_reason = delta.get("stop_reason")
            if isinstance(stop_reason, str) and stop_reason:
                finish_reason = stop_reason
                done_event = emitter.finish()
                if done_event is not None:
                    yield done_event

        if stream_event.get("type") == "content_block_stop":
            index = stream_event.get("index")
            if isinstance(index, int):
                entry = tool_call_map.get(index)
                if entry is not None and entry.id and entry.function.name:
                    yield StreamResponse(
                        response_type=ResponseType.TOOL_CALL,
                        done=False,
                        data={
                            "tool_name": entry.function.name,
                            "tool_call_id": entry.id,
                        },
                    )

        raw_usage = stream_event.get("usage")
        if isinstance(raw_usage, dict):
            usage = merge_anthropic_usage(
                usage,
                _usage_int(raw_usage, "input_tokens"),
                _usage_int(raw_usage, "output_tokens"),
                _usage_optional_int(raw_usage, "cache_read_input_tokens"),
                _usage_optional_int(raw_usage, "cache_creation_input_tokens"),
            )

    log_usage(model, usage)
    done_event = emitter.finish()
    if done_event is not None:
        yield done_event
    yield StreamResponse(
        response_type=ResponseType.ANSWER,
        done=True,
        tool_calls=_ordered_tool_calls(tool_call_map),
        usage=usage,
        finish_reason=finish_reason,
    )


# ── Factory ───────────────────────────────────────────────────────────


def new_anthropic_chat(
    config: ChatConfig, *, client: httpx.AsyncClient | None = None
) -> AnthropicChat:
    """Create an :class:`AnthropicChat` for ``config``."""
    return AnthropicChat(config, client=client)


__all__ = [
    "ANTHROPIC_VERSION",
    "AnthropicChat",
    "anthropic_usage_from",
    "is_anthropic_messages_endpoint",
    "is_anthropic_versioned_base_url",
    "merge_anthropic_cache_counters",
    "merge_anthropic_usage",
    "new_anthropic_chat",
    "parse_anthropic_sse",
    "process_anthropic_stream",
    "text_from_multi_content",
    "tool_call_to_anthropic",
    "tool_choice_to_anthropic",
    "tool_result_to_anthropic",
    "tool_to_anthropic",
    "tool_use_from_block",
]
