"""OpenAI-compatible chat backend over raw HTTP.

``RemoteAPIChat`` owns the generic request / response / stream pipeline; every
provider-specific behavior is delegated to a :class:`ProviderAdapter` (see
``providers``) and a :class:`ThinkingStrategy` (see ``thinking``). The
standard request builder and the completion / stream parsers are the generic
OpenAI-compatible shapes; the provider request/stream PRs build on top of this
class and reuse its raw HTTP path, auth delegation and timeout handling.

The reference implementation routes some calls through an OpenAI SDK client;
this port has no SDK dependency, so every call goes through the raw HTTP path
with the same adapter + thinking composition.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import cast

import httpx

from src.ai.llm.image_resolve import resolve_image_url_for_llm
from src.ai.llm.json_field_extractor import JSONFieldExtractor
from src.ai.llm.prompt_cache import (
    apply_raw_prompt_cache_usage,
    token_usage_from_openai,
)
from src.ai.llm.providers import AuthCreds, resolve_provider
from src.ai.llm.sse_reader import SSEReader
from src.ai.llm.stream_emit import ThinkingEmitter
from src.ai.llm.thinking import parse_thinking_override
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
    ResponseType,
    StreamResponse,
    TokenUsage,
    ToolCallMetadata,
)
from src.ai.llm.usage import log_usage
from src.ai.provider.detect import detect_provider
from src.ai.provider.providers.deepseek import DEEPSEEK_BASE_URL
from src.ai.provider.registry import (
    PROVIDER_AZURE_OPENAI,
    PROVIDER_DEEPSEEK,
    PROVIDER_WEKNORACLOUD,
)
from src.common.exception import AIProviderError
from src.common.json import JsonObject, JsonValue

# Default api-version used for the Azure OpenAI endpoint.
_DEFAULT_AZURE_API_VERSION = "2024-02-15-preview"


def _as_str(value: object) -> str:
    return value if isinstance(value, str) else ""


def _as_object(value: object) -> JsonObject:
    return value if isinstance(value, dict) else {}


def remove_thinking_content(content: str) -> str:
    """Strip ``thinking``-tagged reasoning from a model's answer text."""
    think_start_tag = " thinking"
    think_end_tag = " response"
    trimmed = content.strip()
    if not trimmed.startswith(think_start_tag):
        return content
    last_end = trimmed.rfind(think_end_tag)
    if last_end != -1:
        return trimmed[last_end + len(think_end_tag) :].strip()
    return ""


class StreamState:
    """Accumulates one streaming chat round (thinking + tool-call deltas)."""

    def __init__(self) -> None:
        self._emitter = ThinkingEmitter()
        self.tool_call_map: dict[int, LLMToolCall] = {}
        self.last_function_name: dict[int, str] = {}
        self.name_notified: dict[int, bool] = {}
        self.field_extractors: dict[int, JSONFieldExtractor] = {}
        self.usage: TokenUsage | None = None
        self.last_finish_reason = ""

    def emit(self, content: str) -> StreamResponse:
        """Forward a reasoning chunk."""
        return self._emitter.emit(content)

    def finish(self) -> StreamResponse | None:
        """Emit the thinking-done marker when one is owed."""
        return self._emitter.finish()

    def build_ordered_tool_calls(self) -> list[LLMToolCall]:
        """Return accumulated tool calls ordered by their stream index."""
        if not self.tool_call_map:
            return []
        result: list[LLMToolCall] = []
        for index in range(len(self.tool_call_map)):
            entry = self.tool_call_map.get(index)
            if entry is not None:
                result.append(entry)
        return result

    def set_tool_call_provider_metadata(
        self, index: int, metadata: ToolCallMetadata
    ) -> None:
        if not metadata:
            return
        entry = self.tool_call_map.get(index)
        if entry is None:
            entry = LLMToolCall(
                id="", type="function", function=FunctionCall(name="", arguments="")
            )
            self.tool_call_map[index] = entry
        entry.provider_metadata = metadata


class RemoteAPIChat:
    """OpenAI-compatible chat client.

    ``config`` supplies the endpoint, credentials and provider; ``client``
    allows tests to inject an ``httpx.AsyncClient`` with a mock transport.
    """

    def __init__(
        self,
        config: ChatConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if config.base_url:
            validate_url_for_ssrf(config.base_url)

        api_key = config.api_key
        provider_name = config.provider or detect_provider(config.base_url)
        base_url = config.base_url
        if not base_url and provider_name == PROVIDER_DEEPSEEK:
            base_url = DEEPSEEK_BASE_URL
        base_url = base_url.rstrip("/")

        api_version = ""
        if provider_name == PROVIDER_AZURE_OPENAI and config.extra_config:
            api_version = (config.extra_config.get("api_version") or "").strip()

        if provider_name == PROVIDER_WEKNORACLOUD:
            if not config.app_id:
                raise ValueError("managed cloud provider: AppID is required")
            if not config.app_secret:
                raise ValueError("managed cloud provider: AppSecret is required")

        model_name = config.model_name
        if config.extra_config:
            override = (config.extra_config.get("remote_model_name") or "").strip()
            if override:
                model_name = override

        self._model_name = model_name
        self._model_id = config.model_id
        self._base_url = base_url
        self._api_key = api_key
        self._provider = provider_name
        self._app_id = config.app_id
        self._app_secret = config.app_secret
        self._custom_headers = dict(config.custom_headers or {})
        self._api_version = api_version
        self._client = client or build_ssrf_safe_client()
        self._adapter = resolve_provider(provider_name, model_name)
        self._thinking_override = parse_thinking_override(config.extra_config)

    # ── Getters ─────────────────────────────────────────────────────

    def get_model_name(self) -> str:
        return self._model_name

    def get_model_id(self) -> str:
        return self._model_id

    def get_provider(self) -> str:
        return self._provider

    def get_base_url(self) -> str:
        return self._base_url

    def get_api_key(self) -> str:
        return self._api_key

    def auth_creds(self) -> AuthCreds:
        """Bundle the credentials handed to the adapter's auth hook."""
        return AuthCreds(
            api_key=self._api_key, app_id=self._app_id, app_secret=self._app_secret
        )

    async def aclose(self) -> None:
        """Release the underlying HTTP client."""
        await self._client.aclose()

    # ── Standard request building ────────────────────────────────────

    def convert_messages(self, messages: list[Message]) -> list[JsonObject]:
        """Convert ``Message`` objects into the OpenAI wire message shape."""
        converted: list[JsonObject] = []
        for msg in messages:
            entry: JsonObject = {"role": msg.role}
            if msg.multi_content:
                parts: list[JsonValue] = []
                for part in msg.multi_content:
                    if part.type == "text":
                        parts.append({"type": "text", "text": part.text})
                    elif part.type == "image_url" and part.image_url is not None:
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": part.image_url.url,
                                    "detail": part.image_url.detail,
                                },
                            }
                        )
                entry["multi_content"] = parts
            elif msg.images and msg.role == "user":
                image_parts: list[JsonValue] = []
                for image_url in msg.images:
                    resolved = resolve_image_url_for_llm(image_url)
                    image_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": resolved, "detail": "auto"},
                        }
                    )
                image_parts.append({"type": "text", "text": msg.content})
                entry["multi_content"] = image_parts
            elif msg.content:
                entry["content"] = msg.content

            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in msg.tool_calls
                ]
            if msg.role == "tool":
                entry["tool_call_id"] = msg.tool_call_id
                entry["name"] = msg.name
            if msg.role == "assistant" and msg.reasoning_content:
                entry["reasoning_content"] = msg.reasoning_content
            converted.append(entry)
        return converted

    def build_chat_completion_request(
        self,
        messages: list[Message],
        opts: ChatOptions | None,
        is_stream: bool,
    ) -> JsonObject:
        """Build the standard OpenAI-compatible request body.

        Provider-specific quirks are applied afterwards by the adapter.
        """
        req: JsonObject = {
            "model": self._model_name,
            "messages": cast(JsonValue, self.convert_messages(messages)),
            "stream": is_stream,
        }
        if is_stream:
            req["stream_options"] = {"include_usage": True}
        if opts is None:
            return req

        if opts.temperature != 0:
            req["temperature"] = opts.temperature
        if opts.top_p > 0:
            req["top_p"] = opts.top_p
        if opts.frequency_penalty > 0:
            req["frequency_penalty"] = opts.frequency_penalty
        if opts.presence_penalty > 0:
            req["presence_penalty"] = opts.presence_penalty
        if opts.max_tokens > 0:
            req["max_tokens"] = opts.max_tokens
        if opts.max_completion_tokens > 0:
            req["max_completion_tokens"] = opts.max_completion_tokens

        if opts.tools:
            tools: list[JsonValue] = []
            for tool in opts.tools:
                function: JsonObject = {
                    "name": tool.function.name,
                    "description": tool.function.description,
                }
                if tool.function.parameters is not None:
                    function["parameters"] = tool.function.parameters
                tools.append({"type": tool.type, "function": function})
            req["tools"] = tools

        if opts.parallel_tool_calls is not None:
            req["parallel_tool_calls"] = opts.parallel_tool_calls

        if opts.tool_choice:
            if opts.tool_choice in ("none", "required", "auto"):
                req["tool_choice"] = opts.tool_choice
            else:
                req["tool_choice"] = {
                    "type": "function",
                    "function": {"name": opts.tool_choice},
                }

        if opts.format is not None:
            req["response_format"] = {"type": "json_object"}
            messages_wire = cast(list[JsonObject], req["messages"])
            if messages_wire:
                last = messages_wire[-1]
                last_content = _as_str(last.get("content"))
                last["content"] = (
                    f"{last_content}\nUse this JSON schema: {json.dumps(opts.format)}"
                )
        return req

    def shaped_request(
        self,
        messages: list[Message],
        opts: ChatOptions | None,
        is_stream: bool,
    ) -> JsonObject:
        """Build the standard request and apply adapter message/shape hooks."""
        req = self.build_chat_completion_request(messages, opts, is_stream)
        wire_messages = cast(list[JsonObject], req["messages"])
        req["messages"] = cast(JsonValue, self._adapter.transform_messages(wire_messages))
        self._adapter.shape_request(req, opts, is_stream)
        return req

    def build_provider_openai_request(
        self,
        body: JsonObject,
        openai_messages: list[JsonObject],
        messages: list[Message],
    ) -> JsonObject:
        """Re-emit the body as a plain map, injecting tool-call metadata."""
        out = dict(body)
        provider_messages: list[JsonObject] = []
        for i, msg in enumerate(openai_messages):
            msg_map = dict(msg)
            if (
                i < len(messages)
                and messages[i].tool_calls
                and isinstance(msg_map.get("tool_calls"), list)
            ):
                tool_calls: list[JsonValue] = []
                raw_tool_calls = cast(list[JsonValue], msg_map["tool_calls"])
                for j, raw_tool_call in enumerate(raw_tool_calls):
                    if not isinstance(raw_tool_call, dict):
                        continue
                    tc_map = dict(raw_tool_call)
                    if j < len(messages[i].tool_calls):
                        self._adapter.inject_tool_call_metadata(
                            tc_map, messages[i].tool_calls[j].provider_metadata
                        )
                    tool_calls.append(tc_map)
                msg_map["tool_calls"] = tool_calls
            provider_messages.append(msg_map)
        out["messages"] = cast(JsonValue, provider_messages)
        return out

    def shape_provider_request(
        self,
        body: JsonObject,
        req: JsonObject,
        messages: list[Message],
    ) -> JsonObject:
        """Apply the adapter's raw-path message transform when forced."""
        if not self._adapter.force_raw_http():
            return body
        return self.build_provider_openai_request(
            body, cast(list[JsonObject], req["messages"]), messages
        )

    def build_outbound(
        self,
        messages: list[Message],
        opts: ChatOptions | None,
        is_stream: bool,
    ) -> tuple[JsonObject, str]:
        """Assemble the final outbound body and endpoint.

        This is the single place that composes the adapter and thinking
        strategy.
        """
        req = self.shaped_request(messages, opts, is_stream)
        thinking = self._thinking_override
        if thinking is None:
            thinking = self._adapter.thinking()
        custom_body, _use_raw = thinking.apply(req, opts, is_stream)
        body: JsonObject = req
        if custom_body is not None:
            body = custom_body
        body = self.shape_provider_request(body, req, messages)
        endpoint = self._adapter.endpoint(self._base_url, self._model_id, is_stream)
        if not endpoint:
            endpoint = self._resolve_default_endpoint()
        return body, endpoint

    def _resolve_default_endpoint(self) -> str:
        """Return ``<base_url>/chat/completions`` (Azure uses deployments)."""
        if self._provider == PROVIDER_AZURE_OPENAI:
            api_version = self._api_version or _DEFAULT_AZURE_API_VERSION
            return (
                f"{self._base_url}/openai/deployments/{self._model_name}"
                f"/chat/completions?api-version={api_version}"
            )
        return f"{self._base_url}/chat/completions"

    # ── Non-streaming path ───────────────────────────────────────────

    async def chat(
        self, messages: list[Message], opts: ChatOptions | None = None
    ) -> ChatResponse:
        """Perform a non-streaming chat round-trip."""
        async with with_llm_timeout(DEFAULT_CHAT_TIMEOUT_SECONDS):
            body, endpoint = self.build_outbound(messages, opts, False)
            return await self.chat_with_raw_http(endpoint, body)

    async def chat_with_raw_http(
        self, endpoint: str, custom_req: JsonObject
    ) -> ChatResponse:
        """Send ``custom_req`` verbatim and parse the completion response."""
        json_data = json.dumps(custom_req, ensure_ascii=False).encode("utf-8")
        if not endpoint:
            endpoint = self._resolve_default_endpoint()
        validate_url_for_ssrf(endpoint)

        headers = {"Content-Type": "application/json"}
        headers.update(self._adapter.auth_headers(self.auth_creds(), json_data))
        headers = apply_custom_headers(headers, self._custom_headers)

        response = await self._client.post(endpoint, content=json_data, headers=headers)
        if response.status_code != 200:
            raise AIProviderError(
                f"API request failed with status {response.status_code}: {response.text}"
            )
        body = response.content
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AIProviderError("decode response: failed to parse response JSON") from exc
        if not isinstance(payload, dict):
            raise AIProviderError("decode response: unexpected response shape")

        result = self.parse_completion_response(payload)
        self.apply_completion_tool_call_metadata(body, result)
        apply_raw_prompt_cache_usage(body, result.usage)
        log_usage(self._model_name, result.usage)
        return result

    def parse_completion_response(self, resp: JsonObject) -> ChatResponse:
        """Parse a non-streaming completion response into ``ChatResponse``."""
        choices = resp.get("choices")
        if not isinstance(choices, list) or not choices:
            raise AIProviderError("no response from API")
        first = choices[0]
        message = _as_object(first.get("message") if isinstance(first, dict) else None)

        content = remove_thinking_content(_as_str(message.get("content")))
        usage = token_usage_from_openai(_as_object(resp.get("usage")), self._provider)
        result = ChatResponse(
            content=content,
            finish_reason=_as_str(
                first.get("finish_reason") if isinstance(first, dict) else None
            ),
            usage=usage,
        )

        raw_tool_calls = message.get("tool_calls")
        if isinstance(raw_tool_calls, list):
            for raw_tool_call in raw_tool_calls:
                if not isinstance(raw_tool_call, dict):
                    continue
                function = _as_object(raw_tool_call.get("function"))
                result.tool_calls.append(
                    LLMToolCall(
                        id=_as_str(raw_tool_call.get("id")),
                        type=_as_str(raw_tool_call.get("type")),
                        function=FunctionCall(
                            name=_as_str(function.get("name")),
                            arguments=_as_str(function.get("arguments")),
                        ),
                    )
                )
        return result

    def apply_completion_tool_call_metadata(
        self, body: bytes, result: ChatResponse
    ) -> None:
        """Attach provider tool-call metadata from the raw response body."""
        if not result.tool_calls:
            return
        try:
            raw = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(raw, dict):
            return
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            return
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        raw_tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not isinstance(raw_tool_calls, list):
            return
        for i, raw_tool_call in enumerate(raw_tool_calls):
            if not isinstance(raw_tool_call, dict):
                continue
            index = raw_tool_call.get("index")
            idx = i if not isinstance(index, int) else index
            if 0 <= idx < len(result.tool_calls):
                result.tool_calls[idx].provider_metadata = (
                    self._adapter.extract_tool_call_metadata(raw_tool_call)
                )

    # ── Streaming path ───────────────────────────────────────────────

    async def chat_stream(
        self, messages: list[Message], opts: ChatOptions | None = None
    ) -> AsyncIterator[StreamResponse]:
        """Stream a chat round-trip, yielding one ``StreamResponse`` per event."""
        async with with_llm_timeout(DEFAULT_STREAM_TIMEOUT_SECONDS):
            body, endpoint = self.build_outbound(messages, opts, True)
            async for event in self.chat_stream_with_raw_http(endpoint, body):
                yield event

    async def chat_stream_with_raw_http(
        self, endpoint: str, custom_req: JsonObject
    ) -> AsyncIterator[StreamResponse]:
        """Send ``custom_req`` over SSE and stream parsed events."""
        json_data = json.dumps(custom_req, ensure_ascii=False).encode("utf-8")
        if not endpoint:
            endpoint = self._resolve_default_endpoint()
        validate_url_for_ssrf(endpoint)

        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        headers.update(self._adapter.auth_headers(self.auth_creds(), json_data))
        headers = apply_custom_headers(headers, self._custom_headers)

        async with self._client.stream(
            "POST", endpoint, content=json_data, headers=headers
        ) as response:
            if response.status_code != 200:
                error_body = (await response.aread()).decode("utf-8", errors="replace")
                raise AIProviderError(
                    f"API request failed with status {response.status_code}: {error_body}"
                )
            async for event in self.process_raw_http_stream(response):
                yield event

    async def process_raw_http_stream(
        self, response: httpx.Response
    ) -> AsyncIterator[StreamResponse]:
        """Parse an SSE response into ``StreamResponse`` events."""
        state = StreamState()
        reader = SSEReader(response.aiter_lines())
        async for event in reader:
            if event.done:
                log_usage(self._model_name, state.usage)
                yield StreamResponse(
                    response_type=ResponseType.ANSWER,
                    done=True,
                    tool_calls=state.build_ordered_tool_calls(),
                    usage=state.usage,
                    finish_reason=state.last_finish_reason,
                )
                return
            if event.data is None:
                continue
            try:
                stream_resp = json.loads(event.data)
            except json.JSONDecodeError:
                continue
            if not isinstance(stream_resp, dict):
                continue

            raw_usage = stream_resp.get("usage")
            if isinstance(raw_usage, dict):
                usage = token_usage_from_openai(raw_usage, self._provider)
                apply_raw_prompt_cache_usage(event.data.encode("utf-8"), usage)
                state.usage = usage

            choices = stream_resp.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                self.apply_stream_tool_call_metadata(stream_resp, state)
                async for out in self.process_stream_delta(choices[0], state):
                    yield out

        log_usage(self._model_name, state.usage)
        yield StreamResponse(
            response_type=ResponseType.ANSWER,
            done=True,
            tool_calls=state.build_ordered_tool_calls(),
            usage=state.usage,
            finish_reason=state.last_finish_reason,
        )

    def apply_stream_tool_call_metadata(
        self, stream_resp: JsonObject, state: StreamState
    ) -> None:
        """Attach provider metadata to in-flight tool calls."""
        choices = stream_resp.get("choices")
        if not isinstance(choices, list) or not choices:
            return
        first = choices[0]
        delta = first.get("delta") if isinstance(first, dict) else None
        raw_tool_calls = delta.get("tool_calls") if isinstance(delta, dict) else None
        if not isinstance(raw_tool_calls, list):
            return
        for raw_tool_call in raw_tool_calls:
            if not isinstance(raw_tool_call, dict):
                continue
            metadata = self._adapter.extract_tool_call_metadata(raw_tool_call)
            if not metadata:
                continue
            index = raw_tool_call.get("index")
            idx = index if isinstance(index, int) else 0
            state.set_tool_call_provider_metadata(idx, metadata)

    async def process_stream_delta(
        self, choice: JsonObject, state: StreamState
    ) -> AsyncIterator[StreamResponse]:
        """Process one streaming delta choice."""
        delta = _as_object(choice.get("delta"))
        finish_reason = _as_str(choice.get("finish_reason"))
        is_done = finish_reason != ""
        if is_done:
            state.last_finish_reason = finish_reason

        raw_tool_calls = delta.get("tool_calls")
        if isinstance(raw_tool_calls, list):
            async for out in self.process_tool_calls_delta(raw_tool_calls, state):
                yield out

        reasoning = _as_str(delta.get("reasoning"))
        if not reasoning:
            reasoning = _as_str(delta.get("reasoning_content"))
        if reasoning:
            yield state.emit(reasoning)

        content = _as_str(delta.get("content"))
        if content:
            done_event = state.finish()
            if done_event is not None:
                yield done_event
            yield StreamResponse(
                response_type=ResponseType.ANSWER,
                content=content,
                done=is_done,
                tool_calls=state.build_ordered_tool_calls(),
                finish_reason=finish_reason,
            )

        if is_done and state.tool_call_map:
            yield StreamResponse(
                response_type=ResponseType.ANSWER,
                done=True,
                tool_calls=state.build_ordered_tool_calls(),
                finish_reason=finish_reason,
            )

        if is_done:
            done_event = state.finish()
            if done_event is not None:
                yield done_event

        if is_done and not content and not state.tool_call_map:
            yield StreamResponse(
                response_type=ResponseType.ANSWER,
                done=True,
                finish_reason=finish_reason,
            )

    async def process_tool_calls_delta(
        self,
        tool_calls: list[JsonValue],
        state: StreamState,
    ) -> AsyncIterator[StreamResponse]:
        """Accumulate a tool-call delta and stream high-level markers."""
        for raw_tool_call in tool_calls:
            if not isinstance(raw_tool_call, dict):
                continue
            index = raw_tool_call.get("index")
            tool_call_index = index if isinstance(index, int) else 0

            entry = state.tool_call_map.get(tool_call_index)
            if entry is None:
                entry = LLMToolCall(
                    id="",
                    type=_as_str(raw_tool_call.get("type")) or "function",
                    function=FunctionCall(name="", arguments=""),
                )
                state.tool_call_map[tool_call_index] = entry

            tool_call_id = _as_str(raw_tool_call.get("id"))
            if tool_call_id:
                entry.id = tool_call_id
            tool_call_type = _as_str(raw_tool_call.get("type"))
            if tool_call_type:
                entry.type = tool_call_type

            function = _as_object(raw_tool_call.get("function"))
            name = _as_str(function.get("name"))
            if name and entry.function.name != name:
                entry.function.name += name

            args = _as_str(function.get("arguments"))
            args_updated = False
            if args:
                entry.function.arguments += args
                args_updated = True

            curr_name = entry.function.name
            if (
                curr_name
                and curr_name == state.last_function_name.get(tool_call_index)
                and args_updated
                and not state.name_notified.get(tool_call_index)
                and entry.id
            ):
                state.name_notified[tool_call_index] = True
                data: JsonObject = {"tool_name": curr_name, "tool_call_id": entry.id}
                yield StreamResponse(
                    response_type=ResponseType.TOOL_CALL,
                    done=False,
                    data=data,
                )
            state.last_function_name[tool_call_index] = curr_name

            if entry.function.name == "thinking" and args_updated:
                extractor = state.field_extractors.get(tool_call_index)
                if extractor is None:
                    extractor = JSONFieldExtractor("thought")
                    state.field_extractors[tool_call_index] = extractor
                thought = extractor.feed(args)
                if thought:
                    thinking_data: JsonObject = {
                        "source": "thinking_tool",
                        "tool_call_id": entry.id,
                    }
                    yield StreamResponse(
                        response_type=ResponseType.THINKING,
                        content=thought,
                        done=False,
                        data=thinking_data,
                    )


__all__ = [
    "RemoteAPIChat",
    "StreamState",
    "remove_thinking_content",
]
