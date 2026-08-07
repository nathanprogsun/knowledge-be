"""OpenAI-compatible chat completion parsing and stream parsing.

Maps the stream side of the OpenAI wire protocol: non-streaming completion
parsing (``parseCompletionResponse`` / ``applyCompletionToolCallMetadata``),
the streaming delta loop (``processStream`` / ``processRawHTTPStream`` /
``processStreamDelta`` / ``processToolCallsDelta``), the per-stream
accumulator (``streamState``) and the thinking-content stripper
(``removeThinkingContent``). The functions take the owning ``RemoteAPIChat``
instance where they need its provider / adapter context so ``remote_api`` can
delegate to them while keeping its public API unchanged.

The reference implementation's SDK-backed stream loop has no SDK counterpart
here, so ``process_stream`` consumes already-parsed chunk payloads and the
raw HTTP path feeds it the SSE data lines.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from typing import Protocol

import httpx

from src.ai.llm.json_field_extractor import JSONFieldExtractor
from src.ai.llm.openai import _as_object, _as_str
from src.ai.llm.prompt_cache import (
    apply_raw_prompt_cache_usage,
    token_usage_from_openai,
)
from src.ai.llm.providers import ProviderAdapter
from src.ai.llm.sse_reader import SSEReader
from src.ai.llm.stream_emit import ThinkingEmitter
from src.ai.llm.types import (
    ChatResponse,
    FunctionCall,
    LLMToolCall,
    ResponseType,
    StreamResponse,
    TokenUsage,
    ToolCallMetadata,
)
from src.ai.llm.usage import log_usage
from src.common.exception import AIProviderError
from src.common.json import JsonObject, JsonValue


class OpenAIStreamChat(Protocol):
    """The ``RemoteAPIChat`` surface the stream parser needs."""

    _model_name: str
    _provider: str
    _adapter: ProviderAdapter


@dataclass(frozen=True)
class StreamChunk:
    """One parsed stream payload plus its raw bytes for cache accounting."""

    payload: JsonObject
    raw: bytes | None = None


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

    def set_tool_call_provider_metadata(self, index: int, metadata: ToolCallMetadata) -> None:
        """Attach provider metadata to the tool call at ``index``."""
        if not metadata:
            return
        entry = self.tool_call_map.get(index)
        if entry is None:
            entry = LLMToolCall(
                id="", type="function", function=FunctionCall(name="", arguments="")
            )
            self.tool_call_map[index] = entry
        entry.provider_metadata = metadata


def parse_completion_response(chat: OpenAIStreamChat, resp: JsonObject) -> ChatResponse:
    """Parse a non-streaming completion response into ``ChatResponse``."""
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AIProviderError("no response from API")
    first = choices[0]
    message = _as_object(first.get("message") if isinstance(first, dict) else None)

    content = remove_thinking_content(_as_str(message.get("content")))
    usage = token_usage_from_openai(_as_object(resp.get("usage")), chat._provider)
    result = ChatResponse(
        content=content,
        finish_reason=_as_str(first.get("finish_reason") if isinstance(first, dict) else None),
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
    chat: OpenAIStreamChat, body: bytes, result: ChatResponse
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
            result.tool_calls[idx].provider_metadata = chat._adapter.extract_tool_call_metadata(
                raw_tool_call
            )


def apply_stream_tool_call_metadata(
    chat: OpenAIStreamChat, stream_resp: JsonObject, state: StreamState
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
        metadata = chat._adapter.extract_tool_call_metadata(raw_tool_call)
        if not metadata:
            continue
        index = raw_tool_call.get("index")
        idx = index if isinstance(index, int) else 0
        state.set_tool_call_provider_metadata(idx, metadata)


async def process_stream(
    chat: OpenAIStreamChat,
    chunks: AsyncIterable[StreamChunk],
) -> AsyncIterator[StreamResponse]:
    """Process parsed stream chunks into ``StreamResponse`` events.

    Mirrors the SDK-backed stream loop of the reference implementation: it
    consumes already-parsed chunks (usage capture + delta handling) and yields
    one terminal ``done`` event when the chunk stream is exhausted. The raw
    HTTP path feeds it the SSE payloads via :class:`StreamChunk`.
    """
    state = StreamState()
    async for chunk in chunks:
        stream_resp = chunk.payload
        raw_usage = stream_resp.get("usage")
        if isinstance(raw_usage, dict):
            usage = token_usage_from_openai(raw_usage, chat._provider)
            if chunk.raw is not None:
                apply_raw_prompt_cache_usage(chunk.raw, usage)
            state.usage = usage

        choices = stream_resp.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            apply_stream_tool_call_metadata(chat, stream_resp, state)
            async for out in process_stream_delta(choices[0], state):
                yield out

    log_usage(chat._model_name, state.usage)
    yield StreamResponse(
        response_type=ResponseType.ANSWER,
        done=True,
        tool_calls=state.build_ordered_tool_calls(),
        usage=state.usage,
        finish_reason=state.last_finish_reason,
    )


async def process_raw_http_stream(
    chat: OpenAIStreamChat, response: httpx.Response
) -> AsyncIterator[StreamResponse]:
    """Parse an SSE response into ``StreamResponse`` events."""
    reader = SSEReader(response.aiter_lines())

    async def chunks() -> AsyncIterator[StreamChunk]:
        async for event in reader:
            if event.done:
                return
            if event.data is None:
                continue
            try:
                stream_resp = json.loads(event.data)
            except json.JSONDecodeError:
                continue
            if not isinstance(stream_resp, dict):
                continue
            yield StreamChunk(payload=stream_resp, raw=event.data.encode("utf-8"))

    async for out in process_stream(chat, chunks()):
        yield out


async def process_stream_delta(
    choice: JsonObject, state: StreamState
) -> AsyncIterator[StreamResponse]:
    """Process one streaming delta choice."""
    delta = _as_object(choice.get("delta"))
    finish_reason = _as_str(choice.get("finish_reason"))
    is_done = finish_reason != ""
    if is_done:
        state.last_finish_reason = finish_reason

    raw_tool_calls = delta.get("tool_calls")
    if isinstance(raw_tool_calls, list):
        async for out in process_tool_calls_delta(raw_tool_calls, state):
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
    "OpenAIStreamChat",
    "StreamChunk",
    "StreamState",
    "apply_completion_tool_call_metadata",
    "apply_stream_tool_call_metadata",
    "parse_completion_response",
    "process_raw_http_stream",
    "process_stream",
    "process_stream_delta",
    "process_tool_calls_delta",
    "remove_thinking_content",
]
