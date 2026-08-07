"""Local Ollama chat client (mirrors the upstream ``ollama.go`` chat).

``OllamaChat`` adapts the shared chat request / response types to the local
Ollama ``/api/chat`` endpoint. Non-streaming calls delegate the request to
``OllamaService.chat``; streaming calls consume the SSE frame stream from
``OllamaService.chat_stream`` and project each frame onto the shared
``StreamResponse`` shape, including reasoning chunks and the
non-incremental tool calls Ollama returns.

Tool definitions and calls are converted with the same string↔int id
mapping as the reference: the shared tool-call ``id`` string round-trips
as the numeric ``index`` field on the Ollama wire.
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from typing import cast

from src.ai.llm.image_resolve import resolve_image_url_for_ollama
from src.ai.llm.stream_emit import ThinkingEmitter
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
    Tool,
    ToolCall,
)
from src.ai.llm.usage import log_usage
from src.ai.utils.ollama_service import OllamaService
from src.common.exception import AIProviderError
from src.common.json import JsonObject, JsonValue


def new_ollama_chat(config: ChatConfig, ollama_service: OllamaService | None) -> OllamaChat:
    """Create an Ollama-backed chat client for ``config``.

    ``ollama_service`` is required: the local route always has a service
    instance available, so a missing one is a wiring error.
    """
    if ollama_service is None:
        raise AIProviderError(
            "Ollama service is required for a local chat model",
            code="ollama.chat_service_missing",
        )
    return OllamaChat(
        model_name=config.model_name,
        model_id=config.model_id,
        ollama_service=ollama_service,
    )


class OllamaChat:
    """Chat client backed by a local Ollama service."""

    def __init__(
        self,
        *,
        model_name: str,
        model_id: str,
        ollama_service: OllamaService,
    ) -> None:
        self._model_name = model_name
        self._model_id = model_id
        self._ollama_service = ollama_service

    def get_model_name(self) -> str:
        return self._model_name

    def get_model_id(self) -> str:
        return self._model_id

    # ── Request building ─────────────────────────────────────────────

    def convert_messages(self, messages: list[Message]) -> list[JsonObject]:
        """Convert ``Message`` objects into the Ollama wire message shape."""
        converted: list[JsonObject] = []
        for msg in messages:
            entry: JsonObject = {"role": msg.role, "content": msg.content}
            tool_calls = self.tool_call_from(msg.tool_calls)
            if tool_calls:
                entry["tool_calls"] = cast(JsonValue, tool_calls)
            if msg.role == "tool":
                entry["tool_name"] = msg.name
            if msg.images and msg.role == "user":
                images: list[str] = []
                for image_url in msg.images:
                    data = resolve_image_url_for_ollama(image_url)
                    if data is not None:
                        images.append(base64.b64encode(data).decode("ascii"))
                if images:
                    entry["images"] = cast(JsonValue, images)
            converted.append(entry)
        return converted

    def build_chat_request(
        self, messages: list[Message], opts: ChatOptions | None, is_stream: bool
    ) -> JsonObject:
        """Build the ``/api/chat`` request body for one round-trip."""
        options: dict[str, JsonValue] = {}
        request: JsonObject = {
            "model": self._model_name,
            "messages": cast(JsonValue, self.convert_messages(messages)),
            "stream": is_stream,
            "options": options,
        }
        if opts is None:
            return request
        options["temperature"] = opts.temperature
        if opts.top_p > 0:
            options["top_p"] = opts.top_p
        if opts.max_tokens > 0:
            options["num_predict"] = opts.max_tokens
        if opts.thinking is not None:
            request["think"] = opts.thinking
        if opts.format is not None:
            request["format"] = opts.format
        if opts.tools:
            request["tools"] = cast(JsonValue, self.tool_from(opts.tools))
        return request

    # ── Tool conversion ──────────────────────────────────────────────

    def tool_from(self, tools: list[Tool]) -> list[JsonObject]:
        """Convert shared tool definitions into the Ollama tool shape."""
        converted: list[JsonObject] = []
        for tool in tools:
            function: JsonObject = {
                "name": tool.function.name,
                "description": tool.function.description,
            }
            if tool.function.parameters is not None:
                function["parameters"] = tool.function.parameters
            converted.append({"type": tool.type, "function": function})
        return converted

    def tool_call_from(self, tool_calls: list[ToolCall]) -> list[JsonObject]:
        """Convert outbound assistant tool calls into the Ollama shape.

        The Ollama wire carries the tool-call ``id`` as a numeric
        ``index``, so the shared string id is parsed back to an int.
        """
        converted: list[JsonObject] = []
        for tool_call in tool_calls:
            arguments: JsonValue = {}
            if tool_call.function.arguments:
                try:
                    parsed = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    arguments = parsed
            converted.append(
                {
                    "function": {
                        "index": tools2i(tool_call.id),
                        "name": tool_call.function.name,
                        "arguments": arguments,
                    }
                }
            )
        return converted

    def tool_call_to(self, tool_calls: list[JsonValue]) -> list[LLMToolCall]:
        """Convert inbound Ollama tool calls into shared ``LLMToolCall``."""
        converted: list[LLMToolCall] = []
        for raw in tool_calls:
            if not isinstance(raw, dict):
                continue
            function = raw.get("function")
            function = function if isinstance(function, dict) else {}
            name = function.get("name")
            index = function.get("index")
            converted.append(
                LLMToolCall(
                    id=tooli2s(index) if isinstance(index, int) else "0",
                    type="function",
                    function=FunctionCall(
                        name=name if isinstance(name, str) else "",
                        arguments=_dump_arguments(function.get("arguments")),
                    ),
                )
            )
        return converted

    # ── Non-streaming path ───────────────────────────────────────────

    async def chat(self, messages: list[Message], opts: ChatOptions | None = None) -> ChatResponse:
        """Perform a non-streaming chat round-trip."""
        await self.ensure_model_available()
        request = self.build_chat_request(messages, opts, False)
        try:
            response = await self._ollama_service.chat(request)
        except Exception as exc:
            raise AIProviderError(
                f"Ollama chat request failed: {exc}",
                code="ollama.chat_failed",
            ) from exc
        return self.parse_chat_response(response)

    def parse_chat_response(self, response: JsonObject) -> ChatResponse:
        """Project one non-streaming ``/api/chat`` response."""
        message = response.get("message")
        message = message if isinstance(message, dict) else {}
        content = message.get("content")
        content = content if isinstance(content, str) else ""
        thinking = message.get("thinking")
        thinking = thinking if isinstance(thinking, str) else ""
        # Reasoning models that answered without a thinking parameter still
        # surface their reasoning in ``message.thinking``; use it as the
        # answer when no regular content was returned.
        if not content and thinking:
            content = thinking
        tool_calls = self.tool_call_to(_as_list(message.get("tool_calls")))

        prompt_tokens = 0
        completion_tokens = 0
        eval_count = response.get("eval_count")
        if isinstance(eval_count, int) and eval_count > 0:
            prompt_eval_count = response.get("prompt_eval_count")
            prompt_tokens = prompt_eval_count if isinstance(prompt_eval_count, int) else 0
            completion_tokens = eval_count - prompt_tokens
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        usage.mark_prompt_cache_unsupported()
        log_usage(self._model_name, usage)
        return ChatResponse(content=content, tool_calls=tool_calls, usage=usage)

    # ── Streaming path ───────────────────────────────────────────────

    async def chat_stream(
        self, messages: list[Message], opts: ChatOptions | None = None
    ) -> AsyncIterator[StreamResponse]:
        """Stream a chat round-trip, yielding one ``StreamResponse`` per event."""
        await self.ensure_model_available()
        request = self.build_chat_request(messages, opts, True)
        emitter = ThinkingEmitter()
        try:
            async for frame in self._ollama_service.chat_stream(request):
                async for event in self.process_stream_frame(frame, emitter):
                    yield event
        except Exception as exc:
            yield StreamResponse(
                response_type=ResponseType.ERROR,
                content=str(exc),
                done=True,
            )

    async def process_stream_frame(
        self, frame: JsonObject, emitter: ThinkingEmitter
    ) -> AsyncIterator[StreamResponse]:
        """Project one streamed ``/api/chat`` frame into events."""
        message = frame.get("message")
        message = message if isinstance(message, dict) else {}
        thinking = message.get("thinking")
        if isinstance(thinking, str) and thinking:
            yield emitter.emit(thinking)

        content = message.get("content")
        if isinstance(content, str) and content:
            done_event = emitter.finish()
            if done_event is not None:
                yield done_event
            yield StreamResponse(
                response_type=ResponseType.ANSWER,
                content=content,
                done=False,
            )

        raw_tool_calls = message.get("tool_calls")
        if isinstance(raw_tool_calls, list) and raw_tool_calls:
            yield StreamResponse(
                response_type=ResponseType.TOOL_CALL,
                tool_calls=self.tool_call_to(raw_tool_calls),
                done=False,
            )
            for raw_tool_call in raw_tool_calls:
                if not isinstance(raw_tool_call, dict):
                    continue
                function = raw_tool_call.get("function")
                function = function if isinstance(function, dict) else {}
                if function.get("name") != "thinking":
                    continue
                arguments = function.get("arguments")
                if not isinstance(arguments, dict):
                    continue
                thought = arguments.get("thought")
                if isinstance(thought, str) and thought:
                    index = function.get("index")
                    yield StreamResponse(
                        response_type=ResponseType.THINKING,
                        content=thought,
                        done=False,
                        data={
                            "source": "thinking_tool",
                            "tool_call_id": tooli2s(index) if isinstance(index, int) else "0",
                        },
                    )

        if frame.get("done") is True:
            usage: TokenUsage | None = None
            prompt_eval_count = frame.get("prompt_eval_count")
            eval_count = frame.get("eval_count")
            if (
                isinstance(prompt_eval_count, int)
                and isinstance(eval_count, int)
                and (prompt_eval_count > 0 or eval_count > 0)
            ):
                usage = TokenUsage(
                    prompt_tokens=prompt_eval_count,
                    completion_tokens=eval_count,
                    total_tokens=prompt_eval_count + eval_count,
                )
                usage.mark_prompt_cache_unsupported()
            log_usage(self._model_name, usage)
            yield StreamResponse(
                response_type=ResponseType.ANSWER,
                done=True,
                usage=usage,
            )

    # ── Service plumbing ─────────────────────────────────────────────

    async def ensure_model_available(self) -> None:
        """Pull the model when it is not installed (delegates to the service)."""
        await self._ollama_service.ensure_model_available(self._model_name)


def _as_list(value: JsonValue | None) -> list[JsonValue]:
    return value if isinstance(value, list) else []


def _dump_arguments(value: JsonValue | None) -> str:
    """Serialize tool-call arguments to compact JSON (``json.Marshal`` parity)."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return "{}"


def tools2i(tool_id: str) -> int:
    """Parse a tool-call id into the Ollama ``index`` (reference ``tools2i``)."""
    try:
        return int(tool_id)
    except ValueError:
        return 0


def tooli2s(index: int) -> str:
    """Format an Ollama ``index`` as a tool-call id (reference ``tooli2s``)."""
    return str(index)


__all__ = ["OllamaChat", "new_ollama_chat", "tooli2s", "tools2i"]
