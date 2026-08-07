"""OpenAI-compatible chat request building.

Maps the request side of the OpenAI wire protocol: message conversion
(``ConvertMessages``), the standard Chat Completions request body
(``BuildChatCompletionRequest``), and the provider message-shaping hooks
(``buildProviderOpenAIRequest`` / ``shapeProviderRequest``). The functions
take the owning ``RemoteAPIChat`` instance as the receiver so ``remote_api``
can delegate to them while keeping its public API unchanged.

The reference implementation routes some calls through an OpenAI SDK client;
this port has no SDK dependency, so request bodies are built as plain JSON
maps and the adapter + thinking composition happens in ``remote_api``.
"""

from __future__ import annotations

import json
from typing import Protocol, cast

from src.ai.llm.image_resolve import resolve_image_url_for_llm
from src.ai.llm.providers import ProviderAdapter
from src.ai.llm.types import ChatOptions, Message
from src.common.json import JsonObject, JsonValue


class OpenAIRequestChat(Protocol):
    """The ``RemoteAPIChat`` surface the request builder needs."""

    _model_name: str
    _adapter: ProviderAdapter


def _as_str(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


def _as_object(value: JsonValue) -> JsonObject:
    return value if isinstance(value, dict) else {}


def convert_messages(messages: list[Message]) -> list[JsonObject]:
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
    chat: OpenAIRequestChat,
    messages: list[Message],
    opts: ChatOptions | None,
    is_stream: bool,
) -> JsonObject:
    """Build the standard OpenAI-compatible request body.

    Provider-specific quirks are applied afterwards by the adapter.
    """
    req: JsonObject = {
        "model": chat._model_name,
        "messages": cast(JsonValue, convert_messages(messages)),
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
            last["content"] = f"{last_content}\nUse this JSON schema: {json.dumps(opts.format)}"
    return req


def build_provider_openai_request(
    chat: OpenAIRequestChat,
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
                    chat._adapter.inject_tool_call_metadata(
                        tc_map, messages[i].tool_calls[j].provider_metadata
                    )
                tool_calls.append(tc_map)
            msg_map["tool_calls"] = tool_calls
        provider_messages.append(msg_map)
    out["messages"] = cast(JsonValue, provider_messages)
    return out


def shape_provider_request(
    chat: OpenAIRequestChat,
    body: JsonObject,
    req: JsonObject,
    messages: list[Message],
) -> JsonObject:
    """Apply the adapter's raw-path message transform when forced."""
    if not chat._adapter.force_raw_http():
        return body
    return build_provider_openai_request(
        chat, body, cast(list[JsonObject], req["messages"]), messages
    )


__all__ = [
    "OpenAIRequestChat",
    "build_chat_completion_request",
    "build_provider_openai_request",
    "convert_messages",
    "shape_provider_request",
]
