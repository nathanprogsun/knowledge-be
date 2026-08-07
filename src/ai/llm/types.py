"""Chat request and response contract types.

The request side (``Message`` / ``ChatOptions`` / ``Tool`` / ``FunctionDef`` /
``ToolCall`` / ``MessageContentPart`` / ``ImageURL``) mirrors the chat layer's
domain models. The response side (``ChatResponse`` / ``StreamResponse`` /
``TokenUsage`` / ``LLMToolCall`` / ``ToolCallMetadata`` / ``FunctionCall`` /
``ResponseType`` / ``SearchResult``) is a frozen wire contract: every field
name and JSON serialization name must stay aligned with the upstream contract
field-for-field, because the contract-suite PR compares Python output against
fixtures generated from the reference implementation.

Request types are immutable value objects. Response types stay mutable so the
stream aggregation and usage-normalization helpers can update them in place,
mirroring the reference implementation's pointer-receiver methods.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Protocol, TypeAlias, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject, JsonValue

# ── Request types (chat layer domain models) ─────────────────────────


class FunctionDef(BaseModel):
    """One function/tool definition (``{ "name", "description", "parameters" }``)."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    parameters: JsonValue | None = None


class Tool(BaseModel):
    """A tool entry inside ``ChatOptions.tools``."""

    model_config = ConfigDict(frozen=True)

    type: str = "function"
    function: FunctionDef


class ChatOptions(BaseModel):
    """Sampling and behavior options for one chat call.

    The scalar sampling fields are never omitted (the reference struct has no
    ``omitempty`` on them), so they serialize as zero values when unset.
    """

    model_config = ConfigDict(frozen=True)

    temperature: float = 0.0
    top_p: float = 0.0
    seed: int = 0
    max_tokens: int = 0
    max_completion_tokens: int = 0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    thinking: bool | None = None
    tools: list[Tool] = Field(default_factory=list)
    tool_choice: str = ""
    parallel_tool_calls: bool | None = None
    format: JsonValue | None = None


class ImageURL(BaseModel):
    """Image URL reference inside a multi-content part."""

    model_config = ConfigDict(frozen=True)

    url: str
    detail: str = ""


class MessageContentPart(BaseModel):
    """One part of a multi-content message (text or image)."""

    model_config = ConfigDict(frozen=True)

    type: str
    text: str = ""
    image_url: ImageURL | None = None


#: Provider-specific state that round-trips with an assistant tool call.
ToolCallMetadata: TypeAlias = dict[str, JsonValue]


class FunctionCall(BaseModel):
    """Function details attached to a tool call."""

    name: str
    arguments: str = ""


class ToolCall(BaseModel):
    """A tool call embedded in an assistant message."""

    model_config = ConfigDict(frozen=True)

    id: str
    type: str = "function"
    function: FunctionCall
    provider_metadata: ToolCallMetadata = Field(default_factory=dict)


class Message(BaseModel):
    """One chat message.

    ``images`` carries image URLs for multimodal requests and applies only to
    the current user turn; ``reasoning_content`` round-trips the previous
    assistant turn's reasoning for providers that require it.
    """

    model_config = ConfigDict(frozen=True)

    role: str
    content: str = ""
    multi_content: list[MessageContentPart] = Field(default_factory=list)
    name: str = ""
    tool_call_id: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    reasoning_content: str = ""


class ChatConfig(BaseModel):
    """Runtime configuration for one chat client.

    ``max_concurrency`` caps concurrent background calls to this model (0
    falls back to the process-wide default). ``app_id`` / ``app_secret`` are
    the already-decrypted managed-cloud credentials.
    """

    model_config = ConfigDict(frozen=True)

    source: str
    base_url: str = ""
    model_name: str = ""
    api_key: str = ""
    model_id: str = ""
    provider: str = ""
    max_concurrency: int = 0
    extra_config: dict[str, str] | None = None
    custom_headers: dict[str, str] | None = None
    app_id: str = ""
    app_secret: str = ""


@runtime_checkable
class Chat(Protocol):
    """Interface implemented by every chat client."""

    async def chat(
        self, messages: list[Message], opts: ChatOptions | None = None
    ) -> ChatResponse:
        """Perform a non-streaming chat round-trip."""
        ...

    def chat_stream(
        self, messages: list[Message], opts: ChatOptions | None = None
    ) -> AsyncIterator[StreamResponse]:
        """Stream a chat round-trip as an async iterator of events."""
        ...

    def get_model_name(self) -> str:
        """Return the model name served by this client."""
        ...

    def get_model_id(self) -> str:
        """Return the model id served by this client."""
        ...


# ── Response contract (frozen wire shape) ────────────────────────────


class PromptCacheStatus(StrEnum):
    """Distinguishes a real cache miss from providers that report nothing."""

    UNSUPPORTED = "unsupported"
    UNREPORTED = "unreported"
    MISS = "miss"
    HIT = "hit"


class TokenUsage(BaseModel):
    """Token consumption statistics returned by the model API.

    ``cached_tokens`` is the legacy alias for ``cache_read_tokens`` and stays
    on the wire for existing consumers. The ``set_prompt_cache_usage`` /
    ``mark_prompt_cache_unsupported`` helpers normalize provider-specific
    counters into this shared shape.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cache_miss_tokens: int = 0
    cache_reported: bool = False
    cache_status: PromptCacheStatus | None = None

    def set_prompt_cache_usage(
        self, read: int, write: int, miss: int, reported: bool
    ) -> None:
        """Normalize provider cache counters into the shared usage model.

        ``read`` / ``write`` / ``miss`` are descriptive subsets of the
        provider's total input-token count and are never added to it.
        """
        read = max(read, 0)
        write = max(write, 0)
        miss = max(miss, 0)
        self.cached_tokens = read
        self.cache_read_tokens = read
        self.cache_write_tokens = write
        self.cache_miss_tokens = miss
        self.cache_reported = reported
        if not reported:
            self.cache_status = PromptCacheStatus.UNREPORTED
        elif read > 0:
            self.cache_status = PromptCacheStatus.HIT
        else:
            self.cache_status = PromptCacheStatus.MISS

    def mark_prompt_cache_unsupported(self) -> None:
        """Mark a provider/model path that cannot report prompt-cache usage."""
        self.set_prompt_cache_usage(0, 0, 0, False)
        self.cache_status = PromptCacheStatus.UNSUPPORTED


class LLMToolCall(BaseModel):
    """A function/tool call produced by the model.

    ``model_arguments`` / ``argument_resolution`` / ``unresolved_handles`` are
    request-local observability state and must never be sent to a provider or
    persisted, so they are excluded from serialization.
    """

    id: str = ""
    type: str = "function"
    function: FunctionCall = Field(default_factory=lambda: FunctionCall(name=""))
    provider_metadata: ToolCallMetadata = Field(default_factory=dict)
    model_arguments: str = Field(default="", exclude=True)
    argument_resolution: str = Field(default="", exclude=True)
    unresolved_handles: list[str] = Field(default_factory=list, exclude=True)


class ChatResponse(BaseModel):
    """Non-streaming chat response.

    ``answer_streamed`` / ``answer_event_id`` are transient stream state,
    never persisted.
    """

    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[LLMToolCall] = Field(default_factory=list)
    finish_reason: str = ""
    usage: TokenUsage = Field(default_factory=TokenUsage)
    answer_streamed: bool = Field(default=False, exclude=True)
    answer_event_id: str = Field(default="", exclude=True)


class ResponseType(StrEnum):
    """Stream event discriminator."""

    ANSWER = "answer"
    REFERENCES = "references"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    REFLECTION = "reflection"
    SESSION_TITLE = "session_title"
    AGENT_QUERY = "agent_query"
    COMPLETE = "complete"
    TOOL_APPROVAL_REQUIRED = "tool_approval_required"
    TOOL_APPROVAL_RESOLVED = "tool_approval_resolved"
    MCP_OAUTH_REQUIRED = "mcp_oauth_required"
    MCP_OAUTH_RESOLVED = "mcp_oauth_resolved"


class SearchResult(BaseModel):
    """One knowledge-search hit attached to a stream response."""

    id: str = ""
    content: str = ""
    knowledge_id: str = ""
    chunk_index: int = 0
    knowledge_title: str = ""
    start_at: int = 0
    end_at: int = 0
    seq: int = 0
    score: float = 0.0
    match_type: int = 0
    sub_chunk_id: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    chunk_type: str = ""
    parent_chunk_id: str = ""
    image_info: str = ""
    knowledge_filename: str = ""
    knowledge_source: str = ""
    knowledge_channel: str = ""
    chunk_metadata: JsonObject | None = None
    matched_content: str = ""
    knowledge_description: str = ""
    knowledge_custom_metadata: str = ""
    knowledge_base_id: str = ""


#: ``[]*SearchResult`` equivalent carried by ``StreamResponse``.
References: TypeAlias = list[SearchResult]


class StreamResponse(BaseModel):
    """One event in a streaming chat response."""

    id: str = ""
    response_type: ResponseType = ResponseType.ANSWER
    content: str = ""
    done: bool = False
    knowledge_references: References | None = None
    session_id: str = ""
    assistant_message_id: str = ""
    tool_calls: list[LLMToolCall] = Field(default_factory=list)
    data: JsonObject | None = None
    usage: TokenUsage | None = None
    finish_reason: str = ""


__all__ = [
    "Chat",
    "ChatConfig",
    "ChatOptions",
    "ChatResponse",
    "FunctionCall",
    "FunctionDef",
    "ImageURL",
    "LLMToolCall",
    "Message",
    "MessageContentPart",
    "PromptCacheStatus",
    "References",
    "ResponseType",
    "SearchResult",
    "StreamResponse",
    "TokenUsage",
    "Tool",
    "ToolCall",
    "ToolCallMetadata",
]
