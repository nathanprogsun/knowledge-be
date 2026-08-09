"""Pipeline event and step types — the shared contract pipeline steps consume.

Defines the vocabulary of a chat-pipeline run: the ``EventType`` stages a
plugin can register for, the ``QueryIntent`` classification, the sampling
configuration (``SummaryConfig``), the search-hit / history step payloads,
and the dynamic ``PipelineBuilder`` that assembles a stage list at the
request entry point.

Every value type here is frozen. The only mutable object in the chat
domain is the run carrier ``PipelineContext`` (see ``context.py``), which
plugins read and write as the chain progresses — mirroring the upstream
contract where steps share one in-place carrier.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Protocol, TypeAlias, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from src.ai.retrieval.types import MatchType
from src.common.json import JsonObject


@runtime_checkable
class Context(Protocol):
    """Opaque execution context threaded through pipeline plugins.

    Carries request-scoped handles (cancellation, tracing, background
    markers); the pipeline never inspects its contents.
    """


class EventType(StrEnum):
    """Stage identifiers in the chat pipeline (upstream ``EventType``).

    Event names are string-valued so arbitrary custom events are tolerated
    by the engine; these constants are the sanctioned pipeline stages.
    """

    LOAD_HISTORY = "load_history"
    QUERY_UNDERSTAND = "query_understand"
    CHUNK_SEARCH = "chunk_search"
    CHUNK_SEARCH_PARALLEL = "chunk_search_parallel"
    ENTITY_SEARCH = "entity_search"
    CHUNK_RERANK = "chunk_rerank"
    WEB_FETCH = "web_fetch"
    CHUNK_MERGE = "chunk_merge"
    DATA_ANALYSIS = "data_analysis"
    INTO_CHAT_MESSAGE = "into_chat_message"
    CHAT_COMPLETION = "chat_completion"
    CHAT_COMPLETION_STREAM = "chat_completion_stream"
    FILTER_TOP_K = "filter_top_k"


class QueryIntent(StrEnum):
    """Classified intent of a user query (upstream ``QueryIntent``)."""

    KB_SEARCH = "kb_search"
    WEB_SEARCH = "web_search"
    GREETING = "greeting"
    CHITCHAT = "chitchat"
    FOLLOW_UP = "follow_up"
    IMAGE_ONLY = "image_only"
    DOC_ONLY = "doc_only"
    SUMMARIZE = "summarize"
    CLARIFICATION = "clarification"

    def needs_kb_retrieval(self) -> bool:
        """Return whether this intent requires knowledge-base retrieval.

        The unclassified value (``None``) is handled by the run carrier and
        defaults to retrieval for safety.
        """
        return self in {
            QueryIntent.KB_SEARCH,
            QueryIntent.CLARIFICATION,
            QueryIntent.SUMMARIZE,
        }


class FallbackStrategy(StrEnum):
    """Fallback strategy when a turn cannot be answered (upstream
    ``FallbackStrategy``)."""

    FIXED = "fixed"
    MODEL = "model"


class SearchTargetType(StrEnum):
    """Kind of a search target (upstream ``SearchTargetType``)."""

    KNOWLEDGE_BASE = "knowledge_base"
    KNOWLEDGE = "knowledge"


class SummaryConfig(BaseModel):
    """Sampling + prompt configuration for one turn (upstream
    ``SummaryConfig``)."""

    model_config = ConfigDict(frozen=True)

    max_tokens: int = 0
    repeat_penalty: float = 0.0
    top_k: int = 0
    top_p: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    prompt: str = ""
    context_template: str = ""
    no_match_prefix: str = ""
    temperature: float = 0.0
    seed: int = 0
    max_completion_tokens: int = 0
    thinking: bool | None = None


class SearchTarget(BaseModel):
    """One retrieval scope: a whole KB or specific knowledge within a KB."""

    model_config = ConfigDict(frozen=True)

    type: SearchTargetType = SearchTargetType.KNOWLEDGE_BASE
    knowledge_base_id: str = ""
    tenant_id: int = 0
    knowledge_ids: list[str] = Field(default_factory=list)
    tag_ids: list[str] = Field(default_factory=list)
    scope_tag_ids: list[str] = Field(default_factory=list)
    disable_recall_thresholds: bool = False


class SearchResult(BaseModel):
    """One search hit carried between pipeline steps.

    This is the pipeline-domain projection of a search hit (upstream
    ``SearchResult``); it is the payload that search, rerank and merge
    steps exchange. The retrieval layer exposes its own copy of the same
    shape at its boundary; a step converts across the two when it must.
    """

    model_config = ConfigDict(frozen=True)

    id: str = ""
    content: str = ""
    knowledge_id: str = ""
    chunk_index: int = 0
    knowledge_title: str = ""
    start_at: int = 0
    end_at: int = 0
    seq: int = 0
    score: float = 0.0
    match_type: MatchType = MatchType.EMBEDDING
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


References: TypeAlias = list[SearchResult]


class History(BaseModel):
    """One prior Q&A round replayed into the current turn (upstream
    ``History``)."""

    model_config = ConfigDict(frozen=True)

    query: str = ""
    answer: str = ""
    created_at: datetime | None = None
    references: References = Field(default_factory=list)


class MessageImage(BaseModel):
    """An image attached to a chat message."""

    model_config = ConfigDict(frozen=True)

    url: str = ""
    caption: str = ""


class MessageAttachment(BaseModel):
    """A file attachment carried by a chat message.

    ``url`` is an internal storage locator and is excluded from any
    serialization (upstream keeps it off the wire with ``json:"-"``).
    """

    model_config = ConfigDict(frozen=True)

    id: str = ""
    url: str = Field(default="", exclude=True)
    file_name: str = ""
    file_type: str = ""
    file_size: int = 0
    content: str = ""
    is_truncated: bool = False
    line_count: int = 0
    content_mode: str = ""
    token_count: int = 0
    selected_chunks: int = 0
    total_chunks: int = 0


class GraphNode(BaseModel):
    """A node in an extracted knowledge graph."""

    model_config = ConfigDict(frozen=True)

    name: str = ""
    chunks: list[str] = Field(default_factory=list)
    attributes: list[str] = Field(default_factory=list)


class GraphRelation(BaseModel):
    """A directed relation between two graph nodes."""

    model_config = ConfigDict(frozen=True)

    node1: str = ""
    node2: str = ""
    type: str = ""


class GraphData(BaseModel):
    """One extracted graph fragment (upstream ``GraphData``)."""

    model_config = ConfigDict(frozen=True)

    text: str = ""
    node: list[GraphNode] = Field(default_factory=list)
    relation: list[GraphRelation] = Field(default_factory=list)


class TokenUsage(BaseModel):
    """Token accounting for a model completion (upstream ``TokenUsage``)."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cache_miss_tokens: int = 0
    cache_reported: bool = False
    cache_status: str = ""


class ChatResponse(BaseModel):
    """A model completion produced by the completion step."""

    model_config = ConfigDict(frozen=True)

    content: str = ""
    reasoning_content: str = ""
    finish_reason: str = ""
    usage: TokenUsage = Field(default_factory=TokenUsage)


class PipelineBuilder:
    """Dynamically assemble a pipeline as an ordered list of stages.

    Mirrors the upstream builder: ``add`` appends unconditionally,
    ``add_if`` appends only when its condition holds, and ``build``
    returns a fresh list (the builder must not be reused after build).
    """

    def __init__(self) -> None:
        self._stages: list[EventType] = []

    def add(self, *stages: EventType) -> PipelineBuilder:
        """Append one or more stages unconditionally."""
        self._stages.extend(stages)
        return self

    def add_if(self, condition: bool, *stages: EventType) -> PipelineBuilder:
        """Append stages only when ``condition`` is true."""
        if condition:
            self._stages.extend(stages)
        return self

    def build(self) -> list[EventType]:
        """Return the assembled stage list as a fresh copy."""
        return list(self._stages)


PIPELINE_MODES: dict[str, tuple[EventType, ...]] = {
    "chat": (EventType.CHAT_COMPLETION,),
    "chat_stream": (EventType.CHAT_COMPLETION_STREAM,),
    "chat_history_stream": (
        EventType.LOAD_HISTORY,
        EventType.CHAT_COMPLETION_STREAM,
    ),
    "rag": (
        EventType.CHUNK_SEARCH,
        EventType.CHUNK_RERANK,
        EventType.CHUNK_MERGE,
        EventType.INTO_CHAT_MESSAGE,
        EventType.CHAT_COMPLETION,
    ),
    "rag_stream": (
        EventType.LOAD_HISTORY,
        EventType.QUERY_UNDERSTAND,
        EventType.CHUNK_SEARCH_PARALLEL,
        EventType.CHUNK_RERANK,
        EventType.CHUNK_MERGE,
        EventType.FILTER_TOP_K,
        EventType.DATA_ANALYSIS,
        EventType.INTO_CHAT_MESSAGE,
        EventType.CHAT_COMPLETION_STREAM,
    ),
}


__all__ = [
    "PIPELINE_MODES",
    "ChatResponse",
    "Context",
    "EventType",
    "FallbackStrategy",
    "GraphData",
    "GraphNode",
    "GraphRelation",
    "History",
    "MessageAttachment",
    "MessageImage",
    "PipelineBuilder",
    "QueryIntent",
    "References",
    "SearchResult",
    "SearchTarget",
    "SearchTargetType",
    "SummaryConfig",
    "TokenUsage",
]
