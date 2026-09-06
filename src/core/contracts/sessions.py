from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject


class Session(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    title: str | None = Field(default=None)
    description: str | None = Field(default=None)
    tenant_id: int
    user_id: str | None = Field(default=None)
    is_pinned: bool = False
    pinned_at: datetime | None = Field(default=None)
    im_platform: str | None = Field(default=None)
    im_chat_id: str | None = Field(default=None)
    im_thread_id: str | None = Field(default=None)
    im_user_id: str | None = Field(default=None)
    im_agent_id: str | None = Field(default=None)
    im_channel_id: str | None = Field(default=None)
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = Field(default=None)


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str | None = Field(default=None)
    description: str | None = Field(default=None)


class UpdateSessionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str | None = Field(default=None)
    description: str | None = Field(default=None)


class ListSessionsQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    page: int = 1
    page_size: int = 10
    keyword: str | None = Field(default=None)
    source: str | None = Field(default=None)
    agent_id: str | None = Field(default=None)


class BatchDeleteSessionsRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    ids: list[str] | None = Field(default=None)
    delete_all: bool = False


class GenerateTitleRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    messages: list[TitleGenMessage]


class TitleGenMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str
    content: str


class StopGenerationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    message_id: str


class PinSessionResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    is_pinned: bool


class ContinueStreamQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    message_id: str


class KnowledgeReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    content: str
    knowledge_id: str
    chunk_index: int
    knowledge_title: str | None = Field(default=None)
    start_at: int | None = Field(default=None)
    end_at: int | None = Field(default=None)
    seq: int | None = Field(default=None)
    score: float | None = Field(default=None)
    match_type: int | None = Field(default=None)
    sub_chunk_id: list[str] | None = Field(default=None)
    metadata: JsonObject | None = Field(default=None)
    chunk_type: str | None = Field(default=None)
    parent_chunk_id: str | None = Field(default=None)
    image_info: str | None = Field(default=None)
    knowledge_filename: str | None = Field(default=None)
    knowledge_source: str | None = Field(default=None)


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    session_id: str
    request_id: str | None = Field(default=None)
    role: str
    content: str
    knowledge_references: list[KnowledgeReference] | None = Field(default=None)
    agent_steps: list[JsonObject] | None = Field(default=None)
    is_completed: bool = True
    is_fallback: bool = False
    agent_duration_ms: int | None = Field(default=None)
    mentioned_items: list[JsonObject] | None = Field(default=None)
    images: list[JsonObject] | None = Field(default=None)
    channel: str | None = Field(default=None)
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = Field(default=None)


class LoadMessagesQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    before_time: str | None = Field(default=None)
    limit: int = 20


class SearchMessagesRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    mode: str | None = Field(default="hybrid")
    limit: int = 20
    session_ids: list[str] | None = Field(default=None)


class MessageSearchHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: str
    session_id: str
    session_title: str | None = Field(default=None)
    query_content: str
    answer_content: str
    score: float
    match_type: str | None = Field(default=None)
    created_at: datetime


class MessageSearchResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    items: list[MessageSearchHit]
    total: int


class ChatHistoryStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool
    embedding_model_id: str | None = Field(default=None)
    knowledge_base_id: str | None = Field(default=None)
    knowledge_base_name: str | None = Field(default=None)
    indexed_message_count: int | None = Field(default=None)
    has_indexed_messages: bool | None = Field(default=None)


class MentionedItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    type: str
    kb_type: str | None = Field(default=None)


class ChatImage(BaseModel):
    model_config = ConfigDict(frozen=True)

    data: str


class ChatRequestBase(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    knowledge_base_ids: list[str] | None = Field(default=None)
    knowledge_ids: list[str] | None = Field(default=None)
    agent_id: str | None = Field(default=None)
    summary_model_id: str | None = Field(default=None)
    mentioned_items: list[MentionedItem] | None = Field(default=None)
    disable_title: bool = False
    images: list[ChatImage] | None = Field(default=None)
    channel: str | None = Field(default=None)
    suggestion_attribution: dict[str, str] | None = Field(default=None)


class KnowledgeChatRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    knowledge_base_ids: list[str] | None = Field(default=None)
    knowledge_ids: list[str] | None = Field(default=None)
    agent_id: str | None = Field(default=None)
    summary_model_id: str | None = Field(default=None)
    mentioned_items: list[MentionedItem] | None = Field(default=None)
    disable_title: bool = False
    images: list[ChatImage] | None = Field(default=None)
    channel: str | None = Field(default=None)
    suggestion_attribution: dict[str, str] | None = Field(default=None)


class AgentChatRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    knowledge_base_ids: list[str] | None = Field(default=None)
    knowledge_ids: list[str] | None = Field(default=None)
    agent_enabled: bool = False
    agent_id: str | None = Field(default=None)
    web_search_enabled: bool = False
    summary_model_id: str | None = Field(default=None)
    mentioned_items: list[MentionedItem] | None = Field(default=None)
    disable_title: bool = False
    images: list[ChatImage] | None = Field(default=None)
    channel: str | None = Field(default=None)
    suggestion_attribution: dict[str, str] | None = Field(default=None)


class SuggestionAttribution(BaseModel):
    model_config = ConfigDict(frozen=True)

    suggestion_set_id: str
    question_id: str


class SuggestionQuestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    category: str | None = Field(default=None)


class SuggestionSet(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    session_id: str
    assistant_message_id: str
    position: str
    status: str
    language: str | None = Field(default=None)
    config_snapshot: JsonObject | None = Field(default=None)
    questions: list[SuggestionQuestion] | None = Field(default=None)
    created_at: datetime | None = Field(default=None)
    updated_at: datetime | None = Field(default=None)


class EnsureSuggestionsRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    regenerate: bool = False


class SuggestionEventRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    suggestion_set_id: str
    question_id: str
    event_type: str


__all__ = [
    "AgentChatRequest",
    "BatchDeleteSessionsRequest",
    "ChatHistoryStats",
    "ChatImage",
    "ChatRequestBase",
    "ContinueStreamQuery",
    "CreateSessionRequest",
    "EnsureSuggestionsRequest",
    "GenerateTitleRequest",
    "KnowledgeChatRequest",
    "KnowledgeReference",
    "ListSessionsQuery",
    "LoadMessagesQuery",
    "MentionedItem",
    "Message",
    "MessageSearchHit",
    "MessageSearchResponse",
    "PinSessionResponse",
    "SearchMessagesRequest",
    "Session",
    "StopGenerationRequest",
    "SuggestionAttribution",
    "SuggestionEventRequest",
    "SuggestionQuestion",
    "SuggestionSet",
    "TitleGenMessage",
    "UpdateSessionRequest",
]
