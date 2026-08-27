from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject


class AgentQuestionSuggestionsStarters(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    mode: str | None = Field(default="hybrid")
    items: list[str] = Field(default_factory=list)
    count: int = 6


class AgentQuestionSuggestionsFollowUps(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    mode: str | None = Field(default="hybrid")
    count: int = 3
    model_id: str | None = Field(default=None)
    categories: list[str] = Field(default_factory=lambda: ["clarify", "deepen", "action"])
    max_context_turns: int = 2
    additional_instruction: str | None = Field(default=None)
    suppress_on_fallback: bool = True
    suppress_when_answer_asks_question: bool = True
    knowledge_fallback: bool = True
    allow_regenerate: bool = False


class AgentQuestionSuggestions(BaseModel):
    model_config = ConfigDict(frozen=True)

    starters: AgentQuestionSuggestionsStarters | None = Field(default=None)
    follow_ups: AgentQuestionSuggestionsFollowUps | None = Field(default=None)


class ParserEngineRule(BaseModel):
    """Per-file-type parser dispatch rule."""

    model_config = ConfigDict(frozen=True)

    file_types: list[str]
    engine: str


class AgentConfig(BaseModel):
    """Configuration for a customizable agent.

    Fields used only during request execution are intentionally excluded from
    this serialized model.
    """

    model_config = ConfigDict(frozen=True)

    # ── Basic settings ────────────────────────────────────────────
    agent_mode: str | None = Field(default=None)
    agent_type: str | None = Field(default=None)
    system_prompt: str | None = Field(default=None)
    system_prompt_id: str | None = Field(default=None)
    context_template: str | None = Field(default=None)
    context_template_id: str | None = Field(default=None)

    # ── Model settings ─────────────────────────────────────────────
    model_id: str | None = Field(default=None)
    rerank_model_id: str | None = Field(default=None)
    temperature: float | None = Field(default=0.7)
    max_completion_tokens: int | None = Field(default=2048)
    thinking: bool | None = Field(default=None)
    citation_enabled: bool | None = Field(default=None)
    query_understand_model_id: str | None = Field(default=None)

    # ── Agent-mode settings ────────────────────────────────────────
    max_iterations: int | None = Field(default=10)
    llm_call_timeout: int | None = Field(default=0)
    allowed_tools: list[str] | None = Field(default=None)
    mcp_selection_mode: str | None = Field(default=None)
    mcp_services: list[str] | None = Field(default=None)
    mcp_auth_wait_timeout: int | None = Field(default=0)

    # ── Skills settings ───────────────────────────────────────────
    skills_selection_mode: str | None = Field(default=None)
    selected_skills: list[str] | None = Field(default=None)

    # ── Knowledge-base settings ────────────────────────────────────
    kb_selection_mode: str | None = Field(default=None)
    knowledge_bases: list[str] | None = Field(default=None)
    retrieve_kb_only_when_mentioned: bool = False
    retain_retrieval_history: bool = False

    # ── Image / audio / multimodal ────────────────────────────────
    image_upload_enabled: bool = False
    vlm_model_id: str | None = Field(default=None)
    audio_upload_enabled: bool = False
    asr_model_id: str | None = Field(default=None)
    image_storage_provider: str | None = Field(default=None)

    # ── File-type restriction ─────────────────────────────────────
    supported_file_types: list[str] | None = Field(default=None)

    # ── Chat attachment parsing ────────────────────────────────────
    chat_parser_engine_rules: list[ParserEngineRule] | None = Field(default=None)
    attachment_image_understanding: bool = False
    attachment_ocr_max_pages: int | None = Field(default=0)
    attachment_parse_wait_timeout_sec: int | None = Field(default=0)

    # ── Data analysis ─────────────────────────────────────────────
    data_analysis_enabled: bool = False

    # ── FAQ strategy ──────────────────────────────────────────────
    faq_priority_enabled: bool = True
    faq_direct_answer_threshold: float | None = Field(default=0.9)
    faq_score_boost: float | None = Field(default=1.2)

    # ── Web search ────────────────────────────────────────────────
    web_search_enabled: bool = True
    web_search_max_results: int | None = Field(default=5)
    web_search_provider_id: str | None = Field(default=None)
    web_fetch_enabled: bool = False
    web_fetch_top_n: int | None = Field(default=3)

    # ── Multi-turn ───────────────────────────────────────────────
    multi_turn_enabled: bool = True
    history_turns: int | None = Field(default=5)

    # ── Retrieval strategy ────────────────────────────────────────
    embedding_top_k: int | None = Field(default=10)
    keyword_threshold: float | None = Field(default=0.3)
    vector_threshold: float | None = Field(default=0.5)
    rerank_top_k: int | None = Field(default=5)
    rerank_threshold: float | None = Field(default=0.5)

    # ── Advanced ──────────────────────────────────────────────────
    parallel_tool_calls: bool = False
    max_context_tokens: int | None = Field(default=None)
    max_tool_output_chars: int | None = Field(default=None)
    enable_query_expansion: bool = True
    enable_rewrite: bool = True
    rewrite_prompt_system: str | None = Field(default=None)
    rewrite_prompt_user: str | None = Field(default=None)
    fallback_strategy: str | None = Field(default="model")
    fallback_response: str | None = Field(default=None)
    fallback_prompt: str | None = Field(default=None)
    intent_prompts: dict[str, str] | None = Field(default=None)

    # ── Conversation question suggestions ────────────────────────
    question_suggestions: AgentQuestionSuggestions | None = Field(default=None)


class Agent(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str | None = Field(default=None)
    description: str | None = Field(default=None)
    avatar: str | None = Field(default=None)
    is_builtin: bool
    tenant_id: int
    created_by: str | None = Field(default=None)
    config: AgentConfig | None = Field(default=None)
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = Field(default=None)


class CreateAgentRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str | None = Field(default=None)
    avatar: str | None = Field(default=None)
    config: AgentConfig | None = Field(default=None)


class UpdateAgentRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str | None = Field(default=None)
    description: str | None = Field(default=None)
    avatar: str | None = Field(default=None)
    config: AgentConfig | None = Field(default=None)


class ListAgentsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    agents: list[Agent]
    disabled_own_agent_ids: list[str]


class AgentPlaceholderGroup(BaseModel):
    model_config = ConfigDict(frozen=True)

    all: list[JsonObject]
    system_prompt: list[JsonObject]
    agent_system_prompt: list[JsonObject]
    context_template: list[JsonObject]
    rewrite_system_prompt: list[JsonObject]
    rewrite_prompt: list[JsonObject]
    fallback_prompt: list[JsonObject]


__all__ = [
    "Agent",
    "AgentConfig",
    "AgentPlaceholderGroup",
    "AgentQuestionSuggestions",
    "AgentQuestionSuggestionsFollowUps",
    "AgentQuestionSuggestionsStarters",
    "CreateAgentRequest",
    "ListAgentsResponse",
    "ParserEngineRule",
    "UpdateAgentRequest",
]
