"""Pipeline run carrier (upstream ``ChatManage``).

``PipelineContext`` is the single object shared by every plugin in a
pipeline run. It bundles three concerns, mirroring the upstream shape:

- request configuration (immutable once set at the entry point);
- intermediate state that plugins write as the chain progresses;
- runtime handles (message ids) for the executing turn.

It is intentionally mutable: the event-driven chain hands the same
carrier to each plugin so a step can observe and extend what earlier
steps produced — this is the upstream contract. Every value *inside* the
carrier (``SearchResult``, ``History``, ``SummaryConfig``, ...) is frozen;
plugins replace fields wholesale rather than editing nested objects in
place. Callers needing a pristine copy use ``clone()``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.core.chat.pipeline.types import (
    ChatResponse,
    FallbackStrategy,
    GraphData,
    History,
    MessageAttachment,
    QueryIntent,
    SearchResult,
    SearchTarget,
    SummaryConfig,
)


class PipelineContext(BaseModel):
    """Full configuration, intermediate state and handles for one run."""

    # ── Request configuration (set once at the request entry point) ──

    session_id: str = ""
    user_id: str = ""
    query: str = ""
    max_rounds: int = 4

    knowledge_base_ids: list[str] = Field(default_factory=list)
    knowledge_ids: list[str] = Field(default_factory=list)
    search_targets: list[SearchTarget] = Field(default_factory=list)
    vector_threshold: float = 0.0
    keyword_threshold: float = 0.0
    embedding_top_k: int = 10
    vector_database: str = ""

    rerank_model_id: str = ""
    rerank_top_k: int = 0
    rerank_threshold: float = 0.0

    chat_model_id: str = ""
    summary_config: SummaryConfig = Field(default_factory=SummaryConfig)
    fallback_strategy: FallbackStrategy = FallbackStrategy.MODEL
    fallback_response: str = ""
    fallback_prompt: str = ""
    citation_enabled: bool | None = None

    enable_rewrite: bool = False
    enable_query_expansion: bool = False
    rewrite_prompt_system: str = ""
    rewrite_prompt_user: str = ""
    query_understand_model_id: str = ""

    faq_priority_enabled: bool = False
    faq_direct_answer_threshold: float = 0.0
    faq_score_boost: float = 0.0
    data_analysis_enabled: bool = False

    images: list[str] = Field(default_factory=list)
    vlm_model_id: str = ""
    chat_model_supports_vision: bool = False
    attachments: list[MessageAttachment] = Field(default_factory=list)
    intent_prompt_overrides: dict[str, str] = Field(default_factory=dict)

    tenant_id: int = 0
    web_search_enabled: bool = False
    web_search_provider_id: str = ""
    web_search_max_results: int = 0
    web_fetch_enabled: bool = False
    web_fetch_top_n: int = 3
    language: str = ""

    # ── Intermediate state written by plugins as the chain progresses ──

    rewrite_query: str = ""
    intent: QueryIntent | None = None
    history: list[History] = Field(default_factory=list)

    search_result: list[SearchResult] = Field(default_factory=list)
    rerank_result: list[SearchResult] = Field(default_factory=list)
    merge_result: list[SearchResult] = Field(default_factory=list)

    entity: list[str] = Field(default_factory=list)
    entity_kb_ids: list[str] = Field(default_factory=list)
    entity_knowledge: dict[str, str] = Field(default_factory=dict)
    graph_result: GraphData | None = None

    user_content: str = ""
    rendered_contexts: str = ""
    chat_response: ChatResponse | None = None
    image_description: str = ""
    quoted_context: str = ""
    system_prompt_override: str = ""

    # ── Runtime handles for the executing turn ──

    message_id: str = ""
    user_message_id: str = ""

    # ── Behaviour helpers (mirroring the upstream contract) ──

    def needs_retrieval(self) -> bool:
        """Return whether this run should execute the retrieval stages.

        A web-search intent only retrieves when web search is enabled; the
        unclassified intent (no classification yet) defaults to retrieval
        for safety.
        """
        if self.intent == QueryIntent.WEB_SEARCH:
            return self.web_search_enabled
        if self.intent is None:
            return True
        return self.intent.needs_kb_retrieval()

    def citations_enabled(self) -> bool:
        """Return the effective citation setting for this run.

        ``None`` (unset) defaults to enabled, preserving behaviour for
        requests created before the option existed.
        """
        return self.citation_enabled is None or self.citation_enabled

    def references(self) -> list[SearchResult]:
        """Final reference list for the current turn.

        The merged context wins when the merge stage ran; otherwise the
        search results are returned.
        """
        if self.merge_result:
            return self.merge_result
        return self.search_result

    def clone(self) -> PipelineContext:
        """Return a deep copy of this carrier (used before a parallel run)."""
        return self.model_copy(deep=True)


__all__ = ["PipelineContext"]
