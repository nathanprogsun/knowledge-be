"""Tests for the agent-QA orchestration.

Exercises the pure helpers (config assembly, scope narrowing, query
routing) and the high-level ``run_agent_qa`` entry point. All heavy
seams (model service, tenant loader, history loader, engine factory)
are replaced with tiny stubs so the orchestration logic can be verified
without any live service.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from src.ai.llm.types import Chat, ChatOptions, ChatResponse, Message, StreamResponse
from src.ai.rerank.base import Reranker
from src.ai.rerank.remote_api import RankResult
from src.common.json import JsonObject
from src.core.agents.engine.loop import AgentEngine
from src.core.agents.engine.types import AgentConfig
from src.core.agents.tools.base import (
    TOOL_KNOWLEDGE_SEARCH,
    TOOL_THINKING,
    TOOL_TODO_WRITE,
)
from src.core.chat.bus import Event, EventBus
from src.core.chat.pipeline.types import MessageAttachment, SearchTarget, SearchTargetType
from src.core.chat.sessions.agent_qa import (
    DEFAULT_ALLOWED_TOOLS,
    AgentInfo,
    AgentQARequest,
    ModelInfo,
    TagScope,
    TenantInfo,
    agent_requires_rerank_model,
    apply_per_request_mcp_scope,
    apply_per_request_skill_scope,
    build_agent_config,
    build_agent_query,
    dedup_preserving_order,
    default_allowed_tools,
    intersect_preserving_request_order,
    merge_resolved_tag_knowledge_ids,
    resolve_chat_model_id,
    resolve_retrieval_tenant_id,
    run_agent_qa,
    skills_from_agent_config,
    unique_non_empty,
)
from src.core.chat.types import EventType

# ── Stubs ─────────────────────────────────────────────────────────────


class _Ctx:
    """Opaque execution context satisfying the ``Context`` protocol."""

    tenant_id = 1
    user_id = "test-user"
    request_id = "req-1"
    is_background_task = False


class _RecordingBus(EventBus):
    """In-process event bus that captures every emitted event."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)


class _StubChat:
    """Minimal chat client satisfying the ``Chat`` protocol."""

    async def chat(
        self,
        messages: list[Message],
        opts: ChatOptions | None = None,
    ) -> ChatResponse:
        return ChatResponse(content="stub")

    def chat_stream(
        self,
        messages: list[Message],
        opts: ChatOptions | None = None,
    ) -> AsyncIterator[StreamResponse]:
        async def gen() -> AsyncIterator[StreamResponse]:
            yield StreamResponse(content="stub")

        return gen()

    def get_model_name(self) -> str:
        return "stub-model"

    def get_model_id(self) -> str:
        return "stub"


class _StubReranker:
    """Reranker stub satisfying the ``Reranker`` protocol."""

    async def rerank(self, query: str, documents: list[str]) -> list[RankResult]:
        return []

    def get_model_name(self) -> str:
        return "stub-rerank"

    def get_model_id(self) -> str:
        return "rerank-stub"


class _StubModelService:
    """Model service stub with controllable chat / info lookups."""

    def __init__(
        self,
        *,
        models: dict[str, ModelInfo] | None = None,
        chats: dict[str, Chat] | None = None,
    ) -> None:
        self._models = models or {}
        self._chats = chats or {}
        self.get_chat_calls: list[str] = []
        self.get_info_calls: list[str] = []

    async def get_chat_model(self, ctx: Any, model_id: str) -> Chat:
        self.get_chat_calls.append(model_id)
        return self._chats.get(model_id, _StubChat())

    async def get_model_by_id(self, ctx: Any, model_id: str) -> ModelInfo | None:
        self.get_info_calls.append(model_id)
        return self._models.get(model_id)


class _StubRerankService:
    """Rerank service stub returning a reranker stub."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def get_rerank_model(self, ctx: Any, model_id: str) -> Reranker:
        self.calls.append(model_id)
        return _StubReranker()


class _StubHistoryLoader:
    """History loader stub returning a configured message list."""

    def __init__(
        self,
        messages: list[Message] | None = None,
        fail: bool = False,
    ) -> None:
        self._messages = messages or []
        self._fail = fail
        self.calls: list[tuple[str, int]] = []

    async def load(self, ctx: Any, session_id: str, max_rounds: int) -> list[Message]:
        self.calls.append((session_id, max_rounds))
        if self._fail:
            raise RuntimeError("db unavailable")
        return list(self._messages)


class _StubTenantLoader:
    """Tenant loader stub returning a fixed tenant info row."""

    def __init__(self, tenant: TenantInfo | None = None) -> None:
        self._tenant = tenant
        self.calls: list[int] = []

    async def load(self, ctx: Any, tenant_id: int) -> TenantInfo | None:
        self.calls.append(tenant_id)
        return self._tenant


class _RecordingEngine:
    """Agent engine stub capturing the call and returning a frozen state."""

    def __init__(
        self,
        *,
        state: Any = None,
        raises: BaseException | None = None,
    ) -> None:
        self.state = state
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    async def execute(self, ctx: Any, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.requires_raises():
            raise self.raises  # type: ignore[misc]
        return self.state

    def requires_raises(self) -> bool:
        return self.raises is not None


class _StubEngineFactory:
    """Agent engine factory stub returning a configured engine."""

    def __init__(self, engine: _RecordingEngine) -> None:
        self._engine = engine
        self.calls: list[dict[str, Any]] = []

    def build(
        self,
        *,
        config: AgentConfig,
        chat_model: Chat,
        rerank_model: Reranker | None,
        event_bus: EventBus,
        session_id: str,
        assistant_message_id: str,
    ) -> AgentEngine:
        self.calls.append(
            {
                "config": config,
                "chat_model": chat_model,
                "rerank_model": rerank_model,
                "event_bus": event_bus,
                "session_id": session_id,
                "assistant_message_id": assistant_message_id,
            }
        )
        return cast(AgentEngine, self._engine)


# ── Fixtures ──────────────────────────────────────────────────────────


def _agent_config(
    **overrides: Any,
) -> JsonObject:
    """Build a minimal agent config blob with sensible defaults."""
    defaults: dict[str, Any] = {
        "model_id": "model-1",
        "rerank_model_id": "rerank-default",
        "max_iterations": 10,
        "temperature": 0.7,
        "multi_turn_enabled": True,
        "history_turns": 5,
        "retrieve_kb_only_when_mentioned": False,
        "allowed_tools": list(DEFAULT_ALLOWED_TOOLS),
    }
    defaults.update(overrides)
    return defaults


def _make_agent(**config_overrides: Any) -> AgentInfo:
    """Build an ``AgentInfo`` with overridable config fields."""
    return AgentInfo(
        id="agent-1",
        tenant_id=42,
        config=_agent_config(**config_overrides),
    )


def _make_request(**overrides: Any) -> AgentQARequest:
    """Build an ``AgentQARequest`` with sensible defaults."""
    defaults: dict[str, Any] = {
        "session_id": "sess-1",
        "session_tenant_id": 7,
        "query": "What is RAG?",
        "assistant_message_id": "msg-1",
        "custom_agent": _make_agent(),
        "summary_model_id": "",
        "shared_agent_read_only": False,
        "knowledge_base_ids": ("kb-1",),
        "knowledge_ids": (),
        "tag_scopes": (),
        "mcp_service_ids": (),
        "skill_names": (),
        "image_urls": (),
        "image_description": "",
        "web_search_enabled": False,
        "quoted_context": "",
        "attachments": (),
    }
    defaults.update(overrides)
    return AgentQARequest(**defaults)


# ── default_allowed_tools ────────────────────────────────────────────


class TestDefaultAllowedTools:
    """Tests for the default allowed-tools list."""

    def test_matches_upstream_whitelist(self) -> None:
        tools = default_allowed_tools()
        assert TOOL_THINKING in tools
        assert TOOL_TODO_WRITE in tools
        assert TOOL_KNOWLEDGE_SEARCH in tools
        assert len(tools) == len(DEFAULT_ALLOWED_TOOLS)

    def test_returns_a_fresh_list(self) -> None:
        """Each call returns a fresh list so callers can mutate freely."""
        a = default_allowed_tools()
        a.append("extra")
        b = default_allowed_tools()
        assert "extra" not in b


# ── agent_requires_rerank_model ──────────────────────────────────────


class TestAgentRequiresRerankModel:
    """Tests for the rerank-model requirement heuristic."""

    def test_kb_selection_none_disables_rerank(self) -> None:
        cfg = _agent_config(kb_selection_mode="none")
        assert agent_requires_rerank_model(cfg) is False

    def test_kb_selection_unknown_enables_rerank(self) -> None:
        cfg = _agent_config(kb_selection_mode="custom")
        assert agent_requires_rerank_model(cfg) is True

    def test_default_whitelist_enables_rerank(self) -> None:
        cfg = _agent_config()
        cfg.pop("allowed_tools")
        assert agent_requires_rerank_model(cfg) is True

    def test_explicit_whitelist_without_knowledge_search_disables(self) -> None:
        cfg = _agent_config(allowed_tools=[TOOL_THINKING, TOOL_TODO_WRITE])
        assert agent_requires_rerank_model(cfg) is False

    def test_explicit_whitelist_with_knowledge_search_enables(self) -> None:
        cfg = _agent_config(
            allowed_tools=[TOOL_THINKING, TOOL_KNOWLEDGE_SEARCH]
        )
        assert agent_requires_rerank_model(cfg) is True

    def test_empty_whitelist_falls_back_to_default(self) -> None:
        cfg = _agent_config(allowed_tools=[])
        assert agent_requires_rerank_model(cfg) is True

    def test_non_list_whitelist_falls_back_to_default(self) -> None:
        cfg = _agent_config(allowed_tools="not-a-list")
        assert agent_requires_rerank_model(cfg) is True


# ── dedup_preserving_order / intersect_preserving_request_order ──────


class TestDedupPreservingOrder:
    """Tests for the order-preserving deduplication helper."""

    def test_drops_duplicates_and_blanks(self) -> None:
        assert dedup_preserving_order(["a", "b", "a", "", "c", "b"]) == ["a", "b", "c"]

    def test_empty_input(self) -> None:
        assert dedup_preserving_order([]) == []

    def test_all_unique(self) -> None:
        assert dedup_preserving_order(["x", "y", "z"]) == ["x", "y", "z"]


class TestIntersectPreservingRequestOrder:
    """Tests for the request-order intersection helper."""

    def test_intersection_preserves_request_order(self) -> None:
        assert intersect_preserving_request_order(
            ["c", "a", "b"], ["a", "b", "c"]
        ) == ["c", "a", "b"]

    def test_drops_blanks_and_duplicates(self) -> None:
        assert intersect_preserving_request_order(
            ["", "a", "a", "b"], ["a", "b", "c"]
        ) == ["a", "b"]

    def test_empty_intersection(self) -> None:
        assert intersect_preserving_request_order(["x"], ["a", "b"]) == []

    def test_requested_not_in_allowed(self) -> None:
        assert intersect_preserving_request_order(
            ["x", "y"], ["a", "b"]
        ) == []


class TestUniqueNonEmpty:
    """Tests for the unique-non-empty helper."""

    def test_drops_blanks_and_duplicates(self) -> None:
        assert unique_non_empty(["a", "", "a", "b"]) == ["a", "b"]

    def test_empty(self) -> None:
        assert unique_non_empty([]) == []


# ── merge_resolved_tag_knowledge_ids ─────────────────────────────────


class TestMergeResolvedTagKnowledgeIds:
    """Tests for the tag-knowledge-id merge helper."""

    def test_no_tag_scopes_returns_unique_existing(self) -> None:
        target = SearchTarget(
            type=SearchTargetType.KNOWLEDGE,
            knowledge_base_id="kb-1",
            knowledge_ids=["k-1"],
        )
        assert merge_resolved_tag_knowledge_ids(
            ["k-1", "k-2"], [target], []
        ) == ["k-1", "k-2"]

    def test_merges_tag_targets(self) -> None:
        scope = TagScope(knowledge_base_id="kb-1", tag_ids=("t-1",))
        target = SearchTarget(
            type=SearchTargetType.KNOWLEDGE,
            knowledge_base_id="kb-1",
            knowledge_ids=["k-9"],
        )
        merged = merge_resolved_tag_knowledge_ids(
            ["k-1", "k-2"], [target], [scope]
        )
        assert merged == ["k-1", "k-2", "k-9"]

    def test_skips_kb_base_targets(self) -> None:
        scope = TagScope(knowledge_base_id="kb-1", tag_ids=("t-1",))
        target = SearchTarget(
            type=SearchTargetType.KNOWLEDGE_BASE,
            knowledge_base_id="kb-1",
        )
        assert merge_resolved_tag_knowledge_ids(
            ["k-1"], [target], [scope]
        ) == ["k-1"]

    def test_skips_targets_outside_tag_kbs(self) -> None:
        scope = TagScope(knowledge_base_id="kb-1", tag_ids=("t-1",))
        target = SearchTarget(
            type=SearchTargetType.KNOWLEDGE,
            knowledge_base_id="kb-other",
            knowledge_ids=["k-9"],
        )
        assert merge_resolved_tag_knowledge_ids(
            ["k-1"], [target], [scope]
        ) == ["k-1"]

    def test_dedupes_after_merge(self) -> None:
        scope = TagScope(knowledge_base_id="kb-1", tag_ids=("t-1",))
        target = SearchTarget(
            type=SearchTargetType.KNOWLEDGE,
            knowledge_base_id="kb-1",
            knowledge_ids=["k-1", "k-2"],
        )
        merged = merge_resolved_tag_knowledge_ids(
            ["k-1"], [target], [scope]
        )
        assert merged == ["k-1", "k-2"]


# ── skills_from_agent_config ─────────────────────────────────────────


class TestSkillsFromAgentConfig:
    """Tests for the skill config resolver."""

    def test_skills_unavailable_disables(self) -> None:
        cfg = _agent_config(skills_selection_mode="all")
        enabled, dirs, allowed = skills_from_agent_config(
            cfg, skills_available=False
        )
        assert (enabled, dirs, allowed) == (False, [], [])

    def test_all_mode_enables_preloaded(self) -> None:
        cfg = _agent_config(skills_selection_mode="all")
        enabled, dirs, allowed = skills_from_agent_config(
            cfg, skills_available=True
        )
        assert enabled is True
        assert dirs == ["skills/preloaded"]
        assert allowed == []

    def test_selected_mode_returns_selected_skills(self) -> None:
        cfg = _agent_config(
            skills_selection_mode="selected",
            selected_skills=["alpha", "beta"],
        )
        enabled, dirs, allowed = skills_from_agent_config(
            cfg, skills_available=True
        )
        assert enabled is True
        assert dirs == ["skills/preloaded"]
        assert allowed == ["alpha", "beta"]

    def test_selected_mode_without_skills_disables(self) -> None:
        cfg = _agent_config(
            skills_selection_mode="selected", selected_skills=[]
        )
        enabled, _dirs, allowed = skills_from_agent_config(
            cfg, skills_available=True
        )
        assert (enabled, allowed) == (False, [])

    def test_none_mode_disables(self) -> None:
        cfg = _agent_config(skills_selection_mode="none")
        enabled, dirs, allowed = skills_from_agent_config(
            cfg, skills_available=True
        )
        assert (enabled, dirs, allowed) == (False, [], [])

    def test_unknown_mode_disables(self) -> None:
        cfg = _agent_config(skills_selection_mode="bogus")
        enabled, _dirs, _allowed = skills_from_agent_config(
            cfg, skills_available=True
        )
        assert enabled is False

    def test_non_list_selected_skills_disables(self) -> None:
        cfg = _agent_config(
            skills_selection_mode="selected", selected_skills="not-a-list"
        )
        enabled, _dirs, _allowed = skills_from_agent_config(
            cfg, skills_available=True
        )
        assert enabled is False


# ── apply_per_request_skill_scope ─────────────────────────────────────


class TestApplyPerRequestSkillScope:
    """Tests for the per-request skill scope narrowing."""

    def test_no_request_is_identity(self) -> None:
        assert apply_per_request_skill_scope(
            requested=[],
            skills_mode="all",
            skills_enabled=True,
            allowed_skills=["alpha"],
        ) == (True, ["alpha"])

    def test_skills_already_disabled_is_identity(self) -> None:
        assert apply_per_request_skill_scope(
            requested=["alpha"],
            skills_mode="all",
            skills_enabled=False,
            allowed_skills=[],
        ) == (False, [])

    def test_none_mode_is_identity_with_warning(self) -> None:
        result = apply_per_request_skill_scope(
            requested=["alpha"],
            skills_mode="none",
            skills_enabled=True,
            allowed_skills=["alpha"],
        )
        assert result == (True, ["alpha"])

    def test_selected_mode_intersects(self) -> None:
        assert apply_per_request_skill_scope(
            requested=["alpha", "beta"],
            skills_mode="selected",
            skills_enabled=True,
            allowed_skills=["beta", "gamma"],
        ) == (True, ["beta"])

    def test_selected_mode_disables_when_intersection_empty(self) -> None:
        assert apply_per_request_skill_scope(
            requested=["delta"],
            skills_mode="selected",
            skills_enabled=True,
            allowed_skills=["beta"],
        ) == (False, [])

    def test_all_mode_narrows_to_requested(self) -> None:
        assert apply_per_request_skill_scope(
            requested=["alpha", "beta", "alpha"],
            skills_mode="all",
            skills_enabled=True,
            allowed_skills=["unused"],
        ) == (True, ["alpha", "beta"])


# ── apply_per_request_mcp_scope ──────────────────────────────────────


class TestApplyPerRequestMcpScope:
    """Tests for the per-request MCP scope narrowing."""

    def test_no_request_returns_none(self) -> None:
        assert (
            apply_per_request_mcp_scope(
                requested=[],
                agent_mcps=["m1"],
                selection_mode="all",
                is_shared_agent=False,
            )
            is None
        )

    def test_none_mode_returns_none(self) -> None:
        assert (
            apply_per_request_mcp_scope(
                requested=["m1"],
                agent_mcps=["m1"],
                selection_mode="none",
                is_shared_agent=False,
            )
            is None
        )

    def test_all_mode_returns_requested(self) -> None:
        mode, services = apply_per_request_mcp_scope(
            requested=["m1", "m2"],
            agent_mcps=["m1", "m2", "m3"],
            selection_mode="all",
            is_shared_agent=False,
        ) or ("", [])
        assert mode == "selected"
        assert services == ["m1", "m2"]

    def test_selected_mode_intersects_with_agent(self) -> None:
        mode, services = apply_per_request_mcp_scope(
            requested=["m1", "m2"],
            agent_mcps=["m1", "m3"],
            selection_mode="selected",
            is_shared_agent=False,
        ) or ("", [])
        assert mode == "selected"
        assert services == ["m1"]

    def test_shared_agent_narrows_to_agent_preset(self) -> None:
        # Shared agent cannot register services outside the agent preset.
        assert (
            apply_per_request_mcp_scope(
                requested=["external"],
                agent_mcps=["preset-1"],
                selection_mode="all",
                is_shared_agent=True,
            )
            is None
        )

    def test_shared_agent_keeps_overlap(self) -> None:
        _mode, services = apply_per_request_mcp_scope(
            requested=["preset-1", "external"],
            agent_mcps=["preset-1"],
            selection_mode="all",
            is_shared_agent=True,
        ) or ("", [])
        assert services == ["preset-1"]


# ── resolve_retrieval_tenant_id ──────────────────────────────────────


class TestResolveRetrievalTenantId:
    """Tests for the tenant resolution priority."""

    def test_agent_tenant_wins(self) -> None:
        assert (
            resolve_retrieval_tenant_id(
                session_tenant_id=1,
                agent_tenant_id=42,
                context_tenant_id=99,
            )
            == 42
        )

    def test_context_tenant_used_when_agent_tenant_zero(self) -> None:
        assert (
            resolve_retrieval_tenant_id(
                session_tenant_id=1,
                agent_tenant_id=0,
                context_tenant_id=99,
            )
            == 99
        )

    def test_session_tenant_is_fallback(self) -> None:
        assert (
            resolve_retrieval_tenant_id(
                session_tenant_id=1,
                agent_tenant_id=0,
                context_tenant_id=None,
            )
            == 1
        )

    def test_no_context_uses_session(self) -> None:
        assert (
            resolve_retrieval_tenant_id(
                session_tenant_id=7,
                agent_tenant_id=0,
                context_tenant_id=0,
            )
            == 7
        )


# ── build_agent_config ───────────────────────────────────────────────


class TestBuildAgentConfig:
    """Tests for the runtime ``AgentConfig`` assembly."""

    def test_copies_scalar_fields(self) -> None:
        cfg = build_agent_config(
            custom_agent=_make_agent(max_iterations=12, temperature=0.3),
            session_tenant_id=7,
            shared_agent_read_only=True,
            web_search_enabled=True,
            knowledge_base_ids=["kb-1"],
            knowledge_ids=["k-1"],
            tag_scopes=(),
            search_targets=(),
            mcp_service_ids=(),
            skill_names=(),
            skills_available=False,
        )
        assert cfg.max_iterations == 12
        assert cfg.temperature == 0.3
        assert cfg.knowledge_bases == ["kb-1"]
        assert cfg.knowledge_ids == ["k-1"]

    def test_web_search_requires_both_agent_and_request(self) -> None:
        cfg = build_agent_config(
            custom_agent=_make_agent(web_search_enabled=True),
            session_tenant_id=7,
            shared_agent_read_only=False,
            web_search_enabled=False,
            knowledge_base_ids=[],
            knowledge_ids=[],
            tag_scopes=(),
            search_targets=(),
            mcp_service_ids=(),
            skill_names=(),
            skills_available=False,
        )
        assert cfg.web_search_enabled is False

    def test_web_search_enabled_only_when_both_set(self) -> None:
        cfg = build_agent_config(
            custom_agent=_make_agent(web_search_enabled=True),
            session_tenant_id=7,
            shared_agent_read_only=False,
            web_search_enabled=True,
            knowledge_base_ids=[],
            knowledge_ids=[],
            tag_scopes=(),
            search_targets=(),
            mcp_service_ids=(),
            skill_names=(),
            skills_available=False,
        )
        assert cfg.web_search_enabled is True

    def test_allowed_tools_falls_back_to_default(self) -> None:
        cfg = build_agent_config(
            custom_agent=_make_agent(),
            session_tenant_id=7,
            shared_agent_read_only=False,
            web_search_enabled=False,
            knowledge_base_ids=[],
            knowledge_ids=[],
            tag_scopes=(),
            search_targets=(),
            mcp_service_ids=(),
            skill_names=(),
            skills_available=False,
        )
        assert cfg.allowed_tools == default_allowed_tools()

    def test_allowed_tools_uses_agent_override(self) -> None:
        cfg = build_agent_config(
            custom_agent=_make_agent(allowed_tools=[TOOL_THINKING]),
            session_tenant_id=7,
            shared_agent_read_only=False,
            web_search_enabled=False,
            knowledge_base_ids=[],
            knowledge_ids=[],
            tag_scopes=(),
            search_targets=(),
            mcp_service_ids=(),
            skill_names=(),
            skills_available=False,
        )
        assert cfg.allowed_tools == [TOOL_THINKING]

    def test_skills_disabled_when_unavailable(self) -> None:
        cfg = build_agent_config(
            custom_agent=_make_agent(skills_selection_mode="all"),
            session_tenant_id=7,
            shared_agent_read_only=False,
            web_search_enabled=False,
            knowledge_base_ids=[],
            knowledge_ids=[],
            tag_scopes=(),
            search_targets=(),
            mcp_service_ids=(),
            skill_names=(),
            skills_available=False,
        )
        assert cfg.skills_enabled is False

    def test_system_prompt_used_when_present(self) -> None:
        cfg = build_agent_config(
            custom_agent=_make_agent(system_prompt="custom prompt"),
            session_tenant_id=7,
            shared_agent_read_only=False,
            web_search_enabled=False,
            knowledge_base_ids=[],
            knowledge_ids=[],
            tag_scopes=(),
            search_targets=(),
            mcp_service_ids=(),
            skill_names=(),
            skills_available=False,
        )
        assert cfg.system_prompt == "custom prompt"
        assert cfg.use_custom_system_prompt is True

    def test_per_request_skill_scope_narrows_whitelist(self) -> None:
        cfg = build_agent_config(
            custom_agent=_make_agent(
                skills_selection_mode="all",
                selected_skills=["ignored-in-all-mode"],
            ),
            session_tenant_id=7,
            shared_agent_read_only=False,
            web_search_enabled=False,
            knowledge_base_ids=[],
            knowledge_ids=[],
            tag_scopes=(),
            search_targets=(),
            mcp_service_ids=(),
            skill_names=["alpha", "beta"],
            skills_available=True,
        )
        assert cfg.skills_enabled is True
        assert cfg.allowed_skills == ["alpha", "beta"]

    def test_per_request_mcp_scope_narrows_services(self) -> None:
        cfg = build_agent_config(
            custom_agent=_make_agent(
                mcp_selection_mode="selected",
                mcp_services=["m1", "m2"],
            ),
            session_tenant_id=7,
            shared_agent_read_only=False,
            web_search_enabled=False,
            knowledge_base_ids=[],
            knowledge_ids=[],
            tag_scopes=(),
            search_targets=(),
            mcp_service_ids=["m2", "m3"],
            skill_names=(),
            skills_available=False,
        )
        assert cfg.mcp_selection_mode == "selected"
        assert cfg.mcp_services == ["m2"]

    def test_tag_knowledge_ids_merged(self) -> None:
        scope = TagScope(knowledge_base_id="kb-1", tag_ids=("t-1",))
        target = SearchTarget(
            type=SearchTargetType.KNOWLEDGE,
            knowledge_base_id="kb-1",
            knowledge_ids=["k-merged"],
        )
        cfg = build_agent_config(
            custom_agent=_make_agent(),
            session_tenant_id=7,
            shared_agent_read_only=False,
            web_search_enabled=False,
            knowledge_base_ids=["kb-1"],
            knowledge_ids=["k-existing"],
            tag_scopes=[scope],
            search_targets=[target],
            mcp_service_ids=(),
            skill_names=(),
            skills_available=False,
        )
        assert cfg.knowledge_ids == ["k-existing", "k-merged"]

    def test_llm_call_timeout_falls_back_to_default(self) -> None:
        cfg = build_agent_config(
            custom_agent=_make_agent(),
            session_tenant_id=7,
            shared_agent_read_only=False,
            web_search_enabled=False,
            knowledge_base_ids=[],
            knowledge_ids=[],
            tag_scopes=(),
            search_targets=(),
            mcp_service_ids=(),
            skill_names=(),
            skills_available=False,
            llm_call_timeout_default=0.0,
        )
        assert cfg.llm_call_timeout > 0

    def test_llm_call_timeout_prefers_agent_value(self) -> None:
        cfg = build_agent_config(
            custom_agent=_make_agent(llm_call_timeout=42),
            session_tenant_id=7,
            shared_agent_read_only=False,
            web_search_enabled=False,
            knowledge_base_ids=[],
            knowledge_ids=[],
            tag_scopes=(),
            search_targets=(),
            mcp_service_ids=(),
            skill_names=(),
            skills_available=False,
            llm_call_timeout_default=999,
        )
        assert cfg.llm_call_timeout == 42

    def test_max_context_tokens_default_applied(self) -> None:
        cfg = build_agent_config(
            custom_agent=_make_agent(),
            session_tenant_id=7,
            shared_agent_read_only=False,
            web_search_enabled=False,
            knowledge_base_ids=[],
            knowledge_ids=[],
            tag_scopes=(),
            search_targets=(),
            mcp_service_ids=(),
            skill_names=(),
            skills_available=False,
        )
        assert cfg.max_context_tokens > 0


# ── build_agent_query ────────────────────────────────────────────────


class TestBuildAgentQuery:
    """Tests for the user-query and image-list assembly."""

    def test_query_only(self) -> None:
        query, urls = build_agent_query(
            query="What is RAG?",
            image_urls=(),
            image_description="",
            model_supports_vision=False,
        )
        assert query == "What is RAG?"
        assert urls == []

    def test_vision_passes_image_urls_directly(self) -> None:
        query, urls = build_agent_query(
            query="Q",
            image_urls=("https://img/1", "https://img/2"),
            image_description="a description",
            model_supports_vision=True,
        )
        assert query == "Q"
        assert urls == ["https://img/1", "https://img/2"]

    def test_no_vision_appends_description(self) -> None:
        query, urls = build_agent_query(
            query="Q",
            image_urls=("https://img/1",),
            image_description="A diagram of RAG",
            model_supports_vision=False,
        )
        assert "[用户上传图片内容]" in query
        assert "A diagram of RAG" in query
        assert urls == []

    def test_quoted_context_appended(self) -> None:
        query, _ = build_agent_query(
            query="Q",
            image_urls=(),
            image_description="",
            model_supports_vision=False,
            quoted_context="> previous message",
        )
        assert "> previous message" in query

    def test_attachments_appended(self) -> None:
        att = MessageAttachment(
            id="att-1",
            file_name="report.pdf",
            file_type="pdf",
            file_size=1024,
            content="PDF content",
        )
        query, _ = build_agent_query(
            query="Q",
            image_urls=(),
            image_description="",
            model_supports_vision=False,
            attachments=[att],
        )
        assert "<attachments>" in query
        assert "report.pdf" in query

    def test_vision_with_no_images_appends_description(self) -> None:
        query, urls = build_agent_query(
            query="Q",
            image_urls=(),
            image_description="description",
            model_supports_vision=True,
        )
        assert "description" in query
        assert urls == []


# ── resolve_chat_model_id ────────────────────────────────────────────


class TestResolveChatModelId:
    """Tests for the chat-model-id resolver."""

    @pytest.mark.asyncio
    async def test_uses_agent_model_when_override_invalid(self) -> None:
        service = _StubModelService(
            models={
                "agent-model": ModelInfo(id="agent-model", type="knowledge_qa"),
                "bad-override": ModelInfo(id="bad-override", type="embedding"),
            }
        )
        req = _make_request(
            custom_agent=_make_agent(model_id="agent-model"),
            summary_model_id="bad-override",
        )
        result = await resolve_chat_model_id(
            ctx=_Ctx(),
            req=req,
            knowledge_base_ids=[],
            knowledge_ids=[],
            model_service=service,
        )
        assert result == "agent-model"

    @pytest.mark.asyncio
    async def test_override_wins_when_valid(self) -> None:
        service = _StubModelService(
            models={
                "agent-model": ModelInfo(id="agent-model", type="knowledge_qa"),
                "override-model": ModelInfo(
                    id="override-model", type="knowledge_qa"
                ),
            }
        )
        req = _make_request(
            custom_agent=_make_agent(model_id="agent-model"),
            summary_model_id="override-model",
        )
        result = await resolve_chat_model_id(
            ctx=_Ctx(),
            req=req,
            knowledge_base_ids=[],
            knowledge_ids=[],
            model_service=service,
        )
        assert result == "override-model"

    @pytest.mark.asyncio
    async def test_missing_agent_model_raises(self) -> None:
        agent = _make_agent()
        agent = AgentInfo(
            id=agent.id,
            tenant_id=agent.tenant_id,
            config={**agent.config, "model_id": ""},
        )
        req = _make_request(custom_agent=agent)
        service = _StubModelService()
        with pytest.raises(RuntimeError, match="chat model is not configured"):
            await resolve_chat_model_id(
                ctx=_Ctx(),
                req=req,
                knowledge_base_ids=[],
                knowledge_ids=[],
                model_service=service,
            )

    @pytest.mark.asyncio
    async def test_wrong_model_type_raises(self) -> None:
        agent = _make_agent(model_id="embedding-model")
        req = _make_request(custom_agent=agent)
        service = _StubModelService(
            models={
                "embedding-model": ModelInfo(
                    id="embedding-model", type="embedding"
                )
            }
        )
        with pytest.raises(RuntimeError, match="is unavailable"):
            await resolve_chat_model_id(
                ctx=_Ctx(),
                req=req,
                knowledge_base_ids=[],
                knowledge_ids=[],
                model_service=service,
            )


# ── run_agent_qa ─────────────────────────────────────────────────────


class TestRunAgentQa:
    """Tests for the high-level orchestration entry point."""

    @pytest.mark.asyncio
    async def test_executes_engine_and_passes_inputs(self) -> None:
        agent = _make_agent(
            multi_turn_enabled=True,
            history_turns=3,
            allowed_tools=[TOOL_KNOWLEDGE_SEARCH, TOOL_THINKING],
            rerank_model_id="rerank-1",
        )
        req = _make_request(
            custom_agent=agent,
            query="hello",
            image_description="a picture",
            image_urls=("https://img/1",),
            quoted_context="> quoted",
        )
        model_service = _StubModelService(
            models={
                "model-1": ModelInfo(
                    id="model-1", type="knowledge_qa", supports_vision=False
                )
            },
            chats={"model-1": _StubChat()},
        )
        rerank_service = _StubRerankService()
        history_loader = _StubHistoryLoader()
        tenant_loader = _StubTenantLoader(
            TenantInfo(id=42, web_search_max_results=10)
        )
        engine = _RecordingEngine()
        factory = _StubEngineFactory(engine)
        bus = _RecordingBus()

        await run_agent_qa(
            ctx=_Ctx(),
            req=req,
            event_bus=bus,
            model_service=model_service,
            rerank_service=rerank_service,
            tenant_loader=tenant_loader,
            history_loader=history_loader,
            engine_factory=factory,
            skills_available=False,
        )

        assert factory.calls, "engine factory was invoked"
        factory_call = factory.calls[0]
        assert factory_call["session_id"] == "sess-1"
        assert factory_call["assistant_message_id"] == "msg-1"
        assert factory_call["rerank_model"] is not None

        assert len(engine.calls) == 1
        call = engine.calls[0]
        assert call["session_id"] == "sess-1"
        assert call["message_id"] == "msg-1"
        assert "[用户上传图片内容]" in call["query"]
        assert "a picture" in call["query"]
        assert "> quoted" in call["query"]
        assert call["image_urls"] == []
        assert call["llm_context"] == []
        assert rerank_service.calls == ["rerank-1"]
        assert history_loader.calls == [("sess-1", 3)]

    @pytest.mark.asyncio
    async def test_vision_routes_image_urls_to_engine(self) -> None:
        agent = _make_agent(multi_turn_enabled=False)
        req = _make_request(
            custom_agent=agent,
            image_urls=("https://img/1", "https://img/2"),
        )
        model_service = _StubModelService(
            models={
                "model-1": ModelInfo(
                    id="model-1", type="knowledge_qa", supports_vision=True
                )
            },
            chats={"model-1": _StubChat()},
        )
        engine = _RecordingEngine()
        factory = _StubEngineFactory(engine)
        bus = _RecordingBus()

        await run_agent_qa(
            ctx=_Ctx(),
            req=req,
            event_bus=bus,
            model_service=model_service,
            rerank_service=_StubRerankService(),
            tenant_loader=_StubTenantLoader(),
            history_loader=_StubHistoryLoader(),
            engine_factory=factory,
            skills_available=False,
        )

        assert engine.calls[0]["image_urls"] == [
            "https://img/1",
            "https://img/2",
        ]

    @pytest.mark.asyncio
    async def test_history_loader_failure_does_not_block(self) -> None:
        agent = _make_agent(multi_turn_enabled=True)
        req = _make_request(custom_agent=agent)
        model_service = _StubModelService(
            models={"model-1": ModelInfo(id="model-1", type="knowledge_qa")},
            chats={"model-1": _StubChat()},
        )
        engine = _RecordingEngine()
        factory = _StubEngineFactory(engine)
        bus = _RecordingBus()

        await run_agent_qa(
            ctx=_Ctx(),
            req=req,
            event_bus=bus,
            model_service=model_service,
            rerank_service=_StubRerankService(),
            tenant_loader=_StubTenantLoader(),
            history_loader=_StubHistoryLoader(fail=True),
            engine_factory=factory,
            skills_available=False,
        )

        assert engine.calls[0]["llm_context"] == []

    @pytest.mark.asyncio
    async def test_no_history_when_multiturn_disabled(self) -> None:
        agent = _make_agent(multi_turn_enabled=False)
        req = _make_request(custom_agent=agent)
        model_service = _StubModelService(
            models={"model-1": ModelInfo(id="model-1", type="knowledge_qa")},
            chats={"model-1": _StubChat()},
        )
        history_loader = _StubHistoryLoader()
        engine = _RecordingEngine()
        factory = _StubEngineFactory(engine)
        bus = _RecordingBus()

        await run_agent_qa(
            ctx=_Ctx(),
            req=req,
            event_bus=bus,
            model_service=model_service,
            rerank_service=_StubRerankService(),
            tenant_loader=_StubTenantLoader(),
            history_loader=history_loader,
            engine_factory=factory,
            skills_available=False,
        )

        assert history_loader.calls == []
        assert engine.calls[0]["llm_context"] == []

    @pytest.mark.asyncio
    async def test_engine_error_emits_error_event(self) -> None:
        agent = _make_agent(multi_turn_enabled=False)
        req = _make_request(custom_agent=agent)
        model_service = _StubModelService(
            models={"model-1": ModelInfo(id="model-1", type="knowledge_qa")},
            chats={"model-1": _StubChat()},
        )
        engine = _RecordingEngine(raises=RuntimeError("engine down"))
        factory = _StubEngineFactory(engine)
        bus = _RecordingBus()

        await run_agent_qa(
            ctx=_Ctx(),
            req=req,
            event_bus=bus,
            model_service=model_service,
            rerank_service=_StubRerankService(),
            tenant_loader=_StubTenantLoader(),
            history_loader=_StubHistoryLoader(),
            engine_factory=factory,
            skills_available=False,
        )

        error_events = [e for e in bus.events if e.type == EventType.ERROR]
        assert len(error_events) == 1
        error_data = error_events[0].data
        assert isinstance(error_data, dict)
        assert error_data["stage"] == "agent_execution"
        assert "engine down" in str(error_data["error"])

    @pytest.mark.asyncio
    async def test_no_rerank_when_kb_selection_none(self) -> None:
        agent = _make_agent(
            multi_turn_enabled=False, kb_selection_mode="none"
        )
        req = _make_request(custom_agent=agent)
        model_service = _StubModelService(
            models={"model-1": ModelInfo(id="model-1", type="knowledge_qa")},
            chats={"model-1": _StubChat()},
        )
        rerank_service = _StubRerankService()
        engine = _RecordingEngine()
        factory = _StubEngineFactory(engine)

        await run_agent_qa(
            ctx=_Ctx(),
            req=req,
            event_bus=_RecordingBus(),
            model_service=model_service,
            rerank_service=rerank_service,
            tenant_loader=_StubTenantLoader(),
            history_loader=_StubHistoryLoader(),
            engine_factory=factory,
            skills_available=False,
        )

        assert rerank_service.calls == []
        assert factory.calls[0]["rerank_model"] is None

    @pytest.mark.asyncio
    async def test_rerank_required_but_missing_raises(self) -> None:
        agent = _make_agent(multi_turn_enabled=False, rerank_model_id="")
        req = _make_request(custom_agent=agent)
        model_service = _StubModelService(
            models={"model-1": ModelInfo(id="model-1", type="knowledge_qa")},
            chats={"model-1": _StubChat()},
        )

        with pytest.raises(RuntimeError, match="rerank model is not configured"):
            await run_agent_qa(
                ctx=_Ctx(),
                req=req,
                event_bus=_RecordingBus(),
                model_service=model_service,
                rerank_service=_StubRerankService(),
                tenant_loader=_StubTenantLoader(),
                history_loader=_StubHistoryLoader(),
                engine_factory=_StubEngineFactory(_RecordingEngine()),
                skills_available=False,
            )

    @pytest.mark.asyncio
    async def test_tenant_loader_used_when_agent_tenant_differs(self) -> None:
        # Session tenant = 7, agent tenant = 42 -> loader should be called.
        agent = _make_agent(multi_turn_enabled=False)
        req = _make_request(custom_agent=agent)
        model_service = _StubModelService(
            models={"model-1": ModelInfo(id="model-1", type="knowledge_qa")},
            chats={"model-1": _StubChat()},
        )
        tenant_loader = _StubTenantLoader(
            TenantInfo(id=42, web_search_max_results=10)
        )

        await run_agent_qa(
            ctx=_Ctx(),
            req=req,
            event_bus=_RecordingBus(),
            model_service=model_service,
            rerank_service=_StubRerankService(),
            tenant_loader=tenant_loader,
            history_loader=_StubHistoryLoader(),
            engine_factory=_StubEngineFactory(_RecordingEngine()),
            skills_available=False,
        )

        assert tenant_loader.calls == [42]
