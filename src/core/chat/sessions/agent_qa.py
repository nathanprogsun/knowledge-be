"""Agent-QA orchestration — agent engine dispatch, tool execution, streaming.

Ports the upstream agent-QA service flow into a stateless orchestrator.
The caller supplies request-scoped inputs plus the heavy seams (model
service, tenant loader, history loader, engine factory), and this module
resolves the runtime ``AgentConfig``, loads multi-turn history, and
executes the ReAct engine with streaming through the event bus.

The module never touches storage or model providers directly. Every
side-effecting dependency is injected through a ``Protocol`` seam so the
orchestration stays unit-testable with tiny stubs. The assembly helpers
(``build_agent_config``, per-request scope narrowing, query building)
are pure functions of their inputs and can be exercised without any
service seam in scope.

Mirrors the upstream agent-QA service contract: a ``CustomAgent`` is
required; the chat model must resolve to a knowledge-QA model type; the
rerank model is fetched only when the effective scope actually runs
``knowledge_search``; multi-turn history is rebuilt from the message
store when ``multi_turn_enabled`` is set; vision images are routed
directly when the resolved model supports vision, otherwise the textual
description is appended to the query.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.ai.embedding.base import Context
from src.ai.llm.types import Chat, Message
from src.ai.rerank.base import Reranker
from src.app_context.request_context import get_tenant_id
from src.common.json import JsonObject
from src.core.agents.engine.loop import AgentEngine
from src.core.agents.engine.types import (
    DEFAULT_LLM_CALL_TIMEOUT,
    AgentConfig,
)
from src.core.agents.tools.base import (
    TOOL_DATA_ANALYSIS,
    TOOL_DATA_SCHEMA,
    TOOL_DATABASE_QUERY,
    TOOL_GET_DOCUMENT_INFO,
    TOOL_GREP_CHUNKS,
    TOOL_KNOWLEDGE_SEARCH,
    TOOL_LIST_KNOWLEDGE_CHUNKS,
    TOOL_QUERY_KNOWLEDGE_GRAPH,
    TOOL_THINKING,
    TOOL_TODO_WRITE,
)
from src.core.chat.bus import Event, EventBus
from src.core.chat.pipeline.common import build_attachments_prompt
from src.core.chat.pipeline.types import MessageAttachment, SearchTarget, SearchTargetType
from src.core.chat.types import EventType

logger = logging.getLogger(__name__)

#: Knowledge-QA model type used to validate model resolution for agent-QA.
MODEL_TYPE_KNOWLEDGE_QA = "knowledge_qa"

#: Default ceiling on web-search results when neither the agent nor the
#: tenant configures one.
DEFAULT_WEB_SEARCH_MAX_RESULTS = 5

#: Default history-turn count when ``multi_turn_enabled`` is on but
#: ``history_turns`` is unset.
DEFAULT_HISTORY_TURNS = 5

#: Default context-window budget for agent conversations.
DEFAULT_MAX_CONTEXT_TOKENS = 200_000

#: Directory containing the preloaded skill scripts.
DEFAULT_PRELOADED_SKILLS_DIR = "skills/preloaded"

#: Default allowed tool set when the custom agent does not declare one.
DEFAULT_ALLOWED_TOOLS: tuple[str, ...] = (
    TOOL_THINKING,
    TOOL_TODO_WRITE,
    TOOL_KNOWLEDGE_SEARCH,
    TOOL_GREP_CHUNKS,
    TOOL_LIST_KNOWLEDGE_CHUNKS,
    TOOL_QUERY_KNOWLEDGE_GRAPH,
    TOOL_GET_DOCUMENT_INFO,
    TOOL_DATABASE_QUERY,
    TOOL_DATA_ANALYSIS,
    TOOL_DATA_SCHEMA,
)


# ── Config-blob accessors ──────────────────────────────────────────────


def _config_str(config: JsonObject, key: str) -> str:
    """Return the string value of ``key``, or an empty string."""
    value = config.get(key)
    return value if isinstance(value, str) else ""


def _config_int(config: JsonObject, key: str) -> int:
    """Return the integer value of ``key``, or zero."""
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _config_float(config: JsonObject, key: str) -> float:
    """Return the float value of ``key``, or zero."""
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _config_bool(config: JsonObject, key: str) -> bool:
    """Return the boolean value of ``key``, or ``False``."""
    value = config.get(key)
    return value if isinstance(value, bool) else False


def _config_bool_or_none(config: JsonObject, key: str) -> bool | None:
    """Return the boolean value of ``key``, or ``None`` when unset / non-bool."""
    value = config.get(key)
    return value if isinstance(value, bool) else None


def _config_str_list(config: JsonObject, key: str) -> list[str]:
    """Return the list-of-strings value of ``key``, or an empty list."""
    value = config.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


# ── Request DTOs ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TagScope:
    """A tag-constrained knowledge-base scope from an @mention."""

    knowledge_base_id: str = ""
    tag_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentInfo:
    """The custom agent selected for the turn."""

    id: str
    tenant_id: int
    config: JsonObject


@dataclass(frozen=True, slots=True)
class TenantInfo:
    """Minimal tenant facts used for retrieval / web-search scope."""

    id: int
    web_search_max_results: int = 0


@dataclass(frozen=True, slots=True)
class AgentQARequest:
    """Request payload for an agent-QA turn.

    Field names match the upstream ``QARequest`` wire shape so the
    handler layer can project its request envelope onto this carrier
    without translation.
    """

    session_id: str
    session_tenant_id: int
    query: str
    assistant_message_id: str
    custom_agent: AgentInfo
    summary_model_id: str = ""
    shared_agent_read_only: bool = False
    knowledge_base_ids: tuple[str, ...] = ()
    knowledge_ids: tuple[str, ...] = ()
    tag_scopes: tuple[TagScope, ...] = ()
    mcp_service_ids: tuple[str, ...] = ()
    skill_names: tuple[str, ...] = ()
    image_urls: tuple[str, ...] = ()
    image_description: str = ""
    web_search_enabled: bool = False
    quoted_context: str = ""
    attachments: tuple[MessageAttachment, ...] = ()
    user_message_id: str = ""


# ── Injectable seams ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Minimal model facts consumed by the orchestrator.

    The full model row lives behind the model service; the orchestrator
    only needs the type check (knowledge-QA) and the vision flag.
    """

    id: str
    type: str = ""
    supports_vision: bool = False


@runtime_checkable
class ModelService(Protocol):
    """Resolves chat models and reads model metadata for the orchestrator."""

    async def get_chat_model(self, ctx: Context, model_id: str) -> Chat: ...

    async def get_model_by_id(self, ctx: Context, model_id: str) -> ModelInfo | None: ...


@runtime_checkable
class RerankModelService(Protocol):
    """Resolves a rerank model by id for the knowledge_search tool."""

    async def get_rerank_model(self, ctx: Context, model_id: str) -> Reranker: ...


@runtime_checkable
class AgentHistoryLoader(Protocol):
    """Loads multi-turn history for the LLM context."""

    async def load(self, ctx: Context, session_id: str, max_rounds: int) -> list[Message]: ...


@runtime_checkable
class TenantLoader(Protocol):
    """Loads tenant info by id for retrieval-scope resolution."""

    async def load(self, ctx: Context, tenant_id: int) -> TenantInfo | None: ...


@runtime_checkable
class AgentEngineFactory(Protocol):
    """Builds the configured ``AgentEngine`` for one turn.

    The factory wires the tool registry (with the rerank model bound into
    the ``knowledge_search`` tool), the model-context registry, and any
    knowledge-base / document metadata the engine needs at execution time.
    """

    def build(
        self,
        *,
        config: AgentConfig,
        chat_model: Chat,
        rerank_model: Reranker | None,
        event_bus: EventBus,
        session_id: str,
        assistant_message_id: str,
    ) -> AgentEngine: ...


# ── Pure helpers ───────────────────────────────────────────────────────


def default_allowed_tools() -> list[str]:
    """Return the default allowed tools list.

    Mirrors the upstream ``tools.DefaultAllowedTools()`` exactly — the
    tools every agent runs unless the custom agent overrides the
    whitelist via ``allowed_tools``.
    """
    return list(DEFAULT_ALLOWED_TOOLS)


def agent_requires_rerank_model(config: JsonObject) -> bool:
    """Return whether the agent's scope actually invokes the reranker.

    A knowledge-search-disabled agent never runs the ``knowledge_search``
    tool, so it does not need a rerank model even when the whitelist
    still names it. An empty / unset ``allowed_tools`` falls back to the
    default whitelist, which includes ``knowledge_search``.
    """
    if _config_str(config, "kb_selection_mode") == "none":
        return False
    allowed = _config_str_list(config, "allowed_tools")
    # An empty / unset whitelist falls back to the default tool set, which
    # includes knowledge_search (mirrors the upstream runtime fallback).
    if not allowed:
        allowed = default_allowed_tools()
    return TOOL_KNOWLEDGE_SEARCH in allowed


def dedup_preserving_order(values: Sequence[str]) -> list[str]:
    """Return ``values`` deduplicated, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def intersect_preserving_request_order(
    requested: Sequence[str],
    allowed: Sequence[str],
) -> list[str]:
    """Return requested values that are also in ``allowed``, preserving order."""
    allowed_set = {value for value in allowed if value}
    out: list[str] = []
    seen: set[str] = set()
    for value in requested:
        if not value or value in seen or value not in allowed_set:
            continue
        seen.add(value)
        out.append(value)
    return out


def unique_non_empty(values: Sequence[str]) -> list[str]:
    """Return ``values`` deduplicated, preserving order, dropping blanks."""
    return dedup_preserving_order(values)


def merge_resolved_tag_knowledge_ids(
    existing: Sequence[str],
    search_targets: Sequence[SearchTarget],
    tag_scopes: Sequence[TagScope],
) -> list[str]:
    """Merge tag-resolved knowledge ids into the pinned knowledge set.

    When tag scopes are present, the concrete knowledge ids those tags
    resolved to are appended onto the existing pinned set so the agent
    can see which documents the user explicitly chose this turn. KB
    scopes without tag mentions are kept as-is; knowledge-type targets
    outside the tag KBs are skipped.
    """
    tag_kbs = {
        scope.knowledge_base_id for scope in tag_scopes if scope.knowledge_base_id and scope.tag_ids
    }
    if not tag_kbs:
        return unique_non_empty(existing)
    merged = list(existing)
    for target in search_targets:
        if target is None:
            continue
        if target.knowledge_base_id not in tag_kbs:
            continue
        if target.type is not SearchTargetType.KNOWLEDGE:
            continue
        merged.extend(target.knowledge_ids)
    return unique_non_empty(merged)


def skills_from_agent_config(
    config: JsonObject,
    *,
    skills_available: bool,
) -> tuple[bool, list[str], list[str]]:
    """Resolve ``(skills_enabled, skill_dirs, allowed_skills)`` from the config.

    A disabled script-execution environment disables skills entirely
    regardless of the selection mode — a runtime that cannot execute a
    skill script cannot expose a skill. Selection mode:

    - ``all``: all preloaded skills are enabled;
    - ``selected``: only the explicitly selected skills;
    - ``none`` / unset / unknown: skills are disabled.
    """
    if not skills_available:
        return False, [], []
    mode = _config_str(config, "skills_selection_mode").strip().lower()
    if mode == "all":
        return True, [DEFAULT_PRELOADED_SKILLS_DIR], []
    if mode == "selected":
        selected = _config_str_list(config, "selected_skills")
        if selected:
            return True, [DEFAULT_PRELOADED_SKILLS_DIR], selected
        return False, [], []
    return False, [], []


def apply_per_request_skill_scope(
    *,
    requested: Sequence[str],
    skills_mode: str,
    skills_enabled: bool,
    allowed_skills: Sequence[str],
) -> tuple[bool, list[str]]:
    """Narrow the skill whitelist to the @Skill mentions for this turn.

    Returns the effective ``(skills_enabled, allowed_skills)``. The
    identity pair is returned when no skills were mentioned, when
    mention-driven scope is forbidden (``none`` / empty mode), or when
    skills are already disabled.
    """
    if not requested or not skills_enabled:
        return skills_enabled, list(allowed_skills)
    if skills_mode in ("", "none"):
        logger.warning(
            "Ignoring @skill mention: agent skills selection is disabled (mode=%s)",
            skills_mode,
        )
        return skills_enabled, list(allowed_skills)
    if skills_mode == "selected":
        effective = intersect_preserving_request_order(requested, allowed_skills)
        if not effective:
            return False, []
        return True, effective
    if skills_mode == "all":
        return True, dedup_preserving_order(requested)
    return skills_enabled, list(allowed_skills)


def apply_per_request_mcp_scope(
    *,
    requested: Sequence[str],
    agent_mcps: Sequence[str],
    selection_mode: str,
    is_shared_agent: bool,
) -> tuple[str, list[str]] | None:
    """Narrow MCP services to the @MCP mentions for this turn.

    Returns the effective ``(mcp_selection_mode, mcp_services)`` to
    apply, or ``None`` when the mention should be ignored (no mentions,
    ``none`` mode, or the narrowed set is empty outside an agent preset).
    """
    if not requested:
        return None
    if selection_mode == "none":
        logger.warning("Ignoring @MCP mention: agent MCP selection is disabled (mode=none)")
        return None
    mentioned = dedup_preserving_order(requested)
    effective, mode = _resolve_mcp_scope_inner(
        mentioned, agent_mcps, selection_mode, is_shared_agent
    )
    if not effective:
        logger.warning(
            "Ignoring @MCP scope outside agent preset: requested=%s agent=%s shared=%s",
            list(requested),
            list(agent_mcps),
            is_shared_agent,
        )
        return None
    return mode, effective


def _resolve_mcp_scope_inner(
    mentioned: Sequence[str],
    agent_mcps: Sequence[str],
    selection_mode: str,
    is_shared_agent: bool,
) -> tuple[list[str], str]:
    if is_shared_agent:
        mentioned = intersect_preserving_request_order(mentioned, agent_mcps)
        if not mentioned:
            return [], selection_mode
    if selection_mode == "none":
        return [], selection_mode
    if selection_mode == "selected":
        effective = intersect_preserving_request_order(mentioned, agent_mcps)
    else:
        effective = list(mentioned)
    if not effective:
        return [], selection_mode
    return effective, "selected"


def resolve_retrieval_tenant_id(
    *,
    session_tenant_id: int,
    agent_tenant_id: int,
    context_tenant_id: int | None = None,
) -> int:
    """Determine the tenant id used for retrieval scope.

    Priority: agent tenant > context tenant > session tenant. A
    zero-valued agent tenant falls through (custom agents created in
    the caller's tenant have ``tenant_id == 0`` until persisted).
    """
    if agent_tenant_id:
        return agent_tenant_id
    if context_tenant_id:
        return context_tenant_id
    return session_tenant_id


# ── Resolution helpers ─────────────────────────────────────────────────


async def resolve_chat_model_id(
    *,
    ctx: Context,
    req: AgentQARequest,
    knowledge_base_ids: Sequence[str],
    knowledge_ids: Sequence[str],
    model_service: ModelService,
) -> str:
    """Resolve the chat model id for an agent-QA turn.

    The custom agent's ``model_id`` is mandatory and must resolve to a
    knowledge-QA model type; a request-level ``summary_model_id`` may
    override it for this turn when the override also resolves to a
    knowledge-QA model. Without an agent (not reachable for agent-QA,
    but kept for the shared helper contract), the legacy KB / session /
    system fallback is unavailable — the function requires an agent.
    """
    agent_model_id = _config_str(req.custom_agent.config, "model_id").strip()
    if not agent_model_id:
        raise RuntimeError(
            f"chat model is not configured: please set model_id on agent {req.custom_agent.id}"
        )
    agent_info = await model_service.get_model_by_id(ctx, agent_model_id)
    if agent_info is None or agent_info.type != MODEL_TYPE_KNOWLEDGE_QA:
        raise RuntimeError(
            f"configured chat model {agent_model_id} is unavailable for agent {req.custom_agent.id}"
        )

    override = (req.summary_model_id or "").strip()
    if override:
        override_info = await model_service.get_model_by_id(ctx, override)
        if override_info is not None and override_info.type == MODEL_TYPE_KNOWLEDGE_QA:
            logger.info("Using request's summary model override: %s", override)
            return override
        logger.warning("Request provided invalid summary model ID %s, falling back", override)

    logger.info("Using custom agent's model_id: %s", agent_model_id)
    return agent_model_id


# ── Config assembly ────────────────────────────────────────────────────


def build_agent_config(
    *,
    custom_agent: AgentInfo,
    session_tenant_id: int,
    shared_agent_read_only: bool,
    web_search_enabled: bool,
    knowledge_base_ids: Sequence[str],
    knowledge_ids: Sequence[str],
    tag_scopes: Sequence[TagScope],
    search_targets: Sequence[SearchTarget],
    mcp_service_ids: Sequence[str],
    skill_names: Sequence[str],
    skills_available: bool,
    web_search_provider_id: str = "",
    web_search_max_results_default: int = 0,
    llm_call_timeout_default: float = 0.0,
) -> AgentConfig:
    """Build the runtime ``AgentConfig`` for an agent-QA turn.

    Pure function of its inputs: KB / tenant resolution and history
    loading happen in the caller. Mirrors the upstream
    ``buildAgentConfig`` control flow — fields are copied from the
    custom agent's ``config`` blob, the LLM-call timeout falls back to
    the global default, skills are enabled per the agent's selection
    mode, per-request @Skill / @MCP scopes narrow the whitelists, and
    the web-search ceiling falls back to the tenant value when the
    agent leaves it unset.
    """
    config_blob = custom_agent.config

    # ── skills ──────────────────────────────────────────────────────
    skills_enabled, skill_dirs, allowed_skills = skills_from_agent_config(
        config_blob, skills_available=skills_available
    )

    # ── allowed tools (custom override or default whitelist) ────────
    allowed_tools = _config_str_list(config_blob, "allowed_tools")
    if not allowed_tools:
        allowed_tools = default_allowed_tools()

    # ── per-request scope ───────────────────────────────────────────
    skills_mode = _config_str(config_blob, "skills_selection_mode").strip().lower()
    skills_enabled, allowed_skills = apply_per_request_skill_scope(
        requested=skill_names,
        skills_mode=skills_mode,
        skills_enabled=skills_enabled,
        allowed_skills=allowed_skills,
    )
    mcp_selection_mode = _config_str(config_blob, "mcp_selection_mode").strip().lower()
    agent_mcps = _config_str_list(config_blob, "mcp_services")
    mcp_scope = apply_per_request_mcp_scope(
        requested=mcp_service_ids,
        agent_mcps=agent_mcps,
        selection_mode=mcp_selection_mode,
        is_shared_agent=shared_agent_read_only,
    )
    if mcp_scope is not None:
        mcp_selection_mode, mcp_services = mcp_scope
    else:
        mcp_services = agent_mcps

    # ── system prompt ───────────────────────────────────────────────
    system_prompt = _config_str(config_blob, "system_prompt")
    use_custom_system_prompt = bool(system_prompt)

    # ── web search defaults (agent > tenant > global) ───────────────
    agent_max_results = _config_int(config_blob, "web_search_max_results")
    if agent_max_results > 0:
        web_search_max_results = agent_max_results
    elif web_search_max_results_default > 0:
        web_search_max_results = web_search_max_results_default
    else:
        web_search_max_results = DEFAULT_WEB_SEARCH_MAX_RESULTS

    # ── LLM-call timeout fallback ───────────────────────────────────
    llm_call_timeout = _config_float(config_blob, "llm_call_timeout")
    if llm_call_timeout <= 0:
        llm_call_timeout = llm_call_timeout_default
    if llm_call_timeout <= 0:
        llm_call_timeout = DEFAULT_LLM_CALL_TIMEOUT

    # ── knowledge ids (merge tag-resolved docs onto the pinned set) ─
    merged_knowledge_ids = merge_resolved_tag_knowledge_ids(
        knowledge_ids, search_targets, tag_scopes
    )

    return AgentConfig(
        max_iterations=_config_int(config_blob, "max_iterations"),
        allowed_tools=allowed_tools,
        temperature=_config_float(config_blob, "temperature"),
        knowledge_bases=list(knowledge_base_ids),
        knowledge_ids=merged_knowledge_ids,
        system_prompt=system_prompt,
        use_custom_system_prompt=use_custom_system_prompt,
        web_search_enabled=_config_bool(config_blob, "web_search_enabled") and web_search_enabled,
        web_search_max_results=web_search_max_results,
        web_search_provider_id=web_search_provider_id,
        multi_turn_enabled=_config_bool(config_blob, "multi_turn_enabled"),
        history_turns=_config_int(config_blob, "history_turns"),
        mcp_selection_mode=mcp_selection_mode,
        mcp_services=list(mcp_services),
        thinking=_config_bool_or_none(config_blob, "thinking"),
        citation_enabled=_config_bool_or_none(config_blob, "citation_enabled"),
        retrieve_kb_only_when_mentioned=_config_bool(
            config_blob, "retrieve_kb_only_when_mentioned"
        ),
        retain_retrieval_history=_config_bool(config_blob, "retain_retrieval_history"),
        skills_enabled=skills_enabled,
        skill_dirs=skill_dirs,
        allowed_skills=allowed_skills,
        llm_call_timeout=llm_call_timeout,
        max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS,
    )


# ── Query assembly ────────────────────────────────────────────────────


def build_agent_query(
    *,
    query: str,
    image_urls: Sequence[str],
    image_description: str,
    model_supports_vision: bool,
    quoted_context: str = "",
    attachments: Sequence[MessageAttachment] = (),
) -> tuple[str, list[str]]:
    """Assemble the user query and image list sent to the engine.

    Vision-capable models receive image URLs directly through the
    engine's image channel; non-vision models receive the textual
    description appended to the query. Quoted context (from IM
    quote-replies) and the attachments prompt are always appended to
    the query text — the engine has no separate input for them.
    """
    agent_query = query
    agent_image_urls: list[str] = []
    if model_supports_vision and image_urls:
        agent_image_urls = list(image_urls)
    elif image_description:
        agent_query = query + "\n\n[用户上传图片内容]\n" + image_description
    if quoted_context:
        agent_query += "\n\n" + quoted_context
    if attachments:
        agent_query += build_attachments_prompt(attachments)
    return agent_query, agent_image_urls


# ── High-level entry point ────────────────────────────────────────────


async def run_agent_qa(
    *,
    ctx: Context,
    req: AgentQARequest,
    event_bus: EventBus,
    model_service: ModelService,
    rerank_service: RerankModelService,
    tenant_loader: TenantLoader,
    history_loader: AgentHistoryLoader,
    engine_factory: AgentEngineFactory,
    skills_available: bool = True,
    llm_call_timeout_default: float = 0.0,
    knowledge_base_ids: Sequence[str] | None = None,
    knowledge_ids: Sequence[str] | None = None,
    search_targets: Sequence[SearchTarget] = (),
    web_search_provider_id: str = "",
) -> None:
    """Execute one agent-QA turn with streaming.

    Resolves the retrieval tenant, builds the runtime ``AgentConfig``,
    fetches the chat / rerank models, loads multi-turn history, creates
    the engine via the injected factory, and executes the engine.
    Execution errors are converted into a single ``error`` event on the
    bus; the function itself does not raise on engine failure (events
    are consumed by the handler-layer bus subscription).

    ``knowledge_base_ids`` / ``knowledge_ids`` may be passed in when the
    caller has already resolved the KB scope; when both are ``None`` the
    values carried on ``req`` are used.

    A custom agent is mandatory for the agent path; the handler layer
    guarantees its presence before calling this entry point.
    """
    # ── tenant + retrieval scope ────────────────────────────────────
    agent_tenant_id = int(req.custom_agent.tenant_id)
    context_tenant = get_tenant_id()
    context_tenant_id = int(context_tenant) if context_tenant else None
    retrieval_tenant_id = resolve_retrieval_tenant_id(
        session_tenant_id=req.session_tenant_id,
        agent_tenant_id=agent_tenant_id,
        context_tenant_id=context_tenant_id,
    )

    tenant: TenantInfo = TenantInfo(id=retrieval_tenant_id)
    if retrieval_tenant_id != req.session_tenant_id or context_tenant is None:
        loaded = await tenant_loader.load(ctx, retrieval_tenant_id)
        if loaded is not None:
            tenant = loaded
            logger.info(
                "Using agent tenant info for retrieval scope, tenant ID: %d",
                retrieval_tenant_id,
            )
        else:
            logger.warning(
                "Tenant info not available for agent tenant %d, proceeding with defaults",
                retrieval_tenant_id,
            )

    # ── KB scope ────────────────────────────────────────────────────
    kb_ids = (
        list(knowledge_base_ids) if knowledge_base_ids is not None else list(req.knowledge_base_ids)
    )
    doc_ids = list(knowledge_ids) if knowledge_ids is not None else list(req.knowledge_ids)

    # ── config assembly ────────────────────────────────────────────
    config = build_agent_config(
        custom_agent=req.custom_agent,
        session_tenant_id=req.session_tenant_id,
        shared_agent_read_only=req.shared_agent_read_only,
        web_search_enabled=req.web_search_enabled,
        knowledge_base_ids=kb_ids,
        knowledge_ids=doc_ids,
        tag_scopes=req.tag_scopes,
        search_targets=search_targets,
        mcp_service_ids=req.mcp_service_ids,
        skill_names=req.skill_names,
        skills_available=skills_available,
        web_search_provider_id=web_search_provider_id,
        web_search_max_results_default=tenant.web_search_max_results,
        llm_call_timeout_default=llm_call_timeout_default,
    )

    # ── chat model resolution ───────────────────────────────────────
    effective_model_id = await resolve_chat_model_id(
        ctx=ctx,
        req=req,
        knowledge_base_ids=kb_ids,
        knowledge_ids=doc_ids,
        model_service=model_service,
    )
    if not effective_model_id:
        raise RuntimeError(
            f"summary model (model_id) is not configured in custom agent settings: {req.custom_agent.id}"
        )

    summary_model = await model_service.get_chat_model(ctx, effective_model_id)

    # ── rerank model (only when the effective scope actually uses it) ──
    rerank_model: Reranker | None = None
    if agent_requires_rerank_model(req.custom_agent.config):
        rerank_model_id = _config_str(req.custom_agent.config, "rerank_model_id").strip()
        if not rerank_model_id:
            raise RuntimeError(
                "rerank model is not configured: please set rerank_model_id on the agent"
            )
        rerank_model = await rerank_service.get_rerank_model(ctx, rerank_model_id)
    else:
        logger.info(
            "knowledge_search is unavailable for the effective agent scope, skipping rerank model initialization"
        )

    # ── multi-turn history ─────────────────────────────────────────
    if config.multi_turn_enabled:
        history_turns = config.history_turns
        if history_turns <= 0:
            history_turns = DEFAULT_HISTORY_TURNS
        try:
            llm_context = await history_loader.load(ctx, req.session_id, history_turns)
        except Exception as exc:
            logger.warning(
                "Failed to load agent history from DB: %s, continuing without history",
                exc,
            )
            llm_context = []
        logger.info(
            "Loaded %d history messages from DB (turns=%d)",
            len(llm_context),
            history_turns,
        )
    else:
        logger.info("Multi-turn disabled for this agent, running without history")
        llm_context = []

    # ── engine ──────────────────────────────────────────────────────
    engine = engine_factory.build(
        config=config,
        chat_model=summary_model,
        rerank_model=rerank_model,
        event_bus=event_bus,
        session_id=req.session_id,
        assistant_message_id=req.assistant_message_id,
    )

    # ── vision routing ──────────────────────────────────────────────
    model_info = await model_service.get_model_by_id(ctx, effective_model_id)
    model_supports_vision = bool(model_info is not None and model_info.supports_vision)
    agent_query, agent_image_urls = build_agent_query(
        query=req.query,
        image_urls=req.image_urls,
        image_description=req.image_description,
        model_supports_vision=model_supports_vision,
        quoted_context=req.quoted_context,
        attachments=req.attachments,
    )

    # ── execute (errors are surfaced via the bus) ──────────────────
    try:
        await engine.execute(
            ctx,
            query=agent_query,
            session_id=req.session_id,
            message_id=req.assistant_message_id,
            llm_context=llm_context,
            image_urls=agent_image_urls,
        )
    except Exception as exc:
        logger.error("Agent execution failed: %s", exc)
        await event_bus.emit(
            Event(
                type=EventType.ERROR,
                session_id=req.session_id,
                data={
                    "error": str(exc),
                    "stage": "agent_execution",
                    "session_id": req.session_id,
                },
            )
        )


__all__ = [
    "DEFAULT_ALLOWED_TOOLS",
    "DEFAULT_HISTORY_TURNS",
    "DEFAULT_MAX_CONTEXT_TOKENS",
    "DEFAULT_PRELOADED_SKILLS_DIR",
    "DEFAULT_WEB_SEARCH_MAX_RESULTS",
    "MODEL_TYPE_KNOWLEDGE_QA",
    "AgentEngineFactory",
    "AgentHistoryLoader",
    "AgentInfo",
    "AgentQARequest",
    "ModelInfo",
    "ModelService",
    "RerankModelService",
    "TagScope",
    "TenantInfo",
    "TenantLoader",
    "agent_requires_rerank_model",
    "apply_per_request_mcp_scope",
    "apply_per_request_skill_scope",
    "build_agent_config",
    "build_agent_query",
    "dedup_preserving_order",
    "default_allowed_tools",
    "intersect_preserving_request_order",
    "merge_resolved_tag_knowledge_ids",
    "resolve_chat_model_id",
    "resolve_retrieval_tenant_id",
    "run_agent_qa",
    "skills_from_agent_config",
    "unique_non_empty",
]
