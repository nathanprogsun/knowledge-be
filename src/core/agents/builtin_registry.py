"""Built-in agent presets — registry-backed defaults.

The built-in presets mirror the upstream registry: a fixed display
order plus per-preset default name / description / avatar / config.
Until the registry is fully ported, this module is the single source
of truth the service consults when a built-in id has no customized row
in storage.

The config blobs are the upstream defaults verbatim; the service applies
its own config-defaulting pass on top, so the blobs only need to carry
the preset-specific values.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.core.agents.types import CustomAgentInfo
from src.common.json import JsonObject

# Fixed display order of the built-in presets in the tenant agent list.
# The wiki fixer is intentionally excluded: it is an internal agent
# invoked programmatically from the wiki editor and should not clutter
# the agent picker (it stays resolvable by id).
BUILTIN_AGENT_ORDER: tuple[str, ...] = (
    "builtin-quick-answer",
    "builtin-smart-reasoning",
    "builtin-wiki-researcher",
    "builtin-data-analyst",
)

# Per-preset defaults: name / description / avatar / config.
_BUILTIN_AGENT_DEFAULTS: dict[str, JsonObject] = {
    "builtin-quick-answer": {
        "name": "快速问答",
        "description": "基于知识库的 RAG 问答，快速准确地回答问题",
        "avatar": "",
        "config": {
            "agent_mode": "quick-answer",
            "system_prompt_id": "default_kb",
            "context_template_id": "default_context",
            "temperature": 0.7,
            "max_completion_tokens": 2048,
            "web_search_enabled": True,
            "web_search_max_results": 5,
            "multi_turn_enabled": True,
            "history_turns": 5,
            "kb_selection_mode": "all",
            "retrieve_kb_only_when_mentioned": False,
            "faq_priority_enabled": True,
            "faq_direct_answer_threshold": 0.9,
            "faq_score_boost": 1.2,
            "embedding_top_k": 10,
            "keyword_threshold": 0.3,
            "vector_threshold": 0.5,
            "rerank_top_k": 10,
            "rerank_threshold": 0.3,
            "enable_query_expansion": True,
            "enable_rewrite": True,
            "fallback_strategy": "model",
        },
    },
    "builtin-smart-reasoning": {
        "name": "智能推理",
        "description": "ReAct 推理框架，支持多步思考与工具调用",
        "avatar": "",
        "config": {
            "agent_mode": "smart-reasoning",
            "agent_type": "rag-qa",
            "system_prompt": "",
            "temperature": 0.7,
            "max_completion_tokens": 2048,
            "max_iterations": 50,
            "kb_selection_mode": "all",
            "retrieve_kb_only_when_mentioned": False,
            "allowed_tools": [
                "knowledge_search",
                "grep_chunks",
                "list_knowledge_chunks",
                "query_knowledge_graph",
                "get_document_info",
            ],
            "web_search_enabled": True,
            "web_search_max_results": 5,
            "reflection_enabled": False,
            "multi_turn_enabled": True,
            "history_turns": 5,
            "faq_priority_enabled": True,
            "faq_direct_answer_threshold": 0.9,
            "faq_score_boost": 1.2,
            "embedding_top_k": 10,
            "keyword_threshold": 0.3,
            "vector_threshold": 0.5,
            "rerank_top_k": 10,
            "rerank_threshold": 0.3,
        },
    },
    "builtin-wiki-researcher": {
        "name": "维基问答",
        "description": "专注于在 Wiki 知识库中回答问题的智能体",
        "avatar": "📚",
        "config": {
            "agent_mode": "smart-reasoning",
            "agent_type": "wiki-qa",
            "system_prompt_id": "wiki_researcher",
            "temperature": 0.7,
            "max_completion_tokens": 4096,
            "max_iterations": 30,
            "kb_selection_mode": "all",
            "retrieve_kb_only_when_mentioned": False,
            "allowed_tools": [
                "wiki_search",
                "wiki_read_page",
                "wiki_read_source_doc",
                "wiki_flag_issue",
            ],
            "web_search_enabled": False,
            "web_search_max_results": 0,
            "reflection_enabled": False,
            "multi_turn_enabled": True,
            "history_turns": 10,
            "embedding_top_k": 10,
            "keyword_threshold": 0.3,
            "vector_threshold": 0.5,
            "rerank_top_k": 10,
            "rerank_threshold": 0.3,
        },
    },
    "builtin-data-analyst": {
        "name": "数据分析师",
        "description": "专业的数据分析智能体，支持对 CSV/Excel 文件进行 SQL 查询和统计分析",
        "avatar": "📊",
        "config": {
            "agent_mode": "smart-reasoning",
            "agent_type": "data-analysis",
            "system_prompt_id": "data_analyst",
            "temperature": 0.3,
            "max_completion_tokens": 4096,
            "max_iterations": 30,
            "kb_selection_mode": "all",
            "retrieve_kb_only_when_mentioned": False,
            "supported_file_types": ["csv", "xlsx"],
            "allowed_tools": ["data_schema", "data_analysis"],
            "web_search_enabled": False,
            "web_search_max_results": 0,
            "reflection_enabled": True,
            "multi_turn_enabled": True,
            "history_turns": 10,
            "embedding_top_k": 5,
            "keyword_threshold": 0.3,
            "vector_threshold": 0.5,
            "rerank_top_k": 5,
            "rerank_threshold": 0.3,
        },
    },
}


def get_builtin_agent(agent_id: str, tenant_id: int) -> CustomAgentInfo | None:
    """Return the default preset for ``agent_id``, or ``None``.

    The returned info is a fresh projection per call (immutable model,
    no shared mutable state). ``tenant_id`` scopes the row to the
    caller's workspace; timestamps are pinned to the call time.
    """
    defaults = _BUILTIN_AGENT_DEFAULTS.get(agent_id)
    if defaults is None:
        return None
    now = datetime.now(UTC)
    return CustomAgentInfo(
        id=agent_id,
        name=defaults["name"],
        description=defaults["description"],
        avatar=defaults["avatar"],
        is_builtin=True,
        tenant_id=tenant_id,
        created_by=None,
        config=defaults["config"],
        created_at=now,
        updated_at=now,
    )


__all__ = [
    "BUILTIN_AGENT_ORDER",
    "get_builtin_agent",
]
