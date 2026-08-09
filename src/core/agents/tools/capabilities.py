"""Per-tool capability requirements and knowledge-base filter derivation.

Maps built-in tool names to the knowledge-base capabilities they need
(any-of / all-of / file consumption). The same map backs the KB filter
used to prune out-of-capability knowledge bases from an agent's scope and
the decision of whether the chat input offers the ``@file`` picker.

Tools absent from the map default to "no requirement" and are treated as
always available / file-consuming (permissive fallback so unknown custom
tools never silently break).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class KBCapability(StrEnum):
    """Capability flags a knowledge base can expose."""

    VECTOR = "vector"
    KEYWORD = "keyword"
    WIKI = "wiki"
    GRAPH = "graph"
    FAQ = "faq"


@dataclass(frozen=True, slots=True)
class KBCapabilities:
    """Computed capability flags of one knowledge base."""

    vector: bool = False
    keyword: bool = False
    wiki: bool = False
    graph: bool = False
    faq: bool = False


@dataclass(frozen=True, slots=True)
class ToolRequirement:
    """What one tool needs from the knowledge-base scope.

    ``any_of``: the scope must expose at least one listed capability.
    ``all_of``: the scope must expose every listed capability.
    ``consumes_files``: the tool reads user-provided file references from
    the mention list; used to decide whether to offer the file picker.
    """

    any_of: tuple[KBCapability, ...] = ()
    all_of: tuple[KBCapability, ...] = ()
    consumes_files: bool = False


#: Tool name -> capability requirement. Keep aligned with the registry.
TOOL_CAPABILITY_REQUIREMENTS: dict[str, ToolRequirement] = {
    # Base / reasoning tools (no KB dependency, no file consumption).
    "thinking": ToolRequirement(),
    "todo_write": ToolRequirement(),
    # RAG / chunk retrieval (need at least one chunk-indexed KB).
    "knowledge_search": ToolRequirement(
        any_of=(KBCapability.VECTOR, KBCapability.KEYWORD),
        consumes_files=True,
    ),
    "grep_chunks": ToolRequirement(
        any_of=(KBCapability.VECTOR, KBCapability.KEYWORD),
        consumes_files=True,
    ),
    "list_knowledge_chunks": ToolRequirement(
        any_of=(KBCapability.VECTOR, KBCapability.KEYWORD),
        consumes_files=True,
    ),
    "query_knowledge_graph": ToolRequirement(
        any_of=(KBCapability.VECTOR, KBCapability.KEYWORD),
        consumes_files=True,
    ),
    "get_document_info": ToolRequirement(
        any_of=(KBCapability.VECTOR, KBCapability.KEYWORD),
        consumes_files=True,
    ),
    "database_query": ToolRequirement(
        any_of=(KBCapability.VECTOR, KBCapability.KEYWORD),
        consumes_files=True,
    ),
    # Wiki tools (operate on wiki pages; don't consume arbitrary file ids).
    "wiki_search": ToolRequirement(all_of=(KBCapability.WIKI,)),
    "wiki_read_page": ToolRequirement(all_of=(KBCapability.WIKI,)),
    "wiki_read_source_doc": ToolRequirement(all_of=(KBCapability.WIKI,)),
    "wiki_flag_issue": ToolRequirement(all_of=(KBCapability.WIKI,)),
    "wiki_write_page": ToolRequirement(all_of=(KBCapability.WIKI,)),
    "wiki_replace_text": ToolRequirement(all_of=(KBCapability.WIKI,)),
    "wiki_rename_page": ToolRequirement(all_of=(KBCapability.WIKI,)),
    "wiki_delete_page": ToolRequirement(all_of=(KBCapability.WIKI,)),
    "wiki_read_issue": ToolRequirement(all_of=(KBCapability.WIKI,)),
    "wiki_update_issue": ToolRequirement(all_of=(KBCapability.WIKI,)),
    # Data analysis (reads table summary / column chunks from RAG ingest).
    "data_analysis": ToolRequirement(
        any_of=(KBCapability.VECTOR, KBCapability.KEYWORD),
        consumes_files=True,
    ),
    "data_schema": ToolRequirement(
        any_of=(KBCapability.VECTOR, KBCapability.KEYWORD),
        consumes_files=True,
    ),
}

#: The implicit capability requirement of quick-answer (RAG) agent mode.
#: Quick-answer drives retrieval purely through vector / keyword chunk
#: search, so a knowledge base with neither index cannot contribute
#: context and is filtered out wherever the user can pick one.


@dataclass(frozen=True, slots=True)
class KBFilter:
    """Derived "KB must expose at least one of these" predicate."""

    any_of: tuple[KBCapability, ...] = ()

    def is_empty(self) -> bool:
        """Whether the filter imposes no constraint."""
        return not self.any_of


#: Quick-answer (RAG) agent mode implies a vector|keyword capability filter.
QUICK_ANSWER_KB_FILTER = KBFilter(any_of=(KBCapability.VECTOR, KBCapability.KEYWORD))


def _has_cap(caps: KBCapabilities, capability: KBCapability) -> bool:
    if capability is KBCapability.VECTOR:
        return caps.vector
    if capability is KBCapability.KEYWORD:
        return caps.keyword
    if capability is KBCapability.WIKI:
        return caps.wiki
    if capability is KBCapability.GRAPH:
        return caps.graph
    if capability is KBCapability.FAQ:
        return caps.faq
    return False


def derive_kb_filter_from_tools(allowed_tools: list[str]) -> KBFilter:
    """Derive a capability filter such that a KB passes iff at least one
    allowed tool has a requirement the KB satisfies.

    Tools without any KB requirement don't contribute; if the allowed-tools
    list contains only such tools the returned filter is empty (accept all).
    """
    seen: set[KBCapability] = set()
    for tool_name in allowed_tools:
        requirement = TOOL_CAPABILITY_REQUIREMENTS.get(tool_name)
        if requirement is None:
            continue
        seen.update(requirement.any_of)
        seen.update(requirement.all_of)
    if not seen:
        return KBFilter()
    return KBFilter(any_of=tuple(seen))


def kb_satisfies_tool_requirements(
    caps: KBCapabilities,
    allowed_tools: list[str],
) -> bool:
    """Whether a single KB is compatible with the agent's tool set."""
    kb_filter = derive_kb_filter_from_tools(allowed_tools)
    if kb_filter.is_empty():
        return True
    return any(_has_cap(caps, capability) for capability in kb_filter.any_of)


def derive_kb_filter_for_agent(agent_mode: str, allowed_tools: list[str]) -> KBFilter:
    """Derive the effective KB filter for an agent configuration.

    Combines the implicit constraint from ``agent_mode`` (quick-answer
    forces vector|keyword) with the tool-derived filter. The result is the
    union of both contributions under the existing any-of semantics.
    """
    seen: set[KBCapability] = set()
    if agent_mode == "quick-answer":
        seen.update(QUICK_ANSWER_KB_FILTER.any_of)
    seen.update(derive_kb_filter_from_tools(allowed_tools).any_of)
    if not seen:
        return KBFilter()
    return KBFilter(any_of=tuple(seen))


def kb_satisfies_agent_requirements(
    caps: KBCapabilities,
    agent_mode: str,
    allowed_tools: list[str],
) -> bool:
    """Agent-aware KB compatibility check (also enforces mode constraints)."""
    kb_filter = derive_kb_filter_for_agent(agent_mode, allowed_tools)
    if kb_filter.is_empty():
        return True
    return any(_has_cap(caps, capability) for capability in kb_filter.any_of)


def tools_consume_files(allowed_tools: list[str]) -> bool:
    """Whether any allowed tool can use user-provided file references.

    An empty list is treated as "unknown -> permissive"; unknown tool names
    are assumed file-consuming so the file picker is never hidden from a
    user who just added a custom tool.
    """
    if not allowed_tools:
        return True
    for tool_name in allowed_tools:
        requirement = TOOL_CAPABILITY_REQUIREMENTS.get(tool_name)
        if requirement is None:
            return True
        if requirement.consumes_files:
            return True
    return False


__all__ = [
    "QUICK_ANSWER_KB_FILTER",
    "TOOL_CAPABILITY_REQUIREMENTS",
    "KBCapabilities",
    "KBCapability",
    "KBFilter",
    "ToolRequirement",
    "derive_kb_filter_for_agent",
    "derive_kb_filter_from_tools",
    "kb_satisfies_agent_requirements",
    "kb_satisfies_tool_requirements",
    "tools_consume_files",
]
