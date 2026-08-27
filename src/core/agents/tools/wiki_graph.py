"""Knowledge-graph query agent tool.

``query_knowledge_graph`` explores entity relationships across knowledge
bases that have graph extraction configured. Each KB is queried with a
hybrid search whose hits are deduplicated across KBs and sorted by
relevance; KBs without a graph config fall back to regular search results,
and per-KB failures are collected instead of aborting the call.

When the tool is constructed with search targets (agent scope enforced),
every result is additionally filtered through the document/tag whitelist
before it reaches the model, because the graph backend can only query by
knowledge base.

The tool executes against injected seams — a single-KB loader, a per-KB
search runner, and an optional document lookup for scope filtering — so
tests never touch a vector store.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import cast

from src.ai.embedding.base import Context
from src.common.exception import ApplicationError
from src.common.json import JsonObject, JsonValue
from src.core.agents.tools.base import (
    ToolDefinition,
    ToolResult,
    format_match_type,
    get_relevance_level,
)
from src.core.agents.tools.kb_tool import SearchCall, SearchRunner
from src.core.agents.tools.scope_auth import (
    KnowledgeLookup,
    KnowledgeTagsFetcher,
    filter_search_results_in_search_targets,
    validate_knowledge_base_ids_in_search_targets,
)
from src.core.agents.tools.search_target import SearchTargets
from src.core.agents.tools.text_utils import dedup_non_empty_strings
from src.core.agents.tools.wiki_route import GraphKbLoader
from src.core.knowledge.knowledge_bases.hybrid_search import SearchResult
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo

#: Tool name constant (kept here to avoid a dependency cycle with base).
WIKI_GRAPH_TOOL_NAME = "query_knowledge_graph"

#: Per-KB hybrid-search match count used by the graph tool.
GRAPH_SEARCH_MATCH_COUNT = 10

#: Upper bound on the number of knowledge bases queried in one call.
GRAPH_MAX_KB_IDS = 10

WIKI_GRAPH_TOOL_DESCRIPTION = (
    "Query knowledge graph to explore entity relationships and knowledge networks.\n"
    "\n"
    "## Core Function\n"
    "Explores relationships between entities in knowledge bases that have graph extraction configured.\n"
    "\n"
    "## When to Use\n"
    "✅ **Use for**:\n"
    '- Understanding relationships between entities (e.g., "relationship between Docker and Kubernetes")\n'
    "- Exploring knowledge networks and concept associations\n"
    "- Finding related information about specific entities\n"
    "- Understanding technical architecture and system relationships\n"
    "\n"
    "❌ **Don't use for**:\n"
    "- General text search → use knowledge_search\n"
    "- Knowledge base without graph extraction configured\n"
    "- Need exact document content → use knowledge_search\n"
    "\n"
    "## Parameters\n"
    "- **knowledge_base_ids** (required): Array of short bN knowledge base IDs (1-10). "
    "Only KBs with graph extraction configured will be effective.\n"
    "- **query** (required): Query content - can be entity name, relationship query, or concept search.\n"
    "\n"
    "## Graph Configuration\n"
    "Knowledge graph must be pre-configured in knowledge bases:\n"
    '- **Entity types** (Nodes): e.g., "Technology", "Tool", "Concept"\n'
    '- **Relationship types** (Relations): e.g., "depends_on", "uses", "contains"\n'
    "\n"
    "If KB is not configured with graph, tool will return regular search results.\n"
    "\n"
    "## Workflow\n"
    "1. **Relationship exploration**: query_knowledge_graph → list_knowledge_chunks (for detailed content)\n"
    "2. **Network analysis**: query_knowledge_graph → knowledge_search (for comprehensive understanding)\n"
    "3. **Topic research**: knowledge_search → query_knowledge_graph (for deep entity relationships)\n"
    "\n"
    "## Notes\n"
    "- Results indicate graph configuration status\n"
    "- Cross-KB results are automatically deduplicated\n"
    "- Results are sorted by relevance"
)

WIKI_GRAPH_TOOL_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "knowledge_base_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of short bN knowledge base IDs to query (1-10)",
            },
            "query": {
                "type": "string",
                "description": "Query content (entity name or query text)",
            },
        },
        "required": ["knowledge_base_ids", "query"],
    },
    ensure_ascii=False,
)


def build_wiki_graph_definition() -> ToolDefinition:
    """Return the default tool definition for the knowledge-graph tool."""
    return ToolDefinition(
        name=WIKI_GRAPH_TOOL_NAME,
        description=WIKI_GRAPH_TOOL_DESCRIPTION,
        parameters=WIKI_GRAPH_TOOL_SCHEMA,
    )


@dataclass(frozen=True, slots=True)
class GraphConfigSummary:
    """The entity/relationship vocabulary of a KB's graph extraction."""

    nodes: list[str]
    relations: list[str]


def _node_names(extract: JsonObject) -> list[str]:
    """Extract node names from a raw extract-config blob (lenient)."""
    raw = extract.get("nodes")
    names: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item.strip():
                names.append(item.strip())
            elif isinstance(item, dict):
                name = item.get("name")
                if isinstance(name, str) and name.strip():
                    names.append(name.strip())
    return names


def _relation_names(extract: JsonObject) -> list[str]:
    """Extract relation type names from a raw extract-config blob (lenient)."""
    raw = extract.get("relations")
    names: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item.strip():
                names.append(item.strip())
            elif isinstance(item, dict):
                rel_type = item.get("type")
                if isinstance(rel_type, str) and rel_type.strip():
                    names.append(rel_type.strip())
    return names


def _extract_config(kb: KnowledgeBaseInfo | None) -> JsonObject | None:
    if kb is None:
        return None
    config = kb.extract_config
    return config if isinstance(config, dict) else None


def summarize_graph_config(config: JsonObject | None) -> GraphConfigSummary:
    """Return the unique sorted node/relation names of an extract config."""
    if config is None:
        return GraphConfigSummary(nodes=[], relations=[])
    nodes = dedup_non_empty_strings(_node_names(config))
    relations = dedup_non_empty_strings(_relation_names(config))
    nodes.sort()
    relations.sort()
    return GraphConfigSummary(nodes=nodes, relations=relations)


def graph_configs_to_data(
    graph_configs: dict[str, GraphConfigSummary],
) -> dict[str, JsonObject] | None:
    """Project graph-config summaries onto the structured payload."""
    if not graph_configs:
        return None
    data: dict[str, JsonObject] = {}
    for kb_id, config in graph_configs.items():
        data[kb_id] = {
            "nodes": cast("list[JsonValue]", config.nodes),
            "relations": cast("list[JsonValue]", config.relations),
        }
    return data


def aggregate_graph_config(
    graph_configs: dict[str, GraphConfigSummary],
) -> JsonObject | None:
    """Merge every KB's node/relation vocabulary, deduplicated and sorted."""
    if not graph_configs:
        return None
    merged = GraphConfigSummary(nodes=[], relations=[])
    for config in graph_configs.values():
        merged = GraphConfigSummary(
            nodes=[*merged.nodes, *config.nodes],
            relations=[*merged.relations, *config.relations],
        )
    return {
        "nodes": cast("list[JsonValue]", dedup_non_empty_strings(merged.nodes)),
        "relations": cast("list[JsonValue]", dedup_non_empty_strings(merged.relations)),
    }


def build_graph_visualization_data(results: list[SearchResult]) -> JsonObject:
    """Build a simple node/edge payload for frontend visualization."""
    nodes: list[JsonObject] = []
    edges: list[JsonObject] = []
    seen: set[str] = set()
    for i, result in enumerate(results):
        if result.id in seen:
            continue
        seen.add(result.id)
        nodes.append(
            {
                "id": result.id,
                "label": f"Chunk {i + 1}",
                "content": result.content,
                "kb_id": result.knowledge_id,
                "kb_title": result.knowledge_title,
                "score": result.score,
                "type": "chunk",
            }
        )
    return {
        "nodes": cast("list[JsonValue]", nodes),
        "edges": cast("list[JsonValue]", edges),
        "total_nodes": len(nodes),
        "total_edges": len(edges),
    }


class QueryKnowledgeGraphTool:
    """Queries the knowledge graph for entities and relationships."""

    def __init__(
        self,
        *,
        definition: ToolDefinition,
        kb_loader: GraphKbLoader,
        search_runner: SearchRunner,
        search_targets: SearchTargets | None = None,
        knowledge_service: KnowledgeLookup | None = None,
        tag_fetcher: KnowledgeTagsFetcher | None = None,
    ) -> None:
        self._definition = definition
        self._kb_loader = kb_loader
        self._search_runner = search_runner
        self._search_targets = search_targets
        self._knowledge_service = knowledge_service
        self._tag_fetcher = tag_fetcher

    def name(self) -> str:
        return self._definition.name

    def description(self) -> str:
        return self._definition.description

    def parameters(self) -> str:
        return self._definition.parameters

    async def execute(self, ctx: Context, args: str) -> ToolResult:
        """Query every KB, deduplicate the hits, and render the result."""
        raw = _parse_args(args)
        kb_ids = dedup_non_empty_strings(_as_str_list(raw.get("knowledge_base_ids")))
        if not kb_ids:
            return ToolResult(
                success=False,
                error="knowledge_base_ids is required and must be a non-empty array",
            )
        if len(kb_ids) > GRAPH_MAX_KB_IDS:
            return ToolResult(
                success=False,
                error="knowledge_base_ids must contain at most 10 KB IDs",
            )
        query = _as_str(raw.get("query")).strip()
        if self._search_targets is not None:
            try:
                validate_knowledge_base_ids_in_search_targets(self._search_targets, kb_ids)
            except ApplicationError as exc:
                return ToolResult(success=False, error=exc.message)
        if not query:
            return ToolResult(success=False, error="query is required")

        errors: list[str] = []
        graph_configs: dict[str, GraphConfigSummary] = {}
        kb_counts: dict[str, int] = {}
        seen_chunks: dict[str, SearchResult] = {}

        for kb_id in kb_ids:
            kb, load_error = await self._load_kb(kb_id)
            if load_error:
                errors.append(f"KB {kb_id}: {load_error}")
                continue
            config = _extract_config(kb)
            summary = summarize_graph_config(config)
            if not summary.nodes and not summary.relations:
                errors.append(f"KB {kb_id}: graph extraction not configured")
                continue

            results = await self._search_kb(ctx, kb_id, query, errors)
            if results is None:
                continue
            if self._search_targets is not None:
                try:
                    results = await filter_search_results_in_search_targets(
                        ctx,
                        self._search_targets,
                        kb_id,
                        list(results),
                        self._knowledge_service,
                        self._tag_fetcher,
                    )
                except ApplicationError as exc:
                    errors.append(f"KB {kb_id}: {exc.message}")
                    continue
            graph_configs[kb_id] = summary
            kb_counts[kb_id] = len(results)
            for result in results:
                if result is not None and result.id not in seen_chunks:
                    seen_chunks[result.id] = result

        all_results = sorted(seen_chunks.values(), key=lambda r: r.score, reverse=True)

        if not all_results:
            return ToolResult(
                success=True,
                output="No relevant graph information found.",
                data={
                    "knowledge_base_ids": cast("list[JsonValue]", kb_ids),
                    "query": query,
                    "results": cast("list[JsonValue]", []),
                    "graph_configs": cast(
                        "dict[str, JsonValue]", graph_configs_to_data(graph_configs)
                    ),
                    "graph_config": cast("JsonValue", aggregate_graph_config(graph_configs)),
                    "errors": cast("list[JsonValue]", errors),
                },
            )

        output_parts, formatted_results = self._format_output(
            query, kb_ids, all_results, graph_configs, kb_counts, errors
        )
        has_graph_config = bool(graph_configs)
        return ToolResult(
            success=True,
            output=output_parts,
            data={
                "knowledge_base_ids": cast("list[JsonValue]", kb_ids),
                "query": query,
                "results": cast("list[JsonValue]", formatted_results),
                "count": len(all_results),
                "kb_counts": cast("dict[str, JsonValue]", kb_counts),
                "graph_configs": cast("dict[str, JsonValue]", graph_configs_to_data(graph_configs)),
                "graph_config": cast("JsonValue", aggregate_graph_config(graph_configs)),
                "graph_data": build_graph_visualization_data(all_results),
                "has_graph_config": has_graph_config,
                "errors": cast("list[JsonValue]", errors),
                "display_type": "graph_query_results",
            },
        )

    async def _load_kb(self, kb_id: str) -> tuple[KnowledgeBaseInfo | None, str]:
        """Load one KB record, returning ``(kb, "")`` or ``(None, error)``."""
        try:
            kb = await self._kb_loader.load(knowledge_base_id=kb_id)
        except Exception as exc:
            return None, f"failed to get knowledge base: {exc}"
        if kb is None:
            return None, "failed to get knowledge base"
        return kb, ""

    async def _search_kb(
        self,
        ctx: Context,
        kb_id: str,
        query: str,
        errors: list[str],
    ) -> list[SearchResult] | None:
        """Run one hybrid search, returning ``None`` on failure (error logged)."""
        try:
            results = await self._search_runner.search(
                ctx,
                SearchCall(
                    query_text=query,
                    kb_id=kb_id,
                    knowledge_base_ids=(kb_id,),
                    top_k=GRAPH_SEARCH_MATCH_COUNT,
                ),
            )
        except Exception as exc:
            errors.append(f"KB {kb_id}: query failed: {exc}")
            return None
        return results or []

    def _format_output(
        self,
        query: str,
        kb_ids: list[str],
        all_results: list[SearchResult],
        graph_configs: dict[str, GraphConfigSummary],
        kb_counts: dict[str, int],
        errors: list[str],
    ) -> tuple[str, list[JsonObject]]:
        """Render the markdown output and the structured result list."""
        parts: list[str] = ["=== Knowledge Graph Query ===\n\n"]
        parts.append(f"📊 Query: {query}\n")
        parts.append(f"🎯 Target Knowledge Bases: {kb_ids}\n")
        parts.append(f"✓ Found {len(all_results)} relevant results (deduplicated)\n\n")

        if errors:
            parts.append("=== ⚠️ Partial Failures ===\n")
            for err in errors:
                parts.append(f"  - {err}\n")
            parts.append("\n")

        has_graph_config = False
        parts.append("=== 📈 Graph Configuration Status ===\n\n")
        for kb_id in sorted(graph_configs):
            config = graph_configs[kb_id]
            has_graph_config = True
            parts.append(f"Knowledge Base [{kb_id}]:\n")
            if config.nodes:
                parts.append(f"  ✓ Entity Types ({len(config.nodes)}): {config.nodes}\n")
            else:
                parts.append("  ⚠️ No entity types configured\n")
            if config.relations:
                parts.append(
                    f"  ✓ Relationship Types ({len(config.relations)}): {config.relations}\n"
                )
            else:
                parts.append("  ⚠️ No relationship types configured\n")
            parts.append("\n")

        if not has_graph_config:
            parts.append("⚠️ None of the queried knowledge bases have graph extraction configured\n")
            parts.append(
                "💡 Hint: Configure entity and relationship types in knowledge base settings\n\n"
            )

        if kb_counts:
            parts.append("=== 📚 Knowledge Base Coverage ===\n")
            for kb_id, count in kb_counts.items():
                parts.append(f"  - {kb_id}: {count} results\n")
            parts.append("\n")

        parts.append("=== 🔍 Query Results ===\n\n")
        if not has_graph_config:
            parts.append(
                "💡 Returning relevant document chunks (knowledge base has no graph configuration)\n\n"
            )
        else:
            parts.append("💡 Content retrieval based on graph configuration\n\n")

        formatted_results: list[JsonObject] = []
        current_kb = ""
        for i, result in enumerate(all_results):
            if result.knowledge_id != current_kb:
                current_kb = result.knowledge_id
                if i > 0:
                    parts.append("\n")
                parts.append(f"[Source Document: {result.knowledge_title}]\n\n")

            relevance_level = get_relevance_level(result.score)
            parts.append(f"Result #{i + 1}:\n")
            parts.append(f"  📍 Relevance: {result.score:.2f} ({relevance_level})\n")
            parts.append(f"  🔗 Match Type: {format_match_type(result.match_type)}\n")
            parts.append(f"  📄 Content: {result.content}\n")
            parts.append(f"  🆔 chunk_id: {result.id}\n\n")

            formatted_results.append(
                {
                    "result_index": i + 1,
                    "chunk_id": result.id,
                    "chunk_index": result.chunk_index,
                    "chunk_type": result.chunk_type,
                    "content": result.content,
                    "score": result.score,
                    "relevance_level": relevance_level,
                    "knowledge_id": result.knowledge_id,
                    "knowledge_base_id": result.knowledge_base_id,
                    "knowledge_title": result.knowledge_title,
                    "match_type": format_match_type(result.match_type),
                }
            )

        parts.append("=== 💡 Tips ===\n")
        parts.append(
            "- ✓ Results are deduplicated across knowledge bases and sorted by relevance\n"
        )
        parts.append("- ✓ Use get_chunk_detail to get full content\n")
        parts.append("- ✓ Use list_knowledge_chunks to explore context\n")
        if not has_graph_config:
            parts.append(
                "- ⚠️ Configure graph extraction for more precise entity-relationship results\n"
            )
        parts.append("- ⏳ Full graph query language (Cypher) support is under development\n")
        return "".join(parts), formatted_results


def _parse_args(args: str) -> JsonObject:
    try:
        raw = json.loads(args)
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _as_str(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


def _as_str_list(value: JsonValue) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item != ""]
    return []


__all__ = [
    "GRAPH_MAX_KB_IDS",
    "GRAPH_SEARCH_MATCH_COUNT",
    "WIKI_GRAPH_TOOL_DESCRIPTION",
    "WIKI_GRAPH_TOOL_NAME",
    "WIKI_GRAPH_TOOL_SCHEMA",
    "GraphConfigSummary",
    "QueryKnowledgeGraphTool",
    "aggregate_graph_config",
    "build_graph_visualization_data",
    "build_wiki_graph_definition",
    "graph_configs_to_data",
    "summarize_graph_config",
]
