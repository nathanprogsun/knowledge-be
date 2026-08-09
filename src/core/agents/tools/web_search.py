"""Web-search agent tool.

Runs a web search through an injected dispatch service (the merged
``WebSearchSearchService``, which routes through the
``WebSearchClientRegistry`` to build per-provider HTTP clients). The
tool is constructed with a session id, the agent-level result cap, and
the provider entity id resolved from the agent configuration or the
tenant default; each invocation passes the workspace id explicitly so
the tool can be driven from any context.

Behaviour mirrors the upstream web-search tool:

- the ``query`` argument is required;
- a missing workspace id fails fast;
- results are rendered both as a human-readable block (for the LLM) and
  as structured items (``evidence_type`` ``search_summary``,
  ``page_verified`` ``false``);
- search failures surface as a failed ``ToolResult`` instead of raising.

RAG compression of search results (and the session-scoped temporary
knowledge-base caching behind it) is deferred: it depends on temporary
knowledge-base state that is not yet available in this layer, so raw
search results are returned unchanged.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from src.common.exception import ApplicationError
from src.common.json import JsonObject, JsonValue
from src.core.agents.tools.types import ToolResult
from src.core.infra.web_search.search_service import SearchResult

# Default maximum number of search results (matches the shared config default).
DEFAULT_WEB_SEARCH_MAX_RESULTS: int = 10

# Truncation limit for a single result's content in the rendered output.
_MAX_CONTENT_CHARS: int = 500

_TOOL_NAME = "web_search"

_DESCRIPTION_TEMPLATE = """Search the web for current information and news. This tool searches the internet to find up-to-date information that may not be in the knowledge base.

## CRITICAL - KB First Rule
**ABSOLUTE RULE**: You MUST complete KB retrieval (grep_chunks AND knowledge_search) FIRST before using this tool.
- NEVER use web_search without first trying grep_chunks and knowledge_search
- ONLY use web_search if BOTH grep_chunks AND knowledge_search return insufficient/no results
- KB retrieval is MANDATORY - you CANNOT skip it

## Features
- Real-time web search: Search the internet for current information
- RAG compression: Automatically compresses and extracts relevant content from search results
- Session-scoped caching: Maintains temporary knowledge base for session to avoid re-indexing

## Usage

**Use when**:
- **ONLY after** completing grep_chunks AND knowledge_search
- KB retrieval returned insufficient or no results
- Need current or real-time information (news, events, recent updates)
- Information is not available in knowledge bases
- Need to verify or supplement information from knowledge bases
- Searching for recent developments or trends

**Parameters**:
- query (required): Search query string

**Returns**: Web search results with title, short wN page ID, snippet, and content (up to {max_results} results)

## Examples

```
# Search for current information
{{
  "query": "latest developments in AI"
}}

# Search for recent news
{{
  "query": "Python 3.12 release notes"
}}
```

## Evidence and Fallback

- Results are automatically compressed using RAG to extract relevant content
- Search results are stored in a temporary knowledge base for the session
- Titles, URLs, snippets, and content snippets are usable search-summary evidence
- Use web_fetch only when the snippet is insufficient or full-page verification is important
- If web_fetch fails, keep the search evidence, disclose that page content was not verified, and lower confidence for dynamic facts
- Do not repeat equivalent searches merely because a page could not be fetched
- Maximum {max_results} results will be returned per search"""

# LLM-facing JSON Schema for the tool's ``query`` parameter.
_WEB_SEARCH_SCHEMA: JsonValue = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query string"},
    },
    "required": ["query"],
}


class WebSearchService(Protocol):
    """Dispatch seam the tool depends on (satisfied by ``WebSearchSearchService``)."""

    async def search(
        self,
        *,
        tenant_id: int,
        provider_id: str,
        query: str,
        max_results: int,
    ) -> list[SearchResult]:
        """Run a web search for a workspace via its configured provider."""
        ...


class WebSearchTool:
    """Agent tool that searches the web through the injected search service."""

    name = _TOOL_NAME

    def __init__(
        self,
        *,
        search_service: WebSearchService,
        session_id: str = "",
        max_results: int = DEFAULT_WEB_SEARCH_MAX_RESULTS,
        provider_id: str = "",
    ) -> None:
        self._search_service = search_service
        self._session_id = session_id
        self._max_results = max(1, max_results)
        self._provider_id = provider_id
        self.description = _DESCRIPTION_TEMPLATE.format(max_results=self._max_results)
        self.parameters_schema: JsonValue = _WEB_SEARCH_SCHEMA

    async def execute(self, args: JsonObject, *, tenant_id: int) -> ToolResult:
        """Run a search for ``args["query"]`` and render a structured result."""
        query = _extract_query(args)
        if query == "":
            return ToolResult(success=False, error="query parameter is required")
        if not tenant_id or tenant_id <= 0:
            return ToolResult(success=False, error="workspace ID not found in context")
        try:
            results = await self._search_service.search(
                tenant_id=tenant_id,
                provider_id=self._provider_id,
                query=query,
                max_results=self._max_results,
            )
        except ApplicationError as exc:
            return ToolResult(success=False, error=f"web search failed: {exc.message}")
        return _build_tool_result(query=query, results=results)


# ── Helpers ──────────────────────────────────────────────────────────


def _extract_query(args: JsonObject) -> str:
    raw = args.get("query")
    if isinstance(raw, str):
        return raw
    return ""


def _format_rfc3339(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    normalized = value.astimezone(UTC).isoformat(timespec="seconds")
    return normalized.replace("+00:00", "Z")


def _build_tool_result(*, query: str, results: list[SearchResult]) -> ToolResult:
    if not results:
        return ToolResult(
            success=True,
            output=f"No web search results found for query: {query}",
            data={
                "query": query,
                "results": [],
                "count": 0,
            },
        )

    lines: list[str] = [
        "=== Web Search Results ===",
        f"Query: {query}",
        f"Found {len(results)} result(s)",
        "",
    ]
    formatted: list[JsonValue] = []
    for index, result in enumerate(results, start=1):
        lines.append(f"Result #{index}:")
        lines.append(f"  Title: {result.title}")
        lines.append(f"  URL: {result.url}")
        if result.snippet:
            lines.append(f"  Snippet: {result.snippet}")
        if result.content:
            lines.append(f"  Content: {_truncate_content(result.content)}")
        if result.published_at is not None:
            lines.append(f"  Published: {_format_rfc3339(result.published_at)}")
        lines.append("")

        item: JsonObject = {
            "result_index": index,
            "title": result.title,
            "url": result.url,
            "snippet": result.snippet,
            "content": result.content,
            "source": result.source,
            "evidence_type": "search_summary",
            "page_verified": False,
        }
        if result.published_at is not None:
            item["published_at"] = _format_rfc3339(result.published_at)
        formatted.append(item)

    lines.extend(
        [
            "=== Next Steps ===",
            "- Titles, URLs, snippets, and content snippets are usable search-summary evidence.",
            "- If the evidence is sufficient, answer now. Use web_fetch only for claims that need full-page verification.",
            "- If fetching fails, retain these results, disclose that page content was not verified, and avoid presenting dynamic facts as certain.",
        ]
    )

    return ToolResult(
        success=True,
        output="\n".join(lines),
        data={
            "query": query,
            "results": formatted,
            "count": len(results),
            "display_type": "web_search_results",
        },
    )


def _truncate_content(content: str) -> str:
    if len(content) <= _MAX_CONTENT_CHARS:
        return content
    return content[:_MAX_CONTENT_CHARS] + "..."


__all__ = [
    "DEFAULT_WEB_SEARCH_MAX_RESULTS",
    "WebSearchService",
    "WebSearchTool",
]
