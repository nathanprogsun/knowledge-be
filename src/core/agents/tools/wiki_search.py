"""Wiki-search agent tool.

``wiki_search`` runs a case-insensitive POSIX regex search over the wiki
pages of every knowledge base in scope and renders the matching pages as
XML. The query is a PostgreSQL ``~*`` expression — the tool steers the
model toward regex alternation / chained terms so a single call can
match several concepts at once.

Execution is faithful to the upstream search tool:

- ``query`` (a single string) and ``queries`` (a list) are both accepted;
- a missing ``queries`` fails fast;
- an optional ``knowledge_base_id`` narrows the scopes and is rejected
  when it lies outside the server-owned scope;
- every hit is checked against the scope's document/tag whitelist before
  it is surfaced (the whitelist is server-enforced and never exposed as a
  tool argument);
- slugs already returned earlier in the session render without their
  summary so the model does not re-read identical content.

The tool executes against injected seams — a wiki page service and an
optional tag fetcher — so no test touches storage.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import cast

from src.ai.embedding.base import Context
from src.common.json import JsonObject, JsonValue
from src.core.agents.tools.base import ToolDefinition, ToolResult
from src.core.agents.tools.scope_auth import KnowledgeTagsFetcher
from src.core.agents.tools.text_utils import xml_escape
from src.core.agents.tools.wiki_route import (
    WikiPageServiceProtocol,
    WikiRouteResolver,
    WikiScope,
    page_passes_wiki_scope,
    register_linked_slugs,
    seen_link_key,
)
from src.db.models.wiki_page import WikiPage

WIKI_SEARCH_TOOL_NAME = "wiki_search"

WIKI_SEARCH_TOOL_DESCRIPTION = (
    "Search wiki pages using PostgreSQL POSIX regular expressions (~* operator, case-insensitive).\n"
    "STRONGLY PREFER using regex to search for multiple concepts at once rather than simple plain text queries.\n"
    "Returns matching pages with titles, slugs, and summaries (each tagged with its short bN knowledge_base_id).\n"
    "Examples:\n"
    '- Alternation (RECOMMENDED): "stardust|skyvault" (matches either word)\n'
    '- Multiple terms (RECOMMENDED): "psionic.*engine" (matches both words in order)\n'
    '- Prefix matching: "^entity/.*" (finds all entities)\n'
    '- Plain text: "engine" (matches anywhere in title/content/slug/summary)\n'
    "IMPORTANT — JSON escaping: every backslash in a regex MUST be written as \\\\ inside the JSON tool "
    'arguments (e.g. to search for literal "C++" write "C\\\\+\\\\+", NOT "C\\+\\+"; for "\\d+" write "\\\\d+"). '
    'Plain "\\+" / "\\d" etc. are invalid JSON escapes and will fail to parse.\n'
    "Use this to find relevant wiki pages when you don't know the exact slug."
)

WIKI_SEARCH_TOOL_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of regex search queries to run",
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return per query (default 10)",
            },
            "knowledge_base_id": {
                "type": "string",
                "description": "Optional: restrict search to a single short bN knowledge base ID in scope.",
            },
        },
        "required": ["queries"],
    },
    ensure_ascii=False,
)


def build_wiki_search_definition() -> ToolDefinition:
    """Return the default tool definition for the wiki-search tool."""
    return ToolDefinition(
        name=WIKI_SEARCH_TOOL_NAME,
        description=WIKI_SEARCH_TOOL_DESCRIPTION,
        parameters=WIKI_SEARCH_TOOL_SCHEMA,
    )


@dataclass(frozen=True, slots=True)
class WikiSearchInput:
    """Parsed input for the wiki-search tool."""

    queries: tuple[str, ...] = ()
    query: tuple[str, ...] = ()
    limit: int = 0
    knowledge_base_id: str = ""


def _as_str_list(value: JsonValue) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(item for item in value if isinstance(item, str) and item != "")
    if isinstance(value, str):
        return (value,) if value != "" else ()
    return ()


def _as_str(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


def _as_int(value: JsonValue) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _parse_input(args: str) -> WikiSearchInput:
    try:
        raw = json.loads(args)
    except json.JSONDecodeError:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    return WikiSearchInput(
        queries=_as_str_list(raw.get("queries")),
        query=_as_str_list(raw.get("query")),
        limit=_as_int(raw.get("limit")),
        knowledge_base_id=_as_str(raw.get("knowledge_base_id")),
    )


#: Context window (code points) before / after a regex hit in a snippet.
_SNIPPET_CONTEXT_RUNES = 60
#: Max length of the matched text inside a snippet.
_SNIPPET_MAX_MATCH_RUNES = 100


def extract_snippet(content: str, query: str) -> str:
    """Build a short contextual snippet around the first regex match."""
    if not content or not query:
        return ""
    try:
        pattern = re.compile(query, re.IGNORECASE)
    except re.error:
        return ""
    match = pattern.search(content)
    if match is None:
        return ""
    start, end = match.span()
    match_str = content[start:end]
    before = content[:start]
    after = content[end:]

    before_runes = before[-_SNIPPET_CONTEXT_RUNES:]
    after_runes = after[:_SNIPPET_CONTEXT_RUNES]
    match_runes = match_str[:_SNIPPET_MAX_MATCH_RUNES]
    if len(match_str) > _SNIPPET_MAX_MATCH_RUNES:
        match_runes = match_runes + "..."

    snippet = before_runes + match_runes + after_runes
    snippet = snippet.replace("\n", " ")
    while "  " in snippet:
        snippet = snippet.replace("  ", " ")
    return "... " + snippet.strip() + " ..."


class WikiSearchTool:
    """Searches wiki pages by POSIX regex across the scopes in scope."""

    def __init__(
        self,
        *,
        definition: ToolDefinition,
        wiki_service: WikiPageServiceProtocol,
        scopes: list[WikiScope],
        routes: WikiRouteResolver | None = None,
        tag_fetcher: KnowledgeTagsFetcher | None = None,
    ) -> None:
        self._definition = definition
        self._wiki_service = wiki_service
        self._scopes = list(scopes)
        self._routes = routes if routes is not None else WikiRouteResolver()
        self._tag_fetcher = tag_fetcher
        self._seen_slugs: set[str] = set()

    def name(self) -> str:
        return self._definition.name

    def description(self) -> str:
        return self._definition.description

    def parameters(self) -> str:
        return self._definition.parameters

    async def execute(self, ctx: Context, args: str) -> ToolResult:
        """Run every query across the effective scopes and render the hits."""
        input_ = _parse_input(args)

        queries_to_run = list(input_.queries)
        queries_to_run.extend(input_.query)
        if not queries_to_run:
            return ToolResult(success=False, error="Missing 'queries' parameter")

        limit = input_.limit if input_.limit > 0 else 10

        effective_scopes = self._scopes
        if input_.knowledge_base_id:
            filtered = [
                scope
                for scope in self._scopes
                if scope.knowledge_base_id == input_.knowledge_base_id
            ]
            if not filtered:
                return ToolResult(
                    success=False,
                    error="knowledge_base_id is not within the current wiki scope",
                )
            effective_scopes = filtered

        all_outputs: list[str] = []
        search_errors: list[str] = []
        successful_search_calls = 0
        found_kbs: dict[str, list[str]] = {}

        for query in queries_to_run:
            all_hits: list[tuple[WikiPage, str]] = []
            for scope in effective_scopes:
                kb_id = scope.knowledge_base_id
                if not kb_id:
                    continue
                try:
                    pages = await self._wiki_service.search_pages(
                        knowledge_base_id=kb_id,
                        query=query,
                        limit=limit,
                    )
                except Exception as exc:
                    search_errors.append(f"Wiki search {query!r} failed in KB {kb_id}: {exc}")
                    continue
                successful_search_calls += 1
                for page in pages:
                    if page is None:
                        continue
                    try:
                        passes_scope = await page_passes_wiki_scope(
                            ctx, page, scope, self._tag_fetcher
                        )
                    except Exception as exc:
                        search_errors.append(
                            f"Failed to validate Wiki search result {page.slug!r} in KB {kb_id}: {exc}"
                        )
                        continue
                    if not passes_scope:
                        continue
                    if page.knowledge_base_id and page.knowledge_base_id != kb_id:
                        search_errors.append(
                            f"Wiki search result {page.slug!r} returned KB "
                            f"{page.knowledge_base_id} while resolving allowed KB {kb_id}"
                        )
                        continue
                    all_hits.append((page, kb_id))
                    self._routes.remember_page(page, kb_id)
                    found_kbs.setdefault(page.slug, []).append(kb_id)
                    register_linked_slugs(found_kbs, page, kb_id)

            if not all_hits:
                all_outputs.append(f'<search_results count="0" query="{xml_escape(query)}" />')
                continue

            all_outputs.append(_render_search_results(query, all_hits, self._seen_slugs))

        if successful_search_calls == 0 and search_errors:
            return ToolResult(success=False, error="; ".join(search_errors))

        output = "\n\n".join(all_outputs)
        if search_errors:
            output += "\n\n<errors>\n" + "\n".join(search_errors) + "\n</errors>"
        data: JsonObject = {
            "found_kbs": cast("dict[str, JsonValue]", found_kbs),
        }
        return ToolResult(success=True, output=output, data=data)


def _render_search_results(
    query: str,
    hits: list[tuple[WikiPage, str]],
    seen_slugs: set[str],
) -> str:
    """Render one query's hits as a ``<search_results>`` block."""
    parts: list[str] = [f'<search_results count="{len(hits)}" query="{xml_escape(query)}">\n']
    for page, kb_id in hits:
        key = seen_link_key(kb_id, page.slug)
        seen = key in seen_slugs
        seen_slugs.add(key)

        snippet = extract_snippet(page.content, query)
        snippet_tag = f"\n<match_snippet>{xml_escape(snippet)}</match_snippet>" if snippet else ""
        aliases_tag = (
            f"\n<aliases>{xml_escape(', '.join(page.aliases))}</aliases>" if page.aliases else ""
        )

        summary = page.summary
        if seen:
            summary = "(summary omitted, already seen in previous search)"

        parts.append(
            "<page>\n"
            f"<knowledge_base_id>{xml_escape(kb_id)}</knowledge_base_id>\n"
            f"<link>[[{xml_escape(page.slug)}|{xml_escape(page.title)}]]</link>\n"
            f"<type>{xml_escape(page.page_type)}</type>{aliases_tag}\n"
            f"<summary>{xml_escape(summary)}</summary>{snippet_tag}\n"
            "</page>\n"
        )
    parts.append("</search_results>")
    return "".join(parts)


__all__ = [
    "WIKI_SEARCH_TOOL_DESCRIPTION",
    "WIKI_SEARCH_TOOL_NAME",
    "WIKI_SEARCH_TOOL_SCHEMA",
    "WikiSearchInput",
    "WikiSearchTool",
    "build_wiki_search_definition",
    "extract_snippet",
]
