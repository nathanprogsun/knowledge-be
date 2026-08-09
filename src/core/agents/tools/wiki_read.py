"""Wiki read tools: ``wiki_read_page`` and ``wiki_read_source_doc``.

``wiki_read_page`` resolves one or more wiki slugs against every knowledge
base in scope, applies the server-enforced document/tag whitelist, and
renders each resolved page as a well-formed ``<wiki_page>`` block while
sharing the output budget fairly across the batch. ``wiki_read_source_doc``
drills into one source document — either a bounded chunk range or a regex
query with surrounding context — to recover details omitted from the wiki.

Both tools execute against injected seams (a wiki page service, an optional
tag fetcher, and a paged chunk store) so the layer stays free of direct
storage access.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import cast

from src.ai.embedding.base import Context
from src.common.exception import ApplicationError, NotFoundError
from src.common.json import JsonObject, JsonValue
from src.core.agents.tools.base import ToolDefinition, ToolResult
from src.core.agents.tools.chunk_store import PagedChunkStore
from src.core.agents.tools.output_budget import output_budget, split_budget_fairly
from src.core.agents.tools.scope_auth import (
    KnowledgeLookup,
    KnowledgeTagsFetcher,
    authorize_knowledge_in_search_targets,
)
from src.core.agents.tools.search_target import SearchTargets
from src.core.agents.tools.text_utils import build_image_info_markdown, parse_image_infos
from src.core.agents.tools.wiki_route import (
    WikiPageServiceProtocol,
    WikiRouteResolver,
    WikiScope,
    page_passes_wiki_scope,
    register_linked_slugs,
    scopes_outside_kbs,
    seen_link_key,
)
from src.core.contracts.knowledge import Knowledge
from src.core.knowledge.wiki.types import WIKI_PAGE_TYPE_INDEX, WikiIndexResponse
from src.db.models.chunk import Chunk
from src.db.models.wiki_page import WikiPage

#: Per-type cap when synthesizing the index overview for the agent. The
#: overview gives the model the wiki's shape; deeper exploration goes through
#: wiki_search. Keeping the cap small bounds the content served to the LLM.
WIKI_INDEX_AGENT_TOP_K = 20

#: Neighbour summaries inlined per link section, and the max length of each.
WIKI_MAX_LINK_SUMMARIES = 20
WIKI_LINK_SUMMARY_MAX_RUNES = 150

#: Smallest body slice worth rendering. When the budget cannot give every
#: resolved page at least this much, trailing pages are named in
#: ``<omitted_pages>`` instead of being cut mid-tag.
WIKI_MIN_PAGE_BODY = 400

#: Room held back for the ``<errors>`` / ``<omitted_pages>`` trailers.
WIKI_BUDGET_RESERVE = 600

WIKI_READ_PAGE_TOOL_NAME = "wiki_read_page"

WIKI_READ_PAGE_TOOL_DESCRIPTION = (
    "Read one or more wiki pages by their slugs. Returns the full markdown content, metadata, and links.\n"
    'Use this to read specific wiki pages when you know their slug (e.g. "entity/acme-corp", "concept/rag").\n'
    "Knowledge-base routing is automatic. Known link/search provenance is preferred; otherwise every wiki "
    "knowledge base in scope is checked and ambiguous matches are returned."
)

WIKI_READ_PAGE_TOOL_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "slugs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of wiki page slugs to read (e.g. ['entity/acme-corp', 'index'])",
            },
        },
        "required": ["slugs"],
    },
    ensure_ascii=False,
)

WIKI_READ_SOURCE_DOC_TOOL_NAME = "wiki_read_source_doc"

WIKI_READ_SOURCE_DOC_TOOL_DESCRIPTION = (
    "Read or search within a specific source document to drill down for details omitted from the wiki.\n"
    "Provide the knowledge_id from the <sources> block.\n"
    "You can EITHER search using a regex query OR fetch a specific contiguous range of chunks using "
    "start_chunk_index and end_chunk_index (useful for expanding context around a known chunk).\n"
    "If neither query nor range is provided, it returns the beginning of the document."
)

WIKI_READ_SOURCE_DOC_TOOL_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "knowledge_id": {
                "type": "string",
                "description": "The short dN source document ID from the <sources> block",
            },
            "query": {
                "type": "string",
                "description": (
                    "Optional: A regex query to filter the document chunks. Use this to find specific "
                    'quotes or details efficiently. Remember to double-escape backslashes for JSON: '
                    'write "C\\\\\\\\+\\\\\\\\+" (NOT "C\\\\+\\\\+") and "\\\\\\\\d+" (NOT "\\\\d+").'
                ),
            },
            "start_chunk_index": {
                "type": "integer",
                "description": "Optional: The starting chunk index (1-based) to read a specific range.",
            },
            "end_chunk_index": {
                "type": "integer",
                "description": "Optional: The ending chunk index (1-based) to read a specific range. Must be >= start_chunk_index.",
            },
        },
        "required": ["knowledge_id"],
    },
    ensure_ascii=False,
)


def build_wiki_read_page_definition() -> ToolDefinition:
    """Return the default tool definition for the wiki-read-page tool."""
    return ToolDefinition(
        name=WIKI_READ_PAGE_TOOL_NAME,
        description=WIKI_READ_PAGE_TOOL_DESCRIPTION,
        parameters=WIKI_READ_PAGE_TOOL_SCHEMA,
    )


def build_wiki_read_source_doc_definition() -> ToolDefinition:
    """Return the default tool definition for the source-doc reader."""
    return ToolDefinition(
        name=WIKI_READ_SOURCE_DOC_TOOL_NAME,
        description=WIKI_READ_SOURCE_DOC_TOOL_DESCRIPTION,
        parameters=WIKI_READ_SOURCE_DOC_TOOL_SCHEMA,
    )


# ── wiki_read_page ────────────────────────────────────────────────────


def render_index_overview_for_agent(resp: WikiIndexResponse) -> str:
    """Format a structured index as a compact markdown block for the agent.

    Kept almost identical to the legacy "intro + ## Type (N)" directory so
    existing prompts that reason about index layouts need no retraining.
    """
    type_labels = {
        "summary": "Summary",
        "entity": "Entity",
        "concept": "Concept",
        "synthesis": "Synthesis",
        "comparison": "Comparison",
    }

    parts: list[str] = []
    intro = resp.intro.strip()
    heading = intro.find("\n## ")
    if heading >= 0:
        intro = intro[:heading].strip()
    if intro:
        parts.append(intro)
        parts.append("")

    non_empty = 0
    for group in resp.groups:
        if group.total == 0:
            continue
        label = type_labels.get(group.type, group.type)
        if len(group.items) < group.total:
            parts.append(f"## {label} ({group.total} total, showing top {len(group.items)})\n")
        else:
            parts.append(f"## {label} ({group.total})\n")
        for item in group.items:
            display = item.title or item.slug
            if item.summary:
                parts.append(f"[[{item.slug}|{display}]] — {item.summary}")
            else:
                parts.append(f"[[{item.slug}|{display}]]")
        non_empty += 1

    if non_empty == 0:
        parts.append("\n*No wiki pages yet. Upload documents to get started.*\n")
    else:
        parts.append(
            "\n_To explore more pages under any category, use wiki_search with a query, "
            "or read a specific slug directly._\n"
        )
    return "\n".join(parts)


@dataclass(frozen=True, slots=True)
class _PendingWikiPage:
    """A resolved page whose neighbour summaries, sources, and body are gathered."""

    page: WikiPage
    kb_id: str
    out_links: list[str]
    in_links: list[str]
    sources: list[str]
    body: str

    def render(self, body: str) -> str:
        """Render the page with the given body slice."""
        return (
            "<wiki_page>\n<metadata>\n"
            f"<knowledge_base_id>{self.kb_id}</knowledge_base_id>\n"
            f"<link>[[{self.page.slug}|{self.page.title}]]</link>\n"
            f"<type>{self.page.page_type}</type>\n"
            f"<aliases>{', '.join(self.page.aliases)}</aliases>\n"
            "</metadata>\n<relationships>\n"
            f"<links_to>{', '.join(self.out_links)}</links_to>\n"
            f"<linked_from>{', '.join(self.in_links)}</linked_from>\n"
            "</relationships>\n<sources>\n"
            f"{chr(10).join(self.sources)}\n</sources>\n<summary>\n"
            f"{self.page.summary}\n</summary>\n<content>\n"
            f"{body}\n</content>\n</wiki_page>"
        )


def render_wiki_pages_within_budget(
    pages: list[_PendingWikiPage],
    budget: int,
) -> tuple[str, list[str], list[str]]:
    """Render a batch of pages so the result fits the output ceiling.

    Bodies are trimmed by fair-sharing the budget; only when even a minimal
    body no longer fits are trailing pages dropped — by name, so the model
    knows to re-read them. Returns ``(output, truncated_slugs, omitted_slugs)``.
    """
    if not pages:
        return "", [], []

    separator = "\n\n"
    separator_cost = len(separator)

    rendered = [page.render(page.body) for page in pages]
    body_sizes = [len(page.body) for page in pages]
    overheads = [len(rendered[i]) - body_sizes[i] for i in range(len(pages))]
    total = separator_cost * (len(pages) - 1) + sum(len(item) for item in rendered)

    usable = budget - WIKI_BUDGET_RESERVE
    if usable <= 0 or total <= usable:
        return separator.join(rendered), [], []

    def fixed_cost(keep: int) -> int:
        cost = separator_cost * (keep - 1)
        for i in range(keep):
            cost += overheads[i]
        return cost

    keep = len(pages)
    while keep > 1 and fixed_cost(keep) + keep * WIKI_MIN_PAGE_BODY > usable:
        keep -= 1
    omitted = [page.page.slug for page in pages[keep:]]

    caps = split_budget_fairly(usable - fixed_cost(keep), body_sizes[:keep])
    outputs: list[str] = []
    truncated: list[str] = []
    for i in range(keep):
        if caps[i] >= body_sizes[i]:
            outputs.append(rendered[i])
            continue
        body = "(body omitted: output budget exhausted)"
        if caps[i] > 0:
            body = _truncate_preview(pages[i].body, caps[i])
        outputs.append(pages[i].render(body))
        truncated.append(pages[i].page.slug)
    return separator.join(outputs), truncated, omitted


def _truncate_preview(text: str, max_runes: int) -> str:
    """Truncate ``text`` to ``max_runes`` code points, appending ``...``."""
    if len(text) <= max_runes:
        return text
    return text[:max_runes] + "..."


class WikiReadPageTool:
    """Reads one or more wiki pages by slug across the scopes in scope."""

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
        self._seen_links: set[str] = set()

    def name(self) -> str:
        return self._definition.name

    def description(self) -> str:
        return self._definition.description

    def parameters(self) -> str:
        return self._definition.parameters

    async def execute(self, ctx: Context, args: str) -> ToolResult:
        """Resolve every slug, apply the scope, and render the batch."""
        raw = _parse_args(args)
        slugs_to_fetch: list[str] = []
        slugs_to_fetch.extend(_as_str_list(raw.get("slugs")))
        slugs_to_fetch.extend(_as_str_list(raw.get("slug")))
        slugs_to_fetch = _dedup_non_empty(slugs_to_fetch)
        if not slugs_to_fetch:
            return ToolResult(success=False, error="Missing 'slugs' parameter")

        pending: list[_PendingWikiPage] = []
        errs: list[str] = []
        found_kbs: dict[str, list[str]] = {}
        filtered_out: dict[str, list[str]] = {}
        lookup_failed: set[str] = set()

        for slug in slugs_to_fetch:
            cached_scopes = self._routes.scopes_for_slug(slug, self._scopes)
            effective_scopes = [*cached_scopes, *scopes_outside_kbs(self._scopes, cached_scopes)]
            hits: list[tuple[WikiPage, str]] = []
            for scope in effective_scopes:
                kb_id = scope.knowledge_base_id
                if not kb_id:
                    continue
                try:
                    page = await self._wiki_service.get_page_by_slug(
                        knowledge_base_id=kb_id, slug=slug
                    )
                except NotFoundError:
                    continue
                except Exception as exc:
                    lookup_failed.add(slug)
                    errs.append(f"Failed to read Wiki page '{slug}' in KB {kb_id}: {exc}")
                    continue
                if page is None:
                    continue
                if page.knowledge_base_id and page.knowledge_base_id != kb_id:
                    lookup_failed.add(slug)
                    errs.append(
                        f"Wiki page '{slug}' returned KB {page.knowledge_base_id} "
                        f"while resolving allowed KB {kb_id}"
                    )
                    continue
                try:
                    passes_scope = await page_passes_wiki_scope(ctx, page, scope, self._tag_fetcher)
                except Exception as exc:
                    errs.append(
                        f"Failed to validate wiki scope for '{slug}' in KB {kb_id}: {exc}"
                    )
                    continue
                if not passes_scope:
                    filtered_out.setdefault(slug, [])
                    if kb_id not in filtered_out[slug]:
                        filtered_out[slug].append(kb_id)
                    continue
                hits.append((page, kb_id))
                self._routes.remember_page(page, kb_id)
                found_kbs.setdefault(slug, [])
                if kb_id not in found_kbs[slug]:
                    found_kbs[slug].append(kb_id)
                register_linked_slugs(found_kbs, page, kb_id)
                self._seen_links.add(seen_link_key(kb_id, slug))

            if not hits:
                if slug in filtered_out:
                    errs.append(
                        f"Wiki page '{slug}' exists in {filtered_out[slug]} but none of its "
                        "source documents are within the scope pinned by the user"
                    )
                elif slug not in lookup_failed:
                    errs.append(f"Wiki page '{slug}' not found")
                continue

            for page, kb_id in hits:
                pending.append(await self._resolve_page(ctx, page, kb_id))

        if not pending:
            return ToolResult(success=False, error="; ".join(errs))

        final_output, truncated_slugs, omitted_slugs = render_wiki_pages_within_budget(
            pending, output_budget()
        )
        if omitted_slugs:
            final_output += (
                "\n\n<omitted_pages reason=\"output budget exceeded\">\n"
                + "\n".join(omitted_slugs)
                + "\n</omitted_pages>"
                + "\n<hint>These pages were resolved but not rendered. "
                + "Call wiki_read_page again with fewer slugs to read them.</hint>"
            )
        if errs:
            final_output += "\n\n<errors>\n" + "\n".join(errs) + "\n</errors>"

        ambiguous = {slug: kbs for slug, kbs in found_kbs.items() if len(kbs) > 1}
        return ToolResult(
            success=True,
            output=final_output,
            data={
                "found_kbs": cast("dict[str, JsonValue]", found_kbs),
                "ambiguous_slugs": cast("dict[str, JsonValue]", ambiguous),
                "truncated_slugs": cast("list[JsonValue]", truncated_slugs),
                "omitted_slugs": cast("list[JsonValue]", omitted_slugs),
            },
        )

    async def _resolve_page(self, ctx: Context, page: WikiPage, kb_id: str) -> _PendingWikiPage:
        """Gather everything needed to render ``page`` except its final size."""
        out_links = await self._format_links(ctx, page.out_links, kb_id)
        in_links = await self._format_links(ctx, page.in_links, kb_id)

        sources: list[str] = []
        for ref in page.source_refs:
            kid = ref
            title = ""
            pipe_idx = ref.find("|")
            if pipe_idx > 0:
                kid = ref[:pipe_idx]
                title = ref[pipe_idx + 1 :]
            if title:
                sources.append(f'<source knowledge_id="{kid}">{title}</source>')
            else:
                sources.append(f'<source knowledge_id="{kid}"/>')

        body = page.content
        if page.page_type == WIKI_PAGE_TYPE_INDEX:
            try:
                overview = await self._wiki_service.get_index_view(
                    knowledge_base_id=kb_id,
                    tenant_id=page.tenant_id,
                    limit=WIKI_INDEX_AGENT_TOP_K,
                    cursor="",
                )
            except Exception:
                overview = None
            if overview is not None:
                body = render_index_overview_for_agent(overview)
                for group in overview.groups:
                    for item in group.items:
                        self._routes.remember(item.slug, kb_id)

        return _PendingWikiPage(
            page=page,
            kb_id=kb_id,
            out_links=out_links,
            in_links=in_links,
            sources=sources,
            body=body,
        )

    async def _format_links(self, ctx: Context, slugs: list[str], kb_id: str) -> list[str]:
        """Render one link section, inlining neighbour summaries up to the cap."""
        descs: list[str] = []
        inlined = 0
        for slug in slugs:
            if not slug:
                continue
            if inlined >= WIKI_MAX_LINK_SUMMARIES:
                descs.append(f"[[{slug}]]")
                continue
            key = seen_link_key(kb_id, slug)
            if key in self._seen_links:
                descs.append(f"[[{slug}]] (summary omitted, already seen)")
                continue
            try:
                link_page = await self._wiki_service.get_page_by_slug(
                    knowledge_base_id=kb_id, slug=slug
                )
            except Exception:
                link_page = None
            if link_page is None or not link_page.summary:
                descs.append(f"[[{slug}]]")
                continue
            descs.append(
                f"[[{slug}]] ({_truncate_preview(link_page.summary, WIKI_LINK_SUMMARY_MAX_RUNES)})"
            )
            inlined += 1
            self._seen_links.add(key)
        if not descs:
            return ["(none)"]
        return descs


# ── wiki_read_source_doc ──────────────────────────────────────────────


def _enrich_chunk_content(chunk: Chunk) -> str:
    """Append image markdown for a chunk's embedded image records."""
    content = chunk.content
    if not chunk.image_info:
        return content
    images = parse_image_infos(chunk.image_info)
    if not images:
        return content
    appended = ""
    for img in images:
        url = img.get("url")
        if not isinstance(url, str) or not url:
            continue
        markdown = build_image_info_markdown(url, img)
        if markdown:
            appended += "\n" + markdown
    return content + appended


class WikiReadSourceDocTool:
    """Reads or searches within one source document's chunks."""

    def __init__(
        self,
        *,
        definition: ToolDefinition,
        chunk_store: PagedChunkStore,
        knowledge_service: KnowledgeLookup,
        search_targets: SearchTargets | None = None,
        tag_fetcher: KnowledgeTagsFetcher | None = None,
    ) -> None:
        self._definition = definition
        self._chunk_store = chunk_store
        self._knowledge_service = knowledge_service
        self._search_targets = search_targets
        self._tag_fetcher = tag_fetcher

    def name(self) -> str:
        return self._definition.name

    def description(self) -> str:
        return self._definition.description

    def parameters(self) -> str:
        return self._definition.parameters

    async def execute(self, ctx: Context, args: str) -> ToolResult:
        """Drill into the document's chunks by range or regex query."""
        raw = _parse_args(args)
        knowledge_id = _as_str(raw.get("knowledge_id")).strip()
        if not knowledge_id:
            return ToolResult(success=False, error="knowledge_id is required")

        knowledge = await self._resolve_document(ctx, knowledge_id)
        if knowledge is None:
            return ToolResult(success=False, error="Document not found: empty result")

        start_chunk_index = _as_int(raw.get("start_chunk_index"))
        end_chunk_index = _as_int(raw.get("end_chunk_index"))
        query = _as_str(raw.get("query")).strip()

        has_range = start_chunk_index > 0
        compiled: re.Pattern[str] | None = None

        meta_parts: list[str] = [f"<title>{knowledge.title or ''}</title>"]
        meta_parts.append(f"<knowledge_id>{knowledge_id}</knowledge_id>")
        if has_range:
            if end_chunk_index < start_chunk_index:
                end_chunk_index = start_chunk_index + 10
            if end_chunk_index - start_chunk_index > 50:
                end_chunk_index = start_chunk_index + 50
            meta_parts.append(f'<chunk_range start="{start_chunk_index}" end="{end_chunk_index}"/>')
        elif query:
            try:
                compiled = re.compile(query, re.IGNORECASE)
            except re.error as exc:
                return ToolResult(success=False, error=f"Invalid regex query '{query}': {exc}")
            meta_parts.append(f"<query>{query}</query>")

        match_count = 0
        page_size = 100
        page = (start_chunk_index - 1) // page_size + 1 if has_range else 1

        chunks_output: list[str] = []
        formatted_chunks: list[JsonObject] = []
        total_chunks = 0
        reached_max = False

        prev_chunk: Chunk | None = None
        force_output_next = False
        outputted_indices: set[int] = set()

        def append_formatted_chunk(chunk: Chunk, content: str) -> None:
            formatted_chunks.append(
                {
                    "chunk_id": chunk.id,
                    "chunk_index": chunk.chunk_index,
                    "chunk_type": chunk.chunk_type,
                    "content": content,
                    "knowledge_id": knowledge_id,
                    "knowledge_base": knowledge.knowledge_base_id,
                    "knowledge_title": knowledge.title or "",
                }
            )

        while True:
            chunks, total = await self._chunk_store.list_paged_chunks(
                tenant_id=knowledge.tenant_id,
                knowledge_id=knowledge_id,
                page=page,
                page_size=page_size,
                enabled_only=True,
            )
            if page == 1:
                total_chunks = int(total)
            if not chunks:
                break

            for chunk in chunks:
                chunk_num = chunk.chunk_index + 1
                chunk_content = _enrich_chunk_content(chunk)

                if has_range:
                    if chunk_num < start_chunk_index:
                        continue
                    if chunk_num > end_chunk_index:
                        reached_max = True
                        break
                    chunks_output.append(
                        f'<chunk index="{chunk_num}" type="range">\n{chunk_content}\n</chunk>'
                    )
                    append_formatted_chunk(chunk, chunk_content)
                    match_count += 1
                    continue

                is_match = compiled.search(chunk_content) is not None if compiled else True
                if is_match:
                    match_count += 1
                    if (
                        compiled is not None
                        and prev_chunk is not None
                        and prev_chunk.chunk_index not in outputted_indices
                    ):
                        prev_content = _enrich_chunk_content(prev_chunk)
                        chunks_output.append(
                            f'<chunk index="{prev_chunk.chunk_index + 1}" type="context_before">\n'
                            f"{prev_content}\n</chunk>"
                        )
                        append_formatted_chunk(prev_chunk, prev_content)
                        outputted_indices.add(prev_chunk.chunk_index)
                    if chunk.chunk_index not in outputted_indices:
                        match_attr = ' type="match"' if compiled is not None else ""
                        chunks_output.append(
                            f'<chunk index="{chunk_num}"{match_attr}>\n{chunk_content}\n</chunk>'
                        )
                        append_formatted_chunk(chunk, chunk_content)
                        outputted_indices.add(chunk.chunk_index)
                    if compiled is not None:
                        force_output_next = True
                elif force_output_next:
                    if chunk.chunk_index not in outputted_indices:
                        chunks_output.append(
                            f'<chunk index="{chunk_num}" type="context_after">\n'
                            f"{chunk_content}\n</chunk>"
                        )
                        append_formatted_chunk(chunk, chunk_content)
                        outputted_indices.add(chunk.chunk_index)
                    force_output_next = False

                prev_chunk = chunk
                if compiled is None and match_count >= 10:
                    break
                if compiled is not None and match_count >= 20:
                    break

            if has_range and reached_max:
                break
            if not has_range:
                if compiled is None and match_count >= 10:
                    break
                if compiled is not None and match_count >= 20:
                    reached_max = True
                    break
            if page * page_size >= int(total):
                break
            page += 1

        meta_parts.append(f"<total_chunks>{total_chunks}</total_chunks>")
        output_parts: list[str] = [
            "<source_document>\n<metadata>\n",
            "\n".join(meta_parts),
            "\n</metadata>\n",
        ]
        if match_count > 0:
            output_parts.append(f'<chunks count="{match_count}">\n')
            output_parts.extend(chunks_output)
            output_parts.append("</chunks>\n")
        else:
            output_parts.append('<chunks count="0" />\n')

        if reached_max:
            output_parts.append(
                "<message>Reached maximum limit for fetching chunks in a single call. "
                "Please refine your query or range if needed.</message>\n"
            )
        elif match_count == 0:
            if has_range:
                output_parts.append("<message>No chunks found in the specified range.</message>\n")
            elif compiled is not None:
                output_parts.append(
                    "<message>No chunks matched your query in this document.</message>\n"
                )
            else:
                output_parts.append("<message>Document has no text chunks available.</message>\n")
        elif not has_range and compiled is None:
            output_parts.append(
                "<message>No query or range provided. Showing the first 10 chunks as a preview.</message>\n"
            )
        output_parts.append("</source_document>")

        return ToolResult(
            success=True,
            output="".join(output_parts),
            data={
                "display_type": "knowledge_chunks_list",
                "knowledge_id": knowledge_id,
                "knowledge_title": knowledge.title or "",
                "total_chunks": total_chunks,
                "fetched_chunks": len(formatted_chunks),
                "chunks": cast("list[JsonValue]", formatted_chunks),
            },
        )

    async def _resolve_document(self, ctx: Context, knowledge_id: str) -> Knowledge | None:
        """Resolve the source document, applying the agent scope when enabled.

        Presence of search targets — not their length — enables the agent
        authorization boundary, so an empty scope fails closed.
        """
        if self._search_targets is not None:
            try:
                return await authorize_knowledge_in_search_targets(
                    ctx,
                    self._search_targets,
                    knowledge_id,
                    self._knowledge_service,
                    self._tag_fetcher,
                )
            except ApplicationError:
                return None
        try:
            return await self._knowledge_service.get_document_by_id_only(id=knowledge_id)
        except Exception:
            return None


# ── Shared argument parsing ───────────────────────────────────────────


def _parse_args(args: str) -> JsonObject:
    try:
        raw = json.loads(args)
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


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


def _as_str_list(value: JsonValue) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item != ""]
    if isinstance(value, str):
        return [value] if value != "" else []
    return []


def _dedup_non_empty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


__all__ = [
    "WIKI_BUDGET_RESERVE",
    "WIKI_INDEX_AGENT_TOP_K",
    "WIKI_LINK_SUMMARY_MAX_RUNES",
    "WIKI_MAX_LINK_SUMMARIES",
    "WIKI_MIN_PAGE_BODY",
    "WIKI_READ_PAGE_TOOL_DESCRIPTION",
    "WIKI_READ_PAGE_TOOL_NAME",
    "WIKI_READ_PAGE_TOOL_SCHEMA",
    "WIKI_READ_SOURCE_DOC_TOOL_DESCRIPTION",
    "WIKI_READ_SOURCE_DOC_TOOL_NAME",
    "WIKI_READ_SOURCE_DOC_TOOL_SCHEMA",
    "WikiReadPageTool",
    "WikiReadSourceDocTool",
    "build_wiki_read_page_definition",
    "build_wiki_read_source_doc_definition",
    "render_index_overview_for_agent",
    "render_wiki_pages_within_budget",
]
