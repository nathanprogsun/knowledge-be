"""Grep-chunks tool: regex search over knowledge-base chunk content.

Applies a single POSIX regular expression directly in the database
(PostgreSQL ``~*`` / SQLite ``REGEXP``, case-insensitive) against the
chunk body or the owning document's title, restricted to the session's
search scope. Chunks that were already surfaced by an earlier call in the
same session are rendered compactly with an ``already_seen`` marker.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from re import Pattern
from typing import Protocol, cast, runtime_checkable

from sqlalchemy import text
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncSession

from src.ai.embedding.base import Context
from src.common.json import JsonObject, JsonValue, SqlValue
from src.core.agents.tools.base import ToolDefinition, ToolResult
from src.core.agents.tools.faq_utils import (
    FAQ_CHUNK_TYPE,
    extract_chunk_match_snippet,
    faq_standard_question,
)
from src.core.agents.tools.scope_auth import search_target_scope
from src.core.agents.tools.search_target import SearchTarget, SearchTargets, SearchTargetType
from src.core.agents.tools.text_utils import (
    build_content_signature,
    count_regex_hits,
    jaccard,
    regex_matches_any,
    tokenize_simple,
    xml_escape,
)
from src.db.models.chunk import Chunk

#: Tool name constant.
GREP_TOOL_NAME = "grep_chunks"

GREP_TOOL_DESCRIPTION = (
    "Search knowledge base chunk content with a single POSIX regular "
    "expression, applied directly in the database (PostgreSQL ~* / MySQL/SQLite "
    "REGEXP, case-insensitive). Behaves like grep -E -i.\n"
    "Pack multiple concepts into ONE regex using | alternation — do not call "
    "this tool repeatedly for synonyms.\n"
    "Returns matching chunks with a short cN chunk source ID, a parent dN "
    "document ID, and a <match> snippet around the first match.\n"
    "Examples:\n"
    "- Alternation (RECOMMENDED): \"stardust|skyvault|psionic\" (matches any "
    "of the words)\n"
    "- Multiple terms in order: \"psionic.*engine\" (matches both words in "
    "order)\n"
    "- Word boundary / anchor: \"\\brag\\b\" or \"^chapter\\s+\\d+\"\n"
    "- Plain text: \"engine\" (matches literal substring anywhere in chunk "
    "content)\n"
    "IMPORTANT — JSON escaping: every backslash in a regex MUST be written as "
    "\\\\ inside the JSON tool arguments (e.g. to search for literal \"C++\" "
    "write \"C\\\\+\\\\+\", NOT \"C\\+\\+\"; for \"\\d+\" write \"\\\\d+\"). Plain "
    "\"\\+\" / \"\\d\" etc. are invalid JSON escapes and will fail to parse.\n"
    "Use this to locate candidate chunks by exact identifiers, error codes, "
    "product names, or recurring terms.\n"
    "\n"
    "## Deep read after grep:\n"
    "- **FAQ hit** (chunk type faq): call list_knowledge_chunks with "
    "faq_id=cN from the grep result (NOT the parent dN document ID).\n"
    "- **Document hit**: call list_knowledge_chunks with knowledge_id=dN, or "
    "get_document_info with knowledge_ids=[dN]."
)

GREP_TOOL_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A single POSIX regex applied directly to chunk content "
                    "(case-insensitive). Combine multiple concepts with \"|\" "
                    "alternation in ONE regex (e.g. \"stardust|skyvault|psionic\") "
                    "— do not split into multiple calls."
                ),
                "minLength": 1,
            }
        },
        "required": ["query"],
    },
    ensure_ascii=False,
)

#: Result count is controlled by the backend, not the caller.
_GREP_LIMIT = 30
#: Hard cap on rows fetched from the database before scoring.
_MAX_FETCH_LIMIT = 500
#: MMR is applied only when more than this many candidates survive scoring.
_MMR_THRESHOLD = 10
#: Per-knowledge aggregation rows returned for the UI.
_MAX_KNOWLEDGE_ROWS = 20


@dataclass(frozen=True, slots=True)
class GrepChunksInput:
    """Parsed input for the grep-chunks tool (canonical single ``query``)."""

    query: str = ""

    @classmethod
    def from_json(cls, raw: JsonObject) -> GrepChunksInput:
        return cls(query=_as_str(raw.get("query")))


def build_grep_chunks_definition() -> ToolDefinition:
    """Return the default tool definition for the grep tool."""
    return ToolDefinition(
        name=GREP_TOOL_NAME,
        description=GREP_TOOL_DESCRIPTION,
        parameters=GREP_TOOL_SCHEMA,
    )


def _as_str(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


@dataclass(frozen=True, slots=True)
class ChunkWithTitle:
    """A chunk row joined with its owning document's title."""

    chunk: Chunk
    knowledge_title: str = ""
    match_score: float = 0.0
    matched_patterns: int = 0
    title_match: bool = False
    total_chunk_count: int = 0


@dataclass(slots=True)
class _KnowledgeAggregation:
    """Mutable per-document hit summary built while aggregating."""

    knowledge_id: str
    knowledge_base_id: str
    knowledge_title: str
    faq_question: str = ""
    title_match: bool = False
    chunk_hit_count: int = 0
    total_chunk_count: int = 0
    pattern_counts: dict[str, int] = field(default_factory=dict)
    total_pattern_hits: int = 0
    distinct_patterns: int = 0
    match_snippet: str = ""

    def to_json(self) -> JsonObject:
        """Project onto the UI-facing result map."""
        return {
            "knowledge_id": self.knowledge_id,
            "knowledge_base_id": self.knowledge_base_id,
            "knowledge_title": self.knowledge_title,
            "faq_question": self.faq_question,
            "title_match": self.title_match,
            "chunk_hit_count": self.chunk_hit_count,
            "total_chunk_count": self.total_chunk_count,
            "pattern_counts": cast("dict[str, JsonValue]", self.pattern_counts),
            "total_pattern_hits": self.total_pattern_hits,
            "distinct_patterns": self.distinct_patterns,
            "match_snippet": self.match_snippet,
        }


class GrepChunksTool:
    """Performs regex pattern matching across knowledge-base chunks."""

    def __init__(
        self,
        *,
        definition: ToolDefinition,
        store: ChunkGrepStore,
        search_targets: SearchTargets,
    ) -> None:
        self._definition = definition
        self._store = store
        self._search_targets = search_targets
        # Previously-returned chunk ids for compact re-rendering within the
        # session (one tool instance per agent session).
        self._seen_chunks: set[str] = set()

    def name(self) -> str:
        return self._definition.name

    def description(self) -> str:
        return self._definition.description

    def parameters(self) -> str:
        return self._definition.parameters

    async def execute(self, ctx: Context, args: str) -> ToolResult:
        """Run the regex against the scope and return matching chunks."""
        input_ = _parse_input(args)
        query = input_.query.strip()
        if not query:
            return ToolResult(
                success=False,
                error="query parameter is required and must be a non-empty regex string",
            )

        # Compile with a case-insensitive prefix; compilation also validates
        # the regex syntax before it reaches the database.
        try:
            compiled: list[Pattern[str]] = [re.compile("(?i)" + query)]
        except re.error as exc:
            return ToolResult(
                success=False,
                error=f"invalid regex query {query!r}: {exc}",
            )
        queries = [query]

        kb_tenant_map = self._search_targets.get_kb_tenant_map()
        full_kb_ids, knowledge_ids, tag_targets = self._resolve_grep_scope()
        kb_ids_for_meta = full_kb_ids or self._search_targets.get_all_knowledge_base_ids()

        results = await self._store.search_chunks(
            query=query,
            full_kb_ids=full_kb_ids,
            knowledge_ids=knowledge_ids,
            tag_targets=tag_targets,
            kb_tenant_map=kb_tenant_map,
        )

        deduplicated = self._deduplicate_chunks(results)
        scored = self._score_chunks(deduplicated, compiled)

        final_results = scored
        if len(scored) > _MMR_THRESHOLD:
            mmr_k = min(len(scored), _GREP_LIMIT)
            mmr_results = self._apply_mmr(scored, mmr_k, 0.7)
            if mmr_results:
                final_results = mmr_results

        final_results = sorted(final_results, key=_grep_sort_key)
        if len(final_results) > _GREP_LIMIT:
            final_results = final_results[:_GREP_LIMIT]

        chunk_results = build_grep_chunk_results(final_results, compiled)
        aggregated = self._aggregate_by_knowledge(final_results, queries, compiled)
        document_count = len(aggregated)
        knowledge_results_for_ui = aggregated[:_MAX_KNOWLEDGE_ROWS]

        output = self._format_output(final_results, queries, compiled)

        return ToolResult(
            success=True,
            output=output,
            data={
                "query": query,
                "queries": cast("list[JsonValue]", queries),
                "patterns": cast("list[JsonValue]", queries),
                "chunk_results": cast("list[JsonValue]", chunk_results),
                "knowledge_results": cast("list[JsonValue]", knowledge_results_for_ui),
                "result_count": len(chunk_results),
                "document_count": document_count,
                "total_matches": len(final_results),
                "knowledge_base_ids": cast("list[JsonValue]", kb_ids_for_meta),
                "limit": _GREP_LIMIT,
                "max_results": _GREP_LIMIT,
                "display_type": "grep_results",
            },
        )

    # ── Scope resolution ────────────────────────────────────────────

    def _resolve_grep_scope(self) -> tuple[list[str], list[str], list[SearchTarget]]:
        """Split the search targets into full-KB, document, and tag scopes.

        A target that carries a resolved document whitelist is already the
        intersection of its mention and any tag scope, so it must be grepped
        by document id — falling back to the tag branch would widen the
        search to every document carrying the tag.
        """
        seen_kb: set[str] = set()
        seen_knowledge: set[str] = set()
        seen_tag_scope: set[str] = set()
        full_kb_ids: list[str] = []
        knowledge_ids: list[str] = []
        tag_targets: list[SearchTarget] = []

        for target in self._search_targets:
            if target is None or not target.knowledge_base_id:
                continue
            target_knowledge_ids, target_tag_ids = search_target_scope(target)
            if target_tag_ids:
                tenant_id = target.tenant_id or self._search_targets.get_tenant_id_for_kb(
                    target.knowledge_base_id
                )
                tag_ids = target_tag_ids
                if not tag_ids or tenant_id == 0:
                    continue
                scope_key = (
                    f"{target.knowledge_base_id}:{tenant_id}:"
                    + "\x00".join(tag_ids)
                )
                if scope_key in seen_tag_scope:
                    continue
                seen_tag_scope.add(scope_key)
                tag_targets.append(
                    SearchTarget(
                        type=SearchTargetType.KNOWLEDGE_BASE,
                        knowledge_base_id=target.knowledge_base_id,
                        tenant_id=tenant_id,
                        tag_ids=tuple(tag_ids),
                    )
                )
            elif target_knowledge_ids:
                for kid in target_knowledge_ids:
                    if kid not in seen_knowledge:
                        seen_knowledge.add(kid)
                        knowledge_ids.append(kid)
            else:
                if target.knowledge_base_id not in seen_kb:
                    seen_kb.add(target.knowledge_base_id)
                    full_kb_ids.append(target.knowledge_base_id)

        return full_kb_ids, knowledge_ids, tag_targets

    # ── Dedup / scoring / MMR ───────────────────────────────────────

    def _deduplicate_chunks(self, results: list[ChunkWithTitle]) -> list[ChunkWithTitle]:
        """Remove duplicate / near-duplicate chunks by id and content."""
        seen: set[str] = set()
        content_sig: set[str] = set()
        unique: list[ChunkWithTitle] = []

        for row in results:
            keys = [row.chunk.id]
            if row.chunk.parent_chunk_id:
                keys.append("parent:" + row.chunk.parent_chunk_id)
            if row.chunk.knowledge_id:
                keys.append(f"kb:{row.chunk.knowledge_id}#{row.chunk.chunk_index}")
            if any(key in seen for key in keys):
                continue
            signature = _build_signature(row.chunk.content)
            if signature:
                if signature in content_sig:
                    continue
                content_sig.add(signature)
            seen.update(keys)
            unique.append(row)
        return unique

    def _score_chunks(
        self,
        results: list[ChunkWithTitle],
        compiled: list[Pattern[str]],
    ) -> list[ChunkWithTitle]:
        """Score chunks by regex counts + an earliest-position boost."""
        scored: list[ChunkWithTitle] = []
        for row in results:
            score, pattern_count = self._calculate_match_score(row.chunk.content, compiled)
            title_match = False
            if regex_matches_any(row.knowledge_title, compiled):
                title_match = True
                score = min(score + 0.5, 1.0)
                if pattern_count == 0:
                    pattern_count = 1
            scored.append(
                replace(
                    row,
                    match_score=score,
                    matched_patterns=pattern_count,
                    title_match=title_match,
                )
            )
        return scored

    def _calculate_match_score(
        self,
        content: str,
        compiled: list[Pattern[str]],
    ) -> tuple[float, int]:
        """Count matched patterns and apply a small earliest-position boost."""
        if not content or not compiled:
            return 0.0, 0
        match_count = 0
        earliest_pos = len(content)
        for pattern in compiled:
            match = pattern.search(content)
            if match is None:
                continue
            match_count += 1
            if match.start() < earliest_pos:
                earliest_pos = match.start()
        if match_count == 0:
            return 0.0, 0
        base_score = match_count / len(compiled)
        position_bonus = 0.0
        if earliest_pos < len(content):
            position_ratio = 1.0 - earliest_pos / len(content)
            position_bonus = position_ratio * 0.1
        return min(base_score + position_bonus, 1.0), match_count

    def _apply_mmr(
        self,
        results: list[ChunkWithTitle],
        k: int,
        lambda_value: float,
    ) -> list[ChunkWithTitle]:
        """Apply Maximal Marginal Relevance to reduce redundancy."""
        if k <= 0 or not results:
            return []
        selected: list[ChunkWithTitle] = []
        selected_token_sets: list[set[str]] = []
        candidates = list(results)
        token_sets = [tokenize_simple(row.chunk.content) for row in candidates]

        while len(selected) < k and candidates:
            best_idx = 0
            best_score = -1.0
            for i, row in enumerate(candidates):
                relevance = row.match_score
                redundancy = 0.0
                for selected_tokens in selected_token_sets:
                    redundancy = max(
                        redundancy,
                        jaccard(token_sets[i], selected_tokens),
                    )
                mmr = lambda_value * relevance - (1.0 - lambda_value) * redundancy
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i
            selected.append(candidates[best_idx])
            selected_token_sets.append(token_sets[best_idx])
            last = len(candidates) - 1
            candidates[best_idx] = candidates[last]
            token_sets[best_idx] = token_sets[last]
            candidates.pop()
            token_sets.pop()
        return selected

    # ── Output ──────────────────────────────────────────────────────

    def _format_output(
        self,
        results: list[ChunkWithTitle],
        queries: list[str],
        compiled: list[Pattern[str]],
    ) -> str:
        """Emit per-chunk XML with match snippets / query hits."""
        parts: list[str] = []
        parts.append(f'<grep_results chunk_count="{len(results)}">\n')
        for query in queries:
            parts.append(f"<query>{xml_escape(query)}</query>\n")
        if not results:
            parts.append("</grep_results>")
            return "".join(parts)

        for row in results:
            counts = count_regex_hits(row.chunk.content, compiled, queries)
            snippet = extract_chunk_match_snippet(row.chunk, compiled)
            extra_attr = ""
            question = faq_standard_question(row.chunk)
            if question:
                extra_attr = f' faq_question="{xml_escape(question)}"'
            is_faq = row.chunk.chunk_type == FAQ_CHUNK_TYPE

            seen = row.chunk.id in self._seen_chunks
            self._seen_chunks.add(row.chunk.id)

            if is_faq:
                seen_attr = ' already_seen="true"' if seen else ""
                parts.append(
                    f'<faq faq_id="{xml_escape(row.chunk.id)}" '
                    f'knowledge_title="{xml_escape(row.knowledge_title)}"{extra_attr} '
                    f'index="{row.chunk.chunk_index}" score="{row.match_score:.3f}"'
                    f'{seen_attr}>\n'
                )
            else:
                seen_attr = ' already_seen="true"' if seen else ""
                parts.append(
                    f'<chunk chunk_id="{xml_escape(row.chunk.id)}" '
                    f'knowledge_id="{xml_escape(row.chunk.knowledge_id)}" '
                    f'knowledge_title="{xml_escape(row.knowledge_title)}"{extra_attr} '
                    f'chunk_index="{row.chunk.chunk_index}" score="{row.match_score:.3f}"'
                    f'{seen_attr}>\n'
                )

            for query in queries:
                if counts.get(query, 0) > 0:
                    parts.append(
                        f'<query_hit query="{xml_escape(query)}" count="{counts[query]}" />\n'
                    )
            if seen:
                parts.append(
                    "<note>(snippet omitted, already returned in a previous "
                    "grep_chunks call this session)</note>\n"
                )
            elif snippet:
                parts.append(f"<match_snippet>{xml_escape(snippet)}</match_snippet>\n")

            if is_faq:
                parts.append("</faq>\n")
            else:
                parts.append("</chunk>\n")

        parts.append("</grep_results>")
        return "".join(parts)

    def _aggregate_by_knowledge(
        self,
        results: list[ChunkWithTitle],
        queries: list[str],
        compiled: list[Pattern[str]],
    ) -> list[JsonObject]:
        """Pre-aggregate per-document hit summaries for the UI."""
        if not results:
            return []
        query_keys = [query for query in queries if query.strip()]

        aggregated: dict[str, _KnowledgeAggregation] = {}
        for row in results:
            knowledge_id = row.chunk.knowledge_id or f"chunk-{row.chunk.id}"
            entry = aggregated.get(knowledge_id)
            if entry is None:
                title = row.knowledge_title.strip() or "Untitled"
                entry = _KnowledgeAggregation(
                    knowledge_id=knowledge_id,
                    knowledge_base_id=row.chunk.knowledge_base_id,
                    knowledge_title=title,
                    total_chunk_count=row.total_chunk_count,
                    pattern_counts=dict.fromkeys(query_keys, 0),
                )
                aggregated[knowledge_id] = entry

            entry.chunk_hit_count += 1
            if row.title_match:
                entry.title_match = True
            if not entry.faq_question:
                question = faq_standard_question(row.chunk)
                if question:
                    entry.faq_question = question
            if not entry.match_snippet:
                snippet = extract_chunk_match_snippet(row.chunk, compiled)
                if snippet:
                    entry.match_snippet = snippet

            occurrences = count_regex_hits(row.chunk.content, compiled, query_keys)
            for key in query_keys:
                count = occurrences.get(key, 0)
                if count == 0:
                    continue
                entry.pattern_counts[key] = entry.pattern_counts[key] + count
                entry.total_pattern_hits += count

        result_slice = list(aggregated.values())
        for entry in result_slice:
            entry.distinct_patterns = sum(
                1 for count in entry.pattern_counts.values() if count > 0
            )
        result_slice.sort(key=_aggregation_sort_key)
        return [entry.to_json() for entry in result_slice]


def build_grep_chunk_results(
    results: list[ChunkWithTitle],
    compiled: list[Pattern[str]],
) -> list[JsonObject]:
    """Project per-chunk hits for the UI detail view."""
    out: list[JsonObject] = []
    for row in results:
        item: JsonObject = {
            "knowledge_id": row.chunk.knowledge_id,
            "knowledge_base_id": row.chunk.knowledge_base_id,
            "knowledge_title": row.knowledge_title,
            "chunk_type": row.chunk.chunk_type,
            "title_match": row.title_match,
            "match_snippet": extract_chunk_match_snippet(row.chunk, compiled),
            "score": row.match_score,
        }
        if row.chunk.chunk_type == FAQ_CHUNK_TYPE:
            item["faq_id"] = row.chunk.id
            item["index"] = row.chunk.chunk_index
            question = faq_standard_question(row.chunk)
            if question:
                item["faq_question"] = question
        else:
            item["chunk_id"] = row.chunk.id
            item["chunk_index"] = row.chunk.chunk_index
        out.append(item)
    return out


def _grep_sort_key(row: ChunkWithTitle) -> tuple[bool, int, float, int]:
    """Title hits first, then patterns, then score, then chunk order."""
    return (
        not row.title_match,
        -row.matched_patterns,
        -row.match_score,
        row.chunk.chunk_index,
    )


def _aggregation_sort_key(entry: _KnowledgeAggregation) -> tuple[bool, int, int, int, str]:
    return (
        not entry.title_match,
        -entry.distinct_patterns,
        -entry.total_pattern_hits,
        -entry.chunk_hit_count,
        entry.knowledge_title,
    )


def _build_signature(content: str) -> str:
    return build_content_signature(content)


def _parse_input(args: str) -> GrepChunksInput:
    try:
        raw = json.loads(args)
    except json.JSONDecodeError:
        raw = {}
    return GrepChunksInput.from_json(raw if isinstance(raw, dict) else {})


# ── Database seam ─────────────────────────────────────────────────────


@runtime_checkable
class ChunkGrepStore(Protocol):
    """Applies a regex query to the scoped chunk set and returns matches."""

    async def search_chunks(
        self,
        *,
        query: str,
        full_kb_ids: list[str],
        knowledge_ids: list[str],
        tag_targets: list[SearchTarget],
        kb_tenant_map: dict[str, int],
    ) -> list[ChunkWithTitle]: ...


class SqlChunkGrepStore:
    """``chunks`` / ``documents`` implementation over an ``AsyncSession``."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def search_chunks(
        self,
        *,
        query: str,
        full_kb_ids: list[str],
        knowledge_ids: list[str],
        tag_targets: list[SearchTarget],
        kb_tenant_map: dict[str, int],
    ) -> list[ChunkWithTitle]:
        if not full_kb_ids and not knowledge_ids and not tag_targets:
            return []
        scope = _scope_clause(full_kb_ids, knowledge_ids, tag_targets, kb_tenant_map)
        if scope is None:
            return []

        regex_op = _regex_operator(self._session)
        scope_sql, params = scope
        params["enabled"] = True
        params["pattern"] = query
        params["fetch_limit"] = _MAX_FETCH_LIMIT

        sql = text(
            "select chunks.*, documents.title as knowledge_title "
            "from chunks "
            "join documents on chunks.knowledge_id = documents.id "
            "where chunks.is_enabled = :enabled "
            "and chunks.deleted_at is null "
            "and documents.deleted_at is null "
            f"and (chunks.content {regex_op} :pattern "
            f"or documents.title {regex_op} :pattern) "
            f"and {scope_sql} "
            "order by chunks.created_at desc limit :fetch_limit"
        ).bindparams(**params)

        result = await self._session.execute(sql)
        rows = result.mappings().all()
        chunks = [self._to_chunk_with_title(mapping) for mapping in rows]

        counts = await self._count_by_knowledge_ids(
            [row.chunk.knowledge_id for row in chunks if row.chunk.knowledge_id]
        )
        if counts:
            chunks = [
                replace(row, total_chunk_count=counts.get(row.chunk.knowledge_id, 0))
                for row in chunks
            ]
        return chunks

    async def _count_by_knowledge_ids(self, knowledge_ids: list[str]) -> dict[str, int]:
        """Count live chunks per document for the aggregation view."""
        if not knowledge_ids:
            return {}
        unique = list(dict.fromkeys(knowledge_ids))
        placeholders = ", ".join(f":kid_{i}" for i in range(len(unique)))
        params: dict[str, SqlValue] = {"enabled": True}
        params.update({f"kid_{i}": kid for i, kid in enumerate(unique)})
        sql = text(
            "select knowledge_id, count(*) as cnt from chunks "
            f"where knowledge_id in ({placeholders}) "
            "and is_enabled = :enabled and deleted_at is null "
            "group by knowledge_id"
        ).bindparams(**params)
        result = await self._session.execute(sql)
        return {
            str(mapping["knowledge_id"]): int(mapping["cnt"])
            for mapping in result.mappings().all()
        }

    def _to_chunk_with_title(self, mapping: RowMapping) -> ChunkWithTitle:
        raw = dict(mapping)
        title = raw.pop("knowledge_title", None)
        chunk = cast("Chunk", Chunk.from_row(cast("dict[str, SqlValue]", raw)))
        return ChunkWithTitle(chunk=chunk, knowledge_title=str(title) if title else "")


def _regex_operator(session: AsyncSession) -> str:
    """Return the case-insensitive regex operator for the current dialect."""
    try:
        dialect = session.get_bind().dialect.name
    except Exception:
        dialect = ""
    if dialect == "postgresql":
        return "~*"
    return "REGEXP"


def _scope_clause(
    full_kb_ids: list[str],
    knowledge_ids: list[str],
    tag_targets: list[SearchTarget],
    kb_tenant_map: dict[str, int],
) -> tuple[str, dict[str, SqlValue]] | None:
    """Build the OR'd scope predicate (documents / tag scopes / full KBs).

    Returns ``None`` when no usable scope exists.
    """
    clauses: list[str] = []
    params: dict[str, SqlValue] = {}

    if knowledge_ids:
        placeholders = ", ".join(f":kid_{i}" for i in range(len(knowledge_ids)))
        clauses.append(f"chunks.knowledge_id in ({placeholders})")
        params.update({f"kid_{i}": kid for i, kid in enumerate(knowledge_ids)})

    for tag_index, target in enumerate(tag_targets):
        if (
            target is None
            or not target.knowledge_base_id
            or not target.tag_ids
        ):
            continue
        tenant_id = target.tenant_id or kb_tenant_map.get(target.knowledge_base_id, 0)
        if tenant_id == 0:
            continue
        tag_placeholders = ", ".join(
            f":tag_{tag_index}_{i}" for i in range(len(target.tag_ids))
        )
        clauses.append(
            f"(chunks.knowledge_base_id = :tkb_{tag_index} "
            f"and chunks.tenant_id = :tten_{tag_index} and exists ("
            "select 1 from document_tags ktr "
            "where ktr.knowledge_id = chunks.knowledge_id "
            f"and ktr.tag_id in ({tag_placeholders})))"
        )
        params[f"tkb_{tag_index}"] = target.knowledge_base_id
        params[f"tten_{tag_index}"] = tenant_id
        params.update(
            {f"tag_{tag_index}_{i}": tag_id for i, tag_id in enumerate(target.tag_ids)}
        )

    for kb_index, kb_id in enumerate(full_kb_ids):
        tenant_id = kb_tenant_map.get(kb_id, 0)
        if tenant_id == 0:
            continue
        clauses.append(
            f"(chunks.knowledge_base_id = :fkb_{kb_index} "
            f"and chunks.tenant_id = :ften_{kb_index})"
        )
        params[f"fkb_{kb_index}"] = kb_id
        params[f"ften_{kb_index}"] = tenant_id

    if not clauses:
        return None
    return "(" + " or ".join(clauses) + ")", params


__all__ = [
    "GREP_TOOL_DESCRIPTION",
    "GREP_TOOL_NAME",
    "GREP_TOOL_SCHEMA",
    "ChunkGrepStore",
    "ChunkWithTitle",
    "GrepChunksInput",
    "GrepChunksTool",
    "SqlChunkGrepStore",
    "build_grep_chunk_results",
    "build_grep_chunks_definition",
]
