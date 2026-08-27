"""Knowledge-search tool: semantic retrieval across knowledge bases.

``knowledge_search`` is the vector/keyword retrieval tool of the agent
layer. It fans out hybrid search across the session's search targets,
deduplicates near-duplicate chunks, optionally reranks with a rerank
model or an LLM, applies maximal-marginal-relevance diversification, and
renders the surviving hits as XML (with a ``search_results`` shape shared
with the sibling retrieval tools) plus a structured result map.

The tool executes through injected seams so the layer stays free of
direct storage / LLM access: a ``SearchRunner`` performs the actual
hybrid search, a ``KbLoader`` supplies knowledge-base records for type
and embedding-model grouping, and an optional ``ChunkCounter`` feeds the
per-document totals shown in the retrieval statistics. Reranking is
optional — a ``Reranker`` model or a ``Chat`` model (LLM-based rerank)
or both; when neither is configured results pass through unchanged.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from typing import Protocol, cast, runtime_checkable

from src.ai.embedding.base import Context
from src.ai.llm.types import Chat, ChatOptions, Message
from src.common.json import JsonObject, JsonValue
from src.core.agents.tools.base import ToolDefinition, ToolResult
from src.core.agents.tools.chunk_store import PagedChunkStore
from src.core.agents.tools.faq_utils import (
    FAQChunkMetadata,
    append_similar_questions_to_chunk_data,
    faq_match_snippet_from_queries,
    faq_metadata_from_json,
    write_faq_fields_xml,
)
from src.core.agents.tools.scope_auth import validate_knowledge_base_ids_in_search_targets
from src.core.agents.tools.search_target import SearchTarget, SearchTargets, SearchTargetType
from src.core.agents.tools.text_utils import (
    build_content_signature,
    build_image_info_markdown,
    clamp_float,
    extract_snippet_for_queries,
    jaccard,
    parse_image_infos,
    tokenize_simple,
    xml_escape,
)
from src.core.knowledge.knowledge_bases.hybrid_search import (
    HybridSearchParams,
    SearchDependencies,
    SearchResult,
    hybrid_search,
)
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo

#: Tool name constant (kept here to avoid a dependency cycle with base).
KNOWLEDGE_SEARCH_TOOL_NAME = "knowledge_search"

KNOWLEDGE_SEARCH_TOOL_DESCRIPTION = (
    "Semantic/vector search tool for retrieving knowledge by meaning, "
    "intent, and conceptual relevance.\n"
    "\n"
    "This tool uses embeddings to understand the user's query and find "
    "semantically similar content across knowledge base chunks.\n"
    "\n"
    "## Purpose\n"
    "Designed for high-level understanding tasks, such as:\n"
    "- conceptual explanations\n"
    "- topic overviews\n"
    "- reasoning-based information needs\n"
    "- contextual or intent-driven retrieval\n"
    "- queries that cannot be answered with literal keyword matching\n"
    "\n"
    "The tool searches by MEANING rather than exact text. It identifies "
    "chunks that are conceptually relevant even when the wording differs.\n"
    "\n"
    "## What the Tool Does NOT Do\n"
    "- Does NOT perform exact keyword matching\n"
    "- Does NOT search for specific named entities\n"
    "- Should NOT be used for literal lookup tasks\n"
    "- Should NOT receive long raw text or user messages as queries\n"
    "- Should NOT be used to locate specific strings or error codes\n"
    "\n"
    "For literal/keyword/entity search, another tool should be used.\n"
    "\n"
    "## Required Input Behavior\n"
    '"queries" must contain **1-5 short, well-formed semantic questions or '
    "conceptual statements** that clearly express the meaning the model is "
    "trying to retrieve.\n"
    "\n"
    "Each query should represent a **concept, idea, topic, explanation, or "
    "intent**, such as:\n"
    "- abstract topics\n"
    "- definitions\n"
    "- mechanisms\n"
    "- best practices\n"
    "- comparisons\n"
    "- how/why questions\n"
    "\n"
    "Avoid:\n"
    "- keyword lists\n"
    "- raw text from user messages\n"
    "- full paragraphs\n"
    "- unprocessed input\n"
    "\n"
    "## Examples of valid query shapes (not content):\n"
    '- "What is the main idea of..."\n'
    '- "How does X work in general?"\n'
    '- "Explain the purpose of..."\n'
    '- "What are the key principles behind..."\n'
    '- "Overview of ..."\n'
    "\n"
    "## Parameters\n"
    "- queries (required): 1-5 semantic questions or conceptual statements.\n"
    "  These should reflect the meaning or topic you want embeddings to capture.\n"
    "- knowledge_base_ids (optional): limit the search scope.\n"
    "\n"
    "## Output\n"
    "Returns chunks ranked by semantic similarity, reranked when applicable.  \n"
    "Each chunk has a short cN source ID and belongs to a dN document ID. "
    "Results represent conceptual relevance, not literal keyword overlap. Use "
    "dN for document-level follow-up tool calls."
)

KNOWLEDGE_SEARCH_TOOL_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "description": (
                    "REQUIRED: 1-5 semantic questions/topics (e.g., "
                    "['What is RAG?', 'RAG benefits'])"
                ),
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 5,
            },
            "knowledge_base_ids": {
                "type": "array",
                "description": (
                    "Optional: bound knowledge-base IDs (the short bN values "
                    "shown in runtime context)"
                ),
                "items": {"type": "string"},
                "minItems": 0,
                "maxItems": 10,
            },
        },
        "required": ["queries"],
    },
    ensure_ascii=False,
)

#: Default retrieval parameters (used when no config is supplied).
DEFAULT_TOP_K = 5
DEFAULT_VECTOR_THRESHOLD = 0.6
DEFAULT_KEYWORD_THRESHOLD = 0.5
#: Default rerank gate; results below it are dropped unless preserved.
DEFAULT_RERANK_THRESHOLD = 0.3
#: Lowest relevance a preserved top rerank result may carry.
_RERANK_FALLBACK_MIN_SCORE = 0.15
#: MMR balance between relevance and diversity.
_MMR_LAMBDA = 0.7
#: LLM rerank batch size and per-passage content cap.
_LLM_RERANK_BATCH_SIZE = 15
_LLM_RERANK_MAX_CONTENT_LENGTH = 800


def _indexing_flag(kb: KnowledgeBaseInfo, key: str) -> bool:
    """Read an indexing-strategy flag; a missing strategy defaults to on."""
    strategy = kb.indexing_strategy
    if strategy is None:
        return True
    value = strategy.get(key)
    return value if isinstance(value, bool) else True


def _is_vector_enabled(kb: KnowledgeBaseInfo) -> bool:
    return _indexing_flag(kb, "vector_enabled")


def _is_keyword_enabled(kb: KnowledgeBaseInfo) -> bool:
    return _indexing_flag(kb, "keyword_enabled")


def build_knowledge_search_definition() -> ToolDefinition:
    """Return the default tool definition for the knowledge-search tool."""
    return ToolDefinition(
        name=KNOWLEDGE_SEARCH_TOOL_NAME,
        description=KNOWLEDGE_SEARCH_TOOL_DESCRIPTION,
        parameters=KNOWLEDGE_SEARCH_TOOL_SCHEMA,
    )


@dataclass(frozen=True, slots=True)
class KnowledgeSearchInput:
    """Parsed input for the knowledge-search tool."""

    queries: tuple[str, ...] = ()
    knowledge_base_ids: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, raw: JsonObject) -> KnowledgeSearchInput:
        return cls(
            queries=_as_str_list(raw.get("queries")),
            knowledge_base_ids=_as_str_list(raw.get("knowledge_base_ids")),
        )


def _as_str(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


def _as_str_list(value: JsonValue) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(item for item in value if isinstance(item, str))
    return ()


def _parse_input(args: str) -> KnowledgeSearchInput:
    try:
        raw = json.loads(args)
    except json.JSONDecodeError:
        raw = {}
    return KnowledgeSearchInput.from_json(raw if isinstance(raw, dict) else {})


@dataclass(frozen=True, slots=True)
class SearchCall:
    """One hybrid-search invocation issued by the tool."""

    query_text: str
    kb_id: str
    knowledge_base_ids: tuple[str, ...] = ()
    knowledge_ids: tuple[str, ...] = ()
    tag_ids: tuple[str, ...] = ()
    top_k: int = 0
    vector_threshold: float = 0.0
    keyword_threshold: float = 0.0


@runtime_checkable
class SearchRunner(Protocol):
    """Executes one hybrid-search call and returns the hydrated hits."""

    async def search(self, ctx: Context, call: SearchCall) -> list[SearchResult]: ...


class HybridSearchRunner:
    """``SearchRunner`` adapter over the ``hybrid_search`` orchestrator."""

    def __init__(self, deps: SearchDependencies) -> None:
        self._deps = deps

    async def search(self, ctx: Context, call: SearchCall) -> list[SearchResult]:
        params = HybridSearchParams(
            query_text=call.query_text,
            knowledge_base_ids=call.knowledge_base_ids,
            knowledge_ids=call.knowledge_ids,
            tag_ids=call.tag_ids,
            match_count=call.top_k,
            vector_threshold=call.vector_threshold,
            keyword_threshold=call.keyword_threshold,
        )
        results = await hybrid_search(ctx, kb_id=call.kb_id, params=params, deps=self._deps)
        return results or []


@runtime_checkable
class KbLoader(Protocol):
    """Loads knowledge-base records by id (authorization is the caller's job)."""

    async def load_by_ids(self, ids: list[str]) -> list[KnowledgeBaseInfo]: ...


@runtime_checkable
class ChunkCounter(Protocol):
    """Counts the live text+FAQ chunks of one document."""

    async def count_chunks(self, *, tenant_id: int, knowledge_id: str) -> int: ...


class PagedChunkStoreChunkCounter:
    """``ChunkCounter`` adapter over a ``PagedChunkStore`` (page size 1)."""

    def __init__(self, store: PagedChunkStore) -> None:
        self._store = store

    async def count_chunks(self, *, tenant_id: int, knowledge_id: str) -> int:
        _chunks, total = await self._store.list_paged_chunks(
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
            page=1,
            page_size=1,
            enabled_only=True,
        )
        return int(total)


@runtime_checkable
class RerankItem(Protocol):
    """One ranked hit as consumed by the tool's rerank paths."""

    index: int
    relevance_score: float


@runtime_checkable
class ModelReranker(Protocol):
    """Model-backed reranker consumed by the tool (structural)."""

    async def rerank(self, query: str, documents: list[str]) -> list[RerankItem]: ...


@dataclass(slots=True)
class SearchResultMeta:
    """A search hit plus the query and knowledge-base context it came from."""

    result: SearchResult
    source_query: str = ""
    query_type: str = "hybrid"
    knowledge_base_id: str = ""
    knowledge_base_type: str = ""


@dataclass(frozen=True, slots=True)
class _RankItem:
    """Minimal ranked-hit carrier consumed by the rerank paths."""

    index: int
    relevance_score: float


class KnowledgeSearchTool:
    """Searches knowledge bases with flexible query modes.

    ``_seen_chunks`` lets repeated calls in the same session surface
    previously-returned chunks in a compact form so the model does not
    burn tokens re-reading identical content.
    """

    def __init__(
        self,
        *,
        definition: ToolDefinition,
        search_targets: SearchTargets,
        runner: SearchRunner,
        kb_loader: KbLoader | None = None,
        chunk_counter: ChunkCounter | None = None,
        reranker: ModelReranker | None = None,
        chat: Chat | None = None,
        top_k: int = DEFAULT_TOP_K,
        vector_threshold: float = DEFAULT_VECTOR_THRESHOLD,
        keyword_threshold: float = DEFAULT_KEYWORD_THRESHOLD,
        rerank_threshold: float = DEFAULT_RERANK_THRESHOLD,
    ) -> None:
        self._definition = definition
        self._search_targets = search_targets
        self._runner = runner
        self._kb_loader = kb_loader
        self._chunk_counter = chunk_counter
        self._reranker = reranker
        self._chat = chat
        self._top_k = top_k
        self._vector_threshold = vector_threshold
        self._keyword_threshold = keyword_threshold
        self._rerank_threshold = rerank_threshold
        self._seen_chunks: set[str] = set()

    def name(self) -> str:
        return self._definition.name

    def description(self) -> str:
        return self._definition.description

    def parameters(self) -> str:
        return self._definition.parameters

    # ── Execution ───────────────────────────────────────────────────

    async def execute(self, ctx: Context, args: str) -> ToolResult:
        """Run hybrid search across the scope and format the hits."""
        input_ = _parse_input(args)

        # The user can optionally narrow the search to specific knowledge
        # bases; every one must lie inside the session scope.
        search_targets = self._search_targets
        if input_.knowledge_base_ids:
            user_kb_ids = list(input_.knowledge_base_ids)
            validate_knowledge_base_ids_in_search_targets(search_targets, user_kb_ids)
            search_targets = search_targets.filter_by_kb_ids(tuple(user_kb_ids))
        if not list(search_targets):
            return ToolResult(
                success=False,
                error="no knowledge bases specified and no search targets configured",
            )

        queries = list(input_.queries)
        if not queries:
            return ToolResult(success=False, error="queries parameter is required")

        kb_ids = search_targets.get_all_knowledge_base_ids()
        kb_type_map: dict[str, str] = {}
        if self._kb_loader is not None:
            try:
                kbs = await self._kb_loader.load_by_ids(kb_ids)
            except Exception:
                kbs = []
            for kb in kbs:
                if kb is not None and kb.id:
                    kb_type_map[kb.id] = kb.type

        all_results = await self._concurrent_search_by_targets(
            ctx,
            queries,
            search_targets,
            self._top_k,
            self._vector_threshold,
            self._keyword_threshold,
            kb_type_map,
        )

        deduplicated = self._deduplicate_results(all_results)
        filtered = await self._maybe_rerank(ctx, queries, deduplicated)

        if filtered:
            mmr_k = min(len(filtered), self._top_k) if self._top_k > 0 else len(filtered)
            if mmr_k < 1:
                mmr_k = 1
            mmr_results = self._apply_mmr(filtered, mmr_k, _MMR_LAMBDA)
            if mmr_results:
                filtered = mmr_results

        final_results = self._deduplicate_results(filtered)
        final_results.sort(key=lambda meta: (-meta.result.score, meta.result.knowledge_id))

        return await self._format_output(final_results, kb_ids, queries)

    # ── Retrieval fan-out ───────────────────────────────────────────

    async def _concurrent_search_by_targets(
        self,
        ctx: Context,
        queries: list[str],
        search_targets: SearchTargets,
        top_k: int,
        vector_threshold: float,
        keyword_threshold: float,
        kb_type_map: dict[str, str],
    ) -> list[SearchResultMeta]:
        """Fan hybrid search out across the targets for every query.

        Knowledge-base targets sharing one embedding model are grouped so
        the combined retrieval covers a whole model group in one call;
        document- and tag-scoped targets are searched individually.
        """
        kb_ids = search_targets.get_all_knowledge_base_ids()
        kb_list: list[KnowledgeBaseInfo] = []
        if self._kb_loader is not None:
            try:
                kbs = await self._kb_loader.load_by_ids(kb_ids)
            except Exception:
                # KB records are only used for grouping and searchability; a
                # failed batch load degrades to treating every KB as unknown.
                kbs = []
            kb_list = [kb for kb in kbs if kb is not None]

        # Drop KBs that carry no vector/keyword index (wiki/graph-only scopes
        # must be queried through their own tools). KBs we could not fetch are
        # kept so the downstream search can still surface the real error.
        known_kbs = {kb.id: True for kb in kb_list}
        searchable_kbs = {
            kb.id: True for kb in kb_list if _is_vector_enabled(kb) or _is_keyword_enabled(kb)
        }
        filtered_targets: list[SearchTarget] = []
        for target in search_targets:
            if target is None or not target.knowledge_base_id:
                continue
            if searchable_kbs.get(target.knowledge_base_id):
                filtered_targets.append(target)
                continue
            if target.knowledge_base_id in known_kbs:
                continue
            filtered_targets.append(target)
        if not filtered_targets:
            return []
        search_targets = SearchTargets(targets=tuple(filtered_targets))

        model_key_map = {kb.id: kb.embedding_model_id or "" for kb in kb_list}
        groups: dict[str, list[SearchTarget]] = {}
        for target in search_targets:
            key = model_key_map.get(target.knowledge_base_id, "")
            groups.setdefault(key, []).append(target)

        batches = [
            self._search_group(
                ctx,
                query,
                targets,
                top_k,
                vector_threshold,
                keyword_threshold,
                kb_type_map,
            )
            for query in queries
            for targets in groups.values()
        ]
        if not batches:
            return []
        collected = await asyncio.gather(*batches)
        return [meta for batch in collected for meta in batch]

    async def _search_group(
        self,
        ctx: Context,
        query: str,
        targets: list[SearchTarget],
        top_k: int,
        vector_threshold: float,
        keyword_threshold: float,
        kb_type_map: dict[str, str],
    ) -> list[SearchResultMeta]:
        """Run the combined full-KB call plus per-scope calls for a group."""
        full_kb_ids: list[str] = []
        knowledge_targets: list[SearchTarget] = []
        for target in targets:
            if target.type is SearchTargetType.KNOWLEDGE_BASE and not target.tag_ids:
                full_kb_ids.append(target.knowledge_base_id)
            else:
                knowledge_targets.append(target)

        calls: list[SearchCall] = []
        if full_kb_ids:
            calls.append(
                SearchCall(
                    query_text=query,
                    kb_id=full_kb_ids[0],
                    knowledge_base_ids=tuple(full_kb_ids),
                    top_k=top_k,
                    vector_threshold=vector_threshold,
                    keyword_threshold=keyword_threshold,
                )
            )
        for target in knowledge_targets:
            vec_threshold, kw_threshold = target.recall_thresholds(
                vector_threshold, keyword_threshold
            )
            calls.append(
                SearchCall(
                    query_text=query,
                    kb_id=target.knowledge_base_id,
                    knowledge_ids=tuple(target.knowledge_ids),
                    tag_ids=tuple(list(target.tag_ids) + list(target.scope_tag_ids)),
                    top_k=top_k,
                    vector_threshold=vec_threshold,
                    keyword_threshold=kw_threshold,
                )
            )
        if not calls:
            return []

        async def _run(call: SearchCall) -> list[SearchResult]:
            try:
                return await self._runner.search(ctx, call)
            except Exception:
                # A failed single search degrades to no hits for that scope
                # rather than failing the whole tool call (matches the
                # per-group warn-and-continue behaviour of the upstream tool).
                return []

        results = await asyncio.gather(*(_run(call) for call in calls))
        metas: list[SearchResultMeta] = []
        for search_results in results:
            for result in search_results:
                metas.append(
                    SearchResultMeta(
                        result=result,
                        source_query=query,
                        query_type="hybrid",
                        knowledge_base_id=result.knowledge_base_id,
                        knowledge_base_type=kb_type_map.get(result.knowledge_base_id, ""),
                    )
                )
        return metas

    # ── Dedup / rerank / MMR ────────────────────────────────────────

    def _deduplicate_results(self, results: list[SearchResultMeta]) -> list[SearchResultMeta]:
        """Remove duplicate or near-duplicate chunks, keeping the best score."""
        seen: set[str] = set()
        content_sig: set[str] = set()
        unique: list[SearchResultMeta] = []
        for meta in results:
            result = meta.result
            keys = [result.id]
            if result.parent_chunk_id:
                keys.append("parent:" + result.parent_chunk_id)
            if result.knowledge_id:
                keys.append(f"kb:{result.knowledge_id}#{result.chunk_index}")
            if any(key in seen for key in keys):
                continue
            signature = build_content_signature(result.content)
            if signature:
                if signature in content_sig:
                    continue
                content_sig.add(signature)
            seen.update(keys)
            unique.append(meta)

        seen_by_id: dict[str, SearchResultMeta] = {}
        for meta in unique:
            existing = seen_by_id.get(meta.result.id)
            if existing is None or meta.result.score > existing.result.score:
                seen_by_id[meta.result.id] = meta
        return list(seen_by_id.values())

    async def _maybe_rerank(
        self,
        ctx: Context,
        queries: list[str],
        results: list[SearchResultMeta],
    ) -> list[SearchResultMeta]:
        """Apply reranking when a rerank model or LLM is configured."""
        if not results or (self._reranker is None and self._chat is None):
            return results
        rerank_query = " ".join(queries).strip() if queries else ""
        if not rerank_query:
            return results

        if self._reranker is not None:
            try:
                reranked = await self._rerank_with_model(ctx, rerank_query, results)
            except Exception:
                reranked = []
            if reranked:
                return reranked
            # Model failed or returned nothing above threshold; fall back.
            if self._chat is not None:
                return await self._rerank_with_llm(ctx, rerank_query, results)
            return results
        return await self._rerank_with_llm(ctx, rerank_query, results)

    async def _rerank_with_model(
        self,
        ctx: Context,
        query: str,
        results: list[SearchResultMeta],
    ) -> list[SearchResultMeta]:
        """Rerank with a dedicated rerank model."""
        reranker = self._reranker
        if reranker is None:
            return results
        passages = [self._get_enriched_passage(meta) for meta in results]
        response = await reranker.rerank(query, passages)
        rank_items = [
            _RankItem(index=int(item.index), relevance_score=float(item.relevance_score))
            for item in response
        ]
        return self._apply_model_rerank_scores(
            results,
            rank_items,
            self._rerank_threshold,
            self._search_targets.has_recall_threshold_override(),
        )

    async def _rerank_with_llm(
        self,
        ctx: Context,
        query: str,
        results: list[SearchResultMeta],
    ) -> list[SearchResultMeta]:
        """Rerank with an LLM scoring prompt, processed in batches."""
        chat = self._chat
        if chat is None:
            return results
        all_scores = [0.0] * len(results)
        for batch_start in range(0, len(results), _LLM_RERANK_BATCH_SIZE):
            batch_end = min(batch_start + _LLM_RERANK_BATCH_SIZE, len(results))
            batch = results[batch_start:batch_end]
            passages: list[str] = []
            for meta in batch:
                content = self._get_enriched_passage(meta)
                if len(content) > _LLM_RERANK_MAX_CONTENT_LENGTH:
                    content = content[:_LLM_RERANK_MAX_CONTENT_LENGTH] + "..."
                passages.append(content)

            prompt = _build_rerank_prompt(query, passages)
            messages = [
                Message(
                    role="system",
                    content=(
                        "You are a professional search result reranking expert "
                        "specializing in information retrieval. You evaluate how "
                        "well retrieved passages match user queries in search "
                        "scenarios. Focus on retrieval relevance: whether the "
                        "passage answers the query, provides needed information, "
                        "and matches the user's information need. Always respond "
                        "with scores only, no explanations."
                    ),
                ),
                Message(role="user", content=prompt),
            ]
            max_tokens = len(batch) * 20 + 100
            try:
                response = await chat.chat(
                    messages,
                    ChatOptions(temperature=0.1, max_tokens=max_tokens),
                )
                scores = _parse_scores_from_response(response.content, len(batch))
            except Exception:
                scores = None
            if scores is None:
                for i in range(batch_start, batch_end):
                    all_scores[i] = results[i].result.score
                continue
            for j, score in enumerate(scores):
                all_scores[batch_start + j] = score

        rank_items = [
            _RankItem(index=i, relevance_score=score) for i, score in enumerate(all_scores)
        ]
        rank_items.sort(key=lambda item: item.relevance_score, reverse=True)
        return self._apply_model_rerank_scores(
            results,
            rank_items,
            self._rerank_threshold,
            self._search_targets.has_recall_threshold_override(),
        )

    def _apply_model_rerank_scores(
        self,
        originals: list[SearchResultMeta],
        rank_items: list[_RankItem],
        threshold: float,
        preserve_top: bool,
    ) -> list[SearchResultMeta]:
        """Filter rerank items by threshold and re-score the survivors."""
        filtered = _filter_rerank_rank_items(rank_items, threshold, preserve_top)
        out: list[SearchResultMeta] = []
        for item in filtered:
            if item.index < 0 or item.index >= len(originals):
                continue
            original = originals[item.index]
            base_score = original.result.score
            new_score = self._composite_score(original, item.relevance_score, base_score)
            out.append(
                replace(
                    original,
                    result=original.result.model_copy(update={"score": new_score}),
                )
            )
        out.sort(key=lambda meta: meta.result.score, reverse=True)
        return out

    def _composite_score(
        self,
        meta: SearchResultMeta,
        model_score: float,
        base_score: float,
    ) -> float:
        """Combine the model score, the base score, and provenance priors."""
        source_weight = 1.0
        if meta.result.knowledge_source.lower() == "web_search":
            source_weight = 0.95

        position_prior = 1.0
        start_at = meta.result.start_at
        end_at = meta.result.end_at
        if start_at >= 0 and end_at > start_at:
            position_ratio = 1.0 - start_at / (end_at + 1)
            position_prior += clamp_float(position_ratio, -0.05, 0.05)

        composite = 0.6 * model_score + 0.3 * base_score + 0.1 * source_weight
        composite *= position_prior
        return clamp_float(composite, 0.0, 1.0)

    def _get_enriched_passage(self, meta: SearchResultMeta) -> str:
        """Merge a result's content with its image captions / OCR text."""
        result = meta.result
        if not result.image_info:
            return result.content
        image_texts: list[str] = []
        for img in parse_image_infos(result.image_info):
            caption = _as_str(img.get("caption"))
            if caption:
                image_texts.append(f"Image Caption: {caption}")
            ocr = _as_str(img.get("ocr_text"))
            if ocr:
                image_texts.append(f"Image Text: {ocr}")
        if not image_texts:
            return result.content
        combined = result.content
        if combined:
            combined += "\n\n"
        return combined + "\n".join(image_texts)

    def _apply_mmr(
        self,
        results: list[SearchResultMeta],
        k: int,
        lambda_value: float,
    ) -> list[SearchResultMeta]:
        """Apply maximal marginal relevance to reduce redundancy."""
        if k <= 0 or not results:
            return []
        selected: list[SearchResultMeta] = []
        selected_token_sets: list[set[str]] = []
        candidates = list(results)
        token_sets = [tokenize_simple(self._get_enriched_passage(meta)) for meta in candidates]

        while len(selected) < k and candidates:
            best_idx = 0
            best_score = -1.0
            for i, meta in enumerate(candidates):
                relevance = meta.result.score
                redundancy = 0.0
                for selected_tokens in selected_token_sets:
                    redundancy = max(redundancy, jaccard(token_sets[i], selected_tokens))
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

    async def _format_output(
        self,
        results: list[SearchResultMeta],
        kbs_to_search: list[str],
        queries: list[str],
    ) -> ToolResult:
        """Render the final results as XML plus a structured result map."""
        if not results:
            empty_data: JsonObject = {
                "knowledge_base_ids": cast("list[JsonValue]", kbs_to_search),
                "results": [],
                "count": 0,
            }
            if queries:
                empty_data["queries"] = cast("list[JsonValue]", queries)
            output = f"No relevant content found in {len(kbs_to_search)} knowledge base(s).\n\n"
            output += "=== ⚠️ CRITICAL - Next Steps ===\n"
            output += "- ❌ DO NOT use training data or general knowledge to answer\n"
            output += "- ✅ If web_search is enabled: You MUST use web_search to find information\n"
            output += "- ✅ If web_search is disabled: State 'I couldn't find relevant information in the knowledge base'\n"
            output += "- NEVER fabricate or infer answers - ONLY use retrieved content\n"
            return ToolResult(success=True, output=output, data=empty_data)

        kb_counts: dict[str, int] = {}
        for meta in results:
            kb_counts[meta.knowledge_base_id] = kb_counts.get(meta.knowledge_base_id, 0) + 1

        parts: list[str] = []
        parts.append(f'<search_results count="{len(results)}">\n')
        for query in queries:
            parts.append(f"<query>{xml_escape(query)}</query>\n")
        _write_knowledge_metadata_header(parts, results)

        formatted_results: list[JsonObject] = []
        faq_metadata_cache: dict[str, FAQChunkMetadata | None] = {}
        knowledge_chunk_map: dict[str, set[int]] = {}
        knowledge_total_map: dict[str, int] = {}
        knowledge_title_map: dict[str, str] = {}

        for index, meta in enumerate(results, start=1):
            result = meta.result
            faq_meta: FAQChunkMetadata | None = None
            if meta.knowledge_base_type == "faq":
                faq_meta = _faq_metadata_cached(
                    result.id, result.chunk_metadata, faq_metadata_cache
                )

            knowledge_chunk_map.setdefault(result.knowledge_id, set()).add(result.chunk_index)
            knowledge_title_map[result.knowledge_id] = result.knowledge_title

            if result.knowledge_id not in knowledge_total_map:
                knowledge_total_map[result.knowledge_id] = await self._count_chunks(meta)

            seen = result.id in self._seen_chunks
            self._seen_chunks.add(result.id)

            is_faq = faq_meta is not None
            if seen:
                if is_faq:
                    parts.append(
                        f'<faq rank="{index}" faq_id="{xml_escape(result.id)}" '
                        f'index="{result.chunk_index}" '
                        f'knowledge_base_id="{xml_escape(meta.knowledge_base_id)}" '
                        f'knowledge_title="{xml_escape(result.knowledge_title)}" '
                        f'score="{result.score:.3f}" '
                        f'source_query="{xml_escape(meta.source_query)}" '
                        'already_seen="true">\n'
                    )
                else:
                    parts.append(
                        f'<chunk rank="{index}" chunk_id="{xml_escape(result.id)}" '
                        f'chunk_index="{result.chunk_index}" '
                        f'knowledge_id="{xml_escape(result.knowledge_id)}" '
                        f'knowledge_base_id="{xml_escape(meta.knowledge_base_id)}" '
                        f'knowledge_title="{xml_escape(result.knowledge_title)}" '
                        f'score="{result.score:.3f}" '
                        f'source_query="{xml_escape(meta.source_query)}" '
                        'already_seen="true">\n'
                    )
                parts.append(
                    "<note>(content omitted, already returned in a previous "
                    "knowledge_search call this session)</note>\n"
                )
            else:
                if is_faq:
                    parts.append(
                        f'<faq rank="{index}" faq_id="{xml_escape(result.id)}" '
                        f'index="{result.chunk_index}" '
                        f'knowledge_base_id="{xml_escape(meta.knowledge_base_id)}" '
                        f'knowledge_title="{xml_escape(result.knowledge_title)}" '
                        f'score="{result.score:.3f}" '
                        f'source_query="{xml_escape(meta.source_query)}">\n'
                    )
                else:
                    parts.append(
                        f'<chunk rank="{index}" chunk_id="{xml_escape(result.id)}" '
                        f'chunk_index="{result.chunk_index}" '
                        f'knowledge_id="{xml_escape(result.knowledge_id)}" '
                        f'knowledge_base_id="{xml_escape(meta.knowledge_base_id)}" '
                        f'knowledge_title="{xml_escape(result.knowledge_title)}" '
                        f'score="{result.score:.3f}" '
                        f'source_query="{xml_escape(meta.source_query)}">\n'
                    )

                snippet = ""
                if faq_meta is not None:
                    snippet = faq_match_snippet_from_queries(faq_meta, queries)
                if not snippet:
                    snippet = extract_snippet_for_queries(result.content, queries)
                if snippet:
                    parts.append(f"<match_snippet>{xml_escape(snippet)}</match_snippet>\n")
                parts.append(f"<content>{result.content}</content>\n")

                _write_result_images_markdown(parts, result.image_info)

                if is_faq:
                    write_faq_fields_xml(parts, faq_meta)

            if is_faq:
                parts.append("</faq>\n")
            else:
                parts.append("</chunk>\n")

            entry: JsonObject = {
                "result_index": index,
                "content": result.content,
                "knowledge_id": result.knowledge_id,
                "knowledge_base_id": meta.knowledge_base_id,
                "knowledge_title": result.knowledge_title,
                "knowledge_metadata": result.knowledge_custom_metadata,
                "match_type": int(result.match_type),
                "source_query": meta.source_query,
                "query_type": meta.query_type,
                "knowledge_base_type": meta.knowledge_base_type,
            }
            images = _images_for_result(result.image_info)
            if images:
                entry["images"] = cast("list[JsonValue]", images)
            if faq_meta is not None:
                entry["faq_id"] = result.id
                entry["index"] = result.chunk_index
                if faq_meta.standard_question:
                    entry["faq_standard_question"] = faq_meta.standard_question
                append_similar_questions_to_chunk_data(entry, faq_meta.similar_questions)
                if faq_meta.answers:
                    entry["faq_answers"] = list(faq_meta.answers)
            else:
                entry["chunk_id"] = result.id
                entry["chunk_index"] = result.chunk_index
            formatted_results.append(entry)

        parts.append("<retrieval_statistics>\n")
        for knowledge_id, retrieved_chunks in knowledge_chunk_map.items():
            total_chunks = knowledge_total_map.get(knowledge_id, 0)
            if total_chunks <= 0:
                continue
            retrieved_count = len(retrieved_chunks)
            remaining = total_chunks - retrieved_count
            percentage = retrieved_count / total_chunks * 100
            parts.append(
                f'<document_stat knowledge_id="{xml_escape(knowledge_id)}" '
                f'title="{xml_escape(knowledge_title_map.get(knowledge_id, ""))}" '
                f'total_chunks="{total_chunks}" retrieved="{retrieved_count}" '
                f'remaining="{remaining}" coverage="{percentage:.1f}%" />\n'
            )
        parts.append("</retrieval_statistics>\n")
        parts.append("</search_results>")

        data: JsonObject = {
            "knowledge_base_ids": cast("list[JsonValue]", kbs_to_search),
            "results": cast("list[JsonValue]", formatted_results),
            "count": len(formatted_results),
            "kb_counts": cast("dict[str, JsonValue]", kb_counts),
            "display_type": "search_results",
        }
        if queries:
            data["queries"] = cast("list[JsonValue]", queries)
        return ToolResult(success=True, output="".join(parts), data=data)

    async def _count_chunks(self, meta: SearchResultMeta) -> int:
        """Count the document's live text+FAQ chunks for the statistics."""
        if self._chunk_counter is None:
            return 0
        tenant_id = self._search_targets.get_tenant_id_for_kb(meta.knowledge_base_id)
        if tenant_id == 0:
            return 0
        try:
            return await self._chunk_counter.count_chunks(
                tenant_id=tenant_id, knowledge_id=meta.result.knowledge_id
            )
        except Exception:
            return 0


def _filter_rerank_rank_items(
    rank_items: list[_RankItem],
    threshold: float,
    preserve_top: bool,
) -> list[_RankItem]:
    """Drop rerank items below the threshold, preserving a top fallback."""
    if not rank_items:
        return []
    filtered = [item for item in rank_items if item.relevance_score >= threshold]
    if not filtered:
        top = max(rank_items, key=lambda item: item.relevance_score)
        if preserve_top or top.relevance_score >= _RERANK_FALLBACK_MIN_SCORE:
            return [top]
    return filtered


def _parse_scores_from_response(response_text: str, expected_count: int) -> list[float] | None:
    """Parse ``Passage N: X.XX`` score lines from an LLM response."""
    scores: list[float] = []
    for line in response_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(":")
        score_str = parts[-1].strip() if len(parts) >= 2 else line
        score_str = _trim_nonnumeric(score_str)
        if not score_str:
            continue
        try:
            score = float(score_str)
        except ValueError:
            continue
        scores.append(clamp_float(score, 0.0, 1.0))
    if not scores:
        return None
    while len(scores) < expected_count:
        scores.append(scores[-1])
    if len(scores) > expected_count:
        scores = scores[:expected_count]
    return scores


def _trim_nonnumeric(value: str) -> str:
    """Strip any non-digit / non-dot characters from both ends of ``value``."""
    start = 0
    end = len(value)
    while start < end and not (value[start].isdigit() or value[start] == "."):
        start += 1
    while end > start and not (value[end - 1].isdigit() or value[end - 1] == "."):
        end -= 1
    return value[start:end]


def _build_rerank_prompt(query: str, passages: list[str]) -> str:
    """Build the LLM rerank prompt for one batch of passages."""
    passage_lines: list[str] = []
    for i, content in enumerate(passages, start=1):
        passage_lines.append(
            "─────────────────────────────────────────────────────────────\n"
            f"Passage {i}:\n"
            "─────────────────────────────────────────────────────────────\n"
            f"{content}\n"
        )
    return (
        "You are a search result reranking expert. Your task is to evaluate "
        "how well each retrieved passage matches the user's search query and "
        "information need.\n"
        f"\nUser Query: {query}\n"
        "\nYour task: Rerank these search results by evaluating their retrieval "
        "relevance - how well each passage answers or relates to the query.\n"
        "\nScoring Criteria (0.0 to 1.0):\n"
        "- 1.0 (0.9-1.0): Directly answers the query, contains key information "
        "needed, highly relevant\n"
        "- 0.8 (0.7-0.8): Strongly related, provides substantial relevant information\n"
        "- 0.6 (0.5-0.6): Moderately related, contains some relevant information "
        "but may be incomplete\n"
        "- 0.4 (0.3-0.4): Weakly related, minimal relevance to the query\n"
        "- 0.2 (0.1-0.2): Barely related, mostly irrelevant\n"
        "- 0.0 (0.0): Completely irrelevant, no relation to the query\n"
        "\nEvaluation Factors:\n"
        "1. Query-Answer Match: Does the passage directly address what the "
        "user is asking?\n"
        "2. Information Completeness: Does it provide sufficient information "
        "to answer the query?\n"
        "3. Semantic Relevance: Does the content semantically relate to the "
        "query intent?\n"
        "4. Key Term Coverage: Does it cover important terms/concepts from "
        "the query?\n"
        "5. Information Accuracy: Is the information accurate and trustworthy?\n"
        "\nRetrieved Passages:\n"
        f"{chr(10).join(passage_lines)}\n"
        f"\nIMPORTANT: Return exactly {len(passages)} scores, one per line, in "
        "this exact format:\n"
        "Passage 1: X.XX\n"
        "Passage 2: X.XX\n"
        "Passage 3: X.XX\n"
        "...\n"
        f"Passage {len(passages)}: X.XX\n"
        "\nOutput only the scores, no explanations or additional text."
    )


def _faq_metadata_cached(
    chunk_id: str,
    chunk_metadata: JsonObject | None,
    cache: dict[str, FAQChunkMetadata | None],
) -> FAQChunkMetadata | None:
    """Return a chunk's parsed FAQ metadata, cached per chunk id."""
    if chunk_id in cache:
        return cache[chunk_id]
    meta = faq_metadata_from_json(chunk_metadata)
    cache[chunk_id] = meta
    return meta


def _write_knowledge_metadata_header(
    parts: list[str],
    results: list[SearchResultMeta],
) -> None:
    """Emit the document-scoped metadata block once per document."""
    seen: set[str] = set()
    documents: list[str] = []
    for meta in results:
        result = meta.result
        if not result.knowledge_id or not result.knowledge_custom_metadata:
            continue
        if result.knowledge_id in seen:
            continue
        seen.add(result.knowledge_id)
        documents.append(
            f'<document knowledge_id="{xml_escape(result.knowledge_id)}" '
            f'knowledge_base_id="{xml_escape(meta.knowledge_base_id)}" '
            f'title="{xml_escape(result.knowledge_title)}">\n'
            f"<metadata>{xml_escape(result.knowledge_custom_metadata)}</metadata>\n"
            "</document>\n"
        )
    if not documents:
        return
    parts.append("<documents>\n")
    parts.extend(documents)
    parts.append("</documents>\n")


def _write_result_images_markdown(parts: list[str], image_info: str) -> None:
    """Append the answer-ready image markdown lines for a result."""
    for img in parse_image_infos(image_info):
        url = _as_str(img.get("url"))
        if not url:
            continue
        markdown = build_image_info_markdown(url, img)
        if markdown:
            parts.append(markdown + "\n")


def _images_for_result(image_info: str) -> list[JsonObject]:
    """Project a result's image info onto the structured payload."""
    images: list[JsonObject] = []
    for img in parse_image_infos(image_info):
        entry: JsonObject = {}
        url = _as_str(img.get("url"))
        if url:
            entry["url"] = url
        caption = _as_str(img.get("caption"))
        if caption:
            entry["caption"] = caption
        ocr = _as_str(img.get("ocr_text"))
        if ocr:
            entry["ocr_text"] = ocr
        if entry:
            images.append(entry)
    return images


__all__ = [
    "DEFAULT_KEYWORD_THRESHOLD",
    "DEFAULT_RERANK_THRESHOLD",
    "DEFAULT_TOP_K",
    "DEFAULT_VECTOR_THRESHOLD",
    "KNOWLEDGE_SEARCH_TOOL_DESCRIPTION",
    "KNOWLEDGE_SEARCH_TOOL_NAME",
    "KNOWLEDGE_SEARCH_TOOL_SCHEMA",
    "ChunkCounter",
    "HybridSearchRunner",
    "KbLoader",
    "KnowledgeSearchInput",
    "KnowledgeSearchTool",
    "PagedChunkStoreChunkCounter",
    "SearchCall",
    "SearchResultMeta",
    "SearchRunner",
    "build_knowledge_search_definition",
]
