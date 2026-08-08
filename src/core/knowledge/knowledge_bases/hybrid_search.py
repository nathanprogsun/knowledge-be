"""Knowledge-base hybrid search orchestrator.

``hybrid_search`` runs vector + keyword retrieval over one or more
knowledge bases that share a single embedding model:

1. loads the KBs in scope and picks the primary one;
2. computes the query embedding once (a pre-computed embedding passes
   through, and the injectable query-rewrite seam may rewrite the text);
3. partitions the KBs into store groups by (vector store, owning tenant),
   resolves each group's composite engine through the KB-to-engine
   resolver, and builds the per-group retrieval params;
4. fans retrieval out across the groups with bounded concurrency and a
   per-group timeout, normalizing cross-engine vector scores into [0, 1];
5. fuses vector + keyword hits (RRF) or deduplicates a single retriever;
6. applies FAQ post-processing (iterative retrieval / negative-question
   filtering);
7. hydrates chunk + knowledge rows and assembles the search results.

Everything the stage needs — KB loading, engine resolution, embedding,
query rewriting, FAQ metadata, chunk/knowledge hydration — arrives as an
injectable dependency so the layer stays free of direct storage or LLM
access. A ``None`` chunk/knowledge loader skips hydration (engine-only
callers still get assembled hits).
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from src.ai.embedding import Context, Embedder
from src.ai.retrieval.base import RetrieveEngineRegistry
from src.ai.retrieval.composite import CompositeRetrieveEngine
from src.ai.retrieval.kb_engine_resolver import create_retrieve_engine_for_kb
from src.ai.retrieval.normalizer import EngineAwareNormalizer
from src.ai.retrieval.ownership import TenantStoreOwnership
from src.ai.retrieval.registry import VectorStoreUnavailableError
from src.ai.retrieval.types import (
    IndexWithScore,
    MatchType,
    RetrieveParams,
    RetrieveResult,
    RetrieverType,
)
from src.common.exception import NotFoundError, ValidationError
from src.common.json import JsonObject, JsonValue
from src.core.knowledge.knowledge_bases.search_faq import (
    FaqMetadataLoader,
    apply_faq_post_processing,
)
from src.core.knowledge.knowledge_bases.search_keyword import build_keyword_params
from src.core.knowledge.knowledge_bases.search_mixed import (
    classify_retrieval_results,
    fuse_or_deduplicate,
)
from src.core.knowledge.knowledge_bases.search_query import (
    QueryRewriter,
    prepare_query,
)
from src.core.knowledge.knowledge_bases.search_vector import build_vector_params
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import (
    KNOWLEDGE_BASE_TYPE_FAQ,
    KnowledgeBaseInfo,
)
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.chunk import Chunk
from src.db.models.knowledge import Document

#: Per-group retrieval timeout (seconds) for multi-store fan-out.
_DEFAULT_MULTI_STORE_RETRIEVE_TIMEOUT_SEC = 30.0
#: Bounds concurrent per-request fan-out across store groups.
_DEFAULT_MULTI_STORE_FANOUT_LIMIT = 4
#: Chunk types that are eligible for search results.
_SEARCHABLE_CHUNK_TYPES: frozenset[str] = frozenset(
    {
        "text",
        "summary",
        "table_column",
        "table_summary",
        "faq",
        "image_ocr",
        "image_caption",
    }
)
#: Index statuses that are not yet safe to surface as hits.
_NON_SEARCHABLE_INDEX_STATUSES: frozenset[str] = frozenset({"processing", "failed"})


def _positive_int(value: JsonValue | None, default: int) -> int:
    """Return ``value`` when it is a positive integer, else ``default``."""
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return default


def _positive_float(value: JsonValue | None, default: float) -> float:
    """Return ``value`` when it is a positive number, else ``default``."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return default


# ── Parameters ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class HybridSearchParams:
    """Search scope for one hybrid-search call.

    ``query_text`` is required unless ``query_embedding`` is supplied and
    vector matching stays enabled. ``knowledge_base_ids`` overrides the
    single primary ``kb_id`` when a call spans several KBs that share one
    embedding model. ``only_recommended`` is carried for contract
    fidelity; the retrieval path does not consume it.
    """

    query_text: str = ""
    query_embedding: tuple[float, ...] = ()
    vector_threshold: float = 0.0
    keyword_threshold: float = 0.0
    match_count: int = 0
    disable_keywords_match: bool = False
    disable_vector_match: bool = False
    knowledge_ids: tuple[str, ...] = ()
    tag_ids: tuple[str, ...] = ()
    scope_tag_ids: tuple[str, ...] = ()
    only_recommended: bool = False
    knowledge_base_ids: tuple[str, ...] = ()
    skip_context_enrichment: bool = False
    exclude_chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Tenant retrieval configuration with effective-value defaults."""

    embedding_top_k: int = 50
    vector_threshold: float = 0.15
    keyword_threshold: float = 0.3
    rerank_top_k: int = 10
    rerank_threshold: float = 0.2
    rrf_k: int = 60
    rrf_vector_weight: float = 0.7
    rrf_keyword_weight: float = 0.3

    @classmethod
    def from_json(cls, raw: JsonObject | None) -> RetrievalConfig:
        """Build from a tenant retrieval-config blob; zero values keep defaults."""
        if not raw:
            return cls()
        return cls(
            embedding_top_k=_positive_int(raw.get("embedding_top_k"), 50),
            vector_threshold=_positive_float(raw.get("vector_threshold"), 0.15),
            keyword_threshold=_positive_float(raw.get("keyword_threshold"), 0.3),
            rerank_top_k=_positive_int(raw.get("rerank_top_k"), 10),
            rerank_threshold=_positive_float(raw.get("rerank_threshold"), 0.2),
            rrf_k=_positive_int(raw.get("rrf_k"), 60),
            rrf_vector_weight=_positive_float(raw.get("rrf_vector_weight"), 0.7),
            rrf_keyword_weight=_positive_float(raw.get("rrf_keyword_weight"), 0.3),
        )

    def effective_rrf_weights(self) -> tuple[float, float]:
        """Return the (vector, keyword) RRF weights with defaults applied."""
        if self.rrf_vector_weight <= 0 and self.rrf_keyword_weight <= 0:
            return 0.7, 0.3
        vector = self.rrf_vector_weight if self.rrf_vector_weight > 0 else 0.7
        keyword = self.rrf_keyword_weight if self.rrf_keyword_weight > 0 else 0.3
        return vector, keyword


# ── Dependency seams ──────────────────────────────────────────────────


@runtime_checkable
class KnowledgeBaseLoader(Protocol):
    """Loads knowledge bases by id (authorization is the caller's job)."""

    async def load_by_ids(self, ids: list[str]) -> list[KnowledgeBaseInfo]: ...


class KBServiceKnowledgeBaseLoader:
    """Adapter over ``KBService.get_knowledge_bases_by_ids``."""

    def __init__(self, service: KBService) -> None:
        self._service = service

    async def load_by_ids(self, ids: list[str]) -> list[KnowledgeBaseInfo]:
        return await self._service.get_knowledge_bases_by_ids(ids=ids)


@runtime_checkable
class ChunkLoader(Protocol):
    """Loads live chunk rows for a tenant."""

    async def load_by_ids(
        self, ctx: Context, tenant_id: int, chunk_ids: list[str]
    ) -> list[Chunk]: ...


class ChunkRepositoryLoader:
    """Adapter over ``ChunkRepository.list_by_ids``."""

    def __init__(self, repo: ChunkRepository) -> None:
        self._repo = repo

    async def load_by_ids(self, _ctx: Context, tenant_id: int, chunk_ids: list[str]) -> list[Chunk]:
        return await self._repo.list_by_ids(tenant_id, chunk_ids)


@runtime_checkable
class KnowledgeLoader(Protocol):
    """Loads live document rows for a tenant."""

    async def load_by_ids(
        self, ctx: Context, tenant_id: int, knowledge_ids: list[str]
    ) -> list[Document]: ...


class KnowledgeRepositoryLoader:
    """Adapter over ``KnowledgeRepository.get_batch``."""

    def __init__(self, repo: KnowledgeRepository) -> None:
        self._repo = repo

    async def load_by_ids(
        self, _ctx: Context, tenant_id: int, knowledge_ids: list[str]
    ) -> list[Document]:
        return await self._repo.get_batch(tenant_id, knowledge_ids)


@dataclass(frozen=True, slots=True)
class SearchDependencies:
    """Bundled dependencies for one hybrid-search call."""

    kb_loader: KnowledgeBaseLoader
    engine_registry: RetrieveEngineRegistry
    ownership: TenantStoreOwnership
    embedder: Embedder | None = None
    query_rewriter: QueryRewriter | None = None
    faq_loader: FaqMetadataLoader | None = None
    chunk_loader: ChunkLoader | None = None
    knowledge_loader: KnowledgeLoader | None = None
    retrieval_config: RetrievalConfig | None = None


#: One fan-out unit: KBs sharing a (store id, owning tenant) pair.
@dataclass(frozen=True, slots=True)
class StoreGroup:
    store_id: str
    owner_tenant_id: int
    kb_ids: tuple[str, ...]
    engine: CompositeRetrieveEngine
    base_params: tuple[RetrieveParams, ...] = ()
    top_k: int = 0


# ── Result model ──────────────────────────────────────────────────────


class SearchResult(BaseModel):
    """One hydrated search hit (upstream search-result wire shape)."""

    model_config = ConfigDict(frozen=True)

    id: str = ""
    content: str = ""
    knowledge_id: str = ""
    chunk_index: int = 0
    knowledge_title: str = ""
    start_at: int = 0
    end_at: int = 0
    seq: int = 0
    score: float = 0.0
    match_type: MatchType = MatchType.EMBEDDING
    sub_chunk_id: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    chunk_type: str = ""
    parent_chunk_id: str = ""
    image_info: str = ""
    knowledge_filename: str = ""
    knowledge_source: str = ""
    knowledge_channel: str = ""
    chunk_metadata: JsonObject | None = None
    matched_content: str = ""
    knowledge_description: str = ""
    knowledge_custom_metadata: str = ""
    knowledge_base_id: str = ""


# ── Orchestrator ──────────────────────────────────────────────────────


async def hybrid_search(
    ctx: Context,
    *,
    kb_id: str,
    params: HybridSearchParams,
    deps: SearchDependencies,
) -> list[SearchResult] | None:
    """Run a hybrid (vector + keyword) search across one or more KBs.

    ``kb_id`` is the primary KB: it selects the embedding model and FAQ
    behaviour. ``params.knowledge_base_ids`` overrides the search scope
    when set (every KB must share one embedding model). Returns ``None``
    when no KB is retrievable or no result survives, matching the
    upstream empty contract.
    """
    if not params.query_text.strip() and not params.query_embedding:
        raise ValidationError(
            code="knowledge_base.search_query_required",
            message="search query is required",
        )

    search_kb_ids = list(params.knowledge_base_ids) or [kb_id]
    kbs = await deps.kb_loader.load_by_ids(search_kb_ids)
    if not kbs:
        raise NotFoundError(code="knowledge_base.not_found", message="knowledge base not found")
    kb = _pick_primary(kbs, kb_id)
    if kb is None:
        raise NotFoundError(code="knowledge_base.not_found", message="knowledge base not found")

    config = deps.retrieval_config or RetrievalConfig()
    match_count = params.match_count if params.match_count > 0 else config.embedding_top_k
    over_retrieve = min(max(match_count * 5, 50) * len(search_kb_ids), 500)

    query = await prepare_query(
        ctx,
        query_text=params.query_text,
        needs_embedding=(
            not params.query_embedding
            and not params.disable_vector_match
            and _is_vector_enabled(kb)
            and kb.embedding_model_id != ""
        ),
        embedder=deps.embedder,
        rewriter=deps.query_rewriter,
    )
    effective_embedding = params.query_embedding or query.embedding
    effective_query_text = query.rewritten

    groups = await _resolve_store_groups(
        ctx,
        kb=kb,
        kbs=kbs,
        params=params,
        query_text=effective_query_text,
        embedding=effective_embedding,
        over_retrieve=over_retrieve,
        config=config,
        deps=deps,
    )
    if not groups or all(len(group.base_params) == 0 for group in groups):
        return None

    raw_results = await _retrieve_from_stores(ctx, groups)
    vector_results, keyword_results = classify_retrieval_results(raw_results)
    vector_results, keyword_results = _drop_excluded(
        vector_results, keyword_results, params.exclude_chunk_ids
    )
    if not vector_results and not keyword_results:
        return None

    vector_weight, keyword_weight = config.effective_rrf_weights()
    chunks = fuse_or_deduplicate(
        vector_results,
        keyword_results,
        rrf_k=config.rrf_k,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
    )

    chunks = await apply_faq_post_processing(
        ctx,
        kb_type=kb.type,
        chunks=chunks,
        vector_result_count=len(vector_results),
        requested_count=match_count,
        over_retrieve_count=over_retrieve,
        query_text=effective_query_text,
        retrieve=lambda top_k: _retrieve_and_flatten(ctx, groups, top_k),
        faq_loader=deps.faq_loader,
    )
    if len(chunks) > match_count:
        chunks = chunks[:match_count]

    return await _process_search_results(
        ctx,
        chunks=chunks,
        params=params,
        tenant_id=kb.tenant_id,
        deps=deps,
    )


# ── Helpers ───────────────────────────────────────────────────────────


def _pick_primary(kbs: list[KnowledgeBaseInfo], kb_id: str) -> KnowledgeBaseInfo | None:
    """Return the KB whose id matches ``kb_id``, or ``None``."""
    for kb in kbs:
        if kb.id == kb_id:
            return kb
    return None


def _is_vector_enabled(kb: KnowledgeBaseInfo) -> bool:
    return _indexing_flag(kb, "vector_enabled")


def _is_keyword_enabled(kb: KnowledgeBaseInfo) -> bool:
    return _indexing_flag(kb, "keyword_enabled")


def _indexing_flag(kb: KnowledgeBaseInfo, key: str) -> bool:
    """Read an indexing-strategy flag; a missing strategy defaults to on."""
    strategy = kb.indexing_strategy
    if strategy is None:
        return True
    value = strategy.get(key)
    return value if isinstance(value, bool) else True


async def _resolve_store_groups(
    ctx: Context,
    *,
    kb: KnowledgeBaseInfo,
    kbs: list[KnowledgeBaseInfo],
    params: HybridSearchParams,
    query_text: str,
    embedding: tuple[float, ...],
    over_retrieve: int,
    config: RetrievalConfig,
    deps: SearchDependencies,
) -> list[StoreGroup]:
    """Partition KBs by (store id, owning tenant) and resolve each engine.

    KBs sharing a store and tenant share one composite engine and one
    retrieval-param set; the primary KB supplies the embedding model and
    FAQ type. Resolution failures surface as the typed resolver sentinels
    (which never echo store ids).
    """
    buckets: dict[tuple[str, int], list[KnowledgeBaseInfo]] = {}
    for candidate in kbs:
        key = (candidate.vector_store_id or "", candidate.tenant_id)
        buckets.setdefault(key, []).append(candidate)

    groups: list[StoreGroup] = []
    for (store_id, owner_tenant_id), group_kbs in buckets.items():
        engine = await create_retrieve_engine_for_kb(
            ctx,
            deps.engine_registry,
            deps.ownership,
            owner_tenant_id,
            store_id or None,
        )
        base_params = _build_retrieval_params(
            engine=engine,
            group_kbs=group_kbs,
            params=params,
            query_text=query_text,
            embedding=embedding,
            match_count=over_retrieve,
            config=config,
        )
        groups.append(
            StoreGroup(
                store_id=store_id,
                owner_tenant_id=owner_tenant_id,
                kb_ids=tuple(candidate.id for candidate in group_kbs),
                engine=engine,
                base_params=tuple(base_params),
                top_k=over_retrieve,
            )
        )
    return groups


def _build_retrieval_params(
    *,
    engine: CompositeRetrieveEngine,
    group_kbs: list[KnowledgeBaseInfo],
    params: HybridSearchParams,
    query_text: str,
    embedding: tuple[float, ...],
    match_count: int,
    config: RetrievalConfig,
) -> list[RetrieveParams]:
    """Build one retrieval-param set per index for a store group.

    Document KBs hit the default vector index plus the keyword index;
    FAQ KBs hit only the FAQ vector index. Routing is a per-KB property
    decided from each KB's type and indexing flags — never from the
    primary KB alone.
    """
    faq_vector_ids: list[str] = []
    doc_vector_ids: list[str] = []
    doc_keyword_ids: list[str] = []
    for candidate in group_kbs:
        if _is_vector_enabled(candidate) and candidate.embedding_model_id:
            if candidate.type == KNOWLEDGE_BASE_TYPE_FAQ:
                faq_vector_ids.append(candidate.id)
            else:
                doc_vector_ids.append(candidate.id)
        if _is_keyword_enabled(candidate) and candidate.type != KNOWLEDGE_BASE_TYPE_FAQ:
            doc_keyword_ids.append(candidate.id)

    vector_threshold = (
        params.vector_threshold if params.vector_threshold > 0 else config.vector_threshold
    )
    keyword_threshold = (
        params.keyword_threshold if params.keyword_threshold > 0 else config.keyword_threshold
    )

    retrieve_params: list[RetrieveParams] = []
    if (
        engine.support_retriever(RetrieverType.VECTOR)
        and not params.disable_vector_match
        and (faq_vector_ids or doc_vector_ids)
    ):
        retrieve_params.extend(
            build_vector_params(
                query=query_text,
                embedding=embedding,
                doc_kb_ids=doc_vector_ids,
                faq_kb_ids=faq_vector_ids,
                top_k=match_count,
                threshold=vector_threshold,
                knowledge_ids=params.knowledge_ids,
                tag_ids=params.tag_ids,
            )
        )
    if (
        engine.support_retriever(RetrieverType.KEYWORDS)
        and not params.disable_keywords_match
        and doc_keyword_ids
    ):
        retrieve_params.append(
            build_keyword_params(
                query=query_text,
                kb_ids=doc_keyword_ids,
                top_k=match_count,
                threshold=keyword_threshold,
                knowledge_ids=params.knowledge_ids,
                tag_ids=params.tag_ids,
            )
        )
    return retrieve_params


async def _retrieve_from_stores(
    ctx: Context,
    groups: list[StoreGroup],
    top_k_override: int | None = None,
) -> list[RetrieveResult]:
    """Retrieve every store group with bounded concurrency and a timeout.

    A single group short-circuits to a direct call with zero fan-out
    overhead. Multi-group results are normalized when they span more than
    one engine type (cross-engine vector scores become comparable);
    per-group failures collapse into the typed unavailable sentinel while
    parent cancellation propagates.
    """
    if not groups:
        return []
    if len(groups) == 1:
        return await groups[0].engine.retrieve(ctx, _params_with_top_k(groups[0], top_k_override))

    timeout = _multi_store_retrieve_timeout()
    semaphore = asyncio.Semaphore(_DEFAULT_MULTI_STORE_FANOUT_LIMIT)

    async def _run(group: StoreGroup) -> list[RetrieveResult]:
        params = _params_with_top_k(group, top_k_override)
        try:
            async with semaphore:
                return await asyncio.wait_for(group.engine.retrieve(ctx, params), timeout=timeout)
        except (asyncio.CancelledError, VectorStoreUnavailableError):
            raise
        except Exception as exc:
            raise VectorStoreUnavailableError(
                "vector retrieval failed for one or more bound stores"
            ) from exc

    collected = await asyncio.gather(*(_run(group) for group in groups), return_exceptions=True)
    for item in collected:
        if isinstance(item, BaseException):
            raise item
    flat: list[RetrieveResult] = []
    for item in collected:
        if not isinstance(item, BaseException):
            flat.extend(item)
    return _normalize_scores(ctx, flat)


async def _retrieve_and_flatten(
    ctx: Context, groups: list[StoreGroup], top_k: int
) -> list[IndexWithScore]:
    """Run a fan-out with ``top_k`` and merge every raw hit."""
    results = await _retrieve_from_stores(ctx, groups, top_k)
    flat: list[IndexWithScore] = []
    for result in results:
        flat.extend(result.results)
    return flat


def _params_with_top_k(
    group: StoreGroup, top_k_override: int | None = None
) -> list[RetrieveParams]:
    """Build a fresh param list with only the top-k overridden."""
    top_k = group.top_k if top_k_override is None else top_k_override
    return [params.model_copy(update={"top_k": top_k}) for params in group.base_params]


def _normalize_scores(ctx: Context, results: list[RetrieveResult]) -> list[RetrieveResult]:
    """Rescale vector scores into [0, 1] when results span engine types.

    Same-engine results keep their native scale — raw scores from the
    same engine are directly comparable. Returns new result objects; the
    input is never mutated.
    """
    if not _has_mixed_engine_types(results):
        return results
    normalizer = EngineAwareNormalizer()
    rescaled_results: list[RetrieveResult] = []
    for result in results:
        hits = [
            hit.model_copy(
                update={
                    "score": normalizer.normalize(
                        ctx, hit.score, result.retriever_type, result.retriever_engine_type
                    )
                }
            )
            for hit in result.results
        ]
        rescaled_results.append(result.model_copy(update={"results": hits}))
    return rescaled_results


def _has_mixed_engine_types(results: list[RetrieveResult]) -> bool:
    if len(results) < 2:
        return False
    first = results[0].retriever_engine_type
    return any(result.retriever_engine_type != first for result in results[1:])


def _drop_excluded(
    vector_results: list[IndexWithScore],
    keyword_results: list[IndexWithScore],
    excluded_chunk_ids: tuple[str, ...],
) -> tuple[list[IndexWithScore], list[IndexWithScore]]:
    """Defensively drop excluded chunk ids before fusion."""
    if not excluded_chunk_ids:
        return vector_results, keyword_results
    excluded = frozenset(excluded_chunk_ids)
    return (
        [hit for hit in vector_results if hit.chunk_id not in excluded],
        [hit for hit in keyword_results if hit.chunk_id not in excluded],
    )


def _multi_store_retrieve_timeout() -> float:
    """Return the per-group retrieve timeout; env override falls back to 30s."""
    raw = os.environ.get("MULTI_STORE_RETRIEVE_TIMEOUT_SEC")
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            return float(value)
    return _DEFAULT_MULTI_STORE_RETRIEVE_TIMEOUT_SEC


# ── Result hydration / assembly ───────────────────────────────────────


@dataclass(slots=True)
class _ChunkIndex:
    """Pre-computed lookup structures for assembling search results."""

    knowledge_ids: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    match_types: dict[str, MatchType] = field(default_factory=dict)
    matched_contents: dict[str, str] = field(default_factory=dict)
    processed_ids: set[str] = field(default_factory=set)


async def _process_search_results(
    ctx: Context,
    *,
    chunks: list[IndexWithScore],
    params: HybridSearchParams,
    tenant_id: int,
    deps: SearchDependencies,
) -> list[SearchResult] | None:
    """Hydrate chunk + knowledge rows and assemble the final results.

    When the chunk/knowledge loaders are absent, results are assembled
    directly from the retrieval hits without any store hydration.
    """
    if not chunks:
        return None
    if deps.chunk_loader is None or deps.knowledge_loader is None:
        return [_search_result_from_hit(hit) for hit in chunks]

    index = _build_chunk_index(chunks)
    knowledge_map = {
        doc.id: doc
        for doc in await deps.knowledge_loader.load_by_ids(ctx, tenant_id, index.knowledge_ids)
    }
    chunk_rows = await deps.chunk_loader.load_by_ids(ctx, tenant_id, index.chunk_ids)
    chunk_map = {chunk.id: chunk for chunk in chunk_rows}

    if not params.skip_context_enrichment:
        extra_ids = _collect_enrichment_ids(chunk_rows, index)
        if extra_ids:
            extra_rows = await deps.chunk_loader.load_by_ids(ctx, tenant_id, extra_ids)
            for chunk in extra_rows:
                chunk_map[chunk.id] = chunk

    return _assemble_search_results(
        chunks=chunks,
        chunk_map=chunk_map,
        knowledge_map=knowledge_map,
        index=index,
        skip_enrichment=params.skip_context_enrichment,
    )


def _build_chunk_index(chunks: list[IndexWithScore]) -> _ChunkIndex:
    """Collect knowledge/chunk ids and build score/match-type maps."""
    index = _ChunkIndex()
    for hit in chunks:
        if hit.knowledge_id not in index.knowledge_ids:
            index.knowledge_ids.append(hit.knowledge_id)
        index.chunk_ids.append(hit.chunk_id)
        index.scores[hit.chunk_id] = hit.score
        index.match_types[hit.chunk_id] = hit.match_type
        index.matched_contents[hit.chunk_id] = hit.content
    return index


def _collect_enrichment_ids(chunk_rows: list[Chunk], index: _ChunkIndex) -> list[str]:
    """Gather parent, related, and nearby chunk ids for result enrichment."""
    for chunk in chunk_rows:
        index.processed_ids.add(chunk.id)

    extra_ids: list[str] = []
    for chunk in chunk_rows:
        if chunk.parent_chunk_id and chunk.parent_chunk_id not in index.processed_ids:
            extra_ids.append(chunk.parent_chunk_id)
            index.processed_ids.add(chunk.parent_chunk_id)
            index.scores[chunk.parent_chunk_id] = index.scores[chunk.id]
            index.match_types[chunk.parent_chunk_id] = MatchType.PARENT_CHUNK
        for related_id in _relation_ids(chunk):
            if related_id not in index.processed_ids:
                extra_ids.append(related_id)
                index.processed_ids.add(related_id)
                index.match_types[related_id] = MatchType.RELATION_CHUNK
        if chunk.chunk_type == "text":
            if chunk.next_chunk_id and chunk.next_chunk_id not in index.processed_ids:
                extra_ids.append(chunk.next_chunk_id)
                index.processed_ids.add(chunk.next_chunk_id)
                index.match_types[chunk.next_chunk_id] = MatchType.NEAR_BY_CHUNK
            if chunk.pre_chunk_id and chunk.pre_chunk_id not in index.processed_ids:
                extra_ids.append(chunk.pre_chunk_id)
                index.processed_ids.add(chunk.pre_chunk_id)
                index.match_types[chunk.pre_chunk_id] = MatchType.NEAR_BY_CHUNK
    return extra_ids


def _relation_ids(chunk: Chunk) -> list[str]:
    """Decode the chunk's related-chunk id list leniently."""
    raw = chunk.relation_chunks
    if raw is None:
        return []
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, str)]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return [item for item in parsed if isinstance(item, str)]
    return []


def _assemble_search_results(
    *,
    chunks: list[IndexWithScore],
    chunk_map: Mapping[str, Chunk],
    knowledge_map: Mapping[str, Document],
    index: _ChunkIndex,
    skip_enrichment: bool,
) -> list[SearchResult]:
    """Assemble primary results in input order, then enrichment results."""
    results: list[SearchResult] = []
    added: set[str] = set()

    for hit in chunks:
        chunk = chunk_map.get(hit.chunk_id)
        if (
            chunk is None
            or not _is_searchable_chunk(chunk)
            or chunk.id in added
            or chunk.knowledge_id not in knowledge_map
        ):
            continue
        results.append(
            _build_search_result(
                chunk=chunk,
                knowledge=knowledge_map[chunk.knowledge_id],
                score=index.scores.get(chunk.id, hit.score),
                match_type=index.match_types.get(chunk.id, hit.match_type),
                matched_content=index.matched_contents.get(chunk.id, hit.content),
            )
        )
        added.add(chunk.id)

    if not skip_enrichment:
        for chunk_id, chunk in chunk_map.items():
            if chunk_id in added or not _is_searchable_chunk(chunk):
                continue
            knowledge = knowledge_map.get(chunk.knowledge_id)
            if knowledge is None:
                continue
            match_type = index.match_types.get(chunk_id)
            if match_type is None:
                continue
            score = index.scores.get(chunk_id, 0.0)
            if score <= 0:
                score = 0.0
            results.append(
                _build_search_result(
                    chunk=chunk,
                    knowledge=knowledge,
                    score=score,
                    match_type=match_type,
                    matched_content=index.matched_contents.get(chunk_id, ""),
                )
            )
    return results


def _build_search_result(
    *,
    chunk: Chunk,
    knowledge: Document,
    score: float,
    match_type: MatchType,
    matched_content: str,
) -> SearchResult:
    """Project one chunk + knowledge pair onto the search-result shape."""
    return SearchResult(
        id=chunk.id,
        content=chunk.content,
        knowledge_id=chunk.knowledge_id,
        chunk_index=chunk.chunk_index,
        knowledge_title=knowledge.title,
        start_at=chunk.start_at,
        end_at=chunk.end_at,
        seq=chunk.chunk_index,
        score=score,
        match_type=match_type,
        metadata=_stringify_metadata(knowledge.metadata),
        chunk_type=chunk.chunk_type,
        parent_chunk_id=chunk.parent_chunk_id or "",
        image_info=chunk.image_info or "",
        knowledge_filename=knowledge.file_name or "",
        knowledge_source=knowledge.source,
        knowledge_channel=knowledge.channel,
        chunk_metadata=chunk.metadata,
        matched_content=matched_content,
        knowledge_description=knowledge.description or "",
        knowledge_custom_metadata=_custom_metadata_text(knowledge.custom_metadata),
        knowledge_base_id=knowledge.knowledge_base_id,
    )


def _search_result_from_hit(hit: IndexWithScore) -> SearchResult:
    """Assemble a result from a retrieval hit without store hydration."""
    return SearchResult(
        id=hit.chunk_id,
        content=hit.content,
        knowledge_id=hit.knowledge_id,
        score=hit.score,
        match_type=hit.match_type,
        matched_content=hit.content,
    )


def _is_searchable_chunk(chunk: Chunk) -> bool:
    """Report whether a chunk may be surfaced as a search hit."""
    if not chunk.is_enabled:
        return False
    if chunk.index_status in _NON_SEARCHABLE_INDEX_STATUSES:
        return False
    return chunk.chunk_type in _SEARCHABLE_CHUNK_TYPES


def _stringify_metadata(raw: JsonObject | None) -> dict[str, str]:
    """Narrow a JSON metadata object to a string map, stringifying values."""
    if not raw:
        return {}
    return {key: "" if value is None else str(value) for key, value in raw.items()}


def _custom_metadata_text(raw: JsonObject | None) -> str:
    """Serialize custom metadata to a stable ``key: value`` text block."""
    if not raw:
        return ""
    lines: list[str] = []
    for key in sorted(raw):
        value = raw[key]
        if value is None:
            continue
        key_clean = key.strip()
        text = str(value).strip()
        if key_clean and text:
            lines.append(f"{key_clean}: {text}")
    return "\n".join(lines)


__all__ = [
    "ChunkLoader",
    "ChunkRepositoryLoader",
    "HybridSearchParams",
    "KBServiceKnowledgeBaseLoader",
    "KnowledgeBaseLoader",
    "KnowledgeLoader",
    "KnowledgeRepositoryLoader",
    "RetrievalConfig",
    "SearchDependencies",
    "SearchResult",
    "StoreGroup",
    "hybrid_search",
]
