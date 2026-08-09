"""Chat-pipeline search step (upstream ``PluginSearch``).

``SearchStep`` handles the ``CHUNK_SEARCH`` event. It runs knowledge-base
retrieval across the pre-computed search targets and — when enabled — a web
search concurrently; if the combined recall is still below the requested
top-k, it re-runs retrieval over locally expanded query variants. The
results land on the shared ``PipelineContext.search_result``.

All external work — hybrid retrieval, query embedding, knowledge-base
loading, web search, and tenant web-search configuration — flows through
injected seams declared as structural protocols, so the step is testable
without a vector store, model API, or network. The concrete adapters bind
the step to the retrieval and web-search services.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Protocol, cast, runtime_checkable

from src.ai.embedding.base import Context as RetrievalContext
from src.ai.retrieval.types import MatchType
from src.common.json import JsonObject, JsonValue
from src.core.chat.pipeline.common import pipeline_error, pipeline_info, pipeline_warn
from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.engine import ERR_SEARCH_NOTHING, Next, PluginError
from src.core.chat.pipeline.steps.query_expansion import expand_queries
from src.core.chat.pipeline.types import (
    Context,
    EventType,
    SearchResult,
    SearchTarget,
    SearchTargetType,
)
from src.core.infra.web_search.search_service import WebSearchSearchService
from src.core.knowledge.knowledge_bases.hybrid_search import (
    HybridSearchParams,
    SearchDependencies,
    hybrid_search,
)
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.core.tenants.service import TenantService

#: Default maximum web-search results when no config is stored (upstream
#: ``DefaultWebSearchMaxResults``).
DEFAULT_WEB_SEARCH_MAX_RESULTS = 10
#: Default web-search compression method (upstream
#: ``DefaultWebSearchCompressionMethod``).
DEFAULT_WEB_SEARCH_COMPRESSION_METHOD = "none"
#: Number of rows logged when sampling search-result scores.
_MAX_SCORE_SAMPLE_ROWS = 8
#: Score assigned to a converted web-search hit.
_WEB_HIT_SCORE = 0.6


# ── Retrieval-call carrier ─────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SearchCall:
    """One hybrid-search invocation issued by the search step."""

    query_text: str
    kb_id: str
    knowledge_base_ids: tuple[str, ...] = ()
    query_embedding: tuple[float, ...] = ()
    knowledge_ids: tuple[str, ...] = ()
    tag_ids: tuple[str, ...] = ()
    scope_tag_ids: tuple[str, ...] = ()
    top_k: int = 0
    vector_threshold: float = 0.0
    keyword_threshold: float = 0.0
    disable_vector_match: bool = False
    disable_keywords_match: bool = False
    skip_context_enrichment: bool = True


# ── Dependency seams ───────────────────────────────────────────────────


@runtime_checkable
class SearchRunner(Protocol):
    """Executes one hybrid-search call and returns the hydrated hits."""

    async def search(self, ctx: Context, call: SearchCall) -> list[SearchResult]: ...


@runtime_checkable
class KbLoader(Protocol):
    """Loads knowledge-base records by id (authorization is the caller's job)."""

    async def load_by_ids(self, ids: list[str]) -> list[KnowledgeBaseInfo]: ...


@runtime_checkable
class QueryEmbeddingProvider(Protocol):
    """Computes the query embedding shared by one embedding-model group."""

    async def get_query_embedding(
        self, ctx: Context, kb_id: str, query_text: str
    ) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class WebSearchHit:
    """One web-search hit before conversion (upstream ``WebSearchResult``)."""

    title: str = ""
    url: str = ""
    snippet: str = ""
    content: str = ""
    source: str = ""
    published_at: datetime | None = None


@runtime_checkable
class WebSearchService(Protocol):
    """Runs a web search for one provider and returns the raw hits."""

    async def search(
        self,
        ctx: Context,
        *,
        tenant_id: int,
        provider_id: str,
        query: str,
        max_results: int,
        include_date: bool,
        blacklist: list[str],
        proxy_url: str,
    ) -> list[WebSearchHit]: ...


@runtime_checkable
class WebSearchConfigProvider(Protocol):
    """Loads the tenant's stored web-search configuration blob."""

    async def load(self, ctx: Context, tenant_id: int) -> JsonObject | None: ...


# ── Web-search configuration (upstream ``WebSearchConfig``) ────────────


@dataclass(frozen=True, slots=True)
class WebSearchConfig:
    """Effective web-search settings for one search."""

    provider: str = ""
    api_key: str = ""
    max_results: int = DEFAULT_WEB_SEARCH_MAX_RESULTS
    include_date: bool = False
    compression_method: str = DEFAULT_WEB_SEARCH_COMPRESSION_METHOD
    blacklist: list[str] = field(default_factory=list)
    embedding_model_id: str = ""
    embedding_dimension: int = 0
    rerank_model_id: str = ""
    document_fragments: int = 0
    proxy_url: str = ""


def _as_str(value: JsonValue | None) -> str:
    return value if isinstance(value, str) else ""


def _as_str_list(value: JsonValue | None) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _positive_int(value: JsonValue | None, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return default


def effective_web_search_config(raw: JsonObject | None) -> WebSearchConfig:
    """Normalize a tenant web-search config blob to the effective settings.

    Missing / zero values fall back to the shared defaults (upstream
    ``EffectiveWebSearchConfig``).
    """
    if raw is None:
        return WebSearchConfig()
    include_date = raw.get("include_date")
    return WebSearchConfig(
        provider=_as_str(raw.get("provider")),
        api_key=_as_str(raw.get("api_key")),
        max_results=_positive_int(raw.get("max_results"), DEFAULT_WEB_SEARCH_MAX_RESULTS),
        include_date=include_date if isinstance(include_date, bool) else False,
        compression_method=(
            _as_str(raw.get("compression_method")) or DEFAULT_WEB_SEARCH_COMPRESSION_METHOD
        ),
        blacklist=_as_str_list(raw.get("blacklist")),
        embedding_model_id=_as_str(raw.get("embedding_model_id")),
        embedding_dimension=_positive_int(raw.get("embedding_dimension"), 0),
        rerank_model_id=_as_str(raw.get("rerank_model_id")),
        document_fragments=_positive_int(raw.get("document_fragments"), 0),
        proxy_url=_as_str(raw.get("proxy_url")),
    )


# ── Concrete adapters ─────────────────────────────────────────────────


class KBServiceKbLoader:
    """``KbLoader`` adapter over ``KBService.get_knowledge_bases_by_ids``."""

    def __init__(self, service: KBService) -> None:
        self._service = service

    async def load_by_ids(self, ids: list[str]) -> list[KnowledgeBaseInfo]:
        return await self._service.get_knowledge_bases_by_ids(ids=ids)


class HybridSearchRunner:
    """``SearchRunner`` adapter over the ``hybrid_search`` orchestrator."""

    def __init__(self, deps: SearchDependencies) -> None:
        self._deps = deps

    async def search(self, ctx: Context, call: SearchCall) -> list[SearchResult]:
        params = HybridSearchParams(
            query_text=call.query_text,
            query_embedding=tuple(call.query_embedding),
            knowledge_base_ids=tuple(call.knowledge_base_ids),
            knowledge_ids=tuple(call.knowledge_ids),
            tag_ids=tuple(call.tag_ids),
            scope_tag_ids=tuple(call.scope_tag_ids),
            match_count=call.top_k,
            vector_threshold=call.vector_threshold,
            keyword_threshold=call.keyword_threshold,
            disable_keywords_match=call.disable_keywords_match,
            disable_vector_match=call.disable_vector_match,
            skip_context_enrichment=call.skip_context_enrichment,
        )
        results = await hybrid_search(
            cast("RetrievalContext", ctx), kb_id=call.kb_id, params=params, deps=self._deps
        )
        return [SearchResult(**hit.model_dump()) for hit in (results or [])]


class SearchServiceWebSearch:
    """``WebSearchService`` adapter over the web-search dispatch service."""

    def __init__(self, service: WebSearchSearchService) -> None:
        self._service = service

    async def search(
        self,
        _ctx: Context,
        *,
        tenant_id: int,
        provider_id: str,
        query: str,
        max_results: int,
        include_date: bool,
        blacklist: list[str],
        proxy_url: str,
    ) -> list[WebSearchHit]:
        results = await self._service.search(
            tenant_id=tenant_id,
            provider_id=provider_id,
            query=query,
            max_results=max_results,
            include_date=include_date,
            blacklist=blacklist,
            proxy_url=proxy_url,
        )
        return [
            WebSearchHit(
                title=result.title,
                url=result.url,
                snippet=result.snippet,
                content=result.content,
                source=result.source,
                published_at=result.published_at,
            )
            for result in results
        ]


class TenantServiceWebSearchConfigProvider:
    """``WebSearchConfigProvider`` adapter over the tenant service."""

    def __init__(self, service: TenantService) -> None:
        self._service = service

    async def load(self, ctx: Context, tenant_id: int) -> JsonObject | None:
        info = await self._service.get_tenant(tenant_id)
        return info.web_search_config


# ── Pure helpers ───────────────────────────────────────────────────────


def has_knowledge_retrieval_scope(
    search_targets: Sequence[SearchTarget],
    knowledge_base_ids: Sequence[str],
    knowledge_ids: Sequence[str],
) -> bool:
    """Report whether any knowledge retrieval scope is configured.

    A scope exists when any KB / knowledge id is non-empty or any search
    target selects a KB, specific documents, or a tag scope (upstream
    ``HasKnowledgeRetrievalScope``).
    """
    for kb_id in knowledge_base_ids:
        if kb_id:
            return True
    for knowledge_id in knowledge_ids:
        if knowledge_id:
            return True
    for target in search_targets:
        if target is None or not target.knowledge_base_id:
            continue
        if (
            target.type is SearchTargetType.KNOWLEDGE_BASE
            or bool(target.knowledge_ids)
            or bool(target.tag_ids)
            or bool(target.scope_tag_ids)
        ):
            return True
    return False


def recall_thresholds(
    target: SearchTarget,
    vector_threshold: float,
    keyword_threshold: float,
) -> tuple[float, float]:
    """Return the effective recall thresholds for ``target``.

    A target with recall thresholds disabled keeps recall broad inside an
    already constrained, user-selected scope: both thresholds become zero
    so the gates cannot erase the explicit scope first.
    """
    if target.disable_recall_thresholds:
        return 0.0, 0.0
    return vector_threshold, keyword_threshold


def convert_web_search_results(hits: Sequence[WebSearchHit]) -> list[SearchResult]:
    """Convert web-search hits into pipeline search results.

    Each hit becomes an independent result whose id doubles as its
    knowledge id so results stay distinct during merge (upstream
    ``ConvertWebSearchResults``).
    """
    results: list[SearchResult] = []
    for index, hit in enumerate(hits):
        if hit is None:
            continue
        chunk_id = hit.url
        if not chunk_id:
            chunk_id = f"web_search_{index}"
        content = hit.title
        if hit.snippet:
            content = f"{content}\n\n{hit.snippet}" if content else hit.snippet
        if hit.content:
            content = f"{content}\n\n{hit.content}" if content else hit.content
        metadata: dict[str, str] = {
            "url": hit.url,
            "source": hit.source,
            "title": hit.title,
            "snippet": hit.snippet,
        }
        if hit.published_at is not None:
            metadata["published_at"] = hit.published_at.isoformat()
        results.append(
            SearchResult(
                id=chunk_id,
                content=content,
                knowledge_id=chunk_id,
                chunk_index=0,
                knowledge_title=hit.title,
                start_at=0,
                end_at=len(content),
                seq=1,
                score=_WEB_HIT_SCORE,
                match_type=MatchType.WEB_SEARCH,
                metadata=metadata,
                chunk_type="web_search",
                parent_chunk_id="",
                image_info="",
                knowledge_filename="",
                knowledge_source="web_search",
                knowledge_channel="",
            )
        )
    return results


def log_search_score_sample(action: str, results: Sequence[SearchResult]) -> None:
    """Log the leading result scores for one pipeline moment."""
    limit = min(_MAX_SCORE_SAMPLE_ROWS, len(results))
    for index in range(limit):
        result = results[index]
        pipeline_info(
            "Search",
            action,
            {
                "index": index,
                "chunk_id": result.id,
                "score": f"{result.score:.4f}",
                "match_type": int(result.match_type),
            },
        )
    if len(results) > limit:
        pipeline_info(
            "Search",
            f"{action}_summary",
            {"total": len(results), "logged": limit, "truncated": len(results) - limit},
        )


# ── Step ───────────────────────────────────────────────────────────────


class SearchStep:
    """Runs knowledge-base + web retrieval for the ``CHUNK_SEARCH`` event."""

    def __init__(
        self,
        *,
        runner: SearchRunner,
        kb_loader: KbLoader,
        query_embedding_provider: QueryEmbeddingProvider | None = None,
        web_search: WebSearchService | None = None,
        web_search_config_provider: WebSearchConfigProvider | None = None,
    ) -> None:
        self._runner = runner
        self._kb_loader = kb_loader
        self._query_embedding_provider = query_embedding_provider
        self._web_search = web_search
        self._web_search_config_provider = web_search_config_provider

    def activation_events(self) -> Sequence[EventType]:
        return (EventType.CHUNK_SEARCH,)

    async def on_event(
        self,
        ctx: Context,
        event_type: EventType | str,
        pipeline_ctx: PipelineContext,
        next: Next,
    ) -> PluginError | None:
        has_kb_targets = has_knowledge_retrieval_scope(
            pipeline_ctx.search_targets,
            pipeline_ctx.knowledge_base_ids,
            pipeline_ctx.knowledge_ids,
        )
        if not has_kb_targets and not pipeline_ctx.web_search_enabled:
            pipeline_error("Search", "kb_not_found", {"session_id": pipeline_ctx.session_id})
            return None

        pipeline_info(
            "Search",
            "input",
            {
                "session_id": pipeline_ctx.session_id,
                "rewrite_query": pipeline_ctx.rewrite_query,
                "search_targets": len(pipeline_ctx.search_targets),
                "tenant_id": pipeline_ctx.tenant_id,
                "web_enabled": pipeline_ctx.web_search_enabled,
            },
        )
        pipeline_info(
            "Search",
            "plan",
            {
                "search_targets": len(pipeline_ctx.search_targets),
                "embedding_top_k": pipeline_ctx.embedding_top_k,
                "vector_threshold": pipeline_ctx.vector_threshold,
                "keyword_threshold": pipeline_ctx.keyword_threshold,
            },
        )

        kb_task = self.search_by_targets(ctx, pipeline_ctx)
        web_task = self.search_web_if_enabled(ctx, pipeline_ctx)
        kb_results, web_results = await asyncio.gather(kb_task, web_task)
        pipeline_ctx.search_result = [*kb_results, *web_results]

        log_search_score_sample("result_score_before_normalize", pipeline_ctx.search_result)

        if pipeline_ctx.enable_query_expansion and len(pipeline_ctx.search_result) < max(
            1, pipeline_ctx.embedding_top_k
        ):
            exp_results = await self.run_query_expansion(ctx, pipeline_ctx)
            if exp_results:
                pipeline_ctx.search_result = [*pipeline_ctx.search_result, *exp_results]

        log_search_score_sample("final_score", pipeline_ctx.search_result)

        if pipeline_ctx.search_result:
            pipeline_info(
                "Search",
                "output",
                {
                    "session_id": pipeline_ctx.session_id,
                    "result_count": len(pipeline_ctx.search_result),
                },
            )
            return await next()
        pipeline_warn(
            "Search",
            "output",
            {"session_id": pipeline_ctx.session_id, "result_count": 0},
        )
        return ERR_SEARCH_NOTHING

    # ── Knowledge-base retrieval ────────────────────────────────────

    async def search_by_targets(
        self, ctx: Context, pipeline_ctx: PipelineContext
    ) -> list[SearchResult]:
        """Search the pre-computed targets, grouped by embedding model.

        Targets sharing one embedding model are grouped so the query
        embedding is computed once per model AND all full-KB targets in a
        group are combined into a single retrieval call (upstream
        ``searchByTargets``).
        """
        if not pipeline_ctx.search_targets:
            return []

        query_text = pipeline_ctx.rewrite_query.strip()

        kb_ids = [target.knowledge_base_id for target in pipeline_ctx.search_targets]
        kb_list: list[KnowledgeBaseInfo] = []
        try:
            kb_list = await self._kb_loader.load_by_ids(kb_ids)
        except Exception as exc:
            pipeline_warn("Search", "batch_kb_fetch_error", {"error": str(exc)})

        model_key_map = {kb.id: kb.embedding_model_id or "" for kb in kb_list}
        groups: dict[str, list[SearchTarget]] = {}
        for target in pipeline_ctx.search_targets:
            key = model_key_map.get(target.knowledge_base_id, "")
            groups.setdefault(key, []).append(target)

        pipeline_info(
            "Search",
            "embedding_groups",
            {"total_targets": len(pipeline_ctx.search_targets), "unique_models": len(groups)},
        )

        batches = await asyncio.gather(
            *(
                self._search_group(ctx, pipeline_ctx, model_key, targets, query_text)
                for model_key, targets in groups.items()
            )
        )
        results = [result for batch in batches for result in batch]
        pipeline_info("Search", "kb_result_summary", {"total_hits": len(results)})
        return results

    async def _search_group(
        self,
        ctx: Context,
        pipeline_ctx: PipelineContext,
        model_key: str,
        targets: list[SearchTarget],
        query_text: str,
    ) -> list[SearchResult]:
        """Search one embedding-model group: combined + per-target calls."""
        query_embedding: tuple[float, ...] = ()
        if model_key and self._query_embedding_provider is not None:
            try:
                embedding = await self._query_embedding_provider.get_query_embedding(
                    ctx, targets[0].knowledge_base_id, query_text
                )
                if embedding:
                    query_embedding = tuple(embedding)
            except Exception as exc:
                pipeline_warn(
                    "Search",
                    "group_embed_error",
                    {
                        "model_key": model_key,
                        "kb_id": targets[0].knowledge_base_id,
                        "error": str(exc),
                    },
                )

        full_kb_ids: list[str] = []
        knowledge_targets: list[SearchTarget] = []
        for target in targets:
            if target.type is SearchTargetType.KNOWLEDGE_BASE and not target.tag_ids:
                full_kb_ids.append(target.knowledge_base_id)
            else:
                knowledge_targets.append(target)

        pipeline_info(
            "Search",
            "group_plan",
            {
                "model_key": model_key,
                "combined_kb_count": len(full_kb_ids),
                "individual_targets": len(knowledge_targets),
                "vector_len": len(query_embedding),
            },
        )

        calls: list[tuple[SearchCall, bool]] = []
        if full_kb_ids:
            calls.append(
                (
                    SearchCall(
                        query_text=query_text,
                        kb_id=full_kb_ids[0],
                        knowledge_base_ids=tuple(full_kb_ids),
                        query_embedding=query_embedding,
                        top_k=pipeline_ctx.embedding_top_k,
                        vector_threshold=pipeline_ctx.vector_threshold,
                        keyword_threshold=pipeline_ctx.keyword_threshold,
                        skip_context_enrichment=True,
                    ),
                    True,
                )
            )
        for target in knowledge_targets:
            call = self._build_single_target_call(pipeline_ctx, target, query_text, query_embedding)
            if call is not None:
                calls.append((call, False))

        if not calls:
            return []
        batches = await asyncio.gather(
            *(self._run_search_call(ctx, call, combined) for call, combined in calls)
        )
        return [result for batch in batches for result in batch]

    def _build_single_target_call(
        self,
        pipeline_ctx: PipelineContext,
        target: SearchTarget,
        query_text: str,
        query_embedding: tuple[float, ...],
    ) -> SearchCall | None:
        """Build the retrieval call for one constrained target."""
        if target.type is SearchTargetType.KNOWLEDGE and not target.knowledge_ids:
            return None
        vector_threshold, keyword_threshold = recall_thresholds(
            target, pipeline_ctx.vector_threshold, pipeline_ctx.keyword_threshold
        )
        if target.disable_recall_thresholds:
            pipeline_info(
                "Search",
                "explicit_scope_threshold_override",
                {
                    "kb_id": target.knowledge_base_id,
                    "knowledge_id_count": len(target.knowledge_ids),
                    "tag_id_count": len(target.tag_ids),
                },
            )
        return SearchCall(
            query_text=query_text,
            kb_id=target.knowledge_base_id,
            query_embedding=query_embedding,
            knowledge_ids=(
                tuple(target.knowledge_ids) if target.type is SearchTargetType.KNOWLEDGE else ()
            ),
            tag_ids=tuple(target.tag_ids),
            scope_tag_ids=tuple(target.scope_tag_ids),
            top_k=pipeline_ctx.embedding_top_k,
            vector_threshold=vector_threshold,
            keyword_threshold=keyword_threshold,
            skip_context_enrichment=True,
        )

    async def _run_search_call(
        self,
        ctx: Context,
        call: SearchCall,
        combined: bool,
    ) -> list[SearchResult]:
        """Run one search call; a failure degrades to no hits."""
        try:
            results = await self._runner.search(ctx, call)
        except Exception as exc:
            pipeline_warn(
                "Search",
                "kb_search_error",
                {"kb_id": call.kb_id, "query": call.query_text, "error": str(exc)},
            )
            return []
        pipeline_info(
            "Search",
            "combined_kb_result" if combined else "kb_result",
            {"kb_ids": list(call.knowledge_base_ids) or [call.kb_id], "hit_count": len(results)},
        )
        return results

    # ── Web search ──────────────────────────────────────────────────

    async def search_web_if_enabled(
        self, ctx: Context, pipeline_ctx: PipelineContext
    ) -> list[SearchResult]:
        """Run a web search when enabled and convert the hits (upstream
        ``searchWebIfEnabled``)."""
        if (
            not pipeline_ctx.web_search_enabled
            or self._web_search is None
            or self._web_search_config_provider is None
        ):
            return []

        provider_id = pipeline_ctx.web_search_provider_id
        if not provider_id:
            pipeline_warn("Search", "web_config_missing", {"tenant_id": pipeline_ctx.tenant_id})
            return []

        raw_config: JsonObject | None = None
        try:
            raw_config = await self._web_search_config_provider.load(ctx, pipeline_ctx.tenant_id)
        except Exception:
            raw_config = None
        config = effective_web_search_config(raw_config)
        if pipeline_ctx.web_search_max_results > 0:
            config = replace(config, max_results=pipeline_ctx.web_search_max_results)

        pipeline_info(
            "Search",
            "web_request",
            {"tenant_id": pipeline_ctx.tenant_id, "provider_id": provider_id},
        )
        try:
            hits = await self._web_search.search(
                ctx,
                tenant_id=pipeline_ctx.tenant_id,
                provider_id=provider_id,
                query=pipeline_ctx.rewrite_query,
                max_results=config.max_results,
                include_date=config.include_date,
                blacklist=config.blacklist,
                proxy_url=config.proxy_url,
            )
        except Exception as exc:
            pipeline_warn(
                "Search",
                "web_search_error",
                {"tenant_id": pipeline_ctx.tenant_id, "error": str(exc)},
            )
            return []
        results = convert_web_search_results(hits)
        pipeline_info("Search", "web_hits", {"hit_count": len(results)})
        return results

    # ── Query expansion ─────────────────────────────────────────────

    async def run_query_expansion(
        self, ctx: Context, pipeline_ctx: PipelineContext
    ) -> list[SearchResult]:
        """Re-run retrieval over local query variants when recall is low.

        Every (variant, target) pair is searched concurrently with a
        widened keyword gate; results carry the owning knowledge-base id
        (upstream ``runQueryExpansion``).
        """
        pipeline_info(
            "Search",
            "recall_low",
            {"current": len(pipeline_ctx.search_result), "threshold": pipeline_ctx.embedding_top_k},
        )
        expansions = expand_queries(pipeline_ctx)
        if not expansions:
            return []
        pipeline_info("Search", "expansion_start", {"variants": len(expansions)})

        exp_top_k = max(pipeline_ctx.embedding_top_k * 2, pipeline_ctx.rerank_top_k * 2)
        exp_kw_threshold = pipeline_ctx.keyword_threshold * 0.8

        targets = [
            target
            for target in pipeline_ctx.search_targets
            if target is not None and target.knowledge_base_id
        ]
        jobs = len(expansions) * len(targets)
        cap_sem = min(16, jobs) if jobs > 0 else 1
        pipeline_info("Search", "expansion_concurrency", {"jobs": jobs, "cap": cap_sem})
        semaphore = asyncio.Semaphore(cap_sem)

        async def _expand(query: str, target: SearchTarget) -> list[SearchResult]:
            async with semaphore:
                vector_threshold, keyword_threshold = recall_thresholds(
                    target, pipeline_ctx.vector_threshold, exp_kw_threshold
                )
                call = SearchCall(
                    query_text=query,
                    kb_id=target.knowledge_base_id,
                    knowledge_ids=(
                        tuple(target.knowledge_ids)
                        if target.type is SearchTargetType.KNOWLEDGE
                        else ()
                    ),
                    tag_ids=tuple(target.tag_ids),
                    scope_tag_ids=tuple(target.scope_tag_ids),
                    top_k=exp_top_k,
                    vector_threshold=vector_threshold,
                    keyword_threshold=keyword_threshold,
                    disable_vector_match=False,
                    disable_keywords_match=False,
                    skip_context_enrichment=True,
                )
                try:
                    results = await self._runner.search(ctx, call)
                except Exception as exc:
                    pipeline_warn(
                        "Search",
                        "expansion_error",
                        {"kb_id": target.knowledge_base_id, "error": str(exc)},
                    )
                    return []
                if results:
                    pipeline_info(
                        "Search",
                        "expansion_hits",
                        {
                            "kb_id": target.knowledge_base_id,
                            "query": query,
                            "hits": len(results),
                        },
                    )
                    return [
                        result.model_copy(update={"knowledge_base_id": target.knowledge_base_id})
                        for result in results
                    ]
                return []

        batches = await asyncio.gather(
            *(_expand(query, target) for query in expansions for target in targets)
        )
        exp_results = [result for batch in batches for result in batch]
        if exp_results:
            pipeline_info("Search", "expansion_done", {"added": len(exp_results)})
        return exp_results


__all__ = [
    "DEFAULT_WEB_SEARCH_COMPRESSION_METHOD",
    "DEFAULT_WEB_SEARCH_MAX_RESULTS",
    "HybridSearchRunner",
    "KBServiceKbLoader",
    "KbLoader",
    "QueryEmbeddingProvider",
    "SearchCall",
    "SearchRunner",
    "SearchServiceWebSearch",
    "SearchStep",
    "TenantServiceWebSearchConfigProvider",
    "WebSearchConfig",
    "WebSearchConfigProvider",
    "WebSearchHit",
    "WebSearchService",
    "convert_web_search_results",
    "effective_web_search_config",
    "has_knowledge_retrieval_scope",
    "log_search_score_sample",
    "recall_thresholds",
]
