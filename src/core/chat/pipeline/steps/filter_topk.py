"""Pipeline step: truncate merged/reranked/search results to the top K.

Ports the upstream ``FILTER_TOP_K`` semantics: restore a deterministic
global relevance order, then truncate the first available result list —
merge first, then rerank, then raw search — to ``rerank_top_k`` entries.
"""

from __future__ import annotations

from src.core.chat.pipeline.common import pipeline_info, pipeline_warn
from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.engine import Next, PluginError
from src.core.chat.pipeline.types import Context, EventType, SearchResult


def sort_search_results_deterministically(
    results: list[SearchResult],
) -> list[SearchResult]:
    """Order results by relevance with stable tie-breakers.

    Mirrors the upstream comparator: score descending, then knowledge id,
    chunk type, chunk index and id ascending. Stable ordering keeps
    identical requests reproducible before the TopK truncation, even after
    merge stages reshuffled the list through maps.
    """
    return sorted(
        results,
        key=lambda result: (
            -result.score,
            result.knowledge_id,
            result.chunk_type,
            result.chunk_index,
            result.id,
        ),
    )


class FilterTopKPlugin:
    """Pipeline step that keeps only the top ``rerank_top_k`` results."""

    def activation_events(self) -> list[EventType]:
        return [EventType.FILTER_TOP_K]

    async def on_event(
        self,
        ctx: Context,
        event_type: EventType | str,
        pipeline_ctx: PipelineContext,
        next: Next,
    ) -> PluginError | None:
        if not pipeline_ctx.needs_retrieval():
            return await next()
        pipeline_info(
            "FilterTopK",
            "input",
            {
                "session_id": pipeline_ctx.session_id,
                "top_k": pipeline_ctx.rerank_top_k,
                "merge_cnt": len(pipeline_ctx.merge_result),
                "rerank_cnt": len(pipeline_ctx.rerank_result),
                "search_cnt": len(pipeline_ctx.search_result),
            },
        )

        def filter_top_k(results: list[SearchResult], top_k: int) -> list[SearchResult]:
            ordered = sort_search_results_deterministically(results)
            if top_k > 0 and len(ordered) > top_k:
                pipeline_info(
                    "FilterTopK",
                    "filter",
                    {"before": len(ordered), "after": top_k},
                )
                return ordered[:top_k]
            return ordered

        if pipeline_ctx.merge_result:
            pipeline_ctx.merge_result = filter_top_k(
                pipeline_ctx.merge_result, pipeline_ctx.rerank_top_k
            )
        elif pipeline_ctx.rerank_result:
            pipeline_ctx.rerank_result = filter_top_k(
                pipeline_ctx.rerank_result, pipeline_ctx.rerank_top_k
            )
        elif pipeline_ctx.search_result:
            pipeline_ctx.search_result = filter_top_k(
                pipeline_ctx.search_result, pipeline_ctx.rerank_top_k
            )
        else:
            pipeline_warn("FilterTopK", "skip", {"reason": "no_results"})

        pipeline_info(
            "FilterTopK",
            "output",
            {
                "merge_cnt": len(pipeline_ctx.merge_result),
                "rerank_cnt": len(pipeline_ctx.rerank_result),
                "search_cnt": len(pipeline_ctx.search_result),
            },
        )
        return await next()


__all__ = ["FilterTopKPlugin", "sort_search_results_deterministically"]
