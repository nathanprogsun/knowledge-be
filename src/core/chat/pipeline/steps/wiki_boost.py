"""Pipeline wiki-boost step (upstream ``wiki_boost.go``).

Wiki page chunks carry LLM-synthesized, cross-referenced knowledge and are
preferred over raw document chunks when both are available. After the rerank
stage runs, any ``wiki_page`` chunk whose score matches is multiplied by
``WIKI_BOOST_FACTOR`` and the result set is re-sorted — provided at least
one search target is actually a wiki-enabled knowledge base. Non-wiki runs
skip the knowledge-base lookup entirely.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.core.chat.pipeline.common import pipeline_info
from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.engine import Next, PluginError
from src.core.chat.pipeline.types import Context, EventType
from src.core.knowledge.chunks.types import CHUNK_TYPE_WIKI_PAGE
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo

#: Score multiplier applied to wiki page chunks before the re-sort.
WIKI_BOOST_FACTOR = 1.3


@runtime_checkable
class KnowledgeBaseService(Protocol):
    """Resolves a knowledge base by id without a tenant filter."""

    async def get_knowledge_base_by_id_only(
        self,
        *,
        knowledge_base_id: str,
    ) -> KnowledgeBaseInfo: ...


def _is_wiki_enabled(kb: KnowledgeBaseInfo) -> bool:
    """Return whether the KB's indexing strategy enables the wiki pipeline."""
    strategy = kb.indexing_strategy
    if strategy is None:
        return False
    return strategy.get("wiki_enabled") is True


class WikiBoostPlugin:
    """Boosts the relevance score of wiki page chunks in search results."""

    def __init__(self, kb_service: KnowledgeBaseService) -> None:
        self._kb_service = kb_service

    def activation_events(self) -> list[EventType]:
        return [EventType.CHUNK_RERANK]

    async def on_event(
        self,
        ctx: Context,
        event_type: EventType | str,
        pipeline_ctx: PipelineContext,
        next: Next,
    ) -> PluginError | None:
        # Run the normal reranking first.
        error = await next()
        if error is not None:
            return error

        # Fast path: skip all work when no wiki chunk is in the result set,
        # avoiding a KB lookup on every non-wiki turn.
        if not any(
            result.chunk_type == CHUNK_TYPE_WIKI_PAGE for result in pipeline_ctx.rerank_result
        ):
            return None

        if not await self._has_wiki_kb(pipeline_ctx):
            return None

        boosted = [
            result.model_copy(update={"score": result.score * WIKI_BOOST_FACTOR})
            if result.chunk_type == CHUNK_TYPE_WIKI_PAGE
            else result
            for result in pipeline_ctx.rerank_result
        ]
        pipeline_ctx.rerank_result = boosted
        boosted_count = sum(1 for result in boosted if result.chunk_type == CHUNK_TYPE_WIKI_PAGE)
        if boosted_count > 0:
            pipeline_info(
                "WikiBoost",
                "boosted",
                {"count": boosted_count, "factor": WIKI_BOOST_FACTOR},
            )
            # Re-sort by score after boosting; the sort is stable so ties
            # keep their previous relative order.
            pipeline_ctx.rerank_result = sorted(
                boosted,
                key=lambda result: result.score,
                reverse=True,
            )
        return None

    async def _has_wiki_kb(self, pipeline_ctx: PipelineContext) -> bool:
        """Return whether any search target is a wiki-enabled knowledge base."""
        for target in pipeline_ctx.search_targets:
            if target.knowledge_base_id == "":
                continue
            try:
                kb = await self._kb_service.get_knowledge_base_by_id_only(
                    knowledge_base_id=target.knowledge_base_id
                )
            except Exception:
                continue
            if kb is not None and _is_wiki_enabled(kb):
                return True
        return False


__all__ = ["WIKI_BOOST_FACTOR", "KnowledgeBaseService", "WikiBoostPlugin"]
