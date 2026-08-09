"""Parallel search pipeline step (upstream ``PluginSearchParallel``).

Runs chunk search and entity-graph search concurrently over cloned run
carriers, then merges and deduplicates the hits onto the shared context.

Chunk search is injected as a ``Plugin`` (the step that activates on
``CHUNK_SEARCH``); entity search is composed internally from the graph /
chunk / knowledge repositories so the two searches never write to the
same carrier instance.
"""

from __future__ import annotations

from loguru import logger

from src.ai.graph.types import RetrieveGraphRepository
from src.core.chat.pipeline.common import ParallelTask, pipeline_info, run_parallel
from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.engine import ERR_SEARCH_NOTHING, Next, Plugin, PluginError
from src.core.chat.pipeline.steps.search_entity import (
    ChunkStore,
    KnowledgeStore,
    SearchEntityPlugin,
    remove_duplicate_results,
)
from src.core.chat.pipeline.types import Context, EventType


class SearchParallelPlugin:
    """Runs chunk + entity search concurrently and merges the hits."""

    def __init__(
        self,
        *,
        chunk_search_plugin: Plugin | None = None,
        graph_repo: RetrieveGraphRepository,
        chunk_repo: ChunkStore,
        knowledge_repo: KnowledgeStore,
    ) -> None:
        self._chunk_search_plugin = chunk_search_plugin
        self._entity_search_plugin = SearchEntityPlugin(
            graph_repo=graph_repo,
            chunk_repo=chunk_repo,
            knowledge_repo=knowledge_repo,
        )

    def activation_events(self) -> list[EventType]:
        return [EventType.CHUNK_SEARCH_PARALLEL]

    async def on_event(
        self,
        ctx: Context,
        event_type: EventType | str,
        pipeline_ctx: PipelineContext,
        next: Next,
    ) -> PluginError | None:
        if not pipeline_ctx.needs_retrieval():
            pipeline_info(
                "SearchParallel",
                "skip",
                {"session_id": pipeline_ctx.session_id, "reason": "intent_no_search"},
            )
            return await next()

        pipeline_info(
            "SearchParallel",
            "start",
            {
                "session_id": pipeline_ctx.session_id,
                "has_entities": bool(pipeline_ctx.entity),
                "rewrite_query": pipeline_ctx.rewrite_query,
            },
        )

        # Each search mutates only its own deep copy; the originals are
        # merged afterwards so the parallel branches never race on a shared
        # ``search_result`` slice.
        chunk_cm = pipeline_ctx.clone()
        chunk_cm.search_result = []
        entity_cm = pipeline_ctx.clone()
        entity_cm.search_result = []

        async def _noop() -> PluginError | None:
            return None

        async def _run_chunk_search() -> PluginError | None:
            if self._chunk_search_plugin is None:
                pipeline_info(
                    "SearchParallel",
                    "chunk_search_skip",
                    {"reason": "no_chunk_search_plugin"},
                )
                return None
            error = await self._chunk_search_plugin.on_event(
                ctx,
                EventType.CHUNK_SEARCH,
                chunk_cm,
                _noop,
            )
            pipeline_info(
                "SearchParallel",
                "chunk_search_done",
                {
                    "result_count": len(chunk_cm.search_result),
                    "has_error": error is not None and error != ERR_SEARCH_NOTHING,
                },
            )
            if error == ERR_SEARCH_NOTHING:
                return None
            return error

        async def _run_entity_search() -> PluginError | None:
            if not pipeline_ctx.entity:
                pipeline_info(
                    "SearchParallel",
                    "entity_search_skip",
                    {"reason": "no_entities"},
                )
                return None
            error = await self._entity_search_plugin.on_event(
                ctx,
                EventType.ENTITY_SEARCH,
                entity_cm,
                _noop,
            )
            pipeline_info(
                "SearchParallel",
                "entity_search_done",
                {
                    "result_count": len(entity_cm.search_result),
                    "has_error": error is not None and error != ERR_SEARCH_NOTHING,
                },
            )
            if error == ERR_SEARCH_NOTHING:
                return None
            return error

        errors = await run_parallel(
            [
                ParallelTask(name="chunk_search", run=_run_chunk_search),
                ParallelTask(name="entity_search", run=_run_entity_search),
            ]
        )

        pipeline_ctx.search_result = remove_duplicate_results(
            [*chunk_cm.search_result, *entity_cm.search_result],
        )

        for name, error in errors.items():
            logger.warning("[SearchParallel] {} error: {}", name, error.description)

        pipeline_info(
            "SearchParallel",
            "complete",
            {
                "session_id": pipeline_ctx.session_id,
                "chunk_results": len(chunk_cm.search_result),
                "entity_results": len(entity_cm.search_result),
                "total_results": len(pipeline_ctx.search_result),
                "error_count": len(errors),
            },
        )

        if not pipeline_ctx.search_result:
            chunk_error = errors.get("chunk_search")
            if chunk_error is not None:
                return chunk_error
            return ERR_SEARCH_NOTHING

        return await next()


__all__ = ["SearchParallelPlugin"]
