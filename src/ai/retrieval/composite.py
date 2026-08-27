"""Composite retrieval engine (upstream ``composite.go``).

``CompositeRetrieveEngine`` holds multiple ``RetrieveEngineService``
instances. ``retrieve`` fans each retrieval out to the first engine that
serves the requested retriever type and merges the ranked results; write
operations (indexing, deletes, batch updates, index copy) fan out to every
registered engine concurrently and surface the first error.

``new_composite_retrieve_engine`` builds one from a registry lookup over
the tenant's effective engine params, grouping entries by engine type and
validating that each engine supports the requested retriever types.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from src.ai.embedding import Context, Embedder
from src.ai.retrieval.base import RetrieveEngineRegistry, RetrieveEngineService
from src.ai.retrieval.types import (
    IndexInfo,
    RetrieveParams,
    RetrieverEngineParams,
    RetrieverEngineType,
    RetrieveResult,
    RetrieverType,
)
from src.common.exception import NotFoundError, ValidationError


@dataclass(frozen=True, slots=True)
class EngineInfo:
    """One engine service and the retriever types it serves."""

    retrieve_engine: RetrieveEngineService
    retriever_types: tuple[RetrieverType, ...]


def _dedupe_by_source_id(items: list[IndexInfo]) -> list[IndexInfo]:
    """Keep the first occurrence of each source id, preserving order."""
    seen: set[str] = set()
    result: list[IndexInfo] = []
    for item in items:
        if item.source_id in seen:
            continue
        seen.add(item.source_id)
        result.append(item)
    return result


class CompositeRetrieveEngine:
    """Fan-out retrieval engine delegating to its registered engines."""

    def __init__(self, engine_infos: list[EngineInfo]) -> None:
        self._engine_infos: tuple[EngineInfo, ...] = tuple(engine_infos)

    def support_retriever(self, retriever_type: RetrieverType) -> bool:
        """Report whether any registered engine serves ``retriever_type``."""
        return any(retriever_type in info.retriever_types for info in self._engine_infos)

    async def retrieve(
        self, ctx: Context, retrieve_params: list[RetrieveParams]
    ) -> list[RetrieveResult]:
        """Run each retrieval against the first matching engine, concurrently.

        Results are merged in input order. Raises the first engine error;
        when no engine serves a requested retriever type, raises
        ``NotFoundError``.
        """
        results = await asyncio.gather(
            *(self._retrieve_one(ctx, params) for params in retrieve_params),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return [
            item for result in results if not isinstance(result, BaseException) for item in result
        ]

    async def _retrieve_one(self, ctx: Context, params: RetrieveParams) -> list[RetrieveResult]:
        for engine_info in self._engine_infos:
            if params.retriever_type in engine_info.retriever_types:
                return await engine_info.retrieve_engine.retrieve(ctx, params)
        raise NotFoundError(
            code="retriever_type_not_found",
            message=f"retriever type {params.retriever_type} not found",
        )

    # ── write fan-out (every engine) ─────────────────────────────────

    async def index(self, ctx: Context, embedder: Embedder, index_info: IndexInfo) -> None:
        async def _op(_ctx: Context, info: EngineInfo) -> None:
            await info.retrieve_engine.index(_ctx, embedder, index_info, list(info.retriever_types))

        await self._concurrent_exec(ctx, _op)

    async def batch_index(
        self, ctx: Context, embedder: Embedder, index_info_list: list[IndexInfo]
    ) -> None:
        # Deduplicate source ids before fan-out so a chunk repeated in the
        # batch is not indexed twice.
        deduped = _dedupe_by_source_id(index_info_list)

        async def _op(_ctx: Context, info: EngineInfo) -> None:
            await info.retrieve_engine.batch_index(
                _ctx, embedder, deduped, list(info.retriever_types)
            )

        await self._concurrent_exec(ctx, _op)

    async def delete_by_chunk_id_list(
        self,
        ctx: Context,
        index_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        async def _op(_ctx: Context, info: EngineInfo) -> None:
            await info.retrieve_engine.delete_by_chunk_id_list(
                _ctx, index_id_list, dimension, knowledge_type
            )

        await self._concurrent_exec(ctx, _op)

    async def delete_by_source_id_list(
        self,
        ctx: Context,
        source_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        async def _op(_ctx: Context, info: EngineInfo) -> None:
            await info.retrieve_engine.delete_by_source_id_list(
                _ctx, source_id_list, dimension, knowledge_type
            )

        await self._concurrent_exec(ctx, _op)

    async def copy_indices(
        self,
        ctx: Context,
        source_knowledge_base_id: str,
        source_to_target_kb_id_map: Mapping[str, str],
        source_to_target_chunk_id_map: Mapping[str, str],
        target_knowledge_base_id: str,
        dimension: int,
        knowledge_type: str,
    ) -> None:
        async def _op(_ctx: Context, info: EngineInfo) -> None:
            await info.retrieve_engine.copy_indices(
                _ctx,
                source_knowledge_base_id,
                source_to_target_kb_id_map,
                source_to_target_chunk_id_map,
                target_knowledge_base_id,
                dimension,
                knowledge_type,
            )

        await self._concurrent_exec(ctx, _op)

    async def delete_by_knowledge_id_list(
        self,
        ctx: Context,
        knowledge_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        async def _op(_ctx: Context, info: EngineInfo) -> None:
            await info.retrieve_engine.delete_by_knowledge_id_list(
                _ctx, knowledge_id_list, dimension, knowledge_type
            )

        await self._concurrent_exec(ctx, _op)

    async def batch_update_chunk_enabled_status(
        self, ctx: Context, chunk_status_map: Mapping[str, bool]
    ) -> None:
        async def _op(_ctx: Context, info: EngineInfo) -> None:
            await info.retrieve_engine.batch_update_chunk_enabled_status(_ctx, chunk_status_map)

        await self._concurrent_exec(ctx, _op)

    async def batch_update_chunk_tag_id(
        self, ctx: Context, chunk_tag_map: Mapping[str, str]
    ) -> None:
        async def _op(_ctx: Context, info: EngineInfo) -> None:
            await info.retrieve_engine.batch_update_chunk_tag_id(_ctx, chunk_tag_map)

        await self._concurrent_exec(ctx, _op)

    def estimate_storage_size(
        self,
        ctx: Context,
        embedder: Embedder,
        index_info_list: list[IndexInfo],
    ) -> int:
        """Sum the estimated storage size across every registered engine."""
        return sum(
            info.retrieve_engine.estimate_storage_size(
                ctx, embedder, index_info_list, list(info.retriever_types)
            )
            for info in self._engine_infos
        )

    async def _concurrent_exec(
        self,
        ctx: Context,
        fn: Callable[[Context, EngineInfo], Awaitable[None]],
    ) -> None:
        """Run ``fn`` on every engine concurrently; raise the first error."""
        results = await asyncio.gather(
            *(fn(ctx, engine_info) for engine_info in self._engine_infos),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result


def new_composite_retrieve_engine(
    registry: RetrieveEngineRegistry,
    engine_params: list[RetrieverEngineParams],
) -> CompositeRetrieveEngine:
    """Build a composite from a registry lookup over effective engine params.

    Entries are grouped by engine type so one engine serving several
    retriever types appears once. Raises when an engine is not registered
    (propagated from the registry) or when it does not support a requested
    retriever type.
    """
    by_engine: dict[RetrieverEngineType, tuple[RetrieveEngineService, list[RetrieverType]]] = {}
    for engine_param in engine_params:
        svc = registry.get_retrieve_engine_service(engine_param.retriever_engine_type)
        if engine_param.retriever_type not in svc.support():
            raise ValidationError(
                code="retriever_engine_unsupported",
                message=(
                    f"retrieval engine {svc.engine_type()} does not support "
                    f"retriever type: {engine_param.retriever_type}"
                ),
            )
        engine_type = svc.engine_type()
        entry = by_engine.get(engine_type)
        if entry is None:
            by_engine[engine_type] = (svc, [engine_param.retriever_type])
        else:
            _svc, retriever_types = entry
            retriever_types.append(engine_param.retriever_type)
    engine_infos = [
        EngineInfo(retrieve_engine=svc, retriever_types=tuple(types))
        for svc, types in by_engine.values()
    ]
    return CompositeRetrieveEngine(engine_infos)


__all__ = [
    "CompositeRetrieveEngine",
    "EngineInfo",
    "new_composite_retrieve_engine",
]
