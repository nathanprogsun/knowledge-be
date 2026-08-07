"""Retrieval engine interfaces (upstream ``interfaces/retriever.go``).

``RetrieveEngine`` / ``RetrieveEngineRepository`` / ``RetrieveEngineService``
are the provider-facing and service-facing contracts; ``RetrieveEngineRegistry``
and ``StoreRegistry`` are the lookup contracts. The registry implementation
lives in ``src.ai.retrieval.registry``. The engine factory type and the
store-repository seam used for on-demand rebuilds are declared here too,
mirroring the upstream ``interfaces/vectorstore.go`` surface.

The ai layer never imports core or storage: the ``Embedder`` and ``Context``
protocols come from the embedding package, and store rows are supplied
structurally via ``VectorStoreLike``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Protocol, TypeAlias

from src.ai.embedding import Context, Embedder
from src.ai.retrieval.types import (
    IndexInfo,
    IndexSaveParams,
    RetrieveParams,
    RetrieverEngineType,
    RetrieveResult,
    RetrieverType,
    VectorStoreLike,
)


class Database(Protocol):
    """Opaque database handle consumed by DB-backed engine repositories.

    The wiring layer supplies its storage engine (e.g. an async SQLAlchemy
    engine). The retrieval layer only forwards the handle; it never
    inspects it.
    """


class AppConfig(Protocol):
    """Opaque application configuration consumed by engine repositories.

    Forwarded to repository constructors that need runtime settings; the
    retrieval layer does not inspect it.
    """


class AuditSink(Protocol):
    """Engine audit hook (upstream OpenSearch ``AuditSink``).

    Implemented by the opensearch engine; every other engine treats the
    sink as unused.
    """

    async def emit_index_created(self, ctx: Context, alias: str, dim: int) -> None: ...

    async def emit_reindex_executed(
        self, ctx: Context, src_alias: str, dst_alias: str, docs: int
    ) -> None: ...


class RetrieveEngine(Protocol):
    """Retrieve engine interface (upstream ``RetrieveEngine``)."""

    def engine_type(self) -> RetrieverEngineType:
        """Return the retrieve engine type."""
        ...

    async def retrieve(self, ctx: Context, params: RetrieveParams) -> list[RetrieveResult]:
        """Execute the retrieval and return per-engine results."""
        ...

    def support(self) -> list[RetrieverType]:
        """Return the retriever types supported by this engine."""
        ...


class RetrieveEngineRepository(RetrieveEngine, Protocol):
    """Retrieve engine repository interface (upstream ``RetrieveEngineRepository``)."""

    async def save(self, ctx: Context, index_info: IndexInfo, params: IndexSaveParams) -> None:
        """Save the index info with its embedding map."""
        ...

    async def batch_save(
        self,
        ctx: Context,
        index_info_list: list[IndexInfo],
        params: IndexSaveParams,
    ) -> None:
        """Save a list of index infos in one batch."""
        ...

    def estimate_storage_size(
        self,
        ctx: Context,
        index_info_list: list[IndexInfo],
        params: IndexSaveParams,
    ) -> int:
        """Estimate the storage size needed for the index info list."""
        ...

    async def delete_by_chunk_id_list(
        self,
        ctx: Context,
        index_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        """Delete index records by chunk id list."""
        ...

    async def delete_by_source_id_list(
        self,
        ctx: Context,
        source_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        """Delete index records by source id list."""
        ...

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
        """Copy index data from a source KB to a target KB."""
        ...

    async def delete_by_knowledge_id_list(
        self,
        ctx: Context,
        knowledge_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        """Delete index records by knowledge id list."""
        ...

    async def batch_update_chunk_enabled_status(
        self, ctx: Context, chunk_status_map: Mapping[str, bool]
    ) -> None:
        """Update the enabled status of chunks in batch."""
        ...

    async def batch_update_chunk_tag_id(
        self, ctx: Context, chunk_tag_map: Mapping[str, str]
    ) -> None:
        """Update the tag id of chunks in batch."""
        ...


class RetrieveEngineService(RetrieveEngine, Protocol):
    """Retrieve engine service interface (upstream ``RetrieveEngineService``)."""

    async def index(
        self,
        ctx: Context,
        embedder: Embedder,
        index_info: IndexInfo,
        retriever_types: list[RetrieverType],
    ) -> None:
        """Embed and index a single chunk when vector retrieval is enabled."""
        ...

    async def batch_index(
        self,
        ctx: Context,
        embedder: Embedder,
        index_info_list: list[IndexInfo],
        retriever_types: list[RetrieverType],
    ) -> None:
        """Embed and index a batch of chunks when vector retrieval is enabled."""
        ...

    def estimate_storage_size(
        self,
        ctx: Context,
        embedder: Embedder,
        index_info_list: list[IndexInfo],
        retriever_types: list[RetrieverType],
    ) -> int:
        """Estimate the storage size needed for the index info list."""
        ...

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
        """Copy indices from a source KB to a target KB."""
        ...

    async def delete_by_chunk_id_list(
        self,
        ctx: Context,
        index_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        """Delete index records by chunk id list."""
        ...

    async def delete_by_source_id_list(
        self,
        ctx: Context,
        source_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        """Delete index records by source id list."""
        ...

    async def delete_by_knowledge_id_list(
        self,
        ctx: Context,
        knowledge_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        """Delete index records by knowledge id list."""
        ...

    async def batch_update_chunk_enabled_status(
        self, ctx: Context, chunk_status_map: Mapping[str, bool]
    ) -> None:
        """Update the enabled status of chunks in batch."""
        ...

    async def batch_update_chunk_tag_id(
        self, ctx: Context, chunk_tag_map: Mapping[str, str]
    ) -> None:
        """Update the tag id of chunks in batch."""
        ...


#: Creates a ``RetrieveEngineService`` from a ``VectorStore`` config
#: (upstream ``interfaces.EngineFactory``).
EngineFactory: TypeAlias = Callable[[Context, VectorStoreLike], Awaitable[RetrieveEngineService]]


class StoreRegistry(Protocol):
    """VectorStore-based engine registration / lookup (upstream ``StoreRegistry``)."""

    def register_with_store_id(self, store_id: str, svc: RetrieveEngineService) -> None:
        """Register an engine service by store id (upsert)."""
        ...

    def get_by_store_id(self, store_id: str) -> RetrieveEngineService:
        """Return the engine service registered for a store id."""
        ...

    def unregister_by_store_id(self, store_id: str) -> None:
        """Remove an engine service by store id (idempotent)."""
        ...


class RetrieveEngineRegistry(StoreRegistry, Protocol):
    """Retrieval engine registry interface (upstream ``RetrieveEngineRegistry``)."""

    def register(self, service: RetrieveEngineService) -> None:
        """Register an engine service by engine type.

        Raises ``ConflictError`` when the engine type is already registered.
        """
        ...

    def get_retrieve_engine_service(
        self, engine_type: RetrieverEngineType
    ) -> RetrieveEngineService:
        """Return the engine service registered for an engine type."""
        ...

    def get_all_retrieve_engine_services(self) -> list[RetrieveEngineService]:
        """Return every engine service registered by engine type."""
        ...

    async def get_or_load_by_store_id(
        self, ctx: Context, tenant_id: int, store_id: str
    ) -> RetrieveEngineService:
        """Return the engine for ``store_id``, rebuilding it when missing.

        Scoped to ``tenant_id`` so a caller reaching this method without an
        ownership check cannot hydrate another tenant's store.
        """
        ...


class VectorStoreRepositoryLike(Protocol):
    """Minimal store repository used for on-demand engine rebuilds."""

    async def get_by_id(
        self, ctx: Context, tenant_id: int, store_id: str
    ) -> VectorStoreLike | None:
        """Load a store within a tenant scope, or ``None`` when missing."""
        ...


__all__ = [
    "AppConfig",
    "AuditSink",
    "Context",
    "Database",
    "Embedder",
    "EngineFactory",
    "RetrieveEngine",
    "RetrieveEngineRegistry",
    "RetrieveEngineRepository",
    "RetrieveEngineService",
    "StoreRegistry",
    "VectorStoreRepositoryLike",
]
