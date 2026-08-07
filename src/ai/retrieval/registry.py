"""Retrieval engine registry (upstream ``service/retriever/registry.go``).

Keeps two lookup maps: ``by_engine_type`` for the env stores registered from
``RETRIEVE_DRIVER`` and ``by_store_id`` for DB stores. On-demand rebuilds
(``get_or_load_by_store_id``) are collapsed per store, bounded by a build
timeout, and throttled by a failure cooldown so a backend that stays down
does not cost a full build attempt on every request.

Concurrency follows the asyncio single-threaded model: every dict mutation
here is synchronous (no ``await`` between read and write), so the
generation compare-and-swap used by the rebuild path cannot race a
registration or removal that lands while a build runs.
"""

from __future__ import annotations

import asyncio
import time
from functools import partial

from src.ai.retrieval.base import (
    Context,
    EngineFactory,
    RetrieveEngineService,
    VectorStoreRepositoryLike,
)
from src.ai.retrieval.types import RetrieverEngineType
from src.app_logging import logger
from src.common.exception import ConflictError, ExternalServiceError, NotFoundError

#: Bounds a single on-demand engine build (upstream ``EngineBuildTimeout``).
ENGINE_BUILD_TIMEOUT_SECONDS: float = 10.0

#: Throttles rebuild attempts after a failed build (upstream ``rebuildCooldown``).
REBUILD_COOLDOWN_SECONDS: float = 30.0


class VectorStoreNotFoundError(NotFoundError):
    """The store does not exist for the tenant (upstream ``ErrVectorStoreNotFound``).

    Callers should treat this as non-retryable: no amount of waiting brings
    back a store that is not in the database.
    """

    code = "vector_store_not_found"
    message = "vector store not available"


class VectorStoreUnavailableError(ExternalServiceError):
    """The store exists but its engine cannot be produced right now.

    Mirrors upstream ``ErrVectorStoreUnavailable``: the metadata database was
    unreachable, or building the engine failed against a backend that may
    simply be down. Callers should retry rather than discard the task. The
    message carries no detail because the underlying errors embed endpoints
    and credentials; the cause is logged where it happens.
    """

    code = "vector_store_unavailable"
    message = "vector store engine unavailable"


class RetrieveEngineRegistry:
    """Registry implementing both the ``RetrieveEngineRegistry`` and ``StoreRegistry`` contracts."""

    def __init__(
        self,
        repo: VectorStoreRepositoryLike | None,
        factory: EngineFactory | None,
    ) -> None:
        self._by_engine_type: dict[RetrieverEngineType, RetrieveEngineService] = {}
        self._by_store_id: dict[str, RetrieveEngineService] = {}
        self._store_gen: dict[str, int] = {}
        self._failed_until: dict[str, float] = {}
        self._repo = repo
        self._factory = factory
        self._flights: dict[str, asyncio.Task[RetrieveEngineService]] = {}

    # ── env-store (byEngineType) API ────────────────────────────────

    def register(self, service: RetrieveEngineService) -> None:
        """Register an engine service by engine type.

        Raises ``ConflictError`` when the engine type is already registered.
        """
        engine_type = service.engine_type()
        if engine_type in self._by_engine_type:
            raise ConflictError(
                code="retriever_engine_already_registered",
                message=f"repository type {engine_type} already registered",
            )
        self._by_engine_type[engine_type] = service

    def get_retrieve_engine_service(
        self, engine_type: RetrieverEngineType
    ) -> RetrieveEngineService:
        """Return the engine service for ``engine_type``.

        Only searches the ``by_engine_type`` map (env stores).
        """
        service = self._by_engine_type.get(engine_type)
        if service is None:
            raise NotFoundError(
                code="retriever_engine_not_found",
                message=f"repository of type {engine_type} not found",
            )
        return service

    def get_all_retrieve_engine_services(self) -> list[RetrieveEngineService]:
        """Return every env-store engine service (backward compatible)."""
        return list(self._by_engine_type.values())

    # ── DB-store (byStoreID) API ────────────────────────────────────

    def register_with_store_id(self, store_id: str, service: RetrieveEngineService) -> None:
        """Register an engine service by store id (upsert semantics).

        Unlike ``register``, the same engine type can be registered under
        different store ids (e.g. two Elasticsearch clusters). The
        generation bump invalidates any on-demand build already in flight
        for this store.
        """
        self._by_store_id[store_id] = service
        self._bump_generation_locked(store_id)

    def get_by_store_id(self, store_id: str) -> RetrieveEngineService:
        """Return the engine service for ``store_id``.

        Callers must verify tenant ownership before using the returned
        service.
        """
        service = self._by_store_id.get(store_id)
        if service is None:
            raise NotFoundError(
                code="vector_store_not_found",
                message=f"store {store_id} not found in registry",
            )
        return service

    def unregister_by_store_id(self, store_id: str) -> None:
        """Remove an engine service by store id (idempotent).

        Also clears any cooldown left over from a previous failure so an
        operator can retry immediately after removing a store.
        """
        self._by_store_id.pop(store_id, None)
        self._bump_generation_locked(store_id)
        self._failed_until.pop(store_id, None)

    def can_rebuild_stores(self) -> bool:
        """Report whether the registry can rebuild a store engine on demand."""
        return self._repo is not None and self._factory is not None

    # ── generation tracking ─────────────────────────────────────────

    def _bump_generation_locked(self, store_id: str) -> None:
        self._store_gen[store_id] = self._store_gen.get(store_id, 0) + 1

    def _store_generation(self, store_id: str) -> int:
        return self._store_gen.get(store_id, 0)

    def _register_if_gen_unchanged(
        self,
        store_id: str,
        gen: int,
        service: RetrieveEngineService,
    ) -> bool:
        """Publish ``service`` only when the store entry has not moved."""
        if self._store_gen.get(store_id, 0) != gen:
            return False
        self._by_store_id[store_id] = service
        self._failed_until.pop(store_id, None)
        return True

    # ── failure cooldown ────────────────────────────────────────────

    def _in_failure_cooldown(self, store_id: str) -> bool:
        deadline = self._failed_until.get(store_id)
        return deadline is not None and time.monotonic() < deadline

    def _mark_build_failed(self, store_id: str) -> None:
        self._failed_until[store_id] = time.monotonic() + REBUILD_COOLDOWN_SECONDS

    # ── on-demand rebuild ───────────────────────────────────────────

    async def get_or_load_by_store_id(
        self, ctx: Context, tenant_id: int, store_id: str
    ) -> RetrieveEngineService:
        """Return the engine for ``store_id``, rebuilding it when missing.

        Scoped to ``tenant_id`` so a caller reaching this method without an
        ownership check cannot join another tenant's flight.
        """
        try:
            return self.get_by_store_id(store_id)
        except NotFoundError:
            pass
        if self._repo is None or self._factory is None:
            raise VectorStoreNotFoundError()
        if self._in_failure_cooldown(store_id):
            raise VectorStoreUnavailableError()
        # Sampled before the build starts: an unregistration landing after
        # this point must prevent the finished engine from being published.
        gen = self._store_generation(store_id)
        # Key by tenant as well as store so a caller reaching this method
        # without an ownership check cannot join another tenant's flight.
        key = f"{tenant_id}:{store_id}"
        task = self._flights.get(key)
        if task is None or task.done():
            task = asyncio.create_task(self._run_build(ctx, tenant_id, store_id, gen))
            self._flights[key] = task
            task.add_done_callback(partial(self._flight_done, key))
        return await task

    def _flight_done(self, key: str, task: asyncio.Task[RetrieveEngineService]) -> None:
        if self._flights.get(key) is task:
            self._flights.pop(key, None)
        # Consume a failed task's exception even when no caller awaited it so
        # the event loop does not surface a "task exception never retrieved"
        # warning; the cause is logged here.
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("[retriever.registry] engine build failed: {}", exc)

    async def _run_build(
        self,
        ctx: Context,
        tenant_id: int,
        store_id: str,
        gen: int,
    ) -> RetrieveEngineService:
        try:
            return await asyncio.wait_for(
                self._build_engine(ctx, tenant_id, store_id, gen),
                timeout=ENGINE_BUILD_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            self._mark_build_failed(store_id)
            logger.error("[retriever.registry] engine build timed out for store {}", store_id)
            raise VectorStoreUnavailableError() from None

    async def _build_engine(
        self,
        ctx: Context,
        tenant_id: int,
        store_id: str,
        gen: int,
    ) -> RetrieveEngineService:
        try:
            try:
                # An earlier flight for this key may have finished after the
                # miss above.
                return self.get_by_store_id(store_id)
            except NotFoundError:
                pass
            repo = self._repo
            if repo is None:
                raise VectorStoreNotFoundError()
            store = await repo.get_by_id(ctx, tenant_id, store_id)
        except asyncio.CancelledError:
            raise
        except VectorStoreNotFoundError:
            raise
        except Exception as exc:
            logger.error(
                "[retriever.registry] loading store {} for rebuild failed: {}",
                store_id,
                exc,
            )
            raise VectorStoreUnavailableError() from None
        if store is None:
            raise VectorStoreNotFoundError()
        try:
            factory = self._factory
            if factory is None:
                raise VectorStoreNotFoundError()
            service = await factory(ctx, store)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "[retriever.registry] rebuilding engine for store {} failed, "
                "retrying no sooner than {}s: {}",
                store_id,
                REBUILD_COOLDOWN_SECONDS,
                exc,
            )
            self._mark_build_failed(store_id)
            raise VectorStoreUnavailableError() from None
        if not self._register_if_gen_unchanged(store_id, gen, service):
            # The entry changed while this engine was being built, so this
            # one is stale before it is published. Whatever landed instead
            # is authoritative; the caller retries and picks it up.
            raise VectorStoreUnavailableError()
        return service


def new_retrieve_engine_registry(
    repo: VectorStoreRepositoryLike | None,
    factory: EngineFactory | None,
) -> RetrieveEngineRegistry:
    """Create a retrieval engine registry (upstream ``NewRetrieveEngineRegistry``).

    ``repo`` and ``factory`` enable on-demand engine rebuilds; passing
    ``None`` for either leaves ``get_or_load_by_store_id`` a plain lookup.
    """
    return RetrieveEngineRegistry(repo, factory)


__all__ = [
    "ENGINE_BUILD_TIMEOUT_SECONDS",
    "REBUILD_COOLDOWN_SECONDS",
    "RetrieveEngineRegistry",
    "VectorStoreNotFoundError",
    "VectorStoreUnavailableError",
    "new_retrieve_engine_registry",
]
