"""KB-to-engine resolution (upstream ``factory.go``).

``create_retrieve_engine_for_kb`` resolves a KB's effective retrieval
engines and wraps them in a ``CompositeRetrieveEngine``. An unbound KB (no
vector store binding) falls back to the tenant's effective engines read
from the context; a store-bound KB is ownership-verified and resolved
through the registry's on-demand load path. ``create_retrieve_engine_from_payload``
is the async-task variant that takes the effective engines explicitly, and
``verify_binding`` exposes the same sentinel hierarchy for callers that
only need the ownership check.

The sentinel exceptions intentionally omit store UUIDs so they cannot leak
enumeration surfaces; structured logs record the tenant/store pair.
"""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

from src.ai.embedding import Context
from src.ai.retrieval.base import RetrieveEngineRegistry
from src.ai.retrieval.composite import (
    CompositeRetrieveEngine,
    EngineInfo,
    new_composite_retrieve_engine,
)
from src.ai.retrieval.ownership import TenantStoreOwnership
from src.ai.retrieval.registry import (
    VectorStoreNotFoundError,
    VectorStoreUnavailableError,
)
from src.ai.retrieval.types import RetrieverEngineParams
from src.app_logging import logger
from src.common.exception import ApplicationError, PermissionDeniedError


class TenantInfoMissingError(ApplicationError):
    """The resolver needs a tenant from context and none is present."""

    code = "tenant_info_missing"
    message = "tenant info not found in context"


class VectorStoreForbiddenError(PermissionDeniedError):
    """The resolved store is not owned by the given tenant.

    Guards against cross-tenant access; callers treat it as non-retryable.
    """

    code = "vector_store_forbidden"
    message = "vector store access denied"


@runtime_checkable
class TenantEnginesCarrier(Protocol):
    """Carries a tenant's effective retriever engine params (structural).

    Satisfied by any tenant model exposing ``get_effective_engines()``,
    which returns the tenant's configured engines or the driver defaults
    when none are configured. The ai layer does not import the tenant
    model; the wiring layer supplies the carrier through the context.
    """

    def get_effective_engines(self) -> list[RetrieverEngineParams]: ...


def tenant_info_from_context(ctx: Context) -> TenantEnginesCarrier | None:
    """Read the tenant carrier from ``ctx``, or ``None`` when absent."""
    carrier = getattr(ctx, "tenant_info", None)
    return carrier if isinstance(carrier, TenantEnginesCarrier) else None


def _classify_lookup_error(exc: Exception) -> Exception:
    """Narrow an engine-lookup failure to what the caller may see.

    Context errors and the store sentinels pass through; anything
    unexpected is reported as retryable, because treating an unknown
    failure as permanent is what silently drops work.
    """
    if isinstance(
        exc,
        (
            TimeoutError,
            VectorStoreNotFoundError,
            VectorStoreUnavailableError,
            VectorStoreForbiddenError,
        ),
    ):
        return exc
    return VectorStoreUnavailableError()


async def create_retrieve_engine_for_kb(
    ctx: Context,
    registry: RetrieveEngineRegistry,
    ownership: TenantStoreOwnership,
    tenant_id: int,
    vector_store_id: str | None,
) -> CompositeRetrieveEngine:
    """Resolve a KB's effective engine(s) into a composite.

    A nil or empty ``vector_store_id`` falls back to the tenant's effective
    engines from context (env-store flow); a bound store is ownership
    verified and resolved through the registry.
    """
    if not vector_store_id:
        tenant_info = tenant_info_from_context(ctx)
        if tenant_info is None:
            raise TenantInfoMissingError()
        return new_composite_retrieve_engine(registry, tenant_info.get_effective_engines())
    return await _resolve_bound_engine(ctx, registry, ownership, tenant_id, vector_store_id)


async def create_retrieve_engine_from_payload(
    ctx: Context,
    registry: RetrieveEngineRegistry,
    ownership: TenantStoreOwnership,
    tenant_id: int,
    effective_engines: list[RetrieverEngineParams],
    vector_store_id: str | None,
) -> CompositeRetrieveEngine:
    """Async-task variant: effective engines come from the payload.

    Does not read tenant info from context because async handlers do not
    populate it. Legacy payloads without a binding decode as nil/empty and
    fall back to the pre-serialized ``effective_engines``.
    """
    if not vector_store_id:
        return new_composite_retrieve_engine(registry, effective_engines)
    return await _resolve_bound_engine(ctx, registry, ownership, tenant_id, vector_store_id)


async def verify_binding(
    ctx: Context,
    registry: RetrieveEngineRegistry,
    ownership: TenantStoreOwnership,
    tenant_id: int,
    store_id: str,
) -> None:
    """Assert that a non-empty ``store_id`` is owned and registered.

    Ownership infrastructure errors propagate verbatim; a store the tenant
    does not own yields ``VectorStoreForbiddenError``; an unregistered
    store yields ``VectorStoreNotFoundError``. Never echoes the store UUID.
    """
    owned = await ownership.store_owned_by(ctx, store_id, tenant_id)
    if not owned:
        raise VectorStoreForbiddenError()
    try:
        await registry.get_or_load_by_store_id(ctx, tenant_id, store_id)
    except (
        asyncio.CancelledError,
        TimeoutError,
        VectorStoreNotFoundError,
        VectorStoreUnavailableError,
    ):
        raise
    except Exception as exc:
        raise _classify_lookup_error(exc) from None


async def _resolve_bound_engine(
    ctx: Context,
    registry: RetrieveEngineRegistry,
    ownership: TenantStoreOwnership,
    tenant_id: int,
    store_id: str,
) -> CompositeRetrieveEngine:
    """Ownership-verified lookup shared by both resolver entry points."""
    try:
        owned = await ownership.store_owned_by(ctx, store_id, tenant_id)
    except (asyncio.CancelledError, TimeoutError):
        raise
    except Exception as exc:
        # Infrastructure failure — record the raw error but do not leak it
        # to the caller. The store itself may be fine, so this is retryable
        # rather than not-found.
        logger.error(
            "[retriever.kb_resolver] ownership lookup failed: tenant={} store={} reason={}",
            tenant_id,
            store_id,
            exc,
        )
        raise VectorStoreUnavailableError() from exc
    if not owned:
        # Cross-tenant attempt (or the store was deleted in the meantime).
        logger.warning(
            "[retriever.kb_resolver] cross-tenant store access attempted: tenant={} store={}",
            tenant_id,
            store_id,
        )
        raise VectorStoreForbiddenError()
    try:
        service = await registry.get_or_load_by_store_id(ctx, tenant_id, store_id)
    except (
        asyncio.CancelledError,
        TimeoutError,
        VectorStoreNotFoundError,
        VectorStoreUnavailableError,
    ):
        raise
    except Exception as exc:
        logger.error(
            "[retriever.kb_resolver] store engine could not be resolved: "
            "tenant={} store={} reason={}",
            tenant_id,
            store_id,
            exc,
        )
        raise VectorStoreUnavailableError() from exc
    # A KB bound to a store uses every retriever type that store supports;
    # binding is an explicit opt-out of tenant-default routing.
    return CompositeRetrieveEngine(
        [EngineInfo(retrieve_engine=service, retriever_types=tuple(service.support()))]
    )


__all__ = [
    "TenantEnginesCarrier",
    "TenantInfoMissingError",
    "VectorStoreForbiddenError",
    "create_retrieve_engine_for_kb",
    "create_retrieve_engine_from_payload",
    "tenant_info_from_context",
    "verify_binding",
]
