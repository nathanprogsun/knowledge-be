"""Tests for KB-to-engine resolution.

Covers the unbound fallback (tenant effective engines from context), the
missing-tenant sentinel, the store-bound ownership-verified path, the
cross-tenant / unregistered / infrastructure-failure error paths, the
payload variant used by async workers, ``verify_binding``, and the
sentinel messages that must not leak store UUIDs. The registry and the
ownership lookup are exercised against in-memory fakes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from src.ai.embedding import Context, TaskContext
from src.ai.retrieval.base import RetrieveEngineService
from src.ai.retrieval.kb_engine_resolver import (
    TenantEnginesCarrier,
    TenantInfoMissingError,
    VectorStoreForbiddenError,
    create_retrieve_engine_for_kb,
    create_retrieve_engine_from_payload,
    tenant_info_from_context,
    verify_binding,
)
from src.ai.retrieval.registry import (
    RetrieveEngineRegistry,
    VectorStoreNotFoundError,
    VectorStoreUnavailableError,
    new_retrieve_engine_registry,
)
from src.ai.retrieval.types import (
    RetrieverEngineParams,
    RetrieverEngineType,
    RetrieverType,
)


class _FakeEngine:
    """Engine service exposing only what resolution consumes."""

    def __init__(self, engine_type: RetrieverEngineType, support: list[RetrieverType]) -> None:
        self._engine_type = engine_type
        self._support = support

    def engine_type(self) -> RetrieverEngineType:
        return self._engine_type

    def support(self) -> list[RetrieverType]:
        return list(self._support)


class _FakeOwnership:
    """In-memory tenant ownership: store_id -> owning tenant_id."""

    def __init__(
        self,
        owned: dict[str, int] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._owned = owned or {}
        self._error = error
        self.calls: list[tuple[str, int]] = []

    async def store_owned_by(self, _ctx: Context, store_id: str, tenant_id: int) -> bool:
        self.calls.append((store_id, tenant_id))
        if self._error is not None:
            raise self._error
        return self._owned.get(store_id) == tenant_id


class _Tenant:
    """Tenant carrier exposing effective retriever engines."""

    def __init__(self, engines: list[RetrieverEngineParams]) -> None:
        self._engines = engines

    def get_effective_engines(self) -> list[RetrieverEngineParams]:
        return self._engines


@dataclass(frozen=True, slots=True)
class _TenantCtx:
    """Context carrying a tenant carrier for the unbound path."""

    is_background_task: bool = False
    tenant_info: _Tenant | None = None


def _params(
    engine_type: RetrieverEngineType, retriever_type: RetrieverType
) -> RetrieverEngineParams:
    return RetrieverEngineParams(retriever_engine_type=engine_type, retriever_type=retriever_type)


def _type_registry(*engines: _FakeEngine) -> RetrieveEngineRegistry:
    registry = new_retrieve_engine_registry(None, None)
    for engine in engines:
        registry.register(cast("RetrieveEngineService", engine))
    return registry


def _store_registry(*pairs: tuple[str, _FakeEngine]) -> RetrieveEngineRegistry:
    registry = new_retrieve_engine_registry(None, None)
    for store_id, engine in pairs:
        registry.register_with_store_id(store_id, cast("RetrieveEngineService", engine))
    return registry


# ── tenant-in-context helper ─────────────────────────────────────────


def test_tenant_info_from_context_returns_carrier_when_present() -> None:
    tenant = _Tenant([])
    carrier = tenant_info_from_context(_TenantCtx(tenant_info=tenant))
    assert isinstance(carrier, TenantEnginesCarrier)
    assert carrier is tenant


def test_tenant_info_from_context_returns_none_when_absent() -> None:
    assert tenant_info_from_context(_TenantCtx()) is None
    # A carrier of the wrong shape is treated as absent.
    assert tenant_info_from_context(TaskContext()) is None


# ── unbound (tenant effective engines) ───────────────────────────────


async def test_create_for_kb_unbound_uses_tenant_effective_engines() -> None:
    es_engine = _FakeEngine(RetrieverEngineType.ELASTICSEARCH, [RetrieverType.KEYWORDS])
    registry = _type_registry(es_engine)
    ownership = _FakeOwnership()
    tenant = _Tenant([_params(RetrieverEngineType.ELASTICSEARCH, RetrieverType.KEYWORDS)])

    composite = await create_retrieve_engine_for_kb(
        _TenantCtx(tenant_info=tenant), registry, ownership, 1, None
    )

    infos = composite._engine_infos
    assert len(infos) == 1
    assert infos[0].retrieve_engine is cast("RetrieveEngineService", es_engine)
    assert infos[0].retriever_types == (RetrieverType.KEYWORDS,)
    # The unbound path must not consult ownership.
    assert ownership.calls == []


async def test_create_for_kb_empty_store_id_treated_as_unbound() -> None:
    es_engine = _FakeEngine(RetrieverEngineType.ELASTICSEARCH, [RetrieverType.KEYWORDS])
    registry = _type_registry(es_engine)
    ownership = _FakeOwnership()
    tenant = _Tenant([_params(RetrieverEngineType.ELASTICSEARCH, RetrieverType.KEYWORDS)])

    composite = await create_retrieve_engine_for_kb(
        _TenantCtx(tenant_info=tenant), registry, ownership, 1, ""
    )

    assert composite._engine_infos[0].retrieve_engine is cast("RetrieveEngineService", es_engine)
    assert ownership.calls == []


async def test_create_for_kb_unbound_missing_tenant_raises() -> None:
    registry = _type_registry()
    ownership = _FakeOwnership()

    with pytest.raises(TenantInfoMissingError):
        await create_retrieve_engine_for_kb(TaskContext(), registry, ownership, 1, None)


# ── store-bound (ownership verified) ─────────────────────────────────


async def test_create_for_kb_store_bound() -> None:
    es_engine = _FakeEngine(
        RetrieverEngineType.ELASTICSEARCH,
        [RetrieverType.KEYWORDS, RetrieverType.VECTOR],
    )
    registry = _store_registry(("store-A", es_engine))
    ownership = _FakeOwnership(owned={"store-A": 1})

    composite = await create_retrieve_engine_for_kb(
        TaskContext(), registry, ownership, 1, "store-A"
    )

    infos = composite._engine_infos
    assert len(infos) == 1
    assert infos[0].retrieve_engine is cast("RetrieveEngineService", es_engine)
    # A bound KB uses every retriever type the store supports.
    assert infos[0].retriever_types == (
        RetrieverType.KEYWORDS,
        RetrieverType.VECTOR,
    )


async def test_create_for_kb_cross_tenant_forbidden() -> None:
    es_engine = _FakeEngine(RetrieverEngineType.ELASTICSEARCH, [RetrieverType.VECTOR])
    registry = _store_registry(("store-A", es_engine))
    # store-A is owned by tenant 2, not tenant 1.
    ownership = _FakeOwnership(owned={"store-A": 2})

    with pytest.raises(VectorStoreForbiddenError) as excinfo:
        await create_retrieve_engine_for_kb(TaskContext(), registry, ownership, 1, "store-A")
    # The sentinel must not expose the store UUID.
    assert "store-A" not in str(excinfo.value)


async def test_create_for_kb_unregistered_store_not_found() -> None:
    # Ownership says the tenant owns the store, but the registry has not
    # loaded it (store row exists, engine init failed at startup).
    registry = _type_registry()
    ownership = _FakeOwnership(owned={"store-A": 1})

    with pytest.raises(VectorStoreNotFoundError):
        await create_retrieve_engine_for_kb(TaskContext(), registry, ownership, 1, "store-A")


async def test_create_for_kb_ownership_infra_error_is_unavailable() -> None:
    es_engine = _FakeEngine(RetrieverEngineType.ELASTICSEARCH, [RetrieverType.VECTOR])
    registry = _store_registry(("store-A", es_engine))
    ownership = _FakeOwnership(error=RuntimeError("db connection refused"))

    with pytest.raises(VectorStoreUnavailableError) as excinfo:
        await create_retrieve_engine_for_kb(TaskContext(), registry, ownership, 1, "store-A")
    # A database failure says nothing about whether the store exists, so it
    # must be the retryable sentinel, never the permanent not-found one.
    assert not isinstance(excinfo.value, VectorStoreNotFoundError)


async def test_create_for_kb_ownership_context_error_propagates() -> None:
    es_engine = _FakeEngine(RetrieverEngineType.ELASTICSEARCH, [RetrieverType.VECTOR])
    registry = _store_registry(("store-A", es_engine))
    ownership = _FakeOwnership(error=TimeoutError("deadline exceeded"))

    with pytest.raises(TimeoutError, match="deadline exceeded"):
        await create_retrieve_engine_for_kb(TaskContext(), registry, ownership, 1, "store-A")


# ── payload variant (async workers) ──────────────────────────────────


async def test_create_engine_from_payload_unbound_uses_payload_engines() -> None:
    postgres_engine = _FakeEngine(RetrieverEngineType.POSTGRES, [RetrieverType.VECTOR])
    registry = _type_registry(postgres_engine)
    ownership = _FakeOwnership()
    engines = [_params(RetrieverEngineType.POSTGRES, RetrieverType.VECTOR)]

    composite = await create_retrieve_engine_from_payload(
        TaskContext(), registry, ownership, 1, engines, None
    )

    assert composite._engine_infos[0].retrieve_engine is cast(
        "RetrieveEngineService", postgres_engine
    )
    assert ownership.calls == []


async def test_create_engine_from_payload_bound() -> None:
    qdrant_engine = _FakeEngine(RetrieverEngineType.QDRANT, [RetrieverType.VECTOR])
    registry = _store_registry(("qd-1", qdrant_engine))
    ownership = _FakeOwnership(owned={"qd-1": 42})

    composite = await create_retrieve_engine_from_payload(
        TaskContext(), registry, ownership, 42, [], "qd-1"
    )

    assert composite._engine_infos[0].retrieve_engine is cast(
        "RetrieveEngineService", qdrant_engine
    )


async def test_create_engine_from_payload_tampered_cross_tenant() -> None:
    # The store is owned by tenant 99, but the (possibly tampered) payload
    # claims tenant 1 — the factory must reject.
    es_engine = _FakeEngine(RetrieverEngineType.ELASTICSEARCH, [RetrieverType.VECTOR])
    registry = _store_registry(("store-A", es_engine))
    ownership = _FakeOwnership(owned={"store-A": 99})

    with pytest.raises(VectorStoreForbiddenError):
        await create_retrieve_engine_from_payload(
            TaskContext(), registry, ownership, 1, [], "store-A"
        )


# ── verify_binding ───────────────────────────────────────────────────


async def test_verify_binding_ownership_infra_error_is_verbatim() -> None:
    registry = _type_registry()
    ownership = _FakeOwnership(error=RuntimeError("db boom"))

    with pytest.raises(RuntimeError, match="db boom"):
        await verify_binding(TaskContext(), registry, ownership, 1, "store-A")


async def test_verify_binding_not_owned_is_forbidden() -> None:
    registry = _type_registry()
    ownership = _FakeOwnership(owned={})

    with pytest.raises(VectorStoreForbiddenError):
        await verify_binding(TaskContext(), registry, ownership, 1, "store-A")


async def test_verify_binding_owned_but_unregistered_is_not_found() -> None:
    registry = _type_registry()
    ownership = _FakeOwnership(owned={"store-A": 1})

    with pytest.raises(VectorStoreNotFoundError):
        await verify_binding(TaskContext(), registry, ownership, 1, "store-A")


async def test_verify_binding_owned_and_registered_succeeds() -> None:
    es_engine = _FakeEngine(RetrieverEngineType.ELASTICSEARCH, [RetrieverType.VECTOR])
    registry = _store_registry(("store-A", es_engine))
    ownership = _FakeOwnership(owned={"store-A": 1})

    await verify_binding(TaskContext(), registry, ownership, 1, "store-A")


async def test_verify_binding_cross_tenant_is_forbidden_not_found() -> None:
    # Ownership is checked before the registry lookup, so a store owned by a
    # different tenant yields Forbidden even when it is registered.
    es_engine = _FakeEngine(RetrieverEngineType.ELASTICSEARCH, [RetrieverType.VECTOR])
    registry = _store_registry(("store-A", es_engine))
    ownership = _FakeOwnership(owned={"store-A": 2})

    with pytest.raises(VectorStoreForbiddenError):
        await verify_binding(TaskContext(), registry, ownership, 1, "store-A")


# ── sentinel messages ────────────────────────────────────────────────


def test_sentinel_messages_do_not_leak_store_ids() -> None:
    assert str(VectorStoreForbiddenError()) == "vector store access denied"
    assert str(VectorStoreNotFoundError()) == "vector store not available"
    assert str(VectorStoreUnavailableError()) == "vector store engine unavailable"
    assert str(TenantInfoMissingError()) == "tenant info not found in context"
