"""Tests for the store-tenant ownership adapter.

Covers ownership resolution (store exists vs. missing), tenant scoping of
the underlying ``get_by_id`` call, propagation of infrastructure failures,
and the structural shape of the adapter. The store repository is faked; no
database is contacted.
"""

from __future__ import annotations

from typing import cast

import pytest

from src.ai.embedding import Context, TaskContext
from src.ai.retrieval.base import VectorStoreRepositoryLike
from src.ai.retrieval.ownership import (
    TenantStoreOwnership,
    VectorStoreRepoOwnership,
    new_vector_store_repo_ownership,
)
from src.ai.retrieval.types import RetrieverEngineType, VectorStore, VectorStoreLike

_CTX = TaskContext()


class _FakeRepo:
    """Store repository returning a fixed store (or raising)."""

    def __init__(
        self, store: VectorStoreLike | None, error: Exception | None = None
    ) -> None:
        self._store = store
        self._error = error
        self.calls: list[tuple[int, str]] = []

    async def get_by_id(
        self, _ctx: Context, tenant_id: int, store_id: str
    ) -> VectorStoreLike | None:
        self.calls.append((tenant_id, store_id))
        if self._error is not None:
            raise self._error
        return self._store


def _store(store_id: str = "store-1") -> VectorStore:
    return VectorStore(
        id=store_id,
        tenant_id=1,
        name="Store",
        engine_type=RetrieverEngineType.QDRANT,
    )


def _repo(
    store: VectorStoreLike | None, error: Exception | None = None
) -> _FakeRepo:
    return _FakeRepo(store, error)


# ── ownership resolution ─────────────────────────────────────────────


async def test_store_owned_by_true_when_store_exists() -> None:
    repo = _repo(_store())
    ownership = new_vector_store_repo_ownership(cast("VectorStoreRepositoryLike", repo))

    assert await ownership.store_owned_by(_CTX, "store-1", 1) is True
    assert repo.calls == [(1, "store-1")]


async def test_store_owned_by_false_when_store_missing() -> None:
    repo = _repo(None)
    ownership = new_vector_store_repo_ownership(cast("VectorStoreRepositoryLike", repo))

    assert await ownership.store_owned_by(_CTX, "store-1", 1) is False
    assert repo.calls == [(1, "store-1")]


async def test_store_owned_by_scopes_lookup_by_tenant() -> None:
    # The repository's get_by_id is tenant-scoped; ownership must forward the
    # requested tenant so a store is never resolved across tenants.
    repo = _repo(_store())
    ownership = new_vector_store_repo_ownership(cast("VectorStoreRepositoryLike", repo))

    await ownership.store_owned_by(_CTX, "store-9", 42)
    assert repo.calls == [(42, "store-9")]


async def test_store_owned_by_propagates_infrastructure_errors() -> None:
    repo = _repo(None, error=RuntimeError("db connection refused"))
    ownership = new_vector_store_repo_ownership(cast("VectorStoreRepositoryLike", repo))

    with pytest.raises(RuntimeError, match="db connection refused"):
        await ownership.store_owned_by(_CTX, "store-1", 1)


def test_new_vector_store_repo_ownership_returns_ownership_contract() -> None:
    adapter = new_vector_store_repo_ownership(cast("VectorStoreRepositoryLike", _repo(_store())))
    assert isinstance(adapter, TenantStoreOwnership)
    assert isinstance(adapter, VectorStoreRepoOwnership)


async def test_adapts_any_get_by_id_shape() -> None:
    # The adapter is structural: any object exposing the tenant-scoped
    # get_by_id satisfies the repository seam.
    class _AdHocRepo:
        async def get_by_id(
            self, _ctx: Context, tenant_id: int, store_id: str
        ) -> VectorStoreLike | None:
            return _store(store_id)

    ownership = new_vector_store_repo_ownership(cast("VectorStoreRepositoryLike", _AdHocRepo()))
    assert await ownership.store_owned_by(_CTX, "store-7", 3) is True
