"""Store-tenant ownership adapter (upstream ``ownership.go``).

``TenantStoreOwnership`` abstracts the lookup used by the KB-to-engine
resolver to verify that a vector store belongs to a tenant. Production
implementations wrap the store repository; tests inject in-memory fakes so
they can cover the ownership branches without touching a database.

``VectorStoreRepoOwnership`` adapts ``VectorStoreRepositoryLike.get_by_id``
to the ownership contract. The repository already scopes to the tenant
(``WHERE id = ? AND tenant_id = ?``), so "exists under this tenant" is
equivalent to "owned by this tenant" — no additional comparison is needed.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.ai.embedding import Context
from src.ai.retrieval.base import VectorStoreRepositoryLike


@runtime_checkable
class TenantStoreOwnership(Protocol):
    """Verifies that a vector store with the given ID is owned by a tenant.

    Returns ``True`` iff the store exists under the tenant. A non-existent
    (but well-formed) store ID returns ``False``. Infrastructure failures
    propagate as exceptions.
    """

    async def store_owned_by(self, ctx: Context, store_id: str, tenant_id: int) -> bool: ...


class VectorStoreRepoOwnership:
    """Adapts a store repository to ``TenantStoreOwnership``."""

    def __init__(self, repo: VectorStoreRepositoryLike) -> None:
        self._repo = repo

    async def store_owned_by(self, ctx: Context, store_id: str, tenant_id: int) -> bool:
        store = await self._repo.get_by_id(ctx, tenant_id, store_id)
        return store is not None


def new_vector_store_repo_ownership(
    repo: VectorStoreRepositoryLike,
) -> TenantStoreOwnership:
    """Create the production ownership adapter backed by a store repository."""
    return VectorStoreRepoOwnership(repo)


__all__ = [
    "TenantStoreOwnership",
    "VectorStoreRepoOwnership",
    "new_vector_store_repo_ownership",
]
