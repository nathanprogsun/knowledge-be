"""Web-search-provider persistence — raw SQL only, no ORM.

Maps the methods declared in the upstream
``internal/types/interfaces/web_search_provider.go::WebSearchProviderRepository``
interface. Each method scopes by ``tenant_id`` so a caller can never
read or mutate another workspace's rows.

Reads filter ``deleted_at IS NULL`` (via the base ``GenericRepository``
helpers) so a soft-deleted row behaves as if it no longer exists.

``ClearDefault`` flips ``is_default`` to ``false`` on every live row of
a tenant, optionally excluding one — the service uses it when promoting
a new default so the workspace never has two simultaneous defaults.
"""

from __future__ import annotations

from typing import cast

from sqlalchemy import text
from sqlalchemy.engine import CursorResult

from src.common.json import BindParams, SqlValue
from src.db.dao.generic_repository import GenericRepository
from src.db.models.infra.web_search_provider import WebSearchProvider


class WebSearchProviderRepository(GenericRepository[WebSearchProvider]):
    """`web_search_providers`-table SQL — tenant-scoped CRUD + default flip."""

    model_class = WebSearchProvider

    # ── Reads ───────────────────────────────────────────────────────

    async def get_by_id(
        self,
        tenant_id: int,
        provider_id: str,
    ) -> WebSearchProvider | None:
        """Return one live provider by primary key + tenant scope."""
        return await self.find_unique_by_column_values(
            {"id": provider_id, "tenant_id": tenant_id},
        )

    async def get_default(self, tenant_id: int) -> WebSearchProvider | None:
        """Return the tenant's default provider (is_default=true) or ``None``."""
        return await self.find_unique_by_column_values(
            {"tenant_id": tenant_id, "is_default": True},
        )

    async def list_for_tenant(self, tenant_id: int) -> list[WebSearchProvider]:
        """Return every live provider of the tenant, oldest first."""
        stmt = text(
            "select * from web_search_providers "
            "where tenant_id = :tenant_id and deleted_at is null "
            "order by created_at asc"
        ).bindparams(tenant_id=tenant_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    # ── Mutations ───────────────────────────────────────────────────

    async def clear_default(
        self,
        tenant_id: int,
        exclude_id: str = "",
    ) -> int:
        """Clear ``is_default`` on every live row of ``tenant_id``.

        ``exclude_id`` is optional — when set, the row with that id is
        skipped so the caller can promote it without a TOCTOU window
        where no row holds the flag.

        Returns the row count flipped.
        """
        params: BindParams = {"tenant_id": tenant_id}
        where = "tenant_id = :tenant_id and is_default = true and deleted_at is null"
        if exclude_id:
            where += " and id != :exclude_id"
            params["exclude_id"] = exclude_id
        stmt = text(f"update web_search_providers set is_default = false where {where}").bindparams(
            **params
        )
        result = await self._session.execute(stmt)
        return int(cast("CursorResult[SqlValue]", result).rowcount or 0)


__all__ = ["WebSearchProviderRepository"]
