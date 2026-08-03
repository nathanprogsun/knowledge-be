"""Tenant service — workspace lifecycle.

Create, read (single / batch / list / search), update, soft delete, and
the two storage-counter operations. Constructed per request; the
repository owns the per-request session.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.common.exception import NotFoundError, ValidationError
from src.core.tenants.types import TenantInfo
from src.db.dao.tenants_repository import TenantRepository
from src.db.models.tenants.tenants import DEFAULT_STORAGE_QUOTA_BYTES, Tenant

# A fresh workspace starts in this status, regardless of caller input.
_INITIAL_STATUS = "active"

# Default page size for the search endpoint.
_DEFAULT_PAGE_SIZE = 20


class TenantService:
    """Stateless tenant service, constructed per request."""

    def __init__(self, *, tenants_repo: TenantRepository) -> None:
        self._tenants_repo = tenants_repo

    # ── Create ──────────────────────────────────────────────────────

    async def create_tenant(
        self,
        *,
        name: str,
        description: str | None = None,
        business: str = "",
        retriever_engines: dict[str, object] | list[dict[str, object]] | None = None,
        storage_quota: int | None = None,
    ) -> TenantInfo:
        """Create a workspace in ``active`` status.

        Name is required; status is server-assigned; no API key is
        minted (integrations create keys through ``tenant_api_keys``).
        """
        clean_name = name.strip()
        if not clean_name:
            raise ValidationError(
                code="tenant.name_required",
                message="Workspace name cannot be empty",
            )
        now = datetime.now(UTC)
        row = Tenant(
            name=clean_name,
            description=description,
            business=business,
            status=_INITIAL_STATUS,
            retriever_engines=retriever_engines if retriever_engines is not None else {},
            storage_quota=(
                storage_quota if storage_quota is not None else DEFAULT_STORAGE_QUOTA_BYTES
            ),
            created_at=now,
            updated_at=now,
        )
        return TenantInfo.map_from_db(await self._tenants_repo.insert(row))

    # ── Read ────────────────────────────────────────────────────────

    async def get_tenant(self, tenant_id: int) -> TenantInfo:
        """Return one workspace; raise ``tenant.not_found`` when absent."""
        self._require_valid_id(tenant_id)
        return TenantInfo.map_from_db(await self._tenants_repo.find_by_id(tenant_id))

    async def get_tenants(self, tenant_ids: list[int]) -> dict[int, TenantInfo]:
        """Batch-read workspaces, keyed by id.

        Ids with no live row are absent from the result rather than an
        error.
        """
        rows = await self._tenants_repo.find_by_ids(tenant_ids)
        return {row.id: TenantInfo.map_from_db(row) for row in rows}

    async def list_tenants(self) -> list[TenantInfo]:
        """Return every live workspace, newest first."""
        rows = await self._tenants_repo.list_all()
        return [TenantInfo.map_from_db(row) for row in rows]

    async def search_tenants(
        self,
        *,
        keyword: str | None = None,
        tenant_id: int | None = None,
        page: int = 1,
        page_size: int = _DEFAULT_PAGE_SIZE,
    ) -> tuple[list[TenantInfo], int]:
        """Search workspaces; return one page plus the unpaginated total.

        Pagination applies only when both ``page`` and ``page_size`` are
        positive; otherwise every match is returned.
        """
        paginate = page > 0 and page_size > 0
        rows, total = await self._tenants_repo.search(
            keyword=keyword,
            tenant_id=tenant_id,
            limit=page_size if paginate else None,
            offset=(page - 1) * page_size if paginate else 0,
        )
        return [TenantInfo.map_from_db(row) for row in rows], total

    # ── Update ──────────────────────────────────────────────────────

    async def update_tenant(
        self,
        tenant_id: int,
        *,
        name: str | None = None,
        description: str | None = None,
        business: str | None = None,
        retriever_engines: dict[str, object] | list[dict[str, object]] | None = None,
        storage_quota: int | None = None,
        status: str | None = None,
    ) -> TenantInfo:
        """Patch the supplied columns and stamp ``updated_at``.

        Omitted arguments are left untouched.
        """
        self._require_valid_id(tenant_id)
        columns = self._build_update_columns(
            name=name,
            description=description,
            business=business,
            retriever_engines=retriever_engines,
            storage_quota=storage_quota,
            status=status,
        )
        columns["updated_at"] = datetime.now(UTC)
        row = await self._tenants_repo.update_by_primary_key({"id": tenant_id}, columns)
        if row is None:
            raise NotFoundError(
                code="tenant.not_found",
                message=f"Tenant {tenant_id} not found",
            )
        return TenantInfo.map_from_db(row)

    @staticmethod
    def _build_update_columns(
        *,
        name: str | None,
        description: str | None,
        business: str | None,
        retriever_engines: dict[str, object] | list[dict[str, object]] | None,
        storage_quota: int | None,
        status: str | None,
    ) -> dict[str, object]:
        """Collect the supplied columns, validating the ones with rules."""
        columns: dict[str, object] = {}
        if name is not None:
            clean_name = name.strip()
            if not clean_name:
                raise ValidationError(
                    code="tenant.name_required",
                    message="Workspace name cannot be empty",
                )
            columns["name"] = clean_name
        if description is not None:
            columns["description"] = description
        if business is not None:
            columns["business"] = business
        if retriever_engines is not None:
            columns["retriever_engines"] = retriever_engines
        if storage_quota is not None:
            columns["storage_quota"] = storage_quota
        if status is not None:
            columns["status"] = status
        return columns

    # ── Delete ──────────────────────────────────────────────────────

    async def delete_tenant(self, tenant_id: int) -> bool:
        """Soft-delete a workspace; return whether a row was deleted.

        Idempotent: deleting an unknown or already-deleted workspace is
        not an error, it just reports ``False``.
        """
        self._require_valid_id(tenant_id)
        now = datetime.now(UTC)
        row = await self._tenants_repo.update_by_primary_key(
            {"id": tenant_id},
            {"deleted_at": now, "updated_at": now},
        )
        return row is not None

    # ── Storage counters ────────────────────────────────────────────

    async def adjust_storage_used(self, tenant_id: int, *, delta: int) -> int:
        """Add ``delta`` bytes to the workspace's used storage."""
        self._require_valid_id(tenant_id)
        return await self._tenants_repo.adjust_storage_used(
            tenant_id,
            delta=delta,
            updated_at=datetime.now(UTC),
        )

    async def bulk_set_storage_quota(self, *, quota_bytes: int) -> int:
        """Apply one quota to every workspace; return the row count.

        A non-positive quota is rejected so an operator default is never
        interpreted as "unlimited".
        """
        if quota_bytes <= 0:
            raise ValidationError(
                code="tenant.invalid_quota",
                message="Quota must be positive",
            )
        return await self._tenants_repo.bulk_set_storage_quota(
            quota_bytes=quota_bytes,
            updated_at=datetime.now(UTC),
        )

    # ── Internal helpers ────────────────────────────────────────────

    @staticmethod
    def _require_valid_id(tenant_id: int) -> None:
        """Reject a non-positive id."""
        if tenant_id <= 0:
            raise ValidationError(
                code="tenant.invalid_id",
                message="Tenant ID cannot be 0",
            )


__all__ = ["TenantService"]
