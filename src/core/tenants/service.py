"""Tenant service — workspace lifecycle.

Create, read (single / batch / list / search), update, soft delete, and
the two storage-counter operations. Constructed per request; the
repository owns the per-request session.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

from src.common.exception import NotFoundError, ValidationError
from src.common.json import BindParams, JsonObject, JsonValue
from src.core.tenants.types import TenantInfo
from src.db.dao.tenant_members_repository import TenantMemberRepository
from src.db.dao.tenants_repository import TenantRepository
from src.db.models.tenants.tenants import DEFAULT_STORAGE_QUOTA_BYTES, Tenant

# A fresh workspace starts in this status, regardless of caller input.
_INITIAL_STATUS = "active"

# API-principal authentication strategies (Go ``UpdateAPIPrincipalConfig``).
_PRINCIPAL_MODES: frozenset[str] = frozenset({"tenant", "direct_header", "signed_token"})

# Redaction placeholder an update request may re-submit to mean "keep
# the existing secret" (Go ``apiPrincipalSecretRedacted``).
_PRINCIPAL_SECRET_REDACTED = "***"

# Default page size for the search endpoint.
_DEFAULT_PAGE_SIZE = 20


class TenantService:
    """Stateless tenant service, constructed per request."""

    def __init__(
        self,
        *,
        tenants_repo: TenantRepository,
        members_repo: TenantMemberRepository | None = None,
    ) -> None:
        self._tenants_repo = tenants_repo
        self._members_repo = members_repo

    # ── Create ──────────────────────────────────────────────────────

    async def create_tenant(
        self,
        *,
        name: str,
        description: str | None = None,
        business: str = "",
        retriever_engines: JsonObject | list[JsonObject] | None = None,
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
        retriever_engines: JsonObject | list[JsonObject] | None = None,
        storage_quota: int | None = None,
        status: str | None = None,
        credentials: JsonObject | None = None,
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
            credentials=credentials,
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
        retriever_engines: JsonObject | list[JsonObject] | None,
        storage_quota: int | None,
        status: str | None,
        credentials: JsonObject | None = None,
    ) -> BindParams:
        """Collect the supplied columns, validating the ones with rules."""
        columns: BindParams = {}
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
            columns["retriever_engines"] = cast(JsonValue, retriever_engines)
        if credentials is not None:
            columns["credentials"] = cast(JsonValue, credentials)
        if storage_quota is not None:
            columns["storage_quota"] = storage_quota
        if status is not None:
            columns["status"] = status
        return columns

    # ── Delete ──────────────────────────────────────────────────────

    async def delete_tenant(self, tenant_id: int) -> bool:
        """Soft-delete a workspace; return whether a row was deleted.

        Idempotent: deleting an unknown or already-deleted workspace is
        not an error, it just reports ``False``. Memberships are
        soft-deleted in the same pass — Go's repository deletes
        ``TenantMember`` rows in the same transaction, and leaving them
        would surface the deleted workspace in ``/auth/me``.
        """
        self._require_valid_id(tenant_id)
        now = datetime.now(UTC)
        row = await self._tenants_repo.update_by_primary_key(
            {"id": tenant_id},
            {"deleted_at": now, "updated_at": now},
        )
        if row is not None and self._members_repo is not None:
            await self._members_repo.soft_delete_by_tenant(tenant_id=tenant_id, deleted_at=now)
        return row is not None

    # ── Storage counters ────────────────────────────────────────────

    async def get_api_principal_config(self, tenant_id: int) -> JsonObject | None:
        """Return the workspace's API-principal config, or ``None`` when unset."""
        self._require_valid_id(tenant_id)
        row = await self._tenants_repo.find_by_id(tenant_id)
        return row.api_principal_config

    async def update_api_principal_config(
        self,
        tenant_id: int,
        *,
        config: JsonObject,
    ) -> JsonObject:
        """Persist the workspace's API-principal config; return the stored value."""
        self._require_valid_id(tenant_id)
        # Mirrors Go's ``UpdateAPIPrincipalConfig`` guards: the mode must
        # be one of the three known strategies, and ``signed_token``
        # requires the HMAC secret to be supplied.
        mode = config.get("mode")
        if mode is not None and mode not in _PRINCIPAL_MODES:
            raise ValidationError(
                code="tenant.principal_mode_invalid",
                message="mode must be tenant, direct_header, or signed_token",
            )
        # Resolve the HMAC secret: an omitted value or the ``"***"``
        # redaction placeholder keeps the stored secret (Go
        # ``apiPrincipalSecretRedacted``); anything else replaces it.
        existing = await self._tenants_repo.find_by_id(tenant_id)
        existing_secret = ""
        if existing is not None and existing.api_principal_config:
            raw = existing.api_principal_config.get("hmac_secret")
            if isinstance(raw, str):
                existing_secret = raw
        provided = config.get("hmac_secret")
        hmac_secret = existing_secret
        if isinstance(provided, str) and provided.strip() and provided.strip() != _PRINCIPAL_SECRET_REDACTED:
            hmac_secret = provided.strip()
        if mode == "signed_token" and not hmac_secret:
            raise ValidationError(
                code="tenant.principal_hmac_required",
                message="hmac_secret is required for signed_token mode",
            )
        persisted_config = {**config, "hmac_secret": hmac_secret}
        updated = await self._tenants_repo.update_by_primary_key(
            {"id": tenant_id},
            {
                "api_principal_config": persisted_config,
                "updated_at": datetime.now(UTC),
            },
        )
        if updated is None:
            raise NotFoundError(
                code="tenant.not_found",
                message=f"Tenant {tenant_id} not found",
            )
        stored_config = updated.api_principal_config
        if stored_config is None:
            raise NotFoundError(
                code="tenant.principal_config_unset",
                message="Tenant principal config is not set",
            )
        return stored_config

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
