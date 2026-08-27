"""Organization persistence — raw SQL only, no ORM.

Three tables back the cross-tenant sharing domain:

- ``organizations`` — the collaboration space itself. Reads filter
  soft-deleted rows (``deleted_at is null``).
- ``organization_tenant_members`` — tenant-scoped membership rows.
  The (org, tenant) tuple is unique; the table has no soft-delete
  column, so removal is a hard DELETE.
- ``organization_join_requests`` — join / role-upgrade requests.
  Also hard-deleted; ``status`` carries the lifecycle.

Every query is ``sqlalchemy.text()`` with named ``bindparams``; the
only values interpolated into SQL strings are module-level table-name
constants, so user input never reaches the statement text.
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from sqlalchemy import CursorResult, text

from src.common.exception import DataError
from src.common.json import BindParams, SqlValue
from src.db.dao.generic_repository import GenericRepository
from src.db.models.organization import (
    JOIN_REQUEST_STATUS_PENDING,
    Organization,
    OrganizationJoinRequest,
    OrganizationTenantMember,
)

_LIVE = "deleted_at is null"

# Newest first: the management UI and the discovery list both read as a
# reverse-chronological feed.
_ORGANIZATION_ORDER = "created_at desc, id desc"

# Memberships read oldest-first (the Go side lists join order).
_MEMBER_ORDER = "created_at asc, id asc"

# Join requests read newest-first (the review inbox).
_JOIN_REQUEST_ORDER = "created_at desc, id desc"

# Module-level aliases for the three table names. Every
# ``text(f"...{...}")`` in this file interpolates one of these
# constants; user input never reaches the SQL string.
_ORGANIZATION_TABLE = "organizations"
_ORG_MEMBER_TABLE = "organization_tenant_members"
_JOIN_REQUEST_TABLE = "organization_join_requests"


class OrganizationRepository(GenericRepository[Organization]):
    """`organizations`-table SQL — CRUD + invite-code + discovery reads."""

    model_class = Organization

    # ── Writes ──────────────────────────────────────────────────────

    async def create(self, row: Organization) -> Organization:
        """Insert an organization and return the persisted row."""
        return await self.insert(row)

    async def update(self, row: Organization) -> Organization:
        """Overwrite every mutable column of the row, returning the result.

        ``id`` / ``owner_id`` / ``owner_tenant_id`` / ``created_at`` are
        immutable by contract — ``owner_tenant_id`` is pinned at
        creation time so the owning workspace can never be orphaned — so
        they stay out of the SET clause.
        """
        immutable = {"id", "owner_id", "owner_tenant_id", "created_at"}
        updates = {k: v for k, v in row.model_dump().items() if k not in immutable}
        persisted = await self.update_by_primary_key({"id": row.id}, updates)
        if persisted is None:
            raise DataError(
                code="organization.update_no_row",
                message=f"organization {row.id} not found for update",
            )
        return persisted

    async def soft_delete(self, *, id: str, now: datetime) -> bool:
        """Mark the row deleted. Returns whether a live row was affected."""
        stmt = text(
            f"update {_ORGANIZATION_TABLE} set deleted_at = :now, updated_at = :now "
            f"where id = :id and {_LIVE}"
        ).bindparams(id=id, now=now)
        result = cast("CursorResult[SqlValue]", await self._session.execute(stmt))
        return (result.rowcount or 0) > 0

    async def update_invite_code(
        self,
        *,
        id: str,
        invite_code: str | None,
        expires_at: datetime | None,
        now: datetime,
    ) -> bool:
        """Set the invite code and optional expiry. Returns rows affected."""
        stmt = text(
            f"update {_ORGANIZATION_TABLE} "
            "set invite_code = :invite_code, invite_code_expires_at = :expires_at, "
            "updated_at = :now "
            f"where id = :id and {_LIVE}"
        ).bindparams(
            id=id,
            invite_code=invite_code,
            expires_at=expires_at,
            now=now,
        )
        result = cast("CursorResult[SqlValue]", await self._session.execute(stmt))
        return (result.rowcount or 0) > 0

    # ── Reads ───────────────────────────────────────────────────────

    async def get_by_id_or_none(self, id: str) -> Organization | None:
        """Return the live row for ``id``, or ``None`` when absent."""
        return await self.find_by_primary_key({"id": id})

    async def get_by_invite_code_or_none(self, invite_code: str) -> Organization | None:
        """Return the live row carrying ``invite_code``, or ``None``."""
        if not invite_code:
            return None
        return await self.find_unique_by_column_values({"invite_code": invite_code})

    async def list_by_tenant(self, tenant_id: int) -> list[Organization]:
        """Every organization the tenant participates in, newest first."""
        stmt = text(
            f"select o.* from {_ORGANIZATION_TABLE} o "
            f"join {_ORG_MEMBER_TABLE} m on m.organization_id = o.id "
            f"where m.tenant_id = :tenant_id and o.{_LIVE} "
            f"order by o.{_ORGANIZATION_ORDER}"
        ).bindparams(tenant_id=tenant_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def list_searchable(
        self,
        *,
        query: str,
        limit: int,
    ) -> list[Organization]:
        """Discoverable organizations (``searchable = true``), newest first.

        ``query`` optionally narrows by name, description, or id
        substring. ``limit`` is clamped to a sane page size.
        """
        page_size = limit if limit > 0 else 20
        stmt = text(
            f"select * from {_ORGANIZATION_TABLE} "
            f"where searchable = true and {_LIVE} "
            "and (name ilike :pattern or description ilike :pattern or id::text ilike :pattern) "
            f"order by {_ORGANIZATION_ORDER} limit :limit"
        ).bindparams(pattern=f"%{query}%", limit=page_size)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]


class OrganizationMemberRepository(GenericRepository[OrganizationTenantMember]):
    """`organization_tenant_members`-table SQL — tenant-scoped membership."""

    model_class = OrganizationTenantMember

    # ── Writes ──────────────────────────────────────────────────────

    async def add_member(self, row: OrganizationTenantMember) -> OrganizationTenantMember | None:
        """Insert a membership row, returning ``None`` on a duplicate.

        The conflict target matches the unique index on
        ``(organization_id, tenant_id)``, so a second row for the same
        (org, tenant) tuple is suppressed — the service layer treats
        that as an idempotent no-op.
        """
        return await self.insert_or_none(
            row,
            on_conflict_do_nothing_target_columns=["organization_id", "tenant_id"],
        )

    async def remove_member(self, *, organization_id: str, tenant_id: int) -> bool:
        """Delete the (org, tenant) membership row. Returns rows affected."""
        stmt = text(
            f"delete from {_ORG_MEMBER_TABLE} "
            "where organization_id = :organization_id and tenant_id = :tenant_id"
        ).bindparams(organization_id=organization_id, tenant_id=tenant_id)
        result = cast("CursorResult[SqlValue]", await self._session.execute(stmt))
        return (result.rowcount or 0) > 0

    async def update_member_role(
        self,
        *,
        organization_id: str,
        tenant_id: int,
        role: str,
    ) -> bool:
        """Set the role for a (org, tenant) membership. Returns rows affected."""
        stmt = text(
            f"update {_ORG_MEMBER_TABLE} set role = :role "
            "where organization_id = :organization_id and tenant_id = :tenant_id"
        ).bindparams(organization_id=organization_id, tenant_id=tenant_id, role=role)
        result = cast("CursorResult[SqlValue]", await self._session.execute(stmt))
        return (result.rowcount or 0) > 0

    # ── Reads ───────────────────────────────────────────────────────

    async def get_member(
        self,
        *,
        organization_id: str,
        tenant_id: int,
    ) -> OrganizationTenantMember | None:
        """Return the (org, tenant) membership row, or ``None`` when missing."""
        return await self.find_unique_by_column_values(
            {"organization_id": organization_id, "tenant_id": tenant_id},
            exclude_deleted_or_archived=False,
        )

    async def list_members(self, organization_id: str) -> list[OrganizationTenantMember]:
        """Every membership of one organization, oldest first."""
        stmt = text(
            f"select * from {_ORG_MEMBER_TABLE} "
            f"where organization_id = :organization_id order by {_MEMBER_ORDER}"
        ).bindparams(organization_id=organization_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def count_members(self, organization_id: str) -> int:
        """Count membership rows of one organization."""
        stmt = text(
            f"select count(*) from {_ORG_MEMBER_TABLE} where organization_id = :organization_id"
        ).bindparams(organization_id=organization_id)
        return int((await self._session.execute(stmt)).scalar_one())


class OrganizationJoinRequestRepository(GenericRepository[OrganizationJoinRequest]):
    """`organization_join_requests`-table SQL — join / upgrade requests."""

    model_class = OrganizationJoinRequest

    # ── Writes ──────────────────────────────────────────────────────

    async def create_join_request(self, row: OrganizationJoinRequest) -> OrganizationJoinRequest:
        """Insert a join request and return the persisted row."""
        return await self.insert(row)

    async def update_join_request_status(
        self,
        *,
        id: str,
        status: str,
        reviewed_by: str | None,
        review_message: str | None,
        reviewed_at: datetime,
    ) -> bool:
        """Record the review outcome on a request. Returns rows affected."""
        stmt = text(
            f"update {_JOIN_REQUEST_TABLE} "
            "set status = :status, reviewed_by = :reviewed_by, "
            "review_message = :review_message, reviewed_at = :reviewed_at, "
            "updated_at = :reviewed_at "
            "where id = :id"
        ).bindparams(
            id=id,
            status=status,
            reviewed_by=reviewed_by,
            review_message=review_message,
            reviewed_at=reviewed_at,
        )
        result = cast("CursorResult[SqlValue]", await self._session.execute(stmt))
        return (result.rowcount or 0) > 0

    # ── Reads ───────────────────────────────────────────────────────

    async def get_join_request_by_id(self, id: str) -> OrganizationJoinRequest | None:
        """Return the request for ``id``, or ``None`` when missing."""
        return await self.find_by_primary_key({"id": id}, exclude_deleted_or_archived=False)

    async def get_pending_join_request(
        self,
        *,
        organization_id: str,
        tenant_id: int,
    ) -> OrganizationJoinRequest | None:
        """Return the tenant's pending request for the org, or ``None``."""
        return await self.find_unique_by_column_values(
            {
                "organization_id": organization_id,
                "tenant_id": tenant_id,
                "status": JOIN_REQUEST_STATUS_PENDING,
            },
            exclude_deleted_or_archived=False,
        )

    async def get_pending_request_by_type(
        self,
        *,
        organization_id: str,
        tenant_id: int,
        request_type: str,
    ) -> OrganizationJoinRequest | None:
        """Narrow the pending lookup to one request type (join | upgrade)."""
        return await self.find_unique_by_column_values(
            {
                "organization_id": organization_id,
                "tenant_id": tenant_id,
                "status": JOIN_REQUEST_STATUS_PENDING,
                "request_type": request_type,
            },
            exclude_deleted_or_archived=False,
        )

    async def list_join_requests(
        self,
        organization_id: str,
        *,
        status: str | None = None,
    ) -> list[OrganizationJoinRequest]:
        """Requests of one organization, newest first, optionally by status."""
        where: str = "organization_id = :organization_id"
        params: BindParams = {"organization_id": organization_id}
        if status:
            where = f"{where} and status = :status"
            params["status"] = status
        stmt = text(
            f"select * from {_JOIN_REQUEST_TABLE} where {where} order by {_JOIN_REQUEST_ORDER}"
        ).bindparams(**params)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def count_join_requests(
        self,
        organization_id: str,
        *,
        status: str | None = None,
    ) -> int:
        """Count requests of one organization under the same filter."""
        where: str = "organization_id = :organization_id"
        params: BindParams = {"organization_id": organization_id}
        if status:
            where = f"{where} and status = :status"
            params["status"] = status
        stmt = text(f"select count(*) from {_JOIN_REQUEST_TABLE} where {where}").bindparams(
            **params
        )
        return int((await self._session.execute(stmt)).scalar_one())


__all__ = [
    "OrganizationJoinRequestRepository",
    "OrganizationMemberRepository",
    "OrganizationRepository",
]
