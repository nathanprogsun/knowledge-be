"""Organization service — CRUD + tenant-scoped membership + join-request approval.

Request-scoped: constructed per request by ``factory.build_organization_service``
with fresh repositories on the shared ``AsyncSession``; the web layer never
imports ``db`` directly. Mirrors the upstream organization service semantics:

- ``owner_tenant_id`` is pinned at creation time and never changes, so the
  owning workspace can never be orphaned even if the owner user later moves
  tenants or is soft-deleted.
- Membership is tenant-scoped: every row is a (org, tenant) tuple; the
  ``representative_user_id`` is informational only and does not gate any
  permission check.
- Permission checks ride on (org, tenant, role) — admin gates the update /
  delete / invite-code-regenerate / join-request-review paths.
- Join requests carry a ``request_type`` (``join`` vs ``upgrade``) and a
  ``status`` (``pending`` → ``approved`` / ``rejected``); a partial unique
  index guarantees at most one pending request per (org, tenant, type).

Deferred seams (neutral wording): the cross-tenant share aggregation used by
the searchable-discovery listing (KB / agent share counts) and the audit
recording on review. Those land with the sharing and audit domains.
"""

from __future__ import annotations

import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Final

from src.common.exception import (
    ConflictError,
    NotFoundError,
    PermissionDeniedError,
    ValidationError,
)
from src.core.organizations.types import (
    JOIN_REQUEST_STATUS_APPROVED,
    JOIN_REQUEST_STATUS_PENDING,
    JOIN_REQUEST_STATUS_REJECTED,
    JOIN_REQUEST_TYPE_JOIN,
    JOIN_REQUEST_TYPE_UPGRADE,
    ORG_ROLE_ADMIN,
    ORG_ROLE_EDITOR,
    ORG_ROLE_VIEWER,
    OrganizationInfo,
    OrganizationJoinRequestInfo,
    OrganizationMemberInfo,
)
from src.db.dao.organization_repository import (
    OrganizationJoinRequestRepository,
    OrganizationMemberRepository,
    OrganizationRepository,
)
from src.db.models.organization import (
    JOIN_REQUEST_STATUSES,
    JOIN_REQUEST_TYPES,
    ORG_ROLES,
    Organization,
    OrganizationJoinRequest,
    OrganizationTenantMember,
    has_org_permission,
    is_valid_org_role,
)

# ── Constants ────────────────────────────────────────────────────────

_NOT_FOUND_CODE: Final[str] = "organization.not_found"
_NOT_MEMBER_CODE: Final[str] = "organization.tenant_not_member"

# Default invite-code validity in days (0 = never, 1, 7, 30 allowed).
DEFAULT_INVITE_CODE_VALIDITY_DAYS: Final[int] = 7
# Allowed invite-code validity values, mirrors the upstream allow-list.
VALID_INVITE_CODE_VALIDITY_DAYS: Final[frozenset[int]] = frozenset({0, 1, 7, 30})

# Default max members per organization (0 = unlimited).
DEFAULT_MEMBER_LIMIT: Final[int] = 50

# Generated invite codes are 16 hex chars (8 random bytes).
_INVITE_CODE_BYTES: Final[int] = 8

# Searchable-list default page size.
_DEFAULT_SEARCH_LIMIT: Final[int] = 20


# ── Time / id helpers ────────────────────────────────────────────────


def _now() -> datetime:
    """Return a timezone-aware ``now`` for stamping rows."""
    return datetime.now(UTC)


def _new_id() -> str:
    """Generate a UUID for a freshly created row."""
    return str(uuid.uuid4())


def _generate_invite_code() -> str:
    """Return a fresh 16-character hex invite code."""
    return secrets.token_hex(_INVITE_CODE_BYTES)


def _resolve_invite_expiry(
    validity_days: int,
    now: datetime,
) -> datetime | None:
    """Resolve the invite-code expiry; ``0`` means no expiry (``None``)."""
    if validity_days == 0:
        return None
    return now + timedelta(days=validity_days)


# ── Boundary validators ──────────────────────────────────────────────


def _require_org_id(org_id: str) -> None:
    """Reject an empty organization id."""
    if not org_id or not org_id.strip():
        raise ValidationError(
            code="organization.id_required",
            message="organization ID is required",
        )


def _require_tenant_id(tenant_id: int) -> None:
    """Reject a non-positive tenant id."""
    if not isinstance(tenant_id, int) or tenant_id <= 0:
        raise ValidationError(
            code="organization.tenant_required",
            message="tenant ID is required",
        )


def _require_user_id(user_id: str) -> str:
    """Reject an empty user id; return the trimmed value."""
    clean = user_id.strip()
    if not clean:
        raise ValidationError(
            code="organization.user_required",
            message="user ID is required",
        )
    return clean


def _require_name(name: str) -> str:
    """Reject a blank organization name; return the trimmed value."""
    clean = name.strip()
    if not clean:
        raise ValidationError(
            code="organization.name_required",
            message="organization name is required",
        )
    return clean


def _require_role(role: str) -> str:
    """Reject a role outside the sanctioned set."""
    if not is_valid_org_role(role):
        raise ValidationError(
            code="organization.role_invalid",
            message=f"invalid role: {role}",
        )
    return role


def _require_member_limit(value: int) -> int:
    """Reject a negative member limit; ``0`` means unlimited."""
    if value < 0:
        raise ValidationError(
            code="organization.member_limit_invalid",
            message="member_limit must be >= 0",
        )
    return value


def _require_validity_days(value: int) -> int:
    """Reject an invite-code validity outside the allow-list."""
    if value not in VALID_INVITE_CODE_VALIDITY_DAYS:
        raise ValidationError(
            code="organization.invite_validity_invalid",
            message="invite_code_validity_days must be 0, 1, 7, or 30",
        )
    return value


def _normalise_role(role: str) -> str:
    """Default a blank role to viewer; otherwise validate it."""
    if role == "":
        return ORG_ROLE_VIEWER
    return _require_role(role)


# ── Service ──────────────────────────────────────────────────────────


class OrganizationService:
    """Stateless organization service, constructed per request."""

    def __init__(
        self,
        *,
        org_repo: OrganizationRepository,
        member_repo: OrganizationMemberRepository,
        join_request_repo: OrganizationJoinRequestRepository,
    ) -> None:
        self._org_repo = org_repo
        self._member_repo = member_repo
        self._join_request_repo = join_request_repo

    # ── Organization CRUD ──────────────────────────────────────────

    async def create_organization(
        self,
        *,
        user_id: str,
        tenant_id: int,
        name: str,
        description: str | None = None,
        avatar: str | None = None,
        invite_code_validity_days: int | None = None,
        member_limit: int | None = None,
    ) -> OrganizationInfo:
        """Insert a new organization and enrol the creator's tenant as admin.

        ``owner_tenant_id`` is pinned to ``tenant_id`` so the owning workspace
        can never be orphaned. The creator's tenant is enrolled at admin role
        in the same transaction; on a member-insert failure the org row is
        rolled back to avoid leaving a headless collaboration space.
        """
        clean_user = _require_user_id(user_id)
        _require_tenant_id(tenant_id)
        clean_name = _require_name(name)
        validity_days = (
            DEFAULT_INVITE_CODE_VALIDITY_DAYS
            if invite_code_validity_days is None
            else _require_validity_days(invite_code_validity_days)
        )
        limit = (
            DEFAULT_MEMBER_LIMIT if member_limit is None else _require_member_limit(member_limit)
        )

        now = _now()
        row = Organization(
            id=_new_id(),
            name=clean_name,
            description=description,
            avatar=(avatar or "").strip(),
            owner_id=clean_user,
            owner_tenant_id=tenant_id,
            invite_code=_generate_invite_code(),
            invite_code_expires_at=_resolve_invite_expiry(validity_days, now),
            invite_code_validity_days=validity_days,
            require_approval=False,
            searchable=False,
            member_limit=limit,
            created_at=now,
            updated_at=now,
        )
        persisted = await self._org_repo.create(row)

        member_row = OrganizationTenantMember(
            id=_new_id(),
            organization_id=persisted.id,
            tenant_id=tenant_id,
            role=ORG_ROLE_ADMIN,
            representative_user_id=clean_user,
            joined_at=now,
            created_at=now,
            updated_at=now,
        )
        try:
            await self._member_repo.add_member(member_row)
        except Exception:
            # Best-effort rollback: a headless org would be unrecoverable for
            # the creator; the soft-delete is non-fatal so we surface the
            # original error.
            await self._org_repo.soft_delete(id=persisted.id, now=now)
            raise

        return OrganizationInfo.map_from_db(persisted)

    async def get_organization(self, *, id: str) -> OrganizationInfo:
        """Return one organization by id, or raise ``NotFoundError``."""
        _require_org_id(id)
        row = await self._org_repo.get_by_id_or_none(id)
        if row is None:
            raise self._not_found(id)
        return OrganizationInfo.map_from_db(row)

    async def get_organization_by_invite_code(
        self,
        *,
        invite_code: str,
    ) -> OrganizationInfo:
        """Resolve an invite code to its organization, enforcing expiry."""
        if not invite_code:
            raise self._not_found_for_invite_code(invite_code)
        row = await self._org_repo.get_by_invite_code_or_none(invite_code)
        if row is None:
            raise self._not_found_for_invite_code(invite_code)
        expires_at = row.invite_code_expires_at
        if expires_at is not None and expires_at <= _now():
            raise ValidationError(
                code="organization.invite_code_expired",
                message="invite code has expired",
            )
        return OrganizationInfo.map_from_db(row)

    async def list_tenant_organizations(
        self,
        *,
        tenant_id: int,
    ) -> list[OrganizationInfo]:
        """Return every organization the tenant participates in, newest first."""
        _require_tenant_id(tenant_id)
        rows = await self._org_repo.list_by_tenant(tenant_id)
        return [OrganizationInfo.map_from_db(row) for row in rows]

    async def update_organization(
        self,
        *,
        id: str,
        operator_user_id: str,
        operator_tenant_id: int,
        name: str | None = None,
        description: str | None = None,
        avatar: str | None = None,
        require_approval: bool | None = None,
        searchable: bool | None = None,
        invite_code_validity_days: int | None = None,
        member_limit: int | None = None,
    ) -> OrganizationInfo:
        """Apply the mutable subset of organization fields; admin only."""
        _require_org_id(id)
        _require_user_id(operator_user_id)
        _require_tenant_id(operator_tenant_id)
        is_admin = await self.is_tenant_org_admin(
            org_id=id, tenant_id=operator_tenant_id
        )
        if not is_admin:
            raise self._permission_denied(
                "operator tenant is not an admin of this organization",
            )

        existing = await self._org_repo.get_by_id_or_none(id)
        if existing is None:
            raise self._not_found(id)

        updates: dict[str, object] = {}
        if name is not None:
            updates["name"] = _require_name(name)
        if description is not None:
            updates["description"] = description
        if avatar is not None:
            updates["avatar"] = avatar.strip()
        if require_approval is not None:
            updates["require_approval"] = require_approval
        if searchable is not None:
            updates["searchable"] = searchable
        if invite_code_validity_days is not None:
            updates["invite_code_validity_days"] = _require_validity_days(
                invite_code_validity_days
            )
        if member_limit is not None:
            limit = _require_member_limit(member_limit)
            if limit > 0:
                count = await self._member_repo.count_members(id)
                if count > limit:
                    raise ValidationError(
                        code="organization.member_limit_too_low",
                        message="member limit cannot be lower than current member count",
                    )
            updates["member_limit"] = limit
        updates["updated_at"] = _now()

        updated_row = existing.model_copy(update=updates)
        persisted = await self._org_repo.update(updated_row)
        return OrganizationInfo.map_from_db(persisted)

    async def delete_organization(
        self,
        *,
        id: str,
        operator_user_id: str,
        operator_tenant_id: int,
    ) -> None:
        """Soft-delete an organization; the operator's tenant must be the owner.

        Falls back to the legacy user-level rule (``owner_id == operator_user_id``)
        only for rows where ``owner_tenant_id`` was never persisted, so
        pre-backfill orgs remain deletable by their original creator.
        """
        _require_org_id(id)
        _require_user_id(operator_user_id)
        _require_tenant_id(operator_tenant_id)
        existing = await self._org_repo.get_by_id_or_none(id)
        if existing is None:
            raise self._not_found(id)

        is_owner_tenant = (
            existing.owner_tenant_id != 0
            and existing.owner_tenant_id == operator_tenant_id
        )
        is_legacy_owner_user = (
            existing.owner_tenant_id == 0
            and existing.owner_id == operator_user_id
        )
        if not is_owner_tenant and not is_legacy_owner_user:
            raise self._permission_denied(
                "operator is not the owner of this organization",
            )

        await self._org_repo.soft_delete(id=id, now=_now())

    # ── Invite codes ───────────────────────────────────────────────

    async def generate_invite_code(
        self,
        *,
        org_id: str,
        operator_user_id: str,
        operator_tenant_id: int,
    ) -> str:
        """Rotate the invite code; admin-only."""
        _require_org_id(org_id)
        _require_user_id(operator_user_id)
        _require_tenant_id(operator_tenant_id)
        is_admin = await self.is_tenant_org_admin(
            org_id=org_id, tenant_id=operator_tenant_id
        )
        if not is_admin:
            raise self._permission_denied(
                "operator tenant is not an admin of this organization",
            )

        existing = await self._org_repo.get_by_id_or_none(org_id)
        if existing is None:
            raise self._not_found(org_id)

        validity_days = existing.invite_code_validity_days
        if validity_days not in VALID_INVITE_CODE_VALIDITY_DAYS:
            validity_days = DEFAULT_INVITE_CODE_VALIDITY_DAYS

        new_code = _generate_invite_code()
        now = _now()
        updated = await self._org_repo.update_invite_code(
            id=org_id,
            invite_code=new_code,
            expires_at=_resolve_invite_expiry(validity_days, now),
            now=now,
        )
        if not updated:
            raise self._not_found(org_id)
        return new_code

    # ── Tenant membership ──────────────────────────────────────────

    async def add_tenant_member(
        self,
        *,
        org_id: str,
        tenant_id: int,
        representative_user_id: str,
        role: str,
    ) -> OrganizationMemberInfo:
        """Enrol a tenant in an organization; enforces the member limit."""
        _require_org_id(org_id)
        _require_tenant_id(tenant_id)
        required_role = _require_role(role)
        org = await self._org_repo.get_by_id_or_none(org_id)
        if org is None:
            raise self._not_found(org_id)
        if org.member_limit > 0:
            count = await self._member_repo.count_members(org_id)
            if count >= org.member_limit:
                raise ConflictError(
                    code="organization.member_limit_reached",
                    message="organization member limit reached",
                )

        now = _now()
        row = OrganizationTenantMember(
            id=_new_id(),
            organization_id=org_id,
            tenant_id=tenant_id,
            role=required_role,
            representative_user_id=representative_user_id or "",
            joined_at=now,
            created_at=now,
            updated_at=now,
        )
        stored = await self._member_repo.add_member(row)
        if stored is None:
            # Duplicate (org, tenant) — the partial unique index suppressed
            # the insert; surface the conflict to the caller.
            raise ConflictError(
                code="organization.tenant_already_member",
                message="tenant is already a member of this organization",
            )
        return OrganizationMemberInfo.map_from_db(stored)

    async def remove_tenant_member(
        self,
        *,
        org_id: str,
        member_tenant_id: int,
        operator_user_id: str,
        operator_tenant_id: int,
    ) -> None:
        """Remove a tenant from an organization; owner-tenant is undeletable.

        Self-removal (operator is the target) is always allowed. Admin
        removal requires the operator's tenant to be admin in the org.
        """
        _require_org_id(org_id)
        _require_tenant_id(member_tenant_id)
        _require_user_id(operator_user_id)
        _require_tenant_id(operator_tenant_id)
        org = await self._org_repo.get_by_id_or_none(org_id)
        if org is None:
            raise self._not_found(org_id)
        if self._is_owner_tenant(org, member_tenant_id):
            raise ConflictError(
                code="organization.cannot_remove_owner",
                message="cannot remove organization owner tenant",
            )

        if operator_tenant_id != member_tenant_id:
            is_admin = await self.is_tenant_org_admin(
                org_id=org_id, tenant_id=operator_tenant_id
            )
            if not is_admin:
                raise self._permission_denied(
                    "operator tenant is not an admin of this organization",
                )

        await self._member_repo.remove_member(
            organization_id=org_id, tenant_id=member_tenant_id
        )

    async def update_tenant_member_role(
        self,
        *,
        org_id: str,
        member_tenant_id: int,
        role: str,
        operator_user_id: str,
        operator_tenant_id: int,
    ) -> OrganizationMemberInfo:
        """Change a tenant's role; admin-only and refuses to change the owner."""
        _require_org_id(org_id)
        _require_tenant_id(member_tenant_id)
        required_role = _require_role(role)
        _require_user_id(operator_user_id)
        _require_tenant_id(operator_tenant_id)
        is_admin = await self.is_tenant_org_admin(
            org_id=org_id, tenant_id=operator_tenant_id
        )
        if not is_admin:
            raise self._permission_denied(
                "operator tenant is not an admin of this organization",
            )

        org = await self._org_repo.get_by_id_or_none(org_id)
        if org is None:
            raise self._not_found(org_id)
        if self._is_owner_tenant(org, member_tenant_id):
            raise ConflictError(
                code="organization.cannot_change_owner_role",
                message="cannot change organization owner tenant role",
            )

        updated = await self._member_repo.update_member_role(
            organization_id=org_id,
            tenant_id=member_tenant_id,
            role=required_role,
        )
        if not updated:
            raise self._not_member(member_tenant_id)
        member = await self._member_repo.get_member(
            organization_id=org_id, tenant_id=member_tenant_id
        )
        if member is None:
            raise self._not_member(member_tenant_id)
        return OrganizationMemberInfo.map_from_db(member)

    async def list_tenant_members(
        self,
        *,
        org_id: str,
    ) -> list[OrganizationMemberInfo]:
        """Return every tenant membership of the organization, oldest first."""
        _require_org_id(org_id)
        rows = await self._member_repo.list_members(org_id)
        return [OrganizationMemberInfo.map_from_db(row) for row in rows]

    async def get_tenant_member(
        self,
        *,
        org_id: str,
        tenant_id: int,
    ) -> OrganizationMemberInfo:
        """Return one (org, tenant) membership, or raise ``NotFoundError``."""
        _require_org_id(org_id)
        _require_tenant_id(tenant_id)
        row = await self._member_repo.get_member(
            organization_id=org_id, tenant_id=tenant_id
        )
        if row is None:
            raise self._not_member(tenant_id)
        return OrganizationMemberInfo.map_from_db(row)

    async def is_tenant_org_admin(
        self,
        *,
        org_id: str,
        tenant_id: int,
    ) -> bool:
        """Return whether ``tenant_id`` holds the admin role in ``org_id``."""
        _require_org_id(org_id)
        _require_tenant_id(tenant_id)
        member = await self._member_repo.get_member(
            organization_id=org_id, tenant_id=tenant_id
        )
        if member is None:
            return False
        return member.role == ORG_ROLE_ADMIN

    async def get_tenant_role_in_org(
        self,
        *,
        org_id: str,
        tenant_id: int,
    ) -> str:
        """Return the tenant's role in the org; ``NotFoundError`` if absent."""
        _require_org_id(org_id)
        _require_tenant_id(tenant_id)
        member = await self._member_repo.get_member(
            organization_id=org_id, tenant_id=tenant_id
        )
        if member is None:
            raise self._not_member(tenant_id)
        return member.role

    # ── Join flows ─────────────────────────────────────────────────

    async def join_by_invite_code(
        self,
        *,
        invite_code: str,
        user_id: str,
        tenant_id: int,
    ) -> OrganizationInfo:
        """Join an organization via invite code as a viewer.

        Orgs with ``require_approval`` reject invite-code joins outright;
        those callers must go through ``submit_join_request`` instead.
        """
        clean_user = _require_user_id(user_id)
        _require_tenant_id(tenant_id)
        org = await self.get_organization_by_invite_code(invite_code=invite_code)
        if org.require_approval:
            raise self._permission_denied(
                "organization requires approval; submit a join request",
            )
        await self._join_as_viewer_with_checks(
            org_id=org.id, tenant_id=tenant_id, representative_user_id=clean_user
        )
        return org

    async def join_by_organization_id(
        self,
        *,
        org_id: str,
        user_id: str,
        tenant_id: int,
        message: str = "",
        requested_role: str = "",
    ) -> OrganizationInfo:
        """Join a searchable organization by id (no invite code required).

        ``request_type`` join requests are routed via
        ``submit_join_request`` when ``require_approval`` is set; otherwise
        the tenant is enrolled as a viewer in the same call.
        """
        clean_user = _require_user_id(user_id)
        _require_tenant_id(tenant_id)
        _require_org_id(org_id)
        org = await self._org_repo.get_by_id_or_none(org_id)
        if org is None:
            raise self._not_found(org_id)
        if not org.searchable:
            raise self._permission_denied(
                "organization is not open for searchable join",
            )

        existing = await self._member_repo.get_member(
            organization_id=org_id, tenant_id=tenant_id
        )
        if existing is not None:
            return OrganizationInfo.map_from_db(org)

        requested = _normalise_role(requested_role)
        if org.require_approval:
            await self.submit_join_request(
                org_id=org_id,
                user_id=clean_user,
                tenant_id=tenant_id,
                message=message,
                requested_role=requested,
            )
            return OrganizationInfo.map_from_db(org)

        await self._join_as_viewer_with_checks(
            org_id=org_id, tenant_id=tenant_id, representative_user_id=clean_user
        )
        return OrganizationInfo.map_from_db(org)

    async def _join_as_viewer_with_checks(
        self,
        *,
        org_id: str,
        tenant_id: int,
        representative_user_id: str,
    ) -> None:
        """Idempotent viewer-enrolment; enforces the member limit."""
        existing = await self._member_repo.get_member(
            organization_id=org_id, tenant_id=tenant_id
        )
        if existing is not None:
            return
        org = await self._org_repo.get_by_id_or_none(org_id)
        if org is None:
            raise self._not_found(org_id)
        if org.member_limit > 0:
            count = await self._member_repo.count_members(org_id)
            if count >= org.member_limit:
                raise ConflictError(
                    code="organization.member_limit_reached",
                    message="organization member limit reached",
                )

        now = _now()
        row = OrganizationTenantMember(
            id=_new_id(),
            organization_id=org_id,
            tenant_id=tenant_id,
            role=ORG_ROLE_VIEWER,
            representative_user_id=representative_user_id or "",
            joined_at=now,
            created_at=now,
            updated_at=now,
        )
        stored = await self._member_repo.add_member(row)
        if stored is None:
            # A concurrent caller enrolled the tenant first; idempotent ok.
            return

    # ── Join requests ──────────────────────────────────────────────

    async def submit_join_request(
        self,
        *,
        org_id: str,
        user_id: str,
        tenant_id: int,
        message: str = "",
        requested_role: str = "",
    ) -> OrganizationJoinRequestInfo:
        """Submit a new-member join request for admin review.

        A pending request of the same ``request_type`` already on file
        causes the call to fail with ``ConflictError`` — at most one
        pending request per (org, tenant, type) at a time.
        """
        clean_user = _require_user_id(user_id)
        _require_tenant_id(tenant_id)
        _require_org_id(org_id)

        existing = await self._join_request_repo.get_pending_request_by_type(
            organization_id=org_id,
            tenant_id=tenant_id,
            request_type=JOIN_REQUEST_TYPE_JOIN,
        )
        if existing is not None:
            raise ConflictError(
                code="organization.pending_request_exists",
                message="pending request already exists",
            )

        org = await self._org_repo.get_by_id_or_none(org_id)
        if org is None:
            raise self._not_found(org_id)
        if org.member_limit > 0:
            count = await self._member_repo.count_members(org_id)
            if count >= org.member_limit:
                raise ConflictError(
                    code="organization.member_limit_reached",
                    message="organization member limit reached",
                )

        requested = _normalise_role(requested_role)
        now = _now()
        row = OrganizationJoinRequest(
            id=_new_id(),
            organization_id=org_id,
            user_id=clean_user,
            tenant_id=tenant_id,
            request_type=JOIN_REQUEST_TYPE_JOIN,
            requested_role=requested,
            status=JOIN_REQUEST_STATUS_PENDING,
            message=message or None,
            created_at=now,
            updated_at=now,
        )
        persisted = await self._join_request_repo.create_join_request(row)
        return OrganizationJoinRequestInfo.map_from_db(persisted)

    async def list_join_requests(
        self,
        *,
        org_id: str,
        status: str | None = None,
    ) -> list[OrganizationJoinRequestInfo]:
        """Return join requests of the organization, newest first."""
        _require_org_id(org_id)
        if status is not None and status not in JOIN_REQUEST_STATUSES:
            raise ValidationError(
                code="organization.join_request_status_invalid",
                message=f"invalid join-request status: {status}",
            )
        rows = await self._join_request_repo.list_join_requests(
            org_id, status=status
        )
        return [OrganizationJoinRequestInfo.map_from_db(row) for row in rows]

    async def count_pending_join_requests(self, *, org_id: str) -> int:
        """Return the number of pending join requests for the organization."""
        _require_org_id(org_id)
        return await self._join_request_repo.count_join_requests(
            org_id, status=JOIN_REQUEST_STATUS_PENDING
        )

    async def review_join_request(
        self,
        *,
        org_id: str,
        request_id: str,
        approved: bool,
        reviewer_user_id: str,
        reviewer_tenant_id: int,
        message: str = "",
        assign_role: str | None = None,
    ) -> OrganizationJoinRequestInfo:
        """Approve or reject a pending join / upgrade request.

        On approve the targeted tenant is enrolled (``join``) or has its role
        bumped (``upgrade``); the ``assign_role`` argument overrides the
        request's ``requested_role`` when provided.
        """
        clean_reviewer = _require_user_id(reviewer_user_id)
        _require_tenant_id(reviewer_tenant_id)
        _require_org_id(org_id)
        if not request_id or not request_id.strip():
            raise ValidationError(
                code="organization.join_request_id_required",
                message="join request ID is required",
            )

        request = await self._join_request_repo.get_join_request_by_id(request_id)
        if request is None or request.organization_id != org_id:
            raise NotFoundError(
                code="organization.join_request_not_found",
                message="join request not found",
            )
        if request.status != JOIN_REQUEST_STATUS_PENDING:
            raise ConflictError(
                code="organization.join_request_already_reviewed",
                message="request has already been reviewed",
            )

        if approved:
            role = ORG_ROLE_VIEWER
            if assign_role is not None:
                role = _require_role(assign_role)
            elif is_valid_org_role(request.requested_role):
                role = request.requested_role

            if request.request_type == JOIN_REQUEST_TYPE_UPGRADE:
                updated = await self._member_repo.update_member_role(
                    organization_id=request.organization_id,
                    tenant_id=request.tenant_id,
                    role=role,
                )
                if not updated:
                    raise self._not_member(request.tenant_id)
            else:
                org = await self._org_repo.get_by_id_or_none(request.organization_id)
                if org is None:
                    raise self._not_found(request.organization_id)
                if org.member_limit > 0:
                    count = await self._member_repo.count_members(
                        request.organization_id
                    )
                    if count >= org.member_limit:
                        raise ConflictError(
                            code="organization.member_limit_reached",
                            message="organization member limit reached",
                        )
                now = _now()
                row = OrganizationTenantMember(
                    id=_new_id(),
                    organization_id=request.organization_id,
                    tenant_id=request.tenant_id,
                    role=role,
                    representative_user_id=request.user_id,
                    joined_at=now,
                    created_at=now,
                    updated_at=now,
                )
                stored = await self._member_repo.add_member(row)
                if stored is None:
                    # Tenant already a member — surface the resulting state
                    # by returning the existing membership projection.
                    existing_member = await self._member_repo.get_member(
                        organization_id=request.organization_id,
                        tenant_id=request.tenant_id,
                    )
                    if existing_member is None:
                        raise ConflictError(
                            code="organization.tenant_already_member",
                            message="tenant is already a member of this organization",
                        )

            new_status = JOIN_REQUEST_STATUS_APPROVED
        else:
            new_status = JOIN_REQUEST_STATUS_REJECTED

        now = _now()
        await self._join_request_repo.update_join_request_status(
            id=request_id,
            status=new_status,
            reviewed_by=clean_reviewer,
            review_message=message or None,
            reviewed_at=now,
        )
        updated = await self._join_request_repo.get_join_request_by_id(request_id)
        if updated is None:
            raise NotFoundError(
                code="organization.join_request_not_found",
                message="join request not found",
            )
        return OrganizationJoinRequestInfo.map_from_db(updated)

    async def request_role_upgrade(
        self,
        *,
        org_id: str,
        user_id: str,
        tenant_id: int,
        requested_role: str,
        message: str = "",
    ) -> OrganizationJoinRequestInfo:
        """Submit a role-upgrade request for the caller's tenant.

        The tenant must already be a member of the org; admins cannot
        request an upgrade (they already have the top role), and a
        non-higher role is refused with a conflict.
        """
        clean_user = _require_user_id(user_id)
        _require_tenant_id(tenant_id)
        _require_org_id(org_id)
        required_role = _require_role(requested_role)

        member = await self._member_repo.get_member(
            organization_id=org_id, tenant_id=tenant_id
        )
        if member is None:
            raise self._not_member(tenant_id)
        if member.role == ORG_ROLE_ADMIN:
            raise ConflictError(
                code="organization.already_admin",
                message="tenant is already an admin",
            )
        if (
            not has_org_permission(required_role, member.role)
            or required_role == member.role
        ):
            raise ConflictError(
                code="organization.upgrade_to_same_or_lower_role",
                message="cannot request upgrade to same or lower role",
            )

        existing = await self._join_request_repo.get_pending_request_by_type(
            organization_id=org_id,
            tenant_id=tenant_id,
            request_type=JOIN_REQUEST_TYPE_UPGRADE,
        )
        if existing is not None:
            raise ConflictError(
                code="organization.pending_request_exists",
                message="pending request already exists",
            )

        now = _now()
        row = OrganizationJoinRequest(
            id=_new_id(),
            organization_id=org_id,
            user_id=clean_user,
            tenant_id=tenant_id,
            request_type=JOIN_REQUEST_TYPE_UPGRADE,
            prev_role=member.role,
            requested_role=required_role,
            status=JOIN_REQUEST_STATUS_PENDING,
            message=message or None,
            created_at=now,
            updated_at=now,
        )
        persisted = await self._join_request_repo.create_join_request(row)
        return OrganizationJoinRequestInfo.map_from_db(persisted)

    async def get_pending_upgrade_request(
        self,
        *,
        org_id: str,
        tenant_id: int,
    ) -> OrganizationJoinRequestInfo:
        """Return the tenant's pending upgrade request, or ``NotFoundError``."""
        _require_org_id(org_id)
        _require_tenant_id(tenant_id)
        row = await self._join_request_repo.get_pending_request_by_type(
            organization_id=org_id,
            tenant_id=tenant_id,
            request_type=JOIN_REQUEST_TYPE_UPGRADE,
        )
        if row is None:
            raise NotFoundError(
                code="organization.upgrade_request_not_found",
                message="pending upgrade request not found",
            )
        return OrganizationJoinRequestInfo.map_from_db(row)

    # ── Searchable listing ─────────────────────────────────────────

    async def search_searchable_organizations(
        self,
        *,
        tenant_id: int,
        query: str = "",
        limit: int = _DEFAULT_SEARCH_LIMIT,
    ) -> list[OrganizationInfo]:
        """Return discoverable organizations matching ``query``.

        The caller is expected to enrich each row with member / share counts
        at the presentation layer; the service deliberately returns the
        plain projection to keep storage dependencies local.
        """
        _require_tenant_id(tenant_id)
        page_size = limit if limit > 0 else _DEFAULT_SEARCH_LIMIT
        rows = await self._org_repo.list_searchable(query=query, limit=page_size)
        return [OrganizationInfo.map_from_db(row) for row in rows]

    # ── Internal helpers ───────────────────────────────────────────

    @staticmethod
    def _is_owner_tenant(org: Organization, tenant_id: int) -> bool:
        """Whether ``tenant_id`` is the org's persisted owning tenant.

        Fails-closed: when ``owner_tenant_id`` is zero (legacy / test row),
        every tenant is treated AS IF it were the owner — the only safe
        default because the alternative would let the real owner be
        removed without recovery.
        """
        if org.owner_tenant_id == 0:
            return True
        return org.owner_tenant_id == tenant_id

    @staticmethod
    def _not_found(org_id: str) -> NotFoundError:
        return NotFoundError(
            code=_NOT_FOUND_CODE,
            message=f"organization {org_id} not found",
        )

    @staticmethod
    def _not_found_for_invite_code(invite_code: str) -> NotFoundError:
        return NotFoundError(
            code="organization.invite_code_not_found",
            message=f"invite code {invite_code!r} not found",
        )

    @staticmethod
    def _not_member(tenant_id: int) -> NotFoundError:
        return NotFoundError(
            code=_NOT_MEMBER_CODE,
            message=f"tenant {tenant_id} is not a member of this organization",
        )

    @staticmethod
    def _permission_denied(detail: str) -> PermissionDeniedError:
        return PermissionDeniedError(
            code="organization.permission_denied",
            message=detail,
        )


# Re-export the role / status constants from the storage layer so callers
# can build requests against a single vocabulary.
__all__ = [
    "DEFAULT_INVITE_CODE_VALIDITY_DAYS",
    "DEFAULT_MEMBER_LIMIT",
    "JOIN_REQUEST_STATUSES",
    "JOIN_REQUEST_STATUS_APPROVED",
    "JOIN_REQUEST_STATUS_PENDING",
    "JOIN_REQUEST_STATUS_REJECTED",
    "JOIN_REQUEST_TYPES",
    "JOIN_REQUEST_TYPE_JOIN",
    "JOIN_REQUEST_TYPE_UPGRADE",
    "ORG_ROLES",
    "ORG_ROLE_ADMIN",
    "ORG_ROLE_EDITOR",
    "ORG_ROLE_VIEWER",
    "OrganizationService",
    "VALID_INVITE_CODE_VALIDITY_DAYS",
]
