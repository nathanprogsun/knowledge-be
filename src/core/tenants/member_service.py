"""Tenant membership service — join, remove, role changes, listings.

Invariant it protects: **a workspace never loses its last Owner**. Both
dangerous paths (demoting an Owner, removing an Owner) first lock every
*other* active Owner row and only proceed if at least one remains, which
closes the read-then-write race two concurrent demotions would otherwise
win together.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.common.exception import ConflictError, NotFoundError, ValidationError
from src.core.tenants.types import MembershipInfo
from src.db.dao.tenant_members_repository import TenantMemberRepository
from src.db.models.tenants.tenant_members import TenantMember

ROLE_OWNER = "owner"
ROLE_ADMIN = "admin"
ROLE_CONTRIBUTOR = "contributor"
ROLE_VIEWER = "viewer"

STATUS_ACTIVE = "active"

# Privilege levels, spaced by ten so a new role can be slotted between
# two existing ones without renumbering.
_ROLE_LEVELS: dict[str, int] = {
    ROLE_OWNER: 40,
    ROLE_ADMIN: 30,
    ROLE_CONTRIBUTOR: 20,
    ROLE_VIEWER: 10,
}

# Defensive clamps on the list endpoint.
_DEFAULT_PAGE_SIZE = 20
_MAX_PAGE_SIZE = 100


def is_valid_role(role: str) -> bool:
    """Whether ``role`` is one of the four defined workspace roles."""
    return role in _ROLE_LEVELS


def role_level(role: str) -> int:
    """Privilege level of ``role``; unknown roles rank below every role."""
    return _ROLE_LEVELS.get(role, 0)


def has_permission(role: str, required: str) -> bool:
    """Whether ``role`` is at least as privileged as ``required``."""
    return role_level(role) >= role_level(required)


class TenantMemberService:
    """Stateless membership service, constructed per request."""

    def __init__(self, *, members_repo: TenantMemberRepository) -> None:
        self._members_repo = members_repo

    # ── Joining ─────────────────────────────────────────────────────

    async def add_member(
        self,
        *,
        user_id: str,
        tenant_id: int,
        role: str,
        invited_by: str | None = None,
    ) -> MembershipInfo:
        """Add an active membership; conflict if the user already belongs."""
        self.require_valid_role(role)
        existing = await self._members_repo.find_membership(
            user_id=user_id,
            tenant_id=tenant_id,
        )
        if existing is not None:
            raise self._already_exists()
        stored = await self._members_repo.insert_live_or_none(
            self._new_row(
                user_id=user_id,
                tenant_id=tenant_id,
                role=role,
                invited_by=invited_by,
            )
        )
        if stored is None:
            # A concurrent add slipped between the check and the insert;
            # the partial unique index caught it. Report the same
            # conflict the check would have.
            raise self._already_exists()
        return MembershipInfo.map_from_db(stored)

    async def ensure_owner(self, *, user_id: str, tenant_id: int) -> MembershipInfo:
        """Idempotently make the user an Owner of the workspace.

        Returns the existing membership unchanged when there is one —
        including the row a concurrent caller inserted first — so the
        registration and OIDC paths can re-run safely.
        """
        existing = await self._members_repo.find_membership(
            user_id=user_id,
            tenant_id=tenant_id,
        )
        if existing is not None:
            return MembershipInfo.map_from_db(existing)
        stored = await self._members_repo.insert_live_or_none(
            self._new_row(user_id=user_id, tenant_id=tenant_id, role=ROLE_OWNER)
        )
        if stored is not None:
            return MembershipInfo.map_from_db(stored)
        winner = await self._members_repo.find_membership(
            user_id=user_id,
            tenant_id=tenant_id,
        )
        if winner is None:
            raise ConflictError(
                code="tenant_member.exists",
                message="Tenant membership already exists",
            )
        return MembershipInfo.map_from_db(winner)

    # ── Reads ───────────────────────────────────────────────────────

    async def get_membership(self, *, user_id: str, tenant_id: int) -> MembershipInfo | None:
        """Return the membership for the pair, or ``None`` when absent."""
        row = await self._members_repo.find_membership(user_id=user_id, tenant_id=tenant_id)
        return MembershipInfo.map_from_db(row) if row is not None else None

    async def list_by_user(self, user_id: str) -> list[MembershipInfo]:
        """Every workspace the user belongs to, oldest join first."""
        rows = await self._members_repo.list_by_user(user_id)
        return [MembershipInfo.map_from_db(row) for row in rows]

    async def list_by_tenant(self, tenant_id: int) -> list[MembershipInfo]:
        """Every member of the workspace, oldest join first."""
        rows = await self._members_repo.list_by_tenant(tenant_id)
        return [MembershipInfo.map_from_db(row) for row in rows]

    async def has_any_members(self, tenant_id: int) -> bool:
        """Whether the workspace has at least one active member."""
        return await self._members_repo.has_any_members(tenant_id)

    async def list_members_page(
        self,
        tenant_id: int,
        *,
        query: str | None = None,
        page: int = 1,
        page_size: int = _DEFAULT_PAGE_SIZE,
    ) -> tuple[list[MembershipInfo], int]:
        """One page of members plus the unpaginated total.

        ``query`` matches the member's email or username. Paging inputs
        are clamped rather than rejected.
        """
        page = max(page, 1)
        page_size = page_size if page_size >= 1 else _DEFAULT_PAGE_SIZE
        page_size = min(page_size, _MAX_PAGE_SIZE)
        total = await self._members_repo.count_by_tenant(tenant_id, search=query)
        rows = await self._members_repo.list_page_by_tenant(
            tenant_id,
            search=query,
            limit=page_size,
            offset=(page - 1) * page_size,
        )
        return [MembershipInfo.map_from_db(row) for row in rows], total

    # ── Role changes ────────────────────────────────────────────────

    async def update_role(self, *, user_id: str, tenant_id: int, role: str) -> MembershipInfo:
        """Change a member's role, refusing to demote the last Owner."""
        self.require_valid_role(role)
        current = await self._require_membership(user_id=user_id, tenant_id=tenant_id)
        if current.role == role:
            return MembershipInfo.map_from_db(current)
        if current.role == ROLE_OWNER:
            await self._require_another_owner(tenant_id=tenant_id, user_id=user_id)
        affected = await self._members_repo.update_role(
            user_id=user_id,
            tenant_id=tenant_id,
            role=role,
            updated_at=datetime.now(UTC),
        )
        if affected == 0:
            raise self._not_found()
        return await self._require_membership_info(user_id=user_id, tenant_id=tenant_id)

    # ── Removal ─────────────────────────────────────────────────────

    async def remove_member(self, *, user_id: str, tenant_id: int) -> None:
        """Soft-delete a membership, refusing to remove the last Owner."""
        current = await self._require_membership(user_id=user_id, tenant_id=tenant_id)
        if current.role == ROLE_OWNER:
            await self._require_another_owner(tenant_id=tenant_id, user_id=user_id)
        affected = await self._members_repo.soft_delete(
            user_id=user_id,
            tenant_id=tenant_id,
            deleted_at=datetime.now(UTC),
        )
        if affected == 0:
            raise self._not_found()

    # ── Internal helpers ────────────────────────────────────────────

    @staticmethod
    def _new_row(
        *,
        user_id: str,
        tenant_id: int,
        role: str,
        invited_by: str | None = None,
    ) -> TenantMember:
        now = datetime.now(UTC)
        return TenantMember(
            user_id=user_id,
            tenant_id=tenant_id,
            role=role,
            status=STATUS_ACTIVE,
            invited_by=invited_by,
            joined_at=now,
            created_at=now,
            updated_at=now,
        )

    async def _require_membership(self, *, user_id: str, tenant_id: int) -> TenantMember:
        row = await self._members_repo.find_membership(user_id=user_id, tenant_id=tenant_id)
        if row is None:
            raise self._not_found()
        return row

    async def _require_membership_info(self, *, user_id: str, tenant_id: int) -> MembershipInfo:
        return MembershipInfo.map_from_db(
            await self._require_membership(user_id=user_id, tenant_id=tenant_id)
        )

    async def _require_another_owner(self, *, tenant_id: int, user_id: str) -> None:
        """Lock the other active Owners and refuse if there are none."""
        others = await self._members_repo.count_other_active_owners_for_update(
            tenant_id=tenant_id,
            exclude_user_id=user_id,
        )
        if others == 0:
            raise ConflictError(
                code="tenant_member.last_owner",
                message="Workspace must keep at least one owner",
            )

    @staticmethod
    def require_valid_role(role: str) -> None:
        if not is_valid_role(role):
            raise ValidationError(
                code="tenant_member.invalid_role",
                message=f"Invalid tenant role: {role}",
            )

    @staticmethod
    def _not_found() -> NotFoundError:
        return NotFoundError(
            code="tenant_member.not_found",
            message="Tenant membership not found",
        )

    @staticmethod
    def _already_exists() -> ConflictError:
        return ConflictError(
            code="tenant_member.exists",
            message="Tenant membership already exists",
        )


__all__ = [
    "ROLE_ADMIN",
    "ROLE_CONTRIBUTOR",
    "ROLE_OWNER",
    "ROLE_VIEWER",
    "STATUS_ACTIVE",
    "TenantMemberService",
    "has_permission",
    "is_valid_role",
    "role_level",
]
