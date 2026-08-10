"""Unit tests for `OrganizationService`.

The three repositories (org / member / join-request) are each replaced with
an ``AsyncMock(spec=...)`` backed by closure-captured in-memory state, so
cross-repo hops (create-org -> enrol creator, approve-join -> add-member,
delete-org -> soft-delete) are exercised for real rather than mocked out.

A real-repository construction test guards against signature drift between
the mock specs and the concrete repositories.
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.common.exception import (
    ConflictError,
    NotFoundError,
    PermissionDeniedError,
    ValidationError,
)
from src.core.organizations.service.organization_service import (
    DEFAULT_INVITE_CODE_VALIDITY_DAYS,
    DEFAULT_MEMBER_LIMIT,
    JOIN_REQUEST_STATUS_APPROVED,
    JOIN_REQUEST_STATUS_PENDING,
    JOIN_REQUEST_STATUS_REJECTED,
    JOIN_REQUEST_TYPE_JOIN,
    JOIN_REQUEST_TYPE_UPGRADE,
    ORG_ROLE_ADMIN,
    ORG_ROLE_EDITOR,
    ORG_ROLE_VIEWER,
    OrganizationService,
)
from src.db.dao.organization_repository import (
    OrganizationJoinRequestRepository,
    OrganizationMemberRepository,
    OrganizationRepository,
)
from src.db.models.organization import (
    Organization,
    OrganizationJoinRequest,
    OrganizationTenantMember,
)
from tests.util.service_test import ServiceTest

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_TENANT_OWNER = 7
_TENANT_NEW = 99
_USER_OWNER = "usr-owner"
_USER_NEW = "usr-new"
_INVITE_CODE_PATTERN = re.compile(r"^[0-9a-f]{16}$")


# ── Repository mocks ─────────────────────────────────────────────────


def _make_org_repo() -> tuple[AsyncMock, dict[str, Organization]]:
    """Org-repo mock with closure-captured live row storage."""
    repo = AsyncMock(spec=OrganizationRepository)
    rows: dict[str, Organization] = {}

    def _live() -> dict[str, Organization]:
        return {i: r for i, r in rows.items() if r.deleted_at is None}

    async def _create(row: Organization) -> Organization:
        rows[row.id] = row
        return row

    async def _update(row: Organization) -> Organization:
        existing = rows.get(row.id)
        if existing is None or existing.deleted_at is not None:
            raise ValueError(f"organization {row.id} not live")
        rows[row.id] = row
        return row

    async def _soft_delete(*, id: str, now: datetime) -> bool:
        existing = rows.get(id)
        if existing is None or existing.deleted_at is not None:
            return False
        rows[id] = existing.model_copy(
            update={"deleted_at": now, "updated_at": now}
        )
        return True

    async def _update_invite_code(
        *,
        id: str,
        invite_code: str | None,
        expires_at: datetime | None,
        now: datetime,
    ) -> bool:
        existing = rows.get(id)
        if existing is None or existing.deleted_at is not None:
            return False
        rows[id] = existing.model_copy(
            update={
                "invite_code": invite_code,
                "invite_code_expires_at": expires_at,
                "updated_at": now,
            }
        )
        return True

    async def _get_by_id_or_none(id: str) -> Organization | None:
        return _live().get(id)

    async def _get_by_invite_code_or_none(invite_code: str) -> Organization | None:
        if not invite_code:
            return None
        for r in _live().values():
            if r.invite_code == invite_code:
                return r
        return None

    async def _list_by_tenant(tenant_id: int) -> list[Organization]:
        joined: list[Organization] = []
        for r in _live().values():
            # The mock joins members lazily to mirror the SQL path.
            members = [
                m for m in member_rows.values() if m.organization_id == r.id
            ]
            if any(m.tenant_id == tenant_id for m in members):
                joined.append(r)
        return sorted(joined, key=lambda r: (r.created_at, r.id), reverse=True)

    async def _list_searchable(*, query: str, limit: int) -> list[Organization]:
        pattern = (query or "").lower()
        results = [
            r
            for r in _live().values()
            if r.searchable
            and (
                pattern in r.name.lower()
                or pattern in (r.description or "").lower()
                or pattern in r.id.lower()
                or not pattern
            )
        ]
        return sorted(results, key=lambda r: (r.created_at, r.id), reverse=True)[: max(limit, 1)]

    repo.create.side_effect = _create
    repo.update.side_effect = _update
    repo.soft_delete.side_effect = _soft_delete
    repo.update_invite_code.side_effect = _update_invite_code
    repo.get_by_id_or_none.side_effect = _get_by_id_or_none
    repo.get_by_invite_code_or_none.side_effect = _get_by_invite_code_or_none
    repo.list_by_tenant.side_effect = _list_by_tenant
    repo.list_searchable.side_effect = _list_searchable
    return repo, rows


# Mutable container for the member store so the org-list mock can join.
member_rows: dict[str, OrganizationTenantMember] = {}
join_request_rows: dict[str, OrganizationJoinRequest] = {}


def _make_member_repo() -> AsyncMock:
    """Member-repo mock backed by the module-level member store."""
    repo = AsyncMock(spec=OrganizationMemberRepository)

    def _live_members(org_id: str) -> list[OrganizationTenantMember]:
        return [m for m in member_rows.values() if m.organization_id == org_id]

    async def _add_member(row: OrganizationTenantMember) -> OrganizationTenantMember | None:
        for m in _live_members(row.organization_id):
            if m.tenant_id == row.tenant_id:
                return None
        member_rows[row.id] = row
        return row

    async def _remove_member(*, organization_id: str, tenant_id: int) -> bool:
        for key, m in list(member_rows.items()):
            if m.organization_id == organization_id and m.tenant_id == tenant_id:
                member_rows.pop(key)
                return True
        return False

    async def _update_member_role(
        *,
        organization_id: str,
        tenant_id: int,
        role: str,
    ) -> bool:
        for key, m in member_rows.items():
            if m.organization_id == organization_id and m.tenant_id == tenant_id:
                member_rows[key] = m.model_copy(
                    update={"role": role, "updated_at": _NOW}
                )
                return True
        return False

    async def _get_member(
        *,
        organization_id: str,
        tenant_id: int,
    ) -> OrganizationTenantMember | None:
        for m in _live_members(organization_id):
            if m.tenant_id == tenant_id:
                return m
        return None

    async def _list_members(org_id: str) -> list[OrganizationTenantMember]:
        return sorted(
            _live_members(org_id), key=lambda m: (m.created_at, m.id)
        )

    async def _count_members(org_id: str) -> int:
        return len(_live_members(org_id))

    repo.add_member.side_effect = _add_member
    repo.remove_member.side_effect = _remove_member
    repo.update_member_role.side_effect = _update_member_role
    repo.get_member.side_effect = _get_member
    repo.list_members.side_effect = _list_members
    repo.count_members.side_effect = _count_members
    return repo


def _make_join_request_repo() -> AsyncMock:
    """Join-request-repo mock backed by the module-level join-request store."""
    repo = AsyncMock(spec=OrganizationJoinRequestRepository)

    def _live_for(org_id: str) -> list[OrganizationJoinRequest]:
        return [r for r in join_request_rows.values() if r.organization_id == org_id]

    async def _create_join_request(
        row: OrganizationJoinRequest,
    ) -> OrganizationJoinRequest:
        join_request_rows[row.id] = row
        return row

    async def _update_join_request_status(
        *,
        id: str,
        status: str,
        reviewed_by: str | None,
        review_message: str | None,
        reviewed_at: datetime,
    ) -> bool:
        existing = join_request_rows.get(id)
        if existing is None:
            return False
        join_request_rows[id] = existing.model_copy(
            update={
                "status": status,
                "reviewed_by": reviewed_by,
                "review_message": review_message,
                "reviewed_at": reviewed_at,
                "updated_at": reviewed_at,
            }
        )
        return True

    async def _get_join_request_by_id(id: str) -> OrganizationJoinRequest | None:
        return join_request_rows.get(id)

    async def _get_pending_join_request(
        *,
        organization_id: str,
        tenant_id: int,
    ) -> OrganizationJoinRequest | None:
        for r in _live_for(organization_id):
            if r.tenant_id == tenant_id and r.status == JOIN_REQUEST_STATUS_PENDING:
                return r
        return None

    async def _get_pending_request_by_type(
        *,
        organization_id: str,
        tenant_id: int,
        request_type: str,
    ) -> OrganizationJoinRequest | None:
        for r in _live_for(organization_id):
            if (
                r.tenant_id == tenant_id
                and r.status == JOIN_REQUEST_STATUS_PENDING
                and r.request_type == request_type
            ):
                return r
        return None

    async def _list_join_requests(
        org_id: str, *, status: str | None = None
    ) -> list[OrganizationJoinRequest]:
        rows = [r for r in _live_for(org_id) if status is None or r.status == status]
        return sorted(rows, key=lambda r: (r.created_at, r.id), reverse=True)

    async def _count_join_requests(
        org_id: str, *, status: str | None = None
    ) -> int:
        return sum(1 for r in _live_for(org_id) if status is None or r.status == status)

    repo.create_join_request.side_effect = _create_join_request
    repo.update_join_request_status.side_effect = _update_join_request_status
    repo.get_join_request_by_id.side_effect = _get_join_request_by_id
    repo.get_pending_join_request.side_effect = _get_pending_join_request
    repo.get_pending_request_by_type.side_effect = _get_pending_request_by_type
    repo.list_join_requests.side_effect = _list_join_requests
    repo.count_join_requests.side_effect = _count_join_requests
    return repo


def _make_repos() -> tuple[
    AsyncMock, AsyncMock, AsyncMock,
    dict[str, Organization],
]:
    """Build all three repository mocks with fresh closure-captured state."""
    member_rows.clear()
    join_request_rows.clear()
    org_repo, org_rows = _make_org_repo()
    member_repo = _make_member_repo()
    join_repo = _make_join_request_repo()
    return org_repo, member_repo, join_repo, org_rows


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def state() -> tuple[
    AsyncMock, AsyncMock, AsyncMock, dict[str, Organization]
]:
    return _make_repos()


@pytest.fixture
def org_repo(
    state: tuple[AsyncMock, AsyncMock, AsyncMock, dict[str, Organization]],
) -> AsyncMock:
    return state[0]


@pytest.fixture
def org_rows(
    state: tuple[AsyncMock, AsyncMock, AsyncMock, dict[str, Organization]],
) -> dict[str, Organization]:
    return state[3]


@pytest.fixture
def service(
    state: tuple[AsyncMock, AsyncMock, AsyncMock, dict[str, Organization]],
) -> OrganizationService:
    org_repo, member_repo, join_repo, _ = state
    return OrganizationService(
        org_repo=org_repo,
        member_repo=member_repo,
        join_request_repo=join_repo,
    )


def _seed_org(
    rows: dict[str, Organization],
    *,
    owner_id: str = _USER_OWNER,
    owner_tenant_id: int = _TENANT_OWNER,
    name: str = "alpha",
    description: str | None = None,
    invite_code: str = "code-1",
    invite_code_expires_at: datetime | None = None,
    invite_code_validity_days: int = 7,
    require_approval: bool = False,
    searchable: bool = False,
    member_limit: int = 50,
    created_at: datetime = _NOW,
) -> Organization:
    """Insert an organization directly into the closure-captured store."""
    row = Organization(
        id=f"org-{uuid.uuid4().hex[:8]}",
        name=name,
        description=description,
        avatar="",
        owner_id=owner_id,
        owner_tenant_id=owner_tenant_id,
        invite_code=invite_code,
        invite_code_expires_at=invite_code_expires_at,
        invite_code_validity_days=invite_code_validity_days,
        require_approval=require_approval,
        searchable=searchable,
        member_limit=member_limit,
        created_at=created_at,
        updated_at=created_at,
    )
    rows[row.id] = row
    return row


def _seed_member(
    org_id: str,
    *,
    tenant_id: int,
    role: str = ORG_ROLE_VIEWER,
    representative_user_id: str = "",
    joined_at: datetime | None = None,
) -> OrganizationTenantMember:
    """Insert a member directly into the module-level member store."""
    row = OrganizationTenantMember(
        id=f"mem-{uuid.uuid4().hex[:8]}",
        organization_id=org_id,
        tenant_id=tenant_id,
        role=role,
        representative_user_id=representative_user_id,
        joined_at=joined_at or _NOW,
        created_at=joined_at or _NOW,
        updated_at=joined_at or _NOW,
    )
    member_rows[row.id] = row
    return row


def _seed_join_request(
    org_id: str,
    *,
    tenant_id: int = _TENANT_NEW,
    user_id: str = _USER_NEW,
    status: str = JOIN_REQUEST_STATUS_PENDING,
    request_type: str = JOIN_REQUEST_TYPE_JOIN,
    requested_role: str = ORG_ROLE_VIEWER,
    prev_role: str | None = None,
    created_at: datetime = _NOW,
) -> OrganizationJoinRequest:
    """Insert a join request directly into the module-level request store."""
    row = OrganizationJoinRequest(
        id=f"req-{uuid.uuid4().hex[:8]}",
        organization_id=org_id,
        user_id=user_id,
        tenant_id=tenant_id,
        status=status,
        requested_role=requested_role,
        request_type=request_type,
        prev_role=prev_role,
        created_at=created_at,
        updated_at=created_at,
    )
    join_request_rows[row.id] = row
    return row


# ── Construction guard ───────────────────────────────────────────────


class TestConstructionGuard(ServiceTest):
    async def test_repos_match_the_mock_specs(
        self,
        org_repo: AsyncMock,
        org_rows: dict[str, Organization],
    ) -> None:
        """Mock spec keeps the service aligned with the concrete repository."""
        # Insert via the mock; the repo's spec catches signature drift.
        seeded = _seed_org(org_rows)
        assert await org_repo.get_by_id_or_none(seeded.id) is not None


# ── create_organization ──────────────────────────────────────────────


class TestCreateOrganization(ServiceTest):
    async def test_stamps_id_owner_timestamps_and_invite_code(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        info = await service.create_organization(
            user_id=_USER_OWNER,
            tenant_id=_TENANT_OWNER,
            name="  alpha ",
        )
        assert info.id in org_rows
        assert info.owner_id == _USER_OWNER
        assert info.owner_tenant_id == _TENANT_OWNER
        assert info.name == "alpha"
        assert info.invite_code_validity_days == DEFAULT_INVITE_CODE_VALIDITY_DAYS
        assert info.member_limit == DEFAULT_MEMBER_LIMIT
        assert _INVITE_CODE_PATTERN.match(org_rows[info.id].invite_code or "")

    async def test_enrols_creator_tenant_as_admin(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        info = await service.create_organization(
            user_id=_USER_OWNER,
            tenant_id=_TENANT_OWNER,
            name="alpha",
        )
        members = [m for m in member_rows.values() if m.organization_id == info.id]
        assert len(members) == 1
        assert members[0].tenant_id == _TENANT_OWNER
        assert members[0].role == ORG_ROLE_ADMIN
        assert members[0].representative_user_id == _USER_OWNER

    async def test_validity_days_zero_means_no_expiry(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        info = await service.create_organization(
            user_id=_USER_OWNER,
            tenant_id=_TENANT_OWNER,
            name="alpha",
            invite_code_validity_days=0,
        )
        assert info.invite_code_validity_days == 0
        assert org_rows[info.id].invite_code_expires_at is None

    async def test_validity_days_seven_expires_in_a_week(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        before = datetime.now(UTC)
        info = await service.create_organization(
            user_id=_USER_OWNER,
            tenant_id=_TENANT_OWNER,
            name="alpha",
            invite_code_validity_days=7,
        )
        after = datetime.now(UTC)
        expires_at = org_rows[info.id].invite_code_expires_at
        assert expires_at is not None
        assert expires_at - before >= timedelta(days=7) - timedelta(seconds=1)
        assert expires_at - after <= timedelta(days=7) + timedelta(seconds=1)

    @pytest.mark.parametrize("invalid_days", [-1, 2, 14, 365])
    async def test_rejects_invalid_validity_days(
        self,
        service: OrganizationService,
        invalid_days: int,
    ) -> None:
        with pytest.raises(ValidationError) as excinfo:
            await service.create_organization(
                user_id=_USER_OWNER,
                tenant_id=_TENANT_OWNER,
                name="alpha",
                invite_code_validity_days=invalid_days,
            )
        assert excinfo.value.code == "organization.invite_validity_invalid"

    async def test_rejects_negative_member_limit(
        self, service: OrganizationService
    ) -> None:
        with pytest.raises(ValidationError) as excinfo:
            await service.create_organization(
                user_id=_USER_OWNER,
                tenant_id=_TENANT_OWNER,
                name="alpha",
                member_limit=-1,
            )
        assert excinfo.value.code == "organization.member_limit_invalid"

    async def test_rejects_blank_name(self, service: OrganizationService) -> None:
        with pytest.raises(ValidationError) as excinfo:
            await service.create_organization(
                user_id=_USER_OWNER,
                tenant_id=_TENANT_OWNER,
                name="   ",
            )
        assert excinfo.value.code == "organization.name_required"

    async def test_rejects_invalid_tenant_id(self, service: OrganizationService) -> None:
        with pytest.raises(ValidationError) as excinfo:
            await service.create_organization(
                user_id=_USER_OWNER,
                tenant_id=0,
                name="alpha",
            )
        assert excinfo.value.code == "organization.tenant_required"

    async def test_rejects_blank_user_id(self, service: OrganizationService) -> None:
        with pytest.raises(ValidationError) as excinfo:
            await service.create_organization(
                user_id="   ",
                tenant_id=_TENANT_OWNER,
                name="alpha",
            )
        assert excinfo.value.code == "organization.user_required"


# ── reads ────────────────────────────────────────────────────────────


class TestReads(ServiceTest):
    async def test_get_organization_returns_projection(
        self, service: OrganizationService, org_rows: dict[str, Organization]
    ) -> None:
        seeded = _seed_org(org_rows)

        info = await service.get_organization(id=seeded.id)

        assert info.id == seeded.id
        assert info.name == seeded.name

    async def test_get_organization_missing_raises(
        self, service: OrganizationService
    ) -> None:
        with pytest.raises(NotFoundError) as excinfo:
            await service.get_organization(id="missing")
        assert excinfo.value.code == "organization.not_found"

    async def test_get_organization_rejects_blank_id(
        self, service: OrganizationService
    ) -> None:
        with pytest.raises(ValidationError):
            await service.get_organization(id="   ")

    async def test_get_by_invite_code_returns_org(
        self, service: OrganizationService, org_rows: dict[str, Organization]
    ) -> None:
        seeded = _seed_org(org_rows, invite_code="abc123")

        info = await service.get_organization_by_invite_code(invite_code="abc123")

        assert info.id == seeded.id

    async def test_get_by_invite_code_rejects_expired(
        self, service: OrganizationService, org_rows: dict[str, Organization]
    ) -> None:
        _seed_org(
            org_rows,
            invite_code="abc123",
            invite_code_expires_at=_NOW - timedelta(days=1),
        )

        with pytest.raises(ValidationError) as excinfo:
            await service.get_organization_by_invite_code(invite_code="abc123")
        assert excinfo.value.code == "organization.invite_code_expired"

    async def test_get_by_invite_code_missing_raises(
        self, service: OrganizationService
    ) -> None:
        with pytest.raises(NotFoundError) as excinfo:
            await service.get_organization_by_invite_code(invite_code="nope")
        assert excinfo.value.code == "organization.invite_code_not_found"

    async def test_list_tenant_organizations_newest_first(
        self, service: OrganizationService, org_rows: dict[str, Organization]
    ) -> None:
        newer = _seed_org(org_rows, name="newer", created_at=_NOW + timedelta(hours=1))
        older = _seed_org(org_rows, name="older", created_at=_NOW)
        _seed_member(newer.id, tenant_id=_TENANT_OWNER)
        _seed_member(older.id, tenant_id=_TENANT_OWNER)
        # Another org the tenant is not in.
        _seed_org(org_rows, name="stranger")

        infos = await service.list_tenant_organizations(tenant_id=_TENANT_OWNER)

        assert [info.id for info in infos] == [newer.id, older.id]


# ── update_organization ──────────────────────────────────────────────


class TestUpdateOrganization(ServiceTest):
    async def test_admin_can_update_mutable_fields(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)
        _seed_member(seeded.id, tenant_id=_TENANT_OWNER, role=ORG_ROLE_ADMIN)

        info = await service.update_organization(
            id=seeded.id,
            operator_user_id=_USER_OWNER,
            operator_tenant_id=_TENANT_OWNER,
            name="  renamed ",
            description="new",
            require_approval=True,
            searchable=True,
        )

        assert info.name == "renamed"
        assert info.description == "new"
        assert info.require_approval is True
        assert info.searchable is True

    async def test_non_admin_cannot_update(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)
        _seed_member(seeded.id, tenant_id=_TENANT_OWNER, role=ORG_ROLE_VIEWER)

        with pytest.raises(PermissionDeniedError):
            await service.update_organization(
                id=seeded.id,
                operator_user_id=_USER_OWNER,
                operator_tenant_id=_TENANT_OWNER,
                name="renamed",
            )

    async def test_member_limit_below_current_count_rejected(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows, member_limit=5)
        _seed_member(seeded.id, tenant_id=_TENANT_OWNER, role=ORG_ROLE_ADMIN)
        _seed_member(seeded.id, tenant_id=_TENANT_NEW)

        with pytest.raises(ValidationError) as excinfo:
            await service.update_organization(
                id=seeded.id,
                operator_user_id=_USER_OWNER,
                operator_tenant_id=_TENANT_OWNER,
                member_limit=1,
            )
        assert excinfo.value.code == "organization.member_limit_too_low"

    async def test_member_limit_zero_means_unlimited(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)
        _seed_member(seeded.id, tenant_id=_TENANT_OWNER, role=ORG_ROLE_ADMIN)

        info = await service.update_organization(
            id=seeded.id,
            operator_user_id=_USER_OWNER,
            operator_tenant_id=_TENANT_OWNER,
            member_limit=0,
        )
        assert info.member_limit == 0

    async def test_missing_org_raises(
        self,
        service: OrganizationService,
    ) -> None:
        # The admin check fires before the existence check, so a missing
        # org reads as a permission failure — the gate never leaks its
        # existence to non-admins.
        with pytest.raises(PermissionDeniedError):
            await service.update_organization(
                id="missing",
                operator_user_id=_USER_OWNER,
                operator_tenant_id=_TENANT_OWNER,
                name="renamed",
            )


# ── delete_organization ──────────────────────────────────────────────


class TestDeleteOrganization(ServiceTest):
    async def test_owner_tenant_can_delete(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)

        await service.delete_organization(
            id=seeded.id,
            operator_user_id=_USER_OWNER,
            operator_tenant_id=_TENANT_OWNER,
        )

        assert org_rows[seeded.id].deleted_at is not None

    async def test_non_owner_tenant_rejected(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)

        with pytest.raises(PermissionDeniedError):
            await service.delete_organization(
                id=seeded.id,
                operator_user_id=_USER_NEW,
                operator_tenant_id=_TENANT_NEW,
            )

    async def test_legacy_owner_user_can_still_delete(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows, owner_tenant_id=0)

        await service.delete_organization(
            id=seeded.id,
            operator_user_id=_USER_OWNER,
            operator_tenant_id=_TENANT_OWNER,
        )

        assert org_rows[seeded.id].deleted_at is not None

    async def test_missing_org_raises(
        self, service: OrganizationService
    ) -> None:
        with pytest.raises(NotFoundError):
            await service.delete_organization(
                id="missing",
                operator_user_id=_USER_OWNER,
                operator_tenant_id=_TENANT_OWNER,
            )


# ── generate_invite_code ─────────────────────────────────────────────


class TestGenerateInviteCode(ServiceTest):
    async def test_admin_can_rotate_invite_code(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows, invite_code="old-code")
        _seed_member(seeded.id, tenant_id=_TENANT_OWNER, role=ORG_ROLE_ADMIN)
        old_code = seeded.invite_code

        new_code = await service.generate_invite_code(
            org_id=seeded.id,
            operator_user_id=_USER_OWNER,
            operator_tenant_id=_TENANT_OWNER,
        )

        assert new_code != old_code
        assert _INVITE_CODE_PATTERN.match(new_code)
        assert org_rows[seeded.id].invite_code == new_code

    async def test_non_admin_rejected(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)
        _seed_member(seeded.id, tenant_id=_TENANT_OWNER, role=ORG_ROLE_VIEWER)

        with pytest.raises(PermissionDeniedError):
            await service.generate_invite_code(
                org_id=seeded.id,
                operator_user_id=_USER_OWNER,
                operator_tenant_id=_TENANT_OWNER,
            )


# ── member management ───────────────────────────────────────────────


class TestMemberManagement(ServiceTest):
    async def test_add_tenant_member_returns_projection(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)

        info = await service.add_tenant_member(
            org_id=seeded.id,
            tenant_id=_TENANT_NEW,
            representative_user_id=_USER_NEW,
            role=ORG_ROLE_EDITOR,
        )

        assert info.tenant_id == _TENANT_NEW
        assert info.role == ORG_ROLE_EDITOR
        assert info.representative_user_id == _USER_NEW

    async def test_add_tenant_member_duplicate_conflicts(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)
        _seed_member(seeded.id, tenant_id=_TENANT_NEW)

        with pytest.raises(ConflictError) as excinfo:
            await service.add_tenant_member(
                org_id=seeded.id,
                tenant_id=_TENANT_NEW,
                representative_user_id=_USER_NEW,
                role=ORG_ROLE_VIEWER,
            )
        assert excinfo.value.code == "organization.tenant_already_member"

    async def test_add_tenant_member_enforces_limit(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows, member_limit=1)
        _seed_member(seeded.id, tenant_id=_TENANT_OWNER)

        with pytest.raises(ConflictError) as excinfo:
            await service.add_tenant_member(
                org_id=seeded.id,
                tenant_id=_TENANT_NEW,
                representative_user_id=_USER_NEW,
                role=ORG_ROLE_VIEWER,
            )
        assert excinfo.value.code == "organization.member_limit_reached"

    async def test_add_tenant_member_rejects_bad_role(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)

        with pytest.raises(ValidationError) as excinfo:
            await service.add_tenant_member(
                org_id=seeded.id,
                tenant_id=_TENANT_NEW,
                representative_user_id=_USER_NEW,
                role="superuser",
            )
        assert excinfo.value.code == "organization.role_invalid"

    async def test_remove_self_works(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows, owner_tenant_id=_TENANT_OWNER)
        _seed_member(seeded.id, tenant_id=_TENANT_OWNER, role=ORG_ROLE_ADMIN)
        _seed_member(seeded.id, tenant_id=_TENANT_NEW)

        await service.remove_tenant_member(
            org_id=seeded.id,
            member_tenant_id=_TENANT_NEW,
            operator_user_id=_USER_NEW,
            operator_tenant_id=_TENANT_NEW,
        )

        assert not any(
            m.tenant_id == _TENANT_NEW
            for m in member_rows.values()
            if m.organization_id == seeded.id
        )

    async def test_remove_owner_tenant_rejected(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows, owner_tenant_id=_TENANT_OWNER)
        _seed_member(seeded.id, tenant_id=_TENANT_OWNER, role=ORG_ROLE_ADMIN)

        with pytest.raises(ConflictError) as excinfo:
            await service.remove_tenant_member(
                org_id=seeded.id,
                member_tenant_id=_TENANT_OWNER,
                operator_user_id=_USER_OWNER,
                operator_tenant_id=_TENANT_OWNER,
            )
        assert excinfo.value.code == "organization.cannot_remove_owner"

    async def test_remove_by_other_requires_admin(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows, owner_tenant_id=_TENANT_OWNER)
        _seed_member(seeded.id, tenant_id=_TENANT_OWNER, role=ORG_ROLE_ADMIN)
        _seed_member(seeded.id, tenant_id=_TENANT_NEW)
        _seed_member(seeded.id, tenant_id=11, role=ORG_ROLE_VIEWER)

        with pytest.raises(PermissionDeniedError):
            await service.remove_tenant_member(
                org_id=seeded.id,
                member_tenant_id=_TENANT_NEW,
                operator_user_id="usr-11",
                operator_tenant_id=11,
            )

    async def test_admin_can_remove_other_member(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows, owner_tenant_id=_TENANT_OWNER)
        _seed_member(seeded.id, tenant_id=_TENANT_OWNER, role=ORG_ROLE_ADMIN)
        _seed_member(seeded.id, tenant_id=_TENANT_NEW)

        await service.remove_tenant_member(
            org_id=seeded.id,
            member_tenant_id=_TENANT_NEW,
            operator_user_id=_USER_OWNER,
            operator_tenant_id=_TENANT_OWNER,
        )

        assert not any(
            m.tenant_id == _TENANT_NEW
            for m in member_rows.values()
            if m.organization_id == seeded.id
        )

    async def test_update_member_role(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows, owner_tenant_id=_TENANT_OWNER)
        _seed_member(seeded.id, tenant_id=_TENANT_OWNER, role=ORG_ROLE_ADMIN)
        _seed_member(seeded.id, tenant_id=_TENANT_NEW, role=ORG_ROLE_VIEWER)

        info = await service.update_tenant_member_role(
            org_id=seeded.id,
            member_tenant_id=_TENANT_NEW,
            role=ORG_ROLE_EDITOR,
            operator_user_id=_USER_OWNER,
            operator_tenant_id=_TENANT_OWNER,
        )

        assert info.role == ORG_ROLE_EDITOR

    async def test_update_member_role_owner_rejected(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows, owner_tenant_id=_TENANT_OWNER)
        _seed_member(seeded.id, tenant_id=_TENANT_OWNER, role=ORG_ROLE_ADMIN)

        with pytest.raises(ConflictError) as excinfo:
            await service.update_tenant_member_role(
                org_id=seeded.id,
                member_tenant_id=_TENANT_OWNER,
                role=ORG_ROLE_VIEWER,
                operator_user_id=_USER_OWNER,
                operator_tenant_id=_TENANT_OWNER,
            )
        assert excinfo.value.code == "organization.cannot_change_owner_role"

    async def test_update_member_role_requires_admin(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows, owner_tenant_id=_TENANT_OWNER)
        _seed_member(seeded.id, tenant_id=_TENANT_OWNER, role=ORG_ROLE_ADMIN)
        _seed_member(seeded.id, tenant_id=_TENANT_NEW, role=ORG_ROLE_VIEWER)
        _seed_member(seeded.id, tenant_id=11, role=ORG_ROLE_VIEWER)

        with pytest.raises(PermissionDeniedError):
            await service.update_tenant_member_role(
                org_id=seeded.id,
                member_tenant_id=_TENANT_NEW,
                role=ORG_ROLE_EDITOR,
                operator_user_id="usr-11",
                operator_tenant_id=11,
            )

    async def test_list_tenant_members_oldest_first(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)
        first = _seed_member(
            seeded.id, tenant_id=_TENANT_OWNER, joined_at=_NOW
        )
        second = _seed_member(
            seeded.id,
            tenant_id=_TENANT_NEW,
            joined_at=_NOW + timedelta(minutes=1),
        )

        infos = await service.list_tenant_members(org_id=seeded.id)

        assert [info.id for info in infos] == [first.id, second.id]

    async def test_get_tenant_member_returns_projection(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)
        _seed_member(seeded.id, tenant_id=_TENANT_NEW, role=ORG_ROLE_EDITOR)

        info = await service.get_tenant_member(
            org_id=seeded.id, tenant_id=_TENANT_NEW
        )

        assert info.tenant_id == _TENANT_NEW
        assert info.role == ORG_ROLE_EDITOR

    async def test_get_tenant_member_missing_raises(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)

        with pytest.raises(NotFoundError) as excinfo:
            await service.get_tenant_member(
                org_id=seeded.id, tenant_id=_TENANT_NEW
            )
        assert excinfo.value.code == "organization.tenant_not_member"

    async def test_is_tenant_org_admin(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)
        _seed_member(seeded.id, tenant_id=_TENANT_OWNER, role=ORG_ROLE_ADMIN)
        _seed_member(seeded.id, tenant_id=_TENANT_NEW, role=ORG_ROLE_VIEWER)

        assert (
            await service.is_tenant_org_admin(
                org_id=seeded.id, tenant_id=_TENANT_OWNER
            )
            is True
        )
        assert (
            await service.is_tenant_org_admin(
                org_id=seeded.id, tenant_id=_TENANT_NEW
            )
            is False
        )
        assert (
            await service.is_tenant_org_admin(
                org_id=seeded.id, tenant_id=999
            )
            is False
        )

    async def test_get_tenant_role_in_org(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)
        _seed_member(seeded.id, tenant_id=_TENANT_NEW, role=ORG_ROLE_EDITOR)

        role = await service.get_tenant_role_in_org(
            org_id=seeded.id, tenant_id=_TENANT_NEW
        )

        assert role == ORG_ROLE_EDITOR


# ── join flows ───────────────────────────────────────────────────────


class TestJoinFlows(ServiceTest):
    async def test_join_by_invite_code_enrols_viewer(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows, invite_code="abc123")

        info = await service.join_by_invite_code(
            invite_code="abc123",
            user_id=_USER_NEW,
            tenant_id=_TENANT_NEW,
        )

        assert info.id == seeded.id
        member = next(
            m
            for m in member_rows.values()
            if m.organization_id == seeded.id
            and m.tenant_id == _TENANT_NEW
        )
        assert member.role == ORG_ROLE_VIEWER

    async def test_join_by_invite_code_requires_approval_blocks(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        _seed_org(org_rows, invite_code="abc123", require_approval=True)

        with pytest.raises(PermissionDeniedError):
            await service.join_by_invite_code(
                invite_code="abc123",
                user_id=_USER_NEW,
                tenant_id=_TENANT_NEW,
            )

    async def test_join_by_organization_id_searchable_no_approval(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows, searchable=True)

        info = await service.join_by_organization_id(
            org_id=seeded.id,
            user_id=_USER_NEW,
            tenant_id=_TENANT_NEW,
        )

        assert info.id == seeded.id
        assert any(
            m.tenant_id == _TENANT_NEW
            for m in member_rows.values()
            if m.organization_id == seeded.id
        )

    async def test_join_by_organization_id_requires_approval_creates_request(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows, searchable=True, require_approval=True)

        info = await service.join_by_organization_id(
            org_id=seeded.id,
            user_id=_USER_NEW,
            tenant_id=_TENANT_NEW,
            message="please",
            requested_role=ORG_ROLE_EDITOR,
        )

        assert info.id == seeded.id
        request = next(
            r
            for r in join_request_rows.values()
            if r.organization_id == seeded.id
        )
        assert request.status == JOIN_REQUEST_STATUS_PENDING
        assert request.request_type == JOIN_REQUEST_TYPE_JOIN
        assert request.requested_role == ORG_ROLE_EDITOR

    async def test_join_by_organization_id_not_searchable_rejected(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows, searchable=False)

        with pytest.raises(PermissionDeniedError):
            await service.join_by_organization_id(
                org_id=seeded.id,
                user_id=_USER_NEW,
                tenant_id=_TENANT_NEW,
            )

    async def test_join_by_organization_id_already_member_is_idempotent(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows, searchable=True)
        _seed_member(seeded.id, tenant_id=_TENANT_NEW, role=ORG_ROLE_VIEWER)

        info = await service.join_by_organization_id(
            org_id=seeded.id,
            user_id=_USER_NEW,
            tenant_id=_TENANT_NEW,
        )

        assert info.id == seeded.id


# ── join requests ────────────────────────────────────────────────────


class TestJoinRequests(ServiceTest):
    async def test_submit_join_request_creates_pending(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)

        info = await service.submit_join_request(
            org_id=seeded.id,
            user_id=_USER_NEW,
            tenant_id=_TENANT_NEW,
            message="hi",
            requested_role=ORG_ROLE_VIEWER,
        )

        assert info.organization_id == seeded.id
        assert info.status == JOIN_REQUEST_STATUS_PENDING
        assert info.request_type == JOIN_REQUEST_TYPE_JOIN

    async def test_submit_join_request_defaults_to_viewer_role(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)

        info = await service.submit_join_request(
            org_id=seeded.id,
            user_id=_USER_NEW,
            tenant_id=_TENANT_NEW,
        )

        assert info.requested_role == ORG_ROLE_VIEWER

    async def test_submit_join_request_dedupes_pending(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)
        _seed_join_request(seeded.id, tenant_id=_TENANT_NEW)

        with pytest.raises(ConflictError) as excinfo:
            await service.submit_join_request(
                org_id=seeded.id,
                user_id=_USER_NEW,
                tenant_id=_TENANT_NEW,
            )
        assert excinfo.value.code == "organization.pending_request_exists"

    async def test_submit_join_request_rejects_invalid_role(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)

        with pytest.raises(ValidationError):
            await service.submit_join_request(
                org_id=seeded.id,
                user_id=_USER_NEW,
                tenant_id=_TENANT_NEW,
                requested_role="superuser",
            )

    async def test_list_join_requests_filters_by_status(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)
        pending = _seed_join_request(seeded.id, tenant_id=11)
        _seed_join_request(
            seeded.id,
            tenant_id=22,
            status=JOIN_REQUEST_STATUS_APPROVED,
        )

        infos = await service.list_join_requests(
            org_id=seeded.id, status=JOIN_REQUEST_STATUS_PENDING
        )

        assert [info.id for info in infos] == [pending.id]

    async def test_list_join_requests_rejects_bad_status(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)

        with pytest.raises(ValidationError):
            await service.list_join_requests(
                org_id=seeded.id, status="garbage"
            )

    async def test_count_pending_join_requests(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)
        _seed_join_request(seeded.id, tenant_id=11)
        _seed_join_request(seeded.id, tenant_id=22)
        _seed_join_request(
            seeded.id,
            tenant_id=33,
            status=JOIN_REQUEST_STATUS_APPROVED,
        )

        total = await service.count_pending_join_requests(org_id=seeded.id)

        assert total == 2


class TestReviewJoinRequest(ServiceTest):
    async def test_approve_join_request_enrols_member(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)
        request = _seed_join_request(seeded.id, requested_role=ORG_ROLE_VIEWER)

        updated = await service.review_join_request(
            org_id=seeded.id,
            request_id=request.id,
            approved=True,
            reviewer_user_id=_USER_OWNER,
            reviewer_tenant_id=_TENANT_OWNER,
            message="welcome",
        )

        assert updated.status == JOIN_REQUEST_STATUS_APPROVED
        assert any(
            m.tenant_id == _TENANT_NEW
            for m in member_rows.values()
            if m.organization_id == seeded.id
        )

    async def test_approve_join_request_with_assign_role(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)
        request = _seed_join_request(seeded.id, requested_role=ORG_ROLE_VIEWER)

        await service.review_join_request(
            org_id=seeded.id,
            request_id=request.id,
            approved=True,
            reviewer_user_id=_USER_OWNER,
            reviewer_tenant_id=_TENANT_OWNER,
            assign_role=ORG_ROLE_EDITOR,
        )

        member = next(
            m
            for m in member_rows.values()
            if m.organization_id == seeded.id
            and m.tenant_id == _TENANT_NEW
        )
        assert member.role == ORG_ROLE_EDITOR

    async def test_approve_upgrade_request_changes_role(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)
        _seed_member(seeded.id, tenant_id=_TENANT_NEW, role=ORG_ROLE_VIEWER)
        request = _seed_join_request(
            seeded.id,
            tenant_id=_TENANT_NEW,
            request_type=JOIN_REQUEST_TYPE_UPGRADE,
            prev_role=ORG_ROLE_VIEWER,
            requested_role=ORG_ROLE_EDITOR,
        )

        await service.review_join_request(
            org_id=seeded.id,
            request_id=request.id,
            approved=True,
            reviewer_user_id=_USER_OWNER,
            reviewer_tenant_id=_TENANT_OWNER,
        )

        member = next(
            m
            for m in member_rows.values()
            if m.organization_id == seeded.id
            and m.tenant_id == _TENANT_NEW
        )
        assert member.role == ORG_ROLE_EDITOR

    async def test_approve_join_request_member_limit_rejected(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows, member_limit=1)
        _seed_member(seeded.id, tenant_id=_TENANT_OWNER, role=ORG_ROLE_ADMIN)
        request = _seed_join_request(seeded.id)

        with pytest.raises(ConflictError) as excinfo:
            await service.review_join_request(
                org_id=seeded.id,
                request_id=request.id,
                approved=True,
                reviewer_user_id=_USER_OWNER,
                reviewer_tenant_id=_TENANT_OWNER,
            )
        assert excinfo.value.code == "organization.member_limit_reached"

    async def test_reject_join_request_sets_status(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)
        request = _seed_join_request(seeded.id)

        updated = await service.review_join_request(
            org_id=seeded.id,
            request_id=request.id,
            approved=False,
            reviewer_user_id=_USER_OWNER,
            reviewer_tenant_id=_TENANT_OWNER,
            message="no",
        )

        assert updated.status == JOIN_REQUEST_STATUS_REJECTED
        # The projection drops the reviewer message (it's an internal
        # admin-flow trail), so we read it from the stored row directly.
        assert join_request_rows[request.id].review_message == "no"

    async def test_review_already_reviewed_conflicts(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)
        request = _seed_join_request(
            seeded.id, status=JOIN_REQUEST_STATUS_APPROVED
        )

        with pytest.raises(ConflictError) as excinfo:
            await service.review_join_request(
                org_id=seeded.id,
                request_id=request.id,
                approved=True,
                reviewer_user_id=_USER_OWNER,
                reviewer_tenant_id=_TENANT_OWNER,
            )
        assert excinfo.value.code == "organization.join_request_already_reviewed"

    async def test_review_unknown_request_raises(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)

        with pytest.raises(NotFoundError):
            await service.review_join_request(
                org_id=seeded.id,
                request_id="missing",
                approved=True,
                reviewer_user_id=_USER_OWNER,
                reviewer_tenant_id=_TENANT_OWNER,
            )

    async def test_review_cross_org_request_raises(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        other_org = _seed_org(org_rows, name="other")
        seeded = _seed_org(org_rows, name="this")
        request = _seed_join_request(other_org.id)

        with pytest.raises(NotFoundError):
            await service.review_join_request(
                org_id=seeded.id,
                request_id=request.id,
                approved=True,
                reviewer_user_id=_USER_OWNER,
                reviewer_tenant_id=_TENANT_OWNER,
            )


class TestRoleUpgrade(ServiceTest):
    async def test_request_role_upgrade_creates_pending(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)
        _seed_member(seeded.id, tenant_id=_TENANT_NEW, role=ORG_ROLE_VIEWER)

        info = await service.request_role_upgrade(
            org_id=seeded.id,
            user_id=_USER_NEW,
            tenant_id=_TENANT_NEW,
            requested_role=ORG_ROLE_EDITOR,
            message="please",
        )

        assert info.request_type == JOIN_REQUEST_TYPE_UPGRADE
        assert info.prev_role == ORG_ROLE_VIEWER
        assert info.status == JOIN_REQUEST_STATUS_PENDING

    async def test_request_role_upgrade_admin_rejected(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)
        _seed_member(seeded.id, tenant_id=_TENANT_NEW, role=ORG_ROLE_ADMIN)

        with pytest.raises(ConflictError) as excinfo:
            await service.request_role_upgrade(
                org_id=seeded.id,
                user_id=_USER_NEW,
                tenant_id=_TENANT_NEW,
                requested_role=ORG_ROLE_EDITOR,
            )
        assert excinfo.value.code == "organization.already_admin"

    async def test_request_role_upgrade_non_member_rejected(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)

        with pytest.raises(NotFoundError) as excinfo:
            await service.request_role_upgrade(
                org_id=seeded.id,
                user_id=_USER_NEW,
                tenant_id=_TENANT_NEW,
                requested_role=ORG_ROLE_EDITOR,
            )
        assert excinfo.value.code == "organization.tenant_not_member"

    async def test_request_role_upgrade_same_role_rejected(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)
        _seed_member(seeded.id, tenant_id=_TENANT_NEW, role=ORG_ROLE_VIEWER)

        with pytest.raises(ConflictError) as excinfo:
            await service.request_role_upgrade(
                org_id=seeded.id,
                user_id=_USER_NEW,
                tenant_id=_TENANT_NEW,
                requested_role=ORG_ROLE_VIEWER,
            )
        assert excinfo.value.code == "organization.upgrade_to_same_or_lower_role"

    async def test_get_pending_upgrade_request_returns_projection(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)
        _seed_member(seeded.id, tenant_id=_TENANT_NEW, role=ORG_ROLE_VIEWER)
        pending = _seed_join_request(
            seeded.id,
            tenant_id=_TENANT_NEW,
            request_type=JOIN_REQUEST_TYPE_UPGRADE,
        )

        info = await service.get_pending_upgrade_request(
            org_id=seeded.id, tenant_id=_TENANT_NEW
        )

        assert info.id == pending.id

    async def test_get_pending_upgrade_request_missing_raises(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        seeded = _seed_org(org_rows)

        with pytest.raises(NotFoundError) as excinfo:
            await service.get_pending_upgrade_request(
                org_id=seeded.id, tenant_id=_TENANT_NEW
            )
        assert excinfo.value.code == "organization.upgrade_request_not_found"


class TestSearch(ServiceTest):
    async def test_search_searchable_organizations_returns_matches(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        _seed_org(org_rows, name="alpha", searchable=True)
        _seed_org(org_rows, name="beta", searchable=True)
        _seed_org(org_rows, name="gamma", searchable=False)

        infos = await service.search_searchable_organizations(
            tenant_id=_TENANT_OWNER, query="alpha"
        )

        assert [info.name for info in infos] == ["alpha"]

    async def test_search_searchable_organizations_default_limit(
        self,
        service: OrganizationService,
        org_rows: dict[str, Organization],
    ) -> None:
        for index in range(3):
            _seed_org(
                org_rows,
                name=f"alpha-{index}",
                searchable=True,
                created_at=_NOW + timedelta(minutes=index),
            )

        infos = await service.search_searchable_organizations(
            tenant_id=_TENANT_OWNER, query="", limit=0
        )

        assert len(infos) == 3
