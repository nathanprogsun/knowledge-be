"""Unit tests for `TenantMemberService` and the role hierarchy helpers.

The service is exercised against an
``AsyncMock(spec=TenantMemberRepository)`` with closure-captured state
so membership uniqueness, ordering and live/soft-deleted filtering are
preserved.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from src.common.exception import ConflictError, NotFoundError, ValidationError
from src.core.tenants.member_service import (
    ROLE_ADMIN,
    ROLE_CONTRIBUTOR,
    ROLE_OWNER,
    ROLE_VIEWER,
    TenantMemberService,
    has_permission,
    is_valid_role,
    role_level,
)
from src.db.dao.tenant_members_repository import TenantMemberRepository
from src.db.models.tenants.tenant_members import TenantMember
from tests.util.service_test import ServiceTest

_TENANT_ID = 7
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _make_repo() -> tuple[AsyncMock, dict[int, TenantMember], dict[str, str]]:
    """Tenant-member repo mock with closure-captured live row storage."""
    repo = AsyncMock(spec=TenantMemberRepository)
    rows: dict[int, TenantMember] = {}
    user_index: dict[str, str] = {}
    _next_id = {"value": 0}

    def _live() -> list[TenantMember]:
        return [r for r in rows.values() if r.deleted_at is None]

    @staticmethod
    def _sorted(rs: list[TenantMember]) -> list[TenantMember]:
        return sorted(rs, key=lambda r: (r.joined_at, r.id))

    async def _insert_live_or_none(row: TenantMember) -> TenantMember | None:
        for existing in _live():
            if existing.user_id == row.user_id and existing.tenant_id == row.tenant_id:
                return None
        _next_id["value"] += 1
        stored = row.model_copy(update={"id": _next_id["value"]})
        rows[stored.id] = stored
        return stored

    async def _update_role(
        *,
        user_id: str,
        tenant_id: int,
        role: str,
        updated_at: datetime,
    ) -> int:
        for existing in _live():
            if existing.user_id == user_id and existing.tenant_id == tenant_id:
                rows[existing.id] = existing.model_copy(
                    update={"role": role, "updated_at": updated_at}
                )
                return 1
        return 0

    async def _soft_delete(
        *,
        user_id: str,
        tenant_id: int,
        deleted_at: datetime,
    ) -> int:
        for existing in _live():
            if existing.user_id == user_id and existing.tenant_id == tenant_id:
                rows[existing.id] = existing.model_copy(
                    update={"deleted_at": deleted_at, "updated_at": deleted_at}
                )
                return 1
        return 0

    async def _soft_delete_by_tenant(
        *,
        tenant_id: int,
        deleted_at: datetime,
    ) -> int:
        affected = 0
        for k, r in list(rows.items()):
            if r.tenant_id == tenant_id and r.deleted_at is None:
                rows[k] = r.model_copy(
                    update={"deleted_at": deleted_at, "updated_at": deleted_at}
                )
                affected += 1
        return affected

    async def _find_membership(
        *, user_id: str, tenant_id: int
    ) -> TenantMember | None:
        for r in _live():
            if r.user_id == user_id and r.tenant_id == tenant_id:
                return r
        return None

    async def _list_by_user(user_id: str) -> list[TenantMember]:
        return _sorted([r for r in _live() if r.user_id == user_id])

    async def _list_by_tenant(tenant_id: int) -> list[TenantMember]:
        return _sorted([r for r in _live() if r.tenant_id == tenant_id])

    async def _has_any_members(tenant_id: int) -> bool:
        return any(
            r.tenant_id == tenant_id and r.status == "active" for r in _live()
        )

    def _matching(tenant_id: int, search: str | None) -> list[TenantMember]:
        rs = [r for r in _live() if r.tenant_id == tenant_id]
        term = (search or "").strip().lower()
        if term:
            rs = [r for r in rs if term in user_index.get(r.user_id, "").lower()]
        return _sorted(rs)

    async def _count_by_tenant(
        tenant_id: int, *, search: str | None = None
    ) -> int:
        return len(_matching(tenant_id, search))

    async def _list_page_by_tenant(
        tenant_id: int,
        *,
        search: str | None = None,
        limit: int,
        offset: int,
    ) -> list[TenantMember]:
        return _matching(tenant_id, search)[offset : offset + limit]

    async def _count_active_owners(tenant_id: int) -> int:
        return len(
            [
                r
                for r in _live()
                if r.tenant_id == tenant_id and r.role == "owner" and r.status == "active"
            ]
        )

    async def _count_other_active_owners_for_update(
        *,
        tenant_id: int,
        exclude_user_id: str,
    ) -> int:
        return len(
            [
                r
                for r in _live()
                if r.tenant_id == tenant_id
                and r.user_id != exclude_user_id
                and r.role == "owner"
                and r.status == "active"
            ]
        )

    repo.insert_live_or_none.side_effect = _insert_live_or_none
    repo.update_role.side_effect = _update_role
    repo.soft_delete.side_effect = _soft_delete
    repo.soft_delete_by_tenant.side_effect = _soft_delete_by_tenant
    repo.find_membership.side_effect = _find_membership
    repo.list_by_user.side_effect = _list_by_user
    repo.list_by_tenant.side_effect = _list_by_tenant
    repo.has_any_members.side_effect = _has_any_members
    repo.count_by_tenant.side_effect = _count_by_tenant
    repo.list_page_by_tenant.side_effect = _list_page_by_tenant
    repo.count_active_owners.side_effect = _count_active_owners
    repo.count_other_active_owners_for_update.side_effect = (
        _count_other_active_owners_for_update
    )
    return repo, rows, user_index


@pytest.fixture
def state() -> tuple[AsyncMock, dict[int, TenantMember], dict[str, str]]:
    return _make_repo()


@pytest.fixture
def repo(state: tuple[AsyncMock, dict[int, TenantMember], dict[str, str]]) -> AsyncMock:
    return state[0]


@pytest.fixture
def rows(state: tuple[AsyncMock, dict[int, TenantMember], dict[str, str]]) -> dict[int, TenantMember]:
    return state[1]


@pytest.fixture
def user_index(state: tuple[AsyncMock, dict[int, TenantMember], dict[str, str]]) -> dict[str, str]:
    return state[2]


@pytest.fixture
def service(repo: AsyncMock) -> TenantMemberService:
    return TenantMemberService(members_repo=repo)


def _seed(
    rows: dict[int, TenantMember],
    *,
    user_id: str,
    role: str = ROLE_CONTRIBUTOR,
    tenant_id: int = _TENANT_ID,
    joined_at: datetime = _NOW,
) -> TenantMember:
    """Insert directly into the closure-captured store."""
    next_id = max(rows.keys(), default=0) + 1
    row = TenantMember(
        id=next_id,
        user_id=user_id,
        tenant_id=tenant_id,
        role=role,
        joined_at=joined_at,
        created_at=joined_at,
        updated_at=joined_at,
    )
    rows[next_id] = row
    return row


# ── role hierarchy ──────────────────────────────────────────────────


class TestRoleHierarchy(ServiceTest):
    def test_known_roles_are_valid(self) -> None:
        for role in (ROLE_OWNER, ROLE_ADMIN, ROLE_CONTRIBUTOR, ROLE_VIEWER):
            assert is_valid_role(role) is True

    def test_unknown_role_is_invalid_and_ranks_lowest(self) -> None:
        assert is_valid_role("superuser") is False
        assert role_level("superuser") == 0

    def test_hierarchy_is_ordered(self) -> None:
        assert role_level(ROLE_OWNER) > role_level(ROLE_ADMIN)
        assert role_level(ROLE_ADMIN) > role_level(ROLE_CONTRIBUTOR)
        assert role_level(ROLE_CONTRIBUTOR) > role_level(ROLE_VIEWER)

    def test_has_permission_compares_levels(self) -> None:
        assert has_permission(ROLE_ADMIN, ROLE_CONTRIBUTOR) is True
        assert has_permission(ROLE_ADMIN, ROLE_ADMIN) is True
        assert has_permission(ROLE_VIEWER, ROLE_ADMIN) is False
        assert has_permission("nonsense", ROLE_VIEWER) is False


# ── add_member ──────────────────────────────────────────────────────


class TestAddMember(ServiceTest):
    async def test_creates_active_membership(self, service: TenantMemberService) -> None:
        member = await service.add_member(
            user_id="usr-1",
            tenant_id=_TENANT_ID,
            role=ROLE_CONTRIBUTOR,
        )
        assert member.user_id == "usr-1"
        assert member.status == "active"
        assert member.role == ROLE_CONTRIBUTOR


# ── ensure_owner ────────────────────────────────────────────────────


class TestEnsureOwner(ServiceTest):
    async def test_creates_the_owner_row(self, service: TenantMemberService) -> None:
        member = await service.ensure_owner(user_id="usr-1", tenant_id=_TENANT_ID)
        assert member.role == ROLE_OWNER

    async def test_is_idempotent(
        self, service: TenantMemberService, rows: dict[int, TenantMember]
    ) -> None:
        existing = _seed(rows, user_id="usr-1", role=ROLE_VIEWER)

        member = await service.ensure_owner(user_id="usr-1", tenant_id=_TENANT_ID)

        assert member.id == existing.id
        assert member.role == ROLE_VIEWER


# ── reads ───────────────────────────────────────────────────────────


class TestReads(ServiceTest):
    async def test_get_membership_returns_none_when_absent(
        self, service: TenantMemberService
    ) -> None:
        assert (
            await service.get_membership(user_id="ghost", tenant_id=_TENANT_ID) is None
        )

    async def test_list_by_user_returns_every_workspace(
        self, service: TenantMemberService, rows: dict[int, TenantMember]
    ) -> None:
        _seed(rows, user_id="usr-1", tenant_id=_TENANT_ID)
        _seed(rows, user_id="usr-1", tenant_id=_TENANT_ID + 1)

        memberships = await service.list_by_user("usr-1")

        assert {m.tenant_id for m in memberships} == {_TENANT_ID, _TENANT_ID + 1}

    async def test_list_by_tenant_is_ordered_by_join_time(
        self, service: TenantMemberService, rows: dict[int, TenantMember]
    ) -> None:
        later = _seed(rows, user_id="usr-2", joined_at=_NOW + timedelta(days=1))
        earlier = _seed(rows, user_id="usr-1", joined_at=_NOW)

        memberships = await service.list_by_tenant(_TENANT_ID)

        assert [m.id for m in memberships] == [earlier.id, later.id]

    async def test_has_any_members_reflects_membership(
        self, service: TenantMemberService, rows: dict[int, TenantMember]
    ) -> None:
        assert await service.has_any_members(_TENANT_ID) is False

        _seed(rows, user_id="usr-1")

        assert await service.has_any_members(_TENANT_ID) is True

    async def test_list_members_page_reports_total_and_page(
        self, service: TenantMemberService, rows: dict[int, TenantMember]
    ) -> None:
        for index in range(5):
            _seed(rows, user_id=f"usr-{index}", joined_at=_NOW + timedelta(hours=index))

        members, total = await service.list_members_page(_TENANT_ID, page=2, page_size=2)

        assert total == 5
        assert [m.user_id for m in members] == ["usr-2", "usr-3"]

    async def test_list_members_page_filters_by_query(
        self,
        service: TenantMemberService,
        rows: dict[int, TenantMember],
        user_index: dict[str, str],
    ) -> None:
        _seed(rows, user_id="usr-1")
        _seed(rows, user_id="usr-2")
        user_index["usr-1"] = "alice@example.com alice"
        user_index["usr-2"] = "bob@example.com bob"

        members, total = await service.list_members_page(_TENANT_ID, query="alice")

        assert total == 1
        assert [m.user_id for m in members] == ["usr-1"]

    @pytest.mark.parametrize(
        ("page", "page_size", "expected_count"),
        [(0, 2, 2), (-5, 2, 2), (1, 0, 3), (1, 5000, 3)],
    )
    async def test_list_members_page_clamps_paging_inputs(
        self,
        service: TenantMemberService,
        rows: dict[int, TenantMember],
        page: int,
        page_size: int,
        expected_count: int,
    ) -> None:
        for index in range(3):
            _seed(rows, user_id=f"usr-{index}", joined_at=_NOW + timedelta(hours=index))

        members, _ = await service.list_members_page(
            _TENANT_ID, page=page, page_size=page_size
        )

        assert len(members) == expected_count


# ── update_role ─────────────────────────────────────────────────────


class TestUpdateRole(ServiceTest):
    async def test_changes_the_role(
        self, service: TenantMemberService, rows: dict[int, TenantMember]
    ) -> None:
        _seed(rows, user_id="usr-1", role=ROLE_VIEWER)

        member = await service.update_role(
            user_id="usr-1",
            tenant_id=_TENANT_ID,
            role=ROLE_ADMIN,
        )

        assert member.role == ROLE_ADMIN

    async def test_to_the_same_role_is_a_noop(
        self, service: TenantMemberService, rows: dict[int, TenantMember]
    ) -> None:
        seeded = _seed(rows, user_id="usr-1", role=ROLE_ADMIN)

        member = await service.update_role(
            user_id="usr-1",
            tenant_id=_TENANT_ID,
            role=ROLE_ADMIN,
        )

        assert member.updated_at == seeded.updated_at

    async def test_rejects_unknown_role(
        self, service: TenantMemberService, rows: dict[int, TenantMember]
    ) -> None:
        _seed(rows, user_id="usr-1")

        with pytest.raises(ValidationError):
            await service.update_role(user_id="usr-1", tenant_id=_TENANT_ID, role="root")

    async def test_missing_membership_raises(self, service: TenantMemberService) -> None:
        with pytest.raises(NotFoundError) as excinfo:
            await service.update_role(
                user_id="ghost", tenant_id=_TENANT_ID, role=ROLE_ADMIN
            )
        assert excinfo.value.code == "tenant_member.not_found"

    async def test_last_owner_cannot_be_demoted(
        self, service: TenantMemberService, rows: dict[int, TenantMember]
    ) -> None:
        _seed(rows, user_id="usr-owner", role=ROLE_OWNER)
        _seed(rows, user_id="usr-other", role=ROLE_ADMIN)

        with pytest.raises(ConflictError) as excinfo:
            await service.update_role(
                user_id="usr-owner",
                tenant_id=_TENANT_ID,
                role=ROLE_ADMIN,
            )

        assert excinfo.value.code == "tenant_member.last_owner"

    async def test_owner_can_be_demoted_when_another_owner_remains(
        self, service: TenantMemberService, rows: dict[int, TenantMember]
    ) -> None:
        _seed(rows, user_id="usr-owner", role=ROLE_OWNER)
        _seed(rows, user_id="usr-owner-2", role=ROLE_OWNER)

        member = await service.update_role(
            user_id="usr-owner",
            tenant_id=_TENANT_ID,
            role=ROLE_ADMIN,
        )

        assert member.role == ROLE_ADMIN

    async def test_promoting_to_owner_needs_no_other_owner(
        self, service: TenantMemberService, rows: dict[int, TenantMember]
    ) -> None:
        _seed(rows, user_id="usr-1", role=ROLE_ADMIN)

        member = await service.update_role(
            user_id="usr-1",
            tenant_id=_TENANT_ID,
            role=ROLE_OWNER,
        )

        assert member.role == ROLE_OWNER


# ── remove_member ───────────────────────────────────────────────────


class TestRemoveMember(ServiceTest):
    async def test_soft_deletes_the_row(
        self, service: TenantMemberService, rows: dict[int, TenantMember]
    ) -> None:
        seeded = _seed(rows, user_id="usr-1")

        await service.remove_member(user_id="usr-1", tenant_id=_TENANT_ID)

        assert rows[seeded.id].deleted_at is not None
        assert (
            await service.get_membership(user_id="usr-1", tenant_id=_TENANT_ID) is None
        )

    async def test_missing_membership_raises(self, service: TenantMemberService) -> None:
        with pytest.raises(NotFoundError):
            await service.remove_member(user_id="ghost", tenant_id=_TENANT_ID)

    async def test_last_owner_cannot_be_removed(
        self, service: TenantMemberService, rows: dict[int, TenantMember]
    ) -> None:
        _seed(rows, user_id="usr-owner", role=ROLE_OWNER)
        _seed(rows, user_id="usr-other", role=ROLE_VIEWER)

        with pytest.raises(ConflictError) as excinfo:
            await service.remove_member(user_id="usr-owner", tenant_id=_TENANT_ID)

        assert excinfo.value.code == "tenant_member.last_owner"

    async def test_owner_can_be_removed_when_another_owner_remains(
        self, service: TenantMemberService, rows: dict[int, TenantMember]
    ) -> None:
        _seed(rows, user_id="usr-owner", role=ROLE_OWNER)
        _seed(rows, user_id="usr-owner-2", role=ROLE_OWNER)

        await service.remove_member(user_id="usr-owner", tenant_id=_TENANT_ID)

        assert (
            await service.get_membership(user_id="usr-owner", tenant_id=_TENANT_ID) is None
        )