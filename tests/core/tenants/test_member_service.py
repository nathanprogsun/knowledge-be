"""Unit tests for `TenantMemberService` and the role hierarchy helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

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
from src.db.models.tenants.tenant_members import TenantMember
from tests.fakes.tenant_members import FakeTenantMemberRepository

_TENANT_ID = 7
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def repo() -> FakeTenantMemberRepository:
    return FakeTenantMemberRepository()


@pytest.fixture
def service(repo: FakeTenantMemberRepository) -> TenantMemberService:
    return TenantMemberService(members_repo=repo)  # type: ignore[arg-type]


async def _seed(
    repo: FakeTenantMemberRepository,
    *,
    user_id: str,
    role: str = ROLE_CONTRIBUTOR,
    tenant_id: int = _TENANT_ID,
    joined_at: datetime = _NOW,
    search_text: str = "",
) -> TenantMember:
    if search_text:
        repo.user_search_index[user_id] = search_text
    stored = await repo.insert_live_or_none(
        TenantMember(
            user_id=user_id,
            tenant_id=tenant_id,
            role=role,
            joined_at=joined_at,
            created_at=joined_at,
            updated_at=joined_at,
        )
    )
    assert stored is not None
    return stored


# ── role hierarchy ──────────────────────────────────────────────────


def test_known_roles_are_valid() -> None:
    for role in (ROLE_OWNER, ROLE_ADMIN, ROLE_CONTRIBUTOR, ROLE_VIEWER):
        assert is_valid_role(role) is True


def test_unknown_role_is_invalid_and_ranks_lowest() -> None:
    assert is_valid_role("superuser") is False
    assert role_level("superuser") == 0


def test_role_hierarchy_is_ordered() -> None:
    assert role_level(ROLE_OWNER) > role_level(ROLE_ADMIN)
    assert role_level(ROLE_ADMIN) > role_level(ROLE_CONTRIBUTOR)
    assert role_level(ROLE_CONTRIBUTOR) > role_level(ROLE_VIEWER)


def test_has_permission_compares_levels() -> None:
    assert has_permission(ROLE_ADMIN, ROLE_CONTRIBUTOR) is True
    assert has_permission(ROLE_ADMIN, ROLE_ADMIN) is True
    assert has_permission(ROLE_VIEWER, ROLE_ADMIN) is False
    assert has_permission("nonsense", ROLE_VIEWER) is False


# ── add_member ──────────────────────────────────────────────────────


async def test_add_member_creates_active_membership(service: TenantMemberService) -> None:
    member = await service.add_member(
        user_id="usr-1",
        tenant_id=_TENANT_ID,
        role=ROLE_CONTRIBUTOR,
    )

    assert member.user_id == "usr-1"
    assert member.status == "active"
    assert member.role == ROLE_CONTRIBUTOR


async def test_add_member_records_the_inviter(service: TenantMemberService) -> None:
    member = await service.add_member(
        user_id="usr-1",
        tenant_id=_TENANT_ID,
        role=ROLE_VIEWER,
        invited_by="usr-admin",
    )

    assert member.invited_by == "usr-admin"


async def test_add_member_rejects_unknown_role(service: TenantMemberService) -> None:
    with pytest.raises(ValidationError) as excinfo:
        await service.add_member(user_id="usr-1", tenant_id=_TENANT_ID, role="superuser")

    assert excinfo.value.code == "tenant_member.invalid_role"


async def test_add_member_rejects_duplicate(
    service: TenantMemberService,
    repo: FakeTenantMemberRepository,
) -> None:
    await _seed(repo, user_id="usr-1")

    with pytest.raises(ConflictError) as excinfo:
        await service.add_member(user_id="usr-1", tenant_id=_TENANT_ID, role=ROLE_VIEWER)

    assert excinfo.value.code == "tenant_member.exists"


async def test_removed_member_can_be_added_again(
    service: TenantMemberService,
    repo: FakeTenantMemberRepository,
) -> None:
    await _seed(repo, user_id="usr-1")
    await service.remove_member(user_id="usr-1", tenant_id=_TENANT_ID)

    member = await service.add_member(
        user_id="usr-1",
        tenant_id=_TENANT_ID,
        role=ROLE_ADMIN,
    )

    assert member.role == ROLE_ADMIN


async def test_membership_is_scoped_to_one_workspace(service: TenantMemberService) -> None:
    await service.add_member(user_id="usr-1", tenant_id=_TENANT_ID, role=ROLE_VIEWER)

    other = await service.add_member(
        user_id="usr-1",
        tenant_id=_TENANT_ID + 1,
        role=ROLE_OWNER,
    )

    assert other.tenant_id == _TENANT_ID + 1


# ── ensure_owner ────────────────────────────────────────────────────


async def test_ensure_owner_creates_the_owner_row(service: TenantMemberService) -> None:
    member = await service.ensure_owner(user_id="usr-1", tenant_id=_TENANT_ID)

    assert member.role == ROLE_OWNER


async def test_ensure_owner_is_idempotent(
    service: TenantMemberService,
    repo: FakeTenantMemberRepository,
) -> None:
    existing = await _seed(repo, user_id="usr-1", role=ROLE_VIEWER)

    member = await service.ensure_owner(user_id="usr-1", tenant_id=_TENANT_ID)

    assert member.id == existing.id
    assert member.role == ROLE_VIEWER


# ── reads ───────────────────────────────────────────────────────────


async def test_get_membership_returns_none_when_absent(service: TenantMemberService) -> None:
    assert await service.get_membership(user_id="ghost", tenant_id=_TENANT_ID) is None


async def test_list_by_user_returns_every_workspace(
    service: TenantMemberService,
    repo: FakeTenantMemberRepository,
) -> None:
    await _seed(repo, user_id="usr-1", tenant_id=_TENANT_ID)
    await _seed(repo, user_id="usr-1", tenant_id=_TENANT_ID + 1)

    memberships = await service.list_by_user("usr-1")

    assert {m.tenant_id for m in memberships} == {_TENANT_ID, _TENANT_ID + 1}


async def test_list_by_tenant_is_ordered_by_join_time(
    service: TenantMemberService,
    repo: FakeTenantMemberRepository,
) -> None:
    later = await _seed(repo, user_id="usr-2", joined_at=_NOW + timedelta(days=1))
    earlier = await _seed(repo, user_id="usr-1", joined_at=_NOW)

    memberships = await service.list_by_tenant(_TENANT_ID)

    assert [m.id for m in memberships] == [earlier.id, later.id]


async def test_has_any_members_reflects_membership(
    service: TenantMemberService,
    repo: FakeTenantMemberRepository,
) -> None:
    assert await service.has_any_members(_TENANT_ID) is False

    await _seed(repo, user_id="usr-1")

    assert await service.has_any_members(_TENANT_ID) is True


async def test_list_members_page_reports_total_and_page(
    service: TenantMemberService,
    repo: FakeTenantMemberRepository,
) -> None:
    for index in range(5):
        await _seed(repo, user_id=f"usr-{index}", joined_at=_NOW + timedelta(hours=index))

    members, total = await service.list_members_page(_TENANT_ID, page=2, page_size=2)

    assert total == 5
    assert [m.user_id for m in members] == ["usr-2", "usr-3"]


async def test_list_members_page_filters_by_query(
    service: TenantMemberService,
    repo: FakeTenantMemberRepository,
) -> None:
    await _seed(repo, user_id="usr-1", search_text="alice@example.com alice")
    await _seed(repo, user_id="usr-2", search_text="bob@example.com bob")

    members, total = await service.list_members_page(_TENANT_ID, query="alice")

    assert total == 1
    assert [m.user_id for m in members] == ["usr-1"]


@pytest.mark.parametrize(
    ("page", "page_size", "expected_count"),
    [(0, 2, 2), (-5, 2, 2), (1, 0, 3), (1, 5000, 3)],
)
async def test_list_members_page_clamps_paging_inputs(
    service: TenantMemberService,
    repo: FakeTenantMemberRepository,
    page: int,
    page_size: int,
    expected_count: int,
) -> None:
    for index in range(3):
        await _seed(repo, user_id=f"usr-{index}", joined_at=_NOW + timedelta(hours=index))

    members, _ = await service.list_members_page(_TENANT_ID, page=page, page_size=page_size)

    assert len(members) == expected_count


# ── update_role ─────────────────────────────────────────────────────


async def test_update_role_changes_the_role(
    service: TenantMemberService,
    repo: FakeTenantMemberRepository,
) -> None:
    await _seed(repo, user_id="usr-1", role=ROLE_VIEWER)

    member = await service.update_role(
        user_id="usr-1",
        tenant_id=_TENANT_ID,
        role=ROLE_ADMIN,
    )

    assert member.role == ROLE_ADMIN


async def test_update_role_to_the_same_role_is_a_noop(
    service: TenantMemberService,
    repo: FakeTenantMemberRepository,
) -> None:
    seeded = await _seed(repo, user_id="usr-1", role=ROLE_ADMIN)

    member = await service.update_role(
        user_id="usr-1",
        tenant_id=_TENANT_ID,
        role=ROLE_ADMIN,
    )

    assert member.updated_at == seeded.updated_at


async def test_update_role_rejects_unknown_role(
    service: TenantMemberService,
    repo: FakeTenantMemberRepository,
) -> None:
    await _seed(repo, user_id="usr-1")

    with pytest.raises(ValidationError):
        await service.update_role(user_id="usr-1", tenant_id=_TENANT_ID, role="root")


async def test_update_role_missing_membership_raises(service: TenantMemberService) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        await service.update_role(user_id="ghost", tenant_id=_TENANT_ID, role=ROLE_ADMIN)

    assert excinfo.value.code == "tenant_member.not_found"


async def test_last_owner_cannot_be_demoted(
    service: TenantMemberService,
    repo: FakeTenantMemberRepository,
) -> None:
    await _seed(repo, user_id="usr-owner", role=ROLE_OWNER)
    await _seed(repo, user_id="usr-other", role=ROLE_ADMIN)

    with pytest.raises(ConflictError) as excinfo:
        await service.update_role(
            user_id="usr-owner",
            tenant_id=_TENANT_ID,
            role=ROLE_ADMIN,
        )

    assert excinfo.value.code == "tenant_member.last_owner"


async def test_owner_can_be_demoted_when_another_owner_remains(
    service: TenantMemberService,
    repo: FakeTenantMemberRepository,
) -> None:
    await _seed(repo, user_id="usr-owner", role=ROLE_OWNER)
    await _seed(repo, user_id="usr-owner-2", role=ROLE_OWNER)

    member = await service.update_role(
        user_id="usr-owner",
        tenant_id=_TENANT_ID,
        role=ROLE_ADMIN,
    )

    assert member.role == ROLE_ADMIN


async def test_promoting_to_owner_needs_no_other_owner(
    service: TenantMemberService,
    repo: FakeTenantMemberRepository,
) -> None:
    await _seed(repo, user_id="usr-1", role=ROLE_ADMIN)

    member = await service.update_role(
        user_id="usr-1",
        tenant_id=_TENANT_ID,
        role=ROLE_OWNER,
    )

    assert member.role == ROLE_OWNER


# ── remove_member ───────────────────────────────────────────────────


async def test_remove_member_soft_deletes_the_row(
    service: TenantMemberService,
    repo: FakeTenantMemberRepository,
) -> None:
    seeded = await _seed(repo, user_id="usr-1")

    await service.remove_member(user_id="usr-1", tenant_id=_TENANT_ID)

    assert repo.rows[seeded.id].deleted_at is not None
    assert await service.get_membership(user_id="usr-1", tenant_id=_TENANT_ID) is None


async def test_remove_member_missing_membership_raises(service: TenantMemberService) -> None:
    with pytest.raises(NotFoundError):
        await service.remove_member(user_id="ghost", tenant_id=_TENANT_ID)


async def test_last_owner_cannot_be_removed(
    service: TenantMemberService,
    repo: FakeTenantMemberRepository,
) -> None:
    await _seed(repo, user_id="usr-owner", role=ROLE_OWNER)
    await _seed(repo, user_id="usr-other", role=ROLE_VIEWER)

    with pytest.raises(ConflictError) as excinfo:
        await service.remove_member(user_id="usr-owner", tenant_id=_TENANT_ID)

    assert excinfo.value.code == "tenant_member.last_owner"


async def test_owner_can_be_removed_when_another_owner_remains(
    service: TenantMemberService,
    repo: FakeTenantMemberRepository,
) -> None:
    await _seed(repo, user_id="usr-owner", role=ROLE_OWNER)
    await _seed(repo, user_id="usr-owner-2", role=ROLE_OWNER)

    await service.remove_member(user_id="usr-owner", tenant_id=_TENANT_ID)

    assert await service.get_membership(user_id="usr-owner", tenant_id=_TENANT_ID) is None
