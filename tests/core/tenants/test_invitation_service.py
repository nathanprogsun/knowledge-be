"""Unit tests for `TenantInvitationService`.

The service is wired to the shared in-memory invitation and membership
fakes, so the cross-service hop (accept -> membership) is exercised for
real rather than mocked out.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.common.exception import (
    ConflictError,
    NotFoundError,
    PermissionDeniedError,
    ValidationError,
)
from src.core.tenants.invitation_service import (
    TenantInvitationService,
    generate_share_link_token,
)
from src.core.tenants.member_service import ROLE_OWNER, ROLE_VIEWER, TenantMemberService
from src.db.models.tenants.tenant_invitations import (
    STATUS_ACCEPTED,
    STATUS_DECLINED,
    STATUS_EXPIRED,
    STATUS_PENDING,
    STATUS_REVOKED,
)
from tests.fakes.tenant_invitations import FakeTenantInvitationRepository
from tests.fakes.tenant_members import FakeTenantMemberRepository

_TENANT_ID = 7
_INVITEE = "usr-invitee"
_INVITER = "usr-owner"


@pytest.fixture
def invitations_repo() -> FakeTenantInvitationRepository:
    return FakeTenantInvitationRepository()


@pytest.fixture
def members_repo() -> FakeTenantMemberRepository:
    return FakeTenantMemberRepository()


@pytest.fixture
def member_service(members_repo: FakeTenantMemberRepository) -> TenantMemberService:
    return TenantMemberService(members_repo=members_repo)  # type: ignore[arg-type]


@pytest.fixture
def service(
    invitations_repo: FakeTenantInvitationRepository,
    member_service: TenantMemberService,
) -> TenantInvitationService:
    return TenantInvitationService(
        invitations_repo=invitations_repo,  # type: ignore[arg-type]
        member_service=member_service,
    )


def _expire(repo: FakeTenantInvitationRepository, invitation_id: int) -> None:
    """Backdate a row's expiry so the next sweep flips it."""
    row = repo.rows[invitation_id]
    repo.rows[invitation_id] = row.model_copy(
        update={"expires_at": datetime.now(UTC) - timedelta(minutes=1)}
    )


# ── token helper ────────────────────────────────────────────────────


def test_share_link_token_is_unpadded_and_unique() -> None:
    token = generate_share_link_token()

    assert "=" not in token
    assert token != generate_share_link_token()


# ── create_invitation ───────────────────────────────────────────────


async def test_create_invitation_is_pending_with_ttl(service: TenantInvitationService) -> None:
    invitation = await service.create_invitation(
        tenant_id=_TENANT_ID,
        invitee_user_id=_INVITEE,
        role=ROLE_VIEWER,
        invited_by=_INVITER,
    )

    assert invitation.status == STATUS_PENDING
    assert invitation.role == ROLE_VIEWER
    assert invitation.invited_by == _INVITER
    assert invitation.expires_at > datetime.now(UTC)
    assert invitation.is_share_link is False


async def test_create_invitation_rejects_unknown_role(service: TenantInvitationService) -> None:
    with pytest.raises(ValidationError):
        await service.create_invitation(
            tenant_id=_TENANT_ID,
            invitee_user_id=_INVITEE,
            role="superuser",
        )


async def test_create_invitation_rejects_existing_member(
    service: TenantInvitationService,
    member_service: TenantMemberService,
) -> None:
    await member_service.add_member(
        user_id=_INVITEE,
        tenant_id=_TENANT_ID,
        role=ROLE_VIEWER,
    )

    with pytest.raises(ConflictError) as excinfo:
        await service.create_invitation(
            tenant_id=_TENANT_ID,
            invitee_user_id=_INVITEE,
            role=ROLE_VIEWER,
        )

    assert excinfo.value.code == "tenant_invitation.already_member"


async def test_create_invitation_rejects_duplicate_pending(
    service: TenantInvitationService,
) -> None:
    await service.create_invitation(
        tenant_id=_TENANT_ID,
        invitee_user_id=_INVITEE,
        role=ROLE_VIEWER,
    )

    with pytest.raises(ConflictError) as excinfo:
        await service.create_invitation(
            tenant_id=_TENANT_ID,
            invitee_user_id=_INVITEE,
            role=ROLE_VIEWER,
        )

    assert excinfo.value.code == "tenant_invitation.pending_exists"


async def test_new_invitation_allowed_after_decline(service: TenantInvitationService) -> None:
    first = await service.create_invitation(
        tenant_id=_TENANT_ID,
        invitee_user_id=_INVITEE,
        role=ROLE_VIEWER,
    )
    await service.decline(first.id, user_id=_INVITEE)

    second = await service.create_invitation(
        tenant_id=_TENANT_ID,
        invitee_user_id=_INVITEE,
        role=ROLE_VIEWER,
    )

    assert second.id != first.id
    assert second.status == STATUS_PENDING


# ── accept ──────────────────────────────────────────────────────────


async def test_accept_creates_the_membership_and_finalises_the_row(
    service: TenantInvitationService,
    invitations_repo: FakeTenantInvitationRepository,
) -> None:
    invitation = await service.create_invitation(
        tenant_id=_TENANT_ID,
        invitee_user_id=_INVITEE,
        role=ROLE_VIEWER,
        invited_by=_INVITER,
    )

    member = await service.accept(invitation.id, user_id=_INVITEE)

    assert member.user_id == _INVITEE
    assert member.role == ROLE_VIEWER
    assert member.invited_by == _INVITER
    assert invitations_repo.rows[invitation.id].status == STATUS_ACCEPTED


async def test_accept_is_idempotent_for_an_existing_member(
    service: TenantInvitationService,
    member_service: TenantMemberService,
    invitations_repo: FakeTenantInvitationRepository,
) -> None:
    invitation = await service.create_invitation(
        tenant_id=_TENANT_ID,
        invitee_user_id=_INVITEE,
        role=ROLE_VIEWER,
    )
    # The user joins another way between issuing and accepting.
    await member_service.add_member(
        user_id=_INVITEE,
        tenant_id=_TENANT_ID,
        role=ROLE_OWNER,
    )

    member = await service.accept(invitation.id, user_id=_INVITEE)

    assert member.role == ROLE_OWNER
    assert invitations_repo.rows[invitation.id].status == STATUS_ACCEPTED


async def test_accept_rejects_another_user(service: TenantInvitationService) -> None:
    invitation = await service.create_invitation(
        tenant_id=_TENANT_ID,
        invitee_user_id=_INVITEE,
        role=ROLE_VIEWER,
    )

    with pytest.raises(PermissionDeniedError) as excinfo:
        await service.accept(invitation.id, user_id="usr-someone-else")

    assert excinfo.value.code == "tenant_invitation.forbidden"


async def test_accept_twice_reports_not_pending(service: TenantInvitationService) -> None:
    invitation = await service.create_invitation(
        tenant_id=_TENANT_ID,
        invitee_user_id=_INVITEE,
        role=ROLE_VIEWER,
    )
    await service.accept(invitation.id, user_id=_INVITEE)

    with pytest.raises(ConflictError) as excinfo:
        await service.accept(invitation.id, user_id=_INVITEE)

    assert excinfo.value.code == "tenant_invitation.not_pending"


async def test_accept_unknown_invitation_raises(service: TenantInvitationService) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        await service.accept(4242, user_id=_INVITEE)

    assert excinfo.value.code == "tenant_invitation.not_found"


async def test_accept_expired_invitation_is_refused(
    service: TenantInvitationService,
    invitations_repo: FakeTenantInvitationRepository,
) -> None:
    invitation = await service.create_invitation(
        tenant_id=_TENANT_ID,
        invitee_user_id=_INVITEE,
        role=ROLE_VIEWER,
    )
    _expire(invitations_repo, invitation.id)

    with pytest.raises(ConflictError) as excinfo:
        await service.accept(invitation.id, user_id=_INVITEE)

    assert excinfo.value.code == "tenant_invitation.not_pending"
    assert invitations_repo.rows[invitation.id].status == STATUS_EXPIRED


# ── decline / revoke ────────────────────────────────────────────────


async def test_decline_finalises_without_membership(
    service: TenantInvitationService,
    member_service: TenantMemberService,
    invitations_repo: FakeTenantInvitationRepository,
) -> None:
    invitation = await service.create_invitation(
        tenant_id=_TENANT_ID,
        invitee_user_id=_INVITEE,
        role=ROLE_VIEWER,
    )

    await service.decline(invitation.id, user_id=_INVITEE)

    assert invitations_repo.rows[invitation.id].status == STATUS_DECLINED
    assert await member_service.get_membership(user_id=_INVITEE, tenant_id=_TENANT_ID) is None


async def test_decline_rejects_another_user(service: TenantInvitationService) -> None:
    invitation = await service.create_invitation(
        tenant_id=_TENANT_ID,
        invitee_user_id=_INVITEE,
        role=ROLE_VIEWER,
    )

    with pytest.raises(PermissionDeniedError):
        await service.decline(invitation.id, user_id="usr-other")


async def test_revoke_finalises_a_pending_invitation(
    service: TenantInvitationService,
    invitations_repo: FakeTenantInvitationRepository,
) -> None:
    invitation = await service.create_invitation(
        tenant_id=_TENANT_ID,
        invitee_user_id=_INVITEE,
        role=ROLE_VIEWER,
    )

    await service.revoke(invitation.id)

    assert invitations_repo.rows[invitation.id].status == STATUS_REVOKED


async def test_revoke_after_accept_is_refused(service: TenantInvitationService) -> None:
    invitation = await service.create_invitation(
        tenant_id=_TENANT_ID,
        invitee_user_id=_INVITEE,
        role=ROLE_VIEWER,
    )
    await service.accept(invitation.id, user_id=_INVITEE)

    with pytest.raises(ConflictError):
        await service.revoke(invitation.id)


async def test_accept_after_revoke_is_refused(service: TenantInvitationService) -> None:
    invitation = await service.create_invitation(
        tenant_id=_TENANT_ID,
        invitee_user_id=_INVITEE,
        role=ROLE_VIEWER,
    )
    await service.revoke(invitation.id)

    with pytest.raises(ConflictError):
        await service.accept(invitation.id, user_id=_INVITEE)


# ── share links ─────────────────────────────────────────────────────


async def test_create_share_link_has_no_invitee(service: TenantInvitationService) -> None:
    invitation, token = await service.create_share_link(
        tenant_id=_TENANT_ID,
        role=ROLE_VIEWER,
        invited_by=_INVITER,
    )

    assert invitation.invitee_user_id == ""
    assert invitation.is_share_link is True
    assert token


async def test_share_link_token_is_not_exposed_on_the_dto(
    service: TenantInvitationService,
) -> None:
    invitation, _ = await service.create_share_link(tenant_id=_TENANT_ID, role=ROLE_VIEWER)

    assert "token" not in invitation.model_dump()


async def test_multiple_share_links_can_coexist(service: TenantInvitationService) -> None:
    first, _ = await service.create_share_link(tenant_id=_TENANT_ID, role=ROLE_VIEWER)
    second, _ = await service.create_share_link(tenant_id=_TENANT_ID, role=ROLE_VIEWER)

    assert first.id != second.id


async def test_lookup_by_token_resolves_the_row(service: TenantInvitationService) -> None:
    created, token = await service.create_share_link(tenant_id=_TENANT_ID, role=ROLE_VIEWER)

    found = await service.lookup_by_token(f"  {token}  ")

    assert found.id == created.id


@pytest.mark.parametrize("token", ["", "   ", "sk-never-issued"])
async def test_lookup_by_token_rejects_unknown_tokens(
    service: TenantInvitationService,
    token: str,
) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        await service.lookup_by_token(token)

    assert excinfo.value.code == "tenant_invitation.invalid_token"


async def test_lookup_by_token_rejects_expired_link(
    service: TenantInvitationService,
    invitations_repo: FakeTenantInvitationRepository,
) -> None:
    created, token = await service.create_share_link(tenant_id=_TENANT_ID, role=ROLE_VIEWER)
    _expire(invitations_repo, created.id)

    with pytest.raises(NotFoundError):
        await service.lookup_by_token(token)


async def test_accept_by_token_joins_and_keeps_the_link_pending(
    service: TenantInvitationService,
    invitations_repo: FakeTenantInvitationRepository,
) -> None:
    created, token = await service.create_share_link(tenant_id=_TENANT_ID, role=ROLE_VIEWER)

    member = await service.accept_by_token(token, user_id="usr-new")

    assert member.tenant_id == _TENANT_ID
    assert member.role == ROLE_VIEWER
    stored = invitations_repo.rows[created.id]
    assert stored.status == STATUS_PENDING
    assert stored.accepted_count == 1


async def test_share_link_is_multi_use(
    service: TenantInvitationService,
    invitations_repo: FakeTenantInvitationRepository,
) -> None:
    created, token = await service.create_share_link(tenant_id=_TENANT_ID, role=ROLE_VIEWER)

    await service.accept_by_token(token, user_id="usr-a")
    await service.accept_by_token(token, user_id="usr-b")

    assert invitations_repo.rows[created.id].accepted_count == 2


async def test_accept_by_token_twice_keeps_the_existing_role(
    service: TenantInvitationService,
    member_service: TenantMemberService,
) -> None:
    _, token = await service.create_share_link(tenant_id=_TENANT_ID, role=ROLE_VIEWER)
    await service.accept_by_token(token, user_id="usr-a")
    await member_service.update_role(
        user_id="usr-a",
        tenant_id=_TENANT_ID,
        role=ROLE_OWNER,
    )

    member = await service.accept_by_token(token, user_id="usr-a")

    assert member.role == ROLE_OWNER


# ── listing / sweeping ──────────────────────────────────────────────


async def test_list_by_tenant_hides_terminal_rows_by_default(
    service: TenantInvitationService,
) -> None:
    pending = await service.create_invitation(
        tenant_id=_TENANT_ID,
        invitee_user_id="usr-a",
        role=ROLE_VIEWER,
    )
    declined = await service.create_invitation(
        tenant_id=_TENANT_ID,
        invitee_user_id="usr-b",
        role=ROLE_VIEWER,
    )
    await service.decline(declined.id, user_id="usr-b")

    visible = await service.list_by_tenant(_TENANT_ID)
    everything = await service.list_by_tenant(_TENANT_ID, include_terminal=True)

    assert [i.id for i in visible] == [pending.id]
    assert {i.id for i in everything} == {pending.id, declined.id}


async def test_list_page_reports_total_and_clamps_paging(
    service: TenantInvitationService,
) -> None:
    for index in range(5):
        await service.create_invitation(
            tenant_id=_TENANT_ID,
            invitee_user_id=f"usr-{index}",
            role=ROLE_VIEWER,
        )

    page, total = await service.list_tenant_invitations_page(
        _TENANT_ID,
        page=0,
        page_size=5000,
    )

    assert total == 5
    assert len(page) == 5


async def test_list_by_invitee_is_scoped_to_that_user(
    service: TenantInvitationService,
) -> None:
    mine = await service.create_invitation(
        tenant_id=_TENANT_ID,
        invitee_user_id=_INVITEE,
        role=ROLE_VIEWER,
    )
    await service.create_invitation(
        tenant_id=_TENANT_ID,
        invitee_user_id="usr-other",
        role=ROLE_VIEWER,
    )

    inbox = await service.list_by_invitee(_INVITEE)

    assert [i.id for i in inbox] == [mine.id]


async def test_count_pending_by_invitee_excludes_swept_rows(
    service: TenantInvitationService,
    invitations_repo: FakeTenantInvitationRepository,
) -> None:
    stale = await service.create_invitation(
        tenant_id=_TENANT_ID,
        invitee_user_id=_INVITEE,
        role=ROLE_VIEWER,
    )
    await service.create_invitation(
        tenant_id=_TENANT_ID + 1,
        invitee_user_id=_INVITEE,
        role=ROLE_VIEWER,
    )
    _expire(invitations_repo, stale.id)

    assert await service.count_pending_by_invitee(_INVITEE) == 1


async def test_expire_overdue_reports_swept_rows(
    service: TenantInvitationService,
    invitations_repo: FakeTenantInvitationRepository,
) -> None:
    invitation = await service.create_invitation(
        tenant_id=_TENANT_ID,
        invitee_user_id=_INVITEE,
        role=ROLE_VIEWER,
    )
    _expire(invitations_repo, invitation.id)

    assert await service.expire_overdue() == 1
    assert await service.expire_overdue() == 0


async def test_get_invitation_returns_none_when_missing(
    service: TenantInvitationService,
) -> None:
    assert await service.get_invitation(4242) is None
