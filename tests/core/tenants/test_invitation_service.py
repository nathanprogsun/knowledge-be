"""Unit tests for `TenantInvitationService`.

The invitation and membership repositories are both replaced with
``AsyncMock(spec=...)`` mocks backed by closure-captured state, so the
cross-service hop (accept -> membership) is exercised for real rather
than mocked out.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

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
from src.db.dao.tenant_invitations_repository import TenantInvitationRepository
from src.db.dao.tenant_members_repository import TenantMemberRepository
from src.db.models.tenants.tenant_invitations import (
    STATUS_ACCEPTED,
    STATUS_DECLINED,
    STATUS_EXPIRED,
    STATUS_PENDING,
    TenantInvitation,
    is_expired,
)
from tests.util.service_test import ServiceTest

_TENANT_ID = 7
_INVITEE = "usr-invitee"
_INVITER = "usr-owner"


# ── Invitation repository mock ──────────────────────────────────────


def _make_invitations_repo() -> tuple[AsyncMock, dict[int, TenantInvitation]]:
    """Invitation-repo mock with closure-captured state."""
    repo = AsyncMock(spec=TenantInvitationRepository)
    rows: dict[int, TenantInvitation] = {}
    _next_id = {"value": 0}

    def _live() -> list[TenantInvitation]:
        return [r for r in rows.values() if r.deleted_at is None]

    @staticmethod
    def _filtered(rs: list[TenantInvitation], *, include_terminal: bool) -> list[TenantInvitation]:
        selected = rs if include_terminal else [r for r in rs if r.status == STATUS_PENDING]
        return sorted(selected, key=lambda r: (r.created_at, r.id), reverse=True)

    async def _insert_pending_or_none(row: TenantInvitation) -> TenantInvitation | None:
        if row.invitee_user_id:
            for r in _live():
                if (
                    r.tenant_id == row.tenant_id
                    and r.invitee_user_id == row.invitee_user_id
                    and r.status == STATUS_PENDING
                ):
                    return None
        _next_id["value"] += 1
        stored = row.model_copy(update={"id": _next_id["value"]})
        rows[stored.id] = stored
        return stored

    async def _mark_status_if_pending(
        invitation_id: int,
        *,
        status: str,
        responded_at: datetime,
    ) -> int:
        row = rows.get(invitation_id)
        if row is None or row.status != STATUS_PENDING or row.deleted_at is not None:
            return 0
        rows[invitation_id] = row.model_copy(
            update={
                "status": status,
                "responded_at": responded_at,
                "updated_at": responded_at,
            }
        )
        return 1

    async def _sweep_expired(now: datetime) -> int:
        swept = 0
        for key, row in list(rows.items()):
            if row.status == STATUS_PENDING and row.deleted_at is None and is_expired(row, now):
                rows[key] = row.model_copy(
                    update={
                        "status": STATUS_EXPIRED,
                        "responded_at": now,
                        "updated_at": now,
                    }
                )
                swept += 1
        return swept

    async def _increment_accepted_count(invitation_id: int) -> int:
        row = rows.get(invitation_id)
        if row is None or row.deleted_at is not None:
            return 0
        rows[invitation_id] = row.model_copy(update={"accepted_count": row.accepted_count + 1})
        return 1

    async def _find_by_id_or_none(invitation_id: int) -> TenantInvitation | None:
        row = rows.get(invitation_id)
        return row if row is not None and row.deleted_at is None else None

    async def _find_pending_by_pair(
        *, tenant_id: int, invitee_user_id: str
    ) -> TenantInvitation | None:
        for r in _live():
            if (
                r.tenant_id == tenant_id
                and r.invitee_user_id == invitee_user_id
                and r.status == STATUS_PENDING
            ):
                return r
        return None

    async def _find_pending_by_token(token: str) -> TenantInvitation | None:
        if not token:
            return None
        for r in _live():
            if r.token == token and r.status == STATUS_PENDING:
                return r
        return None

    async def _list_by_tenant(
        tenant_id: int,
        *,
        include_terminal: bool = False,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[TenantInvitation]:
        rs = _filtered(
            [r for r in _live() if r.tenant_id == tenant_id],
            include_terminal=include_terminal,
        )
        return rs[offset : offset + limit] if limit is not None else rs

    async def _count_by_tenant(
        tenant_id: int, *, include_terminal: bool = False
    ) -> int:
        return len(
            _filtered(
                [r for r in _live() if r.tenant_id == tenant_id],
                include_terminal=include_terminal,
            )
        )

    async def _list_by_invitee(
        invitee_user_id: str, *, include_terminal: bool = False
    ) -> list[TenantInvitation]:
        return _filtered(
            [r for r in _live() if r.invitee_user_id == invitee_user_id],
            include_terminal=include_terminal,
        )

    async def _count_pending_by_invitee(invitee_user_id: str) -> int:
        return len(
            [
                r
                for r in _live()
                if r.invitee_user_id == invitee_user_id and r.status == STATUS_PENDING
            ]
        )

    repo.insert_pending_or_none.side_effect = _insert_pending_or_none
    repo.mark_status_if_pending.side_effect = _mark_status_if_pending
    repo.sweep_expired.side_effect = _sweep_expired
    repo.increment_accepted_count.side_effect = _increment_accepted_count
    repo.find_by_id_or_none.side_effect = _find_by_id_or_none
    repo.find_pending_by_pair.side_effect = _find_pending_by_pair
    repo.find_pending_by_token.side_effect = _find_pending_by_token
    repo.list_by_tenant.side_effect = _list_by_tenant
    repo.count_by_tenant.side_effect = _count_by_tenant
    repo.list_by_invitee.side_effect = _list_by_invitee
    repo.count_pending_by_invitee.side_effect = _count_pending_by_invitee
    return repo, rows


# ── Membership repository mock ──────────────────────────────────────


def _make_members_repo() -> tuple[AsyncMock, dict[int, object]]:
    """Minimal membership-repo mock used by ``TenantMemberService``.

    Only the surface exercised by the invitation cross-service hop is
    implemented: ``insert_live_or_none``, ``find_membership``,
    ``list_by_user``, ``list_by_tenant``, ``update_role``,
    ``soft_delete``, ``has_any_members``, ``count_by_tenant``,
    ``list_page_by_tenant``, ``count_active_owners``,
    ``count_other_active_owners_for_update``,
    ``soft_delete_by_tenant``.
    """
    repo = AsyncMock(spec=TenantMemberRepository)
    rows: dict[int, object] = {}

    async def _find_membership(*, user_id: str, tenant_id: int) -> object | None:
        for r in rows.values():
            if (
                r.user_id == user_id  # type: ignore[attr-defined]
                and r.tenant_id == tenant_id  # type: ignore[attr-defined]
                and r.deleted_at is None  # type: ignore[attr-defined]
            ):
                return r
        return None

    async def _insert_live_or_none(row: object) -> object | None:
        if await _find_membership(user_id=row.user_id, tenant_id=row.tenant_id):  # type: ignore[attr-defined]
            return None
        stored = row.model_copy(update={"id": max(rows.keys(), default=0) + 1})  # type: ignore[attr-defined]
        rows[stored.id] = stored
        return stored

    async def _update_role(
        *,
        user_id: str,
        tenant_id: int,
        role: str,
        updated_at: datetime,
    ) -> int:
        for r in rows.values():
            if (
                r.user_id == user_id  # type: ignore[attr-defined]
                and r.tenant_id == tenant_id  # type: ignore[attr-defined]
                and r.deleted_at is None  # type: ignore[attr-defined]
            ):
                rows[r.id] = r.model_copy(  # type: ignore[attr-defined]
                    update={"role": role, "updated_at": updated_at}
                )
                return 1
        return 0

    repo.find_membership.side_effect = _find_membership
    repo.insert_live_or_none.side_effect = _insert_live_or_none
    repo.update_role.side_effect = _update_role
    return repo, rows


@pytest.fixture
def invitations_state() -> tuple[AsyncMock, dict[int, TenantInvitation]]:
    return _make_invitations_repo()


@pytest.fixture
def invitations_repo(
    invitations_state: tuple[AsyncMock, dict[int, TenantInvitation]],
) -> AsyncMock:
    return invitations_state[0]


@pytest.fixture
def invitation_rows(
    invitations_state: tuple[AsyncMock, dict[int, TenantInvitation]],
) -> dict[int, TenantInvitation]:
    return invitations_state[1]


@pytest.fixture
def member_service() -> TenantMemberService:
    repo, _ = _make_members_repo()
    return TenantMemberService(members_repo=repo)


@pytest.fixture
def service(
    invitations_repo: AsyncMock,
    member_service: TenantMemberService,
) -> TenantInvitationService:
    return TenantInvitationService(
        invitations_repo=invitations_repo,
        member_service=member_service,
    )


def _expire(
    rows: dict[int, TenantInvitation], invitation_id: int
) -> None:
    """Backdate a row's expiry so the next sweep flips it."""
    row = rows[invitation_id]
    rows[invitation_id] = row.model_copy(
        update={"expires_at": datetime.now(UTC) - timedelta(minutes=1)}
    )


# ── token helper ────────────────────────────────────────────────────


class TestTokenHelper(ServiceTest):
    def test_share_link_token_is_unpadded_and_unique(self) -> None:
        token = generate_share_link_token()
        assert "=" not in token
        assert token != generate_share_link_token()


# ── create_invitation ───────────────────────────────────────────────


class TestCreateInvitation(ServiceTest):
    async def test_is_pending_with_ttl(
        self,
        service: TenantInvitationService,
    ) -> None:
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


# ── share-link invitation ───────────────────────────────────────────


class TestShareLinkInvitation(ServiceTest):
    async def test_share_link_is_pending_with_token(
        self,
        service: TenantInvitationService,
    ) -> None:
        invitation, token = await service.create_share_link(
            tenant_id=_TENANT_ID, role=ROLE_VIEWER
        )

        assert invitation.status == STATUS_PENDING
        assert invitation.invitee_user_id == ""
        assert invitation.is_share_link is True
        assert token

    async def test_share_link_token_is_not_exposed_on_the_dto(
        self,
        service: TenantInvitationService,
    ) -> None:
        invitation, _ = await service.create_share_link(
            tenant_id=_TENANT_ID, role=ROLE_VIEWER
        )

        assert "token" not in invitation.model_dump()

    async def test_multiple_share_links_can_coexist(
        self,
        service: TenantInvitationService,
    ) -> None:
        first, _ = await service.create_share_link(
            tenant_id=_TENANT_ID, role=ROLE_VIEWER
        )
        second, _ = await service.create_share_link(
            tenant_id=_TENANT_ID, role=ROLE_VIEWER
        )

        assert first.id != second.id

    async def test_lookup_by_token_resolves_the_row(
        self,
        service: TenantInvitationService,
    ) -> None:
        created, token = await service.create_share_link(
            tenant_id=_TENANT_ID, role=ROLE_VIEWER
        )

        found = await service.lookup_by_token(f"  {token}  ")

        assert found.id == created.id

    @pytest.mark.parametrize("token", ["", "   ", "sk-never-issued"])
    async def test_lookup_by_token_rejects_unknown_tokens(
        self,
        service: TenantInvitationService,
        token: str,
    ) -> None:
        with pytest.raises(NotFoundError) as excinfo:
            await service.lookup_by_token(token)
        assert excinfo.value.code == "tenant_invitation.invalid_token"

    async def test_lookup_by_token_rejects_expired_link(
        self,
        service: TenantInvitationService,
        invitation_rows: dict[int, TenantInvitation],
    ) -> None:
        created, token = await service.create_share_link(
            tenant_id=_TENANT_ID, role=ROLE_VIEWER
        )
        _expire(invitation_rows, created.id)

        with pytest.raises(NotFoundError):
            await service.lookup_by_token(token)

    async def test_accept_by_token_joins_and_keeps_the_link_pending(
        self,
        service: TenantInvitationService,
        invitation_rows: dict[int, TenantInvitation],
    ) -> None:
        created, token = await service.create_share_link(
            tenant_id=_TENANT_ID, role=ROLE_VIEWER
        )

        member = await service.accept_by_token(token, user_id="usr-new")

        assert member.tenant_id == _TENANT_ID
        assert member.role == ROLE_VIEWER
        stored = invitation_rows[created.id]
        assert stored.status == STATUS_PENDING
        assert stored.accepted_count == 1

    async def test_share_link_is_multi_use(
        self,
        service: TenantInvitationService,
        invitation_rows: dict[int, TenantInvitation],
    ) -> None:
        created, token = await service.create_share_link(
            tenant_id=_TENANT_ID, role=ROLE_VIEWER
        )

        await service.accept_by_token(token, user_id="usr-a")
        await service.accept_by_token(token, user_id="usr-b")

        assert invitation_rows[created.id].accepted_count == 2

    async def test_accept_by_token_twice_keeps_the_existing_role(
        self,
        service: TenantInvitationService,
        member_service: TenantMemberService,
    ) -> None:
        _, token = await service.create_share_link(
            tenant_id=_TENANT_ID, role=ROLE_VIEWER
        )
        await service.accept_by_token(token, user_id="usr-a")
        await member_service.update_role(
            user_id="usr-a",
            tenant_id=_TENANT_ID,
            role=ROLE_OWNER,
        )

        member = await service.accept_by_token(token, user_id="usr-a")

        assert member.role == ROLE_OWNER


# ── listing / sweeping ──────────────────────────────────────────────


class TestListing(ServiceTest):
    async def test_list_by_tenant_hides_terminal_rows_by_default(
        self,
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
        self,
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
        self,
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
        self,
        service: TenantInvitationService,
        invitation_rows: dict[int, TenantInvitation],
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
        _expire(invitation_rows, stale.id)

        assert await service.count_pending_by_invitee(_INVITEE) == 1

    async def test_expire_overdue_reports_swept_rows(
        self,
        service: TenantInvitationService,
        invitation_rows: dict[int, TenantInvitation],
    ) -> None:
        invitation = await service.create_invitation(
            tenant_id=_TENANT_ID,
            invitee_user_id=_INVITEE,
            role=ROLE_VIEWER,
        )
        _expire(invitation_rows, invitation.id)

        assert await service.expire_overdue() == 1
        assert await service.expire_overdue() == 0

    async def test_get_invitation_returns_none_when_missing(
        self,
        service: TenantInvitationService,
    ) -> None:
        assert await service.get_invitation(4242) is None