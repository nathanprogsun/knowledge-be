"""Tenant invitation service — issue, accept, decline, revoke, expire.

Two kinds of invitation share one table:

- **per-user**: an Owner invites a registered user. At most one pending
  invitation per (workspace, invitee); accepting creates the membership
  and finalises the row.
- **share link**: no specific invitee, a random token in the URL,
  reusable. Accepting adds the caller and leaves the row pending, only
  bumping ``accepted_count``.

Expiry is swept lazily: every list/accept/decline/revoke/count first
flips overdue pending rows to ``expired``, so a stale row never reads
as actionable. Membership creation is delegated to
``TenantMemberService`` so the invitation transition and the
membership insert commit together.
"""

from __future__ import annotations

import base64
import secrets
from datetime import UTC, datetime, timedelta

from src.common.exception import ConflictError, NotFoundError, PermissionDeniedError
from src.core.tenants.member_service import TenantMemberService
from src.core.tenants.types import MembershipInfo, TenantInvitationInfo
from src.db.dao.tenant_invitations_repository import TenantInvitationRepository
from src.db.models.tenants.tenant_invitations import (
    STATUS_ACCEPTED,
    STATUS_DECLINED,
    STATUS_PENDING,
    STATUS_REVOKED,
    TenantInvitation,
    is_expired,
)

# Default TTL for an invitation: seven days.
DEFAULT_INVITATION_TTL = timedelta(days=7)

# 32 raw bytes -> 256 bits, well above the 128-bit floor for an opaque
# unguessable token.
_TOKEN_ENTROPY_BYTES = 32

# Defensive clamps on the list endpoint.
_DEFAULT_PAGE_SIZE = 20
_MAX_PAGE_SIZE = 100


def generate_share_link_token() -> str:
    """Mint a share-link token (base64url, unpadded)."""
    return base64.urlsafe_b64encode(secrets.token_bytes(_TOKEN_ENTROPY_BYTES)).decode().rstrip("=")


class TenantInvitationService:
    """Stateless invitation service, constructed per request."""

    def __init__(
        self,
        *,
        invitations_repo: TenantInvitationRepository,
        member_service: TenantMemberService,
        ttl: timedelta = DEFAULT_INVITATION_TTL,
    ) -> None:
        self._invitations_repo = invitations_repo
        self._member_service = member_service
        self._ttl = ttl

    # ── Issuing ─────────────────────────────────────────────────────

    async def create_invitation(
        self,
        *,
        tenant_id: int,
        invitee_user_id: str,
        role: str,
        invited_by: str | None = None,
        message: str | None = None,
    ) -> TenantInvitationInfo:
        """Invite one registered user to the workspace."""
        self._member_service.require_valid_role(role)
        existing = await self._member_service.get_membership(
            user_id=invitee_user_id,
            tenant_id=tenant_id,
        )
        if existing is not None and existing.status == "active":
            raise ConflictError(
                code="tenant_invitation.already_member",
                message="User is already an active member of the workspace",
            )
        stored = await self._invitations_repo.insert_pending_or_none(
            self._new_row(
                tenant_id=tenant_id,
                invitee_user_id=invitee_user_id,
                role=role,
                invited_by=invited_by,
                message=message,
            )
        )
        if stored is None:
            raise ConflictError(
                code="tenant_invitation.pending_exists",
                message="A pending invitation for this user already exists",
            )
        return TenantInvitationInfo.map_from_db(stored)

    async def create_share_link(
        self,
        *,
        tenant_id: int,
        role: str,
        invited_by: str | None = None,
        message: str | None = None,
    ) -> tuple[TenantInvitationInfo, str]:
        """Issue a reusable share-link invitation; return it and its token.

        The per-user constraints (already-member, duplicate-pending) do
        not apply: a share link has no invitee, several can coexist on
        one workspace, and consuming one does not finalise it.
        """
        self._member_service.require_valid_role(role)
        token = generate_share_link_token()
        stored = await self._invitations_repo.insert_pending_or_none(
            self._new_row(
                tenant_id=tenant_id,
                invitee_user_id="",
                role=role,
                invited_by=invited_by,
                message=message,
                token=token,
            )
        )
        if stored is None:
            raise ConflictError(
                code="tenant_invitation.pending_exists",
                message="Could not issue the share link",
            )
        return TenantInvitationInfo.map_from_db(stored), token

    # ── Responding ──────────────────────────────────────────────────

    async def accept(self, invitation_id: int, *, user_id: str) -> MembershipInfo:
        """Accept a per-user invitation and join the workspace.

        Idempotent against an existing membership: a user who is already
        a member keeps their current role rather than being re-added.
        """
        invitation = await self._require_actionable(invitation_id, user_id=user_id)
        await self._transition(invitation_id, status=STATUS_ACCEPTED)
        return await self._join(
            user_id=user_id,
            tenant_id=invitation.tenant_id,
            role=invitation.role,
            invited_by=invitation.invited_by,
        )

    async def decline(self, invitation_id: int, *, user_id: str) -> None:
        """Decline a per-user invitation."""
        await self._require_actionable(invitation_id, user_id=user_id)
        await self._transition(invitation_id, status=STATUS_DECLINED)

    async def revoke(self, invitation_id: int) -> None:
        """Cancel a pending invitation.

        The caller's authority over the workspace is enforced at the
        route layer.
        """
        await self._sweep()
        invitation = await self._require_pending(invitation_id)
        await self._transition(invitation.id, status=STATUS_REVOKED)

    async def lookup_by_token(self, token: str) -> TenantInvitationInfo:
        """Resolve a share-link token; reject unknown or expired links."""
        await self._sweep()
        row = await self._invitations_repo.find_pending_by_token(token.strip())
        if row is None or is_expired(row, datetime.now(UTC)):
            raise NotFoundError(
                code="tenant_invitation.invalid_token",
                message="Invitation token is invalid or has been revoked",
            )
        return TenantInvitationInfo.map_from_db(row)

    async def accept_by_token(self, token: str, *, user_id: str) -> MembershipInfo:
        """Join a workspace through a share link.

        The row stays pending — share links are multi-use — and only its
        usage counter moves.
        """
        invitation = await self.lookup_by_token(token)
        member = await self._join(
            user_id=user_id,
            tenant_id=invitation.tenant_id,
            role=invitation.role,
            invited_by=invitation.invited_by,
        )
        await self._invitations_repo.increment_accepted_count(invitation.id)
        return member

    # ── Reads ───────────────────────────────────────────────────────

    async def get_invitation(self, invitation_id: int) -> TenantInvitationInfo | None:
        """Return one invitation without sweeping, or ``None``."""
        row = await self._invitations_repo.find_by_id_or_none(invitation_id)
        return TenantInvitationInfo.map_from_db(row) if row is not None else None

    async def list_by_tenant(
        self,
        tenant_id: int,
        *,
        include_terminal: bool = False,
    ) -> list[TenantInvitationInfo]:
        """Invitations of one workspace, newest first."""
        await self._sweep()
        rows = await self._invitations_repo.list_by_tenant(
            tenant_id,
            include_terminal=include_terminal,
        )
        return [TenantInvitationInfo.map_from_db(row) for row in rows]

    async def list_tenant_invitations_page(
        self,
        tenant_id: int,
        *,
        include_terminal: bool = False,
        page: int = 1,
        page_size: int = _DEFAULT_PAGE_SIZE,
    ) -> tuple[list[TenantInvitationInfo], int]:
        """One page of a workspace's invitations plus the total."""
        await self._sweep()
        page = max(page, 1)
        page_size = page_size if page_size >= 1 else _DEFAULT_PAGE_SIZE
        page_size = min(page_size, _MAX_PAGE_SIZE)
        total = await self._invitations_repo.count_by_tenant(
            tenant_id,
            include_terminal=include_terminal,
        )
        rows = await self._invitations_repo.list_by_tenant(
            tenant_id,
            include_terminal=include_terminal,
            limit=page_size,
            offset=(page - 1) * page_size,
        )
        return [TenantInvitationInfo.map_from_db(row) for row in rows], total

    async def list_by_invitee(
        self,
        invitee_user_id: str,
        *,
        include_terminal: bool = False,
    ) -> list[TenantInvitationInfo]:
        """One user's invitation inbox, newest first."""
        await self._sweep()
        rows = await self._invitations_repo.list_by_invitee(
            invitee_user_id,
            include_terminal=include_terminal,
        )
        return [TenantInvitationInfo.map_from_db(row) for row in rows]

    async def count_pending_by_invitee(self, invitee_user_id: str) -> int:
        """Count a user's pending invitations (the inbox badge)."""
        await self._sweep()
        return await self._invitations_repo.count_pending_by_invitee(invitee_user_id)

    async def expire_overdue(self) -> int:
        """Run the expiry sweep explicitly; return rows flipped."""
        return await self._sweep()

    # ── Internal helpers ────────────────────────────────────────────

    def _new_row(
        self,
        *,
        tenant_id: int,
        invitee_user_id: str,
        role: str,
        invited_by: str | None,
        message: str | None,
        token: str = "",
    ) -> TenantInvitation:
        now = datetime.now(UTC)
        return TenantInvitation(
            tenant_id=tenant_id,
            invitee_user_id=invitee_user_id,
            token=token,
            invited_by=invited_by,
            role=role,
            status=STATUS_PENDING,
            message=message,
            expires_at=now + self._ttl,
            created_at=now,
            updated_at=now,
        )

    async def _sweep(self) -> int:
        """Flip overdue pending rows to expired before reading them."""
        return await self._invitations_repo.sweep_expired(datetime.now(UTC))

    async def _require_pending(self, invitation_id: int) -> TenantInvitation:
        row = await self._invitations_repo.find_by_id_or_none(invitation_id)
        if row is None:
            raise NotFoundError(
                code="tenant_invitation.not_found",
                message="Invitation not found",
            )
        if row.status != STATUS_PENDING:
            raise ConflictError(
                code="tenant_invitation.not_pending",
                message="Invitation is no longer pending",
            )
        return row

    async def _require_actionable(self, invitation_id: int, *, user_id: str) -> TenantInvitation:
        """Sweep, then load a row the given user may accept or decline."""
        await self._sweep()
        row = await self._require_pending(invitation_id)
        if row.invitee_user_id != user_id:
            raise PermissionDeniedError(
                code="tenant_invitation.forbidden",
                message="Only the invitee can accept or decline this invitation",
            )
        if is_expired(row, datetime.now(UTC)):
            # The sweep above normally catches this; a row can still age
            # past its expiry between that UPDATE and this read.
            raise ConflictError(
                code="tenant_invitation.expired",
                message="Invitation has expired",
            )
        return row

    async def _transition(self, invitation_id: int, *, status: str) -> None:
        affected = await self._invitations_repo.mark_status_if_pending(
            invitation_id,
            status=status,
            responded_at=datetime.now(UTC),
        )
        if affected == 0:
            # A concurrent responder won; honour the state machine.
            raise ConflictError(
                code="tenant_invitation.not_pending",
                message="Invitation is no longer pending",
            )

    async def _join(
        self,
        *,
        user_id: str,
        tenant_id: int,
        role: str,
        invited_by: str | None,
    ) -> MembershipInfo:
        """Add the membership, tolerating an already-joined user."""
        try:
            return await self._member_service.add_member(
                user_id=user_id,
                tenant_id=tenant_id,
                role=role,
                invited_by=invited_by,
            )
        except ConflictError:
            existing = await self._member_service.get_membership(
                user_id=user_id,
                tenant_id=tenant_id,
            )
            if existing is None:
                raise
            return existing


__all__ = [
    "DEFAULT_INVITATION_TTL",
    "TenantInvitationService",
    "generate_share_link_token",
]
