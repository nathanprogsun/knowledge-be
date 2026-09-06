"""Personal endpoints under ``/me``.

Maps the upstream ``/me/invitations/*`` group of the tenant-invitation
handler. The inbox list, accept, decline, and pending-count routes share the
invitation service. The service enforces invitee ownership.

The routes are authenticated (no tenant-role gate) and resolve the
caller from the authenticated principal — the service enforces "only
the invitee can accept/decline".
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict

from src.common.exception import UnauthorizedError
from src.web.api.me.views import (
    AcceptInvitationEnvelope,
    InvitationListEnvelope,
    SimpleMessageEnvelope,
    accept_invitation_envelope,
    invitation_list_envelope,
)
from src.web.deps import AuthDep, get_request_user_id
from src.web.deps.tenants import TenantInvitationServiceDep

router = APIRouter(prefix="/me", tags=["me"])


def _require_user_id(request: Request) -> str:
    """Return the authenticated user id, or fail."""
    user_id = get_request_user_id(request)
    if not user_id:
        raise UnauthorizedError(
            code="auth.user_missing",
            message="caller user id missing from context",
        )
    return user_id


class PendingInvitationsCountData(BaseModel):
    """Inner ``data`` payload of the pending-count envelope."""

    model_config = ConfigDict(frozen=True)

    pending_count: int


class PendingInvitationsCountResponse(BaseModel):
    """``{"success": true, "data": {"pending_count": <n>}}``."""

    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: PendingInvitationsCountData


@router.get("/invitations/pending-count", response_model=PendingInvitationsCountResponse)
async def pending_invitations_count(
    _auth: AuthDep,
    request: Request,
    invitation_service: TenantInvitationServiceDep,
) -> PendingInvitationsCountResponse:
    """Return the authenticated user's pending-invitation count.

    Lightweight badge endpoint polled by the avatar row; no role gate
    beyond authentication (mirrors the upstream handler).
    """
    user_id = _require_user_id(request)
    count = await invitation_service.count_pending_by_invitee(user_id)
    return PendingInvitationsCountResponse(
        data=PendingInvitationsCountData(pending_count=count),
    )


@router.get("/invitations", response_model=InvitationListEnvelope)
async def list_my_invitations(
    _auth: AuthDep,
    request: Request,
    invitation_service: TenantInvitationServiceDep,
    include_terminal: bool = Query(default=False),
) -> InvitationListEnvelope:
    """Return the caller's invitation inbox."""
    rows = await invitation_service.list_by_invitee(
        _require_user_id(request),
        include_terminal=include_terminal,
    )
    return invitation_list_envelope(rows)


@router.post(
    "/invitations/{invitation_id}/accept",
    response_model=AcceptInvitationEnvelope,
)
async def accept_my_invitation(
    _auth: AuthDep,
    request: Request,
    invitation_id: int,
    invitation_service: TenantInvitationServiceDep,
) -> AcceptInvitationEnvelope:
    """Accept one pending invitation and join that workspace."""
    member = await invitation_service.accept(
        invitation_id,
        user_id=_require_user_id(request),
    )
    return accept_invitation_envelope(member)


@router.post(
    "/invitations/{invitation_id}/decline",
    response_model=SimpleMessageEnvelope,
)
async def decline_my_invitation(
    _auth: AuthDep,
    request: Request,
    invitation_id: int,
    invitation_service: TenantInvitationServiceDep,
) -> SimpleMessageEnvelope:
    """Decline one pending invitation."""
    await invitation_service.decline(
        invitation_id,
        user_id=_require_user_id(request),
    )
    return SimpleMessageEnvelope(message="Invitation declined")


__all__ = ["router"]
