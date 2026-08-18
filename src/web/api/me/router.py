"""Personal endpoints under ``/me``.

Maps the upstream ``/me/invitations/*`` group of the tenant-invitation
handler. Only the lightweight pending-count endpoint is implemented so
far; the list / accept / decline routes land with the full invitation
inbox.

The routes are authenticated (no tenant-role gate) and resolve the
caller from the authenticated principal — the service enforces "only
the invitee can accept/decline".
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict

from src.common.exception import UnauthorizedError
from src.web.deps import AuthDep, get_request_user_id
from src.web.deps.tenants import TenantInvitationServiceDep

router = APIRouter(prefix="/me", tags=["me"])


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
    user_id = get_request_user_id(request)
    if not user_id:
        raise UnauthorizedError(
            code="auth.user_missing",
            message="caller user id missing from context",
        )
    count = await invitation_service.count_pending_by_invitee(user_id)
    return PendingInvitationsCountResponse(
        data=PendingInvitationsCountData(pending_count=count),
    )


__all__ = ["router"]
