"""Wire envelopes for the ``/me/invitations`` inbox."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.core.tenants.types import MembershipInfo, TenantInvitationInfo


class InvitationItem(BaseModel):
    """One inbox row. Extra invitee/tenant display fields stay optional."""

    model_config = ConfigDict(frozen=True)

    id: int
    tenant_id: int
    tenant_name: str | None = None
    invitee_user_id: str
    invited_by: str | None = None
    role: str
    status: str
    message: str | None = None
    expires_at: datetime
    responded_at: datetime | None = None
    accepted_count: int = 0
    is_share_link: bool = False
    created_at: datetime


class InvitationListData(BaseModel):
    model_config = ConfigDict(frozen=True)

    invitations: list[InvitationItem]
    total: int


class InvitationListEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: InvitationListData


class MembershipPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: int
    role: str
    status: str
    joined_at: datetime


class AcceptInvitationData(BaseModel):
    model_config = ConfigDict(frozen=True)

    membership: MembershipPayload


class AcceptInvitationEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: AcceptInvitationData


class SimpleMessageEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool = True
    message: str = Field(default="")


def invitation_item(info: TenantInvitationInfo) -> InvitationItem:
    """Project a service DTO onto the inbox row."""
    return InvitationItem(
        id=info.id,
        tenant_id=info.tenant_id,
        invitee_user_id=info.invitee_user_id,
        invited_by=info.invited_by,
        role=info.role,
        status=info.status,
        message=info.message,
        expires_at=info.expires_at,
        responded_at=info.responded_at,
        accepted_count=info.accepted_count,
        is_share_link=info.is_share_link,
        created_at=info.created_at,
    )


def invitation_list_envelope(rows: list[TenantInvitationInfo]) -> InvitationListEnvelope:
    """Wrap the invitee's inbox."""
    return InvitationListEnvelope(
        data=InvitationListData(
            invitations=[invitation_item(row) for row in rows],
            total=len(rows),
        )
    )


def accept_invitation_envelope(member: MembershipInfo) -> AcceptInvitationEnvelope:
    """Wrap the membership created by accept."""
    return AcceptInvitationEnvelope(
        data=AcceptInvitationData(
            membership=MembershipPayload(
                tenant_id=member.tenant_id,
                role=member.role,
                status=member.status,
                joined_at=member.joined_at,
            )
        )
    )


__all__ = [
    "AcceptInvitationEnvelope",
    "InvitationListEnvelope",
    "SimpleMessageEnvelope",
    "accept_invitation_envelope",
    "invitation_list_envelope",
]
