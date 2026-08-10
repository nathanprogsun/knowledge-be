"""Wire-shape conversion for the organization endpoints.

``OrganizationInfo`` / ``OrganizationMemberInfo`` / ``OrganizationJoinRequestInfo``
are the service-side projections; the wire shapes are the frozen contracts in
``src.core.contracts.organizations``. The conversion functions here perform the
boundary translation, re-emitting service DTOs onto the wire contracts.

Counts that need cross-domain services (``share_count`` / ``agent_share_count``)
are deferred seams and stay at their zero default until the sharing domain is
wired into this layer. The representative-user display fields (``username`` /
``email`` / ``avatar``) likewise stay empty until a user service is wired here.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.core.contracts.organizations import (
    JoinRequestListResponse,
    JoinRequestRecord,
    OrgMember,
    OrgMemberListResponse,
    Organization,
    OrganizationList,
    OrganizationPreview,
    RegenerateInviteCodeResponse,
    SearchOrganizationsResponse,
)
from src.core.organizations.types import (
    OrganizationInfo,
    OrganizationJoinRequestInfo,
    OrganizationMemberInfo,
)


class OrganizationEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` - single-org responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: Organization


class OrganizationListEnvelope(BaseModel):
    """``{"success": true, "data": {"items": [...]}}`` - list responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: OrganizationList


class SearchOrganizationsEnvelope(BaseModel):
    """``{"success": true, "data": {"items": [...]}}`` - search responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: SearchOrganizationsResponse


class OrganizationPreviewEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` - invite-code preview responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: OrganizationPreview


class InviteCodeEnvelope(BaseModel):
    """``{"success": true, "data": {"invite_code": "..."}}`` responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: RegenerateInviteCodeResponse


class MemberListEnvelope(BaseModel):
    """``{"success": true, "data": {"members": [...]}}`` responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: OrgMemberListResponse


class JoinRequestEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` - single join-request responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: JoinRequestRecord


class JoinRequestListEnvelope(BaseModel):
    """``{"success": true, "data": {"requests": [...]}}`` responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: JoinRequestListResponse


class SimpleAckResponse(BaseModel):
    """``{"success": true, "message": "..."}`` - simple ack responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str


class TenantInviteCandidate(BaseModel):
    """One row in the search-tenants-for-invite picker.

    Mirrors the tenant-centric candidate shape: the workspace identity is
    primary and the representative user is a label.
    """

    model_config = ConfigDict(frozen=True)

    tenant_id: int
    tenant_name: str = ""
    representative_user_id: str = ""
    representative_username: str = ""
    representative_email: str = ""
    representative_avatar: str = ""


class TenantInviteListEnvelope(BaseModel):
    """``{"success": true, "data": [...]}`` - invite-candidate responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: list[TenantInviteCandidate]


def org_info_to_contract(
    info: OrganizationInfo,
    *,
    member_count: int = 0,
    share_count: int = 0,
    agent_share_count: int = 0,
    pending_join_request_count: int = 0,
    is_owner: bool = False,
    my_role: str | None = None,
    has_pending_upgrade: bool = False,
) -> Organization:
    """Project the service DTO onto the frozen wire contract.

    ``invite_code`` is deliberately never rendered here: the service drops
    the join credential from its projections and it is exposed only through
    the dedicated regenerate endpoint.
    """
    return Organization(
        id=info.id,
        name=info.name,
        description=info.description,
        avatar=info.avatar,
        owner_id=info.owner_id,
        invite_code=None,
        invite_code_expires_at=None,
        invite_code_validity_days=info.invite_code_validity_days,
        require_approval=info.require_approval,
        searchable=info.searchable,
        member_limit=info.member_limit,
        member_count=member_count,
        share_count=share_count,
        agent_share_count=agent_share_count,
        pending_join_request_count=pending_join_request_count,
        is_owner=is_owner,
        my_role=my_role,
        has_pending_upgrade=has_pending_upgrade,
        created_at=info.created_at,
        updated_at=info.updated_at,
    )


def org_preview_to_contract(
    info: OrganizationInfo,
    *,
    member_count: int = 0,
    share_count: int = 0,
    agent_share_count: int = 0,
    is_already_member: bool = False,
) -> OrganizationPreview:
    """Project the service DTO onto the invite-code preview contract."""
    return OrganizationPreview(
        id=info.id,
        name=info.name,
        description=info.description,
        avatar=info.avatar,
        member_count=member_count,
        share_count=share_count,
        agent_share_count=agent_share_count,
        is_already_member=is_already_member,
        require_approval=info.require_approval,
        created_at=info.created_at,
    )


def org_preview_envelope(
    info: OrganizationInfo,
    *,
    member_count: int = 0,
    share_count: int = 0,
    agent_share_count: int = 0,
    is_already_member: bool = False,
) -> OrganizationPreviewEnvelope:
    """Wrap an invite-code preview in the success envelope."""
    return OrganizationPreviewEnvelope(
        success=True,
        data=org_preview_to_contract(
            info,
            member_count=member_count,
            share_count=share_count,
            agent_share_count=agent_share_count,
            is_already_member=is_already_member,
        ),
    )


def member_info_to_contract(info: OrganizationMemberInfo) -> OrgMember:
    """Project a tenant membership onto the wire ``OrgMember``.

    ``user_id`` is the representative user id (informational); the
    display fields stay empty until a user service is wired here.
    """
    return OrgMember(
        id=info.id,
        user_id=info.representative_user_id,
        username=None,
        email=None,
        avatar=None,
        role=info.role,
        tenant_id=info.tenant_id,
        joined_at=info.joined_at or info.created_at,
    )


def member_list_envelope(
    members: list[OrganizationMemberInfo],
) -> MemberListEnvelope:
    """Wrap a member roster in the success envelope."""
    return MemberListEnvelope(
        success=True,
        data=OrgMemberListResponse(
            members=[member_info_to_contract(m) for m in members],
            total=len(members),
        ),
    )


def join_request_info_to_contract(
    info: OrganizationJoinRequestInfo,
) -> JoinRequestRecord:
    """Project a join request onto the wire ``JoinRequestRecord``.

    ``username`` / ``email`` stay empty until a user service is wired
    here; the internal review trail is already dropped by the service
    projection.
    """
    return JoinRequestRecord(
        id=info.id,
        user_id=info.user_id,
        username=None,
        email=None,
        message=info.message,
        request_type=info.request_type,
        prev_role=info.prev_role,
        requested_role=info.requested_role,
        status=info.status,
        created_at=info.created_at,
    )


def join_request_envelope(
    request: OrganizationJoinRequestInfo,
) -> JoinRequestEnvelope:
    """Wrap one join request in the success envelope."""
    return JoinRequestEnvelope(
        success=True,
        data=join_request_info_to_contract(request),
    )


def join_request_list_envelope(
    requests: list[OrganizationJoinRequestInfo],
) -> JoinRequestListEnvelope:
    """Wrap a join-request queue in the success envelope."""
    return JoinRequestListEnvelope(
        success=True,
        data=JoinRequestListResponse(
            requests=[join_request_info_to_contract(r) for r in requests],
            total=len(requests),
        ),
    )


def search_organizations_envelope(
    previews: list[OrganizationPreview],
    *,
    total: int,
) -> SearchOrganizationsEnvelope:
    """Wrap a searchable-org result page in the success envelope."""
    return SearchOrganizationsEnvelope(
        success=True,
        data=SearchOrganizationsResponse(items=previews, total=total),
    )


def invite_code_envelope(code: str) -> InviteCodeEnvelope:
    """Wrap a freshly generated invite code in the success envelope."""
    return InviteCodeEnvelope(
        success=True,
        data=RegenerateInviteCodeResponse(invite_code=code),
    )


def tenant_invite_list_envelope(
    candidates: list[TenantInviteCandidate],
) -> TenantInviteListEnvelope:
    """Wrap the tenant-candidate picker rows in the success envelope."""
    return TenantInviteListEnvelope(success=True, data=candidates)


__all__ = [
    "InviteCodeEnvelope",
    "JoinRequestEnvelope",
    "JoinRequestListEnvelope",
    "MemberListEnvelope",
    "OrganizationEnvelope",
    "OrganizationListEnvelope",
    "OrganizationPreviewEnvelope",
    "SearchOrganizationsEnvelope",
    "SimpleAckResponse",
    "TenantInviteCandidate",
    "TenantInviteListEnvelope",
    "invite_code_envelope",
    "join_request_envelope",
    "join_request_info_to_contract",
    "join_request_list_envelope",
    "member_info_to_contract",
    "member_list_envelope",
    "org_info_to_contract",
    "org_preview_envelope",
    "org_preview_to_contract",
    "search_organizations_envelope",
    "tenant_invite_list_envelope",
]
