"""Wire-shape conversion for the tenant endpoints.

``include_secrets`` defaults to ``False`` because role resolution is not
wired up yet: the four secret-bearing config blobs (``web_search_config``,
``parser_engine_config``, ``credentials``, ``storage_engine_config``) are
redacted in every response until a caller is allowed to see them.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.app_context.request_context import get_user_id
from src.common.exception import NotFoundError, ValidationError
from src.core.auth.service import AuthService
from src.core.contracts.tenants import (
    RetrieverEngineEntry,
    RetrieverEnginesConfig,
    Tenant,
    TenantList,
)
from src.core.tenants.invitation_service import TenantInvitationService
from src.core.tenants.types import (
    MembershipInfo,
    RetrieverEngines,
    TenantInfo,
    TenantInvitationInfo,
)


class TenantEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` - single-tenant responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: Tenant


class TenantListEnvelope(BaseModel):
    """``{"success": true, "data": [...], "total": ..., "page": ..., "page_size": ...}``.

    Mirrors the project's pagination shape: ``data`` carries the list
    payload (not ``items``) and the pagination metadata are siblings.
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    data: TenantList


class DeleteTenantResponse(BaseModel):
    """``{"success": true, "message": "..."}`` - simple ack response."""

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str


def _to_engines_config(engines: RetrieverEngines) -> RetrieverEnginesConfig:
    return RetrieverEnginesConfig(
        engines=[
            RetrieverEngineEntry(
                retriever_type=entry.retriever_type,
                retriever_engine_type=entry.retriever_engine_type,
            )
            for entry in engines.engines
        ]
    )


def tenant_info_to_contract(info: TenantInfo, *, include_secrets: bool = False) -> Tenant:
    """Project the service DTO onto the frozen wire contract."""
    return Tenant(
        id=info.id,
        name=info.name,
        description=info.description,
        status=info.status,
        retriever_engines=_to_engines_config(info.retriever_engines),
        business=info.business,
        storage_quota=info.storage_quota,
        storage_used=info.storage_used,
        context_config=info.context_config,
        chat_history_config=info.chat_history_config,
        retrieval_config=info.retrieval_config,
        web_search_config=info.web_search_config if include_secrets else None,
        parser_engine_config=info.parser_engine_config if include_secrets else None,
        credentials=info.credentials if include_secrets else None,
        storage_engine_config=info.storage_engine_config if include_secrets else None,
        created_at=info.created_at,
        updated_at=info.updated_at,
        deleted_at=info.deleted_at,
    )


def tenant_envelope(info: TenantInfo) -> TenantEnvelope:
    """Wrap one tenant in the success envelope."""
    return TenantEnvelope(success=True, data=tenant_info_to_contract(info))


def tenant_list_envelope(
    infos: list[TenantInfo],
    *,
    total: int | None = None,
    page: int | None = None,
    page_size: int | None = None,
) -> TenantListEnvelope:
    """Wrap a tenant page in the success envelope.

    Field names mirror Go's pagination contract: ``data`` carries the
    list payload; ``total`` / ``page`` / ``page_size`` are siblings
    on the inner pagination object (defaults 0 / 1 / 20 for unpaginated
    list endpoints so the wire shape stays uniform).
    """
    return TenantListEnvelope(
        success=True,
        data=TenantList(
            data=[tenant_info_to_contract(info) for info in infos],
            total=total or 0,
            page=page or 1,
            page_size=page_size or 20,
        ),
    )


# Share-link copy target. The SPA prefixes the current origin when the
# value is host-relative, which is the usual local-dev case.
_SHARE_LINK_PATH = "/register?token="
_DEFAULT_PAGE = 1
_DEFAULT_PAGE_SIZE = 20
_MAX_PAGE_SIZE = 100


def clamp_page(page: int, page_size: int) -> tuple[int, int]:
    """Clamp page to >= 1 and page_size into [1, 100]."""
    safe_page = page if page >= 1 else _DEFAULT_PAGE
    safe_size = page_size if page_size >= 1 else _DEFAULT_PAGE_SIZE
    return safe_page, min(safe_size, _MAX_PAGE_SIZE)


class TenantMemberItem(BaseModel):
    """One workspace member as the members table already renders it."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    email: str = ""
    username: str = ""
    avatar: str | None = None
    role: str
    status: str
    invited_by: str | None = None
    joined_at: datetime


class MemberListData(BaseModel):
    model_config = ConfigDict(frozen=True)

    members: list[TenantMemberItem]
    total: int
    page: int
    page_size: int


class MemberListEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: MemberListData


class MemberEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: TenantMemberItem


class AddMemberBody(BaseModel):
    model_config = ConfigDict(frozen=True)

    email: str
    role: str


class UpdateMemberRoleBody(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str


class TenantInvitationItem(BaseModel):
    """One invitation row. Display names stay optional when no join filled them."""

    model_config = ConfigDict(frozen=True)

    id: int
    tenant_id: int
    tenant_name: str | None = None
    invitee_user_id: str
    invitee_email: str | None = None
    invitee_name: str | None = None
    invited_by: str | None = None
    inviter_email: str | None = None
    inviter_name: str | None = None
    role: str
    status: str
    message: str | None = None
    expires_at: datetime
    responded_at: datetime | None = None
    created_at: datetime
    invite_url: str | None = None
    is_share_link: bool = False
    accepted_count: int = 0


class InvitationListData(BaseModel):
    model_config = ConfigDict(frozen=True)

    invitations: list[TenantInvitationItem]
    total: int
    page: int
    page_size: int


class InvitationListEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: InvitationListData


class InvitationEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: TenantInvitationItem


class CreateInvitationBody(BaseModel):
    model_config = ConfigDict(frozen=True)

    email: str
    role: str
    message: str | None = None


class CreateInviteLinkBody(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str
    message: str | None = None


class ResolvedUser(BaseModel):
    """Registered user fields needed at the members/invitations boundary."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    email: str
    username: str
    avatar: str | None = Field(default=None)


def invite_url_for_token(token: str) -> str:
    """Build the host-relative share-link URL the SPA copies."""
    return f"{_SHARE_LINK_PATH}{token}"


def require_caller_user_id() -> str:
    """Return the authenticated user id, or fail closed."""
    user_id = get_user_id()
    if user_id is None:
        raise ValidationError(
            code="auth.missing_user_context",
            message="No authenticated user in request context",
        )
    return user_id


async def resolve_registered_user(auth_service: AuthService, email: str) -> ResolvedUser:
    """Resolve an invite/add email to a registered user, or 404."""
    trimmed = email.strip()
    if not trimmed:
        raise ValidationError(
            code="tenant_member.email_required",
            message="email is required",
        )
    user = await auth_service.get_user_row_by_email(trimmed)
    if user is None:
        raise NotFoundError(
            code="user.not_found",
            message="No registered user with that email",
        )
    return ResolvedUser(
        user_id=user.id,
        email=user.email,
        username=user.username,
        avatar=user.avatar,
    )


async def require_tenant_invitation(
    invitation_service: TenantInvitationService,
    *,
    tenant_id: int,
    inv_id: int,
) -> TenantInvitationInfo:
    """Load an invitation that belongs to ``tenant_id``, or hide it as missing."""
    invitation = await invitation_service.get_invitation(inv_id)
    if invitation is None or invitation.tenant_id != tenant_id:
        raise NotFoundError(
            code="tenant_invitation.not_found",
            message="Invitation not found",
        )
    return invitation


def member_item(
    info: MembershipInfo,
    *,
    email: str = "",
    username: str = "",
    avatar: str | None = None,
) -> TenantMemberItem:
    """Project a membership DTO. Email and name stay empty unless a caller has them."""
    return TenantMemberItem(
        user_id=info.user_id,
        email=email,
        username=username,
        avatar=avatar,
        role=info.role,
        status=info.status,
        invited_by=info.invited_by,
        joined_at=info.joined_at,
    )


def member_envelope(
    info: MembershipInfo,
    *,
    email: str = "",
    username: str = "",
    avatar: str | None = None,
) -> MemberEnvelope:
    return MemberEnvelope(data=member_item(info, email=email, username=username, avatar=avatar))


def member_list_envelope(
    rows: list[MembershipInfo],
    *,
    total: int,
    page: int,
    page_size: int,
) -> MemberListEnvelope:
    return MemberListEnvelope(
        data=MemberListData(
            members=[member_item(row) for row in rows],
            total=total,
            page=page,
            page_size=page_size,
        )
    )


def invitation_item(
    info: TenantInvitationInfo,
    *,
    invitee_email: str | None = None,
    invitee_name: str | None = None,
    invite_url: str | None = None,
) -> TenantInvitationItem:
    """Project an invitation DTO onto the members-page row."""
    return TenantInvitationItem(
        id=info.id,
        tenant_id=info.tenant_id,
        invitee_user_id=info.invitee_user_id,
        invitee_email=invitee_email,
        invitee_name=invitee_name,
        invited_by=info.invited_by,
        role=info.role,
        status=info.status,
        message=info.message,
        expires_at=info.expires_at,
        responded_at=info.responded_at,
        created_at=info.created_at,
        invite_url=invite_url,
        is_share_link=info.is_share_link,
        accepted_count=info.accepted_count,
    )


def invitation_envelope(
    info: TenantInvitationInfo,
    *,
    invitee_email: str | None = None,
    invitee_name: str | None = None,
    invite_url: str | None = None,
) -> InvitationEnvelope:
    return InvitationEnvelope(
        data=invitation_item(
            info,
            invitee_email=invitee_email,
            invitee_name=invitee_name,
            invite_url=invite_url,
        )
    )


def invitation_list_envelope(
    rows: list[TenantInvitationInfo],
    *,
    total: int,
    page: int,
    page_size: int,
) -> InvitationListEnvelope:
    return InvitationListEnvelope(
        data=InvitationListData(
            invitations=[invitation_item(row) for row in rows],
            total=total,
            page=page,
            page_size=page_size,
        )
    )


__all__ = [
    "AddMemberBody",
    "CreateInvitationBody",
    "CreateInviteLinkBody",
    "DeleteTenantResponse",
    "InvitationEnvelope",
    "InvitationListEnvelope",
    "MemberEnvelope",
    "MemberListEnvelope",
    "ResolvedUser",
    "TenantEnvelope",
    "TenantInvitationItem",
    "TenantListEnvelope",
    "TenantMemberItem",
    "UpdateMemberRoleBody",
    "clamp_page",
    "invitation_envelope",
    "invitation_item",
    "invitation_list_envelope",
    "invite_url_for_token",
    "member_envelope",
    "member_item",
    "member_list_envelope",
    "require_caller_user_id",
    "require_tenant_invitation",
    "resolve_registered_user",
    "tenant_envelope",
    "tenant_info_to_contract",
    "tenant_list_envelope",
]
