"""Organization HTTP endpoints - CRUD, membership, join requests, invites.

Registered by the app factory.

Route order matters: the static paths (``/search``, ``/join``,
``/join-request``, ``/join-by-id``, ``/preview/{code}``) are declared
before the ``/{id}``-shaped routes so a literal segment is never captured
as an id.

The role floors mirror the upstream route guard: Admin for mutating and
admin-gated actions, Viewer for read-only actions. Every endpoint reads
the caller's workspace id from the request context; a missing context
fails closed with 401.

Cross-domain enrichment (KB / agent share counts) stays a deferred seam
in this layer; those counts default to zero.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict

from src.common.exception import (
    NotFoundError,
    PermissionDeniedError,
    UnauthorizedError,
    ValidationError,
)
from src.core.contracts.organizations import (
    CreateOrganizationRequest,
    JoinOrganizationByCodeRequest,
    JoinRequestByIDRequest,
    JoinRequestRequest,
    Organization,
    OrganizationList,
    RequestRoleUpgradeRequest,
    ReviewJoinRequestRequest,
    UpdateMemberRoleRequest,
    UpdateOrganizationRequest,
)
from src.core.organizations.service.organization_service import (
    ORG_ROLE_ADMIN,
    ORG_ROLE_EDITOR,
    ORG_ROLE_VIEWER,
    OrganizationService,
)
from src.core.organizations.types import OrganizationInfo
from src.web.api.organizations.views import (
    InviteCodeEnvelope,
    JoinRequestEnvelope,
    JoinRequestListEnvelope,
    MemberListEnvelope,
    OrganizationEnvelope,
    OrganizationListEnvelope,
    OrganizationPreviewEnvelope,
    SearchOrganizationsEnvelope,
    SimpleAckResponse,
    TenantInviteCandidate,
    TenantInviteListEnvelope,
    invite_code_envelope,
    join_request_envelope,
    join_request_list_envelope,
    member_list_envelope,
    org_info_to_contract,
    org_preview_envelope,
    org_preview_to_contract,
    search_organizations_envelope,
    tenant_invite_list_envelope,
)
from src.web.deps import AuthDep, RoleAdminDep, RoleViewerDep
from src.web.deps.context import get_tenant_id_dep, get_user_id_dep
from src.web.deps.organizations import OrganizationServiceDep
from src.web.deps.tenants import TenantServiceDep

# Function-arg-style principal deps.
_PrincipalTenant = Annotated[int, Depends(get_tenant_id_dep)]
_PrincipalUser = Annotated[str | None, Depends(get_user_id_dep)]


router = APIRouter(prefix="/organizations", tags=["organizations"])

# Acknowledge messages the UI matches verbatim (mirrors upstream).
_DELETE_MESSAGE = "Organization deleted successfully"
_LEAVE_MESSAGE = "Left organization successfully"
_MEMBER_REMOVED_MESSAGE = "Member removed successfully"
_MEMBER_ROLE_UPDATED_MESSAGE = "Member role updated successfully"
_REVIEW_MESSAGE = "Review completed"
_MEMBER_ADDED_MESSAGE = "Member added successfully"

_VALID_ROLES: frozenset[str] = frozenset(
    {ORG_ROLE_ADMIN, ORG_ROLE_EDITOR, ORG_ROLE_VIEWER}
)


class InviteMemberRequest(BaseModel):
    """Direct-invite body matching the workspace-centric invite shape.

    ``tenant_id`` is the preferred field; ``user_id`` is the legacy
    fallback retained for older SDK callers. ``representative_user_id``
    optionally pins the display user attached to the membership row.
    """

    model_config = ConfigDict(frozen=True)

    tenant_id: int = 0
    representative_user_id: str = ""
    user_id: str = ""
    role: str


def _require_tenant(tenant_id: int) -> int:
    """Return the active workspace id, or fail closed.

    Membership is workspace-scoped; without a workspace context there is
    no safe default, so this rejects rather than guessing.
    """
    if tenant_id == 0:
        raise UnauthorizedError(
            code="organization.tenant_context_missing",
            message="unauthorized: workspace context missing",
        )
    return tenant_id


def _require_user(user_id: str | None) -> str:
    """Return the authenticated user id, or fail closed."""
    if not user_id:
        raise UnauthorizedError(
            code="organization.user_context_missing",
            message="unauthorized: user context missing",
        )
    return user_id


def _require_valid_role(role: str) -> None:
    """Reject a role outside the sanctioned set."""
    if role not in _VALID_ROLES:
        raise ValidationError(
            code="organization.role_invalid",
            message="Invalid role; must be viewer, editor, or admin",
        )


def _is_owner(
    info: OrganizationInfo,
    *,
    tenant_id: int,
    user_id: str | None,
) -> bool:
    """Whether the caller is the owner side of the organization.

    The persisted owning workspace is the canonical check; legacy rows
    with a zero owning workspace fall back to the user-id rule.
    """
    if info.owner_tenant_id != 0:
        return info.owner_tenant_id == tenant_id
    return user_id is not None and info.owner_id == user_id


async def _org_contract(
    *,
    service: OrganizationService,
    info: OrganizationInfo,
    tenant_id: int,
    user_id: str | None,
) -> Organization:
    """Enrich one org DTO with member / role / owner context.

    ``share_count`` / ``agent_share_count`` stay zero because the sharing
    domains are not wired into this layer yet.
    """
    members = await service.list_tenant_members(org_id=info.id)
    my_role: str | None = None
    try:
        my_role = await service.get_tenant_role_in_org(
            org_id=info.id, tenant_id=tenant_id
        )
    except NotFoundError:
        my_role = None
    is_owner = _is_owner(info, tenant_id=tenant_id, user_id=user_id)
    pending = 0
    if is_owner or my_role == ORG_ROLE_ADMIN:
        pending = await service.count_pending_join_requests(org_id=info.id)
    has_pending_upgrade = False
    try:
        await service.get_pending_upgrade_request(
            org_id=info.id, tenant_id=tenant_id
        )
        has_pending_upgrade = True
    except NotFoundError:
        has_pending_upgrade = False
    return org_info_to_contract(
        info,
        member_count=len(members),
        pending_join_request_count=pending,
        is_owner=is_owner,
        my_role=my_role,
        has_pending_upgrade=has_pending_upgrade,
    )


async def _org_envelope(
    *,
    service: OrganizationService,
    info: OrganizationInfo,
    tenant_id: int,
    user_id: str | None,
) -> OrganizationEnvelope:
    """Wrap one enriched organization in the success envelope."""
    data = await _org_contract(
        service=service, info=info, tenant_id=tenant_id, user_id=user_id
    )
    return OrganizationEnvelope(success=True, data=data)


async def _member_visible_org(
    service: OrganizationService, org_id: str, tenant_id: int
) -> None:
    """Gate on membership: non-members of private orgs read as not-found.

    Mirrors the upstream visibility gate: a caller may read an org when
    its workspace is a member, or when the org has opted into discovery.
    """
    org = await service.get_organization(id=org_id)
    if not org.searchable:
        try:
            await service.get_tenant_member(org_id=org_id, tenant_id=tenant_id)
        except NotFoundError:
            raise NotFoundError(
                code="organization.not_found",
                message="Organization not found",
            )


async def _require_admin(
    service: OrganizationService, org_id: str, tenant_id: int
) -> None:
    """Require the caller's workspace to hold the admin role in the org."""
    is_admin = await service.is_tenant_org_admin(org_id=org_id, tenant_id=tenant_id)
    if not is_admin:
        raise PermissionDeniedError(
            code="organization.admin_required",
            message="Only organization admins can perform this action",
        )


async def _is_member(
    service: OrganizationService, org_id: str, tenant_id: int
) -> bool:
    """Whether the caller's workspace is already a member of the org."""
    try:
        await service.get_tenant_member(org_id=org_id, tenant_id=tenant_id)
        return True
    except NotFoundError:
        return False


# ── CRUD ─────────────────────────────────────────────────────────────


@router.post("", response_model=OrganizationEnvelope, status_code=201)
async def create_organization(
    _auth: AuthDep,
    _role: RoleAdminDep,
    body: CreateOrganizationRequest,
    service: OrganizationServiceDep,
    tenant_id: _PrincipalTenant,
    user_id: _PrincipalUser,
) -> OrganizationEnvelope:
    """Create an organization; the creator's workspace becomes admin."""
    tenant_id = _require_tenant(tenant_id)
    caller = _require_user(user_id)
    info = await service.create_organization(
        user_id=caller,
        tenant_id=tenant_id,
        name=body.name,
        description=body.description,
        avatar=body.avatar,
        invite_code_validity_days=body.invite_code_validity_days,
        member_limit=body.member_limit,
    )
    return await _org_envelope(
        service=service, info=info, tenant_id=tenant_id, user_id=caller
    )


@router.get("", response_model=OrganizationListEnvelope)
async def list_my_organizations(
    _auth: AuthDep,
    _role: RoleViewerDep,
    service: OrganizationServiceDep,
    tenant_id: _PrincipalTenant,
    user_id: _PrincipalUser,
) -> OrganizationListEnvelope:
    """List every organization the caller's workspace participates in."""
    tenant_id = _require_tenant(tenant_id)
    infos = await service.list_tenant_organizations(tenant_id=tenant_id)
    items = [
        await _org_contract(
            service=service, info=info, tenant_id=tenant_id, user_id=user_id
        )
        for info in infos
    ]
    return OrganizationListEnvelope(
        success=True,
        data=OrganizationList(items=items, total=len(items)),
    )


@router.get("/search", response_model=SearchOrganizationsEnvelope)
async def search_organizations(
    _auth: AuthDep,
    _role: RoleViewerDep,
    service: OrganizationServiceDep,
    tenant_id: _PrincipalTenant,
    q: str = Query(default="", description="搜索关键词（空间名称或描述）"),
    limit: int = Query(default=20, description="返回数量限制"),
) -> SearchOrganizationsEnvelope:
    """Search discoverable (searchable) organizations to join."""
    tenant_id = _require_tenant(tenant_id)
    if limit <= 0 or limit > 100:
        limit = 20
    infos = await service.search_searchable_organizations(
        tenant_id=tenant_id, query=q, limit=limit
    )
    previews = []
    for info in infos:
        members = await service.list_tenant_members(org_id=info.id)
        previews.append(
            org_preview_to_contract(
                info,
                member_count=len(members),
                is_already_member=await _is_member(service, info.id, tenant_id),
            )
        )
    return search_organizations_envelope(previews, total=len(previews))


@router.get("/preview/{code}", response_model=OrganizationPreviewEnvelope)
async def preview_by_invite_code(
    _auth: AuthDep,
    _role: RoleViewerDep,
    code: str,
    service: OrganizationServiceDep,
    tenant_id: _PrincipalTenant,
) -> OrganizationPreviewEnvelope:
    """Preview an organization by invite code without joining."""
    tenant_id = _require_tenant(tenant_id)
    info = await service.get_organization_by_invite_code(invite_code=code)
    members = await service.list_tenant_members(org_id=info.id)
    return org_preview_envelope(
        info,
        member_count=len(members),
        is_already_member=await _is_member(service, info.id, tenant_id),
    )


@router.post("/join", response_model=OrganizationEnvelope)
async def join_by_invite_code(
    _auth: AuthDep,
    _role: RoleAdminDep,
    body: JoinOrganizationByCodeRequest,
    service: OrganizationServiceDep,
    tenant_id: _PrincipalTenant,
    user_id: _PrincipalUser,
) -> OrganizationEnvelope:
    """Join an organization by invite code as a viewer."""
    tenant_id = _require_tenant(tenant_id)
    caller = _require_user(user_id)
    info = await service.join_by_invite_code(
        invite_code=body.invite_code,
        user_id=caller,
        tenant_id=tenant_id,
    )
    return await _org_envelope(
        service=service, info=info, tenant_id=tenant_id, user_id=caller
    )


@router.post("/join-request", response_model=JoinRequestEnvelope)
async def submit_join_request(
    _auth: AuthDep,
    _role: RoleAdminDep,
    body: JoinRequestRequest,
    service: OrganizationServiceDep,
    tenant_id: _PrincipalTenant,
    user_id: _PrincipalUser,
) -> JoinRequestEnvelope:
    """Submit a join request for organizations that require approval."""
    tenant_id = _require_tenant(tenant_id)
    caller = _require_user(user_id)
    if body.role is not None and body.role != "":
        _require_valid_role(body.role)
    info = await service.get_organization_by_invite_code(invite_code=body.invite_code)
    if not info.require_approval:
        raise ValidationError(
            code="organization.join_not_required",
            message="This organization does not require approval. Use the join endpoint instead.",
        )
    if await _is_member(service, info.id, tenant_id):
        raise ValidationError(
            code="organization.already_member",
            message="Your workspace is already a member of this organization",
        )
    request = await service.submit_join_request(
        org_id=info.id,
        user_id=caller,
        tenant_id=tenant_id,
        message=body.message or "",
        requested_role=body.role or "",
    )
    return join_request_envelope(request)


@router.post("/join-by-id", response_model=OrganizationEnvelope)
async def join_by_organization_id(
    _auth: AuthDep,
    _role: RoleAdminDep,
    body: JoinRequestByIDRequest,
    service: OrganizationServiceDep,
    tenant_id: _PrincipalTenant,
    user_id: _PrincipalUser,
) -> OrganizationEnvelope:
    """Join a searchable organization by id (no invite code required)."""
    tenant_id = _require_tenant(tenant_id)
    caller = _require_user(user_id)
    if body.role is not None and body.role != "":
        _require_valid_role(body.role)
    info = await service.join_by_organization_id(
        org_id=body.organization_id,
        user_id=caller,
        tenant_id=tenant_id,
        message=body.message or "",
        requested_role=body.role or "",
    )
    return await _org_envelope(
        service=service, info=info, tenant_id=tenant_id, user_id=caller
    )


@router.get("/{id}", response_model=OrganizationEnvelope)
async def get_organization(
    _auth: AuthDep,
    _role: RoleViewerDep,
    id: str,
    service: OrganizationServiceDep,
    tenant_id: _PrincipalTenant,
    user_id: _PrincipalUser,
) -> OrganizationEnvelope:
    """Return one organization, gated by membership or discoverability."""
    tenant_id = _require_tenant(tenant_id)
    await _member_visible_org(service, id, tenant_id)
    info = await service.get_organization(id=id)
    return await _org_envelope(
        service=service, info=info, tenant_id=tenant_id, user_id=user_id
    )


@router.put("/{id}", response_model=OrganizationEnvelope)
async def update_organization(
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    body: UpdateOrganizationRequest,
    service: OrganizationServiceDep,
    tenant_id: _PrincipalTenant,
    user_id: _PrincipalUser,
) -> OrganizationEnvelope:
    """Update an organization's mutable fields; admin in the org only."""
    tenant_id = _require_tenant(tenant_id)
    caller = _require_user(user_id)
    info = await service.update_organization(
        id=id,
        operator_user_id=caller,
        operator_tenant_id=tenant_id,
        name=body.name,
        description=body.description,
        avatar=body.avatar,
        require_approval=body.require_approval,
        searchable=body.searchable,
        invite_code_validity_days=body.invite_code_validity_days,
        member_limit=body.member_limit,
    )
    return await _org_envelope(
        service=service, info=info, tenant_id=tenant_id, user_id=caller
    )


@router.delete("/{id}", response_model=SimpleAckResponse)
async def delete_organization(
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    service: OrganizationServiceDep,
    tenant_id: _PrincipalTenant,
    user_id: _PrincipalUser,
) -> SimpleAckResponse:
    """Soft-delete an organization; only the owning workspace may act."""
    tenant_id = _require_tenant(tenant_id)
    caller = _require_user(user_id)
    await service.delete_organization(
        id=id,
        operator_user_id=caller,
        operator_tenant_id=tenant_id,
    )
    return SimpleAckResponse(success=True, message=_DELETE_MESSAGE)


# ── Invite code & leave ──────────────────────────────────────────────


@router.post("/{id}/invite-code", response_model=InviteCodeEnvelope)
async def generate_invite_code(
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    service: OrganizationServiceDep,
    tenant_id: _PrincipalTenant,
    user_id: _PrincipalUser,
) -> InviteCodeEnvelope:
    """Rotate the organization's invite code; admin in the org only."""
    tenant_id = _require_tenant(tenant_id)
    caller = _require_user(user_id)
    code = await service.generate_invite_code(
        org_id=id,
        operator_user_id=caller,
        operator_tenant_id=tenant_id,
    )
    return invite_code_envelope(code)


@router.post("/{id}/leave", response_model=SimpleAckResponse)
async def leave_organization(
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    service: OrganizationServiceDep,
    tenant_id: _PrincipalTenant,
    user_id: _PrincipalUser,
) -> SimpleAckResponse:
    """Leave an organization (self-removal); the owner workspace cannot leave."""
    tenant_id = _require_tenant(tenant_id)
    caller = _require_user(user_id)
    await service.remove_tenant_member(
        org_id=id,
        member_tenant_id=tenant_id,
        operator_user_id=caller,
        operator_tenant_id=tenant_id,
    )
    return SimpleAckResponse(success=True, message=_LEAVE_MESSAGE)


@router.post("/{id}/request-upgrade", response_model=JoinRequestEnvelope)
async def request_role_upgrade(
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    body: RequestRoleUpgradeRequest,
    service: OrganizationServiceDep,
    tenant_id: _PrincipalTenant,
    user_id: _PrincipalUser,
) -> JoinRequestEnvelope:
    """Submit a role-upgrade request for the caller's workspace."""
    tenant_id = _require_tenant(tenant_id)
    caller = _require_user(user_id)
    _require_valid_role(body.requested_role)
    request = await service.request_role_upgrade(
        org_id=id,
        user_id=caller,
        tenant_id=tenant_id,
        requested_role=body.requested_role,
        message=body.message or "",
    )
    return join_request_envelope(request)


# ── Membership ───────────────────────────────────────────────────────


@router.get("/{id}/members", response_model=MemberListEnvelope)
async def list_members(
    _auth: AuthDep,
    _role: RoleViewerDep,
    id: str,
    service: OrganizationServiceDep,
    tenant_id: _PrincipalTenant,
) -> MemberListEnvelope:
    """List the organization's member workspaces; members only."""
    tenant_id = _require_tenant(tenant_id)
    if not await _is_member(service, id, tenant_id):
        raise PermissionDeniedError(
            code="organization.not_member",
            message="Your workspace is not a member of this organization",
        )
    members = await service.list_tenant_members(org_id=id)
    return member_list_envelope(members)


@router.put("/{id}/members/{tenant_id}", response_model=SimpleAckResponse)
async def update_member_role(
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    tenant_id: int,
    body: UpdateMemberRoleRequest,
    service: OrganizationServiceDep,
    caller_tenant_id: _PrincipalTenant,
    user_id: _PrincipalUser,
) -> SimpleAckResponse:
    """Change a member workspace's role; admin in the org only."""
    caller_tenant_id = _require_tenant(caller_tenant_id)
    caller = _require_user(user_id)
    await service.update_tenant_member_role(
        org_id=id,
        member_tenant_id=tenant_id,
        role=body.role,
        operator_user_id=caller,
        operator_tenant_id=caller_tenant_id,
    )
    return SimpleAckResponse(success=True, message=_MEMBER_ROLE_UPDATED_MESSAGE)


@router.delete("/{id}/members/{tenant_id}", response_model=SimpleAckResponse)
async def remove_member(
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    tenant_id: int,
    service: OrganizationServiceDep,
    caller_tenant_id: _PrincipalTenant,
    user_id: _PrincipalUser,
) -> SimpleAckResponse:
    """Remove a member workspace; admin in the org only."""
    caller_tenant_id = _require_tenant(caller_tenant_id)
    caller = _require_user(user_id)
    await service.remove_tenant_member(
        org_id=id,
        member_tenant_id=tenant_id,
        operator_user_id=caller,
        operator_tenant_id=caller_tenant_id,
    )
    return SimpleAckResponse(success=True, message=_MEMBER_REMOVED_MESSAGE)


# ── Join requests (admin) ────────────────────────────────────────────


@router.get("/{id}/join-requests", response_model=JoinRequestListEnvelope)
async def list_join_requests(
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    service: OrganizationServiceDep,
    tenant_id: _PrincipalTenant,
) -> JoinRequestListEnvelope:
    """List pending join requests; admin in the org only."""
    tenant_id = _require_tenant(tenant_id)
    await _require_admin(service, id, tenant_id)
    requests = await service.list_join_requests(org_id=id, status="pending")
    return join_request_list_envelope(requests)


@router.put(
    "/{id}/join-requests/{request_id}/review",
    response_model=SimpleAckResponse,
)
async def review_join_request(
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    request_id: str,
    body: ReviewJoinRequestRequest,
    service: OrganizationServiceDep,
    tenant_id: _PrincipalTenant,
    user_id: _PrincipalUser,
) -> SimpleAckResponse:
    """Approve or reject a join request; admin in the org only."""
    tenant_id = _require_tenant(tenant_id)
    caller = _require_user(user_id)
    await _require_admin(service, id, tenant_id)
    assign_role = None
    if body.role is not None and body.role != "":
        _require_valid_role(body.role)
        assign_role = body.role
    await service.review_join_request(
        org_id=id,
        request_id=request_id,
        approved=body.approved,
        reviewer_user_id=caller,
        reviewer_tenant_id=tenant_id,
        message=body.message or "",
        assign_role=assign_role,
    )
    return SimpleAckResponse(success=True, message=_REVIEW_MESSAGE)


# ── Direct invites ───────────────────────────────────────────────────


async def _search_tenant_candidates(
    *,
    service: OrganizationService,
    tenant_service: TenantServiceDep,
    org_id: str,
    tenant_id: int,
    query: str,
    limit: int,
) -> list[TenantInviteCandidate]:
    """Resolve candidate workspaces for inviting, excluding members.

    Matches by workspace name only (the membership unit); deduplicates by
    workspace id preserving search order; drops rows with no resolvable
    name and caps at ``limit``.
    """
    if limit <= 0 or limit > 50:
        limit = 10
    existing_members = await service.list_tenant_members(org_id=org_id)
    existing_ids = {m.tenant_id for m in existing_members}
    tenants, _ = await tenant_service.search_tenants(
        keyword=query, page=0, page_size=limit * 2
    )
    ordered: list[int] = []
    present: set[int] = set()
    for t in tenants:
        if t.id == 0 or t.id in existing_ids or t.id in present:
            continue
        present.add(t.id)
        ordered.append(t.id)
    by_id = await tenant_service.get_tenants(ordered)
    candidates: list[TenantInviteCandidate] = []
    for tid in ordered:
        info = by_id.get(tid)
        if info is None or not info.name:
            continue
        candidates.append(TenantInviteCandidate(tenant_id=tid, tenant_name=info.name))
        if len(candidates) >= limit:
            break
    return candidates


@router.get("/{id}/search-tenants", response_model=TenantInviteListEnvelope)
async def search_tenants_for_invite(
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    service: OrganizationServiceDep,
    tenant_service: TenantServiceDep,
    tenant_id: _PrincipalTenant,
    q: str = Query(default="", description="搜索关键词（空间名）"),
    limit: int = Query(default=10, description="返回数量限制"),
) -> TenantInviteListEnvelope:
    """Search candidate workspaces for inviting; admin in the org only."""
    tenant_id = _require_tenant(tenant_id)
    await _require_admin(service, id, tenant_id)
    if not q:
        return tenant_invite_list_envelope([])
    candidates = await _search_tenant_candidates(
        service=service,
        tenant_service=tenant_service,
        org_id=id,
        tenant_id=tenant_id,
        query=q,
        limit=limit,
    )
    return tenant_invite_list_envelope(candidates)


@router.get("/{id}/search-users", response_model=TenantInviteListEnvelope)
async def search_users_for_invite(
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    service: OrganizationServiceDep,
    tenant_service: TenantServiceDep,
    tenant_id: _PrincipalTenant,
    q: str = Query(default="", description="搜索关键词（空间名）"),
    limit: int = Query(default=10, description="返回数量限制"),
) -> TenantInviteListEnvelope:
    """Deprecated alias for ``search_tenants_for_invite``.

    Older frontends receive the workspace-grouped shape (not one row per
    user); kept for one release.
    """
    return await search_tenants_for_invite(
        _auth=_auth,
        _role=_role,
        id=id,
        service=service,
        tenant_service=tenant_service,
        tenant_id=tenant_id,
        q=q,
        limit=limit,
    )


@router.post("/{id}/invite", response_model=SimpleAckResponse)
async def invite_member(
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    body: InviteMemberRequest,
    service: OrganizationServiceDep,
    tenant_service: TenantServiceDep,
    tenant_id: _PrincipalTenant,
) -> SimpleAckResponse:
    """Directly enrol a workspace as an organization member; admin only."""
    tenant_id = _require_tenant(tenant_id)
    await _require_admin(service, id, tenant_id)
    _require_valid_role(body.role)
    target_tenant_id = body.tenant_id
    representative_user_id = body.representative_user_id
    if target_tenant_id != 0:
        try:
            await tenant_service.get_tenant(target_tenant_id)
        except NotFoundError:
            raise NotFoundError(
                code="organization.tenant_not_found",
                message="Workspace not found",
            )
        if not representative_user_id:
            representative_user_id = body.user_id
    elif body.user_id:
        # The legacy user-only path resolves the user's workspace through a
        # user service, which is not wired into this layer yet.
        raise ValidationError(
            code="organization.invite_requires_tenant",
            message="Provide tenant_id to invite a workspace",
        )
    else:
        raise ValidationError(
            code="organization.invite_required",
            message="Either tenant_id or user_id is required",
        )
    if await _is_member(service, id, target_tenant_id):
        raise ValidationError(
            code="organization.already_member",
            message="Workspace is already a member of this organization",
        )
    await service.add_tenant_member(
        org_id=id,
        tenant_id=target_tenant_id,
        representative_user_id=representative_user_id,
        role=body.role,
    )
    return SimpleAckResponse(success=True, message=_MEMBER_ADDED_MESSAGE)


__all__ = ["router"]
