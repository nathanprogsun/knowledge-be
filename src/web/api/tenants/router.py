"""Tenant HTTP endpoints - workspace CRUD, listing, search, API keys, KV."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from src.app_context.request_context import get_tenant_id, get_user_id
from src.common.exception import ValidationError
from src.common.json import JsonObject
from src.core.auth.types import UserPreferences
from src.core.contracts.tenants import (
    APIPrincipalConfig,
    CreateAPIKeyRequest,
    CreateTenantRequest,
    TenantAPIKey,
    UpdateAPIPrincipalConfigRequest,
    UpdateTenantRequest,
)
from src.core.tenants.types import TenantAPIKeyInfo
from src.web.api.tenants.views import (
    AddMemberBody,
    CreateInvitationBody,
    CreateInviteLinkBody,
    DeleteTenantResponse,
    InvitationEnvelope,
    InvitationListEnvelope,
    MemberEnvelope,
    MemberListEnvelope,
    TenantEnvelope,
    TenantListEnvelope,
    UpdateMemberRoleBody,
    clamp_page,
    invitation_envelope,
    invitation_list_envelope,
    invite_url_for_token,
    member_envelope,
    member_list_envelope,
    require_caller_user_id,
    require_tenant_invitation,
    resolve_registered_user,
    tenant_envelope,
    tenant_list_envelope,
)
from src.web.deps import (
    AuthDep,
    AuthServiceDep,
    CrossTenantDep,
    CurrentUserContextDep,
    PathTenantMatchDep,
    RoleAdminDep,
    RoleOwnerDep,
    RoleViewerDep,
    TenantAPIKeyServiceDep,
    TenantKVServiceDep,
    TenantMemberServiceDep,
    TenantServiceDep,
)
from src.web.deps.rbac import require_role_dep
from src.web.deps.tenants import TenantInvitationServiceDep

router = APIRouter(prefix="/tenants", tags=["tenants"])


class TenantAPIKeyWithToken(TenantAPIKey):
    """``TenantAPIKey`` plus the one-time plaintext ``api_key``."""

    model_config = ConfigDict(frozen=True)

    api_key: str = Field(default="")


class TenantAPIKeyCreateEnvelope(BaseModel):
    """``{"success": true, "data": {..., "api_key": "<plaintext>"}}``.

    The plaintext token is returned only by the create endpoint (mirrors
    Go's ``tenantWithAPIKey``); every later read carries the key without
    the credential.
    """

    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: TenantAPIKeyWithToken


# Paging values are clamped rather than rejected: a page below 1 becomes
# 1, a page size outside [1, 100] becomes the default (20) or the cap (100).
_DEFAULT_PAGE = 1
_DEFAULT_PAGE_SIZE = 20
_MAX_PAGE_SIZE = 100


@router.post("", response_model=TenantEnvelope, status_code=201)
async def create_tenant(
    _auth: AuthDep,
    body: CreateTenantRequest,
    tenant_service: TenantServiceDep,
    member_service: TenantMemberServiceDep,
    auth_service: AuthServiceDep,
) -> TenantEnvelope:
    """Create a workspace; the status is assigned server-side.

    The creator becomes an Owner of the new workspace (mirrors Go's
    ``memberService.EnsureOwner`` after create), so the workspace is
    reachable through the membership-scoped endpoints immediately.

    The creator's ``preferences.last_active_tenant_id`` is set to the
    new workspace id so the next JWT-mint lands here (mirrors Go's
    home-or-first-membership resolution in
    ``userService.resolveLoginTenantID``). Without this the creator
    would be locked out of their own tenant-scoped endpoints until
    the auth middleware's fallback fires.
    """
    info = await tenant_service.create_tenant(
        name=body.name,
        description=body.description,
        business=body.business or "",
        retriever_engines=_engines_payload(body),
        storage_quota=body.storage_quota,
    )
    user_id = get_user_id()
    if user_id is not None:
        await member_service.ensure_owner(user_id=user_id, tenant_id=info.id)
        # Persist last_active_tenant_id so the next login mints a JWT
        # carrying the new workspace as the active tenant. The
        # auth-middleware fallback covers existing users whose
        # preference was never set; this forward-write keeps new
        # creators from having to rely on it. The PATCH-merge lives in
        # the auth service — web never touches UserRepository.
        await auth_service.update_my_preferences(
            user_id=user_id,
            patch=UserPreferences(last_active_tenant_id=info.id),
        )
    return tenant_envelope(info)


@router.get("/all", response_model=TenantListEnvelope)
async def list_all_tenants(
    _auth: AuthDep,
    _cross: CrossTenantDep,
    tenant_service: TenantServiceDep,
) -> TenantListEnvelope:
    """List every workspace, newest first."""
    return tenant_list_envelope(await tenant_service.list_tenants())


@router.get("/search", response_model=TenantListEnvelope)
async def search_tenants(
    _auth: AuthDep,
    _cross: CrossTenantDep,
    tenant_service: TenantServiceDep,
    keyword: str | None = Query(default=None),
    tenant_id: int | None = Query(default=None),
    page: int = Query(default=_DEFAULT_PAGE),
    page_size: int = Query(default=_DEFAULT_PAGE_SIZE),
) -> TenantListEnvelope:
    """Search workspaces by id and/or keyword, with pagination."""
    page = page if page >= 1 else _DEFAULT_PAGE
    page_size = page_size if page_size >= 1 else _DEFAULT_PAGE_SIZE
    page_size = min(page_size, _MAX_PAGE_SIZE)
    infos, total = await tenant_service.search_tenants(
        keyword=keyword,
        tenant_id=tenant_id,
        page=page,
        page_size=page_size,
    )
    return tenant_list_envelope(infos, total=total, page=page, page_size=page_size)


@router.get("", response_model=TenantListEnvelope)
async def list_my_tenants(
    request: Request,
    _ctx: CurrentUserContextDep,
    tenant_service: TenantServiceDep,
    member_service: TenantMemberServiceDep,
) -> TenantListEnvelope:
    """Return every workspace visible to the authenticated user.

    API-key principals have no associated ``user_id`` and therefore no
    memberships to enumerate. Mirrors Go's behaviour of denying
    user-scoped reads for non-user principals; the workspace list is
    addressable through the per-workspace routes when the key's
    ``tenant_id`` is known.
    """
    if getattr(request.state, "api_key_scope", None) is not None:
        return tenant_list_envelope([])
    user_id = get_user_id()
    if user_id is None:
        raise ValidationError(
            code="auth.missing_user_context",
            message="No authenticated user in request context",
        )
    memberships = await member_service.list_by_user(user_id)
    tenant_ids = [m.tenant_id for m in memberships]
    by_id = await tenant_service.get_tenants(tenant_ids)
    return tenant_list_envelope([by_id[tid] for tid in tenant_ids if tid in by_id])


# ── API keys (Owner) ──────────────────────────────────────────────────


@router.get("/{tenant_id}/api-keys", response_model=list[TenantAPIKey])
async def list_api_keys(
    _auth: AuthDep,
    _owner: RoleOwnerDep,
    _match: PathTenantMatchDep,
    tenant_id: int,
    api_key_service: TenantAPIKeyServiceDep,
) -> list[TenantAPIKey]:
    """List the workspace's API keys (drops credential columns)."""
    keys = await api_key_service.list_api_keys(tenant_id)
    return [_api_key_info_to_contract(k) for k in keys]


@router.post(
    "/{tenant_id}/api-keys",
    response_model=TenantAPIKeyCreateEnvelope,
    status_code=201,
)
async def create_api_key(
    _auth: AuthDep,
    _owner: RoleOwnerDep,
    _match: PathTenantMatchDep,
    tenant_id: int,
    body: CreateAPIKeyRequest,
    api_key_service: TenantAPIKeyServiceDep,
) -> TenantAPIKeyCreateEnvelope:
    """Mint a workspace-scoped API key.

    The plaintext token is embedded in the response once (mirrors Go's
    ``tenantWithAPIKey``); later reads return the key without it.
    """
    result = await api_key_service.create_api_key(
        name=body.name,
        tenant_id=tenant_id,
        full_access=body.full_access,
        knowledge_base_ids=body.knowledge_base_ids,
        capabilities=body.capabilities,
    )
    return TenantAPIKeyCreateEnvelope(
        data=TenantAPIKeyWithToken(
            **_api_key_info_to_contract(result.key).model_dump(),
            api_key=result.token,
        )
    )


@router.delete("/{tenant_id}/api-keys/{key_id}", response_model=DeleteTenantResponse)
async def delete_api_key(
    _auth: AuthDep,
    _owner: RoleOwnerDep,
    _match: PathTenantMatchDep,
    tenant_id: int,
    key_id: int,
    api_key_service: TenantAPIKeyServiceDep,
) -> DeleteTenantResponse:
    """Revoke a workspace API key."""
    await api_key_service.revoke_api_key(key_id, tenant_id=tenant_id)
    return DeleteTenantResponse(success=True, message="API key revoked")


# ── API principal config (Owner) ──────────────────────────────────────


@router.get("/{tenant_id}/api-principal-config", response_model=APIPrincipalConfig)
async def get_api_principal_config(
    _auth: AuthDep,
    _owner: RoleOwnerDep,
    _match: PathTenantMatchDep,
    tenant_id: int,
    tenant_service: TenantServiceDep,
) -> APIPrincipalConfig:
    """Return the workspace's API-principal identity config."""
    config = await tenant_service.get_api_principal_config(tenant_id)
    return _principal_to_contract(config)


@router.put("/{tenant_id}/api-principal-config", response_model=APIPrincipalConfig)
async def update_api_principal_config(
    _auth: AuthDep,
    _owner: RoleOwnerDep,
    _match: PathTenantMatchDep,
    tenant_id: int,
    body: UpdateAPIPrincipalConfigRequest,
    tenant_service: TenantServiceDep,
) -> APIPrincipalConfig:
    """Update the workspace's API-principal identity config."""
    stored = await tenant_service.update_api_principal_config(
        tenant_id,
        config=_principal_update_payload(body),
    )
    return _principal_to_contract(stored)


# ── Tenant KV config ──────────────────────────────────────────────────

# Supported KV keys (Go ``GetTenantKV`` / ``UpdateTenantKV``). The three
# integration-config keys carry secrets and require admin access to read.
_KV_SUPPORTED_KEYS: frozenset[str] = frozenset(
    {
        "web-search-config",
        "prompt-templates",
        "parser-engine-config",
        "storage-engine-config",
        "chat-history-config",
        "retrieval-config",
    }
)
_KV_INTEGRATION_KEYS: frozenset[str] = frozenset(
    {"web-search-config", "parser-engine-config", "storage-engine-config"}
)


def _require_supported_kv_key(key: str) -> None:
    """Reject keys outside the Go-supported set (400 ``unsupported key``)."""
    if key not in _KV_SUPPORTED_KEYS:
        raise ValidationError(
            code="tenant_kv.unsupported_key",
            message=f"unsupported key: {key}",
        )


@router.get("/kv/{key}", response_model=JsonObject)
async def get_tenant_kv(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    request: Request,
    key: str,
    kv_service: TenantKVServiceDep,
) -> JsonObject:
    """Return the current workspace's KV config for ``key``."""
    _require_supported_kv_key(key)
    if key in _KV_INTEGRATION_KEYS:
        # Mirrors Go: integration configs carry secrets and require admin.
        await require_role_dep(request, "admin")
    tenant_id = _require_context_tenant()
    value = await kv_service.get(tenant_id=tenant_id, key=key)
    if value is None:
        # Go returns the typed zero config (``&RetrievalConfig{}`` etc.)
        # when the key is unset — an empty object, not a 404.
        return {}
    if not isinstance(value, dict):
        raise ValidationError(
            code="tenant_kv.bad_shape",
            message=f"KV key '{key}' is not an object",
        )
    return value


@router.put("/kv/{key}", response_model=JsonObject)
async def put_tenant_kv(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    key: str,
    body: JsonObject,
    kv_service: TenantKVServiceDep,
) -> JsonObject:
    """Set the current workspace's KV config for ``key``."""
    _require_supported_kv_key(key)
    tenant_id = _require_context_tenant()
    stored = await kv_service.set(tenant_id=tenant_id, key=key, value=body)
    if not isinstance(stored, dict):
        raise ValidationError(
            code="tenant_kv.bad_shape",
            message=f"KV key '{key}' did not round-trip as an object",
        )
    return stored


@router.get("/{tenant_id}", response_model=TenantEnvelope)
async def get_tenant(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    _match: PathTenantMatchDep,
    tenant_id: int,
    tenant_service: TenantServiceDep,
) -> TenantEnvelope:
    """Return one workspace; an invalid id yields a path validation error."""
    return tenant_envelope(await tenant_service.get_tenant(tenant_id))


@router.put("/{tenant_id}", response_model=TenantEnvelope)
async def update_tenant(
    _auth: AuthDep,
    _owner: RoleOwnerDep,
    _match: PathTenantMatchDep,
    tenant_id: int,
    body: UpdateTenantRequest,
    tenant_service: TenantServiceDep,
) -> TenantEnvelope:
    """Update a workspace's name and/or description.

    Only those two columns are mutable through this endpoint, so a
    caller cannot escalate by sending a larger body.
    """
    info = await tenant_service.update_tenant(
        tenant_id,
        name=body.name,
        description=body.description.strip() if body.description is not None else None,
    )
    return tenant_envelope(info)


@router.delete("/{tenant_id}", response_model=DeleteTenantResponse)
async def delete_tenant(
    _auth: AuthDep,
    _owner: RoleOwnerDep,
    _match: PathTenantMatchDep,
    tenant_id: int,
    tenant_service: TenantServiceDep,
) -> DeleteTenantResponse:
    """Soft-delete a workspace; idempotent for unknown ids."""
    await tenant_service.delete_tenant(tenant_id)
    return DeleteTenantResponse(success=True, message="Workspace deleted successfully")


@router.get("/{tenant_id}/members", response_model=MemberListEnvelope)
async def list_members(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    _match: PathTenantMatchDep,
    tenant_id: int,
    member_service: TenantMemberServiceDep,
    q: str | None = Query(default=None),
    page: int = Query(default=_DEFAULT_PAGE),
    page_size: int = Query(default=_DEFAULT_PAGE_SIZE),
) -> MemberListEnvelope:
    """List workspace members. ``q`` matches email or username when joined."""
    page, page_size = clamp_page(page, page_size)
    query = q.strip() if q is not None and q.strip() else None
    rows, total = await member_service.list_members_page(
        tenant_id,
        query=query,
        page=page,
        page_size=page_size,
    )
    return member_list_envelope(rows, total=total, page=page, page_size=page_size)


@router.post("/{tenant_id}/members", response_model=MemberEnvelope, status_code=201)
async def add_member(
    _auth: AuthDep,
    _owner: RoleOwnerDep,
    _match: PathTenantMatchDep,
    tenant_id: int,
    body: AddMemberBody,
    member_service: TenantMemberServiceDep,
    auth_service: AuthServiceDep,
) -> MemberEnvelope:
    """Add a registered user by email. Unknown emails are 404, not an invite."""
    user = await resolve_registered_user(auth_service, body.email)
    info = await member_service.add_member(
        user_id=user.user_id,
        tenant_id=tenant_id,
        role=body.role,
        invited_by=require_caller_user_id(),
    )
    return member_envelope(
        info,
        email=user.email,
        username=user.username,
        avatar=user.avatar,
    )


@router.put("/{tenant_id}/members/{user_id}", response_model=DeleteTenantResponse)
async def update_member_role(
    _auth: AuthDep,
    _owner: RoleOwnerDep,
    _match: PathTenantMatchDep,
    tenant_id: int,
    user_id: str,
    body: UpdateMemberRoleBody,
    member_service: TenantMemberServiceDep,
) -> DeleteTenantResponse:
    """Change a member's role. Demoting the last owner is a conflict."""
    await member_service.update_role(user_id=user_id, tenant_id=tenant_id, role=body.role)
    return DeleteTenantResponse(success=True, message="Member role updated")


@router.delete("/{tenant_id}/members/{user_id}", response_model=DeleteTenantResponse)
async def delete_member(
    _auth: AuthDep,
    _owner: RoleOwnerDep,
    _match: PathTenantMatchDep,
    tenant_id: int,
    user_id: str,
    member_service: TenantMemberServiceDep,
) -> DeleteTenantResponse:
    """Remove a member. Removing the last owner is a conflict."""
    await member_service.remove_member(user_id=user_id, tenant_id=tenant_id)
    return DeleteTenantResponse(success=True, message="Member removed")


@router.post("/{tenant_id}/leave", response_model=DeleteTenantResponse)
async def leave_tenant(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    _match: PathTenantMatchDep,
    tenant_id: int,
    member_service: TenantMemberServiceDep,
) -> DeleteTenantResponse:
    """Leave the workspace as the caller. Last owner is a typed conflict."""
    await member_service.remove_member(
        user_id=require_caller_user_id(),
        tenant_id=tenant_id,
    )
    return DeleteTenantResponse(success=True, message="Left workspace")


@router.get("/{tenant_id}/invitations", response_model=InvitationListEnvelope)
async def list_invitations(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    _match: PathTenantMatchDep,
    tenant_id: int,
    invitation_service: TenantInvitationServiceDep,
    include_terminal: bool = Query(default=False),
    page: int = Query(default=_DEFAULT_PAGE),
    page_size: int = Query(default=_DEFAULT_PAGE_SIZE),
) -> InvitationListEnvelope:
    """List this workspace's invitations. Defaults to pending rows."""
    page, page_size = clamp_page(page, page_size)
    rows, total = await invitation_service.list_tenant_invitations_page(
        tenant_id,
        include_terminal=include_terminal,
        page=page,
        page_size=page_size,
    )
    return invitation_list_envelope(rows, total=total, page=page, page_size=page_size)


@router.post(
    "/{tenant_id}/invitations",
    response_model=InvitationEnvelope,
    status_code=201,
)
async def create_invitation(
    _auth: AuthDep,
    _owner: RoleOwnerDep,
    _match: PathTenantMatchDep,
    tenant_id: int,
    body: CreateInvitationBody,
    invitation_service: TenantInvitationServiceDep,
    auth_service: AuthServiceDep,
) -> InvitationEnvelope:
    """Invite a registered user. The invitee sees the row on ``/me/invitations``."""
    user = await resolve_registered_user(auth_service, body.email)
    info = await invitation_service.create_invitation(
        tenant_id=tenant_id,
        invitee_user_id=user.user_id,
        role=body.role,
        invited_by=require_caller_user_id(),
        message=body.message,
    )
    return invitation_envelope(info, invitee_email=user.email, invitee_name=user.username)


@router.delete("/{tenant_id}/invitations/{inv_id}", response_model=DeleteTenantResponse)
async def revoke_invitation(
    _auth: AuthDep,
    _owner: RoleOwnerDep,
    _match: PathTenantMatchDep,
    tenant_id: int,
    inv_id: int,
    invitation_service: TenantInvitationServiceDep,
) -> DeleteTenantResponse:
    """Revoke a pending invitation. Other workspaces' rows look missing."""
    await require_tenant_invitation(
        invitation_service,
        tenant_id=tenant_id,
        inv_id=inv_id,
    )
    await invitation_service.revoke(inv_id)
    return DeleteTenantResponse(success=True, message="Invitation revoked")


@router.post(
    "/{tenant_id}/invite-links",
    response_model=InvitationEnvelope,
    status_code=201,
)
async def create_invite_link(
    _auth: AuthDep,
    _owner: RoleOwnerDep,
    _match: PathTenantMatchDep,
    tenant_id: int,
    body: CreateInviteLinkBody,
    invitation_service: TenantInvitationServiceDep,
) -> InvitationEnvelope:
    """Issue a reusable share link and return its copy URL."""
    info, token = await invitation_service.create_share_link(
        tenant_id=tenant_id,
        role=body.role,
        invited_by=require_caller_user_id(),
        message=body.message,
    )
    return invitation_envelope(info, invite_url=invite_url_for_token(token))


def _engines_payload(body: CreateTenantRequest) -> JsonObject | None:
    """Render the request's retriever engines as the stored JSON shape."""
    if body.retriever_engines is None:
        return None
    return body.retriever_engines.model_dump(mode="json")


def _api_key_info_to_contract(info: TenantAPIKeyInfo) -> TenantAPIKey:
    """Project the service DTO to the wire ``TenantAPIKey`` (no token)."""
    return TenantAPIKey(
        id=info.id,
        scope_type=info.scope_type,
        name=info.name,
        full_access=info.full_access,
        knowledge_base_ids=info.knowledge_base_ids,
        capabilities=info.capabilities,
        last_used_at=info.last_used_at,
        expires_at=info.expires_at,
        created_at=info.created_at,
    )


def _principal_to_contract(config: JsonObject | None) -> APIPrincipalConfig:
    """Render a stored principal config object as the wire model."""
    if config is None:
        return APIPrincipalConfig(
            mode="disabled",
            has_hmac_secret=False,
        )
    mode = config.get("mode", "disabled")
    if not isinstance(mode, str):
        mode = "disabled"
    direct = config.get("direct_header_name")
    signed = config.get("signed_token_header_name")
    require_direct = config.get("require_direct_header", False)
    # The stored secret is never disclosed; only its presence is
    # reported (Go GET semantics).
    raw_secret = config.get("hmac_secret")
    has_secret = isinstance(raw_secret, str) and bool(raw_secret)
    return APIPrincipalConfig(
        mode=mode,
        direct_header_name=direct if isinstance(direct, str) else None,
        signed_token_header_name=signed if isinstance(signed, str) else None,
        require_direct_header=require_direct if isinstance(require_direct, bool) else False,
        has_hmac_secret=has_secret,
    )


def _principal_update_payload(body: UpdateAPIPrincipalConfigRequest) -> JsonObject:
    """Build the stored principal config object from an update request.

    ``hmac_secret`` is carried through verbatim; the service resolves
    the ``"***"`` redaction placeholder against the stored secret (Go
    ``apiPrincipalSecretRedacted`` semantics).
    """
    return {
        "mode": body.mode,
        "direct_header_name": body.direct_header_name,
        "signed_token_header_name": body.signed_token_header_name,
        "require_direct_header": body.require_direct_header,
        "hmac_secret": body.hmac_secret,
    }


def _require_context_tenant() -> int:
    """Return the current tenant id from request context, or raise."""
    raw = get_tenant_id()
    if raw is None:
        raise ValidationError(
            code="tenant.context_missing",
            message="No active workspace in request context",
        )
    try:
        return int(raw)
    except ValueError:
        raise ValidationError(
            code="tenant.context_invalid",
            message="Active workspace id is invalid",
        ) from None


__all__ = ["router"]
