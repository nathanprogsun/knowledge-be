"""HTTP endpoints for the MCP service domain.

Maps the 13 endpoints from ``routes_infra.go``:

    POST   /mcp-services                       Admin+   create
    GET    /mcp-services                       Viewer+  list
    GET    /mcp-services/{id}                  Viewer+  get
    PUT    /mcp-services/{id}                  Admin+   update
    DELETE /mcp-services/{id}                  Admin+   delete
    POST   /mcp-services/{id}/test             Admin+   connectivity test
    GET    /mcp-services/{id}/tools            Viewer+  discovered tools
    GET    /mcp-services/{id}/resources        Viewer+  discovered resources
    GET    /mcp-services/{id}/tool-approvals   Viewer+  approval overrides
    PUT    /mcp-services/{id}/tool-approvals/{tool_name}
                                              Admin+   set approval
    POST   /mcp-services/{id}/oauth/authorize-url
                                              Viewer+  start OAuth
    GET    /mcp-services/{id}/oauth/status     Viewer+  per-user status
    DELETE /mcp-services/{id}/oauth/token      Viewer+  revoke token

The credentials subresource (``PUT/DELETE /mcp-services/{id}/credentials``
and ``DELETE /mcp-services/{id}/credentials/{field}``) is
intentionally NOT here so the secret write surface has its own
service module + tests.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from src.common.exception import NotFoundError, ValidationError
from src.common.json import JsonObject, JsonValue
from src.web.api.infra.mcp_services.views import (
    CreateMCPServiceRequest,
    DeleteMCPServiceResponse,
    MCPOAuthAuthorizeURLResponse,
    MCPOAuthStatusEnvelope,
    MCPResourceEnvelope,
    MCPTestResultEnvelope,
    MCPToolApprovalListEnvelope,
    MCPToolEnvelope,
    SetMCPToolApprovalRequest,
    UpdateMCPServiceRequest,
    approval_list_envelope,
    oauth_authorize_envelope,
    oauth_status_envelope,
    resource_envelope,
    service_envelope,
    service_list_envelope,
    test_result_envelope,
    tool_envelope,
)
from src.web.deps import AuthDep, RoleAdminDep, RoleViewerDep
from src.web.deps.infra_mcp import (
    MCPServiceDep,
    RequireTenantIdDep,
    RequireUserIdDep,
)

router = APIRouter(prefix="/mcp-services", tags=["mcp-services"])


class OAuthAuthorizeRequestBody(BaseModel):
    """Body for ``POST /mcp-services/{id}/oauth/authorize-url``.

    ``redirect_uri`` is the absolute backend callback URL registered
    with the authorization server; ``frontend_redirect`` is where the
    callback bounces the browser when done (defaults to ``"/"``).
    """

    model_config = ConfigDict(frozen=True)

    redirect_uri: str
    frontend_redirect: str | None = None


# ── Endpoints ────────────────────────────────────────────────────────


@router.post("", response_model=None, status_code=201)
async def create_mcp_service(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    body: CreateMCPServiceRequest,
    mcp_service: MCPServiceDep,
    tenant_id: RequireTenantIdDep,
) -> dict[str, JsonValue]:
    """Register a new MCP service in the active workspace."""
    info = await mcp_service.create_service(
        tenant_id=tenant_id,
        name=body.name,
        transport_type=body.transport_type,
        description=body.description,
        url=body.url,
        headers=body.headers,
        auth_config=_dict_or_none(body.auth_config),
        advanced_config=_dict_or_none(body.advanced_config),
        stdio_config=_dict_or_none(body.stdio_config),
        env_vars=body.env_vars,
        enabled=body.enabled,
    )
    return service_envelope(info)


@router.get("", response_model=None)
async def list_mcp_services(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    mcp_service: MCPServiceDep,
    tenant_id: RequireTenantIdDep,
) -> dict[str, JsonValue]:
    """List the active workspace's MCP services, newest first."""
    return service_list_envelope(await mcp_service.list_services(tenant_id=tenant_id))


@router.get("/{service_id}", response_model=None)
async def get_mcp_service(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    service_id: str,
    mcp_service: MCPServiceDep,
    tenant_id: RequireTenantIdDep,
) -> dict[str, JsonValue]:
    """Fetch one MCP service by id."""
    info = await mcp_service.get_service(tenant_id=tenant_id, id=service_id)
    return service_envelope(info)


@router.put("/{service_id}", response_model=None)
async def update_mcp_service(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    service_id: str,
    body: UpdateMCPServiceRequest,
    mcp_service: MCPServiceDep,
    tenant_id: RequireTenantIdDep,
) -> dict[str, JsonValue]:
    """Patch the supplied columns of a configured service."""
    info = await mcp_service.update_service(
        tenant_id=tenant_id,
        id=service_id,
        name=body.name,
        description=body.description,
        enabled=body.enabled,
        transport_type=body.transport_type,
        url=body.url,
        headers=body.headers,
        auth_config=_dict_or_none(body.auth_config),
        advanced_config=_dict_or_none(body.advanced_config),
        stdio_config=_dict_or_none(body.stdio_config),
        env_vars=body.env_vars,
    )
    mcp_service.invalidate_discovery_cache(tenant_id=tenant_id, service_id=service_id)
    return service_envelope(info)


@router.delete("/{service_id}", response_model=None)
async def delete_mcp_service(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    service_id: str,
    mcp_service: MCPServiceDep,
    tenant_id: RequireTenantIdDep,
) -> DeleteMCPServiceResponse:
    """Soft-delete an MCP service; builtin services are immutable."""
    deleted = await mcp_service.delete_service(tenant_id=tenant_id, id=service_id)
    if not deleted:
        raise NotFoundError(
            code="mcp_service.not_found",
            message=f"MCP service {service_id} not found",
        )
    mcp_service.invalidate_discovery_cache(tenant_id=tenant_id, service_id=service_id)
    return DeleteMCPServiceResponse(message="MCP service deleted successfully")


@router.post("/{service_id}/test", response_model=MCPTestResultEnvelope)
async def test_mcp_service(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    service_id: str,
    mcp_service: MCPServiceDep,
    tenant_id: RequireTenantIdDep,
) -> MCPTestResultEnvelope:
    """Probe the configured service and report the result."""
    result = await mcp_service.test_service(tenant_id=tenant_id, service_id=service_id)
    return test_result_envelope(result)


@router.get("/{service_id}/tools", response_model=MCPToolEnvelope)
async def list_mcp_service_tools(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    service_id: str,
    mcp_service: MCPServiceDep,
    tenant_id: RequireTenantIdDep,
) -> MCPToolEnvelope:
    """Discover the upstream tools exposed by the service."""
    return tool_envelope(
        await mcp_service.list_tools(tenant_id=tenant_id, service_id=service_id),
    )


@router.get(
    "/{service_id}/resources",
    response_model=MCPResourceEnvelope,
)
async def list_mcp_service_resources(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    service_id: str,
    mcp_service: MCPServiceDep,
    tenant_id: RequireTenantIdDep,
) -> MCPResourceEnvelope:
    """Discover the upstream resources exposed by the service."""
    return resource_envelope(
        await mcp_service.list_resources(tenant_id=tenant_id, service_id=service_id),
    )


@router.get(
    "/{service_id}/tool-approvals",
    response_model=MCPToolApprovalListEnvelope,
)
async def list_mcp_tool_approvals(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    service_id: str,
    mcp_service: MCPServiceDep,
    tenant_id: RequireTenantIdDep,
) -> MCPToolApprovalListEnvelope:
    """Return the persisted approval overrides for the service."""
    return approval_list_envelope(
        await mcp_service.list_tool_approvals(
            tenant_id=tenant_id,
            service_id=service_id,
        ),
    )


@router.put(
    "/{service_id}/tool-approvals/{tool_name}",
    response_model=None,
)
async def set_mcp_tool_approval(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    service_id: str,
    tool_name: str,
    body: SetMCPToolApprovalRequest,
    mcp_service: MCPServiceDep,
    tenant_id: RequireTenantIdDep,
) -> dict[str, JsonValue]:
    """Upsert one tool's ``require_approval`` flag."""
    await mcp_service.set_tool_approval(
        tenant_id=tenant_id,
        service_id=service_id,
        tool_name=tool_name,
        require_approval=body.require_approval,
    )
    return {"success": True}


@router.post(
    "/{service_id}/oauth/authorize-url",
    response_model=None,
)
async def start_oauth_authorization(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    service_id: str,
    body: OAuthAuthorizeRequestBody,
    mcp_service: MCPServiceDep,
    tenant_id: RequireTenantIdDep,
    user_id: RequireUserIdDep,
) -> MCPOAuthAuthorizeURLResponse:
    """Begin OAuth authorisation; return the URL the browser must open."""
    if not body.redirect_uri:
        raise ValidationError(
            code="mcp_service.redirect_uri_required",
            message="redirect_uri is required",
        )
    manager = await mcp_service.fetch_oauth_manager(
        tenant_id=tenant_id,
        service_id=service_id,
    )
    outcome = manager.start_authorization(
        redirect_uri=body.redirect_uri,
        frontend_redirect=body.frontend_redirect,
        user_id=user_id,
    )
    return oauth_authorize_envelope(
        authorization_url=outcome.authorization_url,
        authorization_attempt=outcome.authorization_attempt,
    )


@router.get(
    "/{service_id}/oauth/status",
    response_model=MCPOAuthStatusEnvelope,
)
async def get_oauth_status(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    service_id: str,
    mcp_service: MCPServiceDep,
    tenant_id: RequireTenantIdDep,
    user_id: RequireUserIdDep,
) -> MCPOAuthStatusEnvelope:
    """Report the per-user OAuth authorisation status."""
    manager = await mcp_service.fetch_oauth_manager(
        tenant_id=tenant_id,
        service_id=service_id,
    )
    status = manager.authorization_status(user_id=user_id)
    return oauth_status_envelope(
        JsonObject(
            {
                "authorized": status.authorized,
                "state": status.state,
                "refresh_available": status.refresh_available,
                "expires_at": (status.expires_at.isoformat() if status.expires_at else None),
            },
        ),
    )


@router.delete(
    "/{service_id}/oauth/token",
    response_model=None,
    status_code=204,
)
async def revoke_oauth_token(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    service_id: str,
    mcp_service: MCPServiceDep,
    tenant_id: RequireTenantIdDep,
    user_id: RequireUserIdDep,
) -> None:
    """Revoke the per-user OAuth token."""
    manager = await mcp_service.fetch_oauth_manager(
        tenant_id=tenant_id,
        service_id=service_id,
    )
    manager.revoke(user_id=user_id)
    # FastAPI returns the empty body when ``response_model`` is None
    # and we return None — combine with status_code=204 above.
    return


# ── Helpers ──────────────────────────────────────────────────────────


def _dict_or_none(model: BaseModel | JsonObject | None) -> JsonObject | None:
    """Render a wire DTO (or ``None``) as a JSON-compatible dict.

    The frozen contract DTOs expose ``model_dump``; nested configs
    (e.g. ``auth_config``) come in as those DTOs. A free-form dict is
    also accepted so callers can pass through pre-built dicts.
    """
    if model is None:
        return None
    if isinstance(model, BaseModel):
        dumped = model.model_dump(mode="json", exclude_none=True)
        if isinstance(dumped, dict):
            return JsonObject(dumped)
        return JsonObject({"value": dumped})
    if isinstance(model, dict):
        return JsonObject(model)
    raise ValidationError(
        code="mcp_service.bad_payload",
        message="Expected a mapping or wire DTO",
    )


__all__ = ["OAuthAuthorizeRequestBody", "router"]
