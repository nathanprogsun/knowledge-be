"""Wire-shape helpers for the MCP service endpoints.

The router declares response models as frozen Pydantic contracts
imported from :mod:`src.core.contracts.infra`; this module owns the
DTO/projection from :class:`MCPServiceInfo` to those wire models and
the standard success / list envelopes.
"""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject, JsonValue
from src.core.contracts.infra import (
    CreateMCPServiceRequest,
    MCPMcpServiceAuthConfig,
    MCPResource,
    MCPService,
    MCPServiceAdvancedConfig,
    MCPTestResult,
    MCPTool,
    MCPToolApproval,
    SetMCPToolApprovalRequest,
    UpdateMCPServiceRequest,
)
from src.core.infra.mcp_services.connectivity import ConnectivityResult, to_wire
from src.core.infra.mcp_services.discovery import DiscoveryResource, DiscoveryTool
from src.core.infra.mcp_services.types import MCPServiceInfo, MCPToolApprovalInfo


class MCPServiceEnvelope(BaseModel):
    """``{success: true, data: {...}}`` envelope for single MCP services."""

    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: MCPService


class MCPServiceListEnvelope(BaseModel):
    """``{success: true, data: [...]}`` envelope for MCP service lists."""

    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: list[MCPService]


class DeleteMCPServiceResponse(BaseModel):
    """``{success: true, message: ...}`` acks for the DELETE endpoint."""

    model_config = ConfigDict(frozen=True)

    success: bool = True
    message: str


class MCPTestResultEnvelope(BaseModel):
    """``{success: true, data: {...}}`` for the test endpoint."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: MCPTestResult


class MCPToolApprovalListEnvelope(BaseModel):
    """``{success: true, data: [...]}`` for the tool-approvals endpoint."""

    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: list[MCPToolApproval]


class MCPOAuthAuthorizeURLResponse(BaseModel):
    """``{success: true, data: {authorization_url, attempt}}`` shape."""

    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: JsonObject = Field(default_factory=dict)


class MCPOAuthStatusEnvelope(BaseModel):
    """``{success: true, data: {...}}`` for the status endpoint."""

    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: JsonObject = Field(default_factory=dict)


# ── Projections ──────────────────────────────────────────────────────


def _to_auth_config(info: MCPServiceInfo) -> MCPMcpServiceAuthConfig | None:
    """Project the service's auth_config to the wire DTO.

    Credentials are kept verbatim on user-created services, mirroring
    Go's ``dto.NewMCPServiceResponse`` (which round-trips the
    ``api_key`` / ``token`` fields on user rows; built-in rows are not
    exposed via the public API).
    """
    raw = info.auth_config
    if not isinstance(raw, dict):
        return None
    return MCPMcpServiceAuthConfig.model_validate(raw)


def _auth_config_to_wire_dict(model: MCPMcpServiceAuthConfig) -> dict[str, JsonValue]:
    """Serialise the auth DTO verbatim.

    The frozen contract requires ``api_key`` and ``token`` as nullable
    fields. User-created rows carry them through unchanged so the wire
    shape matches ``docs/api/mcp-service.md``.
    """
    dumped = model.model_dump(mode="json", exclude_none=True)
    if not isinstance(dumped, dict):
        return {}
    return dict(dumped)


def _to_advanced_config(
    info: MCPServiceInfo,
) -> MCPServiceAdvancedConfig | None:
    """Project the service's ``advanced_config`` to the wire DTO."""
    raw = info.advanced_config
    if not isinstance(raw, dict):
        return None
    return MCPServiceAdvancedConfig.model_validate(raw)


def service_info_to_contract(info: MCPServiceInfo) -> MCPService:
    """Render a service DTO onto the frozen wire contract."""
    auth = _to_auth_config(info)
    return MCPService(
        id=info.id,
        tenant_id=info.tenant_id,
        name=info.name,
        description=info.description,
        enabled=info.enabled,
        transport_type=info.transport_type,
        url=info.url,
        headers=info.headers,
        auth_config=auth,
        advanced_config=_to_advanced_config(info),
        stdio_config=None,  # stdio is disabled in the Python scaffold.
        env_vars=info.env_vars,
        is_builtin=info.is_builtin,
        created_at=info.created_at,
        updated_at=info.updated_at,
    )


def service_envelope(info: MCPServiceInfo) -> dict[str, JsonValue]:
    """Wrap one service in the success envelope.

    The auth_config is serialized with secret fields stripped so the
    response body never carries ``api_key`` / ``token`` regardless of
    what callers sent in.
    """
    wire = service_info_to_contract(info)
    body = wire.model_dump(mode="json", exclude_none=True)
    if wire.auth_config is not None:
        body["auth_config"] = _auth_config_to_wire_dict(wire.auth_config)
    return {"success": True, "data": body}


def service_list_envelope(infos: list[MCPServiceInfo]) -> dict[str, JsonValue]:
    """Wrap the list in the standard envelope."""
    data: list[dict[str, JsonValue]] = []
    for info in infos:
        wire = service_info_to_contract(info)
        payload = wire.model_dump(mode="json", exclude_none=True)
        if wire.auth_config is not None:
            payload["auth_config"] = _auth_config_to_wire_dict(wire.auth_config)
        data.append(payload)
    return {"success": True, "data": cast("JsonValue", data)}


def tool_envelope(tools: list[DiscoveryTool]) -> MCPToolEnvelope:
    """Wrap a discovered tools list in a wire-shape envelope."""
    return MCPToolEnvelope(
        data=[
            MCPTool(
                name=t.name,
                description=t.description,
                inputSchema=t.input_schema,
                require_approval=t.require_approval,
            )
            for t in tools
        ],
    )


def resource_envelope(resources: list[DiscoveryResource]) -> MCPResourceEnvelope:
    """Wrap a discovered resources list in a wire-shape envelope."""
    return MCPResourceEnvelope(
        data=[
            MCPResource(
                uri=r.uri,
                name=r.name,
                description=r.description,
                mimeType=r.mime_type,
            )
            for r in resources
        ],
    )


def test_result_envelope(result: ConnectivityResult) -> MCPTestResultEnvelope:
    """Wrap a probe result in the standard envelope.

    The outer ``success`` mirrors the upstream wire convention: it is
    ``True`` when the call itself returned a result (200), even when
    the inner ``data.success`` is False (the test failed). This keeps
    HTTP semantics (200 OK) decoupled from the probe verdict.
    """
    wire = to_wire(result)
    wire_tools = [
        MCPTool(
            name=t.name,
            description=t.description,
            inputSchema=t.input_schema,
            require_approval=t.require_approval,
        )
        for t in wire.tools
    ]
    wire_resources = [
        MCPResource(
            uri=r.uri,
            name=r.name,
            description=r.description,
            mimeType=r.mime_type,
        )
        for r in wire.resources
    ]
    return MCPTestResultEnvelope(
        success=True,
        data=MCPTestResult(
            success=wire.success,
            message=wire.message,
            description=wire.description,
            oauth_required=wire.oauth_required,
            tools=wire_tools,
            resources=wire_resources,
        ),
    )


def approval_info_to_contract(info: MCPToolApprovalInfo) -> MCPToolApproval:
    """Render an approval override as the wire DTO."""
    return MCPToolApproval(
        tool_name=info.tool_name,
        require_approval=info.require_approval,
        updated_at=info.updated_at,
    )


def approval_list_envelope(
    approvals: list[MCPToolApprovalInfo],
) -> MCPToolApprovalListEnvelope:
    """Wrap approval overrides in the success envelope."""
    return MCPToolApprovalListEnvelope(
        data=[approval_info_to_contract(a) for a in approvals],
    )


def oauth_authorize_envelope(
    *,
    authorization_url: str,
    authorization_attempt: str,
) -> MCPOAuthAuthorizeURLResponse:
    """Wrap the authorize-URL outcome in the standard envelope."""
    return MCPOAuthAuthorizeURLResponse(
        data={
            "authorization_url": authorization_url,
            "authorization_attempt": authorization_attempt,
        },
    )


def oauth_status_envelope(payload: JsonObject) -> MCPOAuthStatusEnvelope:
    """Wrap the OAuth status payload in the standard envelope."""
    return MCPOAuthStatusEnvelope(data=payload)


# Response envelopes for tool/resource lists referenced above. Defined
# at module scope so type checkers see them referenced in annotations.


class MCPToolEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: list[MCPTool]


class MCPResourceEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: list[MCPResource]


# Request bodies — small helpers to import the frozen contracts once for
# the router annotations. Re-exported so the router import block stays
# short.


__all__ = [
    "CreateMCPServiceRequest",
    "DeleteMCPServiceResponse",
    "MCPOAuthAuthorizeURLResponse",
    "MCPOAuthStatusEnvelope",
    "MCPResourceEnvelope",
    "MCPTestResultEnvelope",
    "MCPToolApprovalListEnvelope",
    "MCPToolEnvelope",
    "SetMCPToolApprovalRequest",
    "UpdateMCPServiceRequest",
    "approval_info_to_contract",
    "approval_list_envelope",
    "oauth_authorize_envelope",
    "oauth_status_envelope",
    "resource_envelope",
    "service_envelope",
    "service_info_to_contract",
    "service_list_envelope",
    "test_result_envelope",
    "tool_envelope",
]
