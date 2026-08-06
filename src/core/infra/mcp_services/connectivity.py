"""MCP service connectivity test result.

The Go ``TestMCPService`` opens a real MCP client connection, calls
``Initialize`` / ``ListTools`` / ``ListResources``, and either
succeeds or converts a 401-with-OAuth-challenge into the
``oauth_required`` flag the UI uses to prompt the user to switch auth.

Here we expose the same result shape but rely on an injectable
:class:`ConnectivityProbe` callable so tests can verify the success
path, the failure-with-OAuth-challenge path, and the generic failure
path without a live server.

Two implementations ship in this module:

- :class:`StaticConnectivityProbe` — pre-baked result, used by tests.
- :class:`HTTPMCPConnectivityProbe` — drives the live MCP
  transport via :class:`src.ai.mcp_transport.MCPConnectionManager`
  and reports the actual probe outcome.
"""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from src.ai.mcp_transport.connection_manager import MCPSession
from src.ai.mcp_transport.errors import MCPError, OAuthRequiredError
from src.ai.mcp_transport.jsonrpc import JSONRPCResponse
from src.common.json import JsonObject
from src.core.infra.mcp_services.discovery import (
    DiscoveryResource,
    DiscoveryTool,
)
from src.core.infra.mcp_services.types import MCPServiceInfo


class ConnectivityProbe(Protocol):
    """Probe a configured MCP service. Real impls speak MCP-over-HTTP."""

    async def __call__(
        self,
        *,
        tenant_id: int,
        service_id: str,
        transport_type: str,
        url: str | None,
        oauth_required: bool,
    ) -> ConnectivityResult: ...


class ConnectivityResolver(Protocol):
    """Async ``(tenant_id, service_id) → MCPServiceInfo`` resolver for live probes.

    The factory passes a callable that returns the live ``MCPServiceInfo``
    (or a pre-baked dict in tests) so this module does not import the
    repository layer. PR-17.5c C2: the resolver takes the active
    ``tenant_id`` so cross-tenant OAuth tokens / URLs do not leak
    through a hard-coded ``tenant_id=0`` lookup.
    """

    def __call__(
        self,
        tenant_id: int,
        service_id: str,
    ) -> Awaitable[MCPServiceInfo | JsonObject]: ...


@dataclass(frozen=True)
class ConnectivityResult:
    """Raw probe result before rendering to the wire shape."""

    success: bool
    message: str
    description: str | None = None
    oauth_required: bool = False
    tools: tuple[DiscoveryTool, ...] = ()
    resources: tuple[DiscoveryResource, ...] = ()


class MCPTestResultWire(BaseModel):
    """Wire shape mirroring ``MCPTestResult`` in the frozen contract."""

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str
    description: str | None = Field(default=None)
    oauth_required: bool = Field(default=False)
    tools: list[DiscoveryTool] = Field(default_factory=list)
    resources: list[DiscoveryResource] = Field(default_factory=list)


class StaticConnectivityProbe:
    """Default probe returning a pre-baked result.

    Used in unit tests and as the wired-default until a real MCP
    transport lands. Production wiring installs
    :class:`HTTPMCPConnectivityProbe`.
    """

    def __init__(self, *, result: ConnectivityResult) -> None:
        self._result = result

    async def __call__(
        self,
        *,
        tenant_id: int,
        service_id: str,
        transport_type: str,
        url: str | None,
        oauth_required: bool,
    ) -> ConnectivityResult:
        del tenant_id  # static probe does not consult the DB
        return self._result


class ConnectionManagerLike(Protocol):
    """Minimal surface the live connectivity probe needs.

    Declared as a Protocol so callers can pass any object that exposes
    ``get_or_create`` / ``list_tools`` / ``list_resources`` without
    forcing this module to import the AI layer.

    PR-30.6c C7: promoted from ``_ConnectionManagerLike`` to a public
    name so the web-layer forwarder can import a typed alias without
    re-introducing a bare ``object`` annotation.
    """

    async def get_or_create(  # type: ignore[no-untyped-def]
        self,
        *,
        service_id: str,
        transport_type: str,
        url: str,
        headers: dict[str, str] | None,
        advanced_timeout_seconds: int | None = None,
        service_name: str | None = None,
    ): ...

    async def list_tools(self, *, session: MCPSession) -> JSONRPCResponse: ...

    async def list_resources(self, *, session: MCPSession) -> JSONRPCResponse: ...

# Backwards-compatibility alias for code that still imports the
# underscore-prefixed name from earlier PRs.
_ConnectionManagerLike = ConnectionManagerLike


class HTTPMCPConnectivityProbe:
    """Probe a live MCP service through the connection manager.

    The probe runs ``initialize`` + ``tools/list`` +
    ``resources/list``; a 401 that carries an RFC 9728
    ``resource_metadata`` link is translated into
    ``oauth_required=True`` so the UI can prompt the user to switch
    auth strategy instead of surfacing a generic failure.
    """

    def __init__(
        self,
        *,
        connection_manager: ConnectionManagerLike,
        resolver: ConnectivityResolver,
    ) -> None:
        self._manager = connection_manager
        self._resolver = resolver

    async def __call__(
        self,
        *,
        tenant_id: int,
        service_id: str,
        transport_type: str,
        url: str | None,
        oauth_required: bool,
    ) -> ConnectivityResult:
        info = await self._resolver(tenant_id, service_id)
        if isinstance(info, MCPServiceInfo):
            resolved_url = info.url
            resolved_transport = info.transport_type
            resolved_headers = info.headers or {}
            resolved_advanced = _extract_timeout(info)
            resolved_name = info.name
        elif isinstance(info, dict):
            resolved_url_raw = info.get("url")
            resolved_url = str(resolved_url_raw) if resolved_url_raw else ""
            resolved_transport = str(info.get("transport_type") or transport_type)
            raw_headers = info.get("headers") or {}
            resolved_headers = (
                cast("dict[str, str]", raw_headers) if isinstance(raw_headers, dict) else {}
            )
            resolved_advanced_raw = info.get("advanced_timeout_seconds")
            resolved_advanced = (
                int(resolved_advanced_raw)
                if isinstance(resolved_advanced_raw, (int, float))
                else None
            )
            resolved_name = str(info.get("name") or service_id)
        else:
            return ConnectivityResult(
                success=False,
                message="service resolver returned an unsupported payload",
            )

        if not resolved_url:
            return ConnectivityResult(
                success=False,
                message=(
                    f"service {service_id!r} has no URL configured for "
                    f"transport_type={resolved_transport!r}"
                ),
            )

        try:
            session = await self._manager.get_or_create(
                service_id=service_id,
                transport_type=resolved_transport,
                url=resolved_url,
                headers=resolved_headers,
                advanced_timeout_seconds=resolved_advanced,
                service_name=resolved_name,
            )
            tools_response = await self._manager.list_tools(session=session)
            resources_response = await self._manager.list_resources(session=session)
            tools = _as_tools(tools_response)
            resources = _as_resources(resources_response)
        except OAuthRequiredError as exc:
            return ConnectivityResult(
                success=False,
                message=(
                    "the MCP server requires OAuth authorization "
                    f"(advertised metadata URL: {exc.metadata_url})"
                ),
                oauth_required=True,
            )
        except MCPError as exc:
            return ConnectivityResult(
                success=False,
                message=getattr(exc, "message_text", None) or str(exc),
                description=None,
                oauth_required=False,
            )

        return ConnectivityResult(
            success=True,
            message=(f"connected to {resolved_transport} MCP service at {resolved_url}"),
            description=f"session id: {session.session_id or 'n/a'}",
            oauth_required=oauth_required,
            tools=tools,
            resources=resources,
        )


def _extract_timeout(info: MCPServiceInfo) -> int | None:
    if isinstance(info.advanced_config, dict):
        raw_timeout = info.advanced_config.get("timeout")
        if isinstance(raw_timeout, (int, float)) and raw_timeout > 0:
            return int(raw_timeout)
    return None


def _as_tools(response: JSONRPCResponse) -> tuple[DiscoveryTool, ...]:
    result = response.result or {}
    raw = result.get("tools")
    if not isinstance(raw, list):
        return ()
    tools: list[DiscoveryTool] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            continue
        description = entry.get("description")
        input_schema = entry.get("inputSchema")
        if input_schema is not None and not isinstance(input_schema, dict):
            input_schema = None
        tools.append(
            DiscoveryTool(
                name=name,
                description=description if isinstance(description, str) else None,
                input_schema=cast("JsonObject | None", input_schema),
                require_approval=False,
            ),
        )
    return tuple(tools)


def _as_resources(response: JSONRPCResponse) -> tuple[DiscoveryResource, ...]:
    result = response.result or {}
    raw = result.get("resources")
    if not isinstance(raw, list):
        return ()
    resources: list[DiscoveryResource] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        uri = entry.get("uri")
        name = entry.get("name")
        if not isinstance(uri, str) or not isinstance(name, str):
            continue
        description = entry.get("description")
        mime_type = entry.get("mimeType")
        resources.append(
            DiscoveryResource(
                uri=uri,
                name=name,
                description=description if isinstance(description, str) else None,
                mime_type=mime_type if isinstance(mime_type, str) else None,
            ),
        )
    return tuple(resources)


def to_wire(result: ConnectivityResult) -> MCPTestResultWire:
    """Render a :class:`ConnectivityResult` as the wire-shape DTO."""
    return MCPTestResultWire(
        success=result.success,
        message=result.message,
        description=result.description,
        oauth_required=result.oauth_required,
        tools=list(result.tools),
        resources=list(result.resources),
    )


__all__ = [
    "ConnectionManagerLike",
    "ConnectivityProbe",
    "ConnectivityResolver",
    "ConnectivityResult",
    "HTTPMCPConnectivityProbe",
    "MCPTestResultWire",
    "StaticConnectivityProbe",
    "to_wire",
]
