"""Discovery helpers for MCP service tools and resources.

The Go ``mcp_service.go`` ``GetMCPServiceTools`` /
``GetMCPServiceResources`` methods go through an ``mcpManager`` that
maintains a pool of live MCP client connections. This module exposes
the same protocol seam:

- :class:`DiscoveryProvider` — the protocol the service layer depends on.
- :class:`StaticDiscoveryProvider` — kept for tests that do not want a
  live transport wired in.
- :class:`HTTPMCPDiscoveryProvider` — live implementation that
  speaks MCP-over-SSE through
  :class:`src.ai.mcp_transport.MCPConnectionManager`.
- :class:`DiscoveryCache` — in-memory TTL cache for one
  ``(tenant_id, service_id)``.

Discovery results are cached for a short TTL via :class:`DiscoveryCache`
to avoid hammering the upstream MCP server when the UI polls.
"""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from src.ai.mcp_transport.connection_manager import MCPSession
from src.ai.mcp_transport.errors import MCPError, MCPTransportError
from src.ai.mcp_transport.jsonrpc import JSONRPCResponse
from src.common.json import JsonObject, JsonValue
from src.core.infra.mcp_services.types import MCPServiceInfo

# Default cache TTL for tool/resource lists. Short so admin edits are
# picked up promptly without restart; the cache is invalidated
# explicitly on update / delete.
_DISCOVERY_TTL = timedelta(minutes=5)


class DiscoveryTool(BaseModel):
    """One tool returned by ``ListTools``."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str | None = Field(default=None)
    input_schema: JsonObject | None = Field(default=None)
    require_approval: bool = Field(default=False)


class DiscoveryResource(BaseModel):
    """One resource returned by ``ListResources``."""

    model_config = ConfigDict(frozen=True)

    uri: str
    name: str
    description: str | None = Field(default=None)
    mime_type: str | None = Field(default=None)


class DiscoveryProvider(Protocol):
    """Protocol implemented by the live MCP transport client.

    The service layer depends only on this protocol; production code
    wires :class:`HTTPMCPDiscoveryProvider` and tests fall back to
    :class:`StaticDiscoveryProvider`.
    """

    async def list_tools(self, *, tenant_id: int, service_id: str) -> list[DiscoveryTool]: ...

    async def list_resources(
        self,
        *,
        tenant_id: int,
        service_id: str,
    ) -> list[DiscoveryResource]: ...


class StaticDiscoveryProvider:
    """Default provider: returns whatever was pre-baked at construction.

    Used by tests that don't want a live transport wired in. The
    production wiring in :func:`src.core.infra.mcp_services.factory.build_mcp_service`
    installs :class:`HTTPMCPDiscoveryProvider` instead.
    """

    def __init__(
        self,
        *,
        tools: dict[str, list[DiscoveryTool]] | None = None,
        resources: dict[str, list[DiscoveryResource]] | None = None,
    ) -> None:
        self._tools = tools or {}
        self._resources = resources or {}

    async def list_tools(self, *, tenant_id: int, service_id: str) -> list[DiscoveryTool]:
        del tenant_id  # tenant scope is implicit in the pre-baked map
        return list(self._tools.get(service_id, []))

    async def list_resources(
        self,
        *,
        tenant_id: int,
        service_id: str,
    ) -> list[DiscoveryResource]:
        del tenant_id  # tenant scope is implicit in the pre-baked map
        return list(self._resources.get(service_id, []))


class ServiceResolver(Protocol):
    """Async ``(tenant_id, service_id) → MCPServiceInfo`` resolver for live discovery.

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


class _ConnectionManagerLike(Protocol):
    """Minimal surface the discovery provider needs from the connection manager.

    Declared as a Protocol so callers can pass any object that exposes
    ``get_or_create`` / ``list_tools`` / ``list_resources`` without
    forcing this module to import the AI layer.
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


class HTTPMCPDiscoveryProvider:
    """Live discovery through the MCP connection manager.

    The provider is constructed once per request; it borrows the
    APP-scope :class:`src.ai.mcp_transport.MCPConnectionManager` from
    the runtime and looks the service up via the supplied resolver.
    The resolver returns either an :class:`MCPServiceInfo` or a
    ``(transport_type, url, headers, advanced_timeout)`` mapping —
    kept as a callable so this module never imports the service
    repository directly.
    """

    def __init__(
        self,
        *,
        connection_manager: _ConnectionManagerLike,
        service_resolver: ServiceResolver,
    ) -> None:
        self._manager = connection_manager
        self._resolver = service_resolver

    async def list_tools(self, *, tenant_id: int, service_id: str) -> list[DiscoveryTool]:
        response = await self._invoke(tenant_id, service_id, method="tools/list")
        return _extract_tools(response)

    async def list_resources(
        self,
        *,
        tenant_id: int,
        service_id: str,
    ) -> list[DiscoveryResource]:
        response = await self._invoke(tenant_id, service_id, method="resources/list")
        return _extract_resources(response)

    async def _invoke(self, tenant_id: int, service_id: str, *, method: str) -> JSONRPCResponse:
        info = await self._resolve(tenant_id, service_id)
        try:
            session = await self._manager.get_or_create(
                service_id=service_id,
                transport_type=str(info["transport_type"]),
                url=str(info["url"]),
                headers=cast("dict[str, str]", info.get("headers") or {}),
                advanced_timeout_seconds=cast(
                    "int | None",
                    info.get("advanced_timeout_seconds"),
                ),
                service_name=cast("str | None", info.get("name")),
            )
        except MCPError as exc:
            # Re-raise as MCPTransportError so the service-layer
            # ``except MCPError`` degrade-to-empty contract holds even
            # for transport failures during session establishment.
            message = getattr(exc, "message_text", None) or str(exc)
            raise MCPTransportError(
                f"manager could not establish a session for {service_id!r}: {message}",
            ) from exc
        if method == "tools/list":
            response = await self._manager.list_tools(session=session)
        elif method == "resources/list":
            response = await self._manager.list_resources(session=session)
        else:
            raise MCPTransportError(f"unsupported discovery method {method!r}")
        if not isinstance(response, JSONRPCResponse):
            raise MCPTransportError(
                f"unexpected response from manager for {method}: {type(response).__name__}",
            )
        if response.error is not None:
            raise MCPTransportError(
                f"MCP {method} returned error {response.error.code}: {response.error.message}",
            )
        return response

    async def _resolve(self, tenant_id: int, service_id: str) -> JsonObject:
        info = await self._resolver(tenant_id, service_id)
        if isinstance(info, MCPServiceInfo):
            return _info_to_resolver_payload(info)
        if isinstance(info, dict):
            payload: JsonObject = dict(info)
            payload.setdefault("transport_type", "sse")
            payload.setdefault("url", "")
            payload.setdefault("headers", {})
            payload.setdefault("advanced_timeout_seconds", None)
            payload.setdefault("name", service_id)
            return payload
        raise TypeError(
            "service_resolver must return an MCPServiceInfo or a resolver dict",
        )


@dataclass(frozen=True)
class _CacheEntry:
    tools: tuple[DiscoveryTool, ...]
    resources: tuple[DiscoveryResource, ...]
    expires_at: datetime


class DiscoveryCache:
    """In-memory TTL cache for one (tenant_id, service_id).

    Used as a request-scoped helper: a service is created once per
    request, populated by the first discovery call, and expires at the
    end of the TTL.
    """

    def __init__(self, *, ttl: timedelta = _DISCOVERY_TTL) -> None:
        self._ttl = ttl
        self._store: dict[tuple[int, str], _CacheEntry] = {}

    def invalidate(self, *, tenant_id: int, service_id: str) -> None:
        self._store.pop((tenant_id, service_id), None)

    def invalidate_all(self) -> None:
        self._store.clear()

    async def get_or_refresh(
        self,
        *,
        tenant_id: int,
        service_id: str,
        provider: DiscoveryProvider,
    ) -> tuple[list[DiscoveryTool], list[DiscoveryResource]]:
        """Return the cached lists, refreshing on miss or expiry."""
        key = (tenant_id, service_id)
        now = datetime.now(UTC)
        cached = self._store.get(key)
        if cached is not None and cached.expires_at > now:
            return list(cached.tools), list(cached.resources)
        tools = await provider.list_tools(tenant_id=tenant_id, service_id=service_id)
        resources = await provider.list_resources(
            tenant_id=tenant_id,
            service_id=service_id,
        )
        self._store[key] = _CacheEntry(
            tools=tuple(tools),
            resources=tuple(resources),
            expires_at=now + self._ttl,
        )
        return list(tools), list(resources)


# ── Helpers ────────────────────────────────────────────────────────


def _extract_tools(response: JSONRPCResponse) -> list[DiscoveryTool]:
    """Translate an MCP ``tools/list`` JSON-RPC response into ``DiscoveryTool`` rows."""
    result = response.result or {}
    raw = result.get("tools")
    if not isinstance(raw, list):
        return []
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
    return tools


def _extract_resources(response: JSONRPCResponse) -> list[DiscoveryResource]:
    """Translate an MCP ``resources/list`` JSON-RPC response into ``DiscoveryResource`` rows."""
    result = response.result or {}
    raw = result.get("resources")
    if not isinstance(raw, list):
        return []
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
    return resources


def _info_to_resolver_payload(info: MCPServiceInfo) -> JsonObject:
    """Render an :class:`MCPServiceInfo` as the resolver payload the manager consumes."""
    advanced_timeout: int | None = None
    if isinstance(info.advanced_config, dict):
        raw_timeout = info.advanced_config.get("timeout")
        if isinstance(raw_timeout, (int, float)) and raw_timeout > 0:
            advanced_timeout = int(raw_timeout)
    headers: dict[str, str] = dict(info.headers or {})
    return {
        "transport_type": info.transport_type,
        "url": info.url or "",
        "headers": cast(dict[str, JsonValue], headers),
        "advanced_timeout_seconds": advanced_timeout,
        "name": info.name,
    }


__all__ = [
    "DiscoveryCache",
    "DiscoveryProvider",
    "DiscoveryResource",
    "DiscoveryTool",
    "HTTPMCPDiscoveryProvider",
    "ServiceResolver",
    "StaticDiscoveryProvider",
]
