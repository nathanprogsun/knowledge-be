"""Discovery helpers for MCP service tools and resources.

The Go ``mcp_service.go`` ``GetMCPServiceTools`` / ``GetMCPServiceResources``
methods go through an ``mcpManager`` that maintains a pool of live
MCP client connections. Real client construction and protocol
serialization lives behind the ``mcp`` package there; here we model the
equivalent surface without a live MCP runtime so the service layer
keeps testable.

``DiscoveryProvider`` is the seam: a callable the service stack
constructs per request. The default provider returns empty result
lists (the repository-rows-but-no-live-connection shape) so the
endpoints are reachable and testable before a real implementation is
introduced in a later checkpoint.

Discovery results are cached per (tenant_id, service_id) for a short
TTL via ``DiscoveryCache`` to avoid hammering the upstream MCP server
when the UI polls.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject

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

    Real implementations (wired in a later checkpoint) speak the
    MCP-over-HTTP protocol; the default provider used by tests just
    returns empty results.
    """

    async def list_tools(self, *, service_id: str) -> list[DiscoveryTool]: ...

    async def list_resources(self, *, service_id: str) -> list[DiscoveryResource]: ...


class StaticDiscoveryProvider:
    """Default provider: returns whatever was pre-baked at construction.

    Keeps the API reachable when no live MCP transport is wired in.
    A real provider will be installed via DI in a later PR.
    """

    def __init__(
        self,
        *,
        tools: dict[str, list[DiscoveryTool]] | None = None,
        resources: dict[str, list[DiscoveryResource]] | None = None,
    ) -> None:
        self._tools = tools or {}
        self._resources = resources or {}

    async def list_tools(self, *, service_id: str) -> list[DiscoveryTool]:
        return list(self._tools.get(service_id, []))

    async def list_resources(self, *, service_id: str) -> list[DiscoveryResource]:
        return list(self._resources.get(service_id, []))


@dataclass(frozen=True)
class _CacheEntry:
    tools: tuple[DiscoveryTool, ...]
    resources: tuple[DiscoveryResource, ...]
    expires_at: datetime


class DiscoveryCache:
    """In-memory TTL cache for one (tenant_id, service_id).

    Used as a request-scoped helper: a service is created once per
    request, populated by the first discovery call, and expires at
    the end of the TTL.
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
        tools = await provider.list_tools(service_id=service_id)
        resources = await provider.list_resources(service_id=service_id)
        self._store[key] = _CacheEntry(
            tools=tuple(tools),
            resources=tuple(resources),
            expires_at=now + self._ttl,
        )
        return list(tools), list(resources)


__all__ = [
    "DiscoveryCache",
    "DiscoveryProvider",
    "DiscoveryResource",
    "DiscoveryTool",
    "StaticDiscoveryProvider",
]
