"""MCP service connectivity test result.

The Go ``TestMCPService`` opens a real MCP client connection, calls
``Initialize`` / ``ListTools`` / ``ListResources``, and either
succeeds or converts a 401-with-OAuth-challenge into the
``oauth_required`` flag the UI uses to prompt the user to switch auth.

Here we expose the same result shape but rely on an injectable
``ConnectivityProbe`` callable so tests can verify the success path,
the failure-with-OAuth-challenge path, and the generic failure path
without a live server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.core.infra.mcp_services.discovery import DiscoveryResource, DiscoveryTool


class ConnectivityProbe(Protocol):
    """Probe a configured MCP service. Real impls speak MCP-over-HTTP."""

    async def __call__(
        self,
        *,
        service_id: str,
        transport_type: str,
        url: str | None,
        oauth_required: bool,
    ) -> ConnectivityResult: ...


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
    transport lands.
    """

    def __init__(self, *, result: ConnectivityResult) -> None:
        self._result = result

    async def __call__(
        self,
        *,
        service_id: str,
        transport_type: str,
        url: str | None,
        oauth_required: bool,
    ) -> ConnectivityResult:
        return self._result


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
    "ConnectivityProbe",
    "ConnectivityResult",
    "MCPTestResultWire",
    "StaticConnectivityProbe",
    "to_wire",
]
