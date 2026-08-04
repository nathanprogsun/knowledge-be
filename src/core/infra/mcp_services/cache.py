"""Request-scoped discovery cache wiring for the MCP service domain.

Holds a per-request :class:`DiscoveryCache` instance; the discovery
methods on :class:`MCPServiceService` consult this cache first, then
fall back to the live provider.
"""

from __future__ import annotations

from src.core.infra.mcp_services.discovery import DiscoveryCache

__all__ = ["DiscoveryCache"]
