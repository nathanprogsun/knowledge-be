"""Request-scoped factory for the MCP service domain.

Mirrors ``src.core.tenants.factory`` / ``src.core.auth.factory`` —
repositories are constructed per request on the shared
``AsyncSession``; ``web`` never imports ``db``.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.infra.mcp_services.service import MCPServiceService
from src.db.dao.mcp_service_repository import MCPServiceRepository
from src.db.dao.mcp_tool_approval_repository import MCPToolApprovalRepository


def build_mcp_service(session: AsyncSession) -> MCPServiceService:
    """Per-request ``MCPServiceService`` with fresh repositories."""
    return MCPServiceService(
        mcp_repo=MCPServiceRepository(session),
        tool_approvals_repo=MCPToolApprovalRepository(session),
    )


__all__ = ["build_mcp_service"]
