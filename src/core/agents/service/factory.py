"""Custom-agent-domain request-scoped service factories.

See ``src.core.tenants.factory`` for the pattern: the repository is
built per request on the shared ``AsyncSession``; ``web`` never imports
``db``.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.agents.service.custom_agent_service import CustomAgentService
from src.db.dao.custom_agent_repository import CustomAgentRepository


def build_custom_agent_service(session: AsyncSession) -> CustomAgentService:
    """Per-request ``CustomAgentService`` with a fresh repository."""
    return CustomAgentService(agent_repo=CustomAgentRepository(session))


__all__ = ["build_custom_agent_service"]
