"""Agent-domain FastAPI dependency factory.

One-line forwarders to the core factories: the custom-agent service is
built on the request-scoped ``AsyncSession`` (so a mutation and its row
share one transactional unit of work); the skills manager is built from
the configured skill directories with discovery already run. ``web``
never imports ``db``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.core.agents.service.custom_agent_service import CustomAgentService
from src.core.agents.service.factory import build_custom_agent_service
from src.core.agents.skills.factory import build_skills_manager
from src.core.agents.skills.manager import Manager
from src.web.deps.session import SessionDep


def get_custom_agent_service(session: SessionDep) -> CustomAgentService:
    """Build a per-request ``CustomAgentService`` on the shared session."""
    return build_custom_agent_service(session)


def get_skills_manager() -> Manager:
    """Build a per-request skills ``Manager`` with discovery run."""
    return build_skills_manager()


CustomAgentServiceDep = Annotated[CustomAgentService, Depends(get_custom_agent_service)]
SkillsManagerDep = Annotated[Manager, Depends(get_skills_manager)]


__all__ = [
    "CustomAgentServiceDep",
    "SkillsManagerDep",
    "get_custom_agent_service",
    "get_skills_manager",
]
