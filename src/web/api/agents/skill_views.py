"""Skill HTTP endpoints — the preloaded-skill catalog.

Maps the upstream skill handler's single read endpoint: list every
preloaded skill's metadata so the conversation UI can offer skills.

The ``skills_available`` flag mirrors the upstream sandbox-mode probe:
skills are only surfaced when a sandbox backend is enabled, so the
frontend can hide the Skills UI in a deployment without one.

Sandboxed execution is the skills-manager's tool seam, not a separate
HTTP endpoint in the upstream handler, so this module exposes no
execute route.
"""

from __future__ import annotations

import os

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from src.web.deps import AuthDep, RoleViewerDep
from src.web.deps.agents import SkillsManagerDep

#: Environment variable naming the sandbox backend kind. Any value other
#: than empty or ``disabled`` means skills are available.
_SANDBOX_MODE_ENV = "KB_SANDBOX_MODE"
_DISABLED_MODE = "disabled"


class SkillInfoResponse(BaseModel):
    """One preloaded skill's metadata — ``SkillInfoResponse``."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str


class SkillListEnvelope(BaseModel):
    """``{"success": true, "data": [...], "skills_available": bool}``."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: list[SkillInfoResponse] = Field(default_factory=list)
    skills_available: bool


def _skills_available() -> bool:
    """Return whether the sandbox backend is enabled for skills.

    Mirrors the upstream probe: an unset or ``disabled`` sandbox mode
    hides the Skills UI.
    """
    mode = os.getenv(_SANDBOX_MODE_ENV, "")
    return mode != "" and mode != _DISABLED_MODE


skill_router = APIRouter(prefix="/skills", tags=["agents.skills"])


@skill_router.get("", response_model=SkillListEnvelope)
async def list_skills(
    _auth: AuthDep,
    _role: RoleViewerDep,
    manager: SkillsManagerDep,
) -> SkillListEnvelope:
    """List every preloaded skill's name and description.

    The manager is discovered at request construction; a deployment
    without configured skill directories yields an empty catalog.
    """
    metadata = manager.get_all_metadata()
    return SkillListEnvelope(
        success=True,
        data=[SkillInfoResponse(name=meta.name, description=meta.description) for meta in metadata],
        skills_available=_skills_available(),
    )


__all__ = [
    "SkillInfoResponse",
    "SkillListEnvelope",
    "skill_router",
]
