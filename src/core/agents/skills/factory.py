"""Build the skills manager from env-driven sandbox + skill directories.

``KB_SANDBOX_MODE`` selects the backend (``local`` / ``docker`` /
``disabled``). ``KB_SANDBOX_DOCKER_IMAGE`` overrides the Docker image.
``KB_SKILL_DIRS`` is a colon-separated list of skill roots; when unset
and the sandbox is enabled, the manager stays enabled with an empty
catalog until directories are configured.
"""

from __future__ import annotations

import os

from src.core.agents.engine.sandbox.manager import new_manager_from_type
from src.core.agents.skills.manager import Manager
from src.core.agents.skills.types import ManagerConfig

_SANDBOX_MODE_ENV = "KB_SANDBOX_MODE"
_SANDBOX_IMAGE_ENV = "KB_SANDBOX_DOCKER_IMAGE"
_SKILL_DIRS_ENV = "KB_SKILL_DIRS"
_DISABLED = "disabled"


def _sandbox_mode() -> str:
    return os.getenv(_SANDBOX_MODE_ENV, _DISABLED).strip().lower() or _DISABLED


def _skill_dirs() -> list[str]:
    raw = os.getenv(_SKILL_DIRS_ENV, "").strip()
    if not raw:
        return []
    return [part for part in raw.split(":") if part]


def build_skills_manager(*, config: ManagerConfig | None = None) -> Manager:
    """Per-request ``Manager`` with discovery already run.

    When ``config`` is omitted, mode and directories come from the
    environment. A disabled sandbox yields an enabled=False manager
    (empty catalog), matching a deployment without skills.
    """
    if config is not None:
        manager = Manager(config=config)
        manager.initialize()
        return manager

    mode = _sandbox_mode()
    enabled = mode != _DISABLED
    skill_dirs = _skill_dirs()
    sandbox = None
    if enabled:
        sandbox = new_manager_from_type(
            mode if mode in {"docker", "local"} else "local",
            fallback_enabled=True,
            docker_image=os.getenv(_SANDBOX_IMAGE_ENV, "").strip(),
        )
    manager = Manager(
        config=ManagerConfig(skill_dirs=skill_dirs, enabled=enabled),
        sandbox_manager=sandbox,
    )
    manager.initialize()
    return manager


__all__ = ["build_skills_manager"]
