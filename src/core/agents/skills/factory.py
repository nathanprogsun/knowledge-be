"""Skills-domain request-scoped factory.

Builds a fresh ``Manager`` per request on the configured skill
directories; discovery runs at construction so the metadata cache is
populated before the view reads it. ``web`` never imports ``db`` — the
manager is filesystem-backed and stateless, so no session is involved.
"""

from __future__ import annotations

from src.core.agents.skills.manager import Manager
from src.core.agents.skills.types import ManagerConfig


def build_skills_manager(*, config: ManagerConfig | None = None) -> Manager:
    """Per-request ``Manager`` with discovery already run.

    A default (unconfigured) manager is disabled and carries an empty
    metadata cache, mirroring a deployment without skills configured.
    """
    manager = Manager(config=config)
    manager.initialize()
    return manager


__all__ = ["build_skills_manager"]
