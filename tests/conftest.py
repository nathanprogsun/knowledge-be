"""Shared pytest fixtures.

Only cross-cutting infrastructure lives here: the settings singleton and
the anyio backend selector.
"""

from __future__ import annotations

import pytest

# Eagerly import ``src.workers.tasks`` so the @register_task side
# effects fire once at collection time, not the first time a worker
# test happens to import a single task module. Several per-file
# autouse fixtures snapshot/restore ``registry._REGISTRY``; without
# this pre-warm the snapshot is empty, the restore then drops every
# registered handler, and the contract invariant test that follows
# sees a zero-task registry.
import src.workers.tasks  # noqa: F401 — import-time registration
from src.settings import Settings, get_settings, reset_settings_cache


@pytest.fixture
def settings() -> Settings:
    """Return the Settings singleton after clearing any prior cache."""
    reset_settings_cache()
    return get_settings()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
