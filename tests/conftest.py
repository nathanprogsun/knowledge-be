"""Shared pytest fixtures.

Only cross-cutting infrastructure lives here: the settings singleton and
the anyio backend selector.
"""

from __future__ import annotations

import pytest

from src.settings import Settings, get_settings, reset_settings_cache


@pytest.fixture
def settings() -> Settings:
    """Return the Settings singleton after clearing any prior cache."""
    reset_settings_cache()
    return get_settings()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
