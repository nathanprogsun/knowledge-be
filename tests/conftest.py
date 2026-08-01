"""Shared pytest fixtures.

PR-0 fixtures are intentionally minimal — domain-specific fixtures land
with their owning PR (auth fixtures in PR-1, knowledge fixtures in PR-4, …).
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
