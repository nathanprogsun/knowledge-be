"""Shared pytest fixtures.

Only cross-cutting infrastructure lives here: the settings singleton,
the anyio backend selector, and the session-scoped Postgres container
backing every integration test.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from testcontainers.community.postgres import PostgresContainer

from src.settings import Settings, get_settings, reset_settings_cache

_POSTGRES_IMAGE = "postgres:16-alpine"

# The ryuk reaper container exists to clean up after a test run that dies
# before its teardown. Ours always stops the container in a `finally`, and
# on Docker Desktop ryuk is a liability: its socket lives at a per-user
# path that cannot be bind-mounted, and its published port is reported as
# unavailable often enough to turn whole DB suites into flaky skips.
_RYUK_DISABLED_ENV = "TESTCONTAINERS_RYUK_DISABLED"


@pytest.fixture
def settings() -> Settings:
    """Return the Settings singleton after clearing any prior cache."""
    reset_settings_cache()
    return get_settings()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session")
def pg_url() -> Iterator[str]:
    """Yield an asyncpg URL for a Postgres container shared by the run.

    The container stays up for the whole session — each consuming module
    creates and drops its own tables, so a shared instance is still
    hermetic per test. Depending tests skip when no usable Docker daemon
    is present (CI runners without Docker, sandboxes).

    ``driver="asyncpg"`` matters: the default ``psycopg2`` driver would
    put ``postgresql+psycopg2://`` in the URL, which neither our engine
    nor our dependency set can open.
    """
    os.environ.setdefault(_RYUK_DISABLED_ENV, "true")
    try:
        container = PostgresContainer(_POSTGRES_IMAGE, driver="asyncpg")
        container.start()
    except Exception as exc:  # pragma: no cover — env-dependent
        pytest.skip(f"Postgres testcontainer could not start ({type(exc).__name__}): {exc}")
    try:
        yield container.get_connection_url()
    finally:
        container.stop()
