"""Contract-test conftest.

Provides the same live-DB foundation the integration suite uses, scoped
to ``tests/contract``:

- a session-scoped real ``DatabaseEngine`` (NullPool) against the
  configured Postgres;
- a real ``FastAPI`` app built by ``create_app`` with a manually
  attached ``LifeSpanService`` (the real lifespan's OIDC / MCP startup
  is skipped);
- ``make_test_tenant_id`` / ``make_int32_test_tenant_id`` — unique
  tenant ids, the second bounded to PostgreSQL's INTEGER range because
  the ``chunks.tenant_id`` column is ``int4``;
- ``make_user_org`` — mints a real ``User`` + ``Tenant`` +
  ``TenantMember`` row and commits them, so every request resolves a
  freshly-issued ``(user_id, tenant_id)`` pair from the database.

The ``faker_seed`` autouse fixture re-seeds Faker per test so generated
names are varied but reproducible.
"""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from random import randint
from uuid import uuid4

import pytest
import pytest_asyncio
from faker import Faker
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.app_context.lifespan import create_app
from src.app_context.registry import LifeSpanService
from src.core.tenants.member_service import ROLE_OWNER
from src.db.base import DatabaseEngine
from src.db.dao.tenant_members_repository import TenantMemberRepository
from src.db.dao.tenants_repository import TenantRepository
from src.db.dao.users_repository import UserRepository
from src.db.models.auth.users import User
from src.db.models.tenants.tenant_members import TenantMember
from src.db.models.tenants.tenants import Tenant
from src.settings import get_settings, reset_settings_cache
from src.util.security import hash_password

_FAKER_SEED_MAX = 100_000_000

# Tracks every tenant id minted in this module so concurrent test
# workers cannot reuse one. Module-level so the two generators below
# share the same registry.
_used_tenant_ids: set[int] = set()

# PostgreSQL INTEGER upper bound (exclusive) for ``chunks.tenant_id``.
_INT32_TENANT_LOW = 1_100_000_000
_INT32_TENANT_HIGH = 2_147_483_647  # 2**31 - 1


def make_test_tenant_id() -> int:
    """Return a tenant id unique within the test session.

    Values are bounded to PostgreSQL's positive BIGINT range so they fit
    the ``tenants`` / ``tenant_members`` columns. ``0`` is excluded so
    validators that reject non-positive ids stay satisfied.
    """
    while True:
        candidate = secrets.randbelow(2**63 - 1) + 1
        if candidate not in _used_tenant_ids:
            _used_tenant_ids.add(candidate)
            return candidate


def make_int32_test_tenant_id() -> int:
    """Return a unique tenant id that fits PostgreSQL's INTEGER column.

    The ``chunks.tenant_id`` column is ``int4``, so a chunk row seeded
    with a BIGINT tenant id would overflow. The returned value stays in
    ``[1_100_000_000, 2**31 - 1]``, far above the small ids used by
    fixed seeds yet inside the INTEGER range.
    """
    while True:
        candidate = (
            secrets.randbelow(_INT32_TENANT_HIGH - _INT32_TENANT_LOW + 1) + _INT32_TENANT_LOW
        )
        if candidate not in _used_tenant_ids:
            _used_tenant_ids.add(candidate)
            return candidate


@pytest.fixture(autouse=True)
def faker_seed() -> None:
    """Re-seed Faker per test for varied-but-reproducible generation."""
    Faker.seed(randint(1, _FAKER_SEED_MAX))


def pytest_configure(config: pytest.Config) -> None:
    """Collect the stage contract modules.

    The live contract tests live in modules named after the stage (not
    ``test_*``), so the default ``python_files`` pattern skips them. This
    hook registers the explicit module names so ``pytest tests/contract/``
    discovers them.
    """
    config.addinivalue_line("python_files", "stage4_contract.py")
    config.addinivalue_line("python_files", "stage5_contract.py")
config.addinivalue_line("python_files", "stage6_contract.py")
    config.addinivalue_line("python_files", "stage7_contract.py")


@pytest.fixture(scope="session")
def _integration_settings() -> None:
    """Reset the settings cache so DATABASE_URL_OVERRIDE (or defaults) apply."""
    reset_settings_cache()


@pytest.fixture(scope="session")
def _engine(_integration_settings: None) -> DatabaseEngine:
    """Session-scoped engine against the configured Postgres.

    ``NullPool`` disables connection reuse so a connection checked out
    in one event loop is never handed back in another. The TestClient
    runs requests in its own portal loop, so without NullPool a pooled
    connection created in the test loop would be reused across loops and
    raise a cross-loop ``RuntimeError``.
    """
    from sqlalchemy.pool import NullPool

    settings = get_settings()
    return DatabaseEngine(url=settings.database_url, poolclass=NullPool)


@asynccontextmanager
async def _noop_lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """No-op lifespan so TestClient does not run the real startup."""
    yield


@pytest_asyncio.fixture
async def app(_engine: DatabaseEngine) -> AsyncIterator[FastAPI]:
    """Build the real app with a minimal ``LifeSpanService``.

    The real lifespan startup is skipped (it would try to spin up the
    OIDC / MCP singletons); a ``LifeSpanService`` carrying only the test
    engine is attached so the per-request dependency accessors resolve.
    """
    application = create_app()
    application.state.lifespan_service = LifeSpanService(db_engine=_engine)
    application.router.lifespan_context = _noop_lifespan
    yield application
    application.state.lifespan_service = None


@pytest_asyncio.fixture
async def session(_engine: DatabaseEngine) -> AsyncIterator[AsyncSession]:
    """A per-test ``AsyncSession`` against the integration DB."""
    async with _engine.session_factory() as s:
        yield s
        await s.rollback()


@pytest_asyncio.fixture
async def make_user_org(
    _engine: DatabaseEngine,
) -> Callable[..., Awaitable[tuple[str, int]]]:
    """Return a callable that mints a fresh ``(user_id, tenant_id)`` pair.

    Each call commits one ``tenants`` row, one ``users`` row, and one
    ``tenant_members`` row binding the user to the tenant as owner. The
    user's ``tenant_id`` column is left ``None`` so it stays inside the
    INTEGER range regardless of the random tenant id; the header-trust
    auth path resolves the active tenant from ``x-knowledge-tenant-id``.
    """

    async def _factory(
        *,
        tenant_id: int | None = None,
        role: str = ROLE_OWNER,
        suffix: str | None = None,
    ) -> tuple[str, int]:
        now = datetime.now(UTC)
        tag = suffix or uuid4().hex[:8]
        user_id = f"usr-{secrets.token_hex(8)}"
        tenant_id = tenant_id if tenant_id is not None else make_test_tenant_id()
        async with _engine.session_factory() as session, session.begin():
            tenants_repo = TenantRepository(session)
            users_repo = UserRepository(session)
            members_repo = TenantMemberRepository(session)
            tenant_row = await tenants_repo.insert(
                Tenant(
                    name=f"workspace-{tag}",
                    description=f"per-test workspace {tag}",
                    status="active",
                    business="",
                    retriever_engines={"engines": []},
                    created_at=now,
                    updated_at=now,
                )
            )
            await users_repo.insert(
                User(
                    id=user_id,
                    username=f"user-{tag}",
                    email=f"user-{tag}@example.test",
                    password_hash=hash_password("test-password"),
                    tenant_id=None,
                    is_active=True,
                    created_at=now,
                    updated_at=now,
                )
            )
            await members_repo.insert_live_or_none(
                TenantMember(
                    user_id=user_id,
                    tenant_id=tenant_row.id,
                    role=role,
                    status="active",
                    invited_by=None,
                    joined_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
        return user_id, tenant_row.id

    return _factory


@pytest_asyncio.fixture
async def admin_user(
    make_user_org: Callable[..., Awaitable[tuple[str, int]]],
) -> tuple[str, int]:
    """Per-test default principal: ``(user_id, tenant_id)`` minted fresh."""
    return await make_user_org()


@pytest_asyncio.fixture
async def authed_client(
    app: FastAPI,
    admin_user: tuple[str, int],
) -> AsyncIterator[TestClient]:
    """A ``TestClient`` that authenticates via the ``x-knowledge-*`` headers.

    The headers are sourced from the freshly-minted ``admin_user`` so
    every request resolves a real ``User`` row + a real ``Tenant`` row.
    The ``with`` block is required: without it asyncpg raises a
    cross-loop ``RuntimeError`` because TestClient runs requests in its
    own portal loop.
    """
    user_id, tenant_id = admin_user
    with TestClient(app=app) as c:
        c.headers.update(
            {
                "x-knowledge-user-id": user_id,
                "x-knowledge-tenant-id": str(tenant_id),
                "x-knowledge-roles": ROLE_OWNER,
            }
        )
        yield c


__all__ = [
    "_engine",
    "_integration_settings",
    "_noop_lifespan",
    "admin_user",
    "app",
    "authed_client",
    "faker_seed",
    "make_int32_test_tenant_id",
    "make_test_tenant_id",
    "make_user_org",
    "session",
]
