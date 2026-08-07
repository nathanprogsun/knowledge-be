"""Integration-test conftest foundation.

Provides a session-scoped real ``DatabaseEngine`` against the configured
Postgres instance, a real FastAPI app with a manually-constructed
``LifeSpanService`` (so the real lifespan's OIDC / MCP startup is
skipped), and an ``authed_client`` that sets the ``x-knowledge-*``
header trio so the real ``require_auth`` dependency runs end-to-end.

The principal is a real ``User`` row minted by the ``make_user_org``
factory for every test, so each request resolves a freshly-issued
``(user_id, tenant_id)`` pair from the database - no canonical
``usr-int-1`` / ``tenant_id=1`` seed.
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
from src.core.tenants.member_service import ROLE_CONTRIBUTOR, ROLE_OWNER
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

# Tracks every tenant_id minted by ``make_test_tenant_id`` so concurrent
# test workers cannot reuse one. Module-level so all conftests share the
# same registry.
_used_tenant_ids: set[int] = set()


def make_test_tenant_id() -> int:
    """Return a tenant_id guaranteed unique within the test session.

    Values are bounded to PostgreSQL's positive BIGINT range
    (``1..2**63 - 1``) so they fit in the ``tenants`` and
    ``tenant_members`` tables. ``0`` is excluded so callers can use the
    return value with validators that reject non-positive ids.
    """
    while True:
        candidate = secrets.randbelow(2**63 - 1) + 1
        if candidate not in _used_tenant_ids:
            _used_tenant_ids.add(candidate)
            return candidate


@pytest.fixture(autouse=True)
def faker_seed() -> None:
    """Re-seed Faker per test for varied-but-reproducible generation."""
    Faker.seed(randint(1, _FAKER_SEED_MAX))


@pytest.fixture(scope="session")
def _integration_settings() -> None:
    """Reset the settings cache so DATABASE_URL_OVERRIDE (or defaults) apply."""
    reset_settings_cache()


@pytest.fixture(scope="session")
def _engine(_integration_settings: None) -> DatabaseEngine:
    """Session-scoped engine against the configured Postgres.

    ``NullPool`` disables connection reuse so a connection checked out
    in one event loop is never handed back in another (asyncpg binds
    connections to the loop they were created in). The TestClient runs
    requests in its own portal loop, so without NullPool a pooled
    connection created in the test loop would be reused across loops
    and raise a cross-loop ``RuntimeError``.
    """
    from sqlalchemy.pool import NullPool

    settings = get_settings()
    return DatabaseEngine(url=settings.database_url, poolclass=NullPool)


@asynccontextmanager
async def _noop_lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """No-op lifespan so TestClient does not run the real startup.

    The real lifespan wires OIDC and MCP singletons; tests attach a
    minimal ``LifeSpanService`` manually, so the lifespan only needs
    to yield.
    """
    yield


@pytest_asyncio.fixture
async def app(_engine: DatabaseEngine) -> AsyncIterator[FastAPI]:
    """Build the real app with a minimal ``LifeSpanService``.

    The real lifespan startup is skipped (it would try to spin up
    ``OidcClient`` and ``MCPConnectionManager``); instead a
    ``LifeSpanService`` carrying only the test ``db_engine`` is
    attached so ``get_db_engine_from_lifespan`` resolves. The
    lifespan context is replaced with a no-op so TestClient does not
    run the real startup when it enters the app.
    """
    application = create_app()
    lifespan_service = LifeSpanService(db_engine=_engine)
    application.state.lifespan_service = lifespan_service
    application.router.lifespan_context = _noop_lifespan
    yield application
    application.state.lifespan_service = None


@pytest_asyncio.fixture
async def session(_engine: DatabaseEngine) -> AsyncIterator[AsyncSession]:
    """A per-test ``AsyncSession`` against the integration DB.

    Tests use this to seed / assert on real DB rows directly.
    """
    async with _engine.session_factory() as s:
        yield s
        await s.rollback()


# ── Multi-tenant fixtures ─────────────────────────────────────────────
#
# ``make_user_org`` is the project-wide factory every test (web or DAO)
# uses to obtain a unique ``(user_id, tenant_id)`` pair. The factory
# commits one Tenant + one User + one TenantMember row per call; tests
# depend on the freshly-minted pair, so per-test isolation comes from
# random tenant ids rather than DB cleanup.
#
# The factory is function-scoped so each invocation opens its own
# session against the session-scoped engine.


@pytest_asyncio.fixture
async def make_user_org(
    _engine: DatabaseEngine,
) -> Callable[..., Awaitable[tuple[int, int]]]:
    """Return a callable that mints a fresh ``(user_id, tenant_id)`` pair.

    Each call commits:

    - one ``tenants`` row with a unique bigint id, status ``active``;
    - one ``users`` row owned by that tenant;
    - one ``tenant_members`` row binding the user to the tenant as
      ``owner`` (the role most endpoints expect).

    The user's ``tenant_id`` column is intentionally left ``None`` so
    it stays inside the ``INTEGER`` range regardless of the random id;
    the header-trust auth path resolves the active tenant from the
    ``x-knowledge-tenant-id`` header anyway.
    """

    async def _factory(
        *,
        tenant_id: int | None = None,
        role: str = ROLE_OWNER,
        suffix: str | None = None,
    ) -> tuple[int, int]:
        now = datetime.now(UTC)
        tag = suffix or uuid4().hex[:8]
        user_id = f"usr-{secrets.token_hex(8)}"
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
    make_user_org: Callable[..., Awaitable[tuple[int, int]]],
) -> tuple[int, int]:
    """Per-test default principal: ``(user_id, tenant_id)`` minted fresh."""
    return await make_user_org()


@pytest_asyncio.fixture
async def coworker_admin_user(
    admin_user: tuple[int, int],
    make_user_org: Callable[..., Awaitable[tuple[int, int]]],
) -> tuple[int, int]:
    """Second owner inside the same workspace as ``admin_user``.

    Reuses ``admin_user``'s tenant id so the two principals can be
    compared under shared-tenant RBAC rules.
    """
    _user_id, tenant_id = admin_user
    return await make_user_org(tenant_id=tenant_id)


@pytest_asyncio.fixture
async def other_org_admin_user(
    make_user_org: Callable[..., Awaitable[tuple[int, int]]],
) -> tuple[int, int]:
    """Owner of a different freshly-minted workspace for cross-tenant tests."""
    return await make_user_org()


@pytest_asyncio.fixture
async def setup_organization_and_regular_user(
    admin_user: tuple[int, int],
    make_user_org: Callable[..., Awaitable[tuple[int, int]]],
) -> tuple[int, int]:
    """A non-owner user inside the same workspace as ``admin_user``."""
    _user_id, tenant_id = admin_user
    return await make_user_org(tenant_id=tenant_id, role=ROLE_CONTRIBUTOR)


@pytest_asyncio.fixture
async def random_organization(
    make_user_org: Callable[..., Awaitable[tuple[int, int]]],
) -> tuple[int, int]:
    """Fresh owner in a freshly-minted workspace.

    Distinct from ``other_org_admin_user`` only in name; both seed a
    new tenant without sharing state with ``admin_user``.
    """
    return await make_user_org()


@pytest_asyncio.fixture
async def authed_client(
    app: FastAPI,
    admin_user: tuple[int, int],
) -> AsyncIterator[TestClient]:
    """A ``TestClient`` that authenticates via the ``x-knowledge-*`` headers.

    The headers are sourced from the freshly-minted ``admin_user`` so
    every test resolves a real ``User`` row + a real ``Tenant`` row
    created by ``make_user_org``. The ``with`` block is required:
    without it asyncpg raises a cross-loop ``RuntimeError`` because
    TestClient runs requests in its own portal loop.
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
    "coworker_admin_user",
    "faker_seed",
    "make_test_tenant_id",
    "make_user_org",
    "other_org_admin_user",
    "random_organization",
    "session",
    "setup_organization_and_regular_user",
]
