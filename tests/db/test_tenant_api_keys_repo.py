"""Integration tests for `TenantAPIKeyRepository` against a real Postgres.

The session-scoped `pg_url` fixture (tests/conftest.py) provides the
container; the DDL mirrors `alembic/versions/0004_tenant_api_keys.py`
(plus the `tenants` parent needed by the foreign key).

The fixture skips the suite when no Docker daemon is available.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
import sqlalchemy
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.common.exception import NotFoundError
from src.db.dao.tenant_api_keys_repository import TenantAPIKeyRepository
from src.db.models.tenants.tenant_api_keys import TenantAPIKey

_DROP_SQL = sqlalchemy.text("DROP TABLE IF EXISTS tenant_api_keys, tenants CASCADE")

_CREATE_TENANTS_SQL = sqlalchemy.text(
    """
    CREATE TABLE tenants (
        id BIGSERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        deleted_at TIMESTAMP WITH TIME ZONE
    )
    """
)

_CREATE_API_KEYS_SQL = sqlalchemy.text(
    """
    CREATE TABLE tenant_api_keys (
        id BIGSERIAL PRIMARY KEY,
        tenant_id BIGINT REFERENCES tenants(id) ON DELETE CASCADE,
        scope_type VARCHAR(16) NOT NULL DEFAULT 'tenant',
        name VARCHAR(128) NOT NULL,
        key_hash VARCHAR(64) NOT NULL UNIQUE,
        api_key TEXT NOT NULL DEFAULT '',
        full_access BOOLEAN NOT NULL DEFAULT FALSE,
        knowledge_base_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
        capabilities JSONB NOT NULL DEFAULT '[]'::jsonb,
        last_used_at TIMESTAMP WITH TIME ZONE,
        expires_at TIMESTAMP WITH TIME ZONE,
        revoked_at TIMESTAMP WITH TIME ZONE,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT chk_tenant_api_keys_scope CHECK (
            (scope_type = 'tenant' AND tenant_id IS NOT NULL)
            OR (scope_type = 'platform' AND tenant_id IS NULL AND full_access = FALSE)
        )
    )
    """
)

_SEED_TENANT_SQL = sqlalchemy.text(
    "INSERT INTO tenants (name, created_at, updated_at) "
    "VALUES ('acme', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) RETURNING id"
)

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
async def session(pg_url: str) -> AsyncIterator[AsyncSession]:
    engine: AsyncEngine = create_async_engine(pg_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.execute(_DROP_SQL)
        await conn.execute(_CREATE_TENANTS_SQL)
        await conn.execute(_CREATE_API_KEYS_SQL)
    async with factory() as s:
        yield s
    async with engine.begin() as conn:
        await conn.execute(_DROP_SQL)
    await engine.dispose()


@pytest.fixture
async def tenant_id(session: AsyncSession) -> int:
    row = (await session.execute(_SEED_TENANT_SQL)).scalar_one()
    await session.commit()
    return int(row)


def _sample_key(
    *,
    tenant_id: int | None,
    key_hash: str = "hash-1",
    name: str = "ci",
    scope_type: str = "tenant",
    created_at: datetime = _NOW,
    **columns: object,
) -> TenantAPIKey:
    return TenantAPIKey.model_validate(
        {
            "tenant_id": tenant_id,
            "scope_type": scope_type,
            "name": name,
            "key_hash": key_hash,
            "created_at": created_at,
            "updated_at": created_at,
            **columns,
        }
    )


async def _insert(
    repo: TenantAPIKeyRepository,
    session: AsyncSession,
    row: TenantAPIKey,
) -> TenantAPIKey:
    stored = await repo.insert(row)
    await session.commit()
    return stored


# ── insert / find_by_hash ───────────────────────────────────────────


async def test_insert_assigns_id_and_round_trips_json_columns(
    session: AsyncSession,
    tenant_id: int,
) -> None:
    repo = TenantAPIKeyRepository(session)

    stored = await _insert(
        repo,
        session,
        _sample_key(
            tenant_id=tenant_id,
            knowledge_base_ids=["kb-1", "kb-2"],
            capabilities=["chat"],
        ),
    )

    assert stored.id > 0
    found = await repo.find_by_hash("hash-1")
    assert found.knowledge_base_ids == ["kb-1", "kb-2"]
    assert found.capabilities == ["chat"]


async def test_find_by_hash_missing_raises(session: AsyncSession) -> None:
    repo = TenantAPIKeyRepository(session)

    with pytest.raises(NotFoundError) as excinfo:
        await repo.find_by_hash("nope")

    assert excinfo.value.code == "tenant_api_key.not_found"


async def test_find_by_hash_skips_revoked_key(
    session: AsyncSession,
    tenant_id: int,
) -> None:
    repo = TenantAPIKeyRepository(session)
    stored = await _insert(repo, session, _sample_key(tenant_id=tenant_id))
    await repo.revoke(stored.id, tenant_id=tenant_id, revoked_at=_NOW)
    await session.commit()

    with pytest.raises(NotFoundError):
        await repo.find_by_hash("hash-1")


async def test_platform_key_persists_without_a_tenant(session: AsyncSession) -> None:
    repo = TenantAPIKeyRepository(session)

    stored = await _insert(
        repo,
        session,
        _sample_key(tenant_id=None, scope_type="platform", capabilities=["system_audit_read"]),
    )

    assert stored.tenant_id is None
    assert stored.scope_type == "platform"


# ── listing ─────────────────────────────────────────────────────────


async def test_list_for_tenant_returns_live_keys_newest_first(
    session: AsyncSession,
    tenant_id: int,
) -> None:
    repo = TenantAPIKeyRepository(session)
    older = await _insert(
        repo, session, _sample_key(tenant_id=tenant_id, key_hash="h1", created_at=_NOW)
    )
    newer = await _insert(
        repo,
        session,
        _sample_key(tenant_id=tenant_id, key_hash="h2", created_at=_NOW + timedelta(days=1)),
    )

    keys = await repo.list_for_tenant(tenant_id)

    assert [k.id for k in keys] == [newer.id, older.id]


async def test_list_for_tenant_excludes_revoked(
    session: AsyncSession,
    tenant_id: int,
) -> None:
    repo = TenantAPIKeyRepository(session)
    live = await _insert(repo, session, _sample_key(tenant_id=tenant_id, key_hash="h1"))
    gone = await _insert(repo, session, _sample_key(tenant_id=tenant_id, key_hash="h2"))
    await repo.revoke(gone.id, tenant_id=tenant_id, revoked_at=_NOW)
    await session.commit()

    keys = await repo.list_for_tenant(tenant_id)

    assert [k.id for k in keys] == [live.id]


async def test_list_platform_filters_by_scope(
    session: AsyncSession,
    tenant_id: int,
) -> None:
    repo = TenantAPIKeyRepository(session)
    await _insert(repo, session, _sample_key(tenant_id=tenant_id, key_hash="h1"))
    platform = await _insert(
        repo,
        session,
        _sample_key(
            tenant_id=None,
            key_hash="h2",
            scope_type="platform",
            capabilities=["system_audit_read"],
        ),
    )

    keys = await repo.list_platform()

    assert [k.id for k in keys] == [platform.id]


# ── revoke ──────────────────────────────────────────────────────────


async def test_revoke_stamps_revoked_at(session: AsyncSession, tenant_id: int) -> None:
    repo = TenantAPIKeyRepository(session)
    stored = await _insert(repo, session, _sample_key(tenant_id=tenant_id))

    await repo.revoke(stored.id, tenant_id=tenant_id, revoked_at=_NOW)
    await session.commit()

    row = await repo.find_by_primary_key({"id": stored.id})
    assert row is not None
    assert row.revoked_at == _NOW


async def test_revoke_rejects_another_tenants_key(
    session: AsyncSession,
    tenant_id: int,
) -> None:
    repo = TenantAPIKeyRepository(session)
    stored = await _insert(repo, session, _sample_key(tenant_id=tenant_id))

    with pytest.raises(NotFoundError):
        await repo.revoke(stored.id, tenant_id=tenant_id + 999, revoked_at=_NOW)


async def test_revoke_platform_rejects_a_tenant_key(
    session: AsyncSession,
    tenant_id: int,
) -> None:
    repo = TenantAPIKeyRepository(session)
    stored = await _insert(repo, session, _sample_key(tenant_id=tenant_id))

    with pytest.raises(NotFoundError):
        await repo.revoke_platform(stored.id, revoked_at=_NOW)


# ── touch_last_used / update_hash / placeholder queries ─────────────


async def test_touch_last_used_updates_only_live_keys(
    session: AsyncSession,
    tenant_id: int,
) -> None:
    repo = TenantAPIKeyRepository(session)
    stored = await _insert(repo, session, _sample_key(tenant_id=tenant_id))

    affected = await repo.touch_last_used(stored.id, used_at=_NOW)
    await session.commit()

    assert affected == 1
    assert (await repo.find_by_hash("hash-1")).last_used_at == _NOW


async def test_touch_last_used_ignores_revoked_keys(
    session: AsyncSession,
    tenant_id: int,
) -> None:
    repo = TenantAPIKeyRepository(session)
    stored = await _insert(repo, session, _sample_key(tenant_id=tenant_id))
    await repo.revoke(stored.id, tenant_id=tenant_id, revoked_at=_NOW)

    assert await repo.touch_last_used(stored.id, used_at=_NOW) == 0


async def test_update_hash_replaces_the_lookup_value(
    session: AsyncSession,
    tenant_id: int,
) -> None:
    repo = TenantAPIKeyRepository(session)
    stored = await _insert(repo, session, _sample_key(tenant_id=tenant_id))

    await repo.update_hash(stored.id, key_hash="fresh-hash")
    await session.commit()

    assert (await repo.find_by_hash("fresh-hash")).id == stored.id


async def test_placeholder_queries_find_migrated_rows(
    session: AsyncSession,
    tenant_id: int,
) -> None:
    repo = TenantAPIKeyRepository(session)
    legacy = await _insert(
        repo,
        session,
        _sample_key(tenant_id=tenant_id, key_hash=f"migrated-tenant-{tenant_id}"),
    )
    await _insert(repo, session, _sample_key(tenant_id=tenant_id, key_hash="real-hash"))

    assert await repo.has_placeholder_hash() is True
    assert [k.id for k in await repo.list_with_placeholder_hash()] == [legacy.id]


async def test_has_placeholder_hash_is_false_without_migrated_rows(
    session: AsyncSession,
    tenant_id: int,
) -> None:
    repo = TenantAPIKeyRepository(session)
    await _insert(repo, session, _sample_key(tenant_id=tenant_id, key_hash="real-hash"))

    assert await repo.has_placeholder_hash() is False
