"""Integration tests for `TenantKVRepository` against a real Postgres.

The `pg_url` fixture (tests/conftest.py) provides the container; each test
builds a fresh `tenant_kv` schema (plus a minimal `tenants` parent table
for the FK) so writes are hermetic. DDL mirrors
`alembic/versions/0008_tenant_kv.py`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import sqlalchemy
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.db.dao.tenant_kv_repository import TenantKVRepository

_DROP_KV_SQL = sqlalchemy.text("DROP TABLE IF EXISTS tenant_kv")
_DROP_TENANTS_SQL = sqlalchemy.text("DROP TABLE IF EXISTS tenants")

_CREATE_TENANTS_SQL = sqlalchemy.text(
    """
    CREATE TABLE tenants (
        id BIGSERIAL PRIMARY KEY,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """
)

_CREATE_KV_SQL = sqlalchemy.text(
    """
    CREATE TABLE tenant_kv (
        id BIGSERIAL PRIMARY KEY,
        tenant_id BIGINT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
        key VARCHAR(128) NOT NULL,
        value JSONB NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        deleted_at TIMESTAMP WITH TIME ZONE,
        CONSTRAINT uq_tenant_kv_tenant_key_live UNIQUE (tenant_id, key)
    )
    """
)

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
async def session(pg_url: str) -> AsyncIterator[AsyncSession]:
    engine: AsyncEngine = create_async_engine(pg_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.execute(_DROP_KV_SQL)
        await conn.execute(_DROP_TENANTS_SQL)
        await conn.execute(_CREATE_TENANTS_SQL)
        await conn.execute(_CREATE_KV_SQL)
    async with factory() as s:
        yield s
    async with engine.begin() as conn:
        await conn.execute(_DROP_KV_SQL)
        await conn.execute(_DROP_TENANTS_SQL)
    await engine.dispose()


async def _insert_tenant(session: AsyncSession, tenant_id: int) -> None:
    await session.execute(
        sqlalchemy.text("insert into tenants (id) values (:id)"),
        {"id": tenant_id},
    )
    await session.commit()


async def test_upsert_then_read(session: AsyncSession) -> None:
    await _insert_tenant(session, 7)
    repo = TenantKVRepository(session)
    row = await repo.upsert(tenant_id=7, key="web-search-config", value={"max_results": 20})
    assert row.key == "web-search-config"
    assert row.value == {"max_results": 20}
    await session.commit()

    found = await repo.find_value(tenant_id=7, key="web-search-config")
    assert found is not None
    assert found.value == {"max_results": 20}


async def test_upsert_overwrites_existing(session: AsyncSession) -> None:
    await _insert_tenant(session, 7)
    repo = TenantKVRepository(session)
    await repo.upsert(tenant_id=7, key="k", value={"a": 1})
    await session.commit()
    await repo.upsert(tenant_id=7, key="k", value={"a": 2})
    await session.commit()

    found = await repo.find_value(tenant_id=7, key="k")
    assert found is not None
    assert found.value == {"a": 2}


async def test_find_value_missing_returns_none(session: AsyncSession) -> None:
    await _insert_tenant(session, 7)
    repo = TenantKVRepository(session)
    assert await repo.find_value(tenant_id=7, key="missing") is None


async def test_delete_marks_soft_deleted(session: AsyncSession) -> None:
    await _insert_tenant(session, 7)
    repo = TenantKVRepository(session)
    await repo.upsert(tenant_id=7, key="k", value={"x": 1})
    await session.commit()

    deleted = await repo.delete(tenant_id=7, key="k")
    assert deleted is True
    await session.commit()
    assert await repo.find_value(tenant_id=7, key="k") is None

    # Second delete is idempotent.
    assert await repo.delete(tenant_id=7, key="k") is False


__all__ = []
