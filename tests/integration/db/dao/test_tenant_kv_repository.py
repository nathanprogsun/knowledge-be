"""Integration tests for ``TenantKVRepository`` against the real applied schema.

Tests insert unique rows per run; isolation relies on unique tenant ids and
keys. Tests commit explicitly.
"""

from __future__ import annotations

import uuid

import sqlalchemy
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.dao.tenant_kv_repository import TenantKVRepository
from tests.integration.db.dao.conftest import make_test_tenant_id

_SEED_TENANT_SQL = sqlalchemy.text(
    "INSERT INTO tenants (id, name) VALUES (:id, :name) ON CONFLICT (id) DO NOTHING"
)


async def _insert_tenant(session: AsyncSession, tenant_id: int) -> None:
    await session.execute(_SEED_TENANT_SQL.bindparams(id=tenant_id, name=f"kv-tenant-{tenant_id}"))
    await session.commit()


async def test_upsert_then_read(session: AsyncSession) -> None:
    tid = make_test_tenant_id()
    await _insert_tenant(session, tid)
    repo = TenantKVRepository(session)
    key = f"web-search-config-{uuid.uuid4().hex[:8]}"
    row = await repo.upsert(tenant_id=tid, key=key, value={"max_results": 20})
    assert row.key == key
    assert row.value == {"max_results": 20}
    await session.commit()

    found = await repo.find_value(tenant_id=tid, key=key)
    assert found is not None
    assert found.value == {"max_results": 20}


async def test_upsert_overwrites_existing(session: AsyncSession) -> None:
    tid = make_test_tenant_id()
    await _insert_tenant(session, tid)
    repo = TenantKVRepository(session)
    key = f"k-{uuid.uuid4().hex[:8]}"
    await repo.upsert(tenant_id=tid, key=key, value={"a": 1})
    await session.commit()
    await repo.upsert(tenant_id=tid, key=key, value={"a": 2})
    await session.commit()

    found = await repo.find_value(tenant_id=tid, key=key)
    assert found is not None
    assert found.value == {"a": 2}


async def test_find_value_missing_returns_none(session: AsyncSession) -> None:
    tid = make_test_tenant_id()
    await _insert_tenant(session, tid)
    repo = TenantKVRepository(session)
    assert await repo.find_value(tenant_id=tid, key=f"missing-{uuid.uuid4().hex[:8]}") is None


async def test_delete_marks_soft_deleted(session: AsyncSession) -> None:
    tid = make_test_tenant_id()
    await _insert_tenant(session, tid)
    repo = TenantKVRepository(session)
    key = f"k-{uuid.uuid4().hex[:8]}"
    await repo.upsert(tenant_id=tid, key=key, value={"x": 1})
    await session.commit()

    deleted = await repo.delete(tenant_id=tid, key=key)
    assert deleted is True
    await session.commit()
    assert await repo.find_value(tenant_id=tid, key=key) is None

    assert await repo.delete(tenant_id=tid, key=key) is False


# ── tenant isolation ────────────────────────────────────────────────


async def test_find_value_isolated_by_tenant(session: AsyncSession) -> None:
    tid_a = make_test_tenant_id()
    tid_b = make_test_tenant_id()
    await _insert_tenant(session, tid_a)
    await _insert_tenant(session, tid_b)
    repo = TenantKVRepository(session)
    key = f"shared-key-{uuid.uuid4().hex[:8]}"
    await repo.upsert(tenant_id=tid_a, key=key, value={"owner": "a"})
    await session.commit()

    assert await repo.find_value(tenant_id=tid_a, key=key) is not None
    assert await repo.find_value(tenant_id=tid_b, key=key) is None
