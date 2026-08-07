"""Integration tests for ``TenantAPIKeyRepository`` against the real applied schema.

Tests insert unique rows per run; isolation relies on unique key hashes and
tenant ids. Tests commit explicitly.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
import sqlalchemy
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.exception import NotFoundError
from src.db.dao.tenant_api_keys_repository import TenantAPIKeyRepository
from src.db.models.tenants.tenant_api_keys import TenantAPIKey
from tests.integration.db.dao.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 1, tzinfo=UTC)

_SEED_TENANT_SQL = sqlalchemy.text(
    "INSERT INTO tenants (name, created_at, updated_at) "
    "VALUES (:name, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) RETURNING id"
)


def _hash() -> str:
    return f"hash-{uuid.uuid4().hex[:16]}"


@pytest.fixture
async def tenant_id(session: AsyncSession) -> int:
    name = f"apikey-tenant-{uuid.uuid4().hex[:8]}"
    row = (await session.execute(_SEED_TENANT_SQL.bindparams(name=name))).scalar_one()
    await session.commit()
    return int(row)


def _sample_key(
    *,
    tenant_id: int | None,
    key_hash: str | None = None,
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
            "key_hash": key_hash or _hash(),
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
    kh = _hash()

    stored = await _insert(
        repo,
        session,
        _sample_key(
            tenant_id=tenant_id,
            key_hash=kh,
            knowledge_base_ids=["kb-1", "kb-2"],
            capabilities=["chat"],
        ),
    )

    assert stored.id > 0
    found = await repo.find_by_hash(kh)
    assert found.knowledge_base_ids == ["kb-1", "kb-2"]
    assert found.capabilities == ["chat"]


async def test_find_by_hash_missing_raises(session: AsyncSession) -> None:
    repo = TenantAPIKeyRepository(session)

    with pytest.raises(NotFoundError) as excinfo:
        await repo.find_by_hash(_hash())

    assert excinfo.value.code == "tenant_api_key.not_found"


async def test_find_by_hash_skips_revoked_key(
    session: AsyncSession,
    tenant_id: int,
) -> None:
    repo = TenantAPIKeyRepository(session)
    kh = _hash()
    stored = await _insert(repo, session, _sample_key(tenant_id=tenant_id, key_hash=kh))
    await repo.revoke(stored.id, tenant_id=tenant_id, revoked_at=_NOW)
    await session.commit()

    with pytest.raises(NotFoundError):
        await repo.find_by_hash(kh)


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
        repo, session, _sample_key(tenant_id=tenant_id, key_hash=_hash(), created_at=_NOW)
    )
    newer = await _insert(
        repo,
        session,
        _sample_key(tenant_id=tenant_id, key_hash=_hash(), created_at=_NOW + timedelta(days=1)),
    )

    keys = await repo.list_for_tenant(tenant_id)

    assert [k.id for k in keys] == [newer.id, older.id]


async def test_list_for_tenant_excludes_revoked(
    session: AsyncSession,
    tenant_id: int,
) -> None:
    repo = TenantAPIKeyRepository(session)
    live = await _insert(repo, session, _sample_key(tenant_id=tenant_id, key_hash=_hash()))
    gone = await _insert(repo, session, _sample_key(tenant_id=tenant_id, key_hash=_hash()))
    await repo.revoke(gone.id, tenant_id=tenant_id, revoked_at=_NOW)
    await session.commit()

    keys = await repo.list_for_tenant(tenant_id)

    assert [k.id for k in keys] == [live.id]


async def test_list_platform_filters_by_scope(
    session: AsyncSession,
    tenant_id: int,
) -> None:
    repo = TenantAPIKeyRepository(session)
    await _insert(repo, session, _sample_key(tenant_id=tenant_id, key_hash=_hash()))
    platform = await _insert(
        repo,
        session,
        _sample_key(
            tenant_id=None,
            key_hash=_hash(),
            scope_type="platform",
            capabilities=["system_audit_read"],
        ),
    )

    keys = await repo.list_platform()

    assert all(k.scope_type == "platform" for k in keys)
    assert platform.id in {k.id for k in keys}


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
        await repo.revoke(stored.id, tenant_id=make_test_tenant_id(), revoked_at=_NOW)


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
    kh = _hash()
    stored = await _insert(repo, session, _sample_key(tenant_id=tenant_id, key_hash=kh))

    affected = await repo.touch_last_used(stored.id, used_at=_NOW)
    await session.commit()

    assert affected == 1
    assert (await repo.find_by_hash(kh)).last_used_at == _NOW


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
    new_hash = _hash()

    await repo.update_hash(stored.id, key_hash=new_hash)
    await session.commit()

    assert (await repo.find_by_hash(new_hash)).id == stored.id


async def test_placeholder_queries_find_migrated_rows(
    session: AsyncSession,
    tenant_id: int,
) -> None:
    repo = TenantAPIKeyRepository(session)
    migrated_hash = f"migrated-tenant-{uuid.uuid4().hex[:8]}"
    legacy = await _insert(
        repo,
        session,
        _sample_key(tenant_id=tenant_id, key_hash=migrated_hash),
    )
    await _insert(repo, session, _sample_key(tenant_id=tenant_id, key_hash=_hash()))

    assert await repo.has_placeholder_hash() is True
    placeholder_keys = await repo.list_with_placeholder_hash()
    assert legacy.id in {k.id for k in placeholder_keys}


async def test_has_placeholder_hash_is_false_without_migrated_rows(
    session: AsyncSession,
    tenant_id: int,
) -> None:
    repo = TenantAPIKeyRepository(session)
    key = await _insert(repo, session, _sample_key(tenant_id=tenant_id, key_hash=_hash()))

    placeholder_keys = await repo.list_with_placeholder_hash()
    assert key.id not in {k.id for k in placeholder_keys}


# ── tenant isolation ────────────────────────────────────────────────


_SEED_TENANT_WITH_ID_SQL = sqlalchemy.text(
    "INSERT INTO tenants (id, name) VALUES (:id, :name) ON CONFLICT (id) DO NOTHING"
)


async def test_list_for_tenant_isolated_by_tenant(session: AsyncSession) -> None:
    repo = TenantAPIKeyRepository(session)
    tid_a = make_test_tenant_id()
    tid_b = make_test_tenant_id()
    for tid in (tid_a, tid_b):
        await session.execute(_SEED_TENANT_WITH_ID_SQL.bindparams(id=tid, name=f"iso-{tid}"))
    await session.commit()

    key_a = await _insert(repo, session, _sample_key(tenant_id=tid_a))
    key_b = await _insert(repo, session, _sample_key(tenant_id=tid_b))

    keys_a = await repo.list_for_tenant(tid_a)
    keys_b = await repo.list_for_tenant(tid_b)

    assert [k.id for k in keys_a] == [key_a.id]
    assert [k.id for k in keys_b] == [key_b.id]
