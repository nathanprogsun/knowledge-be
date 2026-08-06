"""Integration tests for `TenantRepository` against a real Postgres.

The session-scoped `pg_url` fixture (tests/conftest.py) provides the
container; each test gets a fresh `tenants` schema on top of it so
writes are hermetic. The DDL mirrors `alembic/versions/0003_tenants.py`.

The fixture skips the suite when no Docker daemon is available (CI
runners without Docker, sandboxes).
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
from src.common.json import JsonObject
from src.db.dao.tenants_repository import TenantRepository, escape_like_keyword
from src.db.models.tenants.tenants import DEFAULT_STORAGE_QUOTA_BYTES, Tenant

_DROP_TENANTS_SQL = sqlalchemy.text("DROP TABLE IF EXISTS tenants")

_CREATE_TENANTS_SQL = sqlalchemy.text(
    """
    CREATE TABLE tenants (
        id BIGSERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        description TEXT,
        retriever_engines JSONB NOT NULL DEFAULT '{"engines": []}'::jsonb,
        status VARCHAR(50) NOT NULL DEFAULT 'active',
        business VARCHAR(255) NOT NULL DEFAULT '',
        storage_quota BIGINT NOT NULL DEFAULT 10737418240,
        storage_used BIGINT NOT NULL DEFAULT 0,
        agent_config JSONB,
        context_config JSONB,
        conversation_config JSONB,
        web_search_config JSONB,
        parser_engine_config JSONB,
        storage_engine_config JSONB,
        default_storage_backend_id VARCHAR(36),
        credentials JSONB,
        chat_history_config JSONB,
        retrieval_config JSONB,
        api_principal_config JSONB,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        deleted_at TIMESTAMP WITH TIME ZONE
    )
    """
)

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
async def session(pg_url: str) -> AsyncIterator[AsyncSession]:
    engine: AsyncEngine = create_async_engine(pg_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.execute(_DROP_TENANTS_SQL)
        await conn.execute(_CREATE_TENANTS_SQL)
    async with factory() as s:
        yield s
    async with engine.begin() as conn:
        await conn.execute(_DROP_TENANTS_SQL)
    await engine.dispose()


def _sample_row(
    *,
    name: str = "acme",
    description: str | None = "acme workspace",
    business: str = "saas",
    created_at: datetime = _NOW,
    retriever_engines: JsonObject | list[JsonObject] | None = None,
) -> Tenant:
    return Tenant(
        name=name,
        description=description,
        business=business,
        retriever_engines=retriever_engines if retriever_engines is not None else {"engines": []},
        created_at=created_at,
        updated_at=created_at,
    )


async def _insert(repo: TenantRepository, session: AsyncSession, row: Tenant) -> Tenant:
    stored = await repo.insert(row)
    await session.commit()
    return stored


# ── insert ──────────────────────────────────────────────────────────


async def test_insert_assigns_database_generated_id(session: AsyncSession) -> None:
    repo = TenantRepository(session)

    stored = await _insert(repo, session, _sample_row())

    assert stored.id > 0
    assert stored.name == "acme"
    assert stored.status == "active"
    assert stored.storage_quota == DEFAULT_STORAGE_QUOTA_BYTES
    assert stored.storage_used == 0


async def test_insert_ids_are_distinct_per_row(session: AsyncSession) -> None:
    repo = TenantRepository(session)

    first = await _insert(repo, session, _sample_row(name="a"))
    second = await _insert(repo, session, _sample_row(name="b"))

    assert first.id != second.id


async def test_insert_round_trips_retriever_engines_object(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    engines: JsonObject = {
        "engines": [{"retriever_type": "keywords", "retriever_engine_type": "postgres"}]
    }

    stored = await _insert(repo, session, _sample_row(retriever_engines=engines))

    found = await repo.find_by_id(stored.id)
    assert found.retriever_engines == engines


async def test_insert_round_trips_legacy_retriever_engines_array(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    legacy: list[JsonObject] = [{"retriever_type": "vector", "retriever_engine_type": "postgres"}]

    stored = await _insert(repo, session, _sample_row(retriever_engines=legacy))

    found = await repo.find_by_id(stored.id)
    assert found.retriever_engines == legacy


# ── find_by_id ──────────────────────────────────────────────────────


async def test_find_by_id_returns_row(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    stored = await _insert(repo, session, _sample_row())

    found = await repo.find_by_id(stored.id)

    assert found.id == stored.id
    assert found.description == "acme workspace"


async def test_find_by_id_missing_raises_tenant_not_found(session: AsyncSession) -> None:
    repo = TenantRepository(session)

    with pytest.raises(NotFoundError) as excinfo:
        await repo.find_by_id(4242)

    assert excinfo.value.code == "tenant.not_found"


async def test_find_by_id_skips_soft_deleted_row(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    stored = await _insert(repo, session, _sample_row())
    await repo.update_by_primary_key({"id": stored.id}, {"deleted_at": _NOW})
    await session.commit()

    with pytest.raises(NotFoundError):
        await repo.find_by_id(stored.id)


# ── find_by_ids ─────────────────────────────────────────────────────


async def test_find_by_ids_empty_input_returns_empty_list(session: AsyncSession) -> None:
    repo = TenantRepository(session)

    assert await repo.find_by_ids([]) == []


async def test_find_by_ids_returns_matching_rows_newest_first(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    older = await _insert(repo, session, _sample_row(name="older", created_at=_NOW))
    newer = await _insert(
        repo, session, _sample_row(name="newer", created_at=_NOW + timedelta(days=1))
    )
    await _insert(repo, session, _sample_row(name="unwanted"))

    found = await repo.find_by_ids([older.id, newer.id])

    assert [t.id for t in found] == [newer.id, older.id]


async def test_find_by_ids_ignores_unknown_and_deleted_ids(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    live = await _insert(repo, session, _sample_row(name="live"))
    deleted = await _insert(repo, session, _sample_row(name="deleted"))
    await repo.update_by_primary_key({"id": deleted.id}, {"deleted_at": _NOW})
    await session.commit()

    found = await repo.find_by_ids([live.id, deleted.id, 999_999])

    assert [t.id for t in found] == [live.id]


# ── list ────────────────────────────────────────────────────────────


async def test_list_returns_all_rows_newest_first(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    first = await _insert(repo, session, _sample_row(name="first", created_at=_NOW))
    second = await _insert(
        repo, session, _sample_row(name="second", created_at=_NOW + timedelta(hours=1))
    )

    rows = await repo.list_all()

    assert [t.id for t in rows] == [second.id, first.id]


async def test_list_is_not_capped_by_a_default_page_size(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    for index in range(105):
        await repo.insert(_sample_row(name=f"tenant-{index}"))
    await session.commit()

    rows = await repo.list_all()

    assert len(rows) == 105


async def test_list_applies_explicit_pagination(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    for index in range(5):
        await repo.insert(_sample_row(name=f"t{index}", created_at=_NOW + timedelta(hours=index)))
    await session.commit()

    rows = await repo.list_all(limit=2, offset=1)

    assert [t.name for t in rows] == ["t3", "t2"]


async def test_list_excludes_soft_deleted_rows(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    live = await _insert(repo, session, _sample_row(name="live"))
    gone = await _insert(repo, session, _sample_row(name="gone"))
    await repo.update_by_primary_key({"id": gone.id}, {"deleted_at": _NOW})
    await session.commit()

    rows = await repo.list_all()

    assert [t.id for t in rows] == [live.id]


# ── search ──────────────────────────────────────────────────────────


async def test_search_without_filters_returns_everything(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    await _insert(repo, session, _sample_row(name="alpha"))
    await _insert(repo, session, _sample_row(name="beta"))

    rows, total = await repo.search()

    assert total == 2
    assert len(rows) == 2


async def test_search_matches_keyword_in_name(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    match = await _insert(repo, session, _sample_row(name="alpha corp", description=None))
    await _insert(repo, session, _sample_row(name="beta corp", description=None))

    rows, total = await repo.search(keyword="alpha")

    assert total == 1
    assert [t.id for t in rows] == [match.id]


async def test_search_matches_keyword_in_description(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    match = await _insert(repo, session, _sample_row(name="x", description="research team"))
    await _insert(repo, session, _sample_row(name="y", description="sales team"))

    rows, total = await repo.search(keyword="research")

    assert total == 1
    assert [t.id for t in rows] == [match.id]


async def test_search_treats_wildcards_in_keyword_literally(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    literal = await _insert(repo, session, _sample_row(name="100% owned", description=None))
    await _insert(repo, session, _sample_row(name="anything", description=None))

    rows, total = await repo.search(keyword="100%")

    assert total == 1
    assert [t.id for t in rows] == [literal.id]


async def test_search_filters_by_tenant_id(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    wanted = await _insert(repo, session, _sample_row(name="wanted"))
    await _insert(repo, session, _sample_row(name="other"))

    rows, total = await repo.search(tenant_id=wanted.id)

    assert total == 1
    assert [t.id for t in rows] == [wanted.id]


async def test_search_combines_id_and_keyword_with_or(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    by_id = await _insert(repo, session, _sample_row(name="unrelated", description=None))
    by_keyword = await _insert(repo, session, _sample_row(name="alpha", description=None))
    await _insert(repo, session, _sample_row(name="excluded", description=None))

    rows, total = await repo.search(keyword="alpha", tenant_id=by_id.id)

    assert total == 2
    assert {t.id for t in rows} == {by_id.id, by_keyword.id}


async def test_search_total_ignores_pagination(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    for index in range(5):
        await repo.insert(_sample_row(name=f"t{index}", created_at=_NOW + timedelta(hours=index)))
    await session.commit()

    rows, total = await repo.search(limit=2, offset=0)

    assert total == 5
    assert [t.name for t in rows] == ["t4", "t3"]


async def test_search_excludes_soft_deleted_rows(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    gone = await _insert(repo, session, _sample_row(name="alpha"))
    await repo.update_by_primary_key({"id": gone.id}, {"deleted_at": _NOW})
    await session.commit()

    rows, total = await repo.search(keyword="alpha")

    assert total == 0
    assert rows == []


# ── adjust_storage_used ─────────────────────────────────────────────


async def test_adjust_storage_used_adds_delta(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    stored = await _insert(repo, session, _sample_row())

    used = await repo.adjust_storage_used(stored.id, delta=1024, updated_at=_NOW)
    await session.commit()

    assert used == 1024
    assert (await repo.find_by_id(stored.id)).storage_used == 1024


async def test_adjust_storage_used_clamps_negative_result_to_zero(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    stored = await _insert(repo, session, _sample_row())
    await repo.adjust_storage_used(stored.id, delta=100, updated_at=_NOW)

    used = await repo.adjust_storage_used(stored.id, delta=-500, updated_at=_NOW)
    await session.commit()

    assert used == 0


async def test_adjust_storage_used_missing_tenant_raises(session: AsyncSession) -> None:
    repo = TenantRepository(session)

    with pytest.raises(NotFoundError) as excinfo:
        await repo.adjust_storage_used(4242, delta=1, updated_at=_NOW)

    assert excinfo.value.code == "tenant.not_found"


async def test_adjust_storage_used_stamps_updated_at(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    stored = await _insert(repo, session, _sample_row())
    later = _NOW + timedelta(days=2)

    await repo.adjust_storage_used(stored.id, delta=1, updated_at=later)
    await session.commit()

    assert (await repo.find_by_id(stored.id)).updated_at == later


# ── bulk_set_storage_quota ──────────────────────────────────────────


async def test_bulk_set_storage_quota_updates_every_live_tenant(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    first = await _insert(repo, session, _sample_row(name="a"))
    second = await _insert(repo, session, _sample_row(name="b"))

    affected = await repo.bulk_set_storage_quota(quota_bytes=2048, updated_at=_NOW)
    await session.commit()

    assert affected == 2
    assert (await repo.find_by_id(first.id)).storage_quota == 2048
    assert (await repo.find_by_id(second.id)).storage_quota == 2048


async def test_bulk_set_storage_quota_skips_soft_deleted_tenants(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    await _insert(repo, session, _sample_row(name="live"))
    gone = await _insert(repo, session, _sample_row(name="gone"))
    await repo.update_by_primary_key({"id": gone.id}, {"deleted_at": _NOW})
    await session.commit()

    affected = await repo.bulk_set_storage_quota(quota_bytes=2048, updated_at=_NOW)
    await session.commit()

    assert affected == 1


# ── escape_like_keyword ─────────────────────────────────────────────


def test_escape_like_keyword_escapes_wildcards_and_backslash() -> None:
    assert escape_like_keyword(r"50%_a\b") == r"50\%\_a\\b"


def test_escape_like_keyword_leaves_plain_text_untouched() -> None:
    assert escape_like_keyword("acme") == "acme"
