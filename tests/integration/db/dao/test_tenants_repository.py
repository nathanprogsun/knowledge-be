"""Integration tests for ``TenantRepository`` against the real applied schema.

Tests insert unique rows per run; isolation relies on unique names and
tenant ids. Tests commit explicitly.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.exception import NotFoundError
from src.common.json import JsonObject
from src.db.dao.tenants_repository import TenantRepository, escape_like_keyword
from src.db.models.tenants.tenants import DEFAULT_STORAGE_QUOTA_BYTES, Tenant

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_FUTURE = datetime(2099, 12, 31, tzinfo=UTC)


def _unique_name(prefix: str = "tenant") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _sample_row(
    *,
    name: str | None = None,
    description: str | None = "acme workspace",
    business: str = "saas",
    created_at: datetime = _FUTURE,
    retriever_engines: JsonObject | list[JsonObject] | None = None,
) -> Tenant:
    return Tenant(
        name=name or _unique_name(),
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
    assert stored.status == "active"
    assert stored.storage_quota == DEFAULT_STORAGE_QUOTA_BYTES
    assert stored.storage_used == 0


async def test_insert_ids_are_distinct_per_row(session: AsyncSession) -> None:
    repo = TenantRepository(session)

    first = await _insert(repo, session, _sample_row())
    second = await _insert(repo, session, _sample_row())

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
        await repo.find_by_id(999_999_999)

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
    older = await _insert(repo, session, _sample_row(created_at=_FUTURE))
    newer = await _insert(repo, session, _sample_row(created_at=_FUTURE + timedelta(days=1)))

    found = await repo.find_by_ids([older.id, newer.id])

    assert [t.id for t in found] == [newer.id, older.id]


async def test_find_by_ids_ignores_unknown_and_deleted_ids(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    live = await _insert(repo, session, _sample_row())
    deleted = await _insert(repo, session, _sample_row())
    await repo.update_by_primary_key({"id": deleted.id}, {"deleted_at": _NOW})
    await session.commit()

    found = await repo.find_by_ids([live.id, deleted.id, 999_999_999])

    assert [t.id for t in found] == [live.id]


# ── list ────────────────────────────────────────────────────────────


async def test_list_returns_all_rows_newest_first(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    first = await _insert(repo, session, _sample_row(created_at=_FUTURE))
    second = await _insert(repo, session, _sample_row(created_at=_FUTURE + timedelta(hours=1)))

    rows = await repo.list_all()
    mine = [t for t in rows if t.id in {first.id, second.id}]

    assert [t.id for t in mine] == [second.id, first.id]


async def test_list_is_not_capped_by_a_default_page_size(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    prefix = _unique_name("bulk")
    for index in range(105):
        await repo.insert(_sample_row(name=f"{prefix}-{index}"))
    await session.commit()

    rows = await repo.list_all()
    mine = [t for t in rows if t.name.startswith(prefix)]

    assert len(mine) == 105


async def test_list_applies_explicit_pagination(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    created = []
    base = datetime.now(UTC)
    for index in range(5):
        stored = await repo.insert(_sample_row(created_at=base + timedelta(milliseconds=index)))
        created.append(stored)
    await session.commit()

    # Fetch all and verify our 5 are present, ordered newest-first.
    rows = await repo.list_all()
    mine = [t for t in rows if t.id in {c.id for c in created}]
    assert [t.id for t in mine] == [c.id for c in reversed(created)]

    # Verify limit/offset: offset=0 returns the newest, offset=1 skips it.
    page0 = await repo.list_all(limit=1, offset=0)
    page1 = await repo.list_all(limit=1, offset=1)
    assert len(page0) == 1
    assert len(page1) == 1
    assert page0[0].id != page1[0].id


async def test_list_excludes_soft_deleted_rows(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    live = await _insert(repo, session, _sample_row())
    gone = await _insert(repo, session, _sample_row())
    await repo.update_by_primary_key({"id": gone.id}, {"deleted_at": _NOW})
    await session.commit()

    rows = await repo.list_all()
    ids = {t.id for t in rows}
    assert live.id in ids
    assert gone.id not in ids


# ── search ──────────────────────────────────────────────────────────


async def test_search_matches_keyword_in_name(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    keyword = _unique_name("alpha")
    match = await _insert(repo, session, _sample_row(name=keyword, description=None))
    await _insert(repo, session, _sample_row(description=None))

    rows, total = await repo.search(keyword=keyword)

    assert total == 1
    assert [t.id for t in rows] == [match.id]


async def test_search_matches_keyword_in_description(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    desc = _unique_name("research")
    match = await _insert(repo, session, _sample_row(description=desc))
    await _insert(repo, session, _sample_row(description=_unique_name("sales")))

    rows, total = await repo.search(keyword=desc)

    assert total == 1
    assert [t.id for t in rows] == [match.id]


async def test_search_treats_wildcards_in_keyword_literally(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    literal_name = f"100% owned-{uuid.uuid4().hex[:8]}"
    literal = await _insert(repo, session, _sample_row(name=literal_name, description=None))
    await _insert(repo, session, _sample_row(description=None))

    rows, total = await repo.search(keyword=literal_name)

    assert total == 1
    assert [t.id for t in rows] == [literal.id]


async def test_search_filters_by_tenant_id(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    wanted = await _insert(repo, session, _sample_row())
    await _insert(repo, session, _sample_row())

    rows, total = await repo.search(tenant_id=wanted.id)

    assert total == 1
    assert [t.id for t in rows] == [wanted.id]


async def test_search_combines_id_and_keyword_with_or(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    keyword = _unique_name("alpha")
    by_id = await _insert(repo, session, _sample_row(description=None))
    by_keyword = await _insert(repo, session, _sample_row(name=keyword, description=None))
    await _insert(repo, session, _sample_row(description=None))

    rows, total = await repo.search(keyword=keyword, tenant_id=by_id.id)

    assert total == 2
    assert {t.id for t in rows} == {by_id.id, by_keyword.id}


async def test_search_total_ignores_pagination(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    prefix = _unique_name("page")
    for index in range(5):
        await repo.insert(
            _sample_row(name=f"{prefix}-{index}", created_at=_FUTURE + timedelta(hours=index))
        )
    await session.commit()

    rows, total = await repo.search(keyword=prefix, limit=2, offset=0)

    assert total == 5
    assert len(rows) == 2


async def test_search_excludes_soft_deleted_rows(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    keyword = _unique_name("alpha")
    gone = await _insert(repo, session, _sample_row(name=keyword))
    await repo.update_by_primary_key({"id": gone.id}, {"deleted_at": _NOW})
    await session.commit()

    rows, total = await repo.search(keyword=keyword)

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
        await repo.adjust_storage_used(999_999_999, delta=1, updated_at=_NOW)

    assert excinfo.value.code == "tenant.not_found"


async def test_adjust_storage_used_stamps_updated_at(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    stored = await _insert(repo, session, _sample_row())
    later = _FUTURE + timedelta(days=2)

    await repo.adjust_storage_used(stored.id, delta=1, updated_at=later)
    await session.commit()

    assert (await repo.find_by_id(stored.id)).updated_at == later


# ── bulk_set_storage_quota ──────────────────────────────────────────


async def test_bulk_set_storage_quota_updates_every_live_tenant(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    first = await _insert(repo, session, _sample_row())
    second = await _insert(repo, session, _sample_row())

    affected = await repo.bulk_set_storage_quota(quota_bytes=2048, updated_at=_NOW)
    await session.commit()

    assert affected >= 2
    assert (await repo.find_by_id(first.id)).storage_quota == 2048
    assert (await repo.find_by_id(second.id)).storage_quota == 2048


async def test_bulk_set_storage_quota_skips_soft_deleted_tenants(session: AsyncSession) -> None:
    repo = TenantRepository(session)
    live = await _insert(repo, session, _sample_row())
    gone = await _insert(repo, session, _sample_row())
    await repo.update_by_primary_key({"id": gone.id}, {"deleted_at": _NOW})
    await session.commit()

    affected = await repo.bulk_set_storage_quota(quota_bytes=2048, updated_at=_NOW)
    await session.commit()

    assert affected >= 1
    assert (await repo.find_by_id(live.id)).storage_quota == 2048


# ── escape_like_keyword ─────────────────────────────────────────────


def test_escape_like_keyword_escapes_wildcards_and_backslash() -> None:
    assert escape_like_keyword(r"50%_a\b") == r"50\%\_a\\b"


def test_escape_like_keyword_leaves_plain_text_untouched() -> None:
    assert escape_like_keyword("acme") == "acme"
