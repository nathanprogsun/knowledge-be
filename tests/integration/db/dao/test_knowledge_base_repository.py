"""Integration tests for ``KnowledgeBaseRepository`` against the real applied schema.

Tests insert unique rows per run; isolation relies on unique
knowledge-base ids and tenant ids. Tests commit explicitly.

The three aggregate counts that read sibling tables (``knowledges`` /
``chunks`` / ``kb_shares``) are exercised by the PRs that create those
tables; the repository methods themselves are verified by imports and
typing here.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.exception import DataError
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.models.knowledge_base import KnowledgeBase
from tests.integration.db.dao.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _kbid() -> str:
    return f"kb-{uuid.uuid4().hex[:12]}"


def _kb(
    *,
    name: str = "docs",
    tenant_id: int | None = None,
    type: str = "document",
    is_temporary: bool = False,
    description: str | None = None,
    vector_store_id: str | None = None,
    created_at: datetime = _NOW,
) -> KnowledgeBase:
    return KnowledgeBase(
        id=_kbid(),
        name=name,
        type=type,
        is_temporary=is_temporary,
        description=description,
        tenant_id=tenant_id if tenant_id is not None else make_test_tenant_id(),
        vector_store_id=vector_store_id,
        created_at=created_at,
        updated_at=created_at,
    )


# ── create / read ───────────────────────────────────────────────────


async def test_create_round_trips_persisted_fields(session: AsyncSession) -> None:
    repo = KnowledgeBaseRepository(session)
    row = _kb(name="design-docs", type="faq", description="shared notes")

    stored = await repo.create(row)
    await session.commit()

    assert stored is not None
    assert stored.id == row.id
    fetched = await repo.get_by_id_or_none(row.id)
    assert fetched is not None
    assert fetched.name == "design-docs"
    assert fetched.type == "faq"
    assert fetched.description == "shared notes"


async def test_get_by_id_returns_none_for_unknown_id(session: AsyncSession) -> None:
    repo = KnowledgeBaseRepository(session)

    assert await repo.get_by_id_or_none("missing-kb") is None


async def test_get_by_id_and_tenant_enforces_isolation(session: AsyncSession) -> None:
    repo = KnowledgeBaseRepository(session)
    row = _kb()
    await repo.create(row)
    await session.commit()

    assert await repo.get_by_id_and_tenant(row.id, row.tenant_id) is not None
    assert await repo.get_by_id_and_tenant(row.id, make_test_tenant_id()) is None


async def test_get_by_ids_returns_matching_subset(session: AsyncSession) -> None:
    repo = KnowledgeBaseRepository(session)
    a = _kb()
    b = _kb()
    c = _kb()
    for row in (a, b, c):
        await repo.create(row)
    await session.commit()

    found = await repo.get_by_ids([a.id, c.id])

    assert {r.id for r in found} == {a.id, c.id}


async def test_get_by_ids_empty_input_returns_empty_list(session: AsyncSession) -> None:
    repo = KnowledgeBaseRepository(session)

    assert await repo.get_by_ids([]) == []


# ── listing ─────────────────────────────────────────────────────────


async def test_list_by_tenant_excludes_temporary_rows(session: AsyncSession) -> None:
    repo = KnowledgeBaseRepository(session)
    tid = make_test_tenant_id()
    permanent = _kb(tenant_id=tid)
    temporary = _kb(tenant_id=tid, is_temporary=True)
    await repo.create(permanent)
    await repo.create(temporary)
    await session.commit()

    rows = await repo.list_by_tenant(tid)

    assert [r.id for r in rows] == [permanent.id]


async def test_list_by_tenant_orders_newest_first(session: AsyncSession) -> None:
    repo = KnowledgeBaseRepository(session)
    tid = make_test_tenant_id()
    older = _kb(tenant_id=tid, created_at=_NOW)
    newer = _kb(tenant_id=tid, created_at=_NOW.replace(hour=23))
    await repo.create(older)
    await repo.create(newer)
    await session.commit()

    rows = await repo.list_by_tenant(tid)

    assert [r.id for r in rows] == [newer.id, older.id]


# ── update / delete ─────────────────────────────────────────────────


async def test_update_overwrites_mutable_columns(session: AsyncSession) -> None:
    repo = KnowledgeBaseRepository(session)
    row = _kb(name="before")
    await repo.create(row)
    await session.commit()

    renamed = row.model_copy(update={"name": "after"})
    persisted = await repo.update(renamed)
    await session.commit()

    assert persisted.name == "after"
    assert (await repo.get_by_id_or_none(row.id)) is not None


async def test_update_preserves_immutable_vector_store_id(session: AsyncSession) -> None:
    repo = KnowledgeBaseRepository(session)
    row = _kb(vector_store_id="vs-1")
    await repo.create(row)
    await session.commit()

    updated = await repo.update(row.model_copy(update={"name": "renamed"}))
    await session.commit()

    assert updated.vector_store_id == "vs-1"


async def test_update_raises_for_missing_row(session: AsyncSession) -> None:
    repo = KnowledgeBaseRepository(session)

    with pytest.raises(DataError) as excinfo:
        await repo.update(_kb())
    assert excinfo.value.code == "knowledge_base.update_no_row"


async def test_soft_delete_hides_row_from_reads(session: AsyncSession) -> None:
    repo = KnowledgeBaseRepository(session)
    row = _kb()
    await repo.create(row)
    await session.commit()

    affected = await repo.soft_delete(id=row.id, now=_NOW.replace(year=2027))
    await session.commit()

    assert affected is True
    assert await repo.get_by_id_or_none(row.id) is None
    tombstone = await repo.find_by_primary_key(
        {"id": row.id},
        exclude_deleted_or_archived=False,
    )
    assert tombstone is not None
    assert tombstone.deleted_at == _NOW.replace(year=2027)


# ── aggregate counts ────────────────────────────────────────────────


async def test_count_by_vector_store_id_counts_live_rows_only(
    session: AsyncSession,
) -> None:
    repo = KnowledgeBaseRepository(session)
    tid = make_test_tenant_id()
    for _ in range(2):
        await repo.create(_kb(tenant_id=tid, vector_store_id="vs-1"))
    await repo.create(_kb(tenant_id=tid, vector_store_id="vs-2"))
    doomed = _kb(tenant_id=tid, vector_store_id="vs-1")
    await repo.create(doomed)
    await session.commit()
    await repo.soft_delete(id=doomed.id, now=_NOW)
    await session.commit()

    assert await repo.count_by_vector_store_id(tenant_id=tid, store_id="vs-1") == 2
    assert await repo.count_by_vector_store_id(tenant_id=tid, store_id="vs-2") == 1
    assert await repo.count_by_vector_store_id(tenant_id=tid, store_id="missing") == 0
