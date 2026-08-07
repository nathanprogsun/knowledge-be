"""Integration tests for ``ChunkRevisionRepository`` against the applied schema.

Tests insert unique rows per run; isolation relies on unique tenant ids
and chunk ids. Tests commit explicitly. The ``chunk_revisions`` table is
created by migration ``0019`` (sequenced after the ``chunks`` table), so
these tests run once the full migration chain is applied.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.dao.chunk_revision_repository import ChunkRevisionRepository
from src.db.models.chunk_revision import ChunkRevision
from tests.integration.db.dao.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _revision(
    *,
    tenant_id: int,
    chunk_id: str,
    revision: int,
    **overrides: object,
) -> ChunkRevision:
    values: dict[str, object] = {
        "id": f"{chunk_id}-rev-{revision}",
        "tenant_id": tenant_id,
        "knowledge_base_id": "kb-1",
        "knowledge_id": "knowledge-1",
        "chunk_id": chunk_id,
        "revision": revision,
        "content": f"body-{revision}",
        "is_enabled": True,
        "editor_id": "editor-1",
        "edit_source": "user",
        "edited_at": _NOW,
        "created_at": _NOW,
    }
    values.update(overrides)
    return ChunkRevision.model_validate(values)


# ── create ─────────────────────────────────────────────────────────


async def test_create_persists_every_column(session: AsyncSession) -> None:
    repo = ChunkRevisionRepository(session)
    tenant_id = make_test_tenant_id()
    row = _revision(
        tenant_id=tenant_id,
        chunk_id="chunk-1",
        revision=1,
        content="edited body",
        is_enabled=False,
        editor_id="actor-1",
        edit_source="user",
    )

    stored = await repo.create(row)
    await session.commit()

    assert stored == row
    fetched = await repo.find_by_primary_key({"id": row.id})
    assert fetched is not None
    assert fetched.content == "edited body"
    assert fetched.is_enabled is False
    assert fetched.editor_id == "actor-1"
    assert fetched.edit_source == "user"


async def test_create_applies_model_defaults(session: AsyncSession) -> None:
    repo = ChunkRevisionRepository(session)
    tenant_id = make_test_tenant_id()
    row = _revision(tenant_id=tenant_id, chunk_id="chunk-d", revision=0)

    stored = await repo.create(row)
    await session.commit()

    assert stored.content == ""
    assert stored.is_enabled is True
    assert stored.editor_id == ""
    assert stored.edit_source == "user"


# ── list ───────────────────────────────────────────────────────────


async def test_list_chunk_revisions_orders_newest_first(session: AsyncSession) -> None:
    repo = ChunkRevisionRepository(session)
    tenant_id = make_test_tenant_id()
    for revision in (1, 2, 3):
        await repo.create(_revision(tenant_id=tenant_id, chunk_id="chunk-1", revision=revision))
    await session.commit()

    rows = await repo.list_chunk_revisions(tenant_id=tenant_id, chunk_id="chunk-1")

    assert [r.revision for r in rows] == [3, 2, 1]


async def test_list_chunk_revisions_isolated_by_tenant(session: AsyncSession) -> None:
    repo = ChunkRevisionRepository(session)
    tenant_a = make_test_tenant_id()
    tenant_b = make_test_tenant_id()
    await repo.create(_revision(tenant_id=tenant_a, chunk_id="chunk-1", revision=1))
    await repo.create(_revision(tenant_id=tenant_b, chunk_id="chunk-1", revision=1))
    await session.commit()

    rows = await repo.list_chunk_revisions(tenant_id=tenant_a, chunk_id="chunk-1")

    assert [r.id for r in rows] == ["chunk-1-rev-1"]
    assert rows[0].tenant_id == tenant_a


async def test_list_chunk_revisions_isolated_by_chunk(session: AsyncSession) -> None:
    repo = ChunkRevisionRepository(session)
    tenant_id = make_test_tenant_id()
    await repo.create(_revision(tenant_id=tenant_id, chunk_id="chunk-a", revision=1))
    await repo.create(_revision(tenant_id=tenant_id, chunk_id="chunk-b", revision=1))
    await session.commit()

    rows = await repo.list_chunk_revisions(tenant_id=tenant_id, chunk_id="chunk-a")

    assert [r.chunk_id for r in rows] == ["chunk-a"]


# ── get ────────────────────────────────────────────────────────────


async def test_get_chunk_revision_returns_snapshot(session: AsyncSession) -> None:
    repo = ChunkRevisionRepository(session)
    tenant_id = make_test_tenant_id()
    await repo.create(_revision(tenant_id=tenant_id, chunk_id="chunk-1", revision=1))
    await repo.create(_revision(tenant_id=tenant_id, chunk_id="chunk-1", revision=2))
    await session.commit()

    row = await repo.get_chunk_revision(
        tenant_id=tenant_id,
        chunk_id="chunk-1",
        revision=2,
    )

    assert row is not None
    assert row.revision == 2
    assert row.content == "body-2"


async def test_get_chunk_revision_returns_none_when_absent(session: AsyncSession) -> None:
    repo = ChunkRevisionRepository(session)
    tenant_id = make_test_tenant_id()

    row = await repo.get_chunk_revision(
        tenant_id=tenant_id,
        chunk_id="chunk-1",
        revision=1,
    )

    assert row is None


# ── uniqueness ─────────────────────────────────────────────────────


async def test_duplicate_chunk_revision_raises_integrity_error(
    session: AsyncSession,
) -> None:
    repo = ChunkRevisionRepository(session)
    tenant_id = make_test_tenant_id()
    await repo.create(_revision(tenant_id=tenant_id, chunk_id="chunk-1", revision=1))
    await session.commit()

    duplicate = _revision(
        tenant_id=tenant_id,
        chunk_id="chunk-1",
        revision=1,
        id="chunk-1-rev-1-copy",
    )

    with pytest.raises(IntegrityError):
        await repo.create(duplicate)
        await session.commit()


async def test_same_revision_allowed_across_chunks(session: AsyncSession) -> None:
    repo = ChunkRevisionRepository(session)
    tenant_id = make_test_tenant_id()
    await repo.create(_revision(tenant_id=tenant_id, chunk_id="chunk-a", revision=1))
    await repo.create(_revision(tenant_id=tenant_id, chunk_id="chunk-b", revision=1))
    await session.commit()

    rows = await repo.list_chunk_revisions(tenant_id=tenant_id, chunk_id="chunk-b")

    assert [r.revision for r in rows] == [1]


# ── schema sanity ──────────────────────────────────────────────────


async def test_primary_key_is_the_revision_row_id(session: AsyncSession) -> None:
    """The unique pair is (chunk_id, revision); id stays the primary key."""
    repo = ChunkRevisionRepository(session)
    tenant_id = make_test_tenant_id()
    await repo.create(_revision(tenant_id=tenant_id, chunk_id="chunk-1", revision=1))
    await session.commit()

    fetched = await repo.find_by_primary_key({"id": "chunk-1-rev-1"})

    assert fetched is not None
    assert fetched.tenant_id == tenant_id
