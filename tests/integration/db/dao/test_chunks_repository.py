"""Integration tests for ``ChunkRepository`` against the real applied schema.

Tests insert unique rows per run; isolation relies on unique chunk ids,
knowledge ids, and tenant ids. The ``chunks`` table sits at the tail of
the alembic chain, so these tests run once the chain is applied.
"""

from __future__ import annotations

import itertools
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.exception import ConflictError, NotFoundError, ValidationError
from src.db.dao.chunk_repository import ChunkRepository
from src.db.models.chunk import Chunk

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_tenant_counter = itertools.count(3_000_000)


def _tenant_id() -> int:
    """Return a unique 32-bit tenant id for the chunks table."""
    return next(_tenant_counter)


def _cid() -> str:
    return f"chunk-{uuid.uuid4().hex[:12]}"


def _sample_chunk(
    *,
    tenant_id: int,
    id: str | None = None,
    knowledge_id: str = "knowledge-doc-1",
    knowledge_base_id: str = "kb-1",
    content: str = "The quick brown fox jumps over the lazy dog.",
    chunk_index: int = 0,
    **overrides: object,
) -> Chunk:
    return Chunk.model_validate(
        {
            "id": id or _cid(),
            "tenant_id": tenant_id,
            "knowledge_base_id": knowledge_base_id,
            "knowledge_id": knowledge_id,
            "content": content,
            "chunk_index": chunk_index,
            "is_enabled": True,
            "start_at": 0,
            "end_at": len(content),
            "chunk_type": "text",
            "flags": 1,
            "source_content": "",
            "content_revision": 0,
            "index_status": "ready",
            "last_editor_id": "",
            "created_at": _NOW,
            "updated_at": _NOW,
            "deleted_at": None,
            **overrides,
        }
    )


async def _insert(repo: ChunkRepository, chunk: Chunk) -> Chunk:
    """Insert a row and return the persisted copy (with DB-assigned seq_id)."""
    return await repo.create(chunk)


# ── create / get_by_id ───────────────────────────────────────────────


async def test_create_assigns_seq_id_and_round_trips(session: AsyncSession) -> None:
    repo = ChunkRepository(session)
    tid = _tenant_id()
    cid = _cid()
    persisted = await _insert(repo, _sample_chunk(tenant_id=tid, id=cid))

    assert persisted.seq_id > 0
    assert persisted.index_status == "ready"

    resolved = await repo.get_by_id(tid, cid)
    assert resolved is not None
    assert resolved.id == cid
    assert resolved.content == "The quick brown fox jumps over the lazy dog."


async def test_get_by_id_raises_not_found_for_unknown(session: AsyncSession) -> None:
    repo = ChunkRepository(session)
    with pytest.raises(NotFoundError) as excinfo:
        await repo.get_by_id(_tenant_id(), _cid())
    assert excinfo.value.code == "chunk.not_found"


async def test_get_by_id_or_none_returns_none_for_unknown(session: AsyncSession) -> None:
    repo = ChunkRepository(session)
    assert await repo.get_by_id_or_none(_tenant_id(), _cid()) is None


async def test_get_by_id_only_ignores_tenant_filter(session: AsyncSession) -> None:
    repo = ChunkRepository(session)
    tid = _tenant_id()
    cid = _cid()
    await _insert(repo, _sample_chunk(tenant_id=tid, id=cid))

    assert await repo.get_by_id_only(cid) is not None
    assert await repo.get_by_id_only(_cid()) is None


async def test_get_by_seq_id(session: AsyncSession) -> None:
    repo = ChunkRepository(session)
    tid = _tenant_id()
    persisted = await _insert(repo, _sample_chunk(tenant_id=tid))

    resolved = await repo.get_by_seq_id(tid, persisted.seq_id)
    assert resolved is not None
    assert resolved.id == persisted.id


async def test_tenant_isolation(session: AsyncSession) -> None:
    repo = ChunkRepository(session)
    tid_a = _tenant_id()
    tid_b = _tenant_id()
    cid = _cid()
    await _insert(repo, _sample_chunk(tenant_id=tid_a, id=cid))

    assert await repo.get_by_id_or_none(tid_a, cid) is not None
    assert await repo.get_by_id_or_none(tid_b, cid) is None


# ── create_many / list_by_ids ────────────────────────────────────────


async def test_create_many_and_list_by_ids(session: AsyncSession) -> None:
    repo = ChunkRepository(session)
    tid = _tenant_id()
    rows = [_sample_chunk(tenant_id=tid, content=f"chunk {i}") for i in range(3)]

    persisted = await repo.create_many(rows)

    assert len(persisted) == 3
    assert all(p.seq_id > 0 for p in persisted)
    by_id = await repo.list_by_ids(tid, [p.id for p in persisted])
    assert {c.id for c in by_id} == {p.id for p in persisted}


async def test_create_many_empty_returns_empty(session: AsyncSession) -> None:
    repo = ChunkRepository(session)
    assert await repo.create_many([]) == []
    assert await repo.list_by_ids(_tenant_id(), []) == []


# ── list_by_knowledge_id / list_by_parent_id ─────────────────────────


async def test_list_by_knowledge_id_filters_text_and_orders(session: AsyncSession) -> None:
    repo = ChunkRepository(session)
    tid = _tenant_id()
    kid = "knowledge-doc-9"
    await _insert(repo, _sample_chunk(tenant_id=tid, knowledge_id=kid, content="b", chunk_index=1))
    await _insert(repo, _sample_chunk(tenant_id=tid, knowledge_id=kid, content="a", chunk_index=0))
    await _insert(
        repo,
        _sample_chunk(
            tenant_id=tid,
            knowledge_id=kid,
            content="faq",
            chunk_type="faq",
            chunk_index=2,
        ),
    )

    rows = await repo.list_by_knowledge_id(tid, kid)

    assert [c.content for c in rows] == ["a", "b"]


async def test_list_by_parent_id(session: AsyncSession) -> None:
    repo = ChunkRepository(session)
    tid = _tenant_id()
    parent_id = _cid()
    child_a = _cid()
    child_b = _cid()
    await _insert(repo, _sample_chunk(tenant_id=tid, id=child_a, parent_chunk_id=parent_id))
    await _insert(repo, _sample_chunk(tenant_id=tid, id=child_b, parent_chunk_id=parent_id))
    await _insert(repo, _sample_chunk(tenant_id=tid))

    rows = await repo.list_by_parent_id(tid, parent_id)

    assert {c.id for c in rows} == {child_a, child_b}


# ── update ───────────────────────────────────────────────────────────


async def test_update_overwrites_mutable_columns(session: AsyncSession) -> None:
    repo = ChunkRepository(session)
    tid = _tenant_id()
    persisted = await _insert(repo, _sample_chunk(tenant_id=tid))

    edited = persisted.model_copy(
        update={
            "content": "edited body",
            "content_revision": 1,
            "is_enabled": False,
            "tag_id": "tag-1",
        }
    )
    refreshed = await repo.update(edited)

    assert refreshed.content == "edited body"
    assert refreshed.content_revision == 1
    assert refreshed.is_enabled is False
    assert refreshed.tag_id == "tag-1"


async def test_update_preserves_identity_columns(session: AsyncSession) -> None:
    repo = ChunkRepository(session)
    tid = _tenant_id()
    cid = _cid()
    persisted = await _insert(repo, _sample_chunk(tenant_id=tid, id=cid))
    seq_before = persisted.seq_id

    edited = persisted.model_copy(update={"content": "new", "created_at": _NOW})
    refreshed = await repo.update(edited)

    assert refreshed.id == cid
    assert refreshed.tenant_id == tid
    assert refreshed.seq_id == seq_before


# ── update_document_chunk ────────────────────────────────────────────


async def test_update_document_chunk_bumps_revision(session: AsyncSession) -> None:
    repo = ChunkRepository(session)
    tid = _tenant_id()
    persisted = await _insert(
        repo,
        _sample_chunk(tenant_id=tid, content="  original  ", source_content="original"),
    )

    updated = await repo.update_document_chunk(
        tenant_id=tid,
        chunk_id=persisted.id,
        content="  revised body  ",
        is_enabled=None,
        expected_revision=0,
        last_editor_id="usr-1",
        now=_NOW,
    )

    assert updated.content == "revised body"
    assert updated.content_revision == 1
    assert updated.last_editor_id == "usr-1"
    assert updated.index_status == "processing"


async def test_update_document_chunk_conflict_on_stale_revision(
    session: AsyncSession,
) -> None:
    repo = ChunkRepository(session)
    tid = _tenant_id()
    persisted = await _insert(repo, _sample_chunk(tenant_id=tid))

    with pytest.raises(ConflictError) as excinfo:
        await repo.update_document_chunk(
            tenant_id=tid,
            chunk_id=persisted.id,
            content="edit",
            is_enabled=None,
            expected_revision=5,
            last_editor_id="usr-1",
            now=_NOW,
        )
    assert excinfo.value.code == "chunk.revision_conflict"


async def test_update_document_chunk_rejects_empty_content(
    session: AsyncSession,
) -> None:
    repo = ChunkRepository(session)
    tid = _tenant_id()
    persisted = await _insert(repo, _sample_chunk(tenant_id=tid))

    with pytest.raises(ValidationError) as excinfo:
        await repo.update_document_chunk(
            tenant_id=tid,
            chunk_id=persisted.id,
            content="   ",
            is_enabled=None,
            expected_revision=0,
            last_editor_id="usr-1",
            now=_NOW,
        )
    assert excinfo.value.code == "chunk.content_empty"


async def test_update_document_chunk_rejects_non_text_chunk(
    session: AsyncSession,
) -> None:
    repo = ChunkRepository(session)
    tid = _tenant_id()
    persisted = await _insert(
        repo,
        _sample_chunk(tenant_id=tid, chunk_type="faq"),
    )

    with pytest.raises(ValidationError) as excinfo:
        await repo.update_document_chunk(
            tenant_id=tid,
            chunk_id=persisted.id,
            content="edit",
            is_enabled=None,
            expected_revision=0,
            last_editor_id="usr-1",
            now=_NOW,
        )
    assert excinfo.value.code == "chunk.not_editable"


async def test_update_document_chunk_noop_returns_current(session: AsyncSession) -> None:
    repo = ChunkRepository(session)
    tid = _tenant_id()
    persisted = await _insert(repo, _sample_chunk(tenant_id=tid))

    updated = await repo.update_document_chunk(
        tenant_id=tid,
        chunk_id=persisted.id,
        content=persisted.content,
        is_enabled=persisted.is_enabled,
        expected_revision=0,
        last_editor_id="usr-1",
        now=_NOW,
    )

    assert updated.content_revision == 0
    assert updated.last_editor_id == ""


async def test_update_document_chunk_not_found(session: AsyncSession) -> None:
    repo = ChunkRepository(session)
    with pytest.raises(NotFoundError) as excinfo:
        await repo.update_document_chunk(
            tenant_id=_tenant_id(),
            chunk_id=_cid(),
            content="edit",
            is_enabled=None,
            expected_revision=0,
            last_editor_id="usr-1",
            now=_NOW,
        )
    assert excinfo.value.code == "chunk.not_found"


# ── soft delete / aggregates ─────────────────────────────────────────


async def test_soft_delete_hides_row(session: AsyncSession) -> None:
    repo = ChunkRepository(session)
    tid = _tenant_id()
    cid = _cid()
    await _insert(repo, _sample_chunk(tenant_id=tid, id=cid))

    affected = await repo.soft_delete(tenant_id=tid, id=cid, now=_NOW)

    assert affected is True
    assert await repo.get_by_id_or_none(tid, cid) is None
    assert await repo.soft_delete(tenant_id=tid, id=cid, now=_NOW) is False


async def test_delete_by_knowledge_id(session: AsyncSession) -> None:
    repo = ChunkRepository(session)
    tid = _tenant_id()
    kid = "knowledge-doc-7"
    await _insert(repo, _sample_chunk(tenant_id=tid, knowledge_id=kid))
    await _insert(repo, _sample_chunk(tenant_id=tid, knowledge_id=kid))

    affected = await repo.delete_by_knowledge_id(tenant_id=tid, knowledge_id=kid, now=_NOW)

    assert affected == 2
    assert await repo.list_by_knowledge_id(tid, kid) == []


async def test_count_by_knowledge_base_id(session: AsyncSession) -> None:
    repo = ChunkRepository(session)
    tid = _tenant_id()
    kb = "kb-count-1"
    await _insert(repo, _sample_chunk(tenant_id=tid, knowledge_base_id=kb))
    await _insert(repo, _sample_chunk(tenant_id=tid, knowledge_base_id=kb))

    assert await repo.count_by_knowledge_base_id(tid, kb) == 2
    assert await repo.count_by_knowledge_base_id(tid, "kb-other") == 0


async def test_move_by_knowledge_id(session: AsyncSession) -> None:
    repo = ChunkRepository(session)
    tid = _tenant_id()
    kid = "knowledge-doc-3"
    cid = _cid()
    await _insert(repo, _sample_chunk(tenant_id=tid, id=cid, knowledge_id=kid))

    affected = await repo.move_by_knowledge_id(
        tenant_id=tid,
        knowledge_id=kid,
        target_kb_id="kb-moved",
    )

    assert affected == 1
    resolved = await repo.get_by_id(tid, cid)
    assert resolved.knowledge_base_id == "kb-moved"


# ── JSONB round-trip ─────────────────────────────────────────────────


async def test_json_columns_round_trip(session: AsyncSession) -> None:
    repo = ChunkRepository(session)
    tid = _tenant_id()
    await _insert(
        repo,
        _sample_chunk(
            tenant_id=tid,
            relation_chunks=["rel-1", "rel-2"],
            metadata={"standard_question": "what is x", "answers": ["a"]},
        ),
    )

    rows = await repo.list_by_knowledge_id(tid, "knowledge-doc-1")
    assert len(rows) == 1
    assert rows[0].relation_chunks == ["rel-1", "rel-2"]
    meta = rows[0].metadata
    assert isinstance(meta, dict)
    assert meta["standard_question"] == "what is x"
