"""Integration tests for ``TemporaryDocumentRepository`` against the real schema.

Tests insert unique rows per run; isolation relies on unique tenant ids,
session ids, and document ids. Tests commit explicitly.

NOTE: these tests require the applied migration chain through
``0022_temporary_documents`` (the wave-1 migrations land in order), so
they may not be runnable in a partial worktree — they are written against
the final schema.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.dao.temporary_document_repository import TemporaryDocumentRepository
from src.db.models.temporary_document import (
    TEMPORARY_DOCUMENT_STATUS_FAILED,
    TEMPORARY_DOCUMENT_STATUS_PROCESSING,
    TEMPORARY_DOCUMENT_STATUS_READY,
    TemporaryDocument,
)
from tests.integration.db.dao.conftest import make_test_tenant_id


def _make_row(
    tenant_id: int,
    session_id: str,
    *,
    status: str = "uploaded",
    expires_at: datetime | None = None,
    created_at: datetime | None = None,
) -> TemporaryDocument:
    """Build a minimal valid temporary-document row."""
    now = datetime.now(UTC)
    return TemporaryDocument(
        id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        session_id=session_id,
        resource_ref=f"tmp/{uuid.uuid4().hex[:12]}",
        file_name="note.md",
        file_type=".md",
        mime_type="text/markdown",
        file_size=1024,
        status=status,
        expires_at=expires_at or (now + timedelta(hours=24)),
        created_at=created_at or now,
        updated_at=now,
    )


async def test_create_then_get_by_id(session: AsyncSession) -> None:
    tid = make_test_tenant_id()
    repo = TemporaryDocumentRepository(session)
    row = _make_row(tid, f"session-{uuid.uuid4().hex[:8]}")
    persisted = await repo.create(row)
    assert persisted.id == row.id
    assert persisted.status == "uploaded"
    await session.commit()

    found = await repo.get_by_id(tenant_id=tid, document_id=row.id)
    assert found is not None
    assert found.file_name == "note.md"
    assert found.chunks == []
    assert found.metadata == {}


async def test_get_scoped_requires_session_match(session: AsyncSession) -> None:
    tid = make_test_tenant_id()
    repo = TemporaryDocumentRepository(session)
    row = _make_row(tid, f"session-{uuid.uuid4().hex[:8]}")
    await repo.create(row)
    await session.commit()

    assert await repo.get_scoped(
        tenant_id=tid,
        session_id="other-session",
        document_id=row.id,
    ) is None
    assert await repo.get_scoped(
        tenant_id=tid,
        session_id=row.session_id,
        document_id=row.id,
    ) is not None


async def test_get_by_id_isolated_by_tenant(session: AsyncSession) -> None:
    tid_a = make_test_tenant_id()
    tid_b = make_test_tenant_id()
    repo = TemporaryDocumentRepository(session)
    row = _make_row(tid_a, f"session-{uuid.uuid4().hex[:8]}")
    await repo.create(row)
    await session.commit()

    assert await repo.get_by_id(tenant_id=tid_a, document_id=row.id) is not None
    assert await repo.get_by_id(tenant_id=tid_b, document_id=row.id) is None


async def test_list_scoped_oldest_first(session: AsyncSession) -> None:
    tid = make_test_tenant_id()
    repo = TemporaryDocumentRepository(session)
    sid = f"session-{uuid.uuid4().hex[:8]}"
    base = datetime.now(UTC)
    older = await repo.create(_make_row(tid, sid, created_at=base - timedelta(hours=2)))
    newer = await repo.create(_make_row(tid, sid, created_at=base - timedelta(hours=1)))
    await session.commit()

    rows = await repo.list_scoped(tenant_id=tid, session_id=sid)
    assert [r.id for r in rows] == [older.id, newer.id]


async def test_mark_processing(session: AsyncSession) -> None:
    tid = make_test_tenant_id()
    repo = TemporaryDocumentRepository(session)
    row = await repo.create(_make_row(tid, f"session-{uuid.uuid4().hex[:8]}"))
    await session.commit()

    now = datetime.now(UTC)
    updated = await repo.mark_processing(
        tenant_id=tid,
        document_id=row.id,
        started_at=now,
        now=now,
    )
    assert updated is not None
    assert updated.status == TEMPORARY_DOCUMENT_STATUS_PROCESSING
    assert updated.started_at is not None


async def test_mark_ready(session: AsyncSession) -> None:
    tid = make_test_tenant_id()
    repo = TemporaryDocumentRepository(session)
    row = await repo.create(_make_row(tid, f"session-{uuid.uuid4().hex[:8]}"))
    await session.commit()

    now = datetime.now(UTC)
    chunks = [{"seq": 0, "content": "hello", "start": 0, "end": 5, "token_count": 2}]
    updated = await repo.mark_ready(
        tenant_id=tid,
        document_id=row.id,
        content="hello world",
        chunks=chunks,
        image_refs=[],
        metadata={"parser": "plain_text"},
        token_count=3,
        chunk_count=1,
        ready_at=now,
        now=now,
    )
    assert updated is not None
    assert updated.status == TEMPORARY_DOCUMENT_STATUS_READY
    assert updated.content == "hello world"
    assert updated.chunks == chunks
    assert updated.metadata == {"parser": "plain_text"}
    assert updated.token_count == 3
    assert updated.chunk_count == 1
    assert updated.ready_at is not None


async def test_mark_failed(session: AsyncSession) -> None:
    tid = make_test_tenant_id()
    repo = TemporaryDocumentRepository(session)
    row = await repo.create(_make_row(tid, f"session-{uuid.uuid4().hex[:8]}"))
    await session.commit()

    updated = await repo.mark_failed(
        tenant_id=tid,
        document_id=row.id,
        message="parser unavailable",
        now=datetime.now(UTC),
    )
    assert updated is not None
    assert updated.status == TEMPORARY_DOCUMENT_STATUS_FAILED
    assert updated.error_message == "parser unavailable"


async def test_mark_transitions_scoped_by_tenant(session: AsyncSession) -> None:
    tid_a = make_test_tenant_id()
    tid_b = make_test_tenant_id()
    repo = TemporaryDocumentRepository(session)
    row = await repo.create(_make_row(tid_a, f"session-{uuid.uuid4().hex[:8]}"))
    await session.commit()

    now = datetime.now(UTC)
    updated = await repo.mark_processing(
        tenant_id=tid_b,
        document_id=row.id,
        started_at=now,
        now=now,
    )
    assert updated is None


async def test_delete_scoped_soft_deletes(session: AsyncSession) -> None:
    tid = make_test_tenant_id()
    repo = TemporaryDocumentRepository(session)
    row = _make_row(tid, f"session-{uuid.uuid4().hex[:8]}")
    await repo.create(row)
    await session.commit()

    deleted = await repo.delete_scoped(
        tenant_id=tid,
        session_id=row.session_id,
        document_id=row.id,
        now=datetime.now(UTC),
    )
    assert deleted is True
    await session.commit()

    assert await repo.get_scoped(
        tenant_id=tid,
        session_id=row.session_id,
        document_id=row.id,
    ) is None

    deleted_again = await repo.delete_scoped(
        tenant_id=tid,
        session_id=row.session_id,
        document_id=row.id,
        now=datetime.now(UTC),
    )
    assert deleted_again is False


async def test_list_expired_returns_only_stale(session: AsyncSession) -> None:
    tid = make_test_tenant_id()
    repo = TemporaryDocumentRepository(session)
    sid = f"session-{uuid.uuid4().hex[:8]}"
    now = datetime.now(UTC)
    expired = await repo.create(
        _make_row(tid, sid, expires_at=now - timedelta(minutes=5))
    )
    fresh = await repo.create(
        _make_row(tid, sid, expires_at=now + timedelta(hours=1))
    )
    await session.commit()

    try:
        rows = await repo.list_expired(before=now, limit=10)
        ids = [r.id for r in rows]
        # The global sweep may already contain expired rows left by other
        # runs on the shared dev DB, so assert the sweep CONTRACT rather
        # than this run's specific row landing inside the limit window:
        # only stale rows are returned, ordered oldest-first, and the
        # fresh row is never swept.
        assert rows
        assert all(r.expires_at <= now for r in rows)
        assert [r.expires_at for r in rows] == sorted(r.expires_at for r in rows)
        assert fresh.id not in ids
    finally:
        # Self-clean so the global expiry sweep does not accumulate rows
        # across repeated runs on the shared dev DB.
        await repo.delete_scoped(
            tenant_id=tid, session_id=sid, document_id=expired.id, now=now
        )
        await repo.delete_scoped(
            tenant_id=tid, session_id=sid, document_id=fresh.id, now=now
        )
        await session.commit()
