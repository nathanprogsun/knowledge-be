"""Unit tests for :mod:`src.db.dao.chunk_repository`.

Non-DB tests: exercise the generated SQL text (via a stub session that
records statements) and the pre-write validation logic of
``update_document_chunk`` without a database. The real SQL round-trip is
covered by ``tests/integration/db/dao/test_chunks_repository.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy.sql.expression import TextClause

from src.common.exception import ConflictError, ValidationError
from src.db.dao.chunk_repository import ChunkRepository
from src.db.models.chunk import Chunk

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _row(
    *,
    id: str = "chunk-1",
    tenant_id: int = 1,
    content: str = "original body",
    chunk_type: str = "text",
    content_revision: int = 0,
    **overrides: object,
) -> dict[str, object]:
    row = {
        "id": id,
        "tenant_id": tenant_id,
        "knowledge_base_id": "kb-1",
        "knowledge_id": "knowledge-doc-1",
        "content": content,
        "chunk_index": 0,
        "is_enabled": True,
        "start_at": 0,
        "end_at": len(content),
        "pre_chunk_id": None,
        "next_chunk_id": None,
        "chunk_type": chunk_type,
        "parent_chunk_id": None,
        "image_info": None,
        "relation_chunks": None,
        "indirect_relation_chunks": None,
        "metadata": None,
        "tag_id": None,
        "status": 0,
        "content_hash": None,
        "flags": 1,
        "seq_id": 42,
        "source_content": "",
        "content_revision": content_revision,
        "index_status": "ready",
        "last_editor_id": "",
        "context_header": "",
        "created_at": _NOW,
        "updated_at": _NOW,
        "deleted_at": None,
    }
    row.update(overrides)
    return row


class _FakeMappings:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def all(self) -> list[dict[str, object]]:
        return self._rows

    def first(self) -> dict[str, object] | None:
        return self._rows[0] if self._rows else None


class _FakeResult:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def mappings(self) -> _FakeMappings:
        return _FakeMappings(self._rows)

    def scalar_one(self) -> object:
        return 0


class _FakeSession:
    """Records executed SQL and serves canned rows keyed by SQL prefix."""

    def __init__(self, rows_by_prefix: dict[str, list[dict[str, object]]]) -> None:
        self.executed: list[str] = []
        self._rows_by_prefix = rows_by_prefix

    async def execute(self, stmt: TextClause) -> _FakeResult:
        sql = stmt.text
        self.executed.append(sql)
        for prefix, rows in self._rows_by_prefix.items():
            if sql.lstrip().startswith(prefix):
                return _FakeResult(rows)
        return _FakeResult([])


def _repo(session: _FakeSession) -> ChunkRepository:
    return ChunkRepository(session)  # type: ignore[arg-type]


def _sample_chunk(*, id: str = "chunk-1") -> Chunk:
    return Chunk.model_validate(_row(id=id))


# ── create_many SQL shape ────────────────────────────────────────────


async def test_create_many_builds_multi_row_insert_without_seq_id() -> None:
    session = _FakeSession({})
    repo = _repo(session)

    result = await repo.create_many(
        [
            _sample_chunk(),
            _sample_chunk(id="chunk-2"),
            _sample_chunk(id="chunk-3"),
        ]
    )

    assert isinstance(result, list)
    assert session.executed, "expected an INSERT statement to be recorded"
    sql = session.executed[0]
    assert sql.lstrip().startswith("insert into chunks")
    assert '"seq_id"' not in sql
    assert '"content"' in sql
    assert '"tenant_id"' in sql
    assert '"created_at"' in sql
    # One value group per row, three rows total.
    assert sql.count("), (") == 2
    # No seq_id bindparam was generated.
    assert ":seq_id_" not in sql


async def test_create_many_empty_returns_empty() -> None:
    session = _FakeSession({})
    repo = _repo(session)

    assert await repo.create_many([]) == []
    assert session.executed == []


def test_insert_sql_column_list_excludes_db_generated_seq_id() -> None:
    assert "seq_id" not in Chunk.insert_sql_column_list()
    assert "seq_id" in Chunk.column_fields()


# ── update_document_chunk validation ─────────────────────────────────


async def test_update_document_chunk_rejects_empty_content() -> None:
    session = _FakeSession({"select * from": [_row()]})
    repo = _repo(session)

    with pytest.raises(ValidationError) as excinfo:
        await repo.update_document_chunk(
            tenant_id=1,
            chunk_id="chunk-1",
            content="   ",
            is_enabled=None,
            expected_revision=0,
            last_editor_id="usr-1",
            now=_NOW,
        )
    assert excinfo.value.code == "chunk.content_empty"


async def test_update_document_chunk_rejects_non_text_chunk() -> None:
    session = _FakeSession({"select * from": [_row(chunk_type="faq")]})
    repo = _repo(session)

    with pytest.raises(ValidationError) as excinfo:
        await repo.update_document_chunk(
            tenant_id=1,
            chunk_id="chunk-1",
            content="edit",
            is_enabled=None,
            expected_revision=0,
            last_editor_id="usr-1",
            now=_NOW,
        )
    assert excinfo.value.code == "chunk.not_editable"


async def test_update_document_chunk_conflicts_on_stale_revision() -> None:
    session = _FakeSession({"select * from": [_row(content_revision=3)]})
    repo = _repo(session)

    with pytest.raises(ConflictError) as excinfo:
        await repo.update_document_chunk(
            tenant_id=1,
            chunk_id="chunk-1",
            content="edit",
            is_enabled=None,
            expected_revision=0,
            last_editor_id="usr-1",
            now=_NOW,
        )
    assert excinfo.value.code == "chunk.revision_conflict"


async def test_update_document_chunk_noop_skips_write() -> None:
    session = _FakeSession({"select * from": [_row(content="same body")]})
    repo = _repo(session)

    result = await repo.update_document_chunk(
        tenant_id=1,
        chunk_id="chunk-1",
        content="same body",
        is_enabled=True,
        expected_revision=0,
        last_editor_id="usr-1",
        now=_NOW,
    )

    assert result.content == "same body"
    assert result.content_revision == 0
    assert len(session.executed) == 1, "no-op must not issue an UPDATE"


async def test_update_document_chunk_issues_guarded_update() -> None:
    updated = _row(
        content="new body",
        source_content="original body",
        content_revision=1,
        index_status="processing",
        last_editor_id="usr-1",
    )
    session = _FakeSession(
        {
            "select * from": [_row()],
            "update chunks": [updated],
        }
    )
    repo = _repo(session)

    result = await repo.update_document_chunk(
        tenant_id=1,
        chunk_id="chunk-1",
        content="  new body  ",
        is_enabled=None,
        expected_revision=0,
        last_editor_id="usr-1",
        now=_NOW,
    )

    assert result.content == "new body"
    assert result.content_revision == 1
    assert result.index_status == "processing"
    assert result.last_editor_id == "usr-1"

    update_sql = [s for s in session.executed if s.lstrip().startswith("update chunks")]
    assert len(update_sql) == 1
    assert "content_revision = :expected_revision" in update_sql[0]
    assert "content_revision = :content_revision" in update_sql[0]
    assert "tenant_id = :tenant_id" in update_sql[0]
    assert "source_content = :source_content" in update_sql[0]


async def test_update_document_chunk_trims_content() -> None:
    updated = _row(content="revised", content_revision=1)
    session = _FakeSession(
        {
            "select * from": [_row(content="original")],
            "update chunks": [updated],
        }
    )
    repo = _repo(session)

    result = await repo.update_document_chunk(
        tenant_id=1,
        chunk_id="chunk-1",
        content="  revised  ",
        is_enabled=None,
        expected_revision=0,
        last_editor_id="usr-1",
        now=_NOW,
    )

    assert result.content == "revised"
