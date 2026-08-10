"""Unit tests for :mod:`src.db.dao.message_repository`.

Non-DB tests: exercise the generated SQL text (via a stub session that
records statements) and the in-memory re-sorting of the recent-window
reads, so the session feed, the scoped writes, and the soft-delete
guards stay pinned without a database.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy.sql.expression import TextClause

from src.common.exception import ValidationError
from src.db.dao.message_repository import MessageRepository
from src.db.models.message import Message

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _row(
    *,
    id: str = "msg-1",
    session_id: str = "sess-1",
    role: str = "user",
    content: str = "hello",
    **overrides: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "id": id,
        "request_id": "req-1",
        "session_id": session_id,
        "role": role,
        "content": content,
        "knowledge_references": [],
        "agent_steps": None,
        "is_completed": True,
        "is_fallback": False,
        "agent_duration_ms": 0,
        "rendered_content": "",
        "channel": "",
        "agent_id": "",
        "agent_tenant_id": 0,
        "model_id": "",
        "execution_context": {},
        "knowledge_id": "",
        "mentioned_items": [],
        "images": [],
        "attachments": [],
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


class _FakeScalars:
    def __init__(self, values: list[object]) -> None:
        self._values = values

    def all(self) -> list[object]:
        return self._values


class _FakeResult:
    def __init__(
        self,
        rows: list[dict[str, object]],
        scalars: list[object] | None = None,
        rowcount: int = 1,
    ) -> None:
        self._rows = rows
        self._scalars = scalars or []
        self.rowcount = rowcount

    def mappings(self) -> _FakeMappings:
        return _FakeMappings(self._rows)

    def scalars(self) -> _FakeScalars:
        return _FakeScalars(self._scalars)


class _FakeSession:
    """Records executed SQL and serves canned rows keyed by SQL prefix."""

    def __init__(
        self,
        rows_by_prefix: dict[str, list[dict[str, object]] | _FakeResult],
    ) -> None:
        self.executed: list[str] = []
        self._rows_by_prefix = rows_by_prefix

    async def execute(self, stmt: TextClause) -> _FakeResult:
        sql = stmt.text
        self.executed.append(sql)
        for prefix, rows in self._rows_by_prefix.items():
            if sql.lstrip().startswith(prefix):
                if isinstance(rows, _FakeResult):
                    return rows
                return _FakeResult(rows)
        return _FakeResult([])


def _repo(session: _FakeSession) -> MessageRepository:
    return MessageRepository(session)  # type: ignore[arg-type]


def _sample_message(*, id: str = "msg-1") -> Message:
    return Message.model_validate(_row(id=id))


# ── create ────────────────────────────────────────────────────────────


async def test_create_builds_insert_with_all_columns() -> None:
    session = _FakeSession({"insert into messages": [_row()]})
    repo = _repo(session)

    result = await repo.create(_sample_message())

    assert result.id == "msg-1"
    assert session.executed, "expected an INSERT statement to be recorded"
    sql = session.executed[0]
    assert sql.lstrip().startswith("insert into messages")
    assert '"id"' in sql
    assert '"session_id"' in sql
    assert '"content"' in sql
    assert '"created_at"' in sql
    assert "returning *" in sql


def test_insert_sql_column_list_includes_caller_assigned_id() -> None:
    assert "id" in Message.insert_sql_column_list()
    assert "session_id" in Message.insert_sql_column_list()
    assert "knowledge_references" in Message.insert_sql_column_list()


# ── get ───────────────────────────────────────────────────────────────


async def test_get_by_id_and_session_filters_both_columns() -> None:
    session = _FakeSession({"select * from": [_row()]})
    repo = _repo(session)

    result = await repo.get_by_id_and_session(session_id="sess-1", message_id="msg-1")

    assert result is not None
    assert result.id == "msg-1"
    sql = session.executed[0]
    assert sql.lstrip().startswith("select * from messages")
    assert '"id" = :id' in sql
    assert '"session_id" = :session_id' in sql
    assert "deleted_at is null" in sql


async def test_get_by_request_id_returns_none_for_empty_request_id() -> None:
    session = _FakeSession({})
    repo = _repo(session)

    result = await repo.get_by_request_id(session_id="sess-1", request_id="")

    assert result is None
    assert session.executed == []


# ── list ─────────────────────────────────────────────────────────────


async def test_list_by_session_builds_paginated_feed() -> None:
    session = _FakeSession({})
    repo = _repo(session)

    await repo.list_by_session("sess-1", page=2, page_size=10)

    sql = session.executed[0]
    assert sql.lstrip().startswith("select * from messages")
    assert "session_id = :session_id" in sql
    assert "deleted_at is null" in sql
    assert "order by created_at asc, id asc" in sql
    assert "limit :limit offset :offset" in sql


async def test_list_recent_by_session_sorts_user_turns_first() -> None:
    assistant = _row(id="msg-2", role="assistant", content="answer")
    user = _row(id="msg-3", role="user", content="follow-up")
    session = _FakeSession({"select * from": [assistant, user]})
    repo = _repo(session)

    result = await repo.list_recent_by_session("sess-1", limit=2)

    assert [m.id for m in result] == ["msg-3", "msg-2"]
    sql = session.executed[0]
    assert "order by created_at desc, id desc" in sql
    assert "limit :limit" in sql


async def test_list_by_session_before_time_filters_created_at() -> None:
    session = _FakeSession({})
    repo = _repo(session)

    await repo.list_by_session_before_time("sess-1", before_time=_NOW, limit=5)

    sql = session.executed[0]
    assert "created_at < :before_time" in sql
    assert "limit :limit" in sql


async def test_get_first_user_message_filters_role() -> None:
    session = _FakeSession({"select * from": [_row()]})
    repo = _repo(session)

    result = await repo.get_first_user_message("sess-1")

    assert result is not None
    sql = session.executed[0]
    assert "role = :role" in sql
    assert "order by created_at asc, id asc limit 1" in sql


async def test_list_knowledge_ids_by_session_projects_distinct_values() -> None:
    session = _FakeSession(
        {"select distinct knowledge_id": _FakeResult([], scalars=["kid-1", "kid-2"])}
    )
    repo = _repo(session)

    result = await repo.list_knowledge_ids_by_session("sess-1")

    assert result == ["kid-1", "kid-2"]
    sql = session.executed[0]
    assert "select distinct knowledge_id" in sql
    assert "knowledge_id <> ''" in sql


# ── update ────────────────────────────────────────────────────────────


async def test_update_scopes_by_id_and_session() -> None:
    session = _FakeSession({"update messages": [_row(content="revised")]})
    repo = _repo(session)

    result = await repo.update(
        session_id="sess-1",
        message_id="msg-1",
        column_to_update={"content": "revised"},
    )

    assert result is not None
    assert result.content == "revised"
    sql = session.executed[0]
    assert sql.lstrip().startswith("update messages")
    assert '"content" = :u_content' in sql
    assert "where id = :id and session_id = :session_id" in sql
    assert "deleted_at is null" in sql
    assert "returning *" in sql


async def test_update_rejects_unknown_column() -> None:
    session = _FakeSession({})
    repo = _repo(session)

    with pytest.raises(ValidationError) as excinfo:
        await repo.update(
            session_id="sess-1",
            message_id="msg-1",
            column_to_update={"nonsense": 1},
        )
    assert excinfo.value.code == "db.unknown_column"
    assert session.executed == []


async def test_update_images_binds_jsonb_column() -> None:
    session = _FakeSession({"update messages": [_row(images=[{"url": "u"}])]})
    repo = _repo(session)

    result = await repo.update_images(
        session_id="sess-1",
        message_id="msg-1",
        images=[{"url": "u"}],
    )

    assert result is not None
    sql = session.executed[0]
    assert '"images" = :u_images' in sql


async def test_update_knowledge_id_scopes_by_id_only() -> None:
    session = _FakeSession({"update messages": [_row(knowledge_id="kid-1")]})
    repo = _repo(session)

    result = await repo.update_knowledge_id(
        message_id="msg-1",
        knowledge_id="kid-1",
        now=_NOW,
    )

    assert result is not None
    assert result.knowledge_id == "kid-1"
    sql = session.executed[0]
    assert "knowledge_id = :knowledge_id" in sql
    assert "where id = :id" in sql
    assert "session_id" not in sql


# ── soft delete ───────────────────────────────────────────────────────


async def test_soft_delete_marks_deleted_at_scoped_by_id_and_session() -> None:
    session = _FakeSession({})
    repo = _repo(session)

    affected = await repo.soft_delete(session_id="sess-1", message_id="msg-1", now=_NOW)

    assert affected is True
    sql = session.executed[0]
    assert sql.lstrip().startswith("update messages")
    assert "deleted_at = :now" in sql
    assert "updated_at = :now" in sql
    assert "where id = :id and session_id = :session_id" in sql
    assert "deleted_at is null" in sql


async def test_soft_delete_by_session_marks_every_message() -> None:
    session = _FakeSession({})
    repo = _repo(session)

    affected = await repo.soft_delete_by_session("sess-1", now=_NOW)

    assert affected == 1
    sql = session.executed[0]
    assert "where session_id = :session_id" in sql
    assert "deleted_at is null" in sql
