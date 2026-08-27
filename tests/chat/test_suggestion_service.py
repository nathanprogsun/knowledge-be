"""Unit tests for the message suggestion domain.

Covers the domain vocabulary constants, the service surface (stub
methods raise ``NotImplementedFeatureError`` until the generation pipeline
lands), and the persistence layer via a stub session that records SQL —
so the cache-key lookups, the lease-guarded acquisition, and the
session / message deletes stay pinned without a database.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.sql.expression import TextClause

from src.common.exception import NotImplementedFeatureError
from src.core.chat.messages.suggestion_service import (
    SUGGESTION_EVENT_CLICK,
    SUGGESTION_EVENT_DISMISS,
    SUGGESTION_EVENT_IMPRESSION,
    SUGGESTION_EVENT_REGENERATE,
    SUGGESTION_PLACEMENT_AFTER_ANSWER,
    SUGGESTION_STATUS_FAILED,
    SUGGESTION_STATUS_GENERATING,
    SUGGESTION_STATUS_READY,
    SUGGESTION_STATUS_SUPPRESSED,
    MessageSuggestionService,
)
from src.db.dao.message_suggestion_repository import MessageSuggestionRepository
from src.db.models.message_suggestion import MessageSuggestionSet

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _row(
    *,
    id: str = "set-1",
    tenant_id: int = 1,
    session_id: str = "sess-1",
    assistant_message_id: str = "msg-1",
    status: str = SUGGESTION_STATUS_READY,
    **overrides: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "id": id,
        "tenant_id": tenant_id,
        "session_id": session_id,
        "assistant_message_id": assistant_message_id,
        "agent_id": "",
        "agent_tenant_id": 0,
        "placement": SUGGESTION_PLACEMENT_AFTER_ANSWER,
        "config_hash": "cfg-1",
        "locale": "en",
        "status": status,
        "allow_regenerate": False,
        "suppression_reason": "",
        "questions": [],
        "model_id": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_ms": 0,
        "error_code": "",
        "lease_until": None,
        "generated_at": None,
        "created_at": _NOW,
        "updated_at": _NOW,
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
    def __init__(
        self,
        rows: list[dict[str, object]],
        rowcount: int = 1,
    ) -> None:
        self._rows = rows
        self.rowcount = rowcount

    def mappings(self) -> _FakeMappings:
        return _FakeMappings(self._rows)


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


def _repo(session: _FakeSession) -> MessageSuggestionRepository:
    return MessageSuggestionRepository(session)  # type: ignore[arg-type]


def _sample_set(*, id: str = "set-1") -> MessageSuggestionSet:
    return MessageSuggestionSet.model_validate(_row(id=id))


# ── vocabulary ────────────────────────────────────────────────────────


def test_placement_and_status_vocabulary() -> None:
    assert SUGGESTION_PLACEMENT_AFTER_ANSWER == "after_answer"
    assert SUGGESTION_STATUS_GENERATING == "generating"
    assert SUGGESTION_STATUS_READY == "ready"
    assert SUGGESTION_STATUS_SUPPRESSED == "suppressed"
    assert SUGGESTION_STATUS_FAILED == "failed"


def test_event_vocabulary() -> None:
    assert SUGGESTION_EVENT_IMPRESSION == "impression"
    assert SUGGESTION_EVENT_CLICK == "click"
    assert SUGGESTION_EVENT_DISMISS == "dismiss"
    assert SUGGESTION_EVENT_REGENERATE == "regenerate"


# ── service surface (stub) ────────────────────────────────────────────


async def test_service_stub_methods_raise_not_implemented() -> None:
    service = MessageSuggestionService()

    with pytest.raises(NotImplementedFeatureError):
        await service.ensure_follow_ups(
            session_id="sess-1",
            assistant_message_id="msg-1",
            regenerate=False,
        )
    with pytest.raises(NotImplementedFeatureError):
        await service.get_follow_ups(session_id="sess-1", assistant_message_id="msg-1")
    with pytest.raises(NotImplementedFeatureError):
        await service.record_event(
            session_id="sess-1",
            suggestion_set_id="set-1",
            question_id="q-1",
            event_type=SUGGESTION_EVENT_CLICK,
        )
    with pytest.raises(NotImplementedFeatureError):
        await service.validate_attribution(
            session_id="sess-1",
            query="follow-up",
            suggestion_set_id="set-1",
            question_id="q-1",
        )


# ── create ────────────────────────────────────────────────────────────


async def test_create_builds_insert_with_all_columns() -> None:
    session = _FakeSession({"insert into message_suggestion_sets": [_row()]})
    repo = _repo(session)

    result = await repo.create(_sample_set())

    assert result.id == "set-1"
    sql = session.executed[0]
    assert sql.lstrip().startswith("insert into message_suggestion_sets")
    assert '"tenant_id"' in sql
    assert '"assistant_message_id"' in sql
    assert '"questions"' in sql
    assert "returning *" in sql


# ── get ───────────────────────────────────────────────────────────────


async def test_get_by_id_filters_tenant_and_session() -> None:
    session = _FakeSession({"select * from": [_row()]})
    repo = _repo(session)

    result = await repo.get_by_id(tenant_id=1, session_id="sess-1", id="set-1")

    assert result is not None
    sql = session.executed[0]
    assert '"id" = :id' in sql
    assert '"tenant_id" = :tenant_id' in sql
    assert '"session_id" = :session_id' in sql


async def test_get_by_cache_key_filters_all_key_columns() -> None:
    session = _FakeSession({"select * from": [_row()]})
    repo = _repo(session)

    result = await repo.get_by_cache_key(
        tenant_id=1,
        assistant_message_id="msg-1",
        placement=SUGGESTION_PLACEMENT_AFTER_ANSWER,
        config_hash="cfg-1",
        locale="en",
    )

    assert result is not None
    sql = session.executed[0]
    assert '"assistant_message_id" = :assistant_message_id' in sql
    assert '"placement" = :placement' in sql
    assert '"config_hash" = :config_hash' in sql
    assert '"locale" = :locale' in sql


# ── acquire_generation ────────────────────────────────────────────────


async def test_acquire_generation_inserts_fresh_row() -> None:
    session = _FakeSession(
        {"insert into message_suggestion_sets": [_row(status=SUGGESTION_STATUS_GENERATING)]}
    )
    repo = _repo(session)

    result, acquired = await repo.acquire_generation(_sample_set(), regenerate=False, now=_NOW)

    assert acquired is True
    assert result is not None
    assert result.status == SUGGESTION_STATUS_GENERATING
    sql = session.executed[0]
    assert "on conflict" in sql
    assert "do nothing" in sql


async def test_acquire_generation_returns_ready_row_without_regenerate() -> None:
    session = _FakeSession(
        {
            "insert into message_suggestion_sets": [],
            "select * from": [_row(status=SUGGESTION_STATUS_READY)],
        }
    )
    repo = _repo(session)

    result, acquired = await repo.acquire_generation(_sample_set(), regenerate=False, now=_NOW)

    assert acquired is False
    assert result is not None
    assert result.status == SUGGESTION_STATUS_READY
    assert len(session.executed) == 2, "insert + cache-key lookup, no UPDATE"


async def test_acquire_generation_reacquires_expired_lease() -> None:
    stale = _row(
        status=SUGGESTION_STATUS_GENERATING,
        lease_until=_NOW - timedelta(minutes=1),
    )
    session = _FakeSession(
        {
            "insert into message_suggestion_sets": [],
            "select * from": [stale],
            "update message_suggestion_sets": [
                _row(status=SUGGESTION_STATUS_GENERATING, lease_until=_NOW + timedelta(minutes=3))
            ],
        }
    )
    repo = _repo(session)

    result, acquired = await repo.acquire_generation(_sample_set(), regenerate=False, now=_NOW)

    assert acquired is True
    assert result is not None
    assert result.status == SUGGESTION_STATUS_GENERATING
    update_sql = [s for s in session.executed if s.lstrip().startswith("update")]
    assert len(update_sql) == 1
    assert "lease_until = :lease_until" in update_sql[0]
    assert "status <> :generating" in update_sql[0]


# ── save / delete ─────────────────────────────────────────────────────


async def test_save_rewrites_mutable_columns() -> None:
    session = _FakeSession(
        {"update message_suggestion_sets": [_row(status=SUGGESTION_STATUS_READY)]}
    )
    repo = _repo(session)

    result = await repo.save(_sample_set())

    assert result is not None
    sql = session.executed[0]
    assert sql.lstrip().startswith("update message_suggestion_sets")
    assert '"status" = :u_status' in sql
    assert '"questions" = :u_questions' in sql
    assert '"id" = :id' in sql


async def test_delete_by_message_id_scopes_tenant_session_message() -> None:
    session = _FakeSession({})
    repo = _repo(session)

    affected = await repo.delete_by_message_id(
        tenant_id=1,
        session_id="sess-1",
        message_id="msg-1",
    )

    assert affected == 1
    sql = session.executed[0]
    assert sql.lstrip().startswith("delete from message_suggestion_sets")
    assert "tenant_id = :tenant_id" in sql
    assert "session_id = :session_id" in sql
    assert "assistant_message_id = :message_id" in sql


async def test_delete_by_session_id_scopes_tenant_session() -> None:
    session = _FakeSession({})
    repo = _repo(session)

    affected = await repo.delete_by_session_id(tenant_id=1, session_id="sess-1")

    assert affected == 1
    sql = session.executed[0]
    assert sql.lstrip().startswith("delete from message_suggestion_sets")
    assert "tenant_id = :tenant_id" in sql
    assert "session_id = :session_id" in sql
