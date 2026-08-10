"""Unit tests for :mod:`src.db.dao.embed_channel_repository`.

Non-DB tests: exercise the generated SQL text (via a stub session that
records statements) so the tenant scoping and the soft-delete filters
stay pinned without a database. The real SQL round-trip is covered by
the integration suite.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.sql.expression import TextClause

from src.db.dao.embed_channel_repository import EmbedChannelRepository
from src.db.models.embed_channel import EmbedChannel

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _row(
    *,
    id: str = "embed-1",
    tenant_id: int = 1,
    agent_id: str = "builtin-quick-answer",
    **overrides: object,
) -> dict[str, object]:
    row = {
        "id": id,
        "tenant_id": tenant_id,
        "agent_id": agent_id,
        "name": "Web widget",
        "enabled": True,
        "publish_token": "pub_abc",
        "allowed_origins": ["https://example.com"],
        "welcome_message": "Hi!",
        "rate_limit_per_minute": 30,
        "rate_limit_per_day": 10000,
        "primary_color": "#1f6feb",
        "page_title": "Support",
        "header_title_mode": "channel",
        "show_suggested_questions": True,
        "widget_position": "bottom-right",
        "allow_web_search": False,
        "allow_file_upload": False,
        "default_locale": "",
        "webhook_url": "",
        "webhook_secret": "whsec_xyz",
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
    def __init__(self, rows: list[dict[str, object]], rowcount: int = 1) -> None:
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


def _repo(session: _FakeSession) -> EmbedChannelRepository:
    return EmbedChannelRepository(session)  # type: ignore[arg-type]


def _sample_channel(*, id: str = "embed-1") -> EmbedChannel:
    return EmbedChannel.model_validate(_row(id=id))


# ── insert column list ───────────────────────────────────────────────


def test_insert_sql_column_list_includes_caller_assigned_id() -> None:
    assert "id" in EmbedChannel.insert_sql_column_list()
    assert "allowed_origins" in EmbedChannel.insert_sql_column_list()
    assert "publish_token" in EmbedChannel.insert_sql_column_list()


# ── reads ─────────────────────────────────────────────────────────────


async def test_get_by_id_uses_primary_key_only() -> None:
    session = _FakeSession({"select * from embed_channels": [_row()]})
    repo = _repo(session)

    result = await repo.get_by_id(channel_id="embed-1")

    assert result is not None
    assert result.id == "embed-1"
    sql = session.executed[0]
    assert '"id" = :id' in sql
    assert "tenant_id" not in sql
    assert "deleted_at is null" in sql


async def test_get_by_publish_token_filters_live_rows() -> None:
    session = _FakeSession({"select * from embed_channels": [_row()]})
    repo = _repo(session)

    result = await repo.get_by_publish_token("pub_abc")

    assert result is not None
    sql = session.executed[0]
    assert '"publish_token" = :publish_token' in sql
    assert "deleted_at is null" in sql


async def test_list_by_agent_orders_newest_first() -> None:
    session = _FakeSession({"select * from embed_channels": [_row()]})
    repo = _repo(session)

    result = await repo.list_by_agent(tenant_id=1, agent_id="builtin-quick-answer")

    assert len(result) == 1
    sql = session.executed[0]
    assert "tenant_id = :tenant_id" in sql
    assert "agent_id = :agent_id" in sql
    assert "deleted_at is null" in sql
    assert "order by created_at desc" in sql


async def test_list_by_tenant_orders_newest_first() -> None:
    session = _FakeSession({"select * from embed_channels": [_row()]})
    repo = _repo(session)

    result = await repo.list_by_tenant(tenant_id=1)

    assert len(result) == 1
    sql = session.executed[0]
    assert "tenant_id = :tenant_id" in sql
    assert "deleted_at is null" in sql
    assert "order by created_at desc" in sql


# ── mutations ────────────────────────────────────────────────────────


async def test_create_inserts_row() -> None:
    session = _FakeSession({"insert into embed_channels": [_row()]})
    repo = _repo(session)

    result = await repo.create(_sample_channel())

    assert result.id == "embed-1"
    assert session.executed[0].lstrip().startswith("insert into embed_channels")


async def test_update_overwrites_mutable_columns() -> None:
    session = _FakeSession({"update embed_channels": [_row(name="Renamed")]})
    repo = _repo(session)

    result = await repo.update(_sample_channel())

    assert result.name == "Renamed"
    sql = session.executed[0]
    assert '"name" = :u_name' in sql
    assert 'where "id" = :id' in sql
    assert "deleted_at is null" in sql


async def test_soft_delete_marks_row_deleted() -> None:
    session = _FakeSession({})
    repo = _repo(session)

    affected = await repo.soft_delete(channel_id="embed-1", tenant_id=1, now=_NOW)

    assert affected is True
    sql = session.executed[0]
    assert "set deleted_at = :now, updated_at = :now" in sql
    assert "id = :channel_id" in sql
    assert "tenant_id = :tenant_id" in sql
    assert "deleted_at is null" in sql
