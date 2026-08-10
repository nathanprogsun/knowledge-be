"""Unit tests for :mod:`src.db.dao.im_channel_repository`.

Non-DB tests: exercise the generated SQL text (via a stub session that
records statements) so the tenant scoping, the soft-delete filters, and
the bot-identity predicate stay pinned without a database. The real SQL
round-trip is covered by the integration suite.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.sql.expression import TextClause

from src.db.dao.im_channel_repository import IMChannelRepository
from src.db.models.im_channel import IMChannel

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _row(
    *,
    id: str = "im-1",
    tenant_id: int = 1,
    agent_id: str = "agent-1",
    platform: str = "feishu",
    **overrides: object,
) -> dict[str, object]:
    row = {
        "id": id,
        "tenant_id": tenant_id,
        "agent_id": agent_id,
        "platform": platform,
        "name": "Support bot",
        "enabled": True,
        "mode": "websocket",
        "output_mode": "stream",
        "knowledge_base_id": "",
        "bot_identity": "feishu:cli_abc",
        "session_mode": "user",
        "credentials": {},
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


def _repo(session: _FakeSession) -> IMChannelRepository:
    return IMChannelRepository(session)  # type: ignore[arg-type]


def _sample_channel(*, id: str = "im-1") -> IMChannel:
    return IMChannel.model_validate(_row(id=id))


# ── insert column list ───────────────────────────────────────────────


def test_insert_sql_column_list_includes_caller_assigned_id() -> None:
    assert "id" in IMChannel.insert_sql_column_list()
    assert "credentials" in IMChannel.insert_sql_column_list()
    assert "bot_identity" in IMChannel.insert_sql_column_list()


# ── reads ─────────────────────────────────────────────────────────────


async def test_get_by_id_scopes_by_tenant() -> None:
    session = _FakeSession({"select * from im_channels": [_row()]})
    repo = _repo(session)

    result = await repo.get_by_id(tenant_id=1, channel_id="im-1")

    assert result is not None
    assert result.id == "im-1"
    sql = session.executed[0]
    assert '"id" = :id' in sql
    assert '"tenant_id" = :tenant_id' in sql
    assert "deleted_at is null" in sql


async def test_get_by_id_global_ignores_tenant() -> None:
    session = _FakeSession({"select * from im_channels": [_row()]})
    repo = _repo(session)

    result = await repo.get_by_id_global(channel_id="im-1")

    assert result is not None
    sql = session.executed[0]
    assert '"id" = :id' in sql
    assert "tenant_id" not in sql
    assert "deleted_at is null" in sql


async def test_list_by_agent_orders_newest_first() -> None:
    session = _FakeSession({"select * from im_channels": [_row()]})
    repo = _repo(session)

    result = await repo.list_by_agent(tenant_id=1, agent_id="agent-1")

    assert len(result) == 1
    sql = session.executed[0]
    assert "tenant_id = :tenant_id" in sql
    assert "agent_id = :agent_id" in sql
    assert "deleted_at is null" in sql
    assert "order by created_at desc" in sql


async def test_list_by_tenant_orders_newest_first() -> None:
    session = _FakeSession({"select * from im_channels": [_row()]})
    repo = _repo(session)

    result = await repo.list_by_tenant(tenant_id=1)

    assert len(result) == 1
    sql = session.executed[0]
    assert "tenant_id = :tenant_id" in sql
    assert "deleted_at is null" in sql
    assert "order by created_at desc" in sql


async def test_list_enabled_filters_enabled_live_rows() -> None:
    session = _FakeSession({"select * from im_channels": [_row()]})
    repo = _repo(session)

    result = await repo.list_enabled()

    assert len(result) == 1
    sql = session.executed[0]
    assert "enabled = true" in sql
    assert "deleted_at is null" in sql


async def test_find_by_bot_identity_excludes_own_id() -> None:
    session = _FakeSession({"select * from im_channels": [_row()]})
    repo = _repo(session)

    result = await repo.find_by_bot_identity("feishu:cli_abc", exclude_id="im-1")

    assert result is not None
    sql = session.executed[0]
    assert "bot_identity = :bot_identity" in sql
    assert "id != :exclude_id" in sql
    assert "deleted_at is null" in sql


async def test_find_by_bot_identity_without_exclude() -> None:
    session = _FakeSession({"select * from im_channels": [_row()]})
    repo = _repo(session)

    result = await repo.find_by_bot_identity("feishu:cli_abc")

    assert result is not None
    sql = session.executed[0]
    assert "bot_identity = :bot_identity" in sql
    assert "exclude_id" not in sql


# ── mutations ────────────────────────────────────────────────────────


async def test_create_inserts_row() -> None:
    session = _FakeSession({"insert into im_channels": [_row()]})
    repo = _repo(session)

    result = await repo.create(_sample_channel())

    assert result.id == "im-1"
    assert session.executed[0].lstrip().startswith("insert into im_channels")


async def test_update_overwrites_mutable_columns() -> None:
    session = _FakeSession({"update im_channels": [_row(name="Renamed")]})
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

    affected = await repo.soft_delete(channel_id="im-1", tenant_id=1, now=_NOW)

    assert affected is True
    sql = session.executed[0]
    assert "set deleted_at = :now, updated_at = :now" in sql
    assert "id = :channel_id" in sql
    assert "tenant_id = :tenant_id" in sql
    assert "deleted_at is null" in sql


async def test_soft_delete_by_agent_returns_rowcount() -> None:
    session = _FakeSession({})
    repo = _repo(session)

    affected = await repo.soft_delete_by_agent(agent_id="agent-1", tenant_id=1, now=_NOW)

    assert affected == 1
    sql = session.executed[0]
    assert "agent_id = :agent_id" in sql
    assert "tenant_id = :tenant_id" in sql


async def test_toggle_enabled_flips_flag() -> None:
    session = _FakeSession({"update im_channels": [_row(enabled=False)]})
    repo = _repo(session)

    result = await repo.toggle_enabled(channel_id="im-1", tenant_id=1, now=_NOW)

    assert result is not None
    assert result.enabled is False
    sql = session.executed[0]
    assert "enabled = not enabled" in sql
    assert "returning *" in sql
