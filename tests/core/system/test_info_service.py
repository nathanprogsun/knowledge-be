"""Unit tests for ``SystemInfoService``.

The service is exercised against an in-memory ``AsyncMock`` session
that mimics the two SQL queries the service issues. The tests cover:

- Happy path: both lookups succeed → ``db_version`` carries the
  revision + server version, ``db_migration_error`` is ``None``,
  ``uptime_seconds`` reflects the elapsed wall time, ``started_at``
  matches the lifespan value.
- Alembic lookup fails → ``db_migration_error`` carries the error
  message and ``db_version`` falls back to ``"unknown"``.
- Postgres server lookup fails → ``db_migration_error`` carries the
  error message and ``db_version`` falls back to the alembic revision.
- ``started_at`` is ``None`` (lifespan bypassed) → uptime is 0 and
  ``started_at`` falls back to ``now``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from src.core.system.info_service import SystemInfoService


def _make_session(
    *,
    revision: object,
    server_version: object,
    revision_error: Exception | None = None,
    server_error: Exception | None = None,
) -> AsyncMock:
    """Build an ``AsyncMock`` session that returns canned scalar results."""
    session = AsyncMock()
    revision_result = MagicMock()
    revision_result.scalar_one_or_none.return_value = revision
    server_result = MagicMock()
    server_result.scalar_one_or_none.return_value = server_version

    state = {"call_count": 0}

    async def _execute(_statement: object) -> MagicMock:
        state["call_count"] += 1
        # First call hits ``alembic_version``, second hits ``version()``.
        if state["call_count"] == 1:
            if revision_error is not None:
                raise revision_error
            return revision_result
        if server_error is not None:
            raise server_error
        return server_result

    session.execute.side_effect = _execute
    return session


class TestSystemInfoService:
    async def test_get_info_returns_build_and_db_metadata(self) -> None:
        started = datetime.now(UTC) - timedelta(seconds=42)
        session = _make_session(
            revision="0024_knowledge_processing_spans",
            server_version="PostgreSQL 15.4",
        )
        svc = SystemInfoService(session=session, started_at=started)

        snapshot = await svc.get_info()

        # Build metadata populated.
        assert snapshot.info.version == "0.0.0"
        assert snapshot.info.edition == "standard"
        assert snapshot.info.commit_id == "unknown"
        assert snapshot.info.go_version.startswith("cpython")
        # DB label: revision + server version.
        assert snapshot.info.db_version == "0024_knowledge_processing_spans (PostgreSQL 15.4)"
        assert snapshot.db_migration_error is None
        # Uptime reflects the elapsed wall time.
        assert 40 <= snapshot.uptime_seconds <= 60
        # ``started_at`` is timezone-aware UTC.
        assert snapshot.started_at.tzinfo is not None
        assert snapshot.started_at.utcoffset() == timedelta(0)

    async def test_get_info_handles_alembic_failure(self) -> None:
        session = _make_session(
            revision=None,
            server_version="PostgreSQL 15.4",
            revision_error=RuntimeError("table missing"),
        )
        svc = SystemInfoService(session=session, started_at=None)

        snapshot = await svc.get_info()

        assert snapshot.info.db_version == "unknown"
        assert snapshot.db_migration_error is not None
        assert "alembic_version lookup failed" in snapshot.db_migration_error
        assert "table missing" in snapshot.db_migration_error

    async def test_get_info_handles_server_version_failure(self) -> None:
        session = _make_session(
            revision="0024_knowledge_processing_spans",
            server_version=None,
            server_error=RuntimeError("connection lost"),
        )
        svc = SystemInfoService(session=session, started_at=None)

        snapshot = await svc.get_info()

        # Alembic revision still appears in the label.
        assert "0024_knowledge_processing_spans" in snapshot.info.db_version
        assert snapshot.db_migration_error is not None
        assert "server version lookup failed" in snapshot.db_migration_error

    async def test_get_info_without_lifespan_started_at(self) -> None:
        session = _make_session(
            revision="0024_knowledge_processing_spans",
            server_version="PostgreSQL 15.4",
        )
        svc = SystemInfoService(session=session, started_at=None)

        snapshot = await svc.get_info()

        # ``uptime_seconds`` is 0 and ``started_at`` is recent.
        assert snapshot.uptime_seconds == 0
        now = datetime.now(UTC)
        delta = now - snapshot.started_at
        assert delta.total_seconds() < 5

    async def test_get_info_normalizes_naive_started_at(self) -> None:
        # Naive datetimes (no tzinfo) are coerced to UTC — mirrors the
        # behavior in case the lifespan code stores a naive ``datetime``.
        started = (datetime.now(UTC) - timedelta(seconds=10)).replace(tzinfo=None)
        session = _make_session(
            revision="0024_knowledge_processing_spans",
            server_version="PostgreSQL 15.4",
        )
        svc = SystemInfoService(session=session, started_at=started)

        snapshot = await svc.get_info()

        assert snapshot.started_at.tzinfo is not None
        assert snapshot.started_at.utcoffset() == timedelta(0)
        assert snapshot.uptime_seconds >= 0


__all__ = []
