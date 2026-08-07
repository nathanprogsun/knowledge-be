"""Unit tests for :mod:`src.common.session_provider`.

``session_scope`` is the canonical unit-of-work wrapper: commit on clean
exit, rollback on exception, close always. These tests exercise the
transaction semantics against a stub session - no real DB is involved.
"""

from __future__ import annotations

import pytest

from src.common.session_provider import session_scope


class _StubSession:
    """AsyncSession stand-in tracking commit / rollback / close calls."""

    def __init__(self) -> None:
        self.commits = 0
        self.rollbacks = 0
        self.closes = 0

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1

    async def close(self) -> None:
        self.closes += 1


async def test_session_scope_commits_on_clean_exit() -> None:
    stub = _StubSession()
    async with session_scope(lambda: stub) as session:  # type: ignore[arg-type]
        assert session is stub
    assert stub.commits == 1
    assert stub.rollbacks == 0
    assert stub.closes == 1


async def test_session_scope_rolls_back_on_exception() -> None:
    stub = _StubSession()

    with pytest.raises(RuntimeError, match="boom"):
        async with session_scope(lambda: stub):  # type: ignore[arg-type]
            raise RuntimeError("boom")

    assert stub.commits == 0
    assert stub.rollbacks == 1
    assert stub.closes == 1


async def test_session_scope_closes_only_once() -> None:
    stub = _StubSession()
    async with session_scope(lambda: stub):  # type: ignore[arg-type]
        pass
    assert stub.closes == 1
