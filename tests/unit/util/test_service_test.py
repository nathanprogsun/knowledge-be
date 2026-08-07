"""Smoke tests for the :class:`ServiceTest` DSL.

The DSL is mostly a thin layer of static helpers, so the tests are
correspondingly small: each one configures an ``AsyncMock`` repository
through a helper, then asserts that the mock now returns the expected
value.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from tests.util.service_test import ServiceTest


class _Demo(ServiceTest):
    """Concrete subclass used to exercise the static helpers."""


async def test_mock_repo_find_by_primary_key() -> None:
    repo = AsyncMock()
    await _Demo.mock_repo_find_by_primary_key(repo, "find_by_id", return_value="X")
    assert await repo.find_by_id(1) == "X"


async def test_mock_repo_insert() -> None:
    repo = AsyncMock()
    await _Demo.mock_repo_insert(repo, return_value={"id": 1})
    assert await repo.insert() == {"id": 1}


async def test_mock_repo_update_by_tenanted_primary_key() -> None:
    repo = AsyncMock()
    await _Demo.mock_repo_update_by_tenanted_primary_key(repo, return_value=True)
    assert await repo.update_by_tenanted_primary_key() is True


async def test_mock_repo_find_by_tenanted_primary_key() -> None:
    repo = AsyncMock()
    await _Demo.mock_repo_find_by_tenanted_primary_key(repo, return_value=[{"id": 1}])
    assert await repo.find_by_tenanted_primary_key() == [{"id": 1}]


async def test_mock_repo_find_by_id() -> None:
    repo = AsyncMock()
    await _Demo.mock_repo_find_by_id(repo, return_value={"id": 7})
    assert await repo.find_by_id(7) == {"id": 7}
