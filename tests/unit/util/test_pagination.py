"""Unit tests for :mod:`src.common.pagination`.

Field names are part of the public HTTP contract: the request uses
``page`` / ``page_size`` (capped at 100) and the response carries the
rows under ``data`` (not ``items``). These tests pin both.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.common.pagination import Pagination, PaginationResponse

# ── Pagination request ──────────────────────────────────────────────


def test_pagination_defaults() -> None:
    p = Pagination()
    assert p.page == 1
    assert p.page_size == 20


def test_pagination_accepts_valid_values() -> None:
    p = Pagination(page=3, page_size=50)
    assert p.page == 3
    assert p.page_size == 50


@pytest.mark.parametrize("page_size", [1, 100])
def test_pagination_accepts_page_size_boundaries(page_size: int) -> None:
    assert Pagination(page_size=page_size).page_size == page_size


@pytest.mark.parametrize("page_size", [0, 101])
def test_pagination_rejects_page_size_out_of_range(page_size: int) -> None:
    with pytest.raises(ValidationError):
        Pagination(page_size=page_size)


def test_pagination_rejects_page_below_one() -> None:
    with pytest.raises(ValidationError):
        Pagination(page=0)


def test_pagination_is_frozen() -> None:
    p = Pagination()
    with pytest.raises(ValidationError):
        p.page = 5  # type: ignore[misc]


# ── PaginationResponse ──────────────────────────────────────────────


def test_pagination_response_shape() -> None:
    resp = PaginationResponse(total=5, page=1, page_size=2, data=["a", "b"])
    assert resp.total == 5
    assert resp.page == 1
    assert resp.page_size == 2
    assert resp.data == ["a", "b"]


def test_pagination_response_data_field_not_items() -> None:
    # Pins the upstream-contract field name: ``data``, not ``items``.
    assert "data" in PaginationResponse.model_fields
    assert "items" not in PaginationResponse.model_fields


def test_pagination_response_is_generic() -> None:
    resp: PaginationResponse[int] = PaginationResponse(total=2, page=1, page_size=2, data=[1, 2])
    assert resp.data == [1, 2]


def test_pagination_response_is_frozen() -> None:
    resp = PaginationResponse(total=1, page=1, page_size=1, data=["x"])
    with pytest.raises(ValidationError):
        resp.total = 99  # type: ignore[misc]
