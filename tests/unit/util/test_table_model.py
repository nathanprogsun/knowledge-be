"""Unit tests for :mod:`src.common.table_model`.

``TableModel`` is the frozen-Pydantic row base used by repositories.
Tests cover column / primary-key metadata, INSERT bindparam derivation,
column validation (fail-fast on drift), row hydration, and immutability.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import Field
from pydantic import ValidationError as PydanticValidationError

from src.common.exception import ValidationError
from src.common.json import JsonObject
from src.common.table_model import TableModel


class _DemoRow(TableModel):
    table: ClassVar[str] = "demo"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ("meta",)
    db_generated_columns: ClassVar[tuple[str, ...]] = ("id",)

    id: int
    name: str
    meta: JsonObject = Field(default_factory=dict)


class _DriftRow(TableModel):
    table: ClassVar[str] = "drift"
    primary_keys: ClassVar[tuple[str, ...]] = ("missing",)

    id: int = 0


# ── Read-side metadata ───────────────────────────────────────────────


def test_fq_table_name() -> None:
    assert _DemoRow.fq_table_name() == "demo"


def test_column_fields() -> None:
    assert _DemoRow.column_fields() == ("id", "name", "meta")


def test_ordered_primary_keys() -> None:
    assert _DemoRow.ordered_primary_keys() == ("id",)


def test_get_json_columns() -> None:
    assert _DemoRow.get_json_columns() == ("meta",)


def test_ordered_primary_keys_raises_on_drift() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _DriftRow.ordered_primary_keys()
    assert exc_info.value.code == "db.schema_drift"


# ── Insert-side metadata ─────────────────────────────────────────────


def test_insert_sql_column_list_excludes_db_generated() -> None:
    # ``id`` is db-generated, so it is excluded from INSERT columns.
    assert _DemoRow.insert_sql_column_list() == ("name", "meta")


def test_insert_sql_column_param_list() -> None:
    assert _DemoRow.insert_sql_column_param_list() == (":name", ":meta")


def test_insert_bind_params_omits_db_generated() -> None:
    row = _DemoRow(id=1, name="x", meta={"k": 1})
    assert row.insert_bind_params() == {"name": "x", "meta": {"k": 1}}


# ── Validation helpers ──────────────────────────────────────────────


def test_validate_in_columns_unknown_raises() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _DemoRow.validate_in_columns(["name", "bogus"])
    assert exc_info.value.code == "db.unknown_column"


def test_validate_in_columns_accepts_mapping_keys() -> None:
    _DemoRow.validate_in_columns({"name": "x", "meta": {}})  # no raise


def test_validate_contains_all_primary_keys_missing_raises() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _DemoRow.validate_contains_all_primary_keys(["name"])
    assert exc_info.value.code == "db.missing_primary_key"


def test_validate_contains_all_primary_keys_valid() -> None:
    _DemoRow.validate_contains_all_primary_keys(["id", "name"])  # no raise


# ── Hydration + PK extraction ────────────────────────────────────────


def test_from_row_hydrates() -> None:
    row = _DemoRow.from_row({"id": 7, "name": "y", "meta": {"k": 2}})
    assert row.id == 7
    assert row.name == "y"
    assert row.meta == {"k": 2}


def test_primary_key_to_value() -> None:
    row = _DemoRow(id=9, name="z")
    assert row.primary_key_to_value() == {"id": 9}


# ── Immutability ─────────────────────────────────────────────────────


def test_model_is_frozen() -> None:
    row = _DemoRow(id=1, name="x")
    with pytest.raises(PydanticValidationError):
        row.name = "mutated"  # type: ignore[misc]
