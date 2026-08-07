"""Unit tests for :mod:`src.common.json`.

``JsonObject`` / ``JsonValue`` are type aliases over Pydantic's recursive
JSON type; ``SqlValue`` / ``BindParams`` extend it to cover the
``datetime`` values that SQL bindparam maps carry. The tests verify the
aliases behave as concrete dict types at runtime and round-trip through
a Pydantic model.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel

from src.common.json import BindParams, JsonObject, JsonValue

# ── Aliases ──────────────────────────────────────────────────────────


def test_json_object_is_a_dict() -> None:
    obj: JsonObject = {"a": 1, "b": "two"}
    assert obj["a"] == 1
    assert obj["b"] == "two"
    assert isinstance(obj, dict)


def test_json_value_accepts_recursive_payload() -> None:
    obj: JsonObject = {
        "nested": {"flag": True},
        "items": [1, 2, {"k": None}],
        "name": "x",
    }
    assert obj["nested"]["flag"] is True  # type: ignore[index]
    assert obj["items"][2]["k"] is None  # type: ignore[index]


# ── Pydantic round-trip ─────────────────────────────────────────────


class _Wrapper(BaseModel):
    payload: JsonObject


def test_json_object_round_trips_via_pydantic() -> None:
    model = _Wrapper(payload={"a": 1, "b": [1, 2, {"c": True}]})
    assert model.payload["a"] == 1
    dumped = model.model_dump()
    assert dumped == {"payload": {"a": 1, "b": [1, 2, {"c": True}]}}


# ── SQL bind params ──────────────────────────────────────────────────


def test_bindparams_carries_datetime_and_scalars() -> None:
    ts = datetime.now(UTC)
    params: BindParams = {"ts": ts, "count": 3, "name": "x"}
    assert params["ts"] is ts
    assert params["count"] == 3
    assert params["name"] == "x"


def test_json_value_is_re_exported() -> None:
    # ``JsonValue`` must be importable from src.common.json (the anti-drift
    # rule forbids bare ``object`` / ``Any`` annotations).
    assert JsonValue is not None
