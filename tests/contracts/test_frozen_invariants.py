"""Invariants for every contract in `src.core.contracts`.

Each PR-0.5 contract module declares a set of frozen Pydantic models that
mirror the HTTP wire shape. These tests guard three invariants:

1. `model_config["frozen"]` is `True` — instances are immutable.
2. The model can be constructed with only its required fields filled.
3. Mutating any field of an instance raises (`ValidationError` in v2).

The tests are auto-parameterised across every exported class in every
contract module — adding a new model anywhere under `src.core.contracts`
automatically exercises these invariants without further test edits.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, get_type_hints

import pytest
from pydantic import BaseModel, ValidationError

from src.common.pagination import Pagination, PaginationResponse
from src.core.contracts import (
    agents,
    auth,
    evaluation,
    infra,
    knowledge,
    organizations,
    sessions,
    shared,
    system,
    tenants,
)

ALL_MODULES = (
    agents,
    auth,
    evaluation,
    infra,
    knowledge,
    organizations,
    sessions,
    shared,
    system,
    tenants,
)

# Pagination / PaginationResponse live in `src.common.pagination` and are
# reused by both domain code and the HTTP layer; they get the same three
# invariants applied as every contract model.
EXTRA_MODELS: tuple[tuple[str, str, type[BaseModel]], ...] = (
    ("src.common.pagination", "Pagination", Pagination),
    ("src.common.pagination", "PaginationResponse", PaginationResponse),
)


def _all_models() -> list[tuple[str, str, type[BaseModel]]]:
    out: list[tuple[str, str, type[BaseModel]]] = []
    for mod in ALL_MODULES:
        for name in mod.__all__:
            cls = getattr(mod, name)
            if isinstance(cls, type) and issubclass(cls, BaseModel):
                out.append((mod.__name__, name, cls))
    out.extend(EXTRA_MODELS)
    return out


def _dummy_for(annotation: Any) -> Any:
    """Build a minimal valid value for a Pydantic annotation.

    Recurses into nested contract models so a single dummy can satisfy
    fields whose type is another contract class. Annotations are resolved
    via ``typing.get_type_hints`` (which materialises ``ForwardRef``s from
    the ``from __future__ import annotations`` strings).
    """
    origin = getattr(annotation, "__origin__", None)
    if origin is list:
        return []
    if origin is dict:
        return {}
    if origin is Literal:
        # Pydantic Literal types expose their choices via __args__; pick the
        # first member so required-only construction succeeds.
        args = getattr(annotation, "__args__", ())
        if args:
            return args[0]
    if annotation is str:
        return "x"
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    if annotation is bool:
        return False
    if annotation is datetime:
        return datetime(2020, 1, 1, tzinfo=UTC)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _required_only(annotation)
    return None


def _resolved_hints(model: type[BaseModel]) -> dict[str, Any]:
    return get_type_hints(model)


def _required_only(model: type[BaseModel]) -> dict[str, Any]:
    """Return kwargs covering every required field of `model`."""
    kwargs: dict[str, Any] = {}
    hints = _resolved_hints(model)
    for fname, field in model.model_fields.items():
        if field.is_required():
            kwargs[fname] = _dummy_for(hints.get(fname, field.annotation))
    return kwargs


def _first_required_field(model: type[BaseModel]) -> str | None:
    for fname, field in model.model_fields.items():
        if field.is_required():
            return fname
    return None


ALL_MODELS = _all_models()


@pytest.mark.parametrize(
    ("module_name", "class_name", "model"),
    ALL_MODELS,
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_model_config_is_frozen(module_name: str, class_name: str, model: type[BaseModel]) -> None:
    assert model.model_config.get("frozen") is True, f"{module_name}.{class_name} is not frozen"


@pytest.mark.parametrize(
    ("module_name", "class_name", "model"),
    ALL_MODELS,
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_required_only_construction(
    module_name: str, class_name: str, model: type[BaseModel]
) -> None:
    kwargs = _required_only(model)
    if not kwargs:
        pytest.skip(f"{class_name} has no required fields")
    instance = model(**kwargs)
    assert isinstance(instance, model)


@pytest.mark.parametrize(
    ("module_name", "class_name", "model"),
    ALL_MODELS,
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_assignment_after_init_raises(
    module_name: str, class_name: str, model: type[BaseModel]
) -> None:
    fname = _first_required_field(model)
    if fname is None:
        pytest.skip(f"{class_name} has no required fields")
    kwargs = _required_only(model)
    instance = model(**kwargs)
    annotation = _resolved_hints(model).get(fname, model.model_fields[fname].annotation)
    with pytest.raises(ValidationError):
        setattr(instance, fname, _dummy_for(annotation))


def test_every_module_is_covered() -> None:
    """Sanity guard: at least one model per module."""
    seen = {m for m, _, _ in ALL_MODELS}
    expected = {mod.__name__ for mod in ALL_MODULES} | {"src.common.pagination"}
    assert seen == expected
