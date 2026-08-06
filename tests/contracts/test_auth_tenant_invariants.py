"""Auth + tenant contract invariants.

Asserts the frozen Pydantic contracts match the wire schema captured
in ``fixtures/auth_tenant_responses.json``:

- every contract model is frozen (immutable wire shape);
- the contract's serialized field names exactly equal the fixture's
  expected field-name set (no drift in either direction);
- no fixture key is orphaned (every documented contract is covered).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from tests.contracts.auth_tenant_contract import (
    ALL_AUTH_TENANT_CONTRACTS,
    load_fixture_fields,
    model_wire_fields,
)

_NAMES = {name for name, _, _ in ALL_AUTH_TENANT_CONTRACTS}


def _fixture() -> dict[str, list[str]]:
    return load_fixture_fields()


def test_every_contract_is_frozen() -> None:
    for name, model, _endpoint in ALL_AUTH_TENANT_CONTRACTS:
        assert issubclass(model, BaseModel), name
        assert model.model_config.get("frozen") is True, f"{name} is not frozen"


@pytest.mark.parametrize(
    ("name", "model", "endpoint"),
    ALL_AUTH_TENANT_CONTRACTS,
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_wire_fields_match_fixture(name: str, model: type, endpoint: str) -> None:
    fixture = _fixture()
    assert name in fixture, f"contract '{name}' ({endpoint}) is missing from the fixture file"
    expected = set(fixture[name])
    actual = set(model_wire_fields(model))
    assert actual == expected, (
        f"{name}: wire fields diverge from captured fixture.\n"
        f"  missing from contract: {sorted(expected - actual)}\n"
        f"  extra in contract:     {sorted(actual - expected)}"
    )


def test_no_orphaned_fixture_entries() -> None:
    fixture = _fixture()
    orphaned = sorted(set(fixture) - _NAMES)
    assert not orphaned, f"fixture keys without a contract model: {orphaned}"


def test_fixture_covers_all_auth_tenant_contracts() -> None:
    fixture = _fixture()
    uncovered = sorted(_NAMES - set(fixture))
    assert not uncovered, f"contracts without a fixture entry: {uncovered}"


__all__ = []
