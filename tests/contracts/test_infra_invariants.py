"""Infrastructure-domain contract invariants.

Asserts the frozen Pydantic contracts match the wire schema captured
in ``fixtures/infra_responses.json``:

- every contract model is frozen (immutable wire shape);
- the contract's serialized field names exactly equal the fixture's
  expected field-name set (no drift in either direction);
- no fixture key is orphaned (every documented contract is covered).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from tests.contracts.infra_contract import (
    ALL_INFRA_CONTRACTS,
    load_fixture_fields,
    model_wire_fields,
)

_SRC_ROOT = Path(__file__).parents[2] / "src"

_NAMES = {name for name, _, _ in ALL_INFRA_CONTRACTS}


def _fixture() -> dict[str, list[str]]:
    return load_fixture_fields()


def test_every_contract_is_frozen() -> None:
    for name, model, _endpoint in ALL_INFRA_CONTRACTS:
        assert issubclass(model, BaseModel), name
        assert model.model_config.get("frozen") is True, f"{name} is not frozen"


@pytest.mark.parametrize(
    ("name", "model", "endpoint"),
    ALL_INFRA_CONTRACTS,
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


def test_fixture_covers_all_infra_contracts() -> None:
    fixture = _fixture()
    uncovered = sorted(_NAMES - set(fixture))
    assert not uncovered, f"contracts without a fixture entry: {uncovered}"


def test_placeholder_sweep() -> None:
    """Infrastructure placeholder audit.

    - The MCP live transport must be what production wiring installs;
      ``StaticConnectivityProbe`` / ``StaticDiscoveryProvider`` are
      test doubles only and must not be referenced from the wiring.
    - The WebSearch HTTP clients are pending their own rollout; until
      they land, the ``_NotImplementedClient`` marker must stay
      confined to its owning module
      (``src/ai/web_search_clients.py``).

    Mixin ``NotImplementedError`` bodies (``datasources`` mixins) and
    the explicit ``embed_auth`` stub are intentional and exempt.
    """
    deps_src = (_SRC_ROOT / "web" / "deps" / "infra_mcp.py").read_text(encoding="utf-8")
    assert "HTTPMCPDiscoveryProvider" in deps_src
    assert "HTTPMCPConnectivityProbe" in deps_src
    assert "StaticDiscoveryProvider" not in deps_src
    assert "StaticConnectivityProbe" not in deps_src
    for path in _SRC_ROOT.rglob("*.py"):
        if path.name == "web_search_clients.py":
            continue
        text = path.read_text(encoding="utf-8")
        assert "_NotImplementedClient" not in text, (
            f"{path.relative_to(_SRC_ROOT)} references the WebSearch placeholder"
        )


__all__ = []
