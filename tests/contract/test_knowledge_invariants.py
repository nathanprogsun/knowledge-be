"""Knowledge-domain frozen-contract invariants.

Compares the knowledge wire contracts against the field sets captured in
``fixtures/knowledge_responses.json`` (derived from the reference API
documentation and the upstream handler fixtures):

- every covered contract model is frozen (immutable wire shape);
- the contract's serialized field names exactly equal the fixture's
  expected field-name set for that object (no drift in either
  direction);
- the request-body models carry the exact field set the reference
  documents for each create / update body;
- no fixture key is orphaned.

These checks are read-only and model-only — no I/O, no database. They
are the milestone gate that blocks any future drift between the Python
port and the reference wire shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias

import pytest
from pydantic import BaseModel

from src.core.contracts import knowledge
from src.web.api.knowledge.chunks.views import UpdateChunkRequest as WebUpdateChunkRequest
from src.web.api.knowledge.wiki import views as wiki_views

_FIXTURE_PATH: Path = Path(__file__).parent / "fixtures" / "knowledge_responses.json"

# Mapping of fixture contract-name -> concrete Pydantic model. The wiki
# shapes live in the web view layer; the rest come from the frozen
# contract module.
_CONTRACT_MODELS: dict[str, type[BaseModel]] = {
    "KnowledgeBase": knowledge.KnowledgeBase,
    "Knowledge": knowledge.Knowledge,
    "Chunk": knowledge.Chunk,
    "Tag": knowledge.Tag,
    "TagList": knowledge.TagList,
    "FAQEntry": knowledge.FAQEntry,
    "FAQEntryListResponse": knowledge.FAQEntryListResponse,
    "WikiPageView": wiki_views.WikiPageView,
    "WikiFolderView": wiki_views.WikiFolderView,
    "WikiFolderNodeView": wiki_views.WikiFolderNodeView,
    "WikiStats": wiki_views.WikiStats,
    "WikiPageListData": wiki_views.WikiPageListData,
    "WikiFolderListData": wiki_views.WikiFolderListData,
    "WikiSearchData": wiki_views.WikiSearchData,
}

# Mapping of fixture request-name -> the request model exposed on the
# wire (frozen contract models, plus the web-layer chunk update body).
_REQUEST_MODELS: dict[str, type[BaseModel]] = {
    "CreateKnowledgeBaseRequest": knowledge.CreateKnowledgeBaseRequest,
    "UpdateKnowledgeBaseRequest": knowledge.UpdateKnowledgeBaseRequest,
    "CreateManualKnowledgeRequest": knowledge.CreateManualKnowledgeRequest,
    "CreateKnowledgeFromURLRequest": knowledge.CreateKnowledgeFromURLRequest,
    "UpdateKnowledgeRequest": knowledge.UpdateKnowledgeRequest,
    "CreateTagRequest": knowledge.CreateTagRequest,
    "UpdateTagRequest": knowledge.UpdateTagRequest,
    "FAQEntryPayload": knowledge.FAQEntryPayload,
    "FAQBatchDeleteRequest": knowledge.FAQBatchDeleteRequest,
    "FAQBatchUpsertPayload": knowledge.FAQBatchUpsertPayload,
    "HybridSearchRequest": knowledge.HybridSearchRequest,
    "KnowledgeCopyRequest": knowledge.KnowledgeCopyRequest,
    "WebUpdateChunkRequest": WebUpdateChunkRequest,
}


def _load_fixture() -> dict[str, object]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def model_wire_fields(model: type[BaseModel]) -> list[str]:
    """Return the wire (serialization) field names of a model.

    Respects Pydantic ``alias`` / ``serialization_alias`` so the
    comparison is against the actual JSON keys the API emits.
    """
    out: list[str] = []
    for fname, field in model.model_fields.items():
        if fname == "model_config":
            continue
        out.append(field.serialization_alias or field.alias or fname)
    return out


_FixtureJson: TypeAlias = dict[str, object]


def _fixture_contracts() -> dict[str, list[str]]:
    raw = _load_fixture()
    contracts = raw.get("contracts", {})
    out: dict[str, list[str]] = {}
    if isinstance(contracts, dict):
        for name, fields in contracts.items():
            if isinstance(name, str):
                out[name] = _string_list(fields)
    return out


def _fixture_requests() -> dict[str, list[str]]:
    raw = _load_fixture()
    requests = raw.get("requests", {})
    out: dict[str, list[str]] = {}
    if isinstance(requests, dict):
        for name, fields in requests.items():
            if isinstance(name, str):
                out[name] = _string_list(fields)
    return out


# ── Contract payload invariants ──────────────────────────────────────


def test_every_covered_contract_is_frozen() -> None:
    for name, model in _CONTRACT_MODELS.items():
        assert issubclass(model, BaseModel), name
        assert model.model_config.get("frozen") is True, f"{name} is not frozen"


@pytest.mark.parametrize(
    ("name", "model"),
    sorted(_CONTRACT_MODELS.items()),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_contract_wire_fields_match_fixture(name: str, model: type[BaseModel]) -> None:
    fixture = _fixture_contracts()
    assert name in fixture, f"contract '{name}' is missing from the fixture file"
    expected = set(fixture[name])
    actual = set(model_wire_fields(model))
    assert actual == expected, (
        f"{name}: wire fields diverge from the captured reference.\n"
        f"  missing from contract: {sorted(expected - actual)}\n"
        f"  extra in contract:     {sorted(actual - expected)}"
    )


def test_no_orphaned_contract_fixture_entries() -> None:
    orphaned = sorted(set(_fixture_contracts()) - set(_CONTRACT_MODELS))
    assert not orphaned, f"fixture contract entries without a model: {orphaned}"


# ── Request-body invariants ──────────────────────────────────────────


def test_every_covered_request_is_frozen() -> None:
    for name, model in _REQUEST_MODELS.items():
        assert issubclass(model, BaseModel), name
        assert model.model_config.get("frozen") is True, f"{name} is not frozen"


@pytest.mark.parametrize(
    ("name", "model"),
    sorted(_REQUEST_MODELS.items()),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_request_wire_fields_match_fixture(name: str, model: type[BaseModel]) -> None:
    fixture = _fixture_requests()
    assert name in fixture, f"request '{name}' is missing from the fixture file"
    expected = set(fixture[name])
    actual = set(model_wire_fields(model))
    assert actual == expected, (
        f"{name}: request-body fields diverge from the captured reference.\n"
        f"  missing from model: {sorted(expected - actual)}\n"
        f"  extra in model:     {sorted(actual - expected)}"
    )


def test_no_orphaned_request_fixture_entries() -> None:
    orphaned = sorted(set(_fixture_requests()) - set(_REQUEST_MODELS))
    assert not orphaned, f"fixture request entries without a model: {orphaned}"


__all__ = [
    "model_wire_fields",
]
