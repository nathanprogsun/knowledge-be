"""Session/message frozen-contract invariants.

Compares the session and message wire contracts against the field sets
captured in ``fixtures/session_responses.json`` (derived from the
reference API handlers):

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

from src.core.contracts import sessions as contracts
from src.web.api.chat.messages.views import (
    ChatHistoryStatsEnvelope,
    DeleteMessageResponse,
    MessageLoadEnvelope,
    SearchMessagesEnvelope,
    SuggestionEnvelope,
)
from src.web.api.chat.sessions.views import (
    DeleteSessionResponse,
    PinSessionEnvelope,
    SessionEnvelope,
    SessionListEnvelope,
)

_FIXTURE_PATH: Path = Path(__file__).parent / "fixtures" / "session_responses.json"

# Mapping of fixture contract-name -> concrete Pydantic model.
# Session/Message/related types live in the frozen contract module; the
# envelope shapes live in the web view layer.
_CONTRACT_MODELS: dict[str, type[BaseModel]] = {
    "Session": contracts.Session,
    "SessionListItem": contracts.Session,
    "Message": contracts.Message,
    "KnowledgeReference": contracts.KnowledgeReference,
    "MessageSearchHit": contracts.MessageSearchHit,
    "MessageSearchResponse": contracts.MessageSearchResponse,
    "ChatHistoryStats": contracts.ChatHistoryStats,
    "SuggestionQuestion": contracts.SuggestionQuestion,
    "SuggestionSet": contracts.SuggestionSet,
    "SessionEnvelope": SessionEnvelope,
    "SessionListEnvelope": SessionListEnvelope,
    "DeleteSessionResponse": DeleteSessionResponse,
    "PinSessionEnvelope": PinSessionEnvelope,
    "MessageLoadEnvelope": MessageLoadEnvelope,
    "DeleteMessageResponse": DeleteMessageResponse,
    "SearchMessagesEnvelope": SearchMessagesEnvelope,
    "ChatHistoryStatsEnvelope": ChatHistoryStatsEnvelope,
    "SuggestionEnvelope": SuggestionEnvelope,
}

# Mapping of fixture request-name -> the request model exposed on the
# wire (all frozen contract models).
_REQUEST_MODELS: dict[str, type[BaseModel]] = {
    "CreateSessionRequest": contracts.CreateSessionRequest,
    "UpdateSessionRequest": contracts.UpdateSessionRequest,
    "BatchDeleteSessionsRequest": contracts.BatchDeleteSessionsRequest,
    "LoadMessagesQuery": contracts.LoadMessagesQuery,
    "SearchMessagesRequest": contracts.SearchMessagesRequest,
    "GenerateTitleRequest": contracts.GenerateTitleRequest,
    "TitleGenMessage": contracts.TitleGenMessage,
    "StopGenerationRequest": contracts.StopGenerationRequest,
    "EnsureSuggestionsRequest": contracts.EnsureSuggestionsRequest,
    "SuggestionEventRequest": contracts.SuggestionEventRequest,
}


def _load_fixture() -> dict[str, object]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


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
    actual = set(_model_wire_fields(model))
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
    actual = set(_model_wire_fields(model))
    assert actual == expected, (
        f"{name}: request-body fields diverge from the captured reference.\n"
        f"  missing from model: {sorted(expected - actual)}\n"
        f"  extra in model:     {sorted(actual - expected)}"
    )


def test_no_orphaned_request_fixture_entries() -> None:
    orphaned = sorted(set(_fixture_requests()) - set(_REQUEST_MODELS))
    assert not orphaned, f"fixture request entries without a model: {orphaned}"


# ── Sample payload invariants ────────────────────────────────────────


def test_sample_payloads_validate_against_contracts() -> None:
    """Every captured sample response validates against its contract model."""
    raw = _load_fixture()
    samples = raw.get("samples", {})
    assert isinstance(samples, dict), "fixture 'samples' section missing"

    # Envelope samples hold the inner data against the inner contract.
    envelope_to_inner: dict[str, str] = {
        "SessionEnvelope": "Session",
        "SessionListEnvelope": "SessionListItem",
        "MessageLoadEnvelope": "Message",
        "SearchMessagesEnvelope": "MessageSearchResponse",
        "ChatHistoryStatsEnvelope": "ChatHistoryStats",
        "SuggestionEnvelope": "SuggestionSet",
    }

    for sample_name, sample_payload in samples.items():
        assert isinstance(sample_payload, dict), (
            f"sample {sample_name!r} is missing or not an object"
        )
        target_name = envelope_to_inner.get(sample_name, sample_name)
        model = _CONTRACT_MODELS.get(target_name)
        assert model is not None, (
            f"sample {sample_name!r} targets unknown contract {target_name!r}"
        )

        # For envelopes, validate the inner data payload against the
        # inner contract; the envelope itself is structurally checked
        # above via the contract-invariants gate.
        if sample_name in envelope_to_inner:
            data = sample_payload.get("data")
            assert isinstance(data, (dict, list)), (
                f"envelope sample {sample_name!r} has no data payload"
            )
            # SessionListEnvelope carries data as a list of items;
            # MessageLoadEnvelope also; the rest are single objects.
            if isinstance(data, list):
                for row in data:
                    model.model_validate(row)
            else:
                model.model_validate(data)
        else:
            model.model_validate(sample_payload)


# ── Field-name helper ────────────────────────────────────────────────


def _model_wire_fields(model: type[BaseModel]) -> list[str]:
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


__all__ = ["_model_wire_fields"]
