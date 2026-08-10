"""Chat/agent frozen-contract invariants.

Compares the chat and agent wire contracts against the field sets
captured in ``fixtures/chat_pipeline_responses.json`` (derived from the
reference API fixtures and the upstream handler tests):

- every covered contract model is frozen (immutable wire shape);
- the contract's serialized field names exactly equal the fixture's
  expected field-name set for that object (no drift in either
  direction);
- the request-body models carry the exact field set the reference
  documents for each create / update / QA body;
- the streamed ``response_type`` vocabulary matches the reference;
- every sample response fixture validates against its contract model.

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

from src.core.chat.pipeline.types import SearchResult
from src.core.chat.service import WIRE_RESPONSE_TYPE
from src.core.contracts import agents
from src.web.api.agents.skill_views import SkillInfoResponse, SkillListEnvelope
from src.web.api.agents.views import (
    AgentEnvelope,
    AgentListEnvelope,
    DeleteAgentResponse,
    SuggestedQuestionsData,
    SuggestedQuestionsEnvelope,
    TypePresetsEnvelope,
)
from src.web.api.chat.views import (
    AttachmentUpload,
    CreateKnowledgeQARequest,
    ImageAttachment,
    MentionedItemRequest,
    SearchKnowledgeEnvelope,
    SearchKnowledgeRequest,
    StreamResponse,
    SuggestionAttribution,
)
from tests.contract.test_knowledge_invariants import model_wire_fields

_FIXTURE_PATH: Path = Path(__file__).parent / "fixtures" / "chat_pipeline_responses.json"

# Mapping of fixture contract-name -> concrete Pydantic model. The
# streamed frame, search hit and chat request shapes live in the web
# view layer / pipeline types; the agent shapes come from the frozen
# contract module plus the agent web envelopes.
_CONTRACT_MODELS: dict[str, type[BaseModel]] = {
    "StreamResponse": StreamResponse,
    "SearchResult": SearchResult,
    "SearchKnowledgeEnvelope": SearchKnowledgeEnvelope,
    "Agent": agents.Agent,
    "AgentEnvelope": AgentEnvelope,
    "AgentListEnvelope": AgentListEnvelope,
    "DeleteAgentResponse": DeleteAgentResponse,
    "AgentPlaceholderGroup": agents.AgentPlaceholderGroup,
    "SuggestedQuestionsData": SuggestedQuestionsData,
    "SuggestedQuestionsEnvelope": SuggestedQuestionsEnvelope,
    "TypePresetsEnvelope": TypePresetsEnvelope,
    "SkillInfoResponse": SkillInfoResponse,
    "SkillListEnvelope": SkillListEnvelope,
}

# Mapping of fixture request-name -> the request model exposed on the
# wire (frozen contract models, plus the web-layer chat request bodies).
_REQUEST_MODELS: dict[str, type[BaseModel]] = {
    "CreateKnowledgeQARequest": CreateKnowledgeQARequest,
    "SearchKnowledgeRequest": SearchKnowledgeRequest,
    "MentionedItemRequest": MentionedItemRequest,
    "ImageAttachment": ImageAttachment,
    "AttachmentUpload": AttachmentUpload,
    "SuggestionAttribution": SuggestionAttribution,
    "CreateAgentRequest": agents.CreateAgentRequest,
    "UpdateAgentRequest": agents.UpdateAgentRequest,
}

# Mapping of fixture sample-key -> model used to validate each sample.
_SAMPLE_MODELS: dict[str, type[BaseModel]] = {
    "SearchResult": SearchResult,
    "SearchKnowledgeEnvelope": SearchKnowledgeEnvelope,
    "Agent": agents.Agent,
    "AgentEnvelope": AgentEnvelope,
    "AgentListEnvelope": AgentListEnvelope,
    "AgentPlaceholderGroup": agents.AgentPlaceholderGroup,
    "SuggestedQuestionsData": SuggestedQuestionsData,
    "SkillListEnvelope": SkillListEnvelope,
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


def _fixture_stream_types() -> list[str]:
    raw = _load_fixture()
    return _string_list(raw.get("stream_response_types"))


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


# ── Stream response-type vocabulary ──────────────────────────────────


def test_stream_response_types_match_reference_vocabulary() -> None:
    expected = set(_fixture_stream_types())
    assert expected, "fixture 'stream_response_types' section missing"
    actual = set(WIRE_RESPONSE_TYPE.values())
    assert actual == expected, (
        "wire response_type vocabulary diverges from the captured reference.\n"
        f"  missing: {sorted(expected - actual)}\n"
        f"  extra:   {sorted(actual - expected)}"
    )


# ── Sample payload invariants ────────────────────────────────────────


def test_sample_payloads_validate_against_contracts() -> None:
    raw = _load_fixture()
    samples = raw.get("samples", {})
    assert isinstance(samples, dict), "fixture 'samples' section missing"
    assert set(samples) == {"StreamResponse", *_SAMPLE_MODELS}, (
        f"sample keys diverge from the covered models: {sorted(samples)}"
    )

    frames = samples.get("StreamResponse", {})
    assert isinstance(frames, dict)
    for label, frame in frames.items():
        assert isinstance(frame, dict), f"StreamResponse sample {label!r} is not an object"
        StreamResponse.model_validate(frame)

    for name, model in _SAMPLE_MODELS.items():
        payload = samples.get(name)
        assert isinstance(payload, dict), f"sample {name!r} is missing or not an object"
        model.model_validate(payload)


__all__ = [
    "model_wire_fields",
]
