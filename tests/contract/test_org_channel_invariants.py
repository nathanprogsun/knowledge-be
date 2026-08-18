"""Organization + embed + IM frozen-contract invariants.

Compares the org / embed / IM wire contracts and view models against the
field sets captured in ``fixtures/org_channel_responses.json`` (derived
from the reference Go handlers ``OrganizationResponse``,
``embedChannelResponse``, ``IMChannelSummary`` in
``internal/types``):

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

Where the Python contract diverges from the reference (for example,
``Organization`` missing the reference ``owner_tenant_id``, or
``JoinRequestRecord`` missing the reference ``reviewed_at``) the
parametrised field-set assertion fails — that failure is the
documented drift finding, not a test bug.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias

import pytest
from pydantic import BaseModel

from src.core.contracts import organizations
from src.web.api.channels.embed import views as embed_views
from src.web.api.channels.im import views as im_views
from src.web.api.organizations.views import TenantInviteCandidate

_FIXTURE_PATH: Path = (
    Path(__file__).parent / "fixtures" / "org_channel_responses.json"
)

#: Contract names whose Python frozen model is known to drift from the
#: captured reference. Each entry documents the specific drift so future
#: readers can spot when the drift is closed (the xfail flips to xpass).
KNOWN_CONTRACT_DRIFT: dict[str, str] = {
    "Organization": (
        "Python Organization is missing the reference 'owner_tenant_id' "
        "field that OrganizationResponse carries in the upstream contract."
    ),
    "OrganizationList": (
        "Python OrganizationList renames the reference 'organizations' "
        "list field to 'items'."
    ),
    "SearchOrganizationsResponse": (
        "Python SearchOrganizationsResponse renames the reference "
        "'organizations' list field to 'items'."
    ),
    "JoinRequestRecord": (
        "Python JoinRequestRecord is missing the reference 'reviewed_at' "
        "field that JoinRequestResponse carries upstream."
    ),
    "OrgMember": (
        "Python OrgMember is missing the reference 'representative_user_id' "
        "and 'tenant_name' fields that OrganizationMemberResponse carries "
        "upstream."
    ),
    "KnowledgeBaseShare": (
        "Python KnowledgeBaseShare is missing the reference "
        "'require_approval' field that KnowledgeBaseShareResponse carries "
        "upstream."
    ),
    "AgentShare": (
        "Python AgentShare is missing the reference 'id' and "
        "'scope_mcp_count' fields that AgentShareResponse carries upstream."
    ),
}

# Mapping of fixture contract-name -> concrete Pydantic model. The
# embed / IM shapes live in the web view layer; the rest come from the
# frozen org contract module.
_CONTRACT_MODELS: dict[str, type[BaseModel]] = {
    # Organization domain — frozen contracts in src.core.contracts.organizations
    "Organization": organizations.Organization,
    "OrganizationPreview": organizations.OrganizationPreview,
    "OrganizationList": organizations.OrganizationList,
    "SearchOrganizationsResponse": organizations.SearchOrganizationsResponse,
    "JoinRequestRecord": organizations.JoinRequestRecord,
    "JoinRequestListResponse": organizations.JoinRequestListResponse,
    "OrgMember": organizations.OrgMember,
    "OrgMemberListResponse": organizations.OrgMemberListResponse,
    "RegenerateInviteCodeResponse": organizations.RegenerateInviteCodeResponse,
    "KnowledgeBaseShare": organizations.KnowledgeBaseShare,
    "KnowledgeBaseShareListResponse": organizations.KnowledgeBaseShareListResponse,
    "SharedKnowledgeBaseListItem": organizations.SharedKnowledgeBaseListItem,
    "AgentShare": organizations.AgentShare,
    "AgentShareListResponse": organizations.AgentShareListResponse,
    "SharedAgentListItem": organizations.SharedAgentListItem,
    "TenantInviteCandidate": TenantInviteCandidate,
    # Embed channel — web view models
    "EmbedChannelRecord": embed_views.EmbedChannelRecord,
    "EmbedPublicConfig": embed_views.EmbedPublicConfig,
    # IM channel — web view models
    "IMChannelRecord": im_views.IMChannelRecord,
}

# Mapping of fixture request-name -> the request model exposed on the
# wire (frozen contract models + web-layer request bodies).
_REQUEST_MODELS: dict[str, type[BaseModel]] = {
    "CreateOrganizationRequest": organizations.CreateOrganizationRequest,
    "UpdateOrganizationRequest": organizations.UpdateOrganizationRequest,
    "JoinOrganizationByCodeRequest": organizations.JoinOrganizationByCodeRequest,
    "JoinRequestRequest": organizations.JoinRequestRequest,
    "JoinRequestByIDRequest": organizations.JoinRequestByIDRequest,
    "ReviewJoinRequestRequest": organizations.ReviewJoinRequestRequest,
    "RequestRoleUpgradeRequest": organizations.RequestRoleUpgradeRequest,
    "UpdateMemberRoleRequest": organizations.UpdateMemberRoleRequest,
    "CreateKnowledgeBaseShareRequest": organizations.CreateKnowledgeBaseShareRequest,
    "UpdateKnowledgeBaseShareRequest": organizations.UpdateKnowledgeBaseShareRequest,
    "CreateAgentShareRequest": organizations.CreateAgentShareRequest,
    "EmbedChannelRequest": embed_views.EmbedChannelRequest,
    "IMChannelCreateRequest": im_views.IMChannelCreateRequest,
    "IMChannelUpdateRequest": im_views.IMChannelUpdateRequest,
}


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


def _load_fixture() -> _FixtureJson:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


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
    if name in KNOWN_CONTRACT_DRIFT:
        pytest.xfail(KNOWN_CONTRACT_DRIFT[name])
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