"""Wire-shape conversion for the shared-resource endpoints.

Projects the service-side shared rows (``SharedKnowledgeBaseInfo`` /
``SharedAgentInfo``) onto the frozen wire contracts, embedding the
source resource under the ``knowledge_base`` / ``agent`` key and
applying the cross-tenant strip to the knowledge-base payload so the
owning tenant's vector-store metadata never reaches the wire.

The list envelopes keep ``total`` as a sibling of ``data`` (not nested),
matching the upstream ``{"success": true, "data": [...], "total": n}``
shape.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.common.json import JsonObject
from src.core.contracts.organizations import (
    AgentShare,
    AgentShareListResponse,
    SharedAgentListItem,
    SharedKnowledgeBaseListItem,
)
from src.core.organizations.types import SharedAgentInfo, SharedKnowledgeBaseInfo
from src.db.models.agent_share import AgentShare as AgentShareRow
from src.web.api.agents.views import agent_to_contract
from src.web.api.knowledge_bases.views import knowledge_base_to_contract


class SharedKnowledgeBaseListEnvelope(BaseModel):
    """``{"success": true, "data": [...], "total": n}`` - shared-KB lists."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: list[SharedKnowledgeBaseListItem]
    total: int


class SharedAgentListEnvelope(BaseModel):
    """``{"success": true, "data": [...], "total": n}`` - shared-agent lists."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: list[SharedAgentListItem]
    total: int


class SharedAgentDisabledEnvelope(BaseModel):
    """``{"success": true}`` - hide-preference acknowledgement."""

    model_config = ConfigDict(frozen=True)

    success: bool


def _shared_kb_payload(info: SharedKnowledgeBaseInfo) -> JsonObject:
    """Project one shared knowledge base row onto the wire item.

    The embedded knowledge base runs through the cross-tenant strip:
    the owning tenant's ``vector_store_id`` is removed and the
    ``vector_store_*`` display block is pinned to the shared marker so
    no owner-side store inventory leaks across tenants.
    """
    kb = knowledge_base_to_contract(info.knowledge_base).model_dump(mode="json")
    kb.pop("vector_store_id", None)
    kb["vector_store_name"] = None
    kb["vector_store_source"] = "shared"
    kb["vector_store_engine_type"] = None
    kb["vector_store_status"] = "available"
    return kb


def _shared_kb_row(info: SharedKnowledgeBaseInfo) -> SharedKnowledgeBaseListItem:
    """Project the service DTO onto the frozen shared-KB contract."""
    return SharedKnowledgeBaseListItem(
        knowledge_base=_shared_kb_payload(info),
        share_id=info.share_id,
        organization_id=info.organization_id,
        org_name=info.org_name,
        permission=info.permission,
        source_tenant_id=info.source_tenant_id,
        shared_at=info.shared_at,
        is_mine=None,
        source_from_agent=None,
    )


def _shared_agent_row(info: SharedAgentInfo) -> SharedAgentListItem:
    """Project the service DTO onto the frozen shared-agent contract.

    Empty ``shared_by_*`` values are emitted as ``None`` so the
    exclude-none serialization drops them (mirrors the upstream
    ``omitempty`` tags); ``web_search_ready`` / ``disabled_by_me`` are
    always present.
    """
    return SharedAgentListItem(
        agent=agent_to_contract(info.agent).model_dump(mode="json"),
        share_id=info.share_id,
        organization_id=info.organization_id,
        org_name=info.org_name,
        permission=info.permission,
        source_tenant_id=info.source_tenant_id,
        shared_at=info.shared_at,
        shared_by_user_id=info.shared_by_user_id or None,
        shared_by_username=info.shared_by_username or None,
        web_search_ready=info.web_search_ready,
        disabled_by_me=info.disabled_by_me,
        is_mine=None,
    )


def shared_knowledge_base_list_envelope(
    items: list[SharedKnowledgeBaseInfo],
) -> SharedKnowledgeBaseListEnvelope:
    """Wrap the shared-KB rows in the success envelope."""
    return SharedKnowledgeBaseListEnvelope(
        success=True,
        data=[_shared_kb_row(info) for info in items],
        total=len(items),
    )


def shared_agent_list_envelope(
    items: list[SharedAgentInfo],
) -> SharedAgentListEnvelope:
    """Wrap the shared-agent rows in the success envelope."""
    return SharedAgentListEnvelope(
        success=True,
        data=[_shared_agent_row(info) for info in items],
        total=len(items),
    )


def shared_agent_disabled_envelope() -> SharedAgentDisabledEnvelope:
    """Wrap the hide-preference acknowledgement."""
    return SharedAgentDisabledEnvelope(success=True)


# ── Agent share management ─────────────────────────────────────────


class AgentShareEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` - single agent-share responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: AgentShare


class AgentShareListEnvelope(BaseModel):
    """``{"success": true, "data": {"shares": [...], "total": n}}``."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: AgentShareListResponse


def agent_share_to_contract(
    share: AgentShareRow,
    *,
    org_name: str | None = None,
) -> AgentShare:
    """Project one agent-share row onto the wire contract.

    The display fields the row does not carry (agent name / avatar,
    sharer username) stay ``None``; the organization name is resolved
    by the route layer and passed in.
    """
    return AgentShare(
        id=share.id,
        agent_id=share.agent_id,
        organization_id=share.organization_id,
        organization_name=org_name,
        shared_by_user_id=share.shared_by_user_id,
        source_tenant_id=share.source_tenant_id,
        permission=share.permission,
        created_at=share.created_at,
    )


def agent_share_envelope(
    share: AgentShareRow,
    *,
    org_name: str | None = None,
) -> AgentShareEnvelope:
    """Wrap one agent-share row in the success envelope."""
    return AgentShareEnvelope(
        success=True,
        data=agent_share_to_contract(share, org_name=org_name),
    )


def agent_share_list_envelope(
    shares: list[AgentShareRow],
    *,
    org_names: dict[str, str],
) -> AgentShareListEnvelope:
    """Wrap the agent-share rows in the success envelope."""
    return AgentShareListEnvelope(
        success=True,
        data=AgentShareListResponse(
            shares=[
                agent_share_to_contract(share, org_name=org_names.get(share.organization_id))
                for share in shares
            ],
            total=len(shares),
        ),
    )


__all__ = [
    "SharedAgentDisabledEnvelope",
    "SharedAgentListEnvelope",
    "SharedKnowledgeBaseListEnvelope",
    "shared_agent_disabled_envelope",
    "shared_agent_list_envelope",
    "shared_knowledge_base_list_envelope",
]
