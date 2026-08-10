"""Wire-shape conversion for the agent endpoints.

Projects the service DTOs (``CustomAgentInfo``) onto the frozen
contracts in ``src/core/contracts/agents.py``. The stored ``config``
JSONB blob is parsed onto the typed ``AgentConfig`` contract, leniently:
a stored blob whose field set does not match the contract yields
``None`` rather than failing the whole response, mirroring the
knowledge-base view conversion.

Response-only fields without a backing service in this layer
(``deleted_at``, the list endpoint's ``disabled_own_agent_ids``) are
emitted as ``None`` / empty; the frozen contract types them as nullable.
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.common.json import JsonObject
from src.core.agents.types import CustomAgentInfo
from src.core.contracts.agents import (
    Agent,
    AgentConfig,
    AgentPlaceholderGroup,
)

_Parseable = TypeVar("_Parseable", bound=BaseModel)


class AgentEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` - single-agent responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: Agent


class AgentListEnvelope(BaseModel):
    """``{"success": true, "data": [...], "disabled_own_agent_ids": [...]}``.

    The list response carries the per-workspace disabled-agent ids as a
    sibling of ``data``, mirroring the upstream handler's envelope.
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    data: list[Agent]
    disabled_own_agent_ids: list[str] = Field(default_factory=list)


class DeleteAgentResponse(BaseModel):
    """``{"success": true, "message": "..."}`` - delete acknowledgement."""

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str


class PlaceholdersEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` - prompt placeholder groups."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: AgentPlaceholderGroup


class TypePresetsEnvelope(BaseModel):
    """``{"success": true, "data": [...]}`` - agent-type preset list."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: list[JsonObject] = Field(default_factory=list)


class SuggestedQuestion(BaseModel):
    """One recommended starter question — ``types.SuggestedQuestion``."""

    model_config = ConfigDict(frozen=True)

    question: str
    source: str
    knowledge_base_id: str | None = Field(default=None)


class SuggestedQuestionsData(BaseModel):
    """``{"questions": [...]}`` - the suggested-questions payload."""

    model_config = ConfigDict(frozen=True)

    questions: list[SuggestedQuestion] = Field(default_factory=list)


class SuggestedQuestionsEnvelope(BaseModel):
    """``{"success": true, "data": {"questions": [...]}}``."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: SuggestedQuestionsData


def _parse_optional(
    model: type[_Parseable],
    raw: JsonObject | None,
) -> _Parseable | None:
    """Parse a JSON config blob onto its typed contract, leniently.

    A stored blob whose field set does not match the contract yields
    ``None`` rather than failing the whole response.
    """
    if raw is None:
        return None
    try:
        return model.model_validate(raw)
    except ValidationError:
        return None


def agent_to_contract(info: CustomAgentInfo) -> Agent:
    """Project the service DTO onto the frozen wire contract.

    ``deleted_at`` is emitted as ``None``: the service projection does
    not expose soft-delete state (only live rows reach it).
    """
    return Agent(
        id=info.id,
        name=info.name,
        description=info.description,
        avatar=info.avatar,
        is_builtin=info.is_builtin,
        tenant_id=info.tenant_id,
        created_by=info.created_by,
        config=_parse_optional(AgentConfig, info.config),
        created_at=info.created_at,
        updated_at=info.updated_at,
        deleted_at=None,
    )


def agent_envelope(info: CustomAgentInfo) -> AgentEnvelope:
    """Wrap one agent in the success envelope."""
    return AgentEnvelope(success=True, data=agent_to_contract(info))


def agent_list_envelope(
    infos: list[CustomAgentInfo],
    *,
    disabled_own_agent_ids: list[str] | None = None,
) -> AgentListEnvelope:
    """Wrap a list of agents in the success envelope."""
    return AgentListEnvelope(
        success=True,
        data=[agent_to_contract(info) for info in infos],
        disabled_own_agent_ids=disabled_own_agent_ids or [],
    )


def placeholders_envelope(
    group: dict[str, list[JsonObject]],
) -> PlaceholdersEnvelope:
    """Wrap the placeholder field-group map in the success envelope."""
    return PlaceholdersEnvelope(
        success=True,
        data=AgentPlaceholderGroup(
            all=group.get("all", []),
            system_prompt=group.get("system_prompt", []),
            agent_system_prompt=group.get("agent_system_prompt", []),
            context_template=group.get("context_template", []),
            rewrite_system_prompt=group.get("rewrite_system_prompt", []),
            rewrite_prompt=group.get("rewrite_prompt", []),
            fallback_prompt=group.get("fallback_prompt", []),
        ),
    )


__all__ = [
    "AgentEnvelope",
    "AgentListEnvelope",
    "DeleteAgentResponse",
    "PlaceholdersEnvelope",
    "SuggestedQuestion",
    "SuggestedQuestionsData",
    "SuggestedQuestionsEnvelope",
    "TypePresetsEnvelope",
    "agent_envelope",
    "agent_list_envelope",
    "agent_to_contract",
    "placeholders_envelope",
]
