"""Agent HTTP endpoints — CRUD plus the agent-editor catalogs.

Registered by ``RegisterCustomAgentRoutes`` / ``RegisterSkillRoutes``.

Agents are tenant-scoped: every service call takes the caller's
``tenant_id`` from the request context, and a cross-workspace id reads
as 404 rather than 403 so the id space is not enumerable.

=============================================  ========
Route                                           Role
=============================================  ========
``GET    /agents/placeholders``                 Viewer
``GET    /agents/type-presets``                 Viewer
``POST   /agents``                              Contributor
``GET    /agents``                              Viewer
``GET    /agents/{id}``                         Viewer
``PUT    /agents/{id}``                         Admin
``DELETE /agents/{id}``                         Admin
``POST   /agents/{id}/copy``                    Contributor
``GET    /agents/{id}/suggested-questions``     Viewer
``GET    /skills``                              Viewer
=============================================  ========

Route order matters: the static ``/placeholders`` and ``/type-presets``
paths are declared before the ``/{id}``-shaped routes so a literal
segment is never captured as an id.

Scope notes (this build):

- The per-workspace "disabled by me" list (``disabled_own_agent_ids``)
  is a deferred seam: the shared-agent registry that owns it is not
  ported, so the list response carries an empty id set.
- Agent-type presets load from a startup YAML registry; with no config
  file deployed the registry is empty, so the endpoint returns an
  empty list (the frontend hides the type dropdown).
- The suggested-question generator needs the chunk / wiki / tag /
  knowledge scopes and is a deferred seam; the endpoint validates agent
  access and returns the wire shape with an empty question set.

Query-parameter ``description`` strings are intentionally Chinese
(mirrors the upstream swagger annotations). RUF001 flags the
full-width punctuation; suppressed file-wide for the same reason as
``src/web/api/system/router.py``.
"""
# ruff: noqa: RUF001

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from src.common.exception import UnauthorizedError
from src.common.json import JsonObject
from src.core.agents.placeholders import all_placeholders, prompt_placeholder_group
from src.core.agents.types import CustomAgentInfo
from src.core.contracts.agents import (
    AgentConfig,
    CreateAgentRequest,
    UpdateAgentRequest,
)
from src.web.api.agents.skill_views import skill_router
from src.web.api.agents.views import (
    AgentEnvelope,
    AgentListEnvelope,
    DeleteAgentResponse,
    PlaceholdersEnvelope,
    SuggestedQuestionsData,
    SuggestedQuestionsEnvelope,
    TypePresetsEnvelope,
    agent_envelope,
    agent_list_envelope,
    placeholders_envelope,
)
from src.web.deps import AuthDep, RoleAdminDep, RoleContributorDep, RoleViewerDep
from src.web.deps.agents import CustomAgentServiceDep
from src.web.deps.context import get_tenant_id_dep, get_user_id_dep

# Function-arg-style principal deps.
_PrincipalTenant = Annotated[int, Depends(get_tenant_id_dep)]
_PrincipalUser = Annotated[str | None, Depends(get_user_id_dep)]


router = APIRouter(prefix="/agents", tags=["agents"])

# The handler answers delete with a message string the UI matches.
_DELETE_MESSAGE = "Agent deleted successfully"


def _require_tenant(tenant_id: int) -> int:
    """Return the active workspace id, or fail.

    An agent is always workspace-scoped; without a tenant context there
    is no safe default (tenant 0 is the system scope, which owns no
    agents), so this rejects rather than guessing.
    """
    if tenant_id == 0:
        raise UnauthorizedError(
            code="auth.tenant_context_missing",
            message="unauthorized: workspace context missing",
        )
    return tenant_id


def _config_to_json(value: AgentConfig | None) -> JsonObject | None:
    """Dump an optional typed config contract onto the service JSON shape."""
    if value is None:
        return None
    return value.model_dump(mode="json")


def _filter_by_creator(
    infos: list[CustomAgentInfo],
    *,
    creator: str,
    user_id: str | None,
) -> list[CustomAgentInfo]:
    """Apply the optional [All | Mine | Others] creator filter.

    Built-in agents are workspace fixtures owned by no user, so they are
    always kept regardless of the filter; non-built-in rows with no
    creator id fall out of both ``mine`` and ``others``.
    """
    key = creator.strip().lower()
    if key not in ("mine", "others"):
        return infos
    caller = user_id or ""
    kept: list[CustomAgentInfo] = []
    for info in infos:
        if info.is_builtin:
            kept.append(info)
            continue
        if not info.created_by:
            continue
        if (key == "mine" and info.created_by == caller) or (
            key == "others" and info.created_by != caller
        ):
            kept.append(info)
    return kept


# ── Editor catalogs (declared before /{id} to avoid capture) ─────────


@router.get("/placeholders", response_model=PlaceholdersEnvelope)
async def get_placeholders(
    _auth: AuthDep,
    _role: RoleViewerDep,
) -> PlaceholdersEnvelope:
    """Return the prompt placeholder definitions, grouped by field.

    Backs the editor's placeholder insertion UI; the payload is a static
    catalog shared across the workspace.
    """
    return placeholders_envelope(
        {**prompt_placeholder_group(), "all": all_placeholders()}
    )


@router.get("/type-presets", response_model=TypePresetsEnvelope)
async def get_agent_type_presets(
    _auth: AuthDep,
    _role: RoleViewerDep,
) -> TypePresetsEnvelope:
    """Return the smart-reasoning agent-type presets for the editor.

    The preset registry is a startup-loaded YAML catalog; with no config
    file deployed it stays empty, so the endpoint returns an empty list
    and the frontend hides the type dropdown.
    """
    return TypePresetsEnvelope(success=True, data=[])


# ── CRUD ─────────────────────────────────────────────────────────────


@router.post("", response_model=AgentEnvelope, status_code=201)
async def create_agent(
    _auth: AuthDep,
    _role: RoleContributorDep,
    body: CreateAgentRequest,
    service: CustomAgentServiceDep,
    tenant_id: _PrincipalTenant,
    user_id: _PrincipalUser,
) -> AgentEnvelope:
    """Create a custom agent in the active workspace.

    The domain service stamps id / tenant / timestamps, records the
    creator and applies the contract config defaults before persisting.
    """
    tenant_id = _require_tenant(tenant_id)
    info = await service.create_agent(
        tenant_id=tenant_id,
        name=body.name,
        config=_config_to_json(body.config),
        description=body.description,
        avatar=body.avatar,
        user_id=user_id,
    )
    return agent_envelope(info)


@router.get("", response_model=AgentListEnvelope)
async def list_agents(
    _auth: AuthDep,
    _role: RoleViewerDep,
    service: CustomAgentServiceDep,
    tenant_id: _PrincipalTenant,
    user_id: _PrincipalUser,
    creator: str = Query(default="", description="创建者筛选"),
) -> AgentListEnvelope:
    """List the workspace's agents, newest first.

    The optional ``creator`` query drives the [All | Mine | Others]
    segmented control. The per-workspace disabled-agent id set is a
    deferred seam and is emitted empty.
    """
    tenant_id = _require_tenant(tenant_id)
    infos = await service.list_agents(tenant_id=tenant_id)
    infos = _filter_by_creator(infos, creator=creator, user_id=user_id)
    return agent_list_envelope(infos)


@router.get("/{id}", response_model=AgentEnvelope)
async def get_agent(
    _auth: AuthDep,
    _role: RoleViewerDep,
    id: str,
    service: CustomAgentServiceDep,
    tenant_id: _PrincipalTenant,
) -> AgentEnvelope:
    """Return one agent of the workspace.

    Ownership is enforced by the tenant-scoped read, so a cross-workspace
    id reads as not-found.
    """
    tenant_id = _require_tenant(tenant_id)
    info = await service.get_agent_by_id(tenant_id=tenant_id, agent_id=id)
    return agent_envelope(info)


@router.put("/{id}", response_model=AgentEnvelope)
async def update_agent(
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    body: UpdateAgentRequest,
    service: CustomAgentServiceDep,
    tenant_id: _PrincipalTenant,
) -> AgentEnvelope:
    """Overwrite an agent's mutable fields.

    The name is required and the config is re-defaulted by the service;
    built-in rows reject basic-info edits. The tenant-ownership pre-check
    makes a cross-workspace id read as not-found.
    """
    tenant_id = _require_tenant(tenant_id)
    info = await service.update_agent(
        tenant_id=tenant_id,
        agent_id=id,
        name=body.name or "",
        config=_config_to_json(body.config) or {},
        description=body.description,
        avatar=body.avatar,
    )
    return agent_envelope(info)


@router.delete("/{id}", response_model=DeleteAgentResponse)
async def delete_agent(
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    service: CustomAgentServiceDep,
    tenant_id: _PrincipalTenant,
) -> DeleteAgentResponse:
    """Soft-delete a custom agent.

    Built-in rows and missing ids are rejected by the service with the
    corresponding error status.
    """
    tenant_id = _require_tenant(tenant_id)
    await service.delete_agent(tenant_id=tenant_id, agent_id=id)
    return DeleteAgentResponse(success=True, message=_DELETE_MESSAGE)


# ── Copy ─────────────────────────────────────────────────────────────


@router.post("/{id}/copy", response_model=AgentEnvelope, status_code=201)
async def copy_agent(
    _auth: AuthDep,
    _role: RoleContributorDep,
    id: str,
    service: CustomAgentServiceDep,
    tenant_id: _PrincipalTenant,
    user_id: _PrincipalUser,
) -> AgentEnvelope:
    """Create a copy of an existing agent, owned by the caller.

    The clone is never built-in and inherits the source config with
    defaults applied.
    """
    tenant_id = _require_tenant(tenant_id)
    info = await service.copy_agent(tenant_id=tenant_id, agent_id=id, user_id=user_id)
    return agent_envelope(info)


# ── Suggested questions ──────────────────────────────────────────────


@router.get("/{id}/suggested-questions", response_model=SuggestedQuestionsEnvelope)
async def get_suggested_questions(
    _auth: AuthDep,
    _role: RoleViewerDep,
    id: str,
    service: CustomAgentServiceDep,
    tenant_id: _PrincipalTenant,
    knowledge_base_ids: str = Query(
        default="", description="知识库ID列表（逗号分隔），覆盖智能体默认配置"
    ),
    knowledge_ids: str = Query(
        default="", description="知识ID列表（逗号分隔），限定到具体文档"
    ),
    tag_scopes: str = Query(default="", description="带知识库归属的标签范围（JSON）"),
    limit: int = Query(
        default=0, description="返回数量上限（未传时使用智能体配置的开场问题数量，最大30）"
    ),
) -> SuggestedQuestionsEnvelope:
    """Return recommended starter questions for an agent.

    The question generator needs the chunk / wiki / tag / knowledge
    scopes and is a deferred seam — the endpoint validates agent access
    and returns the wire shape with an empty question set so callers can
    build against the contract.
    """
    tenant_id = _require_tenant(tenant_id)
    await service.get_agent_by_id(tenant_id=tenant_id, agent_id=id)
    return SuggestedQuestionsEnvelope(success=True, data=SuggestedQuestionsData())


__all__ = ["router", "skill_router"]
