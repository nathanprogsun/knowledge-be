"""Unit tests for built-in agent resolution in `CustomAgentService`.

The service falls back to the built-in registry when a preset id has no
customized row, and the agent list leads with the built-in presets in
registry order before the custom rows.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from src.common.exception import NotFoundError
from src.core.agents.builtin_registry import BUILTIN_AGENT_ORDER
from src.core.agents.service.custom_agent_service import CustomAgentService
from src.db.dao.custom_agent_repository import CustomAgentRepository
from src.db.models.custom_agent import CustomAgent

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_TENANT = 42


def _custom_row(*, agent_id: str, name: str, is_builtin: bool = False) -> CustomAgent:
    return CustomAgent(
        id=agent_id,
        name=name,
        description=None,
        avatar=None,
        is_builtin=is_builtin,
        tenant_id=_TENANT,
        created_by=None,
        config={"agent_mode": "quick-answer"},
        created_at=_NOW,
        updated_at=_NOW,
    )


def _make_repo() -> AsyncMock:
    return AsyncMock(spec=CustomAgentRepository)


def _make_service(repo: AsyncMock) -> CustomAgentService:
    return CustomAgentService(agent_repo=repo)


# ── get_agent_by_id ────────────────────────────────────────────────


async def test_get_agent_by_id_falls_back_to_builtin_preset() -> None:
    repo = _make_repo()
    repo.get_by_id_and_tenant.return_value = None
    service = _make_service(repo)

    agent = await service.get_agent_by_id(tenant_id=_TENANT, agent_id="builtin-wiki-researcher")

    assert agent.id == "builtin-wiki-researcher"
    assert agent.is_builtin is True
    assert agent.tenant_id == _TENANT
    repo.get_by_id_and_tenant.assert_awaited_once_with(
        id="builtin-wiki-researcher", tenant_id=_TENANT
    )


async def test_get_agent_by_id_returns_customized_builtin_row() -> None:
    repo = _make_repo()
    row = _custom_row(agent_id="builtin-wiki-researcher", name="定制维基", is_builtin=True)
    repo.get_by_id_and_tenant.return_value = row
    service = _make_service(repo)

    agent = await service.get_agent_by_id(tenant_id=_TENANT, agent_id="builtin-wiki-researcher")

    assert agent.name == "定制维基"
    assert agent.is_builtin is True


async def test_get_agent_by_id_unknown_raises_not_found() -> None:
    repo = _make_repo()
    repo.get_by_id_and_tenant.return_value = None
    service = _make_service(repo)

    with pytest.raises(NotFoundError):
        await service.get_agent_by_id(tenant_id=_TENANT, agent_id="agent-unknown")


# ── list_agents ────────────────────────────────────────────────────


async def test_list_agents_leads_with_builtin_presets() -> None:
    repo = _make_repo()
    repo.list_by_tenant.return_value = [
        _custom_row(agent_id="agent-a", name="自定义 A"),
    ]
    service = _make_service(repo)

    agents = await service.list_agents(tenant_id=_TENANT)

    assert [a.id for a in agents[: len(BUILTIN_AGENT_ORDER)]] == list(BUILTIN_AGENT_ORDER)
    assert agents[-1].id == "agent-a"
    assert all(a.is_builtin for a in agents[: len(BUILTIN_AGENT_ORDER)])
    assert agents[-1].is_builtin is False


async def test_list_agents_uses_customized_builtin_row_when_present() -> None:
    repo = _make_repo()
    repo.list_by_tenant.return_value = [
        _custom_row(agent_id="builtin-quick-answer", name="定制快速问答", is_builtin=True),
    ]
    service = _make_service(repo)

    agents = await service.list_agents(tenant_id=_TENANT)

    quick = next(a for a in agents if a.id == "builtin-quick-answer")
    assert quick.name == "定制快速问答"
