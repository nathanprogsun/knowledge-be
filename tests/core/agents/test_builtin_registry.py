"""Unit tests for the built-in agent preset registry."""

from __future__ import annotations

import pytest

from src.core.agents.builtin_registry import BUILTIN_AGENT_ORDER, get_builtin_agent
from src.core.agents.types import BUILTIN_AGENT_IDS


def test_builtin_agent_order_is_fixed_and_subset_of_known_ids() -> None:
    assert BUILTIN_AGENT_ORDER[0] == "builtin-quick-answer"
    assert set(BUILTIN_AGENT_ORDER) <= set(BUILTIN_AGENT_IDS)
    # The wiki fixer is an internal agent and stays out of the picker.
    assert "builtin-wiki-fixer" not in BUILTIN_AGENT_ORDER


def test_get_builtin_agent_returns_default_preset() -> None:
    agent = get_builtin_agent("builtin-wiki-researcher", tenant_id=42)
    assert agent is not None
    assert agent.id == "builtin-wiki-researcher"
    assert agent.name == "维基问答"
    assert agent.is_builtin is True
    assert agent.tenant_id == 42
    assert agent.config["agent_mode"] == "smart-reasoning"
    assert agent.config["agent_type"] == "wiki-qa"


def test_get_builtin_agent_scopes_tenant_and_fresh_timestamps() -> None:
    first = get_builtin_agent("builtin-quick-answer", tenant_id=7)
    second = get_builtin_agent("builtin-quick-answer", tenant_id=7)
    assert first is not second  # fresh projection per call
    assert first.tenant_id == 7
    assert (second.created_at - first.created_at).total_seconds() < 1  # fresh per call


def test_get_builtin_agent_unknown_id_returns_none() -> None:
    assert get_builtin_agent("builtin-not-a-real-agent", tenant_id=1) is None
    assert get_builtin_agent("agent-abc", tenant_id=1) is None


@pytest.mark.parametrize("agent_id", BUILTIN_AGENT_ORDER)
def test_every_ordered_preset_resolves(agent_id: str) -> None:
    agent = get_builtin_agent(agent_id, tenant_id=1)
    assert agent is not None
    assert agent.id == agent_id
    assert agent.name  # every preset has a display name
    assert agent.config.get("agent_mode")
