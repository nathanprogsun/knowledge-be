"""Unit tests for `AgentShareServiceImpl`.

The four repositories are replaced with ``AsyncMock(spec=...)`` backed
by closure-captured in-memory state so the ownership / org-role checks
and the duplicate-share upgrade are exercised for real.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.common.exception import NotFoundError, PermissionDeniedError, ValidationError
from src.core.sharing.agent_share_service import AgentShareServiceImpl
from src.db.dao.agent_share_repository import AgentShareRepository
from src.db.dao.custom_agent_repository import CustomAgentRepository
from src.db.dao.organization_repository import (
    OrganizationMemberRepository,
    OrganizationRepository,
)
from src.db.models.agent_share import AgentShare
from src.db.models.custom_agent import CustomAgent
from src.db.models.organization import Organization, OrganizationTenantMember

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_TENANT = 42
_OTHER_TENANT = 99
_USER = "usr-owner"


def _agent(agent_id: str, *, tenant_id: int = _TENANT, model_id: str = "model-1") -> CustomAgent:
    return CustomAgent(
        id=agent_id,
        name="Agent",
        description=None,
        avatar=None,
        is_builtin=False,
        tenant_id=tenant_id,
        created_by=None,
        config={"agent_mode": "quick-answer", "model_id": model_id},
        created_at=_NOW,
        updated_at=_NOW,
    )


def _org(org_id: str) -> Organization:
    return Organization(
        id=org_id,
        name="Org",
        description=None,
        avatar="",
        owner_id=_USER,
        owner_tenant_id=_TENANT,
        invite_code=None,
        invite_code_expires_at=None,
        invite_code_validity_days=7,
        require_approval=False,
        searchable=False,
        member_limit=50,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _member(*, org_id: str, tenant_id: int, role: str) -> OrganizationTenantMember:
    return OrganizationTenantMember(
        id=str(uuid.uuid4()),
        organization_id=org_id,
        tenant_id=tenant_id,
        role=role,
        representative_user_id=_USER,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _share(*, agent_id: str, org_id: str, tenant_id: int = _TENANT) -> AgentShare:
    return AgentShare(
        id=str(uuid.uuid4()),
        agent_id=agent_id,
        organization_id=org_id,
        shared_by_user_id=_USER,
        source_tenant_id=tenant_id,
        permission="viewer",
        created_at=_NOW,
        updated_at=_NOW,
    )


def _make_service() -> tuple[AgentShareServiceImpl, SimpleNamespace]:
    """Build the service over closure-backed in-memory repositories.

    Returns the service plus a ``SimpleNamespace`` holding the live
    row stores so tests can seed / inspect state.
    """
    state = SimpleNamespace(
        agents={},
        orgs={},
        members={},
        shares={},
    )

    agent_repo = AsyncMock(spec=CustomAgentRepository)
    org_repo = AsyncMock(spec=OrganizationRepository)
    member_repo = AsyncMock(spec=OrganizationMemberRepository)
    share_repo = AsyncMock(spec=AgentShareRepository)

    async def _get_agent(id: str, tenant_id: int) -> CustomAgent | None:
        row = state.agents.get(id)
        if row is None or row.tenant_id != tenant_id or row.deleted_at is not None:
            return None
        return row

    async def _get_org(id: str) -> Organization | None:
        row = state.orgs.get(id)
        if row is None or row.deleted_at is not None:
            return None
        return row

    async def _get_member(organization_id: str, tenant_id: int) -> OrganizationTenantMember | None:
        return state.members.get((organization_id, tenant_id))

    async def _create_or_none(row: AgentShare) -> AgentShare | None:
        for s in state.shares.values():
            if (
                s.agent_id == row.agent_id
                and s.organization_id == row.organization_id
                and s.source_tenant_id == row.source_tenant_id
                and s.deleted_at is None
            ):
                return None
        state.shares[row.id] = row
        return row

    async def _get_share(share_id: str) -> AgentShare | None:
        row = state.shares.get(share_id)
        if row is None or row.deleted_at is not None:
            return None
        return row

    async def _update(row: AgentShare) -> AgentShare:
        state.shares[row.id] = row
        return row

    async def _soft_delete(id: str, now: datetime) -> bool:
        row = state.shares.get(id)
        if row is None or row.deleted_at is not None:
            return False
        state.shares[id] = row.model_copy(update={"deleted_at": now, "updated_at": now})
        return True

    async def _get_by_agent_and_org(agent_id: str, organization_id: str) -> AgentShare | None:
        for s in state.shares.values():
            if (
                s.agent_id == agent_id
                and s.organization_id == organization_id
                and s.deleted_at is None
            ):
                return s
        return None

    async def _list_by_agent(agent_id: str) -> list[AgentShare]:
        return [s for s in state.shares.values() if s.agent_id == agent_id and s.deleted_at is None]

    agent_repo.get_by_id_and_tenant.side_effect = _get_agent
    org_repo.get_by_id_or_none.side_effect = _get_org
    member_repo.get_member.side_effect = _get_member
    share_repo.create_or_none.side_effect = _create_or_none
    share_repo.get_by_id_or_none.side_effect = _get_share
    share_repo.get_by_agent_and_org_or_none.side_effect = _get_by_agent_and_org
    share_repo.update.side_effect = _update
    share_repo.soft_delete.side_effect = _soft_delete
    share_repo.list_by_agent.side_effect = _list_by_agent

    service = AgentShareServiceImpl(
        agent_repo=agent_repo,
        org_repo=org_repo,
        member_repo=member_repo,
        share_repo=share_repo,
    )
    return service, state


# ── share_agent ─────────────────────────────────────────────────────


async def test_share_agent_forces_viewer_permission() -> None:
    service, state = _make_service()
    state.agents["agent-1"] = _agent("agent-1")
    state.orgs["org-1"] = _org("org-1")
    state.members[("org-1", _TENANT)] = _member(org_id="org-1", tenant_id=_TENANT, role="editor")

    share = await service.share_agent(
        agent_id="agent-1",
        organization_id="org-1",
        user_id=_USER,
        tenant_id=_TENANT,
        permission="editor",
    )

    assert share.permission == "viewer"
    assert share.agent_id == "agent-1"
    assert share.organization_id == "org-1"
    assert share.source_tenant_id == _TENANT


async def test_share_agent_unknown_agent_raises_not_found() -> None:
    service, state = _make_service()
    state.agents["agent-1"] = _agent("agent-1")
    state.orgs["org-1"] = _org("org-1")
    state.members[("org-1", _TENANT)] = _member(org_id="org-1", tenant_id=_TENANT, role="editor")

    with pytest.raises(NotFoundError):
        await service.share_agent(
            agent_id="agent-missing",
            organization_id="org-1",
            user_id=_USER,
            tenant_id=_TENANT,
            permission="viewer",
        )


async def test_share_agent_foreign_agent_raises_not_found() -> None:
    service, state = _make_service()
    state.agents["agent-1"] = _agent("agent-1", tenant_id=_OTHER_TENANT)
    state.orgs["org-1"] = _org("org-1")
    state.members[("org-1", _TENANT)] = _member(org_id="org-1", tenant_id=_TENANT, role="editor")

    # The agent lookup is tenant-scoped, so another tenant's agent is
    # indistinguishable from a missing row (no cross-tenant existence leak).
    with pytest.raises(NotFoundError):
        await service.share_agent(
            agent_id="agent-1",
            organization_id="org-1",
            user_id=_USER,
            tenant_id=_TENANT,
            permission="viewer",
        )


async def test_share_agent_unconfigured_raises_validation() -> None:
    service, state = _make_service()
    state.agents["agent-1"] = _agent("agent-1", model_id="")
    state.orgs["org-1"] = _org("org-1")
    state.members[("org-1", _TENANT)] = _member(org_id="org-1", tenant_id=_TENANT, role="editor")

    with pytest.raises(ValidationError):
        await service.share_agent(
            agent_id="agent-1",
            organization_id="org-1",
            user_id=_USER,
            tenant_id=_TENANT,
            permission="viewer",
        )


async def test_share_agent_unknown_org_raises_not_found() -> None:
    service, state = _make_service()
    state.agents["agent-1"] = _agent("agent-1")
    state.members[("org-1", _TENANT)] = _member(org_id="org-1", tenant_id=_TENANT, role="editor")

    with pytest.raises(NotFoundError):
        await service.share_agent(
            agent_id="agent-1",
            organization_id="org-1",
            user_id=_USER,
            tenant_id=_TENANT,
            permission="viewer",
        )


async def test_share_agent_non_member_raises_permission_denied() -> None:
    service, state = _make_service()
    state.agents["agent-1"] = _agent("agent-1")
    state.orgs["org-1"] = _org("org-1")

    with pytest.raises(PermissionDeniedError):
        await service.share_agent(
            agent_id="agent-1",
            organization_id="org-1",
            user_id=_USER,
            tenant_id=_TENANT,
            permission="viewer",
        )


async def test_share_agent_viewer_role_rejected() -> None:
    service, state = _make_service()
    state.agents["agent-1"] = _agent("agent-1")
    state.orgs["org-1"] = _org("org-1")
    state.members[("org-1", _TENANT)] = _member(org_id="org-1", tenant_id=_TENANT, role="viewer")

    with pytest.raises(PermissionDeniedError):
        await service.share_agent(
            agent_id="agent-1",
            organization_id="org-1",
            user_id=_USER,
            tenant_id=_TENANT,
            permission="viewer",
        )


async def test_share_agent_duplicate_upgrades_to_viewer() -> None:
    service, state = _make_service()
    state.agents["agent-1"] = _agent("agent-1")
    state.orgs["org-1"] = _org("org-1")
    state.members[("org-1", _TENANT)] = _member(org_id="org-1", tenant_id=_TENANT, role="editor")

    existing = _share(agent_id="agent-1", org_id="org-1").model_copy(
        update={"permission": "editor"}
    )
    state.shares[existing.id] = existing

    second = await service.share_agent(
        agent_id="agent-1",
        organization_id="org-1",
        user_id=_USER,
        tenant_id=_TENANT,
        permission="viewer",
    )
    assert second.permission == "viewer"
    assert state.shares[existing.id].permission == "viewer"


# ── remove_share ───────────────────────────────────────────────────


async def test_remove_share_by_original_sharer() -> None:
    service, state = _make_service()
    share = _share(agent_id="agent-1", org_id="org-1")
    state.shares[share.id] = share

    await service.remove_share(
        share_id=share.id,
        user_id=_USER,
        tenant_id=_TENANT,
        tenant_role="viewer",
    )
    assert state.shares[share.id].deleted_at is not None


async def test_remove_share_by_org_admin() -> None:
    service, state = _make_service()
    share = _share(agent_id="agent-1", org_id="org-1").model_copy(
        update={"shared_by_user_id": "usr-other"}
    )
    state.shares[share.id] = share
    state.members[("org-1", _OTHER_TENANT)] = _member(
        org_id="org-1", tenant_id=_OTHER_TENANT, role="admin"
    )

    await service.remove_share(
        share_id=share.id,
        user_id="usr-admin",
        tenant_id=_OTHER_TENANT,
        tenant_role="admin",
    )
    assert state.shares[share.id].deleted_at is not None


async def test_remove_share_by_source_tenant_admin() -> None:
    service, state = _make_service()
    share = _share(agent_id="agent-1", org_id="org-1").model_copy(
        update={"shared_by_user_id": "usr-other"}
    )
    state.shares[share.id] = share

    await service.remove_share(
        share_id=share.id,
        user_id="usr-admin",
        tenant_id=_TENANT,
        tenant_role="admin",
    )
    assert state.shares[share.id].deleted_at is not None


async def test_remove_share_unauthorized_raises() -> None:
    service, state = _make_service()
    share = _share(agent_id="agent-1", org_id="org-1").model_copy(
        update={"shared_by_user_id": "usr-other"}
    )
    state.shares[share.id] = share

    with pytest.raises(PermissionDeniedError):
        await service.remove_share(
            share_id=share.id,
            user_id="usr-random",
            tenant_id=_OTHER_TENANT,
            tenant_role="viewer",
        )


async def test_remove_share_missing_raises_not_found() -> None:
    service, _ = _make_service()
    with pytest.raises(NotFoundError):
        await service.remove_share(
            share_id="missing",
            user_id=_USER,
            tenant_id=_TENANT,
            tenant_role="admin",
        )


# ── list_shares_by_agent ───────────────────────────────────────────


async def test_list_shares_by_agent_returns_live_rows() -> None:
    service, state = _make_service()
    share = _share(agent_id="agent-1", org_id="org-1")
    state.shares[share.id] = share
    gone = share.model_copy(update={"id": "share-gone"})
    state.shares["share-gone"] = gone.model_copy(update={"deleted_at": _NOW})

    shares = await service.list_shares_by_agent(agent_id="agent-1")
    assert [s.id for s in shares] == [share.id]


# ── update_share_permission ────────────────────────────────────────


async def test_update_share_permission_forces_viewer() -> None:
    service, state = _make_service()
    share = _share(agent_id="agent-1", org_id="org-1").model_copy(update={"permission": "editor"})
    state.shares[share.id] = share

    await service.update_share_permission(
        share_id=share.id,
        permission="admin",
        user_id=_USER,
        tenant_id=_TENANT,
        tenant_role="viewer",
    )
    assert state.shares[share.id].permission == "viewer"


async def test_update_share_permission_unauthorized_raises() -> None:
    service, state = _make_service()
    share = _share(agent_id="agent-1", org_id="org-1").model_copy(
        update={"shared_by_user_id": "usr-other"}
    )
    state.shares[share.id] = share

    with pytest.raises(PermissionDeniedError):
        await service.update_share_permission(
            share_id=share.id,
            permission="viewer",
            user_id="usr-stranger",
            tenant_id=_OTHER_TENANT,
            tenant_role="viewer",
        )
