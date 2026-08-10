"""Tests for the agent share repository against the real applied schema.

Tests insert unique rows per run; isolation relies on unique share ids
and tenant ids. Tests commit explicitly.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.db.dao.agent_share_repository import AgentShareRepository
from src.db.models.agent_share import AgentShare
from src.settings import get_settings, reset_settings_cache
from tests.integration.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Per-test session against the real applied schema (no cleanup)."""
    reset_settings_cache()
    settings = get_settings()
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as s:
            yield s
    finally:
        await engine.dispose()


def _uid() -> str:
    return f"share-{uuid.uuid4().hex[:12]}"


def _share(
    *,
    agent_id: str | None = None,
    organization_id: str | None = None,
    tenant_id: int | None = None,
    permission: str = "viewer",
) -> AgentShare:
    return AgentShare(
        id=_uid(),
        agent_id=agent_id or f"agent-{uuid.uuid4().hex[:12]}",
        organization_id=organization_id or f"org-{uuid.uuid4().hex[:12]}",
        shared_by_user_id=f"usr-{uuid.uuid4().hex[:12]}",
        source_tenant_id=tenant_id if tenant_id is not None else make_test_tenant_id(),
        permission=permission,
        created_at=_NOW,
        updated_at=_NOW,
    )


async def test_create_and_get_by_id(session: AsyncSession) -> None:
    repo = AgentShareRepository(session)
    share = _share()

    stored = await repo.create(share)
    await session.commit()

    assert stored.id == share.id
    fetched = await repo.get_by_id_or_none(share.id)
    assert fetched is not None
    assert fetched.agent_id == share.agent_id
    assert fetched.permission == "viewer"


async def test_get_by_agent_and_org(session: AsyncSession) -> None:
    repo = AgentShareRepository(session)
    share = _share()
    await repo.create(share)
    await session.commit()

    fetched = await repo.get_by_agent_and_org_or_none(
        agent_id=share.agent_id,
        organization_id=share.organization_id,
    )

    assert fetched is not None
    assert fetched.id == share.id


async def test_duplicate_live_share_is_suppressed(session: AsyncSession) -> None:
    repo = AgentShareRepository(session)
    share = _share()

    first = await repo.create_or_none(share)
    await session.commit()
    duplicate = await repo.create_or_none(share)
    await session.commit()

    assert first is not None
    assert duplicate is None


async def test_soft_deleted_share_can_be_recreated(session: AsyncSession) -> None:
    repo = AgentShareRepository(session)
    share = _share()
    await repo.create(share)
    await session.commit()

    deleted = await repo.soft_delete(id=share.id, now=_NOW)
    await session.commit()
    # A fresh share of the same (agent, source tenant, org) tuple gets a
    # new id; the partial unique index only guards live rows, so the
    # tuple is reusable.
    recreated = _share(
        agent_id=share.agent_id,
        organization_id=share.organization_id,
        tenant_id=share.source_tenant_id,
    )
    stored = await repo.create_or_none(recreated)
    await session.commit()

    assert deleted is True
    assert stored is not None
    assert stored.id != share.id


async def test_soft_delete_hides_row(session: AsyncSession) -> None:
    repo = AgentShareRepository(session)
    share = _share()
    await repo.create(share)
    await session.commit()

    deleted = await repo.soft_delete(id=share.id, now=_NOW)
    await session.commit()

    assert deleted is True
    assert await repo.get_by_id_or_none(share.id) is None


async def test_list_by_agent(session: AsyncSession) -> None:
    repo = AgentShareRepository(session)
    agent_id = f"agent-{uuid.uuid4().hex[:12]}"
    await repo.create(_share(agent_id=agent_id))
    await repo.create(_share(agent_id=agent_id))
    await session.commit()

    shares = await repo.list_by_agent(agent_id)

    assert len(shares) == 2


async def test_list_by_organization(session: AsyncSession) -> None:
    repo = AgentShareRepository(session)
    org_id = f"org-{uuid.uuid4().hex[:12]}"
    await repo.create(_share(organization_id=org_id))
    await repo.create(_share(organization_id=org_id))
    await session.commit()

    shares = await repo.list_by_organization(org_id)

    assert len(shares) == 2


async def test_count_by_agent(session: AsyncSession) -> None:
    repo = AgentShareRepository(session)
    agent_id = f"agent-{uuid.uuid4().hex[:12]}"
    await repo.create(_share(agent_id=agent_id))
    await repo.create(_share(agent_id=agent_id))
    await session.commit()

    count = await repo.count_by_agent(agent_id)

    assert count == 2


async def test_delete_by_agent(session: AsyncSession) -> None:
    repo = AgentShareRepository(session)
    agent_id = f"agent-{uuid.uuid4().hex[:12]}"
    tenant_id = make_test_tenant_id()
    await repo.create(_share(agent_id=agent_id, tenant_id=tenant_id))
    await repo.create(_share(agent_id=agent_id, tenant_id=tenant_id))
    await session.commit()

    affected = await repo.delete_by_agent(
        agent_id=agent_id,
        source_tenant_id=tenant_id,
        now=_NOW,
    )
    await session.commit()

    assert affected == 2
    assert await repo.list_by_agent(agent_id) == []
