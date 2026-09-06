# The copy-name suffix uses fullwidth parentheses.

"""Unit + integration tests for ``CustomAgentService``.

Unit tests drive the service against an in-memory fake that mirrors the
``CustomAgentRepository`` contract (the same protocol the factory hands
the service at request time): they cover validation, error
classification, config defaulting, built-in protection and the copy
flow.

Integration tests run against the real applied schema. ``custom_agents``
carries an INTEGER (32-bit) ``tenant_id`` column, so those tests use an
int32-safe tenant id (a local counter) instead of
``make_test_tenant_id``'s BIGINT range, which would overflow it.
"""

from __future__ import annotations

import itertools
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from random import randint
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from faker import Faker
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.common.exception import ConflictError, DataError, NotFoundError, ValidationError
from src.common.json import JsonObject
from src.core.agents.builtin_registry import get_builtin_agent
from src.core.agents.service.custom_agent_service import (
    BUILTIN_AGENT_ORDER,
    CustomAgentService,
)
from src.core.agents.service.factory import build_custom_agent_service
from src.core.agents.types import CustomAgentInfo
from src.db.dao.custom_agent_repository import CustomAgentRepository
from src.db.models.custom_agent import CustomAgent
from src.settings import get_settings, reset_settings_cache
from tests.integration.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_FAKER_SEED_MAX = 100_000_000

# ``custom_agents.tenant_id`` is INTEGER (32-bit); integration tests mint
# ids from this counter so they stay inside the range.
_INT32_TENANT_BASE = 2_000_000
_INT32_TENANT_SEQ = itertools.count(start=1)


def _int32_tenant_id() -> int:
    """Return a tenant id unique within the session, safe for INTEGER."""
    return _INT32_TENANT_BASE + next(_INT32_TENANT_SEQ)


@pytest.fixture(autouse=True)
def faker_seed() -> None:
    """Re-seed Faker per test for varied-but-reproducible generation."""
    Faker.seed(randint(1, _FAKER_SEED_MAX))


# ── Sample config ─────────────────────────────────────────────────────


def _config(**overrides: Any) -> JsonObject:
    """Build an agent config dict, defaulting the mode to quick-answer."""
    base: JsonObject = {"agent_mode": "quick-answer"}
    base.update(overrides)
    return base


# ── Unit-test fake repository ─────────────────────────────────────────


def _make_repo() -> tuple[AsyncMock, dict[tuple[str, int], CustomAgent]]:
    """``AsyncMock(spec=CustomAgentRepository)`` with closure-captured state."""
    repo = AsyncMock(spec=CustomAgentRepository)
    rows: dict[tuple[str, int], CustomAgent] = {}
    repo.rows = rows

    async def _create(row: CustomAgent) -> CustomAgent:
        rows[(row.id, row.tenant_id)] = row
        return row

    async def _get(*, id: str, tenant_id: int) -> CustomAgent | None:
        row = rows.get((id, tenant_id))
        if row is None or row.deleted_at is not None:
            return None
        return row

    async def _list(tenant_id: int) -> list[CustomAgent]:
        live = [r for (_, t), r in rows.items() if t == tenant_id and r.deleted_at is None]
        return sorted(live, key=lambda r: r.created_at, reverse=True)

    async def _update(row: CustomAgent) -> CustomAgent:
        if (row.id, row.tenant_id) not in rows:
            raise DataError(
                code="custom_agent.update_no_row",
                message=f"custom agent {row.id} not found for update",
            )
        rows[(row.id, row.tenant_id)] = row
        return row

    async def _soft_delete(*, id: str, tenant_id: int, now: datetime) -> bool:
        existing = rows.get((id, tenant_id))
        if existing is None or existing.deleted_at is not None:
            return False
        rows[(id, tenant_id)] = existing.model_copy(update={"deleted_at": now, "updated_at": now})
        return True

    repo.create.side_effect = _create
    repo.get_by_id_and_tenant.side_effect = _get
    repo.list_by_tenant.side_effect = _list
    repo.update.side_effect = _update
    repo.soft_delete.side_effect = _soft_delete
    return repo, rows


@pytest.fixture
def repo_and_rows() -> tuple[AsyncMock, dict[tuple[str, int], CustomAgent]]:
    return _make_repo()


@pytest.fixture
def repo(repo_and_rows: tuple[AsyncMock, dict[tuple[str, int], CustomAgent]]) -> AsyncMock:
    return repo_and_rows[0]


@pytest.fixture
def rows(
    repo_and_rows: tuple[AsyncMock, dict[tuple[str, int], CustomAgent]],
) -> dict[tuple[str, int], CustomAgent]:
    return repo_and_rows[1]


@pytest.fixture
def service(repo: AsyncMock) -> CustomAgentService:
    return CustomAgentService(agent_repo=repo)


def _seed_builtin(
    rows: dict[tuple[str, int], CustomAgent],
    *,
    tenant_id: int,
    id: str = "builtin-quick-answer",
) -> None:
    """Insert a built-in row directly into the fake for built-in tests."""
    rows[(id, tenant_id)] = CustomAgent(
        id=id,
        name="快速问答",
        description="RAG 问答",
        avatar="💬",
        is_builtin=True,
        tenant_id=tenant_id,
        created_by=None,
        config=_config(),
        created_at=_NOW,
        updated_at=_NOW,
    )


# ── create_agent ──────────────────────────────────────────────────────


async def test_create_agent_persists_a_new_row(
    service: CustomAgentService,
    repo: AsyncMock,
) -> None:
    info = await service.create_agent(
        tenant_id=1001,
        name="QA assistant",
        config=_config(),
    )

    assert isinstance(info, CustomAgentInfo)
    assert info.tenant_id == 1001
    assert info.name == "QA assistant"
    assert info.is_builtin is False
    stored = repo.rows[(info.id, 1001)]
    assert stored.tenant_id == 1001
    assert stored.config["agent_mode"] == "quick-answer"


async def test_create_agent_generates_id_when_unspecified(
    service: CustomAgentService,
) -> None:
    info = await service.create_agent(tenant_id=1001, name="QA", config=_config())

    assert len(info.id.split("-")) == 5


async def test_create_agent_honours_supplied_id(
    service: CustomAgentService,
) -> None:
    info = await service.create_agent(
        tenant_id=1001,
        name="QA",
        config=_config(),
        agent_id="custom-001",
    )

    assert info.id == "custom-001"


async def test_create_agent_rejects_blank_name(
    service: CustomAgentService,
) -> None:
    with pytest.raises(ValidationError) as excinfo:
        await service.create_agent(tenant_id=1001, name="   ", config=_config())

    assert excinfo.value.code == "agent.name_required"


async def test_create_agent_rejects_non_positive_tenant_id(
    service: CustomAgentService,
) -> None:
    with pytest.raises(ValidationError) as excinfo:
        await service.create_agent(tenant_id=0, name="QA", config=_config())

    assert excinfo.value.code == "agent.tenant_required"


async def test_create_agent_stamps_creator_for_real_user(
    service: CustomAgentService,
    repo: AsyncMock,
) -> None:
    info = await service.create_agent(
        tenant_id=1001,
        name="QA",
        config=_config(),
        user_id="usr-abc",
    )

    assert info.created_by == "usr-abc"
    assert repo.rows[(info.id, 1001)].created_by == "usr-abc"


async def test_create_agent_skips_synthetic_creator(
    service: CustomAgentService,
    repo: AsyncMock,
) -> None:
    info = await service.create_agent(
        tenant_id=1001,
        name="QA",
        config=_config(),
        user_id="system-1001",
    )

    assert info.created_by is None
    assert repo.rows[(info.id, 1001)].created_by is None


async def test_create_agent_defaults_quick_answer_mode(
    service: CustomAgentService,
    repo: AsyncMock,
) -> None:
    info = await service.create_agent(
        tenant_id=1001,
        name="QA",
        config={"system_prompt": "be brief"},
    )

    stored = repo.rows[(info.id, 1001)]
    assert stored.config["agent_mode"] == "quick-answer"
    suggestions = stored.config["question_suggestions"]
    assert isinstance(suggestions, dict)
    starters = suggestions["starters"]
    follow_ups = suggestions["follow_ups"]
    assert isinstance(starters, dict)
    assert isinstance(follow_ups, dict)
    assert starters["enabled"] is True
    assert follow_ups["count"] == 3


async def test_create_agent_smart_reasoning_pins_multi_turn(
    service: CustomAgentService,
    repo: AsyncMock,
) -> None:
    info = await service.create_agent(
        tenant_id=1001,
        name="Reasoner",
        config=_config(agent_mode="smart-reasoning"),
    )

    stored = repo.rows[(info.id, 1001)]
    assert stored.config["multi_turn_enabled"] is True


async def test_create_agent_rejects_invalid_suggestion_count(
    service: CustomAgentService,
) -> None:
    config = _config(question_suggestions={"starters": {"count": 9}})

    with pytest.raises(ValidationError) as excinfo:
        await service.create_agent(tenant_id=1001, name="QA", config=config)

    assert excinfo.value.code == "agent.suggestion_count_invalid"


async def test_create_agent_rejects_blank_starter_item(
    service: CustomAgentService,
) -> None:
    config = _config(
        question_suggestions={
            "starters": {"items": ["   "], "count": 2},
        }
    )

    with pytest.raises(ValidationError) as excinfo:
        await service.create_agent(tenant_id=1001, name="QA", config=config)

    assert excinfo.value.code == "agent.suggestion_item_empty"


# ── get_agent_by_id ───────────────────────────────────────────────────


async def test_get_agent_by_id_returns_projection(
    service: CustomAgentService,
) -> None:
    created = await service.create_agent(tenant_id=1001, name="QA", config=_config())

    info = await service.get_agent_by_id(tenant_id=1001, agent_id=created.id)

    assert info.id == created.id
    assert info.name == "QA"


async def test_get_agent_by_id_raises_not_found(
    service: CustomAgentService,
) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        await service.get_agent_by_id(tenant_id=1001, agent_id="missing")

    assert excinfo.value.code == "agent.not_found"


async def test_get_agent_by_id_rejects_empty_id(
    service: CustomAgentService,
) -> None:
    with pytest.raises(ValidationError) as excinfo:
        await service.get_agent_by_id(tenant_id=1001, agent_id="  ")

    assert excinfo.value.code == "agent.id_required"


async def test_get_agent_by_id_and_tenant_enforces_isolation(
    service: CustomAgentService,
) -> None:
    created = await service.create_agent(tenant_id=1001, name="QA", config=_config())

    assert (
        await service.get_agent_by_id_and_tenant(
            agent_id=created.id,
            tenant_id=1001,
        )
    ).id == created.id

    with pytest.raises(NotFoundError):
        await service.get_agent_by_id_and_tenant(
            agent_id=created.id,
            tenant_id=9999,
        )


async def test_get_agent_by_id_applies_config_defaults(
    service: CustomAgentService,
    rows: dict[tuple[str, int], CustomAgent],
) -> None:
    tenant_id = make_test_tenant_id()
    # A legacy row persisted without defaults (config empty).
    rows[(f"legacy-{tenant_id}", tenant_id)] = CustomAgent(
        id=f"legacy-{tenant_id}",
        name="legacy",
        is_builtin=False,
        tenant_id=tenant_id,
        config={},
        created_at=_NOW,
        updated_at=_NOW,
    )

    info = await service.get_agent_by_id(tenant_id=tenant_id, agent_id=f"legacy-{tenant_id}")

    # Reads apply the suggestion-block defaults; the agent-mode default is
    # a create-path concern (mirroring the upstream EnsureDefaults split).
    assert "question_suggestions" in info.config
    suggestions = info.config["question_suggestions"]
    assert isinstance(suggestions, dict)
    starters = suggestions["starters"]
    follow_ups = suggestions["follow_ups"]
    assert isinstance(starters, dict)
    assert isinstance(follow_ups, dict)
    assert starters["count"] == 6
    assert follow_ups["max_context_turns"] == 2


# ── list_agents ───────────────────────────────────────────────────────


async def test_list_agents_returns_tenant_rows_only(
    service: CustomAgentService,
) -> None:
    await service.create_agent(tenant_id=1001, name="a1", config=_config())
    await service.create_agent(tenant_id=1001, name="a2", config=_config())
    await service.create_agent(tenant_id=2002, name="other", config=_config())

    infos = await service.list_agents(tenant_id=1001)

    # Built-in presets lead in registry order; custom tenant rows follow
    # newest-first. The other tenant's row never appears.
    names = [i.name for i in infos]
    assert names[: len(BUILTIN_AGENT_ORDER)] == [
        get_builtin_agent(bid, 1001).name for bid in BUILTIN_AGENT_ORDER
    ]
    assert names[len(BUILTIN_AGENT_ORDER) :] == ["a2", "a1"]


# ── update_agent ──────────────────────────────────────────────────────


async def test_update_agent_overwrites_mutable_fields(
    service: CustomAgentService,
) -> None:
    created = await service.create_agent(tenant_id=1001, name="before", config=_config())

    updated = await service.update_agent(
        tenant_id=1001,
        agent_id=created.id,
        name="after",
        description="new desc",
        config=_config(model_id="chat-2"),
    )

    assert updated.name == "after"
    assert updated.description == "new desc"
    assert updated.config["model_id"] == "chat-2"
    assert updated.is_builtin is False


async def test_update_agent_rejects_blank_name(
    service: CustomAgentService,
) -> None:
    created = await service.create_agent(tenant_id=1001, name="QA", config=_config())

    with pytest.raises(ValidationError) as excinfo:
        await service.update_agent(
            tenant_id=1001,
            agent_id=created.id,
            name="   ",
            config=_config(),
        )

    assert excinfo.value.code == "agent.name_required"


async def test_update_agent_rejects_builtin(
    service: CustomAgentService,
    rows: dict[tuple[str, int], CustomAgent],
) -> None:
    _seed_builtin(rows, tenant_id=1001)

    with pytest.raises(ConflictError) as excinfo:
        await service.update_agent(
            tenant_id=1001,
            agent_id="builtin-quick-answer",
            name="renamed",
            config=_config(),
        )

    assert excinfo.value.code == "agent.cannot_modify_builtin"


async def test_update_agent_allows_builtin_config(
    service: CustomAgentService,
    rows: dict[tuple[str, int], CustomAgent],
) -> None:
    _seed_builtin(rows, tenant_id=1001)

    updated = await service.update_agent(
        tenant_id=1001,
        agent_id="builtin-quick-answer",
        name="快速问答",
        config=_config(model_id="model-qa"),
    )

    assert updated.name == "快速问答"
    assert updated.config["model_id"] == "model-qa"
    assert updated.is_builtin is True


async def test_update_agent_materializes_missing_builtin(
    service: CustomAgentService,
) -> None:
    updated = await service.update_agent(
        tenant_id=1001,
        agent_id="builtin-quick-answer",
        name="快速问答",
        config=_config(model_id="model-qa"),
    )

    assert updated.is_builtin is True
    assert updated.config["model_id"] == "model-qa"


async def test_update_agent_raises_not_found(
    service: CustomAgentService,
) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        await service.update_agent(
            tenant_id=1001,
            agent_id="missing",
            name="x",
            config=_config(),
        )

    assert excinfo.value.code == "agent.not_found"


# ── delete_agent ──────────────────────────────────────────────────────


async def test_delete_agent_soft_deletes_row(
    service: CustomAgentService,
) -> None:
    created = await service.create_agent(tenant_id=1001, name="QA", config=_config())

    await service.delete_agent(tenant_id=1001, agent_id=created.id)

    with pytest.raises(NotFoundError):
        await service.get_agent_by_id(tenant_id=1001, agent_id=created.id)


async def test_delete_agent_rejects_known_builtin_id(
    service: CustomAgentService,
) -> None:
    with pytest.raises(ConflictError) as excinfo:
        await service.delete_agent(tenant_id=1001, agent_id="builtin-quick-answer")

    assert excinfo.value.code == "agent.cannot_delete_builtin"


async def test_delete_agent_rejects_stored_builtin(
    service: CustomAgentService,
    rows: dict[tuple[str, int], CustomAgent],
) -> None:
    _seed_builtin(rows, tenant_id=1001)

    with pytest.raises(ConflictError) as excinfo:
        await service.delete_agent(tenant_id=1001, agent_id="builtin-quick-answer")

    assert excinfo.value.code == "agent.cannot_delete_builtin"


async def test_delete_agent_raises_not_found(
    service: CustomAgentService,
) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        await service.delete_agent(tenant_id=1001, agent_id="missing")

    assert excinfo.value.code == "agent.not_found"


# ── copy_agent ────────────────────────────────────────────────────────


async def test_copy_agent_creates_a_fresh_copy(
    service: CustomAgentService,
    repo: AsyncMock,
) -> None:
    created = await service.create_agent(
        tenant_id=1001,
        name="QA",
        config=_config(model_id="chat-1"),
    )

    copied = await service.copy_agent(
        tenant_id=1001,
        agent_id=created.id,
        user_id="usr-copy",
    )

    assert copied.id != created.id
    assert copied.name == "QA （副本）"
    assert copied.config["model_id"] == "chat-1"
    assert copied.is_builtin is False
    assert copied.created_by == "usr-copy"
    assert (copied.id, 1001) in repo.rows


async def test_copy_agent_of_builtin_is_never_builtin(
    service: CustomAgentService,
    rows: dict[tuple[str, int], CustomAgent],
) -> None:
    _seed_builtin(rows, tenant_id=1001)

    copied = await service.copy_agent(tenant_id=1001, agent_id="builtin-quick-answer")

    assert copied.id != "builtin-quick-answer"
    assert copied.is_builtin is False


async def test_copy_agent_skips_synthetic_owner(
    service: CustomAgentService,
) -> None:
    created = await service.create_agent(tenant_id=1001, name="QA", config=_config())

    copied = await service.copy_agent(
        tenant_id=1001,
        agent_id=created.id,
        user_id="system-1001",
    )

    assert copied.created_by is None


async def test_copy_agent_raises_not_found(
    service: CustomAgentService,
) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        await service.copy_agent(tenant_id=1001, agent_id="missing")

    assert excinfo.value.code == "agent.not_found"


# ── factory ───────────────────────────────────────────────────────────


async def test_factory_builds_service_on_session() -> None:
    session = AsyncMock(spec=AsyncSession)
    service = build_custom_agent_service(session)

    assert isinstance(service, CustomAgentService)


# ── Integration (real applied schema) ─────────────────────────────────


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


def _agent(
    *,
    tenant_id: int,
    id: str | None = None,
    name: str = "QA assistant",
    is_builtin: bool = False,
    config: JsonObject | None = None,
) -> CustomAgent:
    """Build a persisted-shape custom agent row for real-DB inserts."""
    now = datetime.now(UTC)
    return CustomAgent(
        id=id or str(uuid.uuid4()),
        name=name,
        description=None,
        avatar=None,
        is_builtin=is_builtin,
        tenant_id=tenant_id,
        created_by=None,
        config=config or {},
        created_at=now,
        updated_at=now,
    )


async def test_integration_create_and_fetch_round_trip(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    service = CustomAgentService(agent_repo=CustomAgentRepository(session))

    info = await service.create_agent(
        tenant_id=tenant_id,
        name="QA assistant",
        config={"model_id": "chat-1"},
    )
    await session.commit()

    fetched = await service.get_agent_by_id(tenant_id=tenant_id, agent_id=info.id)
    assert fetched.name == "QA assistant"
    assert fetched.config["model_id"] == "chat-1"
    suggestions = fetched.config["question_suggestions"]
    assert isinstance(suggestions, dict)
    starters = suggestions["starters"]
    assert isinstance(starters, dict)
    assert starters["enabled"] is True


async def test_integration_tenant_isolation(session: AsyncSession) -> None:
    owner = _int32_tenant_id()
    stranger = _int32_tenant_id()
    service = CustomAgentService(agent_repo=CustomAgentRepository(session))
    info = await service.create_agent(tenant_id=owner, name="QA", config={})
    await session.commit()

    with pytest.raises(NotFoundError):
        await service.get_agent_by_id(tenant_id=stranger, agent_id=info.id)


async def test_integration_update_persists(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    service = CustomAgentService(agent_repo=CustomAgentRepository(session))
    created = await service.create_agent(tenant_id=tenant_id, name="before", config={})
    await session.commit()

    updated = await service.update_agent(
        tenant_id=tenant_id,
        agent_id=created.id,
        name="after",
        config={"model_id": "chat-2"},
    )
    await session.commit()

    assert updated.name == "after"
    assert updated.config["model_id"] == "chat-2"
    fetched = await service.get_agent_by_id(tenant_id=tenant_id, agent_id=created.id)
    assert fetched.name == "after"


async def test_integration_soft_delete_hides_row(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    service = CustomAgentService(agent_repo=CustomAgentRepository(session))
    created = await service.create_agent(tenant_id=tenant_id, name="QA", config={})
    await session.commit()

    await service.delete_agent(tenant_id=tenant_id, agent_id=created.id)
    await session.commit()

    with pytest.raises(NotFoundError):
        await service.get_agent_by_id(tenant_id=tenant_id, agent_id=created.id)


async def test_integration_list_orders_newest_first(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    repo = CustomAgentRepository(session)
    older = _agent(tenant_id=tenant_id, name="older", config={}).model_copy(
        update={"created_at": _NOW, "updated_at": _NOW}
    )
    newer = _agent(tenant_id=tenant_id, name="newer", config={}).model_copy(
        update={
            "created_at": _NOW.replace(year=2027),
            "updated_at": _NOW.replace(year=2027),
        }
    )
    await repo.create(older)
    await repo.create(newer)
    await session.commit()

    service = CustomAgentService(agent_repo=CustomAgentRepository(session))
    infos = await service.list_agents(tenant_id=tenant_id)

    # Relative order only — the tenant may hold rows left by prior runs
    # (the int32 counter reuses tenant ids across processes).
    ids = [i.id for i in infos]
    assert ids.index(newer.id) < ids.index(older.id)


async def test_integration_count_by_model_id(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    repo = CustomAgentRepository(session)
    # Run-unique model ids keep the count scoped to fresh rows even when
    # the tenant id is reused by a prior pytest run.
    chat_model = f"chat-{uuid.uuid4().hex[:8]}"
    rerank_model = f"rerank-{uuid.uuid4().hex[:8]}"
    fu_model = f"fu-{uuid.uuid4().hex[:8]}"
    missing_model = f"missing-{uuid.uuid4().hex[:8]}"
    for _ in range(2):
        await repo.create(_agent(tenant_id=tenant_id, config={"model_id": chat_model}))
    await repo.create(_agent(tenant_id=tenant_id, config={"rerank_model_id": rerank_model}))
    await repo.create(
        _agent(
            tenant_id=tenant_id,
            config={"question_suggestions": {"follow_ups": {"model_id": fu_model}}},
        )
    )
    doomed = await repo.create(_agent(tenant_id=tenant_id, config={"model_id": chat_model}))
    await session.commit()
    await repo.soft_delete(id=doomed.id, tenant_id=tenant_id, now=datetime.now(UTC))
    await session.commit()

    assert await repo.count_by_model_id(tenant_id=tenant_id, model_id=chat_model) == 2
    assert await repo.count_by_model_id(tenant_id=tenant_id, model_id=rerank_model) == 1
    assert await repo.count_by_model_id(tenant_id=tenant_id, model_id=fu_model) == 1
    assert await repo.count_by_model_id(tenant_id=tenant_id, model_id=missing_model) == 0
