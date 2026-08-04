"""Integration tests for ``ModelRepository`` against a real Postgres.

The session-scoped ``pg_url`` fixture provides the container; each test
gets a fresh ``models`` schema so writes are hermetic. The DDL mirrors
``alembic/versions/models.py``.

The fixture skips the suite when no Docker daemon is available.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import sqlalchemy
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.common.exception import NotFoundError
from src.db.dao.model_repository import ModelRepository
from src.db.models.infra.model import Model

_DROP_MODELS_SQL = sqlalchemy.text("DROP TABLE IF EXISTS models CASCADE")

_CREATE_MODELS_SQL = sqlalchemy.text(
    """
    CREATE TABLE models (
        id VARCHAR(64) PRIMARY KEY,
        tenant_id INTEGER NOT NULL,
        name VARCHAR(255) NOT NULL,
        display_name VARCHAR(255) NOT NULL DEFAULT '',
        type VARCHAR(50) NOT NULL,
        source VARCHAR(50) NOT NULL,
        description TEXT,
        parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
        is_default BOOLEAN NOT NULL DEFAULT FALSE,
        is_builtin BOOLEAN NOT NULL DEFAULT FALSE,
        managed_by VARCHAR(32) NOT NULL DEFAULT '',
        status VARCHAR(50) NOT NULL DEFAULT 'active',
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        deleted_at TIMESTAMP WITH TIME ZONE
    )
    """
)


@pytest.fixture
async def session(pg_url: str) -> AsyncIterator[AsyncSession]:
    engine: AsyncEngine = create_async_engine(pg_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.execute(_DROP_MODELS_SQL)
        await conn.execute(_CREATE_MODELS_SQL)
    async with factory() as s:
        yield s
    async with engine.begin() as conn:
        await conn.execute(_DROP_MODELS_SQL)
    await engine.dispose()


def _sample_row(
    *,
    id: str = "model-1",
    tenant_id: int = 1,
    name: str = "gpt-4o",
    type: str = "KnowledgeQA",
    source: str = "openai",
    is_builtin: bool = False,
    is_default: bool = False,
    parameters: dict[str, object] | None = None,
) -> Model:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return Model.model_validate(
        {
            "id": id,
            "tenant_id": tenant_id,
            "name": name,
            "display_name": name,
            "type": type,
            "source": source,
            "description": None,
            "parameters": parameters or {"provider": "openai"},
            "is_default": is_default,
            "is_builtin": is_builtin,
            "managed_by": "",
            "status": "active",
            "created_at": now,
            "updated_at": now,
            "deleted_at": None,
        }
    )


# ── insert / find_by_tenant_and_id ───────────────────────────────────


async def test_insert_and_resolve_by_tenant_and_id(session: AsyncSession) -> None:
    repo = ModelRepository(session)
    row = _sample_row(id="model-1", tenant_id=1)
    await repo.insert(row)

    resolved = await repo.find_by_tenant_and_id_or_fail(tenant_id=1, id="model-1")
    assert resolved.id == "model-1"
    assert resolved.name == "gpt-4o"


async def test_find_by_tenant_and_id_returns_none_for_unknown(
    session: AsyncSession,
) -> None:
    repo = ModelRepository(session)
    assert await repo.find_by_tenant_and_id(tenant_id=1, id="missing") is None


async def test_find_by_tenant_and_id_raises_not_found_for_unknown(
    session: AsyncSession,
) -> None:
    repo = ModelRepository(session)
    with pytest.raises(NotFoundError) as excinfo:
        await repo.find_by_tenant_and_id_or_fail(tenant_id=1, id="missing")
    assert excinfo.value.code == "model.not_found"


async def test_find_by_tenant_and_id_includes_builtins_by_default(
    session: AsyncSession,
) -> None:
    repo = ModelRepository(session)
    await repo.insert(_sample_row(id="builtin-1", tenant_id=10000, is_builtin=True))

    # Tenant 1 sees the built-in row because include_builtin defaults to True.
    resolved = await repo.find_by_tenant_and_id(tenant_id=1, id="builtin-1")
    assert resolved is not None
    assert resolved.is_builtin is True


async def test_find_by_tenant_and_id_excludes_builtins_when_disabled(
    session: AsyncSession,
) -> None:
    repo = ModelRepository(session)
    await repo.insert(_sample_row(id="builtin-1", tenant_id=10000, is_builtin=True))

    resolved = await repo.find_by_tenant_and_id(tenant_id=1, id="builtin-1", include_builtin=False)
    assert resolved is None


# ── list_by_tenant ───────────────────────────────────────────────────


async def test_list_by_tenant_returns_tenant_rows(session: AsyncSession) -> None:
    repo = ModelRepository(session)
    await repo.insert(_sample_row(id="m-1", tenant_id=1))
    await repo.insert(_sample_row(id="m-2", tenant_id=1))
    await repo.insert(_sample_row(id="m-3", tenant_id=2))

    rows = await repo.list_by_tenant(tenant_id=1)
    ids = {r.id for r in rows}
    assert ids == {"m-1", "m-2"}


async def test_list_by_tenant_includes_builtins_by_default(
    session: AsyncSession,
) -> None:
    repo = ModelRepository(session)
    await repo.insert(_sample_row(id="m-1", tenant_id=1))
    await repo.insert(_sample_row(id="b-1", tenant_id=10000, is_builtin=True))

    rows = await repo.list_by_tenant(tenant_id=1)
    ids = {r.id for r in rows}
    assert ids == {"m-1", "b-1"}


async def test_list_by_tenant_filters_by_type(session: AsyncSession) -> None:
    repo = ModelRepository(session)
    await repo.insert(_sample_row(id="m-1", tenant_id=1, type="KnowledgeQA"))
    await repo.insert(_sample_row(id="m-2", tenant_id=1, type="Embedding"))

    rows = await repo.list_by_tenant(tenant_id=1, model_type="Embedding")
    ids = {r.id for r in rows}
    assert ids == {"m-2"}


async def test_list_by_tenant_filters_by_source(session: AsyncSession) -> None:
    repo = ModelRepository(session)
    await repo.insert(_sample_row(id="m-1", tenant_id=1, source="openai"))
    await repo.insert(_sample_row(id="m-2", tenant_id=1, source="local"))

    rows = await repo.list_by_tenant(tenant_id=1, source="local")
    ids = {r.id for r in rows}
    assert ids == {"m-2"}


# ── update_row ───────────────────────────────────────────────────────


async def test_update_row_returns_refreshed_row(session: AsyncSession) -> None:
    repo = ModelRepository(session)
    await repo.insert(_sample_row(id="m-1", tenant_id=1))

    row = await repo.find_by_tenant_and_id_or_fail(tenant_id=1, id="m-1")
    updated = row.model_copy(update={"name": "renamed"})
    refreshed = await repo.update_row(updated)

    assert refreshed is not None
    assert refreshed.name == "renamed"


async def test_update_row_returns_none_for_unknown_id(session: AsyncSession) -> None:
    repo = ModelRepository(session)
    row = _sample_row(id="m-1", tenant_id=1)
    assert await repo.update_row(row) is None


# ── delete_by_tenant_and_id ──────────────────────────────────────────


async def test_delete_by_tenant_and_id_removes_row(session: AsyncSession) -> None:
    repo = ModelRepository(session)
    await repo.insert(_sample_row(id="m-1", tenant_id=1))

    affected = await repo.delete_by_tenant_and_id(tenant_id=1, id="m-1")
    assert affected == 1
    assert await repo.find_by_tenant_and_id(tenant_id=1, id="m-1") is None


async def test_delete_by_tenant_and_id_returns_zero_when_absent(
    session: AsyncSession,
) -> None:
    repo = ModelRepository(session)
    affected = await repo.delete_by_tenant_and_id(tenant_id=1, id="missing")
    assert affected == 0


# ── clear_default_by_type ────────────────────────────────────────────


async def test_clear_default_by_type_flips_flag(session: AsyncSession) -> None:
    repo = ModelRepository(session)
    await repo.insert(_sample_row(id="m-1", tenant_id=1, is_default=True))
    await repo.insert(_sample_row(id="m-2", tenant_id=1, is_default=True))
    await repo.insert(_sample_row(id="m-3", tenant_id=1, is_default=False))

    affected = await repo.clear_default_by_type(tenant_id=1, model_type="KnowledgeQA")
    assert affected == 2

    after = await repo.list_by_tenant(tenant_id=1, include_builtin=False)
    assert all(not r.is_default for r in after)


async def test_clear_default_by_type_respects_exclude_id(
    session: AsyncSession,
) -> None:
    repo = ModelRepository(session)
    await repo.insert(_sample_row(id="m-1", tenant_id=1, is_default=True))
    await repo.insert(_sample_row(id="m-2", tenant_id=1, is_default=True))

    affected = await repo.clear_default_by_type(
        tenant_id=1,
        model_type="KnowledgeQA",
        exclude_id="m-1",
    )
    assert affected == 1

    keep = await repo.find_by_tenant_and_id_or_fail(tenant_id=1, id="m-1")
    other = await repo.find_by_tenant_and_id_or_fail(tenant_id=1, id="m-2")
    assert keep.is_default is True
    assert other.is_default is False


# ── parameters JSONB round-trip ──────────────────────────────────────


async def test_parameters_round_trip_preserves_jsonb_shape(
    session: AsyncSession,
) -> None:
    repo = ModelRepository(session)
    params: dict[str, object] = {
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "embedding_parameters": {"dimension": 1536},
        "extra_config": {"region": "us"},
    }
    await repo.insert(_sample_row(id="m-1", tenant_id=1, parameters=params))

    resolved = await repo.find_by_tenant_and_id_or_fail(tenant_id=1, id="m-1")
    params_back = resolved.parameters
    assert isinstance(params_back, dict)
    assert params_back["provider"] == "openai"
    embedding = params_back["embedding_parameters"]
    extra = params_back["extra_config"]
    assert isinstance(embedding, dict)
    assert isinstance(extra, dict)
    assert embedding["dimension"] == 1536
    assert extra["region"] == "us"
