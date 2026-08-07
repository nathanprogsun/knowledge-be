"""Integration tests for ``ModelRepository`` against the real applied schema.

Tests insert unique rows per run; isolation relies on unique model ids and
tenant ids. Tests commit explicitly.
"""

from __future__ import annotations

import itertools
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.exception import NotFoundError
from src.db.dao.model_repository import ModelRepository
from src.db.models.infra.model import Model

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_model_tenant_counter = itertools.count(2_000_000)


def _model_tenant_id() -> int:
    """Return a unique 32-bit tenant id for the models table."""
    return next(_model_tenant_counter)


def _mid() -> str:
    return f"model-{uuid.uuid4().hex[:12]}"


def _sample_row(
    *,
    id: str | None = None,
    tenant_id: int | None = None,
    name: str = "gpt-4o",
    type: str = "KnowledgeQA",
    source: str = "openai",
    is_builtin: bool = False,
    is_default: bool = False,
    parameters: dict[str, object] | None = None,
) -> Model:
    return Model.model_validate(
        {
            "id": id or _mid(),
            "tenant_id": tenant_id if tenant_id is not None else _model_tenant_id(),
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
            "created_at": _NOW,
            "updated_at": _NOW,
            "deleted_at": None,
        }
    )


# ── insert / find_by_tenant_and_id ───────────────────────────────────


async def test_insert_and_resolve_by_tenant_and_id(session: AsyncSession) -> None:
    repo = ModelRepository(session)
    tid = _model_tenant_id()
    mid = _mid()
    await repo.insert(_sample_row(id=mid, tenant_id=tid))

    resolved = await repo.find_by_tenant_and_id_or_fail(tenant_id=tid, id=mid)
    assert resolved.id == mid
    assert resolved.name == "gpt-4o"


async def test_find_by_tenant_and_id_returns_none_for_unknown(
    session: AsyncSession,
) -> None:
    repo = ModelRepository(session)
    assert await repo.find_by_tenant_and_id(tenant_id=_model_tenant_id(), id=_mid()) is None


async def test_find_by_tenant_and_id_raises_not_found_for_unknown(
    session: AsyncSession,
) -> None:
    repo = ModelRepository(session)
    with pytest.raises(NotFoundError) as excinfo:
        await repo.find_by_tenant_and_id_or_fail(tenant_id=_model_tenant_id(), id=_mid())
    assert excinfo.value.code == "model.not_found"


async def test_find_by_tenant_and_id_includes_builtins_by_default(
    session: AsyncSession,
) -> None:
    repo = ModelRepository(session)
    builtin_tid = _model_tenant_id()
    regular_tid = _model_tenant_id()
    mid = _mid()
    await repo.insert(_sample_row(id=mid, tenant_id=builtin_tid, is_builtin=True))

    resolved = await repo.find_by_tenant_and_id(tenant_id=regular_tid, id=mid)
    assert resolved is not None
    assert resolved.is_builtin is True


async def test_find_by_tenant_and_id_excludes_builtins_when_disabled(
    session: AsyncSession,
) -> None:
    repo = ModelRepository(session)
    builtin_tid = _model_tenant_id()
    regular_tid = _model_tenant_id()
    mid = _mid()
    await repo.insert(_sample_row(id=mid, tenant_id=builtin_tid, is_builtin=True))

    resolved = await repo.find_by_tenant_and_id(
        tenant_id=regular_tid, id=mid, include_builtin=False
    )
    assert resolved is None


# ── list_by_tenant ───────────────────────────────────────────────────


async def test_list_by_tenant_returns_tenant_rows(session: AsyncSession) -> None:
    repo = ModelRepository(session)
    tid = _model_tenant_id()
    other_tid = _model_tenant_id()
    await repo.insert(_sample_row(id=_mid(), tenant_id=tid))
    await repo.insert(_sample_row(id=_mid(), tenant_id=tid))
    await repo.insert(_sample_row(id=_mid(), tenant_id=other_tid))

    rows = await repo.list_by_tenant(tenant_id=tid, include_builtin=False)
    ids = {r.id for r in rows}
    assert len(ids) == 2


async def test_list_by_tenant_includes_builtins_by_default(
    session: AsyncSession,
) -> None:
    repo = ModelRepository(session)
    tid = _model_tenant_id()
    builtin_tid = _model_tenant_id()
    regular_id = _mid()
    builtin_id = _mid()
    await repo.insert(_sample_row(id=regular_id, tenant_id=tid))
    await repo.insert(_sample_row(id=builtin_id, tenant_id=builtin_tid, is_builtin=True))

    rows = await repo.list_by_tenant(tenant_id=tid)
    ids = {r.id for r in rows}
    assert regular_id in ids
    assert builtin_id in ids


async def test_list_by_tenant_filters_by_type(session: AsyncSession) -> None:
    repo = ModelRepository(session)
    tid = _model_tenant_id()
    mid_qa = _mid()
    mid_emb = _mid()
    await repo.insert(_sample_row(id=mid_qa, tenant_id=tid, type="KnowledgeQA"))
    await repo.insert(_sample_row(id=mid_emb, tenant_id=tid, type="Embedding"))

    rows = await repo.list_by_tenant(tenant_id=tid, model_type="Embedding", include_builtin=False)
    ids = {r.id for r in rows}
    assert ids == {mid_emb}


async def test_list_by_tenant_filters_by_source(session: AsyncSession) -> None:
    repo = ModelRepository(session)
    tid = _model_tenant_id()
    mid_openai = _mid()
    mid_local = _mid()
    await repo.insert(_sample_row(id=mid_openai, tenant_id=tid, source="openai"))
    await repo.insert(_sample_row(id=mid_local, tenant_id=tid, source="local"))

    rows = await repo.list_by_tenant(tenant_id=tid, source="local", include_builtin=False)
    ids = {r.id for r in rows}
    assert ids == {mid_local}


# ── update_row ───────────────────────────────────────────────────────


async def test_update_row_returns_refreshed_row(session: AsyncSession) -> None:
    repo = ModelRepository(session)
    tid = _model_tenant_id()
    mid = _mid()
    await repo.insert(_sample_row(id=mid, tenant_id=tid))

    row = await repo.find_by_tenant_and_id_or_fail(tenant_id=tid, id=mid)
    updated = row.model_copy(update={"name": "renamed"})
    refreshed = await repo.update_row(updated)

    assert refreshed is not None
    assert refreshed.name == "renamed"


async def test_update_row_returns_none_for_unknown_id(session: AsyncSession) -> None:
    repo = ModelRepository(session)
    row = _sample_row()
    assert await repo.update_row(row) is None


# ── delete_by_tenant_and_id ──────────────────────────────────────────


async def test_delete_by_tenant_and_id_removes_row(session: AsyncSession) -> None:
    repo = ModelRepository(session)
    tid = _model_tenant_id()
    mid = _mid()
    await repo.insert(_sample_row(id=mid, tenant_id=tid))

    affected = await repo.delete_by_tenant_and_id(tenant_id=tid, id=mid)
    assert affected == 1
    assert await repo.find_by_tenant_and_id(tenant_id=tid, id=mid) is None


async def test_delete_by_tenant_and_id_returns_zero_when_absent(
    session: AsyncSession,
) -> None:
    repo = ModelRepository(session)
    affected = await repo.delete_by_tenant_and_id(tenant_id=_model_tenant_id(), id=_mid())
    assert affected == 0


# ── clear_default_by_type ────────────────────────────────────────────


async def test_clear_default_by_type_flips_flag(session: AsyncSession) -> None:
    repo = ModelRepository(session)
    tid = _model_tenant_id()
    await repo.insert(_sample_row(id=_mid(), tenant_id=tid, is_default=True, type="KnowledgeQA"))
    await repo.insert(_sample_row(id=_mid(), tenant_id=tid, is_default=True, type="KnowledgeQA"))
    await repo.insert(_sample_row(id=_mid(), tenant_id=tid, is_default=False, type="KnowledgeQA"))

    affected = await repo.clear_default_by_type(tenant_id=tid, model_type="KnowledgeQA")
    assert affected == 2

    after = await repo.list_by_tenant(tenant_id=tid, include_builtin=False)
    assert all(not r.is_default for r in after)


async def test_clear_default_by_type_respects_exclude_id(
    session: AsyncSession,
) -> None:
    repo = ModelRepository(session)
    tid = _model_tenant_id()
    keep_id = _mid()
    other_id = _mid()
    await repo.insert(_sample_row(id=keep_id, tenant_id=tid, is_default=True))
    await repo.insert(_sample_row(id=other_id, tenant_id=tid, is_default=True))

    affected = await repo.clear_default_by_type(
        tenant_id=tid,
        model_type="KnowledgeQA",
        exclude_id=keep_id,
    )
    assert affected == 1

    keep = await repo.find_by_tenant_and_id_or_fail(tenant_id=tid, id=keep_id)
    other = await repo.find_by_tenant_and_id_or_fail(tenant_id=tid, id=other_id)
    assert keep.is_default is True
    assert other.is_default is False


# ── parameters JSONB round-trip ──────────────────────────────────────


async def test_parameters_round_trip_preserves_jsonb_shape(
    session: AsyncSession,
) -> None:
    repo = ModelRepository(session)
    tid = _model_tenant_id()
    mid = _mid()
    params: dict[str, object] = {
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "embedding_parameters": {"dimension": 1536},
        "extra_config": {"region": "us"},
    }
    await repo.insert(_sample_row(id=mid, tenant_id=tid, parameters=params))

    resolved = await repo.find_by_tenant_and_id_or_fail(tenant_id=tid, id=mid)
    params_back = resolved.parameters
    assert isinstance(params_back, dict)
    assert params_back["provider"] == "openai"
    embedding = params_back["embedding_parameters"]
    extra = params_back["extra_config"]
    assert isinstance(embedding, dict)
    assert isinstance(extra, dict)
    assert embedding["dimension"] == 1536
    assert extra["region"] == "us"


# ── tenant isolation ────────────────────────────────────────────────


async def test_find_by_tenant_and_id_isolated_by_tenant(session: AsyncSession) -> None:
    repo = ModelRepository(session)
    tid_a = _model_tenant_id()
    tid_b = _model_tenant_id()
    mid = _mid()
    await repo.insert(_sample_row(id=mid, tenant_id=tid_a))

    assert (
        await repo.find_by_tenant_and_id(tenant_id=tid_a, id=mid, include_builtin=False) is not None
    )
    assert await repo.find_by_tenant_and_id(tenant_id=tid_b, id=mid, include_builtin=False) is None
