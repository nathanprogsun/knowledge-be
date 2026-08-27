"""Integration tests for ``FaqRepository`` against the real applied schema.

Tests insert unique rows per run; isolation relies on unique tenant ids
(from ``make_test_tenant_id``) and unique chunk ids. Tests commit
explicitly. The ``faq`` table is created by the alembic migration chain;
these tests run once the full chain is applied to the test database.
"""

# Chinese test data uses fullwidth punctuation.

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.exception import NotFoundError
from src.db.dao.faq_repository import FaqRepository
from src.db.models.faq import Faq
from tests.integration.db.dao.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _chunk_id() -> str:
    return f"chunk-{uuid.uuid4().hex[:12]}"


def _sample_row(
    *,
    tenant_id: int,
    knowledge_base_id: str = "kb-faq-1",
    knowledge_id: str = "knowledge-faq-1",
    standard_question: str = "如何充值？",
) -> Faq:
    return Faq(
        tenant_id=tenant_id,
        chunk_id=_chunk_id(),
        knowledge_id=knowledge_id,
        knowledge_base_id=knowledge_base_id,
        tag_id=None,
        tag_name=None,
        is_enabled=True,
        is_recommended=False,
        standard_question=standard_question,
        similar_questions=["怎么充值", "充值方法"],
        negative_questions=[],
        answers=["进入设置 -> 账户 -> 充值"],
        answer_strategy="all",
        index_mode=None,
        chunk_type="faq",
        created_at=_NOW,
        updated_at=_NOW,
    )


# ── create / get_by_id ───────────────────────────────────────────────


async def test_create_assigns_id_and_round_trips(session: AsyncSession) -> None:
    repo = FaqRepository(session)
    tid = make_test_tenant_id()
    created = await repo.create(_sample_row(tenant_id=tid))

    assert created.id > 0
    resolved = await repo.get_by_id(tenant_id=tid, id=created.id)
    assert resolved is not None
    assert resolved.standard_question == "如何充值？"
    assert resolved.similar_questions == ["怎么充值", "充值方法"]


async def test_get_by_id_returns_none_for_unknown(session: AsyncSession) -> None:
    repo = FaqRepository(session)
    assert await repo.get_by_id(tenant_id=make_test_tenant_id(), id=999_999) is None


async def test_get_by_id_or_fail_raises_not_found(session: AsyncSession) -> None:
    repo = FaqRepository(session)
    with pytest.raises(NotFoundError) as excinfo:
        await repo.get_by_id_or_fail(tenant_id=make_test_tenant_id(), id=999_999)
    assert excinfo.value.code == "faq.not_found"


async def test_get_by_chunk_id_resolves_entry(session: AsyncSession) -> None:
    repo = FaqRepository(session)
    tid = make_test_tenant_id()
    row = _sample_row(tenant_id=tid)
    created = await repo.create(row)

    resolved = await repo.get_by_chunk_id(tenant_id=tid, chunk_id=created.chunk_id)
    assert resolved is not None
    assert resolved.id == created.id


async def test_get_by_id_isolated_by_tenant(session: AsyncSession) -> None:
    repo = FaqRepository(session)
    tid_a = make_test_tenant_id()
    tid_b = make_test_tenant_id()
    created = await repo.create(_sample_row(tenant_id=tid_a))

    assert await repo.get_by_id(tenant_id=tid_a, id=created.id) is not None
    assert await repo.get_by_id(tenant_id=tid_b, id=created.id) is None


# ── list_by_knowledge_base ───────────────────────────────────────────


async def test_list_by_knowledge_base_returns_page_and_total(session: AsyncSession) -> None:
    repo = FaqRepository(session)
    tid = make_test_tenant_id()
    kb = "kb-faq-list-1"
    other_kb = "kb-faq-list-2"
    await repo.create(_sample_row(tenant_id=tid, knowledge_base_id=kb))
    await repo.create(_sample_row(tenant_id=tid, knowledge_base_id=kb))
    await repo.create(_sample_row(tenant_id=tid, knowledge_base_id=other_kb))
    await repo.create(_sample_row(tenant_id=make_test_tenant_id(), knowledge_base_id=kb))

    rows, total = await repo.list_by_knowledge_base(
        tenant_id=tid,
        knowledge_base_id=kb,
        limit=10,
        offset=0,
    )
    assert total == 2
    assert {r.knowledge_base_id for r in rows} == {kb}


async def test_list_by_knowledge_base_paginates(session: AsyncSession) -> None:
    repo = FaqRepository(session)
    tid = make_test_tenant_id()
    kb = "kb-faq-paged"
    for i in range(3):
        await repo.create(
            _sample_row(
                tenant_id=tid,
                knowledge_base_id=kb,
                standard_question=f"问题{i}？",
            )
        )

    page1, total = await repo.list_by_knowledge_base(
        tenant_id=tid,
        knowledge_base_id=kb,
        limit=2,
        offset=0,
    )
    assert total == 3
    assert len(page1) == 2

    page2, _ = await repo.list_by_knowledge_base(
        tenant_id=tid,
        knowledge_base_id=kb,
        limit=2,
        offset=2,
    )
    assert len(page2) == 1


async def test_list_by_knowledge_base_filters_keyword(session: AsyncSession) -> None:
    repo = FaqRepository(session)
    tid = make_test_tenant_id()
    kb = "kb-faq-keyword"
    await repo.create(
        _sample_row(tenant_id=tid, knowledge_base_id=kb, standard_question="如何退款？")
    )
    await repo.create(
        _sample_row(tenant_id=tid, knowledge_base_id=kb, standard_question="如何充值？")
    )

    rows, total = await repo.list_by_knowledge_base(
        tenant_id=tid,
        knowledge_base_id=kb,
        keyword="退款",
        limit=10,
        offset=0,
    )
    assert total == 1
    assert rows[0].standard_question == "如何退款？"


# ── update ───────────────────────────────────────────────────────────


async def test_update_overwrites_mutable_columns(session: AsyncSession) -> None:
    repo = FaqRepository(session)
    tid = make_test_tenant_id()
    created = await repo.create(_sample_row(tenant_id=tid))

    updated = created.model_copy(
        update={
            "standard_question": "如何变更手机号？",
            "similar_questions": ["换手机号"],
            "answers": ["设置 -> 安全 -> 手机号"],
            "is_recommended": True,
            "updated_at": datetime(2026, 2, 1, tzinfo=UTC),
        }
    )
    refreshed = await repo.update(updated)

    assert refreshed.standard_question == "如何变更手机号？"
    assert refreshed.similar_questions == ["换手机号"]
    assert refreshed.is_recommended is True
    # Immutable scope columns survive the update.
    assert refreshed.chunk_id == created.chunk_id
    assert refreshed.knowledge_base_id == created.knowledge_base_id
    assert refreshed.tenant_id == tid


async def test_update_raises_not_found_for_unknown(session: AsyncSession) -> None:
    repo = FaqRepository(session)
    row = _sample_row(tenant_id=make_test_tenant_id()).model_copy(update={"id": 999_999})
    with pytest.raises(NotFoundError):
        await repo.update(row)


# ── set_enabled ──────────────────────────────────────────────────────


async def test_set_enabled_toggles_and_bumps_updated_at(session: AsyncSession) -> None:
    repo = FaqRepository(session)
    tid = make_test_tenant_id()
    created = await repo.create(_sample_row(tenant_id=tid))

    toggled = await repo.set_enabled(
        tenant_id=tid,
        id=created.id,
        is_enabled=False,
    )
    assert toggled is not None
    assert toggled.is_enabled is False
    assert toggled.updated_at > created.updated_at

    reverted = await repo.set_enabled(
        tenant_id=tid,
        id=created.id,
        is_enabled=True,
    )
    assert reverted is not None
    assert reverted.is_enabled is True


async def test_set_enabled_returns_none_for_unknown(session: AsyncSession) -> None:
    repo = FaqRepository(session)
    toggled = await repo.set_enabled(
        tenant_id=make_test_tenant_id(),
        id=999_999,
        is_enabled=False,
    )
    assert toggled is None


# ── delete_by_ids ────────────────────────────────────────────────────


async def test_delete_by_ids_removes_rows(session: AsyncSession) -> None:
    repo = FaqRepository(session)
    tid = make_test_tenant_id()
    first = await repo.create(_sample_row(tenant_id=tid))
    second = await repo.create(_sample_row(tenant_id=tid))

    affected = await repo.delete_by_ids(tenant_id=tid, ids=[first.id, second.id])
    assert affected == 2
    assert await repo.get_by_id(tenant_id=tid, id=first.id) is None
    assert await repo.get_by_id(tenant_id=tid, id=second.id) is None


async def test_delete_by_ids_scoped_by_tenant(session: AsyncSession) -> None:
    repo = FaqRepository(session)
    tid_a = make_test_tenant_id()
    tid_b = make_test_tenant_id()
    created = await repo.create(_sample_row(tenant_id=tid_a))

    affected = await repo.delete_by_ids(tenant_id=tid_b, ids=[created.id])
    assert affected == 0
    assert await repo.get_by_id(tenant_id=tid_a, id=created.id) is not None


async def test_delete_by_ids_empty_is_noop(session: AsyncSession) -> None:
    repo = FaqRepository(session)
    assert await repo.delete_by_ids(tenant_id=make_test_tenant_id(), ids=[]) == 0


# ── find_duplicate_question ──────────────────────────────────────────


async def test_find_duplicate_question_matches_standard(session: AsyncSession) -> None:
    repo = FaqRepository(session)
    tid = make_test_tenant_id()
    kb = "kb-faq-dup"
    existing = await repo.create(
        _sample_row(tenant_id=tid, knowledge_base_id=kb, standard_question="如何充值？")
    )

    hit = await repo.find_duplicate_question(
        tenant_id=tid,
        knowledge_base_id=kb,
        exclude_id=None,
        questions=["如何充值？"],
    )
    assert hit is not None
    assert hit.id == existing.id


async def test_find_duplicate_question_matches_similar(session: AsyncSession) -> None:
    repo = FaqRepository(session)
    tid = make_test_tenant_id()
    kb = "kb-faq-dup-sim"
    existing = await repo.create(_sample_row(tenant_id=tid, knowledge_base_id=kb))

    hit = await repo.find_duplicate_question(
        tenant_id=tid,
        knowledge_base_id=kb,
        exclude_id=None,
        questions=["充值方法"],
    )
    assert hit is not None
    assert hit.id == existing.id


async def test_find_duplicate_question_honors_exclude_id(session: AsyncSession) -> None:
    repo = FaqRepository(session)
    tid = make_test_tenant_id()
    kb = "kb-faq-dup-excl"
    existing = await repo.create(_sample_row(tenant_id=tid, knowledge_base_id=kb))

    hit = await repo.find_duplicate_question(
        tenant_id=tid,
        knowledge_base_id=kb,
        exclude_id=existing.id,
        questions=["如何充值？"],
    )
    assert hit is None


async def test_find_duplicate_question_returns_none_for_fresh(session: AsyncSession) -> None:
    repo = FaqRepository(session)
    tid = make_test_tenant_id()
    kb = "kb-faq-dup-none"
    await repo.create(_sample_row(tenant_id=tid, knowledge_base_id=kb))

    hit = await repo.find_duplicate_question(
        tenant_id=tid,
        knowledge_base_id=kb,
        exclude_id=None,
        questions=["全新问题"],
    )
    assert hit is None


async def test_find_duplicate_question_empty_questions(session: AsyncSession) -> None:
    repo = FaqRepository(session)
    assert (
        await repo.find_duplicate_question(
            tenant_id=make_test_tenant_id(),
            knowledge_base_id="kb",
            exclude_id=None,
            questions=[],
        )
        is None
    )


# ── JSONB round-trip ─────────────────────────────────────────────────


async def test_question_lists_round_trip_as_jsonb(session: AsyncSession) -> None:
    repo = FaqRepository(session)
    tid = make_test_tenant_id()
    created = await repo.create(_sample_row(tenant_id=tid))

    resolved = await repo.get_by_id(tenant_id=tid, id=created.id)
    assert resolved is not None
    assert resolved.similar_questions == ["怎么充值", "充值方法"]
    assert resolved.negative_questions == []
    assert resolved.answers == ["进入设置 -> 账户 -> 充值"]
