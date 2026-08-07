"""Integration tests for ``TagRepository`` against the real applied schema.

Tests insert unique rows per run; isolation relies on unique tag ids,
knowledge-base ids, and tenant ids. Tests commit explicitly. The ``tags``
and ``document_tags`` tables carry no foreign keys, so no parent rows need
seeding.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.dao.knowledge_tag_repository import TagRepository, escape_like_pattern
from src.db.models.knowledge_tag import KnowledgeTag
from tests.integration.db.dao.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _tid() -> str:
    return f"tag-{uuid.uuid4().hex[:12]}"


def _tag(
    *,
    tenant_id: int | None = None,
    knowledge_base_id: str | None = None,
    name: str | None = None,
    color: str | None = None,
    sort_order: int = 0,
    created_at: datetime = _NOW,
) -> KnowledgeTag:
    return KnowledgeTag(
        id=_tid(),
        tenant_id=tenant_id if tenant_id is not None else make_test_tenant_id(),
        knowledge_base_id=knowledge_base_id or f"kb-{uuid.uuid4().hex[:12]}",
        name=name or f"tag-{uuid.uuid4().hex[:8]}",
        color=color,
        sort_order=sort_order,
        created_at=created_at,
        updated_at=created_at,
    )


async def _store_tag(session: AsyncSession, row: KnowledgeTag) -> KnowledgeTag:
    repo = TagRepository(session)
    stored = await repo.create(row)
    await session.commit()
    return stored


# ── tag CRUD ──────────────────────────────────────────────────────────


async def test_create_assigns_seq_id(session: AsyncSession) -> None:
    stored = await _store_tag(session, _tag())

    assert stored.id != ""
    assert stored.seq_id > 0


async def test_find_by_id(session: AsyncSession) -> None:
    stored = await _store_tag(session, _tag(name="alpha", color="#ff0000"))

    found = await TagRepository(session).get_by_id(stored.tenant_id, stored.id)

    assert found is not None
    assert found.name == "alpha"
    assert found.color == "#ff0000"


async def test_find_by_id_isolated_by_tenant(session: AsyncSession) -> None:
    stored = await _store_tag(session, _tag())
    other_tenant = make_test_tenant_id()

    assert await TagRepository(session).get_by_id(other_tenant, stored.id) is None


async def test_find_by_ids_returns_matching_and_omits_others(session: AsyncSession) -> None:
    repo = TagRepository(session)
    tag_a = await _store_tag(session, _tag())
    tag_b = await _store_tag(session, _tag())

    rows = await repo.get_by_ids(tag_a.tenant_id, [tag_a.id, tag_b.id, _tid()])

    assert {r.id for r in rows} == {tag_a.id, tag_b.id}


async def test_find_by_ids_is_tenant_scoped(session: AsyncSession) -> None:
    repo = TagRepository(session)
    tag_a = await _store_tag(session, _tag())
    tag_b = await _store_tag(session, _tag(tenant_id=make_test_tenant_id()))

    rows = await repo.get_by_ids(tag_a.tenant_id, [tag_a.id, tag_b.id])

    assert [r.id for r in rows] == [tag_a.id]


async def test_find_by_ids_empty_returns_empty(session: AsyncSession) -> None:
    rows = await TagRepository(session).get_by_ids(make_test_tenant_id(), [])

    assert rows == []


async def test_find_by_seq_id(session: AsyncSession) -> None:
    stored = await _store_tag(session, _tag())

    found = await TagRepository(session).get_by_seq_id(stored.tenant_id, stored.seq_id)

    assert found is not None
    assert found.id == stored.id


async def test_find_by_seq_ids(session: AsyncSession) -> None:
    repo = TagRepository(session)
    tag_a = await _store_tag(session, _tag())
    tag_b = await _store_tag(session, _tag())

    rows = await repo.get_by_seq_ids(tag_a.tenant_id, [tag_a.seq_id, tag_b.seq_id])

    assert {r.id for r in rows} == {tag_a.id, tag_b.id}


async def test_find_by_name_within_knowledge_base(session: AsyncSession) -> None:
    repo = TagRepository(session)
    tid = make_test_tenant_id()
    kb_a = f"kb-{uuid.uuid4().hex[:12]}"
    kb_b = f"kb-{uuid.uuid4().hex[:12]}"
    await _store_tag(session, _tag(tenant_id=tid, knowledge_base_id=kb_a, name="shared"))
    await _store_tag(session, _tag(tenant_id=tid, knowledge_base_id=kb_b, name="shared"))

    found = await repo.get_by_name(tid, kb_a, "shared")
    assert found is not None
    assert found.knowledge_base_id == kb_a


async def test_create_duplicate_name_same_kb_raises(session: AsyncSession) -> None:
    repo = TagRepository(session)
    tid = make_test_tenant_id()
    kb = f"kb-{uuid.uuid4().hex[:12]}"
    await _store_tag(session, _tag(tenant_id=tid, knowledge_base_id=kb, name="unique"))

    with pytest.raises(IntegrityError):
        await repo.create(_tag(tenant_id=tid, knowledge_base_id=kb, name="unique"))


async def test_update_overwrites_mutable_columns(session: AsyncSession) -> None:
    repo = TagRepository(session)
    stored = await _store_tag(session, _tag(name="old", color="#000000", sort_order=1))

    renamed = stored.model_copy(
        update={
            "name": "new",
            "color": "#ffffff",
            "sort_order": 5,
            "updated_at": _NOW + timedelta(days=1),
        }
    )
    persisted = await repo.update(renamed)
    await session.commit()

    assert persisted.name == "new"
    assert persisted.color == "#ffffff"
    assert persisted.sort_order == 5
    # Immutable columns survive untouched.
    assert persisted.id == stored.id
    assert persisted.seq_id == stored.seq_id
    assert persisted.tenant_id == stored.tenant_id
    assert persisted.knowledge_base_id == stored.knowledge_base_id


async def test_delete_hard_removes_row(session: AsyncSession) -> None:
    repo = TagRepository(session)
    stored = await _store_tag(session, _tag())

    removed = await repo.delete(tenant_id=stored.tenant_id, id=stored.id)
    await session.commit()

    assert removed is True
    assert await repo.get_by_id(stored.tenant_id, stored.id) is None


async def test_delete_reports_false_for_absent_row(session: AsyncSession) -> None:
    repo = TagRepository(session)
    tid = make_test_tenant_id()

    removed = await repo.delete(tenant_id=tid, id=_tid())

    assert removed is False


# ── list_by_kb ────────────────────────────────────────────────────────


async def test_list_by_kb_paginates_and_counts(session: AsyncSession) -> None:
    repo = TagRepository(session)
    tid = make_test_tenant_id()
    kb = f"kb-{uuid.uuid4().hex[:12]}"
    for index in range(3):
        await _store_tag(session, _tag(tenant_id=tid, knowledge_base_id=kb, sort_order=index))

    page, total = await repo.list_by_kb(tenant_id=tid, knowledge_base_id=kb, page=1, page_size=2)
    _, total_all = await repo.list_by_kb(tenant_id=tid, knowledge_base_id=kb, page=2, page_size=2)

    assert total == 3
    assert len(page) == 2
    assert total_all == 3


async def test_list_by_kb_scopes_to_knowledge_base(session: AsyncSession) -> None:
    repo = TagRepository(session)
    tid = make_test_tenant_id()
    kb_a = f"kb-{uuid.uuid4().hex[:12]}"
    kb_b = f"kb-{uuid.uuid4().hex[:12]}"
    await _store_tag(session, _tag(tenant_id=tid, knowledge_base_id=kb_a, name="in-a"))
    await _store_tag(session, _tag(tenant_id=tid, knowledge_base_id=kb_b, name="in-b"))

    page, total = await repo.list_by_kb(tenant_id=tid, knowledge_base_id=kb_a)

    assert total == 1
    assert [r.name for r in page] == ["in-a"]


async def test_list_by_kb_filters_keyword(session: AsyncSession) -> None:
    repo = TagRepository(session)
    tid = make_test_tenant_id()
    kb = f"kb-{uuid.uuid4().hex[:12]}"
    await _store_tag(session, _tag(tenant_id=tid, knowledge_base_id=kb, name="infrastructure"))
    await _store_tag(session, _tag(tenant_id=tid, knowledge_base_id=kb, name="api"))

    page, total = await repo.list_by_kb(tenant_id=tid, knowledge_base_id=kb, keyword="infra")

    assert total == 1
    assert [r.name for r in page] == ["infrastructure"]


async def test_list_by_kb_treats_wildcards_literally(session: AsyncSession) -> None:
    repo = TagRepository(session)
    tid = make_test_tenant_id()
    kb = f"kb-{uuid.uuid4().hex[:12]}"
    literal_name = f"a%b-{uuid.uuid4().hex[:6]}"
    await _store_tag(session, _tag(tenant_id=tid, knowledge_base_id=kb, name=literal_name))
    await _store_tag(session, _tag(tenant_id=tid, knowledge_base_id=kb, name="axxb"))

    page, total = await repo.list_by_kb(tenant_id=tid, knowledge_base_id=kb, keyword=literal_name)

    assert total == 1
    assert [r.name for r in page] == [literal_name]


async def test_list_by_kb_orders_by_sort_then_recency(session: AsyncSession) -> None:
    repo = TagRepository(session)
    tid = make_test_tenant_id()
    kb = f"kb-{uuid.uuid4().hex[:12]}"
    late = await _store_tag(
        session,
        _tag(
            tenant_id=tid, knowledge_base_id=kb, sort_order=1, created_at=_NOW + timedelta(days=1)
        ),
    )
    early = await _store_tag(
        session,
        _tag(tenant_id=tid, knowledge_base_id=kb, sort_order=1, created_at=_NOW),
    )
    pinned = await _store_tag(
        session,
        _tag(tenant_id=tid, knowledge_base_id=kb, sort_order=0, created_at=_NOW),
    )

    page, _ = await repo.list_by_kb(tenant_id=tid, knowledge_base_id=kb)

    assert [r.id for r in page] == [pinned.id, late.id, early.id]


# ── document-tag bind / unbind ────────────────────────────────────────


async def test_set_knowledge_tags_replaces_bindings(session: AsyncSession) -> None:
    repo = TagRepository(session)
    tid = make_test_tenant_id()
    kb = f"kb-{uuid.uuid4().hex[:12]}"
    tag_a = await _store_tag(session, _tag(tenant_id=tid, knowledge_base_id=kb))
    tag_b = await _store_tag(session, _tag(tenant_id=tid, knowledge_base_id=kb))
    tag_c = await _store_tag(session, _tag(tenant_id=tid, knowledge_base_id=kb))
    knowledge_id = f"kn-{uuid.uuid4().hex[:12]}"

    await repo.set_knowledge_tags(knowledge_id=knowledge_id, tag_ids=[tag_a.id, tag_b.id])
    await session.commit()
    await repo.set_knowledge_tags(knowledge_id=knowledge_id, tag_ids=[tag_b.id, tag_c.id])
    await session.commit()

    tags = await repo.get_knowledge_tags([knowledge_id])
    assert {t.id for t in tags[knowledge_id]} == {tag_b.id, tag_c.id}


async def test_set_knowledge_tags_skips_empty_and_duplicate_ids(session: AsyncSession) -> None:
    repo = TagRepository(session)
    tid = make_test_tenant_id()
    kb = f"kb-{uuid.uuid4().hex[:12]}"
    tag_a = await _store_tag(session, _tag(tenant_id=tid, knowledge_base_id=kb))
    knowledge_id = f"kn-{uuid.uuid4().hex[:12]}"

    await repo.set_knowledge_tags(
        knowledge_id=knowledge_id,
        tag_ids=[tag_a.id, tag_a.id, ""],
    )
    await session.commit()

    tags = await repo.get_knowledge_tags([knowledge_id])
    assert [t.id for t in tags.get(knowledge_id, [])] == [tag_a.id]


async def test_set_knowledge_tags_clears_all_when_empty(session: AsyncSession) -> None:
    repo = TagRepository(session)
    tid = make_test_tenant_id()
    kb = f"kb-{uuid.uuid4().hex[:12]}"
    tag_a = await _store_tag(session, _tag(tenant_id=tid, knowledge_base_id=kb))
    knowledge_id = f"kn-{uuid.uuid4().hex[:12]}"
    await repo.set_knowledge_tags(knowledge_id=knowledge_id, tag_ids=[tag_a.id])
    await session.commit()

    await repo.set_knowledge_tags(knowledge_id=knowledge_id, tag_ids=[])
    await session.commit()

    tags = await repo.get_knowledge_tags([knowledge_id])
    assert tags == {}


async def test_get_knowledge_tags_groups_by_document(session: AsyncSession) -> None:
    repo = TagRepository(session)
    tid = make_test_tenant_id()
    kb = f"kb-{uuid.uuid4().hex[:12]}"
    tag_a = await _store_tag(session, _tag(tenant_id=tid, knowledge_base_id=kb))
    tag_b = await _store_tag(session, _tag(tenant_id=tid, knowledge_base_id=kb))
    doc_x = f"kn-{uuid.uuid4().hex[:12]}"
    doc_y = f"kn-{uuid.uuid4().hex[:12]}"
    await repo.set_knowledge_tags(knowledge_id=doc_x, tag_ids=[tag_a.id, tag_b.id])
    await repo.set_knowledge_tags(knowledge_id=doc_y, tag_ids=[tag_a.id])
    await session.commit()

    tags = await repo.get_knowledge_tags([doc_x, doc_y])

    assert {t.id for t in tags[doc_x]} == {tag_a.id, tag_b.id}
    assert [t.id for t in tags[doc_y]] == [tag_a.id]


async def test_get_knowledge_tags_empty_input(session: AsyncSession) -> None:
    tags = await TagRepository(session).get_knowledge_tags([])

    assert tags == {}


async def test_delete_knowledge_tag_relations(session: AsyncSession) -> None:
    repo = TagRepository(session)
    tid = make_test_tenant_id()
    kb = f"kb-{uuid.uuid4().hex[:12]}"
    tag_a = await _store_tag(session, _tag(tenant_id=tid, knowledge_base_id=kb))
    knowledge_id = f"kn-{uuid.uuid4().hex[:12]}"
    await repo.set_knowledge_tags(knowledge_id=knowledge_id, tag_ids=[tag_a.id])
    await session.commit()

    removed = await repo.delete_knowledge_tag_relations(knowledge_id)
    await session.commit()

    assert removed == 1
    assert await repo.get_knowledge_tags([knowledge_id]) == {}


# ── pure helpers ──────────────────────────────────────────────────────


def test_escape_like_pattern_neutralises_wildcards() -> None:
    assert escape_like_pattern(r"a%b_c\d") == r"a\%b\_c\\d"


def test_escape_like_pattern_leaves_plain_text_unchanged() -> None:
    assert escape_like_pattern("alpha") == "alpha"
