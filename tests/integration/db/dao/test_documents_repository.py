"""Integration tests for ``KnowledgeRepository`` against the real applied schema.

Tests insert unique rows per run; isolation relies on unique ids and
tenant ids. Tests commit explicitly.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.exception import DataError
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.knowledge import Document
from tests.integration.db.dao.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _did() -> str:
    return f"doc-{uuid.uuid4().hex[:12]}"


def _kbid() -> str:
    return f"kb-{uuid.uuid4().hex[:12]}"


def _sample_row(
    *,
    id: str | None = None,
    tenant_id: int | None = None,
    knowledge_base_id: str | None = None,
    title: str = "Q3 budget",
    type: str = "file",
    source: str = "budget-2026.pdf",
    parse_status: str = "completed",
    file_name: str = "budget-2026.pdf",
    **columns: object,
) -> Document:
    return Document.model_validate(
        {
            "id": id or _did(),
            "tenant_id": tenant_id if tenant_id is not None else make_test_tenant_id(),
            "knowledge_base_id": knowledge_base_id or _kbid(),
            "type": type,
            "title": title,
            "description": None,
            "source": source,
            "channel": "web",
            "parse_status": parse_status,
            "pending_subtasks_count": 0,
            "summary_status": "none",
            "enable_status": "enabled",
            "embedding_model_id": None,
            "file_name": file_name,
            "file_type": "pdf",
            "file_size": 1024,
            "file_hash": None,
            "file_path": None,
            "storage_size": 2048,
            "metadata": None,
            "custom_metadata": {},
            "last_faq_import_result": None,
            "created_at": _NOW,
            "updated_at": _NOW,
            "processed_at": None,
            "error_message": None,
            "deleted_at": None,
            **columns,
        }
    )


# ── create / get_by_id ──────────────────────────────────────────────


async def test_create_and_resolve_by_id(session: AsyncSession) -> None:
    repo = KnowledgeRepository(session)
    row = _sample_row()
    await repo.create(row)

    resolved = await repo.get_by_id(tenant_id=row.tenant_id, id=row.id)
    assert resolved is not None
    assert resolved.title == "Q3 budget"
    assert resolved.source == "budget-2026.pdf"
    assert resolved.parse_status == "completed"


async def test_get_by_id_returns_none_for_unknown(session: AsyncSession) -> None:
    repo = KnowledgeRepository(session)
    assert await repo.get_by_id(tenant_id=make_test_tenant_id(), id=_did()) is None


async def test_get_by_id_only_resolves_without_tenant_scope(session: AsyncSession) -> None:
    repo = KnowledgeRepository(session)
    row = _sample_row()
    await repo.create(row)

    resolved = await repo.get_by_id_only(id=row.id)
    assert resolved is not None
    assert resolved.id == row.id


async def test_get_by_id_only_returns_none_for_unknown(session: AsyncSession) -> None:
    repo = KnowledgeRepository(session)
    assert await repo.get_by_id_only(id=_did()) is None


async def test_get_by_id_isolated_by_tenant(session: AsyncSession) -> None:
    repo = KnowledgeRepository(session)
    tid_a = make_test_tenant_id()
    tid_b = make_test_tenant_id()
    row = _sample_row(tenant_id=tid_a)
    await repo.create(row)

    assert await repo.get_by_id(tenant_id=tid_a, id=row.id) is not None
    assert await repo.get_by_id(tenant_id=tid_b, id=row.id) is None


# ── list_by_knowledge_base ──────────────────────────────────────────


async def test_list_by_knowledge_base_orders_newest_first(session: AsyncSession) -> None:
    repo = KnowledgeRepository(session)
    tid = make_test_tenant_id()
    kbid = _kbid()
    older = _sample_row(tenant_id=tid, knowledge_base_id=kbid, created_at=_NOW)
    newer = _sample_row(
        tenant_id=tid,
        knowledge_base_id=kbid,
        created_at=_NOW + timedelta(hours=1),
    )
    await repo.create(older)
    await repo.create(newer)

    rows = await repo.list_by_knowledge_base(tenant_id=tid, knowledge_base_id=kbid)
    assert [r.id for r in rows] == [newer.id, older.id]


async def test_list_by_knowledge_base_scoped_to_kb_and_tenant(session: AsyncSession) -> None:
    repo = KnowledgeRepository(session)
    tid = make_test_tenant_id()
    kbid = _kbid()
    other_kbid = _kbid()
    other_tid = make_test_tenant_id()
    await repo.create(_sample_row(tenant_id=tid, knowledge_base_id=kbid))
    await repo.create(_sample_row(tenant_id=tid, knowledge_base_id=other_kbid))
    await repo.create(_sample_row(tenant_id=other_tid, knowledge_base_id=kbid))

    rows = await repo.list_by_knowledge_base(tenant_id=tid, knowledge_base_id=kbid)
    assert len(rows) == 1


async def test_list_by_knowledge_base_excludes_soft_deleted(
    session: AsyncSession,
) -> None:
    repo = KnowledgeRepository(session)
    tid = make_test_tenant_id()
    kbid = _kbid()
    kept = _sample_row(tenant_id=tid, knowledge_base_id=kbid)
    removed = _sample_row(tenant_id=tid, knowledge_base_id=kbid)
    await repo.create(kept)
    await repo.create(removed)
    await repo.soft_delete(tenant_id=tid, id=removed.id, now=_NOW + timedelta(hours=1))

    rows = await repo.list_by_knowledge_base(tenant_id=tid, knowledge_base_id=kbid)
    assert [r.id for r in rows] == [kept.id]


# ── list_paged_by_knowledge_base ────────────────────────────────────


async def test_list_paged_returns_page_and_total(session: AsyncSession) -> None:
    repo = KnowledgeRepository(session)
    tid = make_test_tenant_id()
    kbid = _kbid()
    rows = [_sample_row(tenant_id=tid, knowledge_base_id=kbid) for _ in range(3)]
    for r in rows:
        await repo.create(r)

    page, total = await repo.list_paged_by_knowledge_base(
        tenant_id=tid,
        knowledge_base_id=kbid,
        limit=2,
        offset=0,
    )
    assert len(page) == 2
    assert total == 3


async def test_list_paged_filters_by_keyword_case_insensitive(
    session: AsyncSession,
) -> None:
    repo = KnowledgeRepository(session)
    tid = make_test_tenant_id()
    kbid = _kbid()
    match = _sample_row(tenant_id=tid, knowledge_base_id=kbid, title="Quarterly REPORT")
    other = _sample_row(tenant_id=tid, knowledge_base_id=kbid, title="Unrelated notes")
    await repo.create(match)
    await repo.create(other)

    page, total = await repo.list_paged_by_knowledge_base(
        tenant_id=tid,
        knowledge_base_id=kbid,
        limit=10,
        offset=0,
        keyword="quarterly",
    )
    assert total == 1
    assert [r.id for r in page] == [match.id]


async def test_list_paged_routes_file_type_manual_to_type_column(
    session: AsyncSession,
) -> None:
    repo = KnowledgeRepository(session)
    tid = make_test_tenant_id()
    kbid = _kbid()
    manual = _sample_row(
        tenant_id=tid,
        knowledge_base_id=kbid,
        type="manual",
        file_type="manual",
        title="Manual note",
    )
    file_row = _sample_row(
        tenant_id=tid,
        knowledge_base_id=kbid,
        type="file",
        file_type="pdf",
        title="PDF upload",
    )
    await repo.create(manual)
    await repo.create(file_row)

    page, total = await repo.list_paged_by_knowledge_base(
        tenant_id=tid,
        knowledge_base_id=kbid,
        limit=10,
        offset=0,
        file_type="manual",
    )
    assert total == 1
    assert [r.id for r in page] == [manual.id]


async def test_list_paged_filters_by_parse_status(session: AsyncSession) -> None:
    repo = KnowledgeRepository(session)
    tid = make_test_tenant_id()
    kbid = _kbid()
    failed = _sample_row(
        tenant_id=tid,
        knowledge_base_id=kbid,
        parse_status="failed",
        title="Broken doc",
    )
    completed = _sample_row(tenant_id=tid, knowledge_base_id=kbid, title="Fine doc")
    await repo.create(failed)
    await repo.create(completed)

    page, total = await repo.list_paged_by_knowledge_base(
        tenant_id=tid,
        knowledge_base_id=kbid,
        limit=10,
        offset=0,
        parse_status="failed",
    )
    assert total == 1
    assert [r.id for r in page] == [failed.id]


async def test_list_paged_hides_deleting_rows_by_default(session: AsyncSession) -> None:
    repo = KnowledgeRepository(session)
    tid = make_test_tenant_id()
    kbid = _kbid()
    live = _sample_row(tenant_id=tid, knowledge_base_id=kbid, title="Live doc")
    deleting = _sample_row(
        tenant_id=tid,
        knowledge_base_id=kbid,
        parse_status="deleting",
        title="Being deleted",
    )
    await repo.create(live)
    await repo.create(deleting)

    page, total = await repo.list_paged_by_knowledge_base(
        tenant_id=tid,
        knowledge_base_id=kbid,
        limit=10,
        offset=0,
    )
    assert total == 1
    assert [r.id for r in page] == [live.id]


async def test_list_paged_filters_by_source_channel(session: AsyncSession) -> None:
    repo = KnowledgeRepository(session)
    tid = make_test_tenant_id()
    kbid = _kbid()
    notion = _sample_row(
        tenant_id=tid,
        knowledge_base_id=kbid,
        channel="notion",
        title="Notion page",
    )
    web = _sample_row(tenant_id=tid, knowledge_base_id=kbid, title="Web doc")
    await repo.create(notion)
    await repo.create(web)

    page, total = await repo.list_paged_by_knowledge_base(
        tenant_id=tid,
        knowledge_base_id=kbid,
        limit=10,
        offset=0,
        source="notion",
    )
    assert total == 1
    assert [r.id for r in page] == [notion.id]


async def test_list_paged_filters_by_updated_range(session: AsyncSession) -> None:
    repo = KnowledgeRepository(session)
    tid = make_test_tenant_id()
    kbid = _kbid()
    early = _sample_row(
        tenant_id=tid,
        knowledge_base_id=kbid,
        updated_at=_NOW,
        title="Early doc",
    )
    late = _sample_row(
        tenant_id=tid,
        knowledge_base_id=kbid,
        updated_at=_NOW + timedelta(days=2),
        title="Late doc",
    )
    await repo.create(early)
    await repo.create(late)

    page, total = await repo.list_paged_by_knowledge_base(
        tenant_id=tid,
        knowledge_base_id=kbid,
        limit=10,
        offset=0,
        updated_from=_NOW + timedelta(days=1),
        updated_to=_NOW + timedelta(days=3),
    )
    assert total == 1
    assert [r.id for r in page] == [late.id]


# ── update / update_columns / update_active_deleting_columns ────────


async def test_update_overwrites_mutable_columns(session: AsyncSession) -> None:
    repo = KnowledgeRepository(session)
    row = _sample_row()
    await repo.create(row)

    updated = row.model_copy(update={"title": "Renamed budget", "error_message": None})
    refreshed = await repo.update(updated)

    assert refreshed.title == "Renamed budget"
    # immutable / owned columns are preserved
    assert refreshed.tenant_id == row.tenant_id
    assert refreshed.knowledge_base_id == row.knowledge_base_id
    assert refreshed.pending_subtasks_count == row.pending_subtasks_count


async def test_update_preserves_pending_subtasks_counter(session: AsyncSession) -> None:
    repo = KnowledgeRepository(session)
    row = _sample_row(pending_subtasks_count=2, parse_status="finalizing")
    await repo.create(row)

    updated = row.model_copy(update={"title": "Retitled"})
    refreshed = await repo.update(updated)

    assert refreshed.title == "Retitled"
    assert refreshed.pending_subtasks_count == 2


async def test_update_raises_for_unknown_id(session: AsyncSession) -> None:
    repo = KnowledgeRepository(session)
    row = _sample_row()
    with pytest.raises(DataError):
        await repo.update(row)


async def test_update_columns_writes_only_named_columns(session: AsyncSession) -> None:
    repo = KnowledgeRepository(session)
    row = _sample_row()
    await repo.create(row)

    refreshed = await repo.update_columns(
        row.id,
        {"parse_status": "failed", "error_message": "dead-lettered"},
    )
    assert refreshed is not None
    assert refreshed.parse_status == "failed"
    assert refreshed.error_message == "dead-lettered"
    assert refreshed.title == row.title


async def test_update_columns_returns_none_for_unknown_id(session: AsyncSession) -> None:
    repo = KnowledgeRepository(session)
    assert await repo.update_columns(_did(), {"title": "x"}) is None


async def test_update_active_deleting_columns_only_when_deleting(
    session: AsyncSession,
) -> None:
    repo = KnowledgeRepository(session)
    deleting_row = _sample_row(parse_status="deleting")
    live_row = _sample_row()
    await repo.create(deleting_row)
    await repo.create(live_row)

    affected = await repo.update_active_deleting_columns(
        deleting_row.id,
        {"parse_status": "failed"},
    )
    assert affected is True

    untouched = await repo.update_active_deleting_columns(
        live_row.id,
        {"parse_status": "failed"},
    )
    assert untouched is False
    resolved = await repo.get_by_id(tenant_id=live_row.tenant_id, id=live_row.id)
    assert resolved is not None
    assert resolved.parse_status == "completed"


# ── soft_delete / soft_delete_list ──────────────────────────────────


async def test_soft_delete_marks_row_and_hides_it(session: AsyncSession) -> None:
    repo = KnowledgeRepository(session)
    tid = make_test_tenant_id()
    row = _sample_row(tenant_id=tid)
    await repo.create(row)

    affected = await repo.soft_delete(tenant_id=tid, id=row.id, now=_NOW + timedelta(hours=1))
    assert affected is True
    assert await repo.get_by_id(tenant_id=tid, id=row.id) is None


async def test_soft_delete_returns_false_when_absent(session: AsyncSession) -> None:
    repo = KnowledgeRepository(session)
    affected = await repo.soft_delete(
        tenant_id=make_test_tenant_id(),
        id=_did(),
        now=_NOW,
    )
    assert affected is False


async def test_soft_delete_list_marks_batch(session: AsyncSession) -> None:
    repo = KnowledgeRepository(session)
    tid = make_test_tenant_id()
    kbid = _kbid()
    ids = []
    for _ in range(3):
        row = _sample_row(tenant_id=tid, knowledge_base_id=kbid)
        await repo.create(row)
        ids.append(row.id)

    affected = await repo.soft_delete_list(tenant_id=tid, ids=ids, now=_NOW + timedelta(hours=1))
    assert affected == 3
    assert await repo.list_by_knowledge_base(tenant_id=tid, knowledge_base_id=kbid) == []


async def test_soft_delete_list_returns_zero_for_empty(session: AsyncSession) -> None:
    repo = KnowledgeRepository(session)
    affected = await repo.soft_delete_list(tenant_id=make_test_tenant_id(), ids=[], now=_NOW)
    assert affected == 0


# ── get_batch ───────────────────────────────────────────────────────


async def test_get_batch_returns_matching_rows(session: AsyncSession) -> None:
    repo = KnowledgeRepository(session)
    tid = make_test_tenant_id()
    kbid = _kbid()
    row_a = _sample_row(tenant_id=tid, knowledge_base_id=kbid)
    row_b = _sample_row(tenant_id=tid, knowledge_base_id=kbid)
    await repo.create(row_a)
    await repo.create(row_b)

    rows = await repo.get_batch(tenant_id=tid, ids=[row_a.id, row_b.id, _did()])
    assert {r.id for r in rows} == {row_a.id, row_b.id}


async def test_get_batch_returns_empty_for_empty_input(session: AsyncSession) -> None:
    repo = KnowledgeRepository(session)
    assert await repo.get_batch(tenant_id=make_test_tenant_id(), ids=[]) == []


# ── counts ──────────────────────────────────────────────────────────


async def test_count_by_knowledge_base(session: AsyncSession) -> None:
    repo = KnowledgeRepository(session)
    tid = make_test_tenant_id()
    kbid = _kbid()
    for _ in range(2):
        await repo.create(_sample_row(tenant_id=tid, knowledge_base_id=kbid))

    assert await repo.count_by_knowledge_base(tenant_id=tid, knowledge_base_id=kbid) == 2
    assert await repo.count_by_knowledge_base(tenant_id=tid, knowledge_base_id=_kbid()) == 0


async def test_count_by_status_filters_and_returns_zero_for_empty(
    session: AsyncSession,
) -> None:
    repo = KnowledgeRepository(session)
    tid = make_test_tenant_id()
    kbid = _kbid()
    for _ in range(2):
        await repo.create(
            _sample_row(tenant_id=tid, knowledge_base_id=kbid, parse_status="processing")
        )
    await repo.create(_sample_row(tenant_id=tid, knowledge_base_id=kbid, parse_status="completed"))

    count = await repo.count_by_status(tid, kbid, ["processing"])
    assert count == 2
    assert await repo.count_by_status(tid, kbid, []) == 0


# ── JSONB round-trip ────────────────────────────────────────────────


async def test_json_columns_round_trip(session: AsyncSession) -> None:
    repo = KnowledgeRepository(session)
    row = _sample_row(
        metadata={"external_id": "wiki/123", "source_node": "wiki"},
        custom_metadata={"owner": "finance"},
        last_faq_import_result={"imported": 5, "skipped": 1},
    )
    await repo.create(row)

    resolved = await repo.get_by_id(tenant_id=row.tenant_id, id=row.id)
    assert resolved is not None
    assert resolved.metadata is not None
    assert resolved.metadata["external_id"] == "wiki/123"
    assert resolved.custom_metadata["owner"] == "finance"
    assert resolved.last_faq_import_result is not None
    assert resolved.last_faq_import_result["imported"] == 5
