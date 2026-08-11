"""Unit + integration tests for ``KnowledgeService``.

Unit tests drive the service with a stateful repository mock (closure
captured storage, the same pattern used across the core service tests):
they cover validation, error classification, and the happy paths.

Integration tests run against the real applied schema (``documents``
table) using the tenant-id factory from the integration conftest. They
require a reachable database — run with ``DATABASE_URL_OVERRIDE``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from random import randint
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from faker import Faker
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.common.exception import NotFoundError, ValidationError
from src.common.pagination import PaginationResponse
from src.core.contracts.knowledge import Knowledge
from src.core.knowledge.documents.factory import build_knowledge_service
from src.core.knowledge.documents.service.knowledge_service import KnowledgeService
from src.core.knowledge.documents.types import (
    CHANNEL_API,
    CHANNEL_WEB,
    PARSE_STATUS_PENDING,
    DocumentListFilter,
)
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.knowledge import Document
from src.settings import get_settings, reset_settings_cache
from tests.integration.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_FAKER_SEED_MAX = 100_000_000


@pytest.fixture(autouse=True)
def faker_seed() -> None:
    """Re-seed Faker per test for varied-but-reproducible generation."""
    Faker.seed(randint(1, _FAKER_SEED_MAX))


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
    custom_metadata: dict[str, object] | None = None,
    **columns: object,
) -> Document:
    """Build a persisted-shape document row for seeding mocks / DB."""
    return Document.model_validate(
        {
            "id": id or _did(),
            "tenant_id": tenant_id if tenant_id is not None else make_test_tenant_id(),
            "knowledge_base_id": knowledge_base_id or _kbid(),
            "type": type,
            "title": title,
            "description": None,
            "source": source,
            "channel": CHANNEL_WEB,
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
            "custom_metadata": custom_metadata if custom_metadata is not None else {},
            "last_faq_import_result": None,
            "created_at": _NOW,
            "updated_at": _NOW,
            "processed_at": None,
            "error_message": None,
            "deleted_at": None,
            **columns,
        }
    )


# ── Repository mock (stateful via side_effect closures) ────────────────


def _make_repo() -> tuple[AsyncMock, dict[str, Document]]:
    """Repository mock with closure-captured storage."""
    repo = AsyncMock(spec=KnowledgeRepository)
    rows: dict[str, Document] = {}

    async def _create(row: Document) -> Document:
        rows[row.id] = row
        return row

    async def _get_by_id(tenant_id: int, id: str) -> Document | None:
        row = rows.get(id)
        if row is not None and row.tenant_id == tenant_id and row.deleted_at is None:
            return row
        return None

    async def _get_by_id_only(id: str) -> Document | None:
        row = rows.get(id)
        if row is not None and row.deleted_at is None:
            return row
        return None

    async def _get_batch(tenant_id: int, ids: list[str]) -> list[Document]:
        out: list[Document] = []
        for id in ids:
            row = rows.get(id)
            if row is not None and row.tenant_id == tenant_id and row.deleted_at is None:
                out.append(row)
        return out

    async def _list_by_knowledge_base(
        tenant_id: int,
        knowledge_base_id: str,
    ) -> list[Document]:
        out = [
            row
            for row in rows.values()
            if (
                row.tenant_id == tenant_id
                and row.knowledge_base_id == knowledge_base_id
                and row.deleted_at is None
            )
        ]
        out.sort(key=lambda r: r.created_at, reverse=True)
        return out

    async def _list_paged_by_knowledge_base(
        tenant_id: int,
        knowledge_base_id: str,
        *,
        limit: int,
        offset: int,
        keyword: str | None = None,
        file_type: str | None = None,
        parse_status: str | None = None,
        source: str | None = None,
        updated_from: datetime | None = None,
        updated_to: datetime | None = None,
    ) -> tuple[list[Document], int]:
        candidates = [
            row
            for row in rows.values()
            if (
                row.tenant_id == tenant_id
                and row.knowledge_base_id == knowledge_base_id
                and row.deleted_at is None
            )
        ]
        if keyword:
            kw = keyword.lower()
            candidates = [
                row
                for row in candidates
                if kw in (row.file_name or "").lower() or kw in row.title.lower()
            ]
        if file_type:
            if file_type in ("manual", "url"):
                candidates = [row for row in candidates if row.type == file_type]
            else:
                candidates = [row for row in candidates if row.file_type == file_type]
        if parse_status:
            candidates = [row for row in candidates if row.parse_status == parse_status]
        if source:
            if source in ("manual", "url"):
                candidates = [row for row in candidates if row.type == source]
            else:
                candidates = [row for row in candidates if row.channel == source]
        if updated_from is not None:
            candidates = [row for row in candidates if row.updated_at >= updated_from]
        if updated_to is not None:
            candidates = [row for row in candidates if row.updated_at <= updated_to]
        candidates.sort(key=lambda r: r.created_at, reverse=True)
        total = len(candidates)
        return candidates[offset : offset + limit], total

    async def _count_by_knowledge_base(tenant_id: int, knowledge_base_id: str) -> int:
        return sum(
            1
            for row in rows.values()
            if (
                row.tenant_id == tenant_id
                and row.knowledge_base_id == knowledge_base_id
                and row.deleted_at is None
            )
        )

    async def _count_by_status(
        tenant_id: int,
        knowledge_base_id: str,
        parse_statuses: list[str],
    ) -> int:
        if not parse_statuses:
            return 0
        statuses = set(parse_statuses)
        return sum(
            1
            for row in rows.values()
            if (
                row.tenant_id == tenant_id
                and row.knowledge_base_id == knowledge_base_id
                and row.parse_status in statuses
                and row.deleted_at is None
            )
        )

    async def _update(row: Document) -> Document:
        rows[row.id] = row
        return row

    async def _soft_delete(*, tenant_id: int, id: str, now: datetime) -> bool:
        row = rows.get(id)
        if row is None or row.tenant_id != tenant_id or row.deleted_at is not None:
            return False
        rows[id] = row.model_copy(update={"deleted_at": now, "updated_at": now})
        return True

    async def _soft_delete_list(*, tenant_id: int, ids: list[str], now: datetime) -> int:
        removed = 0
        for id in ids:
            row = rows.get(id)
            if row is not None and row.tenant_id == tenant_id and row.deleted_at is None:
                rows[id] = row.model_copy(update={"deleted_at": now, "updated_at": now})
                removed += 1
        return removed

    repo.create.side_effect = _create
    repo.get_by_id.side_effect = _get_by_id
    repo.get_by_id_only.side_effect = _get_by_id_only
    repo.get_batch.side_effect = _get_batch
    repo.list_by_knowledge_base.side_effect = _list_by_knowledge_base
    repo.list_paged_by_knowledge_base.side_effect = _list_paged_by_knowledge_base
    repo.count_by_knowledge_base.side_effect = _count_by_knowledge_base
    repo.count_by_status.side_effect = _count_by_status
    repo.update.side_effect = _update
    repo.soft_delete.side_effect = _soft_delete
    repo.soft_delete_list.side_effect = _soft_delete_list
    return repo, rows


def _service(repo: AsyncMock) -> KnowledgeService:
    return KnowledgeService(knowledge_repo=repo)


# ── Factory ────────────────────────────────────────────────────────────


def test_factory_builds_request_scoped_service() -> None:
    session = AsyncMock(spec=AsyncSession)
    service = build_knowledge_service(session)
    assert isinstance(service, KnowledgeService)
    assert isinstance(service._knowledge_repo, KnowledgeRepository)
    assert service._knowledge_repo._session is session


# ── Create ─────────────────────────────────────────────────────────────


async def test_create_document_applies_service_defaults() -> None:
    repo, _rows = _make_repo()
    created = await _service(repo).create_document(
        tenant_id=1,
        knowledge_base_id="kb-1",
        type="manual",
        title="Manual notes",
        source="manual",
    )
    assert isinstance(created, Knowledge)
    assert created.id
    assert created.tenant_id == 1
    assert created.knowledge_base_id == "kb-1"
    assert created.type == "manual"
    assert created.title == "Manual notes"
    assert created.source == "manual"
    assert created.channel == CHANNEL_WEB
    assert created.parse_status == PARSE_STATUS_PENDING
    assert created.enable_status == "enabled"
    repo.create.assert_awaited_once()


async def test_create_document_round_trips_custom_metadata_and_file_columns() -> None:
    repo, rows = _make_repo()
    created = await _service(repo).create_document(
        tenant_id=1,
        knowledge_base_id="kb-1",
        type="file",
        title="budget",
        source="budget-2026.pdf",
        file_name="budget-2026.pdf",
        file_type="pdf",
        file_size=2048,
        custom_metadata={"owner": "finance"},
    )
    stored = rows[created.id]
    assert stored.custom_metadata == {"owner": "finance"}
    assert stored.file_name == "budget-2026.pdf"
    assert stored.file_size == 2048


async def test_create_document_rejects_invalid_scope() -> None:
    service = _service(_make_repo()[0])
    with pytest.raises(ValidationError) as exc_info:
        await service.create_document(
            tenant_id=0,
            knowledge_base_id="kb-1",
            type="manual",
            title="t",
            source="manual",
        )
    assert exc_info.value.code == "knowledge.tenant_required"

    with pytest.raises(ValidationError) as exc_info:
        await service.create_document(
            tenant_id=1,
            knowledge_base_id="",
            type="manual",
            title="t",
            source="manual",
        )
    assert exc_info.value.code == "knowledge.kb_required"


async def test_create_document_rejects_blank_required_fields() -> None:
    service = _service(_make_repo()[0])
    with pytest.raises(ValidationError) as exc_info:
        await service.create_document(
            tenant_id=1,
            knowledge_base_id="kb-1",
            type="",
            title="t",
            source="manual",
        )
    assert exc_info.value.code == "knowledge.type_required"

    with pytest.raises(ValidationError) as exc_info:
        await service.create_document(
            tenant_id=1,
            knowledge_base_id="kb-1",
            type="manual",
            title="  ",
            source="manual",
        )
    assert exc_info.value.code == "knowledge.title_required"

    with pytest.raises(ValidationError) as exc_info:
        await service.create_document(
            tenant_id=1,
            knowledge_base_id="kb-1",
            type="manual",
            title="t",
            source="",
        )
    assert exc_info.value.code == "knowledge.source_required"


# ── Read ───────────────────────────────────────────────────────────────


async def test_get_document_returns_projected_row() -> None:
    repo, rows = _make_repo()
    row = _sample_row()
    rows[row.id] = row
    fetched = await _service(repo).get_document(tenant_id=row.tenant_id, id=row.id)
    assert fetched.id == row.id
    assert fetched.title == "Q3 budget"
    assert fetched.parse_status == "completed"


async def test_get_document_raises_for_missing_row() -> None:
    repo, rows = _make_repo()
    row = _sample_row()
    rows[row.id] = row
    service = _service(repo)
    with pytest.raises(NotFoundError) as exc_info:
        await service.get_document(tenant_id=row.tenant_id, id="doc-missing")
    assert exc_info.value.code == "knowledge.not_found"
    # Cross-tenant id is also "not found" — the row exists but is out of scope.
    with pytest.raises(NotFoundError):
        await service.get_document(tenant_id=row.tenant_id + 1, id=row.id)


async def test_get_document_ignores_soft_deleted_row() -> None:
    repo, rows = _make_repo()
    row = _sample_row(deleted_at=datetime.now(UTC))
    rows[row.id] = row
    with pytest.raises(NotFoundError):
        await _service(repo).get_document(tenant_id=row.tenant_id, id=row.id)


async def test_get_document_validates_scope() -> None:
    service = _service(_make_repo()[0])
    with pytest.raises(ValidationError) as exc_info:
        await service.get_document(tenant_id=-1, id="doc-1")
    assert exc_info.value.code == "knowledge.tenant_required"
    with pytest.raises(ValidationError) as exc_info:
        await service.get_document(tenant_id=1, id=" ")
    assert exc_info.value.code == "knowledge.id_required"


async def test_get_document_by_id_only_resolves_without_tenant_scope() -> None:
    repo, rows = _make_repo()
    row = _sample_row()
    rows[row.id] = row
    resolved = await _service(repo).get_document_by_id_only(id=row.id)
    assert resolved is not None
    assert resolved.id == row.id
    assert await _service(repo).get_document_by_id_only(id="doc-missing") is None


async def test_get_documents_returns_matching_and_drops_missing() -> None:
    repo, rows = _make_repo()
    tenant_id = make_test_tenant_id()
    row_a = _sample_row(tenant_id=tenant_id)
    row_b = _sample_row(tenant_id=tenant_id, title="Q4 plan")
    rows[row_a.id] = row_a
    rows[row_b.id] = row_b
    docs = await _service(repo).get_documents(
        tenant_id=row_a.tenant_id,
        ids=[row_a.id, row_b.id, "doc-missing"],
    )
    assert {doc.id for doc in docs} == {row_a.id, row_b.id}


async def test_get_documents_empty_ids_skips_database() -> None:
    repo, _rows = _make_repo()
    docs = await _service(repo).get_documents(tenant_id=1, ids=[])
    assert docs == []
    repo.get_batch.assert_not_awaited()


async def test_list_documents_orders_newest_first() -> None:
    repo, rows = _make_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    older = _sample_row(tenant_id=tenant_id, knowledge_base_id=kb_id, created_at=_NOW - timedelta(days=1))
    newer = _sample_row(tenant_id=tenant_id, knowledge_base_id=kb_id, created_at=_NOW)
    rows[older.id] = older
    rows[newer.id] = newer
    docs = await _service(repo).list_documents(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
    )
    assert [doc.id for doc in docs] == [newer.id, older.id]


async def test_list_documents_returns_empty_for_unknown_kb() -> None:
    repo, rows = _make_repo()
    row = _sample_row()
    rows[row.id] = row
    docs = await _service(repo).list_documents(
        tenant_id=row.tenant_id,
        knowledge_base_id="kb-other",
    )
    assert docs == []


# ── Paged list ─────────────────────────────────────────────────────────


async def test_list_documents_paged_applies_filters() -> None:
    repo, rows = _make_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    completed = _sample_row(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        title="Budget draft",
        file_name="budget-2026.pdf",
        file_type="pdf",
        parse_status="completed",
        channel=CHANNEL_WEB,
    )
    failed = _sample_row(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        title="Report",
        file_name="report-2026.pdf",
        file_type="pdf",
        parse_status="failed",
        channel=CHANNEL_API,
    )
    rows[completed.id] = completed
    rows[failed.id] = failed

    result = await _service(repo).list_documents_paged(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        page=1,
        page_size=10,
        list_filter=DocumentListFilter(keyword="budget"),
    )
    assert isinstance(result, PaginationResponse)
    assert result.total == 1
    assert result.page == 1
    assert result.page_size == 10
    assert [doc.id for doc in result.data] == [completed.id]

    status_result = await _service(repo).list_documents_paged(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        list_filter=DocumentListFilter(parse_status="failed"),
    )
    assert status_result.total == 1
    assert status_result.data[0].id == failed.id


async def test_list_documents_paged_slices_page() -> None:
    repo, rows = _make_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    for i in range(3):
        row = _sample_row(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            title=f"doc-{i}",
            created_at=_NOW - timedelta(days=1) + timedelta(days=i),
        )
        rows[row.id] = row
    result = await _service(repo).list_documents_paged(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        page=2,
        page_size=2,
    )
    assert result.total == 3
    assert [doc.title for doc in result.data] == ["doc-0"]


async def test_list_documents_paged_validates_pagination() -> None:
    repo, rows = _make_repo()
    row = _sample_row()
    rows[row.id] = row
    service = _service(repo)
    with pytest.raises(ValidationError) as exc_info:
        await service.list_documents_paged(
            tenant_id=row.tenant_id,
            knowledge_base_id=row.knowledge_base_id,
            page=0,
        )
    assert exc_info.value.code == "knowledge.invalid_page"
    with pytest.raises(ValidationError) as exc_info:
        await service.list_documents_paged(
            tenant_id=row.tenant_id,
            knowledge_base_id=row.knowledge_base_id,
            page_size=101,
        )
    assert exc_info.value.code == "knowledge.invalid_page_size"


async def test_list_documents_paged_rejects_tag_filter() -> None:
    repo, rows = _make_repo()
    row = _sample_row()
    rows[row.id] = row
    with pytest.raises(ValidationError) as exc_info:
        await _service(repo).list_documents_paged(
            tenant_id=row.tenant_id,
            knowledge_base_id=row.knowledge_base_id,
            list_filter=DocumentListFilter(tag_ids=["tag-1"]),
        )
    assert exc_info.value.code == "knowledge.tag_filter_unsupported"


# ── Counts ─────────────────────────────────────────────────────────────


async def test_count_documents_and_by_status() -> None:
    repo, rows = _make_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    rows[_sample_row(tenant_id=tenant_id, knowledge_base_id=kb_id).id] = _sample_row(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
    )
    rows[_sample_row(tenant_id=tenant_id, knowledge_base_id=kb_id, parse_status="processing").id] = (
        _sample_row(tenant_id=tenant_id, knowledge_base_id=kb_id, parse_status="processing")
    )
    service = _service(repo)
    assert await service.count_documents(tenant_id=tenant_id, knowledge_base_id=kb_id) == 2
    assert (
        await service.count_documents_by_status(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            parse_statuses=["completed"],
        )
        == 1
    )
    assert (
        await service.count_documents_by_status(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            parse_statuses=["completed", "processing"],
        )
        == 2
    )


async def test_count_by_status_empty_statuses_counts_zero() -> None:
    repo, rows = _make_repo()
    row = _sample_row()
    rows[row.id] = row
    count = await _service(repo).count_documents_by_status(
        tenant_id=row.tenant_id,
        knowledge_base_id=row.knowledge_base_id,
        parse_statuses=[],
    )
    assert count == 0


# ── Update ─────────────────────────────────────────────────────────────


async def test_update_document_applies_mutable_fields() -> None:
    repo, rows = _make_repo()
    row = _sample_row(custom_metadata={"owner": "finance"})
    rows[row.id] = row
    updated = await _service(repo).update_document(
        tenant_id=row.tenant_id,
        id=row.id,
        title="Q3 budget v2",
        description="revised",
        custom_metadata={"owner": "finance", "status": "review"},
    )
    assert updated.title == "Q3 budget v2"
    assert updated.description == "revised"
    stored = rows[row.id]
    assert stored.custom_metadata == {"owner": "finance", "status": "review"}
    assert stored.updated_at >= row.updated_at


async def test_update_document_skips_empty_fields_and_keeps_others() -> None:
    repo, rows = _make_repo()
    row = _sample_row(title="Q3 budget", description="original")
    rows[row.id] = row
    updated = await _service(repo).update_document(
        tenant_id=row.tenant_id,
        id=row.id,
        title="",
        description="new description",
    )
    assert updated.title == "Q3 budget"
    assert updated.description == "new description"


async def test_update_document_no_change_skips_write() -> None:
    repo, rows = _make_repo()
    row = _sample_row()
    rows[row.id] = row
    result = await _service(repo).update_document(tenant_id=row.tenant_id, id=row.id)
    assert result.id == row.id
    assert result.title == "Q3 budget"
    repo.update.assert_not_awaited()


async def test_update_document_raises_for_missing_row() -> None:
    repo, _rows = _make_repo()
    with pytest.raises(NotFoundError) as exc_info:
        await _service(repo).update_document(tenant_id=1, id="doc-missing", title="x")
    assert exc_info.value.code == "knowledge.not_found"


async def test_update_document_validates_custom_metadata() -> None:
    repo, rows = _make_repo()
    row = _sample_row()
    rows[row.id] = row
    service = _service(repo)

    too_many = {f"key-{i}": "v" for i in range(21)}
    with pytest.raises(ValidationError) as exc_info:
        await service.update_document(tenant_id=row.tenant_id, id=row.id, custom_metadata=too_many)
    assert exc_info.value.code == "knowledge.custom_metadata_too_many_fields"

    with pytest.raises(ValidationError) as exc_info:
        await service.update_document(
            tenant_id=row.tenant_id,
            id=row.id,
            custom_metadata={"blank key": "v", "  ": "v"},
        )
    assert exc_info.value.code == "knowledge.invalid_custom_metadata_field"

    with pytest.raises(ValidationError) as exc_info:
        await service.update_document(
            tenant_id=row.tenant_id,
            id=row.id,
            custom_metadata={"long": "x" * 1001},
        )
    assert exc_info.value.code == "knowledge.invalid_custom_metadata_field"

    with pytest.raises(ValidationError) as exc_info:
        await service.update_document(
            tenant_id=row.tenant_id,
            id=row.id,
            custom_metadata={"nested": {"a": 1}},
        )
    assert exc_info.value.code == "knowledge.invalid_custom_metadata_value"

    with pytest.raises(ValidationError) as exc_info:
        await service.update_document(
            tenant_id=row.tenant_id,
            id=row.id,
            custom_metadata={"items": [1, 2]},
        )
    assert exc_info.value.code == "knowledge.invalid_custom_metadata_value"


async def test_update_document_accepts_scalar_custom_metadata_kinds() -> None:
    repo, rows = _make_repo()
    row = _sample_row()
    rows[row.id] = row
    updated = await _service(repo).update_document(
        tenant_id=row.tenant_id,
        id=row.id,
        custom_metadata={"s": "text", "n": 3, "f": 1.5, "b": True, "z": None},
    )
    assert updated.id == row.id
    assert rows[row.id].custom_metadata == {"s": "text", "n": 3, "f": 1.5, "b": True, "z": None}


# ── Delete ─────────────────────────────────────────────────────────────


async def test_delete_document_soft_deletes_and_is_idempotent() -> None:
    repo, rows = _make_repo()
    row = _sample_row()
    rows[row.id] = row
    service = _service(repo)
    assert await service.delete_document(tenant_id=row.tenant_id, id=row.id) is True
    assert rows[row.id].deleted_at is not None
    # Second delete reports False; the row is gone from reads.
    assert await service.delete_document(tenant_id=row.tenant_id, id=row.id) is False
    with pytest.raises(NotFoundError):
        await service.get_document(tenant_id=row.tenant_id, id=row.id)


async def test_delete_document_reports_false_for_unknown_and_cross_tenant() -> None:
    repo, rows = _make_repo()
    row = _sample_row()
    rows[row.id] = row
    service = _service(repo)
    assert await service.delete_document(tenant_id=row.tenant_id, id="doc-missing") is False
    assert await service.delete_document(tenant_id=row.tenant_id + 1, id=row.id) is False


async def test_delete_documents_batch() -> None:
    repo, rows = _make_repo()
    tenant_id = make_test_tenant_id()
    row_a = _sample_row(tenant_id=tenant_id)
    row_b = _sample_row(tenant_id=tenant_id)
    rows[row_a.id] = row_a
    rows[row_b.id] = row_b
    service = _service(repo)
    removed = await service.delete_documents(
        tenant_id=row_a.tenant_id,
        ids=[row_a.id, row_b.id, "", "  "],
    )
    assert removed == 2
    assert await service.delete_documents(tenant_id=row_a.tenant_id, ids=[]) == 0


# ── Integration (real applied schema) ──────────────────────────────────


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


async def test_integration_create_get_round_trip(session: AsyncSession) -> None:
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    service = KnowledgeService(knowledge_repo=KnowledgeRepository(session))
    created = await service.create_document(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        type="manual",
        title="Integration notes",
        source="manual",
        custom_metadata={"project": "q3"},
    )
    assert created.id
    assert created.title == "Integration notes"
    assert created.parse_status == PARSE_STATUS_PENDING

    fetched = await service.get_document(tenant_id=tenant_id, id=created.id)
    assert fetched.id == created.id
    assert fetched.title == "Integration notes"

    row = await KnowledgeRepository(session).get_by_id(tenant_id=tenant_id, id=created.id)
    assert row is not None
    assert row.custom_metadata == {"project": "q3"}
    assert row.channel == CHANNEL_WEB

    with pytest.raises(NotFoundError):
        await service.get_document(tenant_id=tenant_id, id="doc-missing")


async def test_integration_update_persists_custom_metadata(session: AsyncSession) -> None:
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    service = KnowledgeService(knowledge_repo=KnowledgeRepository(session))
    created = await service.create_document(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        type="manual",
        title="Before",
        source="manual",
    )
    updated = await service.update_document(
        tenant_id=tenant_id,
        id=created.id,
        title="After",
        custom_metadata={"owner": "finance"},
    )
    assert updated.title == "After"
    row = await KnowledgeRepository(session).get_by_id(tenant_id=tenant_id, id=created.id)
    assert row is not None
    assert row.title == "After"
    assert row.custom_metadata == {"owner": "finance"}


async def test_integration_list_count_status_delete(session: AsyncSession) -> None:
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    service = KnowledgeService(knowledge_repo=KnowledgeRepository(session))
    doc_a = await service.create_document(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        type="file",
        title="a",
        source="a.pdf",
        parse_status="completed",
    )
    doc_b = await service.create_document(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        type="file",
        title="b",
        source="b.pdf",
        parse_status="failed",
    )

    assert await service.count_documents(tenant_id=tenant_id, knowledge_base_id=kb_id) == 2
    assert (
        await service.count_documents_by_status(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            parse_statuses=["completed"],
        )
        == 1
    )
    listed = await service.list_documents(tenant_id=tenant_id, knowledge_base_id=kb_id)
    assert {doc.id for doc in listed} == {doc_a.id, doc_b.id}
    paged = await service.list_documents_paged(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        list_filter=DocumentListFilter(parse_status="completed"),
    )
    assert paged.total == 1
    assert paged.data[0].id == doc_a.id

    assert await service.delete_document(tenant_id=tenant_id, id=doc_b.id) is True
    assert await service.count_documents(tenant_id=tenant_id, knowledge_base_id=kb_id) == 1
    assert await service.delete_document(tenant_id=tenant_id, id=doc_b.id) is False
