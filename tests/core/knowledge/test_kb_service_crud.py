"""Unit tests for `KBService`.

The service is exercised against an ``AsyncMock(spec=KnowledgeBaseRepository)``
with closure-captured in-memory state so the SQL-touching methods keep
working without a database. Count queries resolve from a controllable
map so the list enrichment is deterministic. A real-repository
construction test guards against signature drift between the mock spec
and the concrete ``KnowledgeBaseRepository``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from src.common.exception import DataError, NotFoundError, ValidationError
from src.common.json import JsonObject
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import (
    KNOWLEDGE_BASE_TYPE_DOCUMENT,
    KNOWLEDGE_BASE_TYPE_FAQ,
)
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.models.knowledge_base import KnowledgeBase
from tests.util.service_test import ServiceTest

_NOW = datetime(2026, 1, 1, tzinfo=UTC)

_DEFAULT_STRATEGY = {
    "vector_enabled": True,
    "keyword_enabled": True,
    "wiki_enabled": False,
    "graph_enabled": False,
}


def _make_repo() -> tuple[AsyncMock, dict[str, KnowledgeBase], dict[tuple[int, str], int]]:
    """Knowledge-base-repo mock with closure-captured in-memory storage."""
    repo = AsyncMock(spec=KnowledgeBaseRepository)
    rows: dict[str, KnowledgeBase] = {}
    counts: dict[tuple[int, str], int] = {}

    def _live() -> dict[str, KnowledgeBase]:
        return {i: r for i, r in rows.items() if r.deleted_at is None}

    async def _create(row: KnowledgeBase) -> KnowledgeBase:
        rows[row.id] = row
        return row

    async def _get_by_id_or_none(id: str) -> KnowledgeBase | None:
        return _live().get(id)

    async def _get_by_id_and_tenant(id: str, tenant_id: int) -> KnowledgeBase | None:
        row = _live().get(id)
        if row is not None and row.tenant_id == tenant_id:
            return row
        return None

    async def _get_by_ids(ids: list[str]) -> list[KnowledgeBase]:
        live = _live()
        return [live[i] for i in ids if i in live]

    async def _list_by_tenant(tenant_id: int) -> list[KnowledgeBase]:
        return sorted(
            (r for r in _live().values() if r.tenant_id == tenant_id and not r.is_temporary),
            key=lambda r: r.created_at,
            reverse=True,
        )

    async def _update(row: KnowledgeBase) -> KnowledgeBase:
        if row.id not in rows or rows[row.id].deleted_at is not None:
            raise DataError(code="knowledge_base.update_no_row", message="no live row")
        rows[row.id] = row
        return row

    async def _soft_delete(*, id: str, now: datetime) -> bool:
        row = rows.get(id)
        if row is None or row.deleted_at is not None:
            return False
        rows[id] = row.model_copy(update={"deleted_at": now})
        return True

    async def _count_documents(*, tenant_id: int, knowledge_base_id: str) -> int:
        return counts.get((tenant_id, knowledge_base_id), 0)

    async def _count_chunks(*, tenant_id: int, knowledge_base_id: str) -> int:
        return counts.get((tenant_id, knowledge_base_id), 0)

    async def _count_members(*, tenant_id: int, knowledge_base_id: str) -> int:
        return counts.get((tenant_id, knowledge_base_id), 0)

    repo.create.side_effect = _create
    repo.get_by_id_or_none.side_effect = _get_by_id_or_none
    repo.get_by_id_and_tenant.side_effect = _get_by_id_and_tenant
    repo.get_by_ids.side_effect = _get_by_ids
    repo.list_by_tenant.side_effect = _list_by_tenant
    repo.update.side_effect = _update
    repo.soft_delete.side_effect = _soft_delete
    repo.count_documents.side_effect = _count_documents
    repo.count_chunks.side_effect = _count_chunks
    repo.count_members.side_effect = _count_members
    return repo, rows, counts


def _seed(
    rows: dict[str, KnowledgeBase],
    *,
    name: str = "docs",
    tenant_id: int = 7,
    kb_type: str = KNOWLEDGE_BASE_TYPE_DOCUMENT,
    is_temporary: bool = False,
    description: str | None = None,
    chunking_config: JsonObject | None = None,
    extract_config: JsonObject | None = None,
    faq_config: JsonObject | None = None,
    indexing_strategy: JsonObject | None = None,
    created_at: datetime = _NOW,
) -> KnowledgeBase:
    """Insert a row directly into the closure-captured store."""
    row = KnowledgeBase(
        id=f"kb-{uuid.uuid4().hex[:8]}",
        name=name,
        tenant_id=tenant_id,
        type=kb_type,
        is_temporary=is_temporary,
        description=description,
        chunking_config=chunking_config,
        extract_config=extract_config,
        faq_config=faq_config,
        indexing_strategy=indexing_strategy,
        created_at=created_at,
        updated_at=created_at,
    )
    rows[row.id] = row
    return row


@pytest.fixture
def repo_and_state() -> tuple[AsyncMock, dict[str, KnowledgeBase], dict[tuple[int, str], int]]:
    return _make_repo()


@pytest.fixture
def repo(
    repo_and_state: tuple[AsyncMock, dict[str, KnowledgeBase], dict[tuple[int, str], int]],
) -> AsyncMock:
    return repo_and_state[0]


@pytest.fixture
def rows(
    repo_and_state: tuple[AsyncMock, dict[str, KnowledgeBase], dict[tuple[int, str], int]],
) -> dict[str, KnowledgeBase]:
    return repo_and_state[1]


@pytest.fixture
def counts(
    repo_and_state: tuple[AsyncMock, dict[str, KnowledgeBase], dict[tuple[int, str], int]],
) -> dict[tuple[int, str], int]:
    return repo_and_state[2]


@pytest.fixture
def service(
    repo_and_state: tuple[AsyncMock, dict[str, KnowledgeBase], dict[tuple[int, str], int]],
) -> KBService:
    return KBService(kb_repo=repo_and_state[0])


# ── create_knowledge_base ───────────────────────────────────────────


class TestCreateKnowledgeBase(ServiceTest):
    async def test_stamps_id_tenant_timestamps_and_defaults(
        self, service: KBService, rows: dict[str, KnowledgeBase]
    ) -> None:
        info = await service.create_knowledge_base(tenant_id=7, name="docs")

        assert info.id in rows
        assert info.tenant_id == 7
        assert info.name == "docs"
        assert info.type == KNOWLEDGE_BASE_TYPE_DOCUMENT
        assert info.created_at is not None
        assert info.indexing_strategy == _DEFAULT_STRATEGY
        assert info.faq_config is None

    async def test_trims_the_name(self, service: KBService, rows: dict[str, KnowledgeBase]) -> None:
        info = await service.create_knowledge_base(tenant_id=7, name="  docs  ")
        assert info.name == "docs"
        assert rows[info.id].name == "docs"

    async def test_stamps_creator_id(
        self, service: KBService, rows: dict[str, KnowledgeBase]
    ) -> None:
        info = await service.create_knowledge_base(tenant_id=7, name="docs", creator_id="usr-1")
        assert info.creator_id == "usr-1"

    async def test_faq_type_gets_default_faq_config(
        self, service: KBService, rows: dict[str, KnowledgeBase]
    ) -> None:
        info = await service.create_knowledge_base(
            tenant_id=7, name="faq", kb_type=KNOWLEDGE_BASE_TYPE_FAQ
        )
        assert info.faq_config == {
            "index_mode": "question_answer",
            "question_index_mode": "combined",
        }

    async def test_faq_defaults_merge_with_supplied_config(
        self, service: KBService, rows: dict[str, KnowledgeBase]
    ) -> None:
        info = await service.create_knowledge_base(
            tenant_id=7,
            name="faq",
            kb_type=KNOWLEDGE_BASE_TYPE_FAQ,
            faq_config={"index_mode": "question_only"},
        )
        assert info.faq_config == {
            "index_mode": "question_only",
            "question_index_mode": "combined",
        }

    async def test_document_type_clears_faq_config(
        self, service: KBService, rows: dict[str, KnowledgeBase]
    ) -> None:
        info = await service.create_knowledge_base(
            tenant_id=7,
            name="docs",
            kb_type=KNOWLEDGE_BASE_TYPE_DOCUMENT,
            faq_config={"index_mode": "question_answer"},
        )
        assert info.faq_config is None

    async def test_empty_vector_store_id_normalised_to_none(
        self, service: KBService, rows: dict[str, KnowledgeBase]
    ) -> None:
        info = await service.create_knowledge_base(tenant_id=7, name="docs", vector_store_id="")
        assert info.vector_store_id is None
        assert rows[info.id].vector_store_id is None

    async def test_explicit_vector_store_id_preserved(
        self, service: KBService, rows: dict[str, KnowledgeBase]
    ) -> None:
        info = await service.create_knowledge_base(tenant_id=7, name="docs", vector_store_id="vs-1")
        assert info.vector_store_id == "vs-1"

    async def test_extract_enabled_syncs_graph_enabled(
        self, service: KBService, rows: dict[str, KnowledgeBase]
    ) -> None:
        info = await service.create_knowledge_base(
            tenant_id=7, name="docs", extract_config={"enabled": True}
        )
        assert info.indexing_strategy == {**_DEFAULT_STRATEGY, "graph_enabled": True}

    async def test_storage_config_surfaces_under_wire_name(
        self, service: KBService, rows: dict[str, KnowledgeBase]
    ) -> None:
        info = await service.create_knowledge_base(
            tenant_id=7, name="docs", storage_config={"secret_id": "s"}
        )
        assert info.storage_config == {"secret_id": "s"}
        assert rows[info.id].cos_config == {"secret_id": "s"}

    async def test_rejects_blank_name(self, service: KBService) -> None:
        with pytest.raises(ValidationError) as excinfo:
            await service.create_knowledge_base(tenant_id=7, name="   ")
        assert excinfo.value.code == "knowledge_base.name_required"

    async def test_rejects_unknown_type(self, service: KBService) -> None:
        with pytest.raises(ValidationError) as excinfo:
            await service.create_knowledge_base(tenant_id=7, name="docs", kb_type="graph")
        assert excinfo.value.code == "knowledge_base.type_invalid"

    @pytest.mark.parametrize("tenant_id", [0, -1])
    async def test_rejects_invalid_tenant(self, service: KBService, tenant_id: int) -> None:
        with pytest.raises(ValidationError) as excinfo:
            await service.create_knowledge_base(tenant_id=tenant_id, name="docs")
        assert excinfo.value.code == "knowledge_base.tenant_required"


# ── get_knowledge_base_by_id ────────────────────────────────────────


class TestGetKnowledgeBase(ServiceTest):
    async def test_returns_projection_with_defaults(
        self, service: KBService, rows: dict[str, KnowledgeBase]
    ) -> None:
        stored = _seed(rows, name="docs")
        info = await service.get_knowledge_base_by_id(knowledge_base_id=stored.id)
        assert info.id == stored.id
        assert info.name == "docs"
        assert info.indexing_strategy == _DEFAULT_STRATEGY

    async def test_faq_row_gets_default_config_on_read(
        self, service: KBService, rows: dict[str, KnowledgeBase]
    ) -> None:
        stored = _seed(rows, name="faq", kb_type=KNOWLEDGE_BASE_TYPE_FAQ)
        info = await service.get_knowledge_base_by_id(knowledge_base_id=stored.id)
        assert info.faq_config == {
            "index_mode": "question_answer",
            "question_index_mode": "combined",
        }

    async def test_missing_raises_not_found(self, service: KBService) -> None:
        with pytest.raises(NotFoundError) as excinfo:
            await service.get_knowledge_base_by_id(knowledge_base_id="missing")
        assert excinfo.value.code == "knowledge_base.not_found"

    async def test_empty_id_rejected(self, service: KBService) -> None:
        with pytest.raises(ValidationError) as excinfo:
            await service.get_knowledge_base_by_id(knowledge_base_id="  ")
        assert excinfo.value.code == "knowledge_base.id_required"

    async def test_id_only_variant_matches_unscoped_read(
        self, service: KBService, rows: dict[str, KnowledgeBase]
    ) -> None:
        stored = _seed(rows, name="docs")
        info = await service.get_knowledge_base_by_id_only(knowledge_base_id=stored.id)
        assert info.id == stored.id


class TestGetKnowledgeBaseByTenant(ServiceTest):
    async def test_returns_row_owned_by_tenant(
        self, service: KBService, rows: dict[str, KnowledgeBase]
    ) -> None:
        stored = _seed(rows, name="docs", tenant_id=7)
        info = await service.get_knowledge_base_by_id_and_tenant(
            tenant_id=7, knowledge_base_id=stored.id
        )
        assert info.id == stored.id

    async def test_other_tenant_reads_as_absent(
        self, service: KBService, rows: dict[str, KnowledgeBase]
    ) -> None:
        stored = _seed(rows, name="docs", tenant_id=7)
        with pytest.raises(NotFoundError):
            await service.get_knowledge_base_by_id_and_tenant(
                tenant_id=99, knowledge_base_id=stored.id
            )


class TestGetKnowledgeBasesByIDs(ServiceTest):
    async def test_returns_matching_subset(
        self, service: KBService, rows: dict[str, KnowledgeBase]
    ) -> None:
        a = _seed(rows, name="a")
        b = _seed(rows, name="b")
        _seed(rows, name="c")

        infos = await service.get_knowledge_bases_by_ids(ids=[a.id, b.id, "missing"])

        assert [i.id for i in infos] == [a.id, b.id]

    async def test_empty_input_returns_empty_list(self, service: KBService) -> None:
        assert await service.get_knowledge_bases_by_ids(ids=[]) == []


# ── list_knowledge_bases ────────────────────────────────────────────


class TestListKnowledgeBases(ServiceTest):
    async def test_orders_newest_first_and_excludes_temporary(
        self, service: KBService, rows: dict[str, KnowledgeBase]
    ) -> None:
        tid = 7
        older = _seed(rows, name="older", tenant_id=tid, created_at=_NOW)
        newer = _seed(rows, name="newer", tenant_id=tid, created_at=_NOW + timedelta(days=1))
        _seed(rows, name="tmp", tenant_id=tid, is_temporary=True)
        _seed(rows, name="other", tenant_id=99)

        infos = await service.list_knowledge_bases(tenant_id=tid)

        assert [i.id for i in infos] == [newer.id, older.id]

    async def test_excludes_soft_deleted(
        self, service: KBService, rows: dict[str, KnowledgeBase]
    ) -> None:
        tid = 7
        live = _seed(rows, name="live", tenant_id=tid)
        gone = _seed(rows, name="gone", tenant_id=tid)
        await service.delete_knowledge_base(knowledge_base_id=gone.id)

        infos = await service.list_knowledge_bases(tenant_id=tid)

        assert [i.id for i in infos] == [live.id]

    async def test_document_rows_get_knowledge_count(
        self,
        service: KBService,
        rows: dict[str, KnowledgeBase],
        counts: dict[tuple[int, str], int],
        repo: AsyncMock,
    ) -> None:
        tid = 7
        stored = _seed(rows, name="docs", tenant_id=tid)
        counts[(tid, stored.id)] = 3

        infos = await service.list_knowledge_bases(tenant_id=tid)

        assert infos[0].knowledge_count == 3
        repo.count_documents.assert_awaited_once_with(tenant_id=tid, knowledge_base_id=stored.id)

    async def test_faq_rows_get_chunk_count(
        self,
        service: KBService,
        rows: dict[str, KnowledgeBase],
        counts: dict[tuple[int, str], int],
        repo: AsyncMock,
    ) -> None:
        tid = 7
        stored = _seed(rows, name="faq", tenant_id=tid, kb_type=KNOWLEDGE_BASE_TYPE_FAQ)
        counts[(tid, stored.id)] = 5

        infos = await service.list_knowledge_bases(tenant_id=tid)

        assert infos[0].chunk_count == 5
        repo.count_chunks.assert_awaited_once_with(tenant_id=tid, knowledge_base_id=stored.id)
        repo.count_documents.assert_not_awaited()

    async def test_count_failure_leaves_zero_default(
        self, service: KBService, rows: dict[str, KnowledgeBase], repo: AsyncMock
    ) -> None:
        tid = 7
        _seed(rows, name="docs", tenant_id=tid)
        repo.count_documents.side_effect = RuntimeError("count boom")

        infos = await service.list_knowledge_bases(tenant_id=tid)

        assert infos[0].knowledge_count == 0

    async def test_rejects_invalid_tenant(self, service: KBService) -> None:
        with pytest.raises(ValidationError):
            await service.list_knowledge_bases(tenant_id=0)


# ── update_knowledge_base ───────────────────────────────────────────


class TestUpdateKnowledgeBase(ServiceTest):
    async def test_renames_and_updates_description(
        self, service: KBService, rows: dict[str, KnowledgeBase]
    ) -> None:
        stored = _seed(rows, name="before")

        info = await service.update_knowledge_base(
            knowledge_base_id=stored.id, name="after", description="desc"
        )

        assert info.name == "after"
        assert info.description == "desc"
        assert rows[stored.id].name == "after"

    async def test_applies_chunking_config(
        self, service: KBService, rows: dict[str, KnowledgeBase]
    ) -> None:
        stored = _seed(rows, name="docs")

        info = await service.update_knowledge_base(
            knowledge_base_id=stored.id,
            name="docs",
            config={"chunking_config": {"chunk_size": 512, "chunk_overlap": 64}},
        )

        assert info.chunking_config == {"chunk_size": 512, "chunk_overlap": 64}

    async def test_config_without_chunking_clears_it(
        self, service: KBService, rows: dict[str, KnowledgeBase]
    ) -> None:
        stored = _seed(
            rows, name="docs", chunking_config={"chunk_size": 1000, "chunk_overlap": 200}
        )

        info = await service.update_knowledge_base(
            knowledge_base_id=stored.id,
            name="docs",
            config={"image_processing_config": {"model_id": "m"}},
        )

        assert info.chunking_config is None
        assert info.image_processing_config == {"model_id": "m"}

    async def test_graph_sync_writes_extract_config(
        self, service: KBService, rows: dict[str, KnowledgeBase]
    ) -> None:
        stored = _seed(rows, name="docs")

        info = await service.update_knowledge_base(
            knowledge_base_id=stored.id,
            name="docs",
            config={
                "indexing_strategy": {
                    "vector_enabled": True,
                    "keyword_enabled": True,
                    "wiki_enabled": False,
                    "graph_enabled": True,
                }
            },
        )

        assert info.extract_config == {"enabled": True}
        assert info.indexing_strategy == {
            "vector_enabled": True,
            "keyword_enabled": True,
            "wiki_enabled": False,
            "graph_enabled": True,
        }

    async def test_graph_sync_merges_existing_extract(
        self, service: KBService, rows: dict[str, KnowledgeBase]
    ) -> None:
        stored = _seed(rows, name="docs", extract_config={"enabled": False, "text": "t"})

        info = await service.update_knowledge_base(
            knowledge_base_id=stored.id,
            name="docs",
            config={
                "indexing_strategy": {
                    "vector_enabled": True,
                    "keyword_enabled": True,
                    "wiki_enabled": False,
                    "graph_enabled": True,
                }
            },
        )

        assert info.extract_config == {"enabled": True, "text": "t"}

    async def test_wiki_strategy_creates_wiki_config(
        self, service: KBService, rows: dict[str, KnowledgeBase]
    ) -> None:
        stored = _seed(rows, name="docs")

        info = await service.update_knowledge_base(
            knowledge_base_id=stored.id,
            name="docs",
            config={
                "indexing_strategy": {
                    "vector_enabled": False,
                    "keyword_enabled": False,
                    "wiki_enabled": True,
                    "graph_enabled": False,
                }
            },
        )

        assert info.wiki_config == {}

    async def test_rejects_all_disabled_indexing(
        self, service: KBService, rows: dict[str, KnowledgeBase]
    ) -> None:
        stored = _seed(rows, name="docs")

        with pytest.raises(ValidationError) as excinfo:
            await service.update_knowledge_base(
                knowledge_base_id=stored.id,
                name="docs",
                config={
                    "indexing_strategy": {
                        "vector_enabled": False,
                        "keyword_enabled": False,
                        "wiki_enabled": False,
                        "graph_enabled": False,
                    }
                },
            )

        assert excinfo.value.code == "knowledge_base.indexing_required"

    async def test_missing_row_raises_not_found(self, service: KBService) -> None:
        with pytest.raises(NotFoundError):
            await service.update_knowledge_base(knowledge_base_id="missing", name="x")

    async def test_empty_id_rejected(self, service: KBService) -> None:
        with pytest.raises(ValidationError):
            await service.update_knowledge_base(knowledge_base_id="", name="x")


# ── delete_knowledge_base ───────────────────────────────────────────


class TestDeleteKnowledgeBase(ServiceTest):
    async def test_soft_deletes_and_hides_from_reads(
        self, service: KBService, rows: dict[str, KnowledgeBase]
    ) -> None:
        stored = _seed(rows, name="docs")

        deleted = await service.delete_knowledge_base(knowledge_base_id=stored.id)

        assert deleted is True
        assert rows[stored.id].deleted_at is not None
        with pytest.raises(NotFoundError):
            await service.get_knowledge_base_by_id(knowledge_base_id=stored.id)

    async def test_missing_row_raises_not_found(self, service: KBService) -> None:
        with pytest.raises(NotFoundError):
            await service.delete_knowledge_base(knowledge_base_id="missing")


# ── aggregate counts ────────────────────────────────────────────────


class TestCountMethods(ServiceTest):
    async def test_count_documents_delegates(
        self,
        service: KBService,
        repo: AsyncMock,
        counts: dict[tuple[int, str], int],
    ) -> None:
        counts[(7, "kb-1")] = 4

        assert await service.count_documents(tenant_id=7, knowledge_base_id="kb-1") == 4
        repo.count_documents.assert_awaited_once_with(tenant_id=7, knowledge_base_id="kb-1")

    async def test_count_chunks_delegates(
        self,
        service: KBService,
        repo: AsyncMock,
        counts: dict[tuple[int, str], int],
    ) -> None:
        counts[(7, "kb-1")] = 2

        assert await service.count_chunks(tenant_id=7, knowledge_base_id="kb-1") == 2
        repo.count_chunks.assert_awaited_once_with(tenant_id=7, knowledge_base_id="kb-1")

    async def test_count_members_delegates(
        self,
        service: KBService,
        repo: AsyncMock,
        counts: dict[tuple[int, str], int],
    ) -> None:
        counts[(7, "kb-1")] = 9

        assert await service.count_members(tenant_id=7, knowledge_base_id="kb-1") == 9
        repo.count_members.assert_awaited_once_with(tenant_id=7, knowledge_base_id="kb-1")

    async def test_rejects_invalid_scope(self, service: KBService) -> None:
        with pytest.raises(ValidationError):
            await service.count_documents(tenant_id=0, knowledge_base_id="kb-1")
        with pytest.raises(ValidationError):
            await service.count_chunks(tenant_id=7, knowledge_base_id="")
        with pytest.raises(ValidationError):
            await service.count_members(tenant_id=0, knowledge_base_id="")


# ── signature drift guard ───────────────────────────────────────────


def test_service_accepts_the_real_repository_type() -> None:
    """Construction with the concrete repo must keep type-checking.

    ``KnowledgeBaseRepository`` needs a session only when a query runs,
    so ``None`` is enough to prove the constructor contract holds.
    """
    service = KBService(kb_repo=KnowledgeBaseRepository(None))  # type: ignore[arg-type]

    assert isinstance(service, KBService)


def test_factory_builds_service_with_a_session() -> None:
    """``build_kb_service`` wires a fresh repo onto the shared session."""
    from unittest.mock import MagicMock

    from src.core.knowledge.knowledge_bases.factory import build_kb_service

    service = build_kb_service(MagicMock())  # type: ignore[arg-type]

    assert isinstance(service, KBService)
