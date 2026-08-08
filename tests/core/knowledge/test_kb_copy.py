"""Unit + integration tests for the knowledge-base copy / duplicate modules.

Unit tests drive ``copy_kb`` and ``duplicate_kb`` against a real
``KBService`` backed by a stateful repository mock (closure-captured
storage), covering validation, error classification and the happy paths
including the clone defenses.

Integration tests run against the real applied schema (``knowledge_bases``
and ``chunks`` tables) using the tenant-id factory from the integration
conftest. They require a reachable database — run with
``DATABASE_URL_OVERRIDE``. Chunk-bearing rows use a 32-bit-safe tenant id
because the ``chunks.tenant_id`` column is an ``INTEGER``.
"""

from __future__ import annotations

import secrets
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from random import randint
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from faker import Faker
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.common.exception import NotFoundError, ValidationError
from src.common.json import JsonObject
from src.core.knowledge.knowledge_bases.copy import copy_kb
from src.core.knowledge.knowledge_bases.duplicate import duplicate_kb
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import (
    KNOWLEDGE_BASE_TYPE_DOCUMENT,
    KNOWLEDGE_BASE_TYPE_FAQ,
)
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.dao.tenants_repository import TenantRepository
from src.db.models.chunk import Chunk
from src.db.models.knowledge_base import KnowledgeBase
from src.db.models.tenants.tenants import Tenant
from src.settings import get_settings, reset_settings_cache
from tests.integration.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_FAKER_SEED_MAX = 100_000_000

# ``chunks.tenant_id`` is a 32-bit INTEGER, so integration rows touching
# it need a 32-bit-safe unique id (the workspace ids do not fit). Values
# are random so leftover rows from earlier test runs never collide.
_used_int32_tenant_ids: set[int] = set()

_DEFAULT_STRATEGY = {
    "vector_enabled": True,
    "keyword_enabled": True,
    "wiki_enabled": False,
    "graph_enabled": False,
}


@pytest.fixture(autouse=True)
def faker_seed() -> None:
    """Re-seed Faker per test for varied-but-reproducible generation."""
    Faker.seed(randint(1, _FAKER_SEED_MAX))


def _int32_tenant_id() -> int:
    """Return a unique 32-bit tenant id for chunk-bearing integration rows."""
    while True:
        candidate = secrets.randbelow(2**31 - 1) + 1
        if candidate not in _used_int32_tenant_ids:
            _used_int32_tenant_ids.add(candidate)
            return candidate


# ── Repository mock (stateful via side_effect closures) ──────────────────


def _make_repo() -> tuple[AsyncMock, dict[str, KnowledgeBase]]:
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
            (
                r
                for r in _live().values()
                if r.tenant_id == tenant_id and not r.is_temporary
            ),
            key=lambda r: r.created_at,
            reverse=True,
        )

    async def _update(row: KnowledgeBase) -> KnowledgeBase:
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
    return repo, rows


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
    embedding_model_id: str = "",
    summary_model_id: str = "",
    storage_provider_config: JsonObject | None = None,
    storage_backend_id: str | None = None,
    storage_config: JsonObject | None = None,
    vector_store_id: str | None = None,
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
        embedding_model_id=embedding_model_id,
        summary_model_id=summary_model_id,
        storage_provider_config=storage_provider_config,
        storage_backend_id=storage_backend_id,
        cos_config=storage_config,
        vector_store_id=vector_store_id,
        created_at=created_at,
        updated_at=created_at,
    )
    rows[row.id] = row
    return row


@pytest.fixture
def repo_and_state() -> tuple[AsyncMock, dict[str, KnowledgeBase]]:
    return _make_repo()


@pytest.fixture
def repo(
    repo_and_state: tuple[AsyncMock, dict[str, KnowledgeBase]],
) -> AsyncMock:
    return repo_and_state[0]


@pytest.fixture
def rows(
    repo_and_state: tuple[AsyncMock, dict[str, KnowledgeBase]],
) -> dict[str, KnowledgeBase]:
    return repo_and_state[1]


@pytest.fixture
def service(
    repo_and_state: tuple[AsyncMock, dict[str, KnowledgeBase]],
) -> KBService:
    return KBService(kb_repo=repo_and_state[0])


def _patch_storage_tenant(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tenant: SimpleNamespace | None = None,
) -> None:
    """Stub the workspace-storage read used by the clone-into-target defense.

    ``None`` means "no workspace configuration" (the defense is skipped);
    a ``SimpleNamespace`` with ``default_storage_backend_id`` /
    ``storage_engine_config`` exercises the comparison.
    """
    fake_repo = MagicMock()
    fake_repo.find_by_primary_key = AsyncMock(return_value=tenant)
    monkeypatch.setattr(
        "src.core.knowledge.knowledge_bases.copy.TenantRepository",
        MagicMock(return_value=fake_repo),
    )


# ── copy_kb: clone into a newly created knowledge base ─────────────────


class TestCopyIntoNewTarget:
    async def test_creates_shallow_copy_with_matching_settings(
        self,
        service: KBService,
        rows: dict[str, KnowledgeBase],
    ) -> None:
        source = _seed(
            rows,
            name="docs",
            chunking_config={"chunk_size": 512},
            embedding_model_id="emb-1",
            storage_config={"provider": "minio"},
        )

        _, target = await copy_kb(
            service=service,
            session=AsyncMock(),
            tenant_id=7,
            source_kb_id=source.id,
        )

        assert target.id != source.id
        assert target.id in rows
        assert target.tenant_id == 7
        assert target.name == source.name
        assert target.type == KNOWLEDGE_BASE_TYPE_DOCUMENT
        assert target.chunking_config == {"chunk_size": 512}
        assert target.embedding_model_id == "emb-1"
        assert target.storage_config == {"provider": "minio"}
        # The source projection is never mutated.
        assert rows[source.id].name == "docs"

    async def test_stamps_creator_id(
        self,
        service: KBService,
        rows: dict[str, KnowledgeBase],
    ) -> None:
        source = _seed(rows, name="docs")

        _, target = await copy_kb(
            service=service,
            session=AsyncMock(),
            tenant_id=7,
            source_kb_id=source.id,
            creator_id="usr-1",
        )

        assert target.creator_id == "usr-1"
        assert rows[target.id].creator_id == "usr-1"

    async def test_preserves_vector_store_binding(
        self,
        service: KBService,
        rows: dict[str, KnowledgeBase],
    ) -> None:
        source = _seed(rows, name="docs", vector_store_id="vs-1")

        _, target = await copy_kb(
            service=service,
            session=AsyncMock(),
            tenant_id=7,
            source_kb_id=source.id,
        )

        assert target.vector_store_id == "vs-1"

    async def test_faq_source_copies_faq_config(
        self,
        service: KBService,
        rows: dict[str, KnowledgeBase],
    ) -> None:
        source = _seed(
            rows,
            name="faq",
            kb_type=KNOWLEDGE_BASE_TYPE_FAQ,
            faq_config={"index_mode": "question_only"},
        )

        _, target = await copy_kb(
            service=service,
            session=AsyncMock(),
            tenant_id=7,
            source_kb_id=source.id,
        )

        assert target.faq_config == {
            "index_mode": "question_only",
            "question_index_mode": "combined",
        }

    async def test_missing_source_raises_not_found(
        self,
        service: KBService,
        rows: dict[str, KnowledgeBase],
    ) -> None:
        _seed(rows, name="docs")

        with pytest.raises(NotFoundError) as excinfo:
            await copy_kb(
                service=service,
                session=AsyncMock(),
                tenant_id=7,
                source_kb_id="missing",
            )

        assert excinfo.value.code == "knowledge_base.not_found"

    async def test_cross_tenant_source_reads_as_absent(
        self,
        service: KBService,
        rows: dict[str, KnowledgeBase],
    ) -> None:
        source = _seed(rows, name="docs", tenant_id=7)

        with pytest.raises(NotFoundError):
            await copy_kb(
                service=service,
                session=AsyncMock(),
                tenant_id=99,
                source_kb_id=source.id,
            )

    async def test_empty_source_id_rejected(
        self,
        service: KBService,
        rows: dict[str, KnowledgeBase],
    ) -> None:
        with pytest.raises(ValidationError) as excinfo:
            await copy_kb(
                service=service,
                session=AsyncMock(),
                tenant_id=7,
                source_kb_id="   ",
            )

        assert excinfo.value.code == "knowledge_base.id_required"


# ── copy_kb: clone into an existing target ─────────────────────────────


class TestCopyIntoExistingTarget:
    async def test_returns_source_and_target_without_new_row(
        self,
        service: KBService,
        rows: dict[str, KnowledgeBase],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = _seed(rows, name="docs")
        target = _seed(rows, name="tgt")
        _patch_storage_tenant(monkeypatch, tenant=None)

        src_out, tgt_out = await copy_kb(
            service=service,
            session=AsyncMock(),
            tenant_id=7,
            source_kb_id=source.id,
            target_kb_id=target.id,
        )

        assert src_out.id == source.id
        assert tgt_out.id == target.id
        assert set(rows) == {source.id, target.id}

    async def test_rejects_embedding_model_mismatch(
        self,
        service: KBService,
        rows: dict[str, KnowledgeBase],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = _seed(rows, name="docs", embedding_model_id="emb-a")
        target = _seed(rows, name="tgt", embedding_model_id="emb-b")
        _patch_storage_tenant(monkeypatch, tenant=None)

        with pytest.raises(ValidationError) as excinfo:
            await copy_kb(
                service=service,
                session=AsyncMock(),
                tenant_id=7,
                source_kb_id=source.id,
                target_kb_id=target.id,
            )

        assert excinfo.value.code == "knowledge_base.copy_embedding_mismatch"

    async def test_matching_embedding_models_pass(
        self,
        service: KBService,
        rows: dict[str, KnowledgeBase],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = _seed(rows, name="docs", embedding_model_id="emb-a")
        target = _seed(rows, name="tgt", embedding_model_id="emb-a")
        _patch_storage_tenant(monkeypatch, tenant=None)

        _, tgt_out = await copy_kb(
            service=service,
            session=AsyncMock(),
            tenant_id=7,
            source_kb_id=source.id,
            target_kb_id=target.id,
        )

        assert tgt_out.id == target.id

    async def test_rejects_vector_store_mismatch(
        self,
        service: KBService,
        rows: dict[str, KnowledgeBase],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = _seed(rows, name="docs", vector_store_id="vs-a")
        target = _seed(rows, name="tgt", vector_store_id="vs-b")
        _patch_storage_tenant(monkeypatch, tenant=None)

        with pytest.raises(ValidationError) as excinfo:
            await copy_kb(
                service=service,
                session=AsyncMock(),
                tenant_id=7,
                source_kb_id=source.id,
                target_kb_id=target.id,
            )

        assert excinfo.value.code == "knowledge_base.copy_vector_store_mismatch"

    async def test_matching_vector_store_passes(
        self,
        service: KBService,
        rows: dict[str, KnowledgeBase],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = _seed(rows, name="docs", vector_store_id="vs-a")
        target = _seed(rows, name="tgt", vector_store_id="vs-a")
        _patch_storage_tenant(monkeypatch, tenant=None)

        _, tgt_out = await copy_kb(
            service=service,
            session=AsyncMock(),
            tenant_id=7,
            source_kb_id=source.id,
            target_kb_id=target.id,
        )

        assert tgt_out.id == target.id

    async def test_empty_vector_store_binding_equals_none(
        self,
        service: KBService,
        rows: dict[str, KnowledgeBase],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = _seed(rows, name="docs", vector_store_id="")
        target = _seed(rows, name="tgt", vector_store_id=None)
        _patch_storage_tenant(monkeypatch, tenant=None)

        _, tgt_out = await copy_kb(
            service=service,
            session=AsyncMock(),
            tenant_id=7,
            source_kb_id=source.id,
            target_kb_id=target.id,
        )

        assert tgt_out.id == target.id

    async def test_rejects_storage_backend_mismatch(
        self,
        service: KBService,
        rows: dict[str, KnowledgeBase],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = _seed(rows, name="docs", storage_backend_id="sb-a")
        target = _seed(rows, name="tgt", storage_backend_id="sb-b")
        _patch_storage_tenant(
            monkeypatch,
            tenant=SimpleNamespace(default_storage_backend_id=None, storage_engine_config=None),
        )

        with pytest.raises(ValidationError) as excinfo:
            await copy_kb(
                service=service,
                session=AsyncMock(),
                tenant_id=7,
                source_kb_id=source.id,
                target_kb_id=target.id,
            )

        assert excinfo.value.code == "knowledge_base.copy_storage_mismatch"

    async def test_storage_defense_skipped_without_tenant_config(
        self,
        service: KBService,
        rows: dict[str, KnowledgeBase],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = _seed(rows, name="docs", storage_backend_id="sb-a")
        target = _seed(rows, name="tgt", storage_backend_id="sb-b")
        _patch_storage_tenant(monkeypatch, tenant=None)

        _, tgt_out = await copy_kb(
            service=service,
            session=AsyncMock(),
            tenant_id=7,
            source_kb_id=source.id,
            target_kb_id=target.id,
        )

        assert tgt_out.id == target.id

    async def test_storage_backend_compare_falls_back_to_provider(
        self,
        service: KBService,
        rows: dict[str, KnowledgeBase],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = _seed(rows, name="docs", storage_provider_config={"provider": "minio"})
        target = _seed(rows, name="tgt", storage_provider_config={"provider": "cos"})
        _patch_storage_tenant(
            monkeypatch,
            tenant=SimpleNamespace(default_storage_backend_id=None, storage_engine_config=None),
        )

        with pytest.raises(ValidationError) as excinfo:
            await copy_kb(
                service=service,
                session=AsyncMock(),
                tenant_id=7,
                source_kb_id=source.id,
                target_kb_id=target.id,
            )

        assert excinfo.value.code == "knowledge_base.copy_storage_mismatch"

    async def test_cross_tenant_target_reads_as_absent(
        self,
        service: KBService,
        rows: dict[str, KnowledgeBase],
    ) -> None:
        source = _seed(rows, name="docs", tenant_id=7)
        _seed(rows, name="tgt", tenant_id=99)

        with pytest.raises(NotFoundError):
            await copy_kb(
                service=service,
                session=AsyncMock(),
                tenant_id=7,
                source_kb_id=source.id,
                target_kb_id="tgt",
            )


# ── duplicate_kb ────────────────────────────────────────────────────────


class TestDuplicateKnowledgeBase:
    async def test_duplicates_settings_only_with_copy_suffix(
        self,
        service: KBService,
        rows: dict[str, KnowledgeBase],
    ) -> None:
        source = _seed(
            rows,
            name="docs",
            chunking_config={"chunk_size": 512},
            embedding_model_id="emb-1",
            extract_config={"enabled": True},
        )

        target = await duplicate_kb(
            service=service,
            session=AsyncMock(),
            tenant_id=7,
            source_kb_id=source.id,
        )

        assert target.id != source.id
        assert target.id in rows
        assert target.tenant_id == 7
        assert target.name == "docs Copy"
        assert target.chunking_config == {"chunk_size": 512}
        assert target.embedding_model_id == "emb-1"
        assert target.extract_config == {"enabled": True}
        # Runtime state is reset.
        assert target.is_temporary is False
        assert target.knowledge_count == 0
        assert target.chunk_count == 0

    async def test_stamps_creator_id(
        self,
        service: KBService,
        rows: dict[str, KnowledgeBase],
    ) -> None:
        source = _seed(rows, name="docs")

        target = await duplicate_kb(
            service=service,
            session=AsyncMock(),
            tenant_id=7,
            source_kb_id=source.id,
            creator_id="usr-2",
        )

        assert target.creator_id == "usr-2"

    async def test_name_dedup_appends_number(
        self,
        service: KBService,
        rows: dict[str, KnowledgeBase],
    ) -> None:
        source = _seed(rows, name="docs")
        _seed(rows, name="docs Copy")
        _seed(rows, name="docs Copy 2")

        target = await duplicate_kb(
            service=service,
            session=AsyncMock(),
            tenant_id=7,
            source_kb_id=source.id,
        )

        assert target.name == "docs Copy 3"

    @pytest.mark.parametrize(
        ("locale", "suffix"),
        [
            ("en", "docs Copy"),
            ("en-US", "docs Copy"),
            ("zh-CN", "docs 副本"),
            ("ko-KR", "docs 사본"),
            ("ru-RU", "docs копия"),
        ],
    )
    async def test_locale_specific_suffix(
        self,
        service: KBService,
        rows: dict[str, KnowledgeBase],
        locale: str,
        suffix: str,
    ) -> None:
        source = _seed(rows, name="docs")

        target = await duplicate_kb(
            service=service,
            session=AsyncMock(),
            tenant_id=7,
            source_kb_id=source.id,
            locale=locale,
        )

        assert target.name == suffix

    async def test_blank_source_name_uses_default_base(
        self,
        service: KBService,
        rows: dict[str, KnowledgeBase],
    ) -> None:
        source = _seed(rows, name="  ")

        target = await duplicate_kb(
            service=service,
            session=AsyncMock(),
            tenant_id=7,
            source_kb_id=source.id,
            locale="zh-CN",
        )

        assert target.name == "知识库 副本"

    async def test_listing_failure_falls_back_to_bare_suffix(
        self,
        service: KBService,
        rows: dict[str, KnowledgeBase],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = _seed(rows, name="docs")
        monkeypatch.setattr(
            service,
            "list_knowledge_bases",
            AsyncMock(side_effect=RuntimeError("listing boom")),
        )

        target = await duplicate_kb(
            service=service,
            session=AsyncMock(),
            tenant_id=7,
            source_kb_id=source.id,
        )

        assert target.name == "docs Copy"

    async def test_empty_source_id_rejected(
        self,
        service: KBService,
        rows: dict[str, KnowledgeBase],
    ) -> None:
        with pytest.raises(ValidationError) as excinfo:
            await duplicate_kb(
                service=service,
                session=AsyncMock(),
                tenant_id=7,
                source_kb_id="   ",
            )

        assert excinfo.value.code == "knowledge_base.id_required"

    async def test_missing_source_raises_not_found(
        self,
        service: KBService,
        rows: dict[str, KnowledgeBase],
    ) -> None:
        _seed(rows, name="docs")

        with pytest.raises(NotFoundError) as excinfo:
            await duplicate_kb(
                service=service,
                session=AsyncMock(),
                tenant_id=7,
                source_kb_id="missing",
            )

        assert excinfo.value.code == "knowledge_base.not_found"

    async def test_cross_tenant_source_reads_as_absent(
        self,
        service: KBService,
        rows: dict[str, KnowledgeBase],
    ) -> None:
        source = _seed(rows, name="docs", tenant_id=7)

        with pytest.raises(NotFoundError):
            await duplicate_kb(
                service=service,
                session=AsyncMock(),
                tenant_id=99,
                source_kb_id=source.id,
            )


# ── Integration (real applied schema) ───────────────────────────────────


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


def _service(session: AsyncSession) -> KBService:
    return KBService(kb_repo=KnowledgeBaseRepository(session))


async def test_integration_copy_into_new_target_creates_row(session: AsyncSession) -> None:
    service = _service(session)
    tid = make_test_tenant_id()
    source = await service.create_knowledge_base(
        tenant_id=tid,
        name="docs",
        chunking_config={"chunk_size": 512},
        embedding_model_id="emb-1",
    )
    await session.commit()

    _, target = await copy_kb(
        service=service,
        session=session,
        tenant_id=tid,
        source_kb_id=source.id,
    )
    await session.commit()

    assert target.id != source.id
    assert target.tenant_id == tid
    assert target.name == "docs"
    assert target.chunking_config == {"chunk_size": 512}
    assert target.embedding_model_id == "emb-1"
    fetched = await service.get_knowledge_base_by_id_and_tenant(
        tenant_id=tid, knowledge_base_id=target.id
    )
    assert fetched.id == target.id


async def test_integration_copy_into_existing_target_returns_pair(session: AsyncSession) -> None:
    service = _service(session)
    tid = make_test_tenant_id()
    source = await service.create_knowledge_base(tenant_id=tid, name="docs")
    target = await service.create_knowledge_base(tenant_id=tid, name="tgt")
    await session.commit()

    src_out, tgt_out = await copy_kb(
        service=service,
        session=session,
        tenant_id=tid,
        source_kb_id=source.id,
        target_kb_id=target.id,
    )

    # The clone is settings-level only: the very rows created above are
    # returned unchanged and nothing new is inserted.
    assert src_out.id == source.id
    assert tgt_out.id == target.id


async def test_integration_copy_rejects_embedding_mismatch(session: AsyncSession) -> None:
    service = _service(session)
    tid = make_test_tenant_id()
    source = await service.create_knowledge_base(
        tenant_id=tid, name="docs", embedding_model_id="emb-a"
    )
    target = await service.create_knowledge_base(
        tenant_id=tid, name="tgt", embedding_model_id="emb-b"
    )
    await session.commit()

    with pytest.raises(ValidationError) as excinfo:
        await copy_kb(
            service=service,
            session=session,
            tenant_id=tid,
            source_kb_id=source.id,
            target_kb_id=target.id,
        )

    assert excinfo.value.code == "knowledge_base.copy_embedding_mismatch"


async def test_integration_copy_rejects_vector_store_mismatch(session: AsyncSession) -> None:
    service = _service(session)
    tid = make_test_tenant_id()
    source = await service.create_knowledge_base(
        tenant_id=tid, name="docs", vector_store_id="vs-a"
    )
    target = await service.create_knowledge_base(
        tenant_id=tid, name="tgt", vector_store_id="vs-b"
    )
    await session.commit()

    with pytest.raises(ValidationError) as excinfo:
        await copy_kb(
            service=service,
            session=session,
            tenant_id=tid,
            source_kb_id=source.id,
            target_kb_id=target.id,
        )

    assert excinfo.value.code == "knowledge_base.copy_vector_store_mismatch"


async def test_integration_copy_rejects_cross_tenant_source(session: AsyncSession) -> None:
    service = _service(session)
    other_tid = make_test_tenant_id()
    source = await service.create_knowledge_base(tenant_id=other_tid, name="docs")
    await session.commit()

    with pytest.raises(NotFoundError):
        await copy_kb(
            service=service,
            session=session,
            tenant_id=make_test_tenant_id(),
            source_kb_id=source.id,
        )


async def test_integration_copy_rejects_storage_backend_mismatch(session: AsyncSession) -> None:
    now = datetime.now(UTC)
    tenant = await TenantRepository(session).insert(
        Tenant(
            name="ws",
            description=None,
            status="active",
            business="",
            retriever_engines={"engines": []},
            created_at=now,
            updated_at=now,
        )
    )
    await session.commit()
    tid = tenant.id
    service = _service(session)
    source = await service.create_knowledge_base(
        tenant_id=tid, name="docs", storage_backend_id="sb-a"
    )
    target = await service.create_knowledge_base(
        tenant_id=tid, name="tgt", storage_backend_id="sb-b"
    )
    await session.commit()

    with pytest.raises(ValidationError) as excinfo:
        await copy_kb(
            service=service,
            session=session,
            tenant_id=tid,
            source_kb_id=source.id,
            target_kb_id=target.id,
        )

    assert excinfo.value.code == "knowledge_base.copy_storage_mismatch"


async def test_integration_duplicate_copies_settings_only(session: AsyncSession) -> None:
    tid = _int32_tenant_id()
    service = _service(session)
    source = await service.create_knowledge_base(
        tenant_id=tid,
        name="docs",
        kb_type=KNOWLEDGE_BASE_TYPE_FAQ,
        chunking_config={"chunk_size": 512},
        embedding_model_id="emb-1",
    )
    now = datetime.now(UTC)
    await ChunkRepository(session).create(
        Chunk(
            id=f"chunk-{uuid.uuid4().hex[:12]}",
            tenant_id=tid,
            knowledge_base_id=source.id,
            knowledge_id="doc-1",
            content="hello world",
            chunk_index=0,
            start_at=0,
            end_at=11,
            created_at=now,
            updated_at=now,
        )
    )
    await session.commit()

    target = await duplicate_kb(
        service=service,
        session=session,
        tenant_id=tid,
        source_kb_id=source.id,
    )
    await session.commit()

    assert target.id != source.id
    assert target.name == "docs Copy"
    assert target.chunking_config == {"chunk_size": 512}
    assert target.embedding_model_id == "emb-1"
    # Content is not copied: the source keeps its chunk, the duplicate has none.
    assert await service.count_chunks(tenant_id=tid, knowledge_base_id=source.id) == 1
    assert await service.count_chunks(tenant_id=tid, knowledge_base_id=target.id) == 0


async def test_integration_duplicate_name_dedup(session: AsyncSession) -> None:
    service = _service(session)
    tid = _int32_tenant_id()
    source = await service.create_knowledge_base(
        tenant_id=tid, name="docs", kb_type=KNOWLEDGE_BASE_TYPE_FAQ
    )
    await service.create_knowledge_base(
        tenant_id=tid, name="docs Copy", kb_type=KNOWLEDGE_BASE_TYPE_FAQ
    )
    await session.commit()

    target = await duplicate_kb(
        service=service,
        session=session,
        tenant_id=tid,
        source_kb_id=source.id,
    )
    await session.commit()

    assert target.name == "docs Copy 2"


async def test_integration_duplicate_rejects_cross_tenant_source(session: AsyncSession) -> None:
    service = _service(session)
    other_tid = make_test_tenant_id()
    source = await service.create_knowledge_base(tenant_id=other_tid, name="docs")
    await session.commit()

    with pytest.raises(NotFoundError):
        await duplicate_kb(
            service=service,
            session=session,
            tenant_id=make_test_tenant_id(),
            source_kb_id=source.id,
        )


async def test_integration_duplicate_rejects_blank_source_id(session: AsyncSession) -> None:
    service = _service(session)

    with pytest.raises(ValidationError) as excinfo:
        await duplicate_kb(
            service=service,
            session=session,
            tenant_id=make_test_tenant_id(),
            source_kb_id="   ",
        )

    assert excinfo.value.code == "knowledge_base.id_required"
