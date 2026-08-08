"""Unit + integration tests for the knowledge create variants.

Unit tests drive the create-variant modules (URL / passage / manual)
with stateful repository mocks (closure-captured storage, the same
pattern used across the core service tests): they cover validation,
error classification, duplicate detection, tag attachment, and the
happy paths.

Integration tests run against the real applied schema (``documents`` /
``chunks`` / ``tags`` tables) using the tenant-id factory from the
integration conftest. The passage-sync path writes ``chunks`` rows whose
``tenant_id`` is a 32-bit integer, so it uses an int32-safe tenant id
from a local counter rather than ``make_test_tenant_id``'s BIGINT range.
They require a reachable database — run with ``DATABASE_URL_OVERRIDE``.
"""

from __future__ import annotations

import hashlib
import itertools
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

from src.common.exception import ConflictError, NotFoundError, ValidationError
from src.common.json import BindParams
from src.core.contracts.knowledge import Knowledge
from src.core.knowledge.documents.create_manual import create_knowledge_from_manual
from src.core.knowledge.documents.create_passage import create_knowledge_from_passage
from src.core.knowledge.documents.create_url import create_knowledge_from_url
from src.core.knowledge.documents.types import (
    CHANNEL_API,
    CHANNEL_WEB,
    KNOWLEDGE_TYPE_MANUAL,
    MANUAL_KNOWLEDGE_FORMAT_MARKDOWN,
    MANUAL_KNOWLEDGE_STATUS_DRAFT,
    PARSE_STATUS_COMPLETED,
    PARSE_STATUS_PENDING,
)
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.dao.knowledge_tag_repository import TagRepository
from src.db.models.chunk import Chunk
from src.db.models.knowledge import Document
from src.db.models.knowledge_base import KnowledgeBase as KnowledgeBaseRow
from src.db.models.knowledge_tag import KnowledgeTag
from src.settings import get_settings, reset_settings_cache
from tests.integration.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_LATER = _NOW + timedelta(days=1)
_FAKER_SEED_MAX = 100_000_000

# int32-safe tenant ids for the passage-sync path (``chunks.tenant_id``
# is a 32-bit integer column).
_int32_tenant_counter = itertools.count(2_000_000)


def _int32_tenant_id() -> int:
    """Return a unique tenant id that fits PostgreSQL's INTEGER range."""
    return next(_int32_tenant_counter)


@pytest.fixture(autouse=True)
def faker_seed() -> None:
    """Re-seed Faker per test for varied-but-reproducible generation."""
    Faker.seed(randint(1, _FAKER_SEED_MAX))


def _did() -> str:
    return f"doc-{uuid.uuid4().hex[:12]}"


def _kbid() -> str:
    return f"kb-{uuid.uuid4().hex[:12]}"


def _hash(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()


# ── Mock helpers ──────────────────────────────────────────────────────


def _kb_info(
    *,
    tenant_id: int,
    kb_id: str,
    kb_type: str = "document",
    embedding_model_id: str = "emb-1",
) -> KnowledgeBaseInfo:
    return KnowledgeBaseInfo(
        id=kb_id,
        name="workspace knowledge",
        type=kb_type,
        tenant_id=tenant_id,
        embedding_model_id=embedding_model_id,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _kb_service(*, info: KnowledgeBaseInfo | None = None) -> AsyncMock:
    svc = AsyncMock(spec=KBService)
    if info is None:
        raise AssertionError("_kb_service requires an info")
    async def _get_kb(*, knowledge_base_id: str) -> KnowledgeBaseInfo:
        return info

    svc.get_knowledge_base_by_id.side_effect = _get_kb
    return svc


def _kb_service_not_found() -> AsyncMock:
    svc = AsyncMock(spec=KBService)

    async def _get_kb(*, knowledge_base_id: str) -> KnowledgeBaseInfo:
        raise NotFoundError(
            code="knowledge_base.not_found",
            message="knowledge base not found",
        )

    svc.get_knowledge_base_by_id.side_effect = _get_kb
    return svc


def _make_knowledge_repo() -> tuple[AsyncMock, dict[str, Document]]:
    """Knowledge repository mock with closure-captured storage."""
    repo = AsyncMock(spec=KnowledgeRepository)
    rows: dict[str, Document] = {}

    async def _create(row: Document) -> Document:
        rows[row.id] = row
        return row

    async def _list_by_knowledge_base(
        tenant_id: int,
        knowledge_base_id: str,
    ) -> list[Document]:
        return [
            row
            for row in rows.values()
            if (
                row.tenant_id == tenant_id
                and row.knowledge_base_id == knowledge_base_id
                and row.deleted_at is None
            )
        ]

    async def _update(row: Document) -> Document:
        rows[row.id] = row
        return row

    async def _update_columns(id: str, values: BindParams) -> Document | None:
        row = rows.get(id)
        if row is None:
            return None
        updated = row.model_copy(update=dict(values))
        rows[id] = updated
        return updated

    repo.create.side_effect = _create
    repo.list_by_knowledge_base.side_effect = _list_by_knowledge_base
    repo.update.side_effect = _update
    repo.update_columns.side_effect = _update_columns
    return repo, rows


def _sample_row(
    *,
    tenant_id: int,
    kb_id: str,
    id: str | None = None,
    type: str = "file",
    title: str = "Q3 budget",
    source: str = "budget-2026.pdf",
    parse_status: str = "completed",
    file_hash: str | None = None,
    file_name: str | None = "budget-2026.pdf",
    **columns: object,
) -> Document:
    """Build a persisted-shape document row for seeding mocks / DB."""
    return Document.model_validate(
        {
            "id": id or _did(),
            "tenant_id": tenant_id,
            "knowledge_base_id": kb_id,
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
            "file_hash": file_hash,
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


def _make_tag_repo() -> tuple[AsyncMock, dict[str, KnowledgeTag]]:
    """Tag repository mock with closure-captured storage."""
    repo = AsyncMock(spec=TagRepository)
    tags: dict[str, KnowledgeTag] = {}

    async def _get_by_ids(tenant_id: int, ids: list[str]) -> list[KnowledgeTag]:
        return [
            tag for tag_id in ids if (tag := tags.get(tag_id)) is not None and tag.tenant_id == tenant_id
        ]

    repo.get_by_ids.side_effect = _get_by_ids
    repo.set_knowledge_tags.return_value = None
    return repo, tags


def _seed_tag(
    *,
    tags: dict[str, KnowledgeTag],
    tag_id: str,
    tenant_id: int,
    kb_id: str,
) -> None:
    tags[tag_id] = KnowledgeTag(
        id=tag_id,
        seq_id=1,
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        name=tag_id,
        color="",
        sort_order=0,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _make_chunk_repo() -> tuple[AsyncMock, list[Chunk]]:
    """Chunk repository mock that records created rows."""
    repo = AsyncMock(spec=ChunkRepository)
    created: list[Chunk] = []

    async def _create_many(rows: list[Chunk]) -> list[Chunk]:
        created.extend(rows)
        return rows

    repo.create_many.side_effect = _create_many
    return repo, created


async def _url_guard_ok(url: str) -> None:
    return None


async def _url_guard_blocked(url: str) -> None:
    raise ValidationError(code="oidc.ssrf_blocked", message="restricted host")


# ── create_from_url: happy paths ──────────────────────────────────────


async def test_create_url_from_web_creates_url_row() -> None:
    repo, rows = _make_knowledge_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    url = "https://example.com/docs/guide"
    result = await create_knowledge_from_url(
        tenant_id=tenant_id,
        kb_id=kb_id,
        url=url,
        title="Guide",
        knowledge_repo=repo,
        kb_service=_kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id)),
        url_guard=_url_guard_ok,
        now=_NOW,
    )
    assert isinstance(result, Knowledge)
    assert result.type == "url"
    assert result.source == url
    assert result.title == "Guide"
    assert result.file_type == "html"
    assert result.file_hash == _hash(url)
    assert result.parse_status == PARSE_STATUS_PENDING
    assert result.enable_status == "disabled"
    assert result.channel == CHANNEL_WEB
    assert result.embedding_model_id == "emb-1"
    stored = rows[result.id]
    assert stored.tenant_id == tenant_id
    assert stored.knowledge_base_id == kb_id


async def test_create_url_accepts_custom_channel_and_filename_hints() -> None:
    repo, _rows = _make_knowledge_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    result = await create_knowledge_from_url(
        tenant_id=tenant_id,
        kb_id=kb_id,
        url="https://example.com/a",
        file_name="custom.pdf",
        file_type="pdf",
        title="Custom",
        channel=CHANNEL_API,
        knowledge_repo=repo,
        kb_service=_kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id)),
        url_guard=_url_guard_ok,
        now=_NOW,
    )
    assert result.type == "file_url"
    assert result.file_name == "custom.pdf"
    assert result.file_type == "pdf"
    assert result.channel == CHANNEL_API
    assert result.title == "Custom"


async def test_create_url_routes_file_url_and_extracts_name() -> None:
    repo, _rows = _make_knowledge_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    url = "https://example.com/files/report.pdf"
    result = await create_knowledge_from_url(
        tenant_id=tenant_id,
        kb_id=kb_id,
        url=url,
        knowledge_repo=repo,
        kb_service=_kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id)),
        url_guard=_url_guard_ok,
        now=_NOW,
    )
    assert result.type == "file_url"
    assert result.file_name == "report.pdf"
    assert result.file_type == "pdf"
    assert result.source == url
    assert result.title == "report.pdf"
    assert result.file_hash == _hash(url)


# ── create_from_url: validation ───────────────────────────────────────


async def test_create_url_rejects_blank_and_invalid_url() -> None:
    repo, _rows = _make_knowledge_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    service = _kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id))

    with pytest.raises(ValidationError) as exc_info:
        await create_knowledge_from_url(
            tenant_id=tenant_id,
            kb_id=kb_id,
            url="  ",
            knowledge_repo=repo,
            kb_service=service,
        )
    assert exc_info.value.code == "knowledge.url_required"

    with pytest.raises(ValidationError) as exc_info:
        await create_knowledge_from_url(
            tenant_id=tenant_id,
            kb_id=kb_id,
            url="ftp://example.com/x",
            knowledge_repo=repo,
            kb_service=service,
        )
    assert exc_info.value.code == "knowledge.invalid_url"

    with pytest.raises(ValidationError) as exc_info:
        await create_knowledge_from_url(
            tenant_id=tenant_id,
            kb_id=kb_id,
            url="https://example.com/javascript:alert(1)",
            knowledge_repo=repo,
            kb_service=service,
        )
    assert exc_info.value.code == "knowledge.invalid_url"


async def test_create_url_rejects_ssrf_blocked_host() -> None:
    repo, _rows = _make_knowledge_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    with pytest.raises(ValidationError) as exc_info:
        await create_knowledge_from_url(
            tenant_id=tenant_id,
            kb_id=kb_id,
            url="http://localhost:8000/x",
            knowledge_repo=repo,
            kb_service=_kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id)),
            url_guard=_url_guard_blocked,
        )
    assert exc_info.value.code == "knowledge.invalid_url"


async def test_create_url_kb_not_found_raises_not_found() -> None:
    repo, _rows = _make_knowledge_repo()
    with pytest.raises(NotFoundError) as exc_info:
        await create_knowledge_from_url(
            tenant_id=make_test_tenant_id(),
            kb_id=_kbid(),
            url="https://example.com/a",
            knowledge_repo=repo,
            kb_service=_kb_service_not_found(),
            url_guard=_url_guard_ok,
        )
    assert exc_info.value.code == "knowledge_base.not_found"


async def test_create_url_file_url_rejects_faq_kb() -> None:
    repo, _rows = _make_knowledge_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    with pytest.raises(ValidationError) as exc_info:
        await create_knowledge_from_url(
            tenant_id=tenant_id,
            kb_id=kb_id,
            url="https://example.com/file.pdf",
            knowledge_repo=repo,
            kb_service=_kb_service(
                info=_kb_info(tenant_id=tenant_id, kb_id=kb_id, kb_type="faq")
            ),
            url_guard=_url_guard_ok,
        )
    assert exc_info.value.code == "knowledge.faq_file_unsupported"


async def test_create_url_file_url_rejects_unsupported_type() -> None:
    repo, _rows = _make_knowledge_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    with pytest.raises(ValidationError) as exc_info:
        await create_knowledge_from_url(
            tenant_id=tenant_id,
            kb_id=kb_id,
            url="https://example.com/a",
            file_name="data.xyz",
            knowledge_repo=repo,
            kb_service=_kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id)),
            url_guard=_url_guard_ok,
        )
    assert exc_info.value.code == "knowledge.file_type_unsupported"


async def test_create_url_file_url_rejects_video() -> None:
    repo, _rows = _make_knowledge_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    with pytest.raises(ValidationError) as exc_info:
        await create_knowledge_from_url(
            tenant_id=tenant_id,
            kb_id=kb_id,
            url="https://example.com/a",
            file_name="movie.mp4",
            knowledge_repo=repo,
            kb_service=_kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id)),
            url_guard=_url_guard_ok,
        )
    assert exc_info.value.code == "knowledge.video_unsupported"


async def test_create_url_rejects_invalid_file_name() -> None:
    repo, _rows = _make_knowledge_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    with pytest.raises(ValidationError) as exc_info:
        await create_knowledge_from_url(
            tenant_id=tenant_id,
            kb_id=kb_id,
            url="https://example.com/a",
            file_name="javascript:x.pdf",
            knowledge_repo=repo,
            kb_service=_kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id)),
            url_guard=_url_guard_ok,
        )
    assert exc_info.value.code == "knowledge.file_name_invalid"


# ── create_from_url: duplicates ───────────────────────────────────────


async def test_create_url_duplicate_web_url_raises_conflict() -> None:
    repo, rows = _make_knowledge_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    url = "https://example.com/dup"
    existing = _sample_row(
        tenant_id=tenant_id,
        kb_id=kb_id,
        type="url",
        title="existing",
        source=url,
        file_hash=_hash(url),
        file_name=None,
    )
    rows[existing.id] = existing
    with pytest.raises(ConflictError) as exc_info:
        await create_knowledge_from_url(
            tenant_id=tenant_id,
            kb_id=kb_id,
            url=url,
            knowledge_repo=repo,
            kb_service=_kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id)),
            url_guard=_url_guard_ok,
            now=_LATER,
        )
    assert exc_info.value.code == "knowledge.duplicate_url"
    existing = next(row for row in rows.values() if row.type == "url")
    assert existing.created_at == _LATER
    assert existing.updated_at == _LATER


async def test_create_url_duplicate_file_url_raises_conflict() -> None:
    repo, rows = _make_knowledge_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    url = "https://example.com/dup.pdf"
    existing = _sample_row(
        tenant_id=tenant_id,
        kb_id=kb_id,
        type="file_url",
        title="existing",
        source=url,
        file_hash=_hash(url),
        file_name="dup.pdf",
    )
    rows[existing.id] = existing
    with pytest.raises(ConflictError) as exc_info:
        await create_knowledge_from_url(
            tenant_id=tenant_id,
            kb_id=kb_id,
            url=url,
            knowledge_repo=repo,
            kb_service=_kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id)),
            url_guard=_url_guard_ok,
        )
    assert exc_info.value.code == "knowledge.duplicate_file"


# ── create_from_url: tags ─────────────────────────────────────────────


async def test_create_url_attaches_valid_tags() -> None:
    repo, _rows = _make_knowledge_repo()
    tag_repo, tags = _make_tag_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    _seed_tag(tags=tags, tag_id="tag-1", tenant_id=tenant_id, kb_id=kb_id)
    result = await create_knowledge_from_url(
        tenant_id=tenant_id,
        kb_id=kb_id,
        url="https://example.com/a",
        tag_ids=["tag-1", "tag-1"],
        knowledge_repo=repo,
        kb_service=_kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id)),
        tag_repo=tag_repo,
        url_guard=_url_guard_ok,
    )
    tag_repo.set_knowledge_tags.assert_awaited_once()
    kwargs = tag_repo.set_knowledge_tags.await_args.kwargs
    assert kwargs["knowledge_id"] == result.id
    assert kwargs["tag_ids"] == ["tag-1"]


async def test_create_url_rejects_unknown_tag() -> None:
    repo, _rows = _make_knowledge_repo()
    tag_repo, _tags = _make_tag_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    with pytest.raises(ValidationError) as exc_info:
        await create_knowledge_from_url(
            tenant_id=tenant_id,
            kb_id=kb_id,
            url="https://example.com/a",
            tag_ids=["tag-missing"],
            knowledge_repo=repo,
            kb_service=_kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id)),
            tag_repo=tag_repo,
            url_guard=_url_guard_ok,
        )
    assert exc_info.value.code == "knowledge.tag_not_found"


async def test_create_url_rejects_tag_of_another_kb() -> None:
    repo, _rows = _make_knowledge_repo()
    tag_repo, tags = _make_tag_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    _seed_tag(tags=tags, tag_id="tag-other", tenant_id=tenant_id, kb_id="kb-other")
    with pytest.raises(ValidationError) as exc_info:
        await create_knowledge_from_url(
            tenant_id=tenant_id,
            kb_id=kb_id,
            url="https://example.com/a",
            tag_ids=["tag-other"],
            knowledge_repo=repo,
            kb_service=_kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id)),
            tag_repo=tag_repo,
            url_guard=_url_guard_ok,
        )
    assert exc_info.value.code == "knowledge.tag_not_in_kb"


async def test_create_url_tags_require_tag_repo() -> None:
    repo, _rows = _make_knowledge_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    with pytest.raises(ValidationError) as exc_info:
        await create_knowledge_from_url(
            tenant_id=tenant_id,
            kb_id=kb_id,
            url="https://example.com/a",
            tag_ids=["tag-1"],
            knowledge_repo=repo,
            kb_service=_kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id)),
            url_guard=_url_guard_ok,
        )
    assert exc_info.value.code == "knowledge.tag_repo_required"


# ── create_from_passage ───────────────────────────────────────────────


async def test_create_passage_creates_row_async() -> None:
    repo, rows = _make_knowledge_repo()
    chunk_repo, _created = _make_chunk_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    result = await create_knowledge_from_passage(
        tenant_id=tenant_id,
        kb_id=kb_id,
        passages=["alpha", "beta"],
        knowledge_repo=repo,
        kb_service=_kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id)),
        chunk_repo=chunk_repo,
        now=_NOW,
    )
    assert result.type == "passage"
    assert result.parse_status == PARSE_STATUS_PENDING
    assert result.enable_status == "disabled"
    assert result.title == ""
    stored = rows[result.id]
    assert stored.embedding_model_id == "emb-1"
    assert stored.channel == CHANNEL_WEB
    chunk_repo.create_many.assert_not_awaited()


async def test_create_passage_rejects_empty_list() -> None:
    repo, _rows = _make_knowledge_repo()
    with pytest.raises(ValidationError) as exc_info:
        await create_knowledge_from_passage(
            tenant_id=make_test_tenant_id(),
            kb_id=_kbid(),
            passages=[],
            knowledge_repo=repo,
            kb_service=_kb_service(
                info=_kb_info(tenant_id=make_test_tenant_id(), kb_id=_kbid())
            ),
        )
    assert exc_info.value.code == "knowledge.passage_required"


async def test_create_passage_rejects_invalid_passage() -> None:
    repo, _rows = _make_knowledge_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    for bad in ("bad\x00passage", "<script>alert(1)</script>"):
        with pytest.raises(ValidationError) as exc_info:
            await create_knowledge_from_passage(
                tenant_id=tenant_id,
                kb_id=kb_id,
                passages=["ok", bad],
                knowledge_repo=repo,
                kb_service=_kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id)),
            )
        assert exc_info.value.code == "knowledge.passage_invalid"


async def test_create_passage_kb_not_found() -> None:
    repo, _rows = _make_knowledge_repo()
    with pytest.raises(NotFoundError) as exc_info:
        await create_knowledge_from_passage(
            tenant_id=make_test_tenant_id(),
            kb_id=_kbid(),
            passages=["a"],
            knowledge_repo=repo,
            kb_service=_kb_service_not_found(),
        )
    assert exc_info.value.code == "knowledge_base.not_found"


async def test_create_passage_sync_writes_chunks_and_settles() -> None:
    repo, rows = _make_knowledge_repo()
    chunk_repo, created = _make_chunk_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    result = await create_knowledge_from_passage(
        tenant_id=tenant_id,
        kb_id=kb_id,
        passages=["alpha", "", "beta"],
        sync=True,
        knowledge_repo=repo,
        kb_service=_kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id)),
        chunk_repo=chunk_repo,
        now=_NOW,
    )
    assert result.parse_status == PARSE_STATUS_COMPLETED
    assert result.enable_status == "enabled"
    assert result.processed_at == _NOW
    stored = rows[result.id]
    assert stored.parse_status == PARSE_STATUS_COMPLETED
    # Empty passage skipped; offsets accumulate over trimmed content.
    assert [chunk.content for chunk in created] == ["alpha", "beta"]
    assert created[0].chunk_index == 0
    assert created[1].chunk_index == 2
    assert (created[0].start_at, created[0].end_at) == (0, 5)
    assert (created[1].start_at, created[1].end_at) == (5, 9)
    for chunk in created:
        assert chunk.knowledge_id == result.id
        assert chunk.tenant_id == tenant_id


async def test_create_passage_sync_requires_chunk_repo() -> None:
    repo, _rows = _make_knowledge_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    with pytest.raises(ValueError):
        await create_knowledge_from_passage(
            tenant_id=tenant_id,
            kb_id=kb_id,
            passages=["a"],
            sync=True,
            knowledge_repo=repo,
            kb_service=_kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id)),
        )


async def test_create_passage_sync_trims_content() -> None:
    repo, _rows = _make_knowledge_repo()
    chunk_repo, created = _make_chunk_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    await create_knowledge_from_passage(
        tenant_id=tenant_id,
        kb_id=kb_id,
        passages=["  padded  "],
        sync=True,
        knowledge_repo=repo,
        kb_service=_kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id)),
        chunk_repo=chunk_repo,
        now=_NOW,
    )
    assert created[0].content == "padded"


# ── create_from_manual ────────────────────────────────────────────────


async def test_create_manual_draft_creates_row() -> None:
    repo, rows = _make_knowledge_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    result = await create_knowledge_from_manual(
        tenant_id=tenant_id,
        kb_id=kb_id,
        title="Notes",
        content="# Hello",
        knowledge_repo=repo,
        kb_service=_kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id)),
        now=_NOW,
    )
    assert result.type == KNOWLEDGE_TYPE_MANUAL
    assert result.title == "Notes"
    assert result.parse_status == MANUAL_KNOWLEDGE_STATUS_DRAFT
    assert result.enable_status == "disabled"
    assert result.file_name == "Notes.md"
    assert result.file_type == KNOWLEDGE_TYPE_MANUAL
    assert result.source == KNOWLEDGE_TYPE_MANUAL
    stored = rows[result.id]
    assert stored.channel == CHANNEL_WEB
    assert stored.metadata == {
        "content": "# Hello",
        "format": MANUAL_KNOWLEDGE_FORMAT_MARKDOWN,
        "status": MANUAL_KNOWLEDGE_STATUS_DRAFT,
        "version": 1,
        "updated_at": _NOW.isoformat(),
    }


async def test_create_manual_publish_stamps_pending_and_overrides() -> None:
    repo, rows = _make_knowledge_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    result = await create_knowledge_from_manual(
        tenant_id=tenant_id,
        kb_id=kb_id,
        title="Post",
        content="content",
        status="publish",
        process_overrides={"chunk_size": 512},
        knowledge_repo=repo,
        kb_service=_kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id)),
        now=_NOW,
    )
    assert result.parse_status == PARSE_STATUS_PENDING
    stored = rows[result.id]
    assert stored.metadata is not None
    assert stored.metadata["status"] == "publish"
    assert stored.metadata["process_overrides"] == {"chunk_size": 512}


async def test_create_manual_default_title_and_status() -> None:
    repo, rows = _make_knowledge_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    result = await create_knowledge_from_manual(
        tenant_id=tenant_id,
        kb_id=kb_id,
        title="  ",
        content="content",
        status=" DRAFT ",
        knowledge_repo=repo,
        kb_service=_kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id)),
        now=_NOW,
    )
    assert result.title == "Knowledge-20260101-000000"
    assert result.file_name == "Knowledge-20260101-000000.md"
    assert result.parse_status == MANUAL_KNOWLEDGE_STATUS_DRAFT
    stored = rows[result.id]
    assert stored.metadata is not None
    assert stored.metadata["status"] == MANUAL_KNOWLEDGE_STATUS_DRAFT


async def test_create_manual_cleans_content() -> None:
    repo, rows = _make_knowledge_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    result = await create_knowledge_from_manual(
        tenant_id=tenant_id,
        kb_id=kb_id,
        title="Clean",
        content="<script>alert(1)</script>Hello",
        knowledge_repo=repo,
        kb_service=_kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id)),
        now=_NOW,
    )
    stored = rows[result.id]
    assert stored.metadata is not None
    assert stored.metadata["content"] == "Hello"


async def test_create_manual_rejects_empty_and_blank_content() -> None:
    repo, _rows = _make_knowledge_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    service = _kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id))
    for blank in ("   ", "<script></script>"):
        with pytest.raises(ValidationError) as exc_info:
            await create_knowledge_from_manual(
                tenant_id=tenant_id,
                kb_id=kb_id,
                title="t",
                content=blank,
                knowledge_repo=repo,
                kb_service=service,
            )
        assert exc_info.value.code == "knowledge.content_required"


async def test_create_manual_rejects_too_long_content() -> None:
    repo, _rows = _make_knowledge_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    with pytest.raises(ValidationError) as exc_info:
        await create_knowledge_from_manual(
            tenant_id=tenant_id,
            kb_id=kb_id,
            title="t",
            content="x" * 200001,
            knowledge_repo=repo,
            kb_service=_kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id)),
        )
    assert exc_info.value.code == "knowledge.content_too_long"


async def test_create_manual_rejects_invalid_title() -> None:
    repo, _rows = _make_knowledge_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    with pytest.raises(ValidationError) as exc_info:
        await create_knowledge_from_manual(
            tenant_id=tenant_id,
            kb_id=kb_id,
            title="<script>alert(1)</script>",
            content="content",
            knowledge_repo=repo,
            kb_service=_kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id)),
        )
    assert exc_info.value.code == "knowledge.title_invalid"


async def test_create_manual_rejects_invalid_status() -> None:
    repo, _rows = _make_knowledge_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    with pytest.raises(ValidationError) as exc_info:
        await create_knowledge_from_manual(
            tenant_id=tenant_id,
            kb_id=kb_id,
            title="t",
            content="content",
            status="archived",
            knowledge_repo=repo,
            kb_service=_kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id)),
        )
    assert exc_info.value.code == "knowledge.status_invalid"


async def test_create_manual_kb_not_found() -> None:
    repo, _rows = _make_knowledge_repo()
    with pytest.raises(NotFoundError) as exc_info:
        await create_knowledge_from_manual(
            tenant_id=make_test_tenant_id(),
            kb_id=_kbid(),
            title="t",
            content="content",
            knowledge_repo=repo,
            kb_service=_kb_service_not_found(),
        )
    assert exc_info.value.code == "knowledge_base.not_found"


async def test_create_manual_attaches_tags() -> None:
    repo, _rows = _make_knowledge_repo()
    tag_repo, tags = _make_tag_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    _seed_tag(tags=tags, tag_id="tag-1", tenant_id=tenant_id, kb_id=kb_id)
    result = await create_knowledge_from_manual(
        tenant_id=tenant_id,
        kb_id=kb_id,
        title="t",
        content="content",
        tag_ids=["tag-1"],
        knowledge_repo=repo,
        kb_service=_kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id)),
        tag_repo=tag_repo,
    )
    tag_repo.set_knowledge_tags.assert_awaited_once()
    assert tag_repo.set_knowledge_tags.await_args.kwargs["knowledge_id"] == result.id
    assert tag_repo.set_knowledge_tags.await_args.kwargs["tag_ids"] == ["tag-1"]


async def test_create_manual_rejects_unknown_tag_before_insert() -> None:
    repo, _rows = _make_knowledge_repo()
    tag_repo, _tags = _make_tag_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    with pytest.raises(ValidationError) as exc_info:
        await create_knowledge_from_manual(
            tenant_id=tenant_id,
            kb_id=kb_id,
            title="t",
            content="content",
            tag_ids=["tag-missing"],
            knowledge_repo=repo,
            kb_service=_kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id)),
            tag_repo=tag_repo,
        )
    assert exc_info.value.code == "knowledge.tag_not_found"
    repo.create.assert_not_awaited()


async def test_create_manual_custom_channel_and_suffix_preserved() -> None:
    repo, rows = _make_knowledge_repo()
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    result = await create_knowledge_from_manual(
        tenant_id=tenant_id,
        kb_id=kb_id,
        title="already.md",
        content="content",
        channel=CHANNEL_API,
        knowledge_repo=repo,
        kb_service=_kb_service(info=_kb_info(tenant_id=tenant_id, kb_id=kb_id)),
        now=_NOW,
    )
    assert result.channel == CHANNEL_API
    assert result.file_name == "already.md"
    assert rows[result.id].channel == CHANNEL_API


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


async def _insert_kb(
    session: AsyncSession,
    *,
    tenant_id: int,
    kb_id: str,
    kb_type: str = "document",
) -> None:
    await KnowledgeBaseRepository(session).create(
        KnowledgeBaseRow(
            id=kb_id,
            name=f"kb-{kb_id}",
            type=kb_type,
            tenant_id=tenant_id,
            embedding_model_id="emb-1",
            created_at=_NOW,
            updated_at=_NOW,
        )
    )


async def test_integration_create_url_round_trip_and_duplicate(
    session: AsyncSession,
) -> None:
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    await _insert_kb(session, tenant_id=tenant_id, kb_id=kb_id)
    kb_service = KBService(kb_repo=KnowledgeBaseRepository(session))
    url = "https://example.com/docs/integration"
    result = await create_knowledge_from_url(
        tenant_id=tenant_id,
        kb_id=kb_id,
        url=url,
        title="Integration",
        knowledge_repo=KnowledgeRepository(session),
        kb_service=kb_service,
        now=_NOW,
    )
    assert result.type == "url"
    assert result.file_hash == _hash(url)
    row = await KnowledgeRepository(session).get_by_id(tenant_id=tenant_id, id=result.id)
    assert row is not None
    assert row.source == url
    assert row.embedding_model_id == "emb-1"

    with pytest.raises(ConflictError) as exc_info:
        await create_knowledge_from_url(
            tenant_id=tenant_id,
            kb_id=kb_id,
            url=url,
            knowledge_repo=KnowledgeRepository(session),
            kb_service=kb_service,
            now=_LATER,
        )
    assert exc_info.value.code == "knowledge.duplicate_url"


async def test_integration_create_manual_round_trip(session: AsyncSession) -> None:
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    await _insert_kb(session, tenant_id=tenant_id, kb_id=kb_id)
    kb_service = KBService(kb_repo=KnowledgeBaseRepository(session))
    result = await create_knowledge_from_manual(
        tenant_id=tenant_id,
        kb_id=kb_id,
        title="Integration notes",
        content="# Hello",
        status="publish",
        knowledge_repo=KnowledgeRepository(session),
        kb_service=kb_service,
        now=_NOW,
    )
    assert result.parse_status == PARSE_STATUS_PENDING
    row = await KnowledgeRepository(session).get_by_id(tenant_id=tenant_id, id=result.id)
    assert row is not None
    assert row.type == KNOWLEDGE_TYPE_MANUAL
    assert row.file_name == "Integration notes.md"
    assert row.metadata is not None
    assert row.metadata["content"] == "# Hello"
    assert row.metadata["status"] == "publish"
    assert row.metadata["format"] == MANUAL_KNOWLEDGE_FORMAT_MARKDOWN


async def test_integration_create_passage_sync_writes_chunks(
    session: AsyncSession,
) -> None:
    tenant_id = _int32_tenant_id()
    kb_id = _kbid()
    await _insert_kb(session, tenant_id=tenant_id, kb_id=kb_id)
    kb_service = KBService(kb_repo=KnowledgeBaseRepository(session))
    result = await create_knowledge_from_passage(
        tenant_id=tenant_id,
        kb_id=kb_id,
        passages=["alpha", "beta"],
        sync=True,
        knowledge_repo=KnowledgeRepository(session),
        kb_service=kb_service,
        chunk_repo=ChunkRepository(session),
        now=_NOW,
    )
    assert result.parse_status == PARSE_STATUS_COMPLETED
    chunks = await ChunkRepository(session).list_by_knowledge_id(tenant_id, result.id)
    assert {chunk.content for chunk in chunks} == {"alpha", "beta"}
    assert all(chunk.tenant_id == tenant_id for chunk in chunks)
    row = await KnowledgeRepository(session).get_by_id(tenant_id=tenant_id, id=result.id)
    assert row is not None
    assert row.parse_status == PARSE_STATUS_COMPLETED
    assert row.enable_status == "enabled"


async def test_integration_create_manual_with_tag_binding(
    session: AsyncSession,
) -> None:
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    await _insert_kb(session, tenant_id=tenant_id, kb_id=kb_id)
    kb_service = KBService(kb_repo=KnowledgeBaseRepository(session))
    tag_repo = TagRepository(session)
    await tag_repo.create(
        KnowledgeTag(
            id="tag-int-1",
            seq_id=0,
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            name="docs",
            color="",
            sort_order=0,
            created_at=_NOW,
            updated_at=_NOW,
        )
    )
    result = await create_knowledge_from_manual(
        tenant_id=tenant_id,
        kb_id=kb_id,
        title="Tagged",
        content="content",
        tag_ids=["tag-int-1"],
        knowledge_repo=KnowledgeRepository(session),
        kb_service=kb_service,
        tag_repo=tag_repo,
        now=_NOW,
    )
    bindings = await tag_repo.get_knowledge_tags([result.id])
    assert [tag.id for tag in bindings[result.id]] == ["tag-int-1"]
