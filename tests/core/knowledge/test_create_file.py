"""Unit + integration tests for the knowledge-from-file orchestration.

Unit tests drive ``create_knowledge_from_file`` and the
``upload_pipeline`` helpers with stateful repository / service mocks
(closure-captured storage, the pattern used across the core service
tests): they cover validation, error classification, the storage /
duplicate / quota gates, and the async dispatch seam.

Integration tests run against the real applied schema (``documents`` +
``knowledge_bases`` tables) using ``make_test_tenant_id`` and
``faker_seed`` from the integration conftest and a real local-storage
backend over a per-test temp dir. They require a reachable database —
run with ``DATABASE_URL_OVERRIDE``.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.ai.storage.local_backend import LocalStorageAdapter
from src.common.exception import ConflictError, NotFoundError, ValidationError
from src.common.json import JsonObject
from src.core.contracts.knowledge import Knowledge
from src.core.knowledge.documents.create_file import (
    TenantStorageInfo,
    create_knowledge_from_file,
)
from src.core.knowledge.documents.service.knowledge_service import KnowledgeService
from src.core.knowledge.documents.types import (
    CHANNEL_API,
    CHANNEL_WEB,
    PARSE_STATUS_FAILED,
    PARSE_STATUS_PENDING,
)
from src.core.knowledge.documents.upload_pipeline import (
    DEFAULT_CHANNEL,
    DEFAULT_QUESTION_COUNT,
    UNKNOWN_FILE_TYPE,
    DocumentProcessPayload,
    EffectiveProcessConfig,
    calculate_file_hash,
    check_file_knowledge_exists,
    default_channel,
    file_type_of,
    is_audio_type,
    is_image_type,
    is_supported_import_type,
    is_valid_file_type,
    is_video_type,
    normalize_file_extension,
    resolve_effective_process_config,
    validate_file_name,
    validate_import_file_type,
    validate_media_prerequisites,
)
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.core.knowledge.tags.service.tag_service import TagService
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.knowledge import Document
from src.settings import get_settings, reset_settings_cache
from tests.integration.conftest import faker_seed, make_test_tenant_id  # noqa: F401  (fixture)

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_PDF_BYTES = b"%PDF-1.4 sample content for create-from-file tests"


def _short() -> str:
    return uuid.uuid4().hex[:12]


def _kbid() -> str:
    return f"kb-{uuid.uuid4().hex[:12]}"


# ── Fakes ───────────────────────────────────────────────────────────────


class _FakeFile:
    """Re-readable ``FileUpload``-shaped upload for tests."""

    def __init__(
        self,
        *,
        filename: str,
        data: bytes,
        content_type: str = "application/pdf",
    ) -> None:
        self.filename = filename
        self.size = len(data)
        self.content_type = content_type
        self._data = data

    async def read(self) -> bytes:
        return self._data


class _FakeFileService:
    """In-memory file service recording saves / deletes."""

    def __init__(self, *, saved_path: str = "local://1/k/1/obj") -> None:
        self.saved_path = saved_path
        self.saved: list[tuple[str, int, str]] = []
        self.deleted: list[str] = []

    async def save_file(
        self, *, file: object, tenant_id: int, knowledge_id: str
    ) -> str:
        self.saved.append((file.filename, tenant_id, knowledge_id))  # type: ignore[attr-defined]
        return self.saved_path

    async def delete_file(self, file_path: str) -> None:
        self.deleted.append(file_path)


class _FakeDispatcher:
    """Task-dispatch seam capturing payloads; ``fail`` simulates a broken queue."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.payloads: list[DocumentProcessPayload] = []
        self.calls = 0

    async def dispatch(self, *, payload: DocumentProcessPayload) -> str:
        self.calls += 1
        if self.fail:
            raise RuntimeError("queue unavailable")
        self.payloads.append(payload)
        return f"task-{self.calls}"


# ── Repository mock (stateful via side_effect closures) ────────────────


def _make_repo() -> tuple[AsyncMock, dict[str, Document]]:
    repo = AsyncMock(spec=KnowledgeRepository)
    rows: dict[str, Document] = {}

    async def _find_all_by_column_values(columns: dict[str, object]) -> list[Document]:
        return [
            row
            for row in rows.values()
            if all(getattr(row, key) == value for key, value in columns.items())
        ]

    async def _create(row: Document) -> Document:
        rows[row.id] = row
        return row

    async def _update_columns(id: str, values: dict[str, object]) -> Document | None:
        row = rows.get(id)
        if row is None:
            return None
        updated = row.model_copy(update=values)
        rows[id] = updated
        return updated

    repo.find_all_by_column_values.side_effect = _find_all_by_column_values
    repo.create.side_effect = _create
    repo.update_columns.side_effect = _update_columns
    return repo, rows


def _sample_doc(
    *,
    id: str,
    tenant_id: int,
    knowledge_base_id: str,
    file_name: str,
    file_type: str,
    file_hash: str,
    parse_status: str = "pending",
) -> Document:
    return Document(
        id=id,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        type="file",
        title=file_name,
        description=None,
        source="",
        channel=CHANNEL_WEB,
        parse_status=parse_status,
        pending_subtasks_count=0,
        summary_status="none",
        enable_status="disabled",
        embedding_model_id=None,
        file_name=file_name,
        file_type=file_type,
        file_size=len(_PDF_BYTES),
        file_hash=file_hash,
        file_path="local://1/k/1/obj",
        storage_size=0,
        metadata=None,
        custom_metadata={},
        last_faq_import_result=None,
        created_at=_NOW,
        updated_at=_NOW,
        processed_at=None,
        error_message=None,
        deleted_at=None,
    )


def _kb_info(
    *,
    knowledge_base_id: str = "kb-1",
    kb_type: str = "document",
    embedding_model_id: str = "emb-1",
    chunking_config: JsonObject | None = None,
    question_generation_config: JsonObject | None = None,
    vlm_config: JsonObject | None = None,
    asr_config: JsonObject | None = None,
) -> KnowledgeBaseInfo:
    return KnowledgeBaseInfo(
        id=knowledge_base_id,
        name="Docs",
        type=kb_type,
        tenant_id=1,
        embedding_model_id=embedding_model_id,
        chunking_config=chunking_config,
        question_generation_config=question_generation_config,
        vlm_config=vlm_config,
        asr_config=asr_config,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _kb_service_mock(kb: KnowledgeBaseInfo) -> AsyncMock:
    service = AsyncMock(spec=KBService)
    service.get_knowledge_base_by_id.return_value = kb
    return service


def _upload_file(
    *,
    filename: str = "report.pdf",
    data: bytes = _PDF_BYTES,
) -> _FakeFile:
    return _FakeFile(filename=filename, data=data)


# ── upload_pipeline helpers (unit) ─────────────────────────────────────


def test_file_type_helpers() -> None:
    assert file_type_of("report.pdf") == "pdf"
    assert file_type_of("report") == UNKNOWN_FILE_TYPE
    assert file_type_of("a.b.c") == "c"
    assert is_video_type("MP4") is True
    assert is_video_type("pdf") is False
    assert is_image_type("png") is True
    assert is_image_type("txt") is False
    assert is_audio_type("mp3") is True
    assert is_audio_type("mp4") is False
    assert is_supported_import_type("pdf") is True
    assert is_supported_import_type("exe") is False
    assert is_supported_import_type("unknown") is False
    assert is_valid_file_type("report.pdf") is True
    assert is_valid_file_type("report.xyz") is False
    assert normalize_file_extension(".XLSX") == "xlsx"


def test_validate_import_file_type_accepts_and_normalises() -> None:
    assert validate_import_file_type(".PDF") == "pdf"


def test_validate_import_file_type_rejects_unknown_video_unsupported() -> None:
    with pytest.raises(ValidationError) as exc:
        validate_import_file_type("unknown")
    assert exc.value.code == "knowledge.invalid_file_type"

    with pytest.raises(ValidationError) as exc:
        validate_import_file_type("mp4")
    assert exc.value.code == "knowledge.video_not_supported"

    with pytest.raises(ValidationError) as exc:
        validate_import_file_type("exe")
    assert exc.value.code == "knowledge.unsupported_file_type"


def test_calculate_file_hash_matches_md5() -> None:
    assert calculate_file_hash(b"hello") == hashlib.md5(b"hello").hexdigest()


def test_validate_file_name_rejects_unsafe_names() -> None:
    assert validate_file_name(" report.pdf ") == "report.pdf"
    with pytest.raises(ValidationError) as exc:
        validate_file_name("")
    assert exc.value.code == "knowledge.empty_file_name"
    with pytest.raises(ValidationError) as exc:
        validate_file_name("bad\nname.pdf")
    assert exc.value.code == "knowledge.invalid_file_name"


def test_default_channel_falls_back_to_web() -> None:
    assert default_channel("") == DEFAULT_CHANNEL
    assert default_channel(CHANNEL_API) == CHANNEL_API


def test_resolve_effective_process_config_precedence() -> None:
    # KB defaults when nothing overrides.
    eff = resolve_effective_process_config(
        chunking_config={"enable_multimodal": True},
        question_generation_config={"enabled": True, "question_count": 5},
        enable_multimodel=None,
        process_overrides=None,
    )
    assert eff == EffectiveProcessConfig(
        enable_multimodel=True,
        enable_question_generation=True,
        question_count=5,
    )

    # Override wins over the request flag; non-positive count defaults.
    eff = resolve_effective_process_config(
        chunking_config={},
        question_generation_config=None,
        enable_multimodel=True,
        process_overrides={
            "enable_multimodel": False,
            "question_generation_config": {"enabled": True, "question_count": 0},
        },
    )
    assert eff.enable_multimodel is False
    assert eff.enable_question_generation is True
    assert eff.question_count == DEFAULT_QUESTION_COUNT

    # Request flag applies when no override pins the value.
    eff = resolve_effective_process_config(
        chunking_config={"enable_multimodal": False},
        question_generation_config=None,
        enable_multimodel=True,
        process_overrides=None,
    )
    assert eff.enable_multimodel is True
    assert eff.enable_question_generation is False


def test_validate_media_prerequisites() -> None:
    validate_media_prerequisites(file_type="pdf", vlm_config=None, asr_config=None)

    with pytest.raises(ValidationError) as exc:
        validate_media_prerequisites(file_type="png", vlm_config=None, asr_config=None)
    assert exc.value.code == "knowledge.vlm_required"

    validate_media_prerequisites(
        file_type="png",
        vlm_config={"enabled": True, "model_id": "vlm-1"},
        asr_config=None,
    )

    with pytest.raises(ValidationError) as exc:
        validate_media_prerequisites(file_type="mp3", vlm_config=None, asr_config=None)
    assert exc.value.code == "knowledge.asr_required"


async def test_check_file_knowledge_exists_uses_hash_within_type() -> None:
    # Same hash, same type -> duplicate.
    repo, rows = _make_repo()
    rows["doc-1"] = _sample_doc(
        id="doc-1",
        tenant_id=1,
        knowledge_base_id="kb-1",
        file_name="old.pdf",
        file_type="pdf",
        file_hash="abc123",
    )
    found = await check_file_knowledge_exists(
        repo,
        tenant_id=1,
        knowledge_base_id="kb-1",
        file_name="new.pdf",
        file_type="pdf",
        file_size=1024,
        file_hash="abc123",
    )
    assert found is not None
    assert found.id == "doc-1"

    # Same hash but a different file type is importable again.
    found = await check_file_knowledge_exists(
        repo,
        tenant_id=1,
        knowledge_base_id="kb-1",
        file_name="new.md",
        file_type="md",
        file_size=1024,
        file_hash="abc123",
    )
    assert found is None

    # A failed row is never treated as a duplicate.
    repo_failed, rows_failed = _make_repo()
    rows_failed["doc-3"] = _sample_doc(
        id="doc-3",
        tenant_id=1,
        knowledge_base_id="kb-1",
        file_name="failed.pdf",
        file_type="pdf",
        file_hash="abc123",
        parse_status=PARSE_STATUS_FAILED,
    )
    found = await check_file_knowledge_exists(
        repo_failed,
        tenant_id=1,
        knowledge_base_id="kb-1",
        file_name="new.pdf",
        file_type="pdf",
        file_size=1024,
        file_hash="abc123",
    )
    assert found is None

    # Without a hash the identity falls back to name + size.
    repo_nohash, rows_nohash = _make_repo()
    rows_nohash["doc-4"] = _sample_doc(
        id="doc-4",
        tenant_id=1,
        knowledge_base_id="kb-1",
        file_name="old.pdf",
        file_type="pdf",
        file_hash="",
    )
    found = await check_file_knowledge_exists(
        repo_nohash,
        tenant_id=1,
        knowledge_base_id="kb-1",
        file_name="old.pdf",
        file_type="pdf",
        file_size=len(_PDF_BYTES),
        file_hash="",
    )
    assert found is not None
    assert found.id == "doc-4"


# ── create_knowledge_from_file (unit) ──────────────────────────────────


async def test_create_from_file_happy_path() -> None:
    repo, rows = _make_repo()
    file_service = _FakeFileService()
    dispatcher = _FakeDispatcher()
    file = _upload_file(filename="report.pdf")
    created = await create_knowledge_from_file(
        tenant_id=1,
        knowledge_base_id="kb-1",
        file=file,
        knowledge_repo=repo,
        kb_service=_kb_service_mock(_kb_info()),
        file_service=file_service,
        dispatcher=dispatcher,
        channel=CHANNEL_API,
        language="en-US",
    )

    assert isinstance(created, Knowledge)
    assert created.id in rows
    assert created.tenant_id == 1
    assert created.knowledge_base_id == "kb-1"
    assert created.type == "file"
    assert created.title == "report.pdf"
    assert created.file_name == "report.pdf"
    assert created.file_type == "pdf"
    assert created.file_hash == calculate_file_hash(_PDF_BYTES)
    assert created.file_size == len(_PDF_BYTES)
    assert created.channel == CHANNEL_API
    assert created.parse_status == PARSE_STATUS_PENDING
    assert created.enable_status == "disabled"
    assert created.embedding_model_id == "emb-1"

    stored = rows[created.id]
    assert stored.file_path == file_service.saved_path
    assert stored.metadata is None

    assert file_service.saved == [("report.pdf", 1, created.id)]
    assert dispatcher.calls == 1
    payload = dispatcher.payloads[0]
    assert payload.knowledge_id == created.id
    assert payload.tenant_id == 1
    assert payload.file_path == file_service.saved_path
    assert payload.file_name == "report.pdf"
    assert payload.file_type == "pdf"
    assert payload.enable_multimodel is False
    assert payload.enable_question_generation is False
    assert payload.language == "en-US"


async def test_create_from_file_custom_name_and_process_overrides() -> None:
    repo, rows = _make_repo()
    file_service = _FakeFileService()
    dispatcher = _FakeDispatcher()
    kb = _kb_info(
        chunking_config={"enable_multimodal": True},
        question_generation_config={"enabled": True, "question_count": 4},
    )
    created = await create_knowledge_from_file(
        tenant_id=1,
        knowledge_base_id="kb-1",
        file=_upload_file(filename="original.pdf"),
        knowledge_repo=repo,
        kb_service=_kb_service_mock(kb),
        file_service=file_service,
        dispatcher=dispatcher,
        custom_file_name="renamed.pdf",
        metadata={"owner": "finance"},
        process_overrides={"enable_multimodel": True},
    )

    stored = rows[created.id]
    assert stored.title == "renamed.pdf"
    assert stored.file_name == "renamed.pdf"
    assert stored.metadata == {"owner": "finance", "process_overrides": {"enable_multimodel": True}}
    assert dispatcher.payloads[0].enable_multimodel is True
    assert dispatcher.payloads[0].enable_question_generation is True
    assert dispatcher.payloads[0].question_count == 4


async def test_create_from_file_rejects_video() -> None:
    repo, _rows = _make_repo()
    with pytest.raises(ValidationError) as exc:
        await create_knowledge_from_file(
            tenant_id=1,
            knowledge_base_id="kb-1",
            file=_upload_file(filename="clip.mp4"),
            knowledge_repo=repo,
            kb_service=_kb_service_mock(_kb_info()),
            file_service=_FakeFileService(),
        )
    assert exc.value.code == "knowledge.video_not_supported"


async def test_create_from_file_rejects_faq_knowledge_base() -> None:
    repo, _rows = _make_repo()
    with pytest.raises(ValidationError) as exc:
        await create_knowledge_from_file(
            tenant_id=1,
            knowledge_base_id="kb-1",
            file=_upload_file(),
            knowledge_repo=repo,
            kb_service=_kb_service_mock(_kb_info(kb_type="faq")),
            file_service=_FakeFileService(),
        )
    assert exc.value.code == "knowledge.faq_file_upload_unsupported"


async def test_create_from_file_propagates_missing_knowledge_base() -> None:
    repo, _rows = _make_repo()
    kb_service = AsyncMock(spec=KBService)
    kb_service.get_knowledge_base_by_id.side_effect = NotFoundError(
        code="knowledge_base.not_found",
        message="knowledge base kb-1 not found",
    )
    with pytest.raises(NotFoundError):
        await create_knowledge_from_file(
            tenant_id=1,
            knowledge_base_id="kb-1",
            file=_upload_file(),
            knowledge_repo=repo,
            kb_service=kb_service,
            file_service=_FakeFileService(),
        )


async def test_create_from_file_rejects_unsupported_extension() -> None:
    repo, _rows = _make_repo()
    with pytest.raises(ValidationError) as exc:
        await create_knowledge_from_file(
            tenant_id=1,
            knowledge_base_id="kb-1",
            file=_upload_file(filename="script.exe"),
            knowledge_repo=repo,
            kb_service=_kb_service_mock(_kb_info()),
            file_service=_FakeFileService(),
        )
    assert exc.value.code == "knowledge.unsupported_file_type"


async def test_create_from_file_requires_storage_engine() -> None:
    repo, _rows = _make_repo()

    class _NoStorage:
        async def resolve_file_service(
            self, *, knowledge_base_id: str, tenant_id: int
        ) -> object:
            return None

    with pytest.raises(ValidationError) as exc:
        await create_knowledge_from_file(
            tenant_id=1,
            knowledge_base_id="kb-1",
            file=_upload_file(),
            knowledge_repo=repo,
            kb_service=_kb_service_mock(_kb_info()),
            storage_resolver=_NoStorage(),  # type: ignore[arg-type]
        )
    assert exc.value.code == "knowledge.storage_engine_required"


async def test_create_from_file_duplicate_raises_and_refreshes_created_at() -> None:
    repo, rows = _make_repo()
    hash_value = calculate_file_hash(_PDF_BYTES)
    existing = _sample_doc(
        id="doc-dup",
        tenant_id=1,
        knowledge_base_id="kb-1",
        file_name="report.pdf",
        file_type="pdf",
        file_hash=hash_value,
    )
    rows[existing.id] = existing
    created_at_before = existing.created_at

    with pytest.raises(ConflictError) as exc:
        await create_knowledge_from_file(
            tenant_id=1,
            knowledge_base_id="kb-1",
            file=_upload_file(filename="report.pdf"),
            knowledge_repo=repo,
            kb_service=_kb_service_mock(_kb_info()),
            file_service=_FakeFileService(),
        )
    assert exc.value.code == "knowledge.duplicate_file"
    assert exc.value.details == {"knowledge_id": "doc-dup"}
    assert rows["doc-dup"].created_at >= created_at_before
    # No new row is created for a duplicate.
    assert set(rows) == {"doc-dup"}


async def test_create_from_file_duplicate_respects_file_type() -> None:
    repo, rows = _make_repo()
    hash_value = calculate_file_hash(_PDF_BYTES)
    existing = _sample_doc(
        id="doc-md",
        tenant_id=1,
        knowledge_base_id="kb-1",
        file_name="report.md",
        file_type="md",
        file_hash=hash_value,
    )
    rows[existing.id] = existing
    created = await create_knowledge_from_file(
        tenant_id=1,
        knowledge_base_id="kb-1",
        file=_upload_file(filename="report.txt"),
        knowledge_repo=repo,
        kb_service=_kb_service_mock(_kb_info()),
        file_service=_FakeFileService(),
    )
    assert created.file_type == "txt"


async def test_create_from_file_storage_quota_gate() -> None:
    repo, _rows = _make_repo()
    with pytest.raises(ValidationError) as exc:
        await create_knowledge_from_file(
            tenant_id=1,
            knowledge_base_id="kb-1",
            file=_upload_file(),
            knowledge_repo=repo,
            kb_service=_kb_service_mock(_kb_info()),
            file_service=_FakeFileService(),
            tenant_storage=TenantStorageInfo(storage_quota=100, storage_used=100),
        )
    assert exc.value.code == "knowledge.storage_quota_exceeded"

    # Quota not exceeded does not block.
    created = await create_knowledge_from_file(
        tenant_id=1,
        knowledge_base_id="kb-1",
        file=_upload_file(),
        knowledge_repo=repo,
        kb_service=_kb_service_mock(_kb_info()),
        file_service=_FakeFileService(),
        tenant_storage=TenantStorageInfo(storage_quota=100, storage_used=50),
    )
    assert created.id


async def test_create_from_file_rejects_invalid_file_name() -> None:
    repo, _rows = _make_repo()
    with pytest.raises(ValidationError) as exc:
        await create_knowledge_from_file(
            tenant_id=1,
            knowledge_base_id="kb-1",
            file=_upload_file(filename="bad\nname.pdf"),
            knowledge_repo=repo,
            kb_service=_kb_service_mock(_kb_info()),
            file_service=_FakeFileService(),
        )
    assert exc.value.code == "knowledge.invalid_file_name"


async def test_create_from_file_requires_vlm_for_images() -> None:
    repo, _rows = _make_repo()
    kb = _kb_info(vlm_config={"enabled": False})
    with pytest.raises(ValidationError) as exc:
        await create_knowledge_from_file(
            tenant_id=1,
            knowledge_base_id="kb-1",
            file=_FakeFile(filename="pic.png", data=b"png-data", content_type="image/png"),
            knowledge_repo=repo,
            kb_service=_kb_service_mock(kb),
            file_service=_FakeFileService(),
        )
    assert exc.value.code == "knowledge.vlm_required"


async def test_create_from_file_requires_asr_for_audio() -> None:
    repo, _rows = _make_repo()
    kb = _kb_info(asr_config={"enabled": False})
    with pytest.raises(ValidationError) as exc:
        await create_knowledge_from_file(
            tenant_id=1,
            knowledge_base_id="kb-1",
            file=_FakeFile(filename="clip.mp3", data=b"mp3-data", content_type="audio/mpeg"),
            knowledge_repo=repo,
            kb_service=_kb_service_mock(kb),
            file_service=_FakeFileService(),
        )
    assert exc.value.code == "knowledge.asr_required"


async def test_create_from_file_marks_failed_on_dispatch_error() -> None:
    repo, rows = _make_repo()
    created = await create_knowledge_from_file(
        tenant_id=1,
        knowledge_base_id="kb-1",
        file=_upload_file(),
        knowledge_repo=repo,
        kb_service=_kb_service_mock(_kb_info()),
        file_service=_FakeFileService(),
        dispatcher=_FakeDispatcher(fail=True),
    )
    assert created.id
    stored = rows[created.id]
    assert stored.parse_status == PARSE_STATUS_FAILED
    assert stored.error_message == "Failed to enqueue processing task"


async def test_create_from_file_without_dispatcher_stays_pending() -> None:
    repo, rows = _make_repo()
    created = await create_knowledge_from_file(
        tenant_id=1,
        knowledge_base_id="kb-1",
        file=_upload_file(),
        knowledge_repo=repo,
        kb_service=_kb_service_mock(_kb_info()),
        file_service=_FakeFileService(),
    )
    assert rows[created.id].parse_status == PARSE_STATUS_PENDING


async def test_create_from_file_cleans_up_file_when_insert_fails() -> None:
    repo, _rows = _make_repo()

    async def _fail_create(row: Document) -> Document:
        raise RuntimeError("insert failed")

    repo.create.side_effect = _fail_create
    file_service = _FakeFileService()
    with pytest.raises(RuntimeError, match="insert failed"):
        await create_knowledge_from_file(
            tenant_id=1,
            knowledge_base_id="kb-1",
            file=_upload_file(),
            knowledge_repo=repo,
            kb_service=_kb_service_mock(_kb_info()),
            file_service=file_service,
        )
    assert file_service.deleted == [file_service.saved_path]


async def test_create_from_file_attaches_tags() -> None:
    repo, _rows = _make_repo()
    tag_service = AsyncMock(spec=TagService)
    created = await create_knowledge_from_file(
        tenant_id=1,
        knowledge_base_id="kb-1",
        file=_upload_file(),
        knowledge_repo=repo,
        kb_service=_kb_service_mock(_kb_info()),
        file_service=_FakeFileService(),
        tag_service=tag_service,
        tag_ids=["tag-1", "", "tag-2"],
    )
    tag_service.set_knowledge_tags.assert_awaited_once_with(
        knowledge_id=created.id,
        tag_ids=["tag-1", "tag-2"],
    )


async def test_create_from_file_defaults_channel_to_web() -> None:
    repo, rows = _make_repo()
    created = await create_knowledge_from_file(
        tenant_id=1,
        knowledge_base_id="kb-1",
        file=_upload_file(),
        knowledge_repo=repo,
        kb_service=_kb_service_mock(_kb_info()),
        file_service=_FakeFileService(),
    )
    assert rows[created.id].channel == CHANNEL_WEB


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


async def _seed_kb(session: AsyncSession, tenant_id: int) -> KnowledgeBaseInfo:
    kb_service = KBService(kb_repo=KnowledgeBaseRepository(session))
    return await kb_service.create_knowledge_base(
        tenant_id=tenant_id,
        name="Integration KB",
        embedding_model_id="emb-integration",
    )


async def test_integration_create_from_file_round_trip(
    session: AsyncSession, tmp_path: Path
) -> None:
    tenant_id = make_test_tenant_id()
    kb = await _seed_kb(session, tenant_id)
    file_service = LocalStorageAdapter(path_prefix="", base_dir=str(tmp_path))

    created = await create_knowledge_from_file(
        tenant_id=tenant_id,
        knowledge_base_id=kb.id,
        file=_upload_file(filename="integration.pdf"),
        knowledge_repo=KnowledgeRepository(session),
        kb_service=KBService(kb_repo=KnowledgeBaseRepository(session)),
        file_service=file_service,
        dispatcher=_FakeDispatcher(),
    )

    assert created.id
    assert created.tenant_id == tenant_id
    assert created.knowledge_base_id == kb.id
    assert created.title == "integration.pdf"
    assert created.file_type == "pdf"
    assert created.parse_status == PARSE_STATUS_PENDING
    assert created.embedding_model_id == "emb-integration"

    row = await KnowledgeRepository(session).get_by_id(tenant_id=tenant_id, id=created.id)
    assert row is not None
    assert row.file_name == "integration.pdf"
    assert row.file_hash == calculate_file_hash(_PDF_BYTES)
    assert row.file_path and row.file_path.startswith("local://")

    # The object is really on disk under {base}/{tenant}/{knowledge_id}/.
    stored_dir = tmp_path / str(tenant_id) / created.id
    objects = [p for p in stored_dir.iterdir() if p.is_file()]
    assert len(objects) == 1
    assert objects[0].read_bytes() == _PDF_BYTES


async def test_integration_create_from_file_duplicate(
    session: AsyncSession, tmp_path: Path
) -> None:
    tenant_id = make_test_tenant_id()
    kb = await _seed_kb(session, tenant_id)
    file_service = LocalStorageAdapter(path_prefix="", base_dir=str(tmp_path))

    created = await create_knowledge_from_file(
        tenant_id=tenant_id,
        knowledge_base_id=kb.id,
        file=_upload_file(filename="dup.pdf"),
        knowledge_repo=KnowledgeRepository(session),
        kb_service=KBService(kb_repo=KnowledgeBaseRepository(session)),
        file_service=file_service,
    )
    assert created.id

    with pytest.raises(ConflictError) as exc:
        await create_knowledge_from_file(
            tenant_id=tenant_id,
            knowledge_base_id=kb.id,
            file=_upload_file(filename="dup.pdf"),
            knowledge_repo=KnowledgeRepository(session),
            kb_service=KBService(kb_repo=KnowledgeBaseRepository(session)),
            file_service=file_service,
        )
    assert exc.value.code == "knowledge.duplicate_file"
    assert exc.value.details == {"knowledge_id": created.id}

    service = KnowledgeService(knowledge_repo=KnowledgeRepository(session))
    listed = await service.list_documents(tenant_id=tenant_id, knowledge_base_id=kb.id)
    assert len(listed) == 1
