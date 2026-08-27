"""Unit and integration tests for the FAQ import pipeline.

The unit section drives the parser with real CSV bytes and the import
orchestration against spec'd ``AsyncMock`` repositories with
closure-captured state, so it runs without Postgres. The integration
section runs ``import_faq`` against the real applied schema and skips
when the database is unreachable.

``chunks.tenant_id`` is a 32-bit INTEGER, so integration rows use an
int32-safe local counter rather than the 64-bit tenant factory.
"""

# Chinese test data uses fullwidth punctuation.

from __future__ import annotations

import itertools
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from faker import Faker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.pool import NullPool

from src.common.exception import ValidationError
from src.core.knowledge.documents.faq_import import (
    FAQ_BATCH_MODE_APPEND,
    FAQ_BATCH_MODE_REPLACE,
    FAQImportResult,
    build_faq_chunk_content,
    import_faq,
)
from src.core.knowledge.faq import import_parser as parser_module
from src.core.knowledge.faq.import_parser import (
    IMPORT_HEADERS,
    parse_csv,
    parse_excel,
    parse_import_file,
)
from src.core.knowledge.faq.types import FAQContent
from src.db.base import DatabaseEngine
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.faq_repository import FaqRepository
from src.db.models.chunk import Chunk
from src.db.models.faq import Faq
from src.settings import get_settings, reset_settings_cache
from tests.integration.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_HEADER = ",".join(IMPORT_HEADERS)

# ``chunks.tenant_id`` is a 32-bit INTEGER column; integration rows need a
# 32-bit-safe unique id (the tenants table's 64-bit ids do not fit).
_tenant_counter = itertools.count(3_000_000)


def _tenant_id() -> int:
    """Return a unique 32-bit tenant id for the ``chunks`` table."""
    return next(_tenant_counter)


def _csv(*rows: str) -> bytes:
    """Build UTF-8 CSV bytes from the template header plus data rows."""
    return ("\n".join([_HEADER, *rows]) + "\n").encode("utf-8")


# One canonical valid row (tag, question, similar, negative, answers,
# answer-all, disabled, disable-recommend).
_R1 = "账户,如何充值？,怎么充值##充值方法,,进入设置-账户-充值,true,false,false"
_R2 = "账户,如何退款？,,退款是充值吗,联系客服,false,true,true"


# ── Parser: format and structure ──────────────────────────────────────


def test_parse_csv_maps_every_template_column() -> None:
    parsed = parse_csv(_csv(_R1, _R2))
    assert parsed.total == 2
    assert parsed.skipped_rows == 0
    assert parsed.errors == []

    first, second = parsed.entries
    assert first.row_number == 2
    assert first.payload.standard_question == "如何充值？"
    assert first.payload.similar_questions == ["怎么充值", "充值方法"]
    assert first.payload.negative_questions == []
    assert first.payload.answers == ["进入设置-账户-充值"]
    assert first.payload.answer_strategy == "all"
    assert first.payload.tag_name == "账户"
    assert first.payload.is_enabled is True
    assert first.payload.is_recommended is True

    assert second.row_number == 3
    assert second.payload.answer_strategy == "random"
    assert second.payload.is_enabled is False
    assert second.payload.is_recommended is False


def test_parse_csv_tolerates_bom_and_empty_toggles() -> None:
    data = "﻿" + _csv(_R1).decode("utf-8")
    parsed = parse_csv(data.encode("utf-8"))
    assert parsed.total == 1
    # Unspecified toggles keep their defaults.
    assert parsed.entries[0].payload.answer_strategy == "all"


def test_parse_csv_skips_blank_data_rows() -> None:
    parsed = parse_csv(_csv(_R1, "", _R2))
    assert parsed.total == 2
    assert parsed.skipped_rows == 1
    assert [e.row_number for e in parsed.entries] == [2, 4]


def test_parse_csv_rejects_mismatched_header() -> None:
    data = "问题,答案\n如何充值？,进入设置\n".encode()
    with pytest.raises(ValidationError) as excinfo:
        parse_csv(data)
    assert excinfo.value.code == "faq.import_invalid_header"


def test_parse_csv_accepts_header_without_optional_toggles() -> None:
    header = ",".join(IMPORT_HEADERS[:6])
    data = f"{header}\n账户,如何充值？,,,进入设置,true\n".encode()
    parsed = parse_csv(data)
    assert parsed.total == 1
    assert parsed.entries[0].payload.is_enabled is True


def test_parse_csv_rejects_too_few_columns() -> None:
    data = f"{_HEADER}\n账户,如何充值？,怎么充值\n".encode()
    parsed = parse_csv(data)
    assert parsed.total == 1
    assert parsed.entries == []
    assert parsed.errors[0].code == "faq.import_invalid_row"
    assert parsed.errors[0].row_number == 2


def test_parse_csv_rejects_invalid_boolean_token() -> None:
    data = f"{_HEADER}\n账户,如何充值？,,,进入设置,maybe\n".encode()
    parsed = parse_csv(data)
    assert parsed.entries == []
    assert parsed.errors[0].code == "faq.import_invalid_boolean"


def test_parse_csv_handles_quoted_fields_with_commas() -> None:
    data = (f'{_HEADER}\n"客服,账户",如何充值？,,,"进入设置,然后充值",true,false,false\n').encode()
    parsed = parse_csv(data)
    assert parsed.total == 1
    assert parsed.entries[0].payload.tag_name == "客服,账户"
    assert parsed.entries[0].payload.answers == ["进入设置,然后充值"]


def test_parse_csv_reports_non_utf8_encoding() -> None:
    with pytest.raises(ValidationError) as excinfo:
        parse_csv(b"\xff\xfe\x00garbage")
    assert excinfo.value.code == "faq.import_decode_error"


def test_parse_csv_empty_file_has_no_rows() -> None:
    parsed = parse_csv(_csv())
    assert parsed.total == 0
    assert parsed.entries == []


def test_parse_csv_entirely_empty_is_an_error() -> None:
    with pytest.raises(ValidationError) as excinfo:
        parse_csv(b"\r\n\r\n")
    assert excinfo.value.code == "faq.import_empty"


def test_parse_import_file_dispatch_by_extension() -> None:
    assert parse_import_file(_csv(_R1), filename="faq.csv").total == 1
    assert parse_import_file(_csv(_R1), filename="faq.CSV").total == 1


def test_parse_import_file_rejects_unknown_extension() -> None:
    with pytest.raises(ValidationError) as excinfo:
        parse_import_file(b"data", filename="faq.txt")
    assert excinfo.value.code == "faq.import_unsupported_file"


# ── Parser: Excel path ────────────────────────────────────────────────


class _FakeSheet:
    """Stands in for an ``openpyxl`` worksheet (values-only row stream)."""

    def iter_rows(self, *, values_only: bool = True) -> object:
        del values_only
        rows: list[tuple[object, ...]] = [
            tuple(IMPORT_HEADERS),
            ("账户", "如何充值？", "怎么充值", "", "进入设置", True, "", ""),
            ("账户", "如何退款？", "", "退款是充值吗", "联系客服", False, True, True),
        ]
        return iter(rows)


class _FakeWorkbook:
    """Stands in for an ``openpyxl`` workbook whose first sheet is used."""

    active = _FakeSheet()


def test_parse_excel_reads_first_sheet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parser_module, "_load_workbook", lambda _data: _FakeWorkbook())
    parsed = parse_excel(b"fake-xlsx-bytes")
    assert parsed.total == 2
    assert parsed.entries[0].payload.standard_question == "如何充值？"
    assert parsed.entries[1].payload.answer_strategy == "random"


def test_parse_import_file_dispatches_excel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parser_module, "_load_workbook", lambda _data: _FakeWorkbook())
    parsed = parse_import_file(b"fake-xlsx-bytes", filename="faq.xlsx")
    assert parsed.total == 2


def test_parse_excel_reports_missing_openpyxl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _missing(_name: str) -> object:
        raise ImportError("No module named 'openpyxl'")

    monkeypatch.setattr("importlib.import_module", _missing)
    with pytest.raises(ValidationError) as excinfo:
        parse_excel(b"fake-xlsx-bytes")
    assert excinfo.value.code == "faq.import_excel_unsupported"


# ── Chunk content builder ─────────────────────────────────────────────


def test_build_faq_chunk_content_question_only_mode() -> None:
    content = FAQContent(
        standard_question="如何充值？",
        similar_questions=["怎么充值"],
        negative_questions=["如何退款？"],
        answers=["进入设置"],
    )
    text = build_faq_chunk_content(content, index_mode="question_only")
    assert text == "Q: 如何充值？\nSimilar Questions:\n- 怎么充值"
    # Negative questions and answers never enter the index content.
    assert "如何退款" not in text
    assert "进入设置" not in text


def test_build_faq_chunk_content_question_answer_mode_includes_answers() -> None:
    content = FAQContent(
        standard_question="如何充值？",
        similar_questions=[],
        negative_questions=[],
        answers=["进入设置", "联系客服"],
    )
    text = build_faq_chunk_content(content, index_mode="question_answer")
    assert text == "Q: 如何充值？\nAnswers:\n- 进入设置\n- 联系客服"


def test_build_faq_chunk_content_defaults_to_question_only() -> None:
    content = FAQContent(standard_question="如何充值？", answers=["进入设置"])
    assert build_faq_chunk_content(content, index_mode=None) == "Q: 如何充值？"


# ── Mock repositories ─────────────────────────────────────────────────


def _mock_faq_repo(*, existing: Faq | None = None) -> tuple[AsyncMock, list[Faq]]:
    """A spec'd ``FaqRepository`` that assigns ids and records inserted rows."""
    repo = AsyncMock(spec=FaqRepository)
    stored: list[Faq] = []
    id_counter = itertools.count(100)

    async def _create(row: Faq) -> Faq:
        persisted = row.model_copy(update={"id": next(id_counter)})
        stored.append(persisted)
        return persisted

    repo.create.side_effect = _create
    repo.find_duplicate_question.return_value = existing
    return repo, stored


def _mock_chunk_repo() -> tuple[AsyncMock, list[Chunk]]:
    """A spec'd ``ChunkRepository`` that assigns seq ids and records rows."""
    repo = AsyncMock(spec=ChunkRepository)
    stored: list[Chunk] = []

    async def _create_many(chunks: list[Chunk]) -> list[Chunk]:
        persisted = [
            chunk.model_copy(update={"seq_id": i}) for i, chunk in enumerate(chunks, start=1)
        ]
        stored.extend(persisted)
        return persisted

    repo.create_many.side_effect = _create_many
    return repo, stored


def _existing_row(
    *,
    tenant_id: int,
    standard_question: str = "如何充值？",
) -> Faq:
    return Faq(
        id=1,
        tenant_id=tenant_id,
        chunk_id="chunk-existing",
        knowledge_id="knowledge-1",
        knowledge_base_id="kb-1",
        tag_name="账户",
        is_enabled=True,
        is_recommended=False,
        standard_question=standard_question,
        similar_questions=[],
        negative_questions=[],
        answers=["进入设置"],
        answer_strategy="all",
        index_mode=None,
        chunk_type="faq",
        created_at=_NOW,
        updated_at=_NOW,
    )


async def _run_import(
    *,
    data: bytes,
    faq_repo: AsyncMock,
    chunk_repo: AsyncMock,
    mode: str = FAQ_BATCH_MODE_APPEND,
    dry_run: bool = False,
    tenant_id: int | None = None,
) -> FAQImportResult:
    return await import_faq(
        file_data=data,
        filename="faq.csv",
        tenant_id=tenant_id if tenant_id is not None else 7,
        knowledge_base_id="kb-1",
        knowledge_id="knowledge-1",
        faq_repo=faq_repo,
        chunk_repo=chunk_repo,
        mode=mode,
        dry_run=dry_run,
    )


# ── Import pipeline: happy paths ──────────────────────────────────────


async def test_import_faq_append_persists_entries_and_links_chunks() -> None:
    faq_repo, faq_stored = _mock_faq_repo()
    chunk_repo, chunk_stored = _mock_chunk_repo()

    result = await _run_import(data=_csv(_R1, _R2), faq_repo=faq_repo, chunk_repo=chunk_repo)

    assert result.mode == FAQ_BATCH_MODE_APPEND
    assert result.total == 2
    assert result.success_count == 2
    assert result.added_count == 2
    assert result.failed_count == 0
    assert result.skipped_count == 0
    assert len(faq_stored) == 2
    assert len(chunk_stored) == 2

    # Each faq row is linked to a FAQ chunk by a shared chunk id.
    for entry, faq_row, chunk_row in zip(
        result.success_entries,
        faq_stored,
        chunk_stored,
        strict=True,
    ):
        assert faq_row.chunk_id == chunk_row.id
        assert entry.chunk_id == chunk_row.id
        assert entry.id == faq_row.id
        assert entry.seq_id == chunk_row.seq_id
        assert chunk_row.chunk_type == "faq"
        assert chunk_row.metadata is not None
        assert chunk_row.metadata["standard_question"] == faq_row.standard_question
        assert chunk_row.content.startswith("Q: ")


async def test_import_faq_uses_question_only_index_content_by_default() -> None:
    faq_repo, _ = _mock_faq_repo()
    chunk_repo, chunk_stored = _mock_chunk_repo()

    await _run_import(data=_csv(_R1), faq_repo=faq_repo, chunk_repo=chunk_repo)

    assert chunk_stored[0].content == ("Q: 如何充值？\nSimilar Questions:\n- 怎么充值\n- 充值方法")
    assert "进入设置" not in chunk_stored[0].content


async def test_import_faq_replace_skips_cross_entry_duplicate_check() -> None:
    existing = _existing_row(tenant_id=7)
    faq_repo, _ = _mock_faq_repo(existing=existing)
    chunk_repo, _ = _mock_chunk_repo()

    result = await _run_import(
        data=_csv(_R1),
        faq_repo=faq_repo,
        chunk_repo=chunk_repo,
        mode=FAQ_BATCH_MODE_REPLACE,
    )

    assert result.success_count == 1
    faq_repo.find_duplicate_question.assert_not_awaited()


async def test_import_faq_with_generated_question() -> None:
    Faker.seed(0)
    question = f"如何{Faker().word()}？"
    data = _csv(f"账户,{question},,,进入设置,true,false,false")
    faq_repo, faq_stored = _mock_faq_repo()
    chunk_repo, _ = _mock_chunk_repo()

    result = await _run_import(data=data, faq_repo=faq_repo, chunk_repo=chunk_repo)

    assert result.success_count == 1
    assert faq_stored[0].standard_question == question


async def test_import_faq_propagates_tenant_scope() -> None:
    tenant_id = make_test_tenant_id()
    faq_repo, faq_stored = _mock_faq_repo()
    chunk_repo, chunk_stored = _mock_chunk_repo()

    result = await _run_import(
        data=_csv(_R1),
        faq_repo=faq_repo,
        chunk_repo=chunk_repo,
        tenant_id=tenant_id,
    )

    assert result.success_count == 1
    assert faq_stored[0].tenant_id == tenant_id
    assert chunk_stored[0].tenant_id == tenant_id


# ── Import pipeline: validation failures ──────────────────────────────


async def test_import_faq_rejects_invalid_mode() -> None:
    faq_repo, _ = _mock_faq_repo()
    chunk_repo, _ = _mock_chunk_repo()
    with pytest.raises(ValidationError) as excinfo:
        await _run_import(data=_csv(_R1), faq_repo=faq_repo, chunk_repo=chunk_repo, mode="merge")
    assert excinfo.value.code == "faq.invalid_import_mode"


async def test_import_faq_rejects_empty_entry_set() -> None:
    faq_repo, _ = _mock_faq_repo()
    chunk_repo, _ = _mock_chunk_repo()
    with pytest.raises(ValidationError) as excinfo:
        await _run_import(data=_csv(), faq_repo=faq_repo, chunk_repo=chunk_repo)
    assert excinfo.value.code == "faq.entries_required"


async def test_import_faq_reports_structural_errors_as_failed() -> None:
    data = _csv(_R1, "账户,如何充值？,,,进入设置,maybe")
    faq_repo, faq_stored = _mock_faq_repo()
    chunk_repo, chunk_stored = _mock_chunk_repo()

    result = await _run_import(data=data, faq_repo=faq_repo, chunk_repo=chunk_repo)

    assert result.total == 2
    assert result.success_count == 1
    assert result.failed_count == 1
    assert result.failed_entries[0].code == "faq.import_invalid_boolean"
    assert result.failed_entries[0].row_number == 3
    assert len(faq_stored) == 1
    assert len(chunk_stored) == 1


async def test_import_faq_rejects_duplicate_standard_within_batch() -> None:
    data = _csv(_R1, _R1)
    faq_repo, faq_stored = _mock_faq_repo()
    chunk_repo, _ = _mock_chunk_repo()

    result = await _run_import(data=data, faq_repo=faq_repo, chunk_repo=chunk_repo)

    assert result.success_count == 1
    assert result.failed_count == 1
    assert result.failed_entries[0].code == "faq.duplicate_in_batch"
    assert len(faq_stored) == 1


async def test_import_faq_rejects_similar_colliding_with_earlier_row() -> None:
    # The second row's similar question equals the first row's standard.
    data = _csv(_R1, "账户,如何退款？,如何充值？,,联系客服,false,false,false")
    faq_repo, _ = _mock_faq_repo()
    chunk_repo, _ = _mock_chunk_repo()

    result = await _run_import(data=data, faq_repo=faq_repo, chunk_repo=chunk_repo)

    assert result.success_count == 1
    assert result.failed_count == 1
    assert result.failed_entries[0].code == "faq.duplicate_in_batch"


async def test_import_faq_rejects_semantically_invalid_entry() -> None:
    # Empty standard question is a content error, reported per-row.
    data = _csv("账户,,怎么充值,,进入设置,true,false,false", _R2)
    faq_repo, faq_stored = _mock_faq_repo()
    chunk_repo, _ = _mock_chunk_repo()

    result = await _run_import(data=data, faq_repo=faq_repo, chunk_repo=chunk_repo)

    assert result.success_count == 1
    assert result.failed_count == 1
    assert result.failed_entries[0].code == "faq.standard_question_required"
    assert len(faq_stored) == 1


async def test_import_faq_reports_cross_entry_duplicate_in_append() -> None:
    existing = _existing_row(tenant_id=7)
    faq_repo, faq_stored = _mock_faq_repo(existing=existing)
    chunk_repo, _ = _mock_chunk_repo()

    result = await _run_import(data=_csv(_R1), faq_repo=faq_repo, chunk_repo=chunk_repo)

    assert result.success_count == 0
    assert result.failed_count == 1
    assert result.failed_entries[0].code == "faq.duplicate_question"
    assert len(faq_stored) == 0


async def test_import_faq_dry_run_validates_without_persisting() -> None:
    faq_repo, faq_stored = _mock_faq_repo()
    chunk_repo, chunk_stored = _mock_chunk_repo()

    result = await _run_import(
        data=_csv(_R1, _R2),
        faq_repo=faq_repo,
        chunk_repo=chunk_repo,
        dry_run=True,
    )

    assert result.success_count == 2
    assert result.added_count == 0
    assert result.failed_count == 0
    assert result.success_entries == []
    assert faq_stored == []
    assert chunk_stored == []
    faq_repo.create.assert_not_awaited()
    chunk_repo.create_many.assert_not_awaited()


async def test_import_faq_all_failed_returns_empty_result() -> None:
    faq_repo, faq_stored = _mock_faq_repo()
    chunk_repo, chunk_stored = _mock_chunk_repo()
    data = _csv("账户,如何充值？,,,进入设置,maybe")

    result = await _run_import(data=data, faq_repo=faq_repo, chunk_repo=chunk_repo)

    assert result.success_count == 0
    assert result.failed_count == 1
    assert faq_stored == []
    assert chunk_stored == []
    chunk_repo.create_many.assert_not_awaited()


async def test_import_faq_result_counts_are_consistent() -> None:
    data = _csv(_R1, _R1, _R2)
    faq_repo, _ = _mock_faq_repo()
    chunk_repo, _ = _mock_chunk_repo()

    result = await _run_import(data=data, faq_repo=faq_repo, chunk_repo=chunk_repo)

    assert result.total == 3
    assert result.success_count + result.failed_count == result.total
    assert result.failed_count == 1
    assert result.failed_entries[0].code == "faq.duplicate_in_batch"


# ── Integration against the real schema ───────────────────────────────


@pytest_asyncio.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    """Per-test session against the real applied schema; skips without a DB."""
    reset_settings_cache()
    engine = DatabaseEngine(url=get_settings().database_url, poolclass=NullPool)
    try:
        await engine.prewarm()
    except Exception as exc:
        await engine.close()
        pytest.skip(f"integration database unavailable: {exc}")
    async with engine.session_factory() as s:
        yield s
        await s.rollback()
    await engine.close()


class TestFaqImportIntegration:
    async def test_import_persists_faq_rows_and_linked_faq_chunks(
        self,
        db_session: AsyncSession,
    ) -> None:
        tenant_id = _tenant_id()
        kb_id = f"kb-import-{tenant_id}"
        knowledge_id = f"knowledge-import-{tenant_id}"
        faq_repo = FaqRepository(db_session)
        chunk_repo = ChunkRepository(db_session)

        result = await import_faq(
            file_data=_csv(_R1, _R2),
            filename="faq.csv",
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            knowledge_id=knowledge_id,
            faq_repo=faq_repo,
            chunk_repo=chunk_repo,
        )

        assert result.success_count == 2
        rows, total = await faq_repo.list_by_knowledge_base(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            limit=10,
            offset=0,
        )
        assert total == 2
        assert {r.standard_question for r in rows} == {"如何充值？", "如何退款？"}

        for entry, row in zip(result.success_entries, rows, strict=True):
            assert entry.id == row.id
            assert entry.chunk_id == row.chunk_id
            chunk = await chunk_repo.get_by_id(tenant_id, row.chunk_id)
            assert chunk.chunk_type == "faq"
            assert chunk.metadata is not None
            assert chunk.metadata["standard_question"] == row.standard_question
            assert chunk.content.startswith("Q: ")
            assert chunk.tenant_id == tenant_id
            assert chunk.status == 1

    async def test_import_append_reports_duplicate_against_existing_row(
        self,
        db_session: AsyncSession,
    ) -> None:
        tenant_id = _tenant_id()
        kb_id = f"kb-import-dup-{tenant_id}"
        knowledge_id = f"knowledge-import-dup-{tenant_id}"
        faq_repo = FaqRepository(db_session)
        chunk_repo = ChunkRepository(db_session)
        existing = await faq_repo.create(
            _existing_row(
                tenant_id=tenant_id,
                standard_question="如何退款？",
            ).model_copy(
                update={
                    "chunk_id": f"chunk-seed-{tenant_id}",
                    "knowledge_base_id": kb_id,
                    "knowledge_id": knowledge_id,
                }
            )
        )

        result = await import_faq(
            file_data=_csv(_R1, _R2),
            filename="faq.csv",
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            knowledge_id=knowledge_id,
            faq_repo=faq_repo,
            chunk_repo=chunk_repo,
        )

        assert existing.id > 0
        # The colliding entry is rejected; the other one imports.
        assert result.success_count == 1
        assert result.failed_count == 1
        assert result.failed_entries[0].code == "faq.duplicate_question"
        assert result.failed_entries[0].standard_question == "如何退款？"

        _rows, total = await faq_repo.list_by_knowledge_base(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            limit=10,
            offset=0,
        )
        assert total == 2  # the seed plus the one imported entry

    async def test_import_replace_ignores_existing_rows(
        self,
        db_session: AsyncSession,
    ) -> None:
        tenant_id = _tenant_id()
        kb_id = f"kb-import-replace-{tenant_id}"
        knowledge_id = f"knowledge-import-replace-{tenant_id}"
        faq_repo = FaqRepository(db_session)
        chunk_repo = ChunkRepository(db_session)
        await faq_repo.create(
            _existing_row(tenant_id=tenant_id).model_copy(
                update={
                    "chunk_id": f"chunk-seed-{tenant_id}",
                    "knowledge_base_id": kb_id,
                    "knowledge_id": knowledge_id,
                }
            )
        )

        result = await import_faq(
            file_data=_csv(_R1),
            filename="faq.csv",
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            knowledge_id=knowledge_id,
            faq_repo=faq_repo,
            chunk_repo=chunk_repo,
            mode=FAQ_BATCH_MODE_REPLACE,
        )

        assert result.success_count == 1
        assert result.failed_count == 0
        _rows, total = await faq_repo.list_by_knowledge_base(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            limit=10,
            offset=0,
        )
        # Replace mode leaves existing rows in place (diffing is deferred).
        assert total == 2
