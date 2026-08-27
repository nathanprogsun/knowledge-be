"""FAQ batch import orchestration.

``import_faq`` parses an uploaded CSV / Excel file into FAQ entries,
validates the batch (per-entry content rules, within-file question
collisions, and — in append mode — collisions with entries already in
the knowledge base), and persists the valid entries. Each entry becomes
one ``faq`` row plus one ``chunks`` row of FAQ type linked by a shared
``chunk_id``; the chunk carries the sanitised content metadata and the
indexable question/answer content so a later retrieval wave can rebuild
the vector index from it.

The pipeline is a standalone module: repositories are injected so the web
layer wires the real ones behind a transaction and unit tests drive the
module with fakes. ``import_faq`` is an async function that performs the
validation and persistence itself; running it on a background task queue
(e.g. for very large files) is the caller's concern. ``dry_run`` reuses
the same validation pass but returns counts without persisting, matching
the dry-run mode of the FAQ import contract.

Scope notes
-----------

- ``replace`` mode re-validates the same way as ``append`` but skips the
  cross-entry duplicate scan; diffing an existing knowledge base against
  the file (deleting the removed set) is deferred to the FAQ
  synchronization wave that also owns tag resolution.
- A within-file question collision (an entry whose standard or similar
  question matches an earlier row) rejects the whole entry; finer-grained
  partial-removal of just the colliding alias is deferred.
- ``chunks.tenant_id`` is a 32-bit INTEGER column, so the caller must
  pass a tenant id that fits that range when persisting.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from src.common.exception import ValidationError
from src.common.json import JsonObject
from src.core.contracts.knowledge import FAQEntryPayload
from src.core.knowledge.chunks.types import (
    CHUNK_FLAG_RECOMMENDED,
    CHUNK_STATUS_STORED,
    CHUNK_TYPE_FAQ,
)
from src.core.knowledge.documents.faq_ops import build_faq_row
from src.core.knowledge.faq.import_parser import (
    ImportRowError,
    parse_import_file,
)
from src.core.knowledge.faq.types import (
    ANSWER_STRATEGY_ALL,
    FAQ_INDEX_MODE_QUESTION_ANSWER,
    FAQContent,
    sanitize_faq_content,
)
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.faq_repository import FaqRepository
from src.db.models.chunk import Chunk
from src.db.models.faq import Faq

FAQ_BATCH_MODE_APPEND = "append"
FAQ_BATCH_MODE_REPLACE = "replace"

_IMPORT_MODES: frozenset[str] = frozenset({FAQ_BATCH_MODE_APPEND, FAQ_BATCH_MODE_REPLACE})


@dataclass(frozen=True)
class ImportedEntry:
    """One successfully imported entry and its persisted identity."""

    row_number: int
    id: int
    chunk_id: str
    seq_id: int
    tag_name: str | None
    standard_question: str


@dataclass(frozen=True)
class FailedEntry:
    """One rejected row plus the reason it did not import.

    The question / answer fields mirror the original row so a caller can
    render the failure back to the user (or export it to a CSV).
    """

    row_number: int
    code: str
    reason: str
    tag_name: str | None
    standard_question: str
    similar_questions: list[str] = field(default_factory=list)
    negative_questions: list[str] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    answer_all: bool = False
    is_disabled: bool = False


@dataclass(frozen=True)
class FAQImportResult:
    """Summary of one FAQ import run, mirroring the import progress shape.

    ``success_count`` counts the entries that passed validation
    (``added_count`` equals it because this pipeline only creates new
    entries); ``skipped_count`` counts blank rows in the file.
    """

    mode: str
    total: int
    success_count: int
    added_count: int
    failed_count: int
    skipped_count: int
    failed_entries: list[FailedEntry] = field(default_factory=list)
    success_entries: list[ImportedEntry] = field(default_factory=list)


def build_faq_chunk_content(content: FAQContent, *, index_mode: str | None) -> str:
    """Build the indexable chunk text for one FAQ entry.

    The standard question is always present; similar questions follow as
    a bulleted block; answers are included only in question+answer index
    mode. Negative questions never appear in the indexed content.
    """
    parts = [f"Q: {content.standard_question}"]
    if content.similar_questions:
        parts.append("Similar Questions:")
        parts.extend(f"- {q}" for q in content.similar_questions)
    if index_mode == FAQ_INDEX_MODE_QUESTION_ANSWER and content.answers:
        parts.append("Answers:")
        parts.extend(f"- {a}" for a in content.answers)
    return "\n".join(parts)


def _content_hash(content: FAQContent) -> str:
    """Return a deterministic content hash for FAQ dedupe bookkeeping."""
    payload = json.dumps(
        content.model_dump(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def import_faq(
    *,
    file_data: bytes,
    filename: str,
    tenant_id: int,
    knowledge_base_id: str,
    knowledge_id: str,
    faq_repo: FaqRepository,
    chunk_repo: ChunkRepository,
    mode: str = FAQ_BATCH_MODE_APPEND,
    index_mode: str | None = None,
    dry_run: bool = False,
) -> FAQImportResult:
    """Import FAQ entries from ``file_data`` into the knowledge base.

    Parses the file, validates every entry, and (unless ``dry_run``)
    persists the valid ones as FAQ chunks and ``faq`` rows. Failed rows
    are reported in the result rather than aborting the whole import.
    """
    mode = mode or FAQ_BATCH_MODE_APPEND
    if mode not in _IMPORT_MODES:
        raise ValidationError(
            code="faq.invalid_import_mode",
            message="模式仅支持 append 或 replace",
        )

    parsed = parse_import_file(file_data, filename=filename)
    if parsed.total == 0:
        raise ValidationError(
            code="faq.entries_required",
            message="FAQ 条目不能为空",
        )

    failed: list[FailedEntry] = [_parser_error_to_failed(err) for err in parsed.errors]
    valid: list[tuple[int, FAQEntryPayload, FAQContent]] = []
    seen_questions: dict[str, int] = {}

    for parsed_entry in parsed.entries:
        row_number = parsed_entry.row_number
        payload = parsed_entry.payload
        try:
            content = sanitize_faq_content(
                standard_question=payload.standard_question,
                similar_questions=payload.similar_questions,
                negative_questions=payload.negative_questions,
                answers=payload.answers,
                answer_strategy=payload.answer_strategy,
            )
        except ValidationError as exc:
            failed.append(
                _failed_from_payload(
                    payload,
                    row_number=row_number,
                    code=exc.code or "faq.invalid_entry",
                    reason=exc.message,
                )
            )
            continue

        conflict = _find_batch_conflict(content, seen_questions)
        if conflict is not None:
            question, first_row = conflict
            failed.append(
                _failed_from_payload(
                    payload,
                    row_number=row_number,
                    code="faq.duplicate_in_batch",
                    reason=f"问题「{question}」与批次内第 {first_row} 行重复",
                )
            )
            continue
        for question in (content.standard_question, *content.similar_questions):
            seen_questions.setdefault(question, row_number)

        if mode == FAQ_BATCH_MODE_APPEND:
            existing = await faq_repo.find_duplicate_question(
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
                exclude_id=None,
                questions=[content.standard_question, *content.similar_questions],
            )
            if existing is not None:
                failed.append(
                    _failed_from_payload(
                        payload,
                        row_number=row_number,
                        code="faq.duplicate_question",
                        reason=f"标准问「{content.standard_question}」已存在",
                    )
                )
                continue

        valid.append((row_number, payload, content))

    success_count = len(valid)
    if dry_run or success_count == 0:
        return FAQImportResult(
            mode=mode,
            total=parsed.total,
            success_count=success_count,
            added_count=0,
            failed_count=len(failed),
            skipped_count=parsed.skipped_rows,
            failed_entries=failed,
            success_entries=[],
        )

    now = datetime.now(UTC)
    chunk_rows: list[Chunk] = []
    faq_rows: list[Faq] = []
    for _row_number, payload, content in valid:
        chunk_id = str(uuid4())
        is_enabled = payload.is_enabled if payload.is_enabled is not None else True
        is_recommended = payload.is_recommended if payload.is_recommended is not None else False
        chunk_content = build_faq_chunk_content(content, index_mode=index_mode)
        chunk_rows.append(
            Chunk(
                id=chunk_id,
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
                knowledge_id=knowledge_id,
                content=chunk_content,
                chunk_index=0,
                is_enabled=is_enabled,
                start_at=0,
                end_at=0,
                chunk_type=CHUNK_TYPE_FAQ,
                metadata=cast(JsonObject, content.model_dump()),
                tag_id=None,
                status=CHUNK_STATUS_STORED,
                flags=CHUNK_FLAG_RECOMMENDED if is_recommended else 0,
                source_content="",
                content_revision=0,
                index_status="ready",
                last_editor_id="",
                content_hash=_content_hash(content),
                created_at=now,
                updated_at=now,
            )
        )
        faq_rows.append(
            build_faq_row(
                tenant_id=tenant_id,
                chunk_id=chunk_id,
                knowledge_id=knowledge_id,
                knowledge_base_id=knowledge_base_id,
                content=content,
                tag_name=payload.tag_name,
                is_enabled=is_enabled,
                is_recommended=is_recommended,
                index_mode=index_mode,
            )
        )

    persisted_chunks = await chunk_repo.create_many(chunk_rows)
    persisted_faqs: list[Faq] = []
    for row in faq_rows:
        persisted_faqs.append(await faq_repo.create(row))

    success_entries = [
        ImportedEntry(
            row_number=row_number,
            id=faq_row.id,
            chunk_id=chunk_row.id,
            seq_id=chunk_row.seq_id,
            tag_name=faq_row.tag_name,
            standard_question=faq_row.standard_question,
        )
        for (row_number, _payload, _content), faq_row, chunk_row in zip(
            valid,
            persisted_faqs,
            persisted_chunks,
            strict=True,
        )
    ]

    return FAQImportResult(
        mode=mode,
        total=parsed.total,
        success_count=len(success_entries),
        added_count=len(success_entries),
        failed_count=len(failed),
        skipped_count=parsed.skipped_rows,
        failed_entries=failed,
        success_entries=success_entries,
    )


# ── Validation helpers ────────────────────────────────────────────────


def _find_batch_conflict(
    content: FAQContent,
    seen_questions: dict[str, int],
) -> tuple[str, int] | None:
    """Return the first question that collides with an earlier batch row.

    ``seen_questions`` maps every question (standard + similar) of the
    accepted rows to the row that first introduced it.
    """
    for question in (content.standard_question, *content.similar_questions):
        first_row = seen_questions.get(question)
        if first_row is not None:
            return question, first_row
    return None


def _failed_from_payload(
    payload: FAQEntryPayload,
    *,
    row_number: int,
    code: str,
    reason: str,
) -> FailedEntry:
    """Build a failed-entry report from the row's payload."""
    return FailedEntry(
        row_number=row_number,
        code=code,
        reason=reason,
        tag_name=payload.tag_name,
        standard_question=payload.standard_question,
        similar_questions=list(payload.similar_questions or []),
        negative_questions=list(payload.negative_questions or []),
        answers=list(payload.answers or []),
        answer_all=payload.answer_strategy == ANSWER_STRATEGY_ALL,
        is_disabled=payload.is_enabled is False,
    )


def _parser_error_to_failed(error: ImportRowError) -> FailedEntry:
    """Promote a structural parse error into a failed-entry report."""
    return FailedEntry(
        row_number=error.row_number,
        code=error.code,
        reason=error.message,
        tag_name=None,
        standard_question="",
    )


__all__ = [
    "FAQ_BATCH_MODE_APPEND",
    "FAQ_BATCH_MODE_REPLACE",
    "FAQImportResult",
    "FailedEntry",
    "ImportedEntry",
    "build_faq_chunk_content",
    "import_faq",
]
