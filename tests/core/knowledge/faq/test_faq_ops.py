"""Unit tests for FAQ entry mapping helpers."""

# Chinese test data uses fullwidth punctuation.

from __future__ import annotations

from datetime import UTC, datetime

from src.common.exception import ValidationError
from src.core.contracts.knowledge import FAQEntry
from src.core.knowledge.documents.faq_ops import (
    build_faq_row,
    duplicate_error_for,
    faq_row_to_entry,
)
from src.core.knowledge.faq.service.faq_service import score_faq_keyword_match
from src.core.knowledge.faq.types import CHUNK_TYPE_FAQ, FAQContent
from src.db.models.faq import Faq

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _content() -> FAQContent:
    return FAQContent(
        standard_question="如何充值？",
        similar_questions=["怎么充值"],
        negative_questions=["怎么退款"],
        answers=["进入设置"],
        answer_strategy="all",
    )


def _row() -> Faq:
    return Faq(
        id=42,
        tenant_id=7,
        chunk_id="chunk-abc",
        knowledge_id="knowledge-1",
        knowledge_base_id="kb-1",
        tag_id=None,
        tag_name=None,
        is_enabled=True,
        is_recommended=False,
        standard_question="如何充值？",
        similar_questions=["怎么充值"],
        negative_questions=[],
        answers=["进入设置"],
        answer_strategy="all",
        index_mode=None,
        chunk_type=CHUNK_TYPE_FAQ,
        created_at=_NOW,
        updated_at=_NOW,
    )


# ── build_faq_row ────────────────────────────────────────────────────


def test_build_faq_row_copies_content_and_stamps_timestamps() -> None:
    row = build_faq_row(
        tenant_id=7,
        chunk_id="chunk-abc",
        knowledge_id="knowledge-1",
        knowledge_base_id="kb-1",
        content=_content(),
        is_enabled=False,
        is_recommended=True,
        index_mode="question_only",
    )
    assert row.tenant_id == 7
    assert row.chunk_id == "chunk-abc"
    assert row.knowledge_base_id == "kb-1"
    assert row.standard_question == "如何充值？"
    assert row.similar_questions == ["怎么充值"]
    assert row.negative_questions == ["怎么退款"]
    assert row.answers == ["进入设置"]
    assert row.answer_strategy == "all"
    assert row.chunk_type == CHUNK_TYPE_FAQ
    assert row.is_enabled is False
    assert row.is_recommended is True
    assert row.index_mode == "question_only"
    assert row.created_at is not None
    assert row.updated_at == row.created_at


def test_build_faq_row_defaults_flags() -> None:
    row = build_faq_row(
        tenant_id=7,
        chunk_id="chunk-abc",
        knowledge_id="knowledge-1",
        knowledge_base_id="kb-1",
        content=_content(),
    )
    assert row.is_enabled is True
    assert row.is_recommended is False
    assert row.index_mode is None
    assert row.tag_id is None
    assert row.tag_name is None


def test_build_faq_row_copies_scope_lists_by_value() -> None:
    content = _content()
    row = build_faq_row(
        tenant_id=7,
        chunk_id="chunk-abc",
        knowledge_id="knowledge-1",
        knowledge_base_id="kb-1",
        content=content,
    )
    assert row.similar_questions is not content.similar_questions
    assert row.similar_questions == content.similar_questions


# ── faq_row_to_entry ─────────────────────────────────────────────────


def test_faq_row_to_entry_projects_wire_shape() -> None:
    entry = faq_row_to_entry(_row())
    assert isinstance(entry, FAQEntry)
    assert entry.id == 42
    assert entry.chunk_id == "chunk-abc"
    assert entry.knowledge_id == "knowledge-1"
    assert entry.knowledge_base_id == "kb-1"
    assert entry.standard_question == "如何充值？"
    assert entry.similar_questions == ["怎么充值"]
    assert entry.answers == ["进入设置"]
    assert entry.answer_strategy == "all"
    assert entry.chunk_type == CHUNK_TYPE_FAQ
    assert entry.created_at == _NOW
    assert entry.updated_at == _NOW
    # Search-only fields are left unset.
    assert entry.score is None
    assert entry.match_type is None
    assert entry.matched_question is None


# ── duplicate_error_for ──────────────────────────────────────────────


def test_duplicate_error_for_standard_collision() -> None:
    error = duplicate_error_for(
        FAQContent(standard_question="如何充值？", answers=["答案"]),
        _row(),
    )
    assert error is not None
    assert error.code == "faq.duplicate_question"
    assert "标准问" in error.message


def test_duplicate_error_for_similar_collision() -> None:
    error = duplicate_error_for(
        FAQContent(
            standard_question="如何开户？", similar_questions=["怎么充值"], answers=["答案"]
        ),
        _row(),
    )
    assert error is not None
    assert error.code == "faq.duplicate_question"
    assert "相似问" in error.message


def test_duplicate_error_for_returns_none_when_no_collision() -> None:
    error = duplicate_error_for(
        FAQContent(
            standard_question="如何开户？", similar_questions=["怎么开户"], answers=["答案"]
        ),
        _row(),
    )
    assert error is None


def test_duplicate_error_for_reports_standard_before_similar() -> None:
    error = duplicate_error_for(
        FAQContent(
            standard_question="如何充值？", similar_questions=["怎么充值"], answers=["答案"]
        ),
        _row(),
    )
    assert error is not None
    assert isinstance(error, ValidationError)
    assert "标准问" in error.message


# ── keyword search scoring ───────────────────────────────────────────


def test_keyword_score_matches_standard_question_substring() -> None:
    entry = faq_row_to_entry(_row())
    scored = score_faq_keyword_match("充值", entry)
    assert scored is not None
    score, matched = scored
    assert score > 0
    assert matched == "如何充值？"


def test_keyword_score_prefers_similar_question_when_only_it_hits() -> None:
    entry = faq_row_to_entry(_row())
    scored = score_faq_keyword_match("怎么充值", entry)
    assert scored is not None
    _score, matched = scored
    assert matched == "怎么充值"


def test_keyword_score_returns_none_when_nothing_overlaps() -> None:
    entry = faq_row_to_entry(_row())
    assert score_faq_keyword_match("退货流程", entry) is None
