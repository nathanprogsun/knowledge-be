"""Unit tests for FAQ domain types and content validation."""

# Chinese test data uses fullwidth punctuation.

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from src.common.exception import ValidationError
from src.core.knowledge.faq.types import (
    ANSWER_STRATEGY_ALL,
    ANSWER_STRATEGY_RANDOM,
    FAQ_CONTENT_INITIAL_VERSION,
    FAQ_CONTENT_SOURCE,
    FAQContent,
    sanitize_faq_content,
    validate_question_sets,
)

# ── sanitize_faq_content ─────────────────────────────────────────────


def test_sanitize_returns_sanitized_content() -> None:
    content = sanitize_faq_content(
        standard_question="  如何充值？  ",
        similar_questions=[" 怎么充值 ", " 怎么绑定手机 ", "  "],
        negative_questions=["退款是充值吗"],
        answers=[" 进入设置 ", "进入设置"],
    )
    assert isinstance(content, FAQContent)
    assert content.standard_question == "如何充值？"
    assert content.similar_questions == ["怎么充值", "怎么绑定手机"]
    assert content.negative_questions == ["退款是充值吗"]
    assert content.answers == ["进入设置"]
    assert content.answer_strategy == ANSWER_STRATEGY_ALL
    assert content.version == FAQ_CONTENT_INITIAL_VERSION
    assert content.source == FAQ_CONTENT_SOURCE


def test_sanitize_defaults_to_empty_lists_and_all_strategy() -> None:
    content = sanitize_faq_content(standard_question="问题", answers=["答案"])
    assert content.similar_questions == []
    assert content.negative_questions == []
    assert content.answer_strategy == ANSWER_STRATEGY_ALL


def test_sanitize_accepts_random_strategy() -> None:
    content = sanitize_faq_content(
        standard_question="问题",
        answers=["答案"],
        answer_strategy=ANSWER_STRATEGY_RANDOM,
    )
    assert content.answer_strategy == ANSWER_STRATEGY_RANDOM


def test_sanitize_rejects_invalid_answer_strategy() -> None:
    with pytest.raises(ValidationError) as excinfo:
        sanitize_faq_content(standard_question="问题", answers=["答案"], answer_strategy="both")
    assert excinfo.value.code == "faq.invalid_answer_strategy"


def test_sanitize_rejects_empty_standard_question() -> None:
    with pytest.raises(ValidationError) as excinfo:
        sanitize_faq_content(standard_question="   ", answers=["答案"])
    assert excinfo.value.code == "faq.standard_question_required"


def test_sanitize_rejects_empty_answers() -> None:
    with pytest.raises(ValidationError) as excinfo:
        sanitize_faq_content(standard_question="问题", answers=["  "])
    assert excinfo.value.code == "faq.answers_required"


def test_sanitize_rejects_similar_equal_standard() -> None:
    with pytest.raises(ValidationError) as excinfo:
        sanitize_faq_content(
            standard_question="如何充值？",
            similar_questions=["如何充值？"],
            answers=["答案"],
        )
    assert excinfo.value.code == "faq.duplicate_question"


def test_sanitize_rejects_duplicate_similar() -> None:
    with pytest.raises(ValidationError) as excinfo:
        sanitize_faq_content(
            standard_question="如何充值？",
            similar_questions=["怎么充值", "怎么充值"],
            answers=["答案"],
        )
    assert excinfo.value.code == "faq.duplicate_question"


def test_sanitize_rejects_negative_equal_standard() -> None:
    with pytest.raises(ValidationError) as excinfo:
        sanitize_faq_content(
            standard_question="如何充值？",
            negative_questions=["如何充值？"],
            answers=["答案"],
        )
    assert excinfo.value.code == "faq.duplicate_question"


def test_sanitize_rejects_negative_equal_similar() -> None:
    with pytest.raises(ValidationError) as excinfo:
        sanitize_faq_content(
            standard_question="如何充值？",
            similar_questions=["怎么充值"],
            negative_questions=["怎么充值"],
            answers=["答案"],
        )
    assert excinfo.value.code == "faq.duplicate_question"


def test_sanitize_rejects_duplicate_negative() -> None:
    with pytest.raises(ValidationError) as excinfo:
        sanitize_faq_content(
            standard_question="如何充值？",
            negative_questions=["怎么退款", "怎么退款"],
            answers=["答案"],
        )
    assert excinfo.value.code == "faq.duplicate_question"


# ── validate_question_sets (direct) ──────────────────────────────────


def test_validate_question_sets_accepts_clean_sets() -> None:
    validate_question_sets(
        standard_question="如何充值？",
        similar_questions=["怎么充值"],
        negative_questions=["怎么退款"],
    )


def test_validate_question_sets_rejects_negative_duplicate_standard() -> None:
    with pytest.raises(ValidationError):
        validate_question_sets(
            standard_question="如何充值？",
            similar_questions=[],
            negative_questions=["如何充值？"],
        )


def test_validate_question_sets_rejects_similar_duplicate_standard() -> None:
    with pytest.raises(ValidationError):
        validate_question_sets(
            standard_question="如何充值？",
            similar_questions=["如何充值？"],
            negative_questions=[],
        )


def test_validate_question_sets_rejects_duplicate_similar() -> None:
    with pytest.raises(ValidationError):
        validate_question_sets(
            standard_question="如何充值？",
            similar_questions=["怎么充值", "怎么充值"],
            negative_questions=[],
        )


def test_validate_question_sets_rejects_duplicate_negative() -> None:
    with pytest.raises(ValidationError):
        validate_question_sets(
            standard_question="如何充值？",
            similar_questions=[],
            negative_questions=["怎么退款", "怎么退款"],
        )


# ── FAQContent model ─────────────────────────────────────────────────


def test_faq_content_is_frozen() -> None:
    content = FAQContent(
        standard_question="问题",
        answers=["答案"],
    )
    with pytest.raises(PydanticValidationError):
        content.standard_question = "changed"
