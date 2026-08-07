"""Internal types and constants for the FAQ domain.

The wire shapes live in ``src/core/contracts/knowledge.py`` (frozen);
this module holds the domain-level constants, the sanitised FAQ entry
content model, and the pure validation used by the FAQ operations.
Field and JSON names mirror the FAQ entry contract so a value round-trips
unchanged from a request payload to a persisted row and back to the wire.

``FAQContent`` is the sanitised question/answer body of one entry: the
standard question, the similar and negative question aliases, and the
answers. It is the persistence content; entry-level scope columns
(tenant / knowledge base / chunk) live on the ``faq`` row model.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.common.exception import ValidationError

# ── Answer strategy ─────────────────────────────────────────────────

ANSWER_STRATEGY_ALL = "all"
ANSWER_STRATEGY_RANDOM = "random"

# ── FAQ index modes ─────────────────────────────────────────────────

FAQ_INDEX_MODE_QUESTION_ONLY = "question_only"
FAQ_INDEX_MODE_QUESTION_ANSWER = "question_answer"

# ── FAQ question index modes ────────────────────────────────────────

FAQ_QUESTION_INDEX_MODE_COMBINED = "combined"
FAQ_QUESTION_INDEX_MODE_SEPARATE = "separate"

# ── Chunk-type tag carried by every FAQ entry ───────────────────────

CHUNK_TYPE_FAQ = "faq"

# Default tag name for entries without an explicit tag.
UNTAGGED_TAG_NAME = "未分类"

# Version stamped on a freshly-sanitised entry's content.
FAQ_CONTENT_INITIAL_VERSION = 1

# The entry content's source tag (distinguishes FAQ content in later
# retrieval/indexing stages).
FAQ_CONTENT_SOURCE = "faq"

ANSWER_STRATEGIES: frozenset[str] = frozenset(
    {ANSWER_STRATEGY_ALL, ANSWER_STRATEGY_RANDOM}
)

FAQ_INDEX_MODES: frozenset[str] = frozenset(
    {FAQ_INDEX_MODE_QUESTION_ONLY, FAQ_INDEX_MODE_QUESTION_ANSWER}
)


class FAQContent(BaseModel):
    """Sanitised question/answer content of one FAQ entry.

    ``similar_questions`` and ``negative_questions`` are retrieval
    aliases; ``answers`` is the reply set. ``answer_strategy`` controls
    whether every answer is returned or a single random one.
    """

    model_config = ConfigDict(frozen=True)

    standard_question: str
    similar_questions: list[str] = Field(default_factory=list)
    negative_questions: list[str] = Field(default_factory=list)
    answers: list[str] = Field(default_factory=list)
    answer_strategy: str = ANSWER_STRATEGY_ALL
    version: int = FAQ_CONTENT_INITIAL_VERSION
    source: str = FAQ_CONTENT_SOURCE


def _dedupe_trimmed(values: list[str] | None) -> list[str]:
    """Trim and dedupe a string list, preserving first-occurrence order.

    Empty entries are dropped; duplicates keep their first occurrence.
    """
    if not values:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for raw in values:
        value = raw.strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _trim_non_empty(values: list[str] | None) -> list[str]:
    """Trim each value and drop empty entries, preserving duplicates.

    Unlike :func:`_dedupe_trimmed`, duplicates are kept — the duplicate
    validation must see them so an exact duplicate is rejected rather
    than silently collapsed.
    """
    if not values:
        return []
    return [v.strip() for v in values if v.strip()]


def validate_question_sets(
    *,
    standard_question: str,
    similar_questions: list[str],
    negative_questions: list[str],
) -> None:
    """Validate the intra-entry question relationships.

    Mirrors the entry-level duplicate rules: a similar question must not
    equal the standard question, similar questions must be unique among
    themselves, and a negative question must not equal the standard
    question, any similar question, or another negative question.
    Raises ``ValidationError`` on the first violation.
    """
    similar_set: set[str] = set()
    for q in similar_questions:
        if q == standard_question:
            raise ValidationError(
                code="faq.duplicate_question",
                message=f"相似问「{q}」不能与标准问相同",
            )
        if q in similar_set:
            raise ValidationError(
                code="faq.duplicate_question",
                message=f"相似问「{q}」重复",
            )
        similar_set.add(q)

    positive_questions = {standard_question, *similar_set}
    negative_seen: set[str] = set()
    for q in negative_questions:
        if q in positive_questions:
            if q == standard_question:
                raise ValidationError(
                    code="faq.duplicate_question",
                    message=f"反例问题「{q}」不能与标准问相同",
                )
            raise ValidationError(
                code="faq.duplicate_question",
                message=f"反例问题「{q}」不能与相似问相同",
            )
        if q in negative_seen:
            raise ValidationError(
                code="faq.duplicate_question",
                message=f"反例问题「{q}」重复",
            )
        negative_seen.add(q)


def sanitize_faq_content(
    *,
    standard_question: str,
    similar_questions: list[str] | None = None,
    negative_questions: list[str] | None = None,
    answers: list[str] | None = None,
    answer_strategy: str | None = None,
) -> FAQContent:
    """Validate and sanitize FAQ entry content from a request payload.

    Trims and dedupes every list, applies the entry-level duplicate
    rules, and enforces the required fields: a non-empty standard
    question and at least one answer. The duplicate rules are checked on
    the trimmed lists *before* deduplication, so an exact duplicate is
    rejected rather than silently collapsed; answers are deduped without
    a duplicate error. Raises ``ValidationError`` when the entry is
    invalid, mirroring the request-validation semantics.
    """
    strategy = answer_strategy or ANSWER_STRATEGY_ALL
    if strategy not in ANSWER_STRATEGIES:
        raise ValidationError(
            code="faq.invalid_answer_strategy",
            message="answer_strategy 必须是 'all' 或 'random'",
        )

    question = standard_question.strip()
    if not question:
        raise ValidationError(
            code="faq.standard_question_required",
            message="标准问不能为空",
        )

    clean_answers = _dedupe_trimmed(answers)
    if not clean_answers:
        raise ValidationError(
            code="faq.answers_required",
            message="至少提供一个答案",
        )

    similar = _trim_non_empty(similar_questions)
    negative = _trim_non_empty(negative_questions)
    validate_question_sets(
        standard_question=question,
        similar_questions=similar,
        negative_questions=negative,
    )

    return FAQContent(
        standard_question=question,
        similar_questions=similar,
        negative_questions=negative,
        answers=clean_answers,
        answer_strategy=strategy,
        version=FAQ_CONTENT_INITIAL_VERSION,
        source=FAQ_CONTENT_SOURCE,
    )


__all__ = [
    "ANSWER_STRATEGIES",
    "ANSWER_STRATEGY_ALL",
    "ANSWER_STRATEGY_RANDOM",
    "CHUNK_TYPE_FAQ",
    "FAQ_CONTENT_INITIAL_VERSION",
    "FAQ_CONTENT_SOURCE",
    "FAQ_INDEX_MODES",
    "FAQ_INDEX_MODE_QUESTION_ANSWER",
    "FAQ_INDEX_MODE_QUESTION_ONLY",
    "FAQ_QUESTION_INDEX_MODE_COMBINED",
    "FAQ_QUESTION_INDEX_MODE_SEPARATE",
    "UNTAGGED_TAG_NAME",
    "FAQContent",
    "sanitize_faq_content",
    "validate_question_sets",
]
