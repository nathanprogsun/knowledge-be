"""FAQ entry operations — row construction and projection.

Pure mapping helpers between the sanitised FAQ content
(``src/core/knowledge/faq/types.py``), the persisted ``faq`` row
(``src/db/models/faq.py``), and the wire entry shape
(``src/core/contracts/knowledge.py``). No database access lives here;
the service combines these with the repository.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.common.exception import ValidationError
from src.core.contracts.knowledge import FAQEntry
from src.core.knowledge.faq.types import CHUNK_TYPE_FAQ, FAQContent
from src.db.models.faq import Faq


def build_faq_row(
    *,
    tenant_id: int,
    chunk_id: str,
    knowledge_id: str,
    knowledge_base_id: str,
    content: FAQContent,
    tag_id: int | None = None,
    tag_name: str | None = None,
    is_enabled: bool = True,
    is_recommended: bool = False,
    index_mode: str | None = None,
) -> Faq:
    """Build a new ``faq`` row from sanitised content and entry scope.

    ``content`` is expected to come from
    :func:`src.core.knowledge.faq.types.sanitize_faq_content`; the row
    copies the content fields verbatim. ``created_at`` / ``updated_at``
    are stamped with the current UTC time.
    """
    now = datetime.now(UTC)
    return Faq(
        tenant_id=tenant_id,
        chunk_id=chunk_id,
        knowledge_id=knowledge_id,
        knowledge_base_id=knowledge_base_id,
        tag_id=tag_id,
        tag_name=tag_name,
        is_enabled=is_enabled,
        is_recommended=is_recommended,
        standard_question=content.standard_question,
        similar_questions=list(content.similar_questions),
        negative_questions=list(content.negative_questions),
        answers=list(content.answers),
        answer_strategy=content.answer_strategy,
        index_mode=index_mode,
        chunk_type=CHUNK_TYPE_FAQ,
        created_at=now,
        updated_at=now,
    )


def faq_row_to_entry(row: Faq) -> FAQEntry:
    """Project a persisted ``faq`` row to the wire entry shape.

    The search-only result fields (``score`` / ``match_type`` /
    ``matched_question``) are left at their defaults; the search path
    populates them from the retrieval result.
    """
    return FAQEntry(
        id=row.id,
        chunk_id=row.chunk_id,
        knowledge_id=row.knowledge_id,
        knowledge_base_id=row.knowledge_base_id,
        tag_id=row.tag_id,
        tag_name=row.tag_name,
        is_enabled=row.is_enabled,
        is_recommended=row.is_recommended,
        standard_question=row.standard_question,
        similar_questions=list(row.similar_questions),
        negative_questions=list(row.negative_questions),
        answers=list(row.answers),
        answer_strategy=row.answer_strategy,
        index_mode=row.index_mode,
        chunk_type=row.chunk_type,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def duplicate_error_for(new_content: FAQContent, existing: Faq) -> ValidationError | None:
    """Return the specific duplicate error when ``new_content`` collides.

    Compares the new standard/similar questions against the existing
    entry's standard and similar questions, mirroring the duplicate
    reporting semantics: a standard-question collision is reported
    before per-similar-question collisions. Returns ``None`` when the
    entry passed in was not actually a duplicate (e.g. the collision was
    with the entry being edited, which callers exclude beforehand).
    """
    existing_similar = set(existing.similar_questions)
    if (
        new_content.standard_question
        and (
            new_content.standard_question == existing.standard_question
            or new_content.standard_question in existing_similar
        )
    ):
        return ValidationError(
            code="faq.duplicate_question",
            message=f"标准问「{new_content.standard_question}」已存在",
        )
    for q in new_content.similar_questions:
        if q == existing.standard_question or q in existing_similar:
            return ValidationError(
                code="faq.duplicate_question",
                message=f"相似问「{q}」已存在",
            )
    return None


__all__ = ["build_faq_row", "duplicate_error_for", "faq_row_to_entry"]
