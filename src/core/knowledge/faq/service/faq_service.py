"""FAQ entry service — entry-level operations over the ``faq`` table.

Implements the self-contained part of the FAQ operations: create / get /
list / update / delete / toggle of FAQ entries, the entry-internal
question validation, and the cross-entry duplicate guard. It operates on
a resolved ``(tenant_id, knowledge_base_id)`` scope and needs no other
domain service.

Wiring that requires not-yet-merged domain services is deferred and
completed in a later change: knowledge-base type validation, tag
resolution (the payload's ``tag_id`` / ``tag_name`` are stored as given),
the FAQ container knowledge lookup, and the vector-index synchronisation
that keeps retrieval in sync with entry edits. Those callers will pass
the resolved scope and, where applicable, the index mode.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from src.common.exception import NotFoundError, ValidationError
from src.core.contracts.knowledge import FAQEntry, FAQEntryListResponse, FAQEntryPayload
from src.core.knowledge.documents.faq_ops import (
    build_faq_row,
    duplicate_error_for,
    faq_row_to_entry,
)
from src.core.knowledge.faq.types import FAQContent, sanitize_faq_content
from src.db.dao.faq_repository import FaqRepository
from src.db.models.faq import Faq


class FAQService:
    """Entry-level FAQ operations against the ``faq`` table."""

    def __init__(self, *, faq_repo: FaqRepository) -> None:
        self._faq_repo = faq_repo

    # ── Create ──────────────────────────────────────────────────────

    async def create_entry(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        knowledge_id: str,
        payload: FAQEntryPayload,
        chunk_id: str | None = None,
        index_mode: str | None = None,
    ) -> FAQEntry:
        """Create one FAQ entry and return its wire shape.

        ``knowledge_id`` is the FAQ container the entry belongs to;
        callers resolve it (together with KB-type validation) before
        invoking this method. ``chunk_id`` defaults to a fresh UUID and
        is the reference the later index-sync wiring keys on.
        """
        content = sanitize_faq_content(
            standard_question=payload.standard_question,
            similar_questions=payload.similar_questions,
            negative_questions=payload.negative_questions,
            answers=payload.answers,
            answer_strategy=payload.answer_strategy,
        )
        await self._reject_cross_entry_duplicate(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            exclude_id=None,
            content=content,
        )
        row = build_faq_row(
            tenant_id=tenant_id,
            chunk_id=chunk_id or str(uuid4()),
            knowledge_id=knowledge_id,
            knowledge_base_id=knowledge_base_id,
            content=content,
            tag_id=payload.tag_id,
            tag_name=payload.tag_name,
            is_enabled=payload.is_enabled if payload.is_enabled is not None else True,
            is_recommended=(
                payload.is_recommended if payload.is_recommended is not None else False
            ),
            index_mode=index_mode,
        )
        persisted = await self._faq_repo.create(row)
        return faq_row_to_entry(persisted)

    # ── Read ────────────────────────────────────────────────────────

    async def get_entry(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        entry_id: int,
    ) -> FAQEntry:
        """Return one entry, verifying it belongs to the knowledge base."""
        row = await self._get_scoped_row(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            entry_id=entry_id,
        )
        return faq_row_to_entry(row)

    async def list_entries(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        keyword: str | None = None,
        limit: int,
        offset: int,
    ) -> FAQEntryListResponse:
        """Return one page of the knowledge base's entries and the total."""
        rows, total = await self._faq_repo.list_by_knowledge_base(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            keyword=keyword,
            limit=limit,
            offset=offset,
        )
        entries = [faq_row_to_entry(row) for row in rows]
        page = (offset // limit) + 1 if limit > 0 else 1
        return FAQEntryListResponse(
            total=total,
            page=page,
            page_size=limit,
            data=entries,
        )

    # ── Update / toggle / delete ────────────────────────────────────

    async def update_entry(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        entry_id: int,
        payload: FAQEntryPayload,
    ) -> FAQEntry:
        """Update one entry's content and flags, returning the new shape."""
        existing = await self._get_scoped_row(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            entry_id=entry_id,
        )
        content = sanitize_faq_content(
            standard_question=payload.standard_question,
            similar_questions=payload.similar_questions,
            negative_questions=payload.negative_questions,
            answers=payload.answers,
            answer_strategy=payload.answer_strategy,
        )
        await self._reject_cross_entry_duplicate(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            exclude_id=existing.id,
            content=content,
        )
        updated = existing.model_copy(
            update={
                "standard_question": content.standard_question,
                "similar_questions": list(content.similar_questions),
                "negative_questions": list(content.negative_questions),
                "answers": list(content.answers),
                "answer_strategy": content.answer_strategy,
                "tag_id": payload.tag_id,
                "tag_name": payload.tag_name,
                "is_enabled": (
                    payload.is_enabled
                    if payload.is_enabled is not None
                    else existing.is_enabled
                ),
                "is_recommended": (
                    payload.is_recommended
                    if payload.is_recommended is not None
                    else existing.is_recommended
                ),
                "updated_at": datetime.now(UTC),
            }
        )
        persisted = await self._faq_repo.update(updated)
        return faq_row_to_entry(persisted)

    async def set_entry_enabled(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        entry_id: int,
        is_enabled: bool,
    ) -> FAQEntry:
        """Toggle an entry's enabled flag, returning the new shape."""
        existing = await self._get_scoped_row(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            entry_id=entry_id,
        )
        if existing.is_enabled == is_enabled:
            return faq_row_to_entry(existing)
        persisted = await self._faq_repo.set_enabled(
            tenant_id=tenant_id,
            id=entry_id,
            is_enabled=is_enabled,
        )
        if persisted is None:
            raise NotFoundError(
                code="faq.not_found",
                message="FAQ条目不存在",
            )
        return faq_row_to_entry(persisted)

    async def delete_entries(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        entry_ids: list[int],
    ) -> int:
        """Delete the given entries, verifying each belongs to the KB.

        Returns the number of entries removed. Any id that is absent or
        belongs to another knowledge base fails the batch before any
        delete runs, mirroring the FAQ delete semantics.
        """
        for entry_id in entry_ids:
            row = await self._faq_repo.get_by_id(
                tenant_id=tenant_id,
                id=entry_id,
            )
            if row is None:
                raise NotFoundError(
                    code="faq.not_found",
                    message="FAQ条目不存在",
                )
            if row.knowledge_base_id != knowledge_base_id:
                raise ValidationError(
                    code="faq.invalid_batch_entry",
                    message="包含无效的 FAQ 条目",
                )
        return await self._faq_repo.delete_by_ids(
            tenant_id=tenant_id,
            ids=entry_ids,
        )

    # ── Shared helpers ──────────────────────────────────────────────

    async def _get_scoped_row(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        entry_id: int,
    ) -> Faq:
        """Return one entry, verifying tenant and KB ownership.

        KB-type validation happens at the wiring layer; here we only
        enforce that the entry actually belongs to the requested
        knowledge base, so one tenant's KB id cannot reach another
        workspace's entries.
        """
        row = await self._faq_repo.get_by_id(tenant_id=tenant_id, id=entry_id)
        if row is None:
            raise NotFoundError(
                code="faq.not_found",
                message="FAQ条目不存在",
            )
        if row.knowledge_base_id != knowledge_base_id:
            raise NotFoundError(
                code="faq.not_found",
                message="FAQ条目不存在",
            )
        return row

    async def _reject_cross_entry_duplicate(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        exclude_id: int | None,
        content: FAQContent,
    ) -> None:
        """Raise the specific duplicate error when another entry collides.

        Collisions are reported with the same standard-first semantics as
        the entry-internal validation, using the existing entry's stored
        question sets.
        """
        candidate_questions = [content.standard_question, *content.similar_questions]
        if not candidate_questions:
            return
        existing = await self._faq_repo.find_duplicate_question(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            exclude_id=exclude_id,
            questions=candidate_questions,
        )
        if existing is None:
            return
        error = duplicate_error_for(content, existing)
        if error is not None:
            raise error


__all__ = ["FAQService"]
