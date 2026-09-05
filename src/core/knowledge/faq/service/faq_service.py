"""FAQ entry service — entry-level operations over the ``faq`` table.

Implements the self-contained part of the FAQ operations: create / get /
list / update / delete / toggle of FAQ entries, keyword search, JSON
batch upsert, batch field / tag updates, the entry-internal question
validation, and the cross-entry duplicate guard. It operates on a
resolved ``(tenant_id, knowledge_base_id)`` scope and needs no other
domain service.

Tag names on the payload are stored as given. Vector-index synchronisation
stays with a later retrieval wave; keyword search does not call an
embedder.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from src.common.exception import NotFoundError, ValidationError
from src.core.contracts.knowledge import (
    FAQEntry,
    FAQEntryFieldsBatchUpdate,
    FAQEntryFieldsUpdate,
    FAQEntryListResponse,
    FAQEntryPayload,
    FAQSearchRequest,
)
from src.core.knowledge.documents.faq_import import (
    FAQ_BATCH_MODE_APPEND,
    FAQ_BATCH_MODE_REPLACE,
)
from src.core.knowledge.documents.faq_ops import (
    build_faq_row,
    duplicate_error_for,
    faq_row_to_entry,
)
from src.core.knowledge.faq.types import FAQContent, sanitize_faq_content
from src.db.dao.faq_repository import FaqRepository
from src.db.models.faq import Faq

#: Keyword search writes this ``match_type`` so the manager can label the hit.
_SEARCH_MATCH_TYPE = "keywords"

#: SPA default when ``match_count`` is omitted.
_DEFAULT_MATCH_COUNT = 10

#: Page size while walking every entry of a knowledge base.
_LIST_PAGE_SIZE = 1000

_UPSERT_MODES: frozenset[str] = frozenset({FAQ_BATCH_MODE_APPEND, FAQ_BATCH_MODE_REPLACE})


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
                    payload.is_enabled if payload.is_enabled is not None else existing.is_enabled
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

    # ── Search / batch upsert / batch patch ─────────────────────────

    async def search_entries(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        request: FAQSearchRequest,
    ) -> list[FAQEntry]:
        """Return keyword-overlap hits, capped by ``match_count``.

        ``vector_threshold`` is ignored. An empty query is a client error
        so the manager's empty-box warning and the API agree.
        """
        query = request.query_text.strip()
        if not query:
            raise ValidationError(
                code="faq.empty_query",
                message="搜索内容不能为空",
            )
        cap = request.match_count if request.match_count is not None else _DEFAULT_MATCH_COUNT
        hits: list[FAQEntry] = []
        for entry in await self._list_all_entries(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
        ):
            if request.only_recommended and not entry.is_recommended:
                continue
            scored = score_faq_keyword_match(query, entry)
            if scored is None:
                continue
            score, matched = scored
            hits.append(
                entry.model_copy(
                    update={
                        "score": score,
                        "match_type": _SEARCH_MATCH_TYPE,
                        "matched_question": matched,
                    }
                )
            )
        hits.sort(key=lambda item: item.score or 0.0, reverse=True)
        return hits[: max(cap, 0)]

    async def upsert_entries(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        knowledge_id: str,
        entries: list[FAQEntryPayload],
        mode: str,
        dry_run: bool = False,
    ) -> list[FAQEntry]:
        """Append or replace FAQ rows from a JSON payload list.

        Replace clears the knowledge base first, then creates. Dry-run
        validates without writing. File bytes stay on ``FAQImportRunner``.
        """
        if mode not in _UPSERT_MODES:
            raise ValidationError(
                code="faq.invalid_import_mode",
                message="模式仅支持 append 或 replace",
            )
        if not entries:
            raise ValidationError(
                code="faq.entries_required",
                message="FAQ 条目不能为空",
            )
        if dry_run:
            await self._validate_upsert_batch(
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
                entries=entries,
                mode=mode,
            )
            return []
        if mode == FAQ_BATCH_MODE_REPLACE:
            await self._clear_knowledge_base_entries(
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
            )
        return await self._create_payloads(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            knowledge_id=knowledge_id,
            entries=entries,
        )

    async def update_entry_tags(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        updates: dict[int, int | None],
    ) -> int:
        """Set or clear ``tag_id`` for each entry in ``updates``.

        ``None`` clears the tag so the manager can send a null value
        without a second body shape.
        """
        changed = 0
        for entry_id, tag_id in updates.items():
            await self._patch_entry(
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
                entry_id=entry_id,
                is_enabled=None,
                is_recommended=None,
                tag_id=tag_id,
                set_tag=True,
            )
            changed += 1
        return changed

    async def update_entry_fields(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        batch: FAQEntryFieldsBatchUpdate,
    ) -> int:
        """Apply field patches by entry id and/or tag, honoring ``exclude_ids``."""
        exclude = set(batch.exclude_ids or [])
        patches = await self._collect_field_patches(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            batch=batch,
            exclude=exclude,
        )
        for entry_id, patch in patches.items():
            await self._patch_entry(
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
                entry_id=entry_id,
                is_enabled=patch.is_enabled,
                is_recommended=patch.is_recommended,
                tag_id=patch.tag_id,
                set_tag="tag_id" in patch.model_fields_set,
            )
        return len(patches)

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

    async def _list_all_entries(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
    ) -> list[FAQEntry]:
        """Page through every entry of the knowledge base."""
        collected: list[FAQEntry] = []
        while True:
            page = await self.list_entries(
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
                keyword=None,
                limit=_LIST_PAGE_SIZE,
                offset=len(collected),
            )
            collected.extend(page.data)
            if len(collected) >= page.total:
                return collected

    async def _clear_knowledge_base_entries(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
    ) -> None:
        """Delete every entry in the knowledge base (replace mode)."""
        existing = await self._list_all_entries(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
        )
        if not existing:
            return
        await self.delete_entries(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            entry_ids=[row.id for row in existing],
        )

    async def _create_payloads(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        knowledge_id: str,
        entries: list[FAQEntryPayload],
    ) -> list[FAQEntry]:
        """Create one row per payload via the existing create path."""
        created: list[FAQEntry] = []
        for payload in entries:
            created.append(
                await self.create_entry(
                    tenant_id=tenant_id,
                    knowledge_base_id=knowledge_base_id,
                    knowledge_id=knowledge_id,
                    payload=payload,
                )
            )
        return created

    async def _validate_upsert_batch(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        entries: list[FAQEntryPayload],
        mode: str,
    ) -> None:
        """Sanitize each payload and reject append-mode collisions."""
        for payload in entries:
            content = sanitize_faq_content(
                standard_question=payload.standard_question,
                similar_questions=payload.similar_questions,
                negative_questions=payload.negative_questions,
                answers=payload.answers,
                answer_strategy=payload.answer_strategy,
            )
            if mode == FAQ_BATCH_MODE_APPEND:
                await self._reject_cross_entry_duplicate(
                    tenant_id=tenant_id,
                    knowledge_base_id=knowledge_base_id,
                    exclude_id=None,
                    content=content,
                )

    async def _collect_field_patches(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        batch: FAQEntryFieldsBatchUpdate,
        exclude: set[int],
    ) -> dict[int, FAQEntryFieldsUpdate]:
        """Merge by-tag then by-id patches; by-id wins on the same entry."""
        patches: dict[int, FAQEntryFieldsUpdate] = {}
        by_tag = batch.by_tag or {}
        if by_tag:
            for entry in await self._list_all_entries(
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
            ):
                if entry.id in exclude or entry.tag_id is None:
                    continue
                tagged = by_tag.get(entry.tag_id)
                if tagged is not None:
                    patches[entry.id] = tagged
        for entry_id, patch in (batch.by_id or {}).items():
            if entry_id not in exclude:
                patches[entry_id] = patch
        return patches

    async def _patch_entry(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        entry_id: int,
        is_enabled: bool | None,
        is_recommended: bool | None,
        tag_id: int | None,
        set_tag: bool,
    ) -> FAQEntry:
        """Apply a sparse field patch to one scoped entry."""
        existing = await self._get_scoped_row(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            entry_id=entry_id,
        )
        next_tag_id = tag_id if set_tag else existing.tag_id
        next_tag_name = None if set_tag and tag_id is None else existing.tag_name
        updated = existing.model_copy(
            update={
                "is_enabled": existing.is_enabled if is_enabled is None else is_enabled,
                "is_recommended": (
                    existing.is_recommended if is_recommended is None else is_recommended
                ),
                "tag_id": next_tag_id,
                "tag_name": next_tag_name,
                "updated_at": datetime.now(UTC),
            }
        )
        persisted = await self._faq_repo.update(updated)
        return faq_row_to_entry(persisted)


def score_faq_keyword_match(query: str, entry: FAQEntry) -> tuple[float, str] | None:
    """Return keyword-overlap score and the question field that matched.

    Tokens are whitespace-split; a query with no spaces is one token so a
    CJK substring still hits. Answers are a fallback when no question
    field matches.
    """
    terms = _keyword_terms(query)
    if not terms:
        return None
    folded_query = query.strip().lower()
    best = _best_question_match(terms, folded_query, entry)
    if best is not None:
        return best
    for answer in entry.answers:
        hits = sum(1 for term in terms if term in answer.lower())
        if hits > 0:
            return float(hits), entry.standard_question
    return None


def _keyword_terms(query: str) -> list[str]:
    stripped = query.strip().lower()
    if not stripped:
        return []
    parts = [part for part in stripped.split() if part]
    return parts if parts else [stripped]


def _best_question_match(
    terms: list[str],
    folded_query: str,
    entry: FAQEntry,
) -> tuple[float, str] | None:
    best_score = 0.0
    matched = entry.standard_question
    for field in (entry.standard_question, *entry.similar_questions):
        hay = field.lower()
        hits = sum(1 for term in terms if term in hay)
        if hits == 0:
            continue
        score = float(hits)
        if folded_query and folded_query in hay:
            score += 1.0
        if score > best_score:
            best_score = score
            matched = field
    if best_score <= 0:
        return None
    return best_score, matched


__all__ = ["FAQService", "score_faq_keyword_match"]
