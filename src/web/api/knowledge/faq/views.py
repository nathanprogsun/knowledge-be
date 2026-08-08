"""Wire-shape conversion for the FAQ endpoints.

Projects the FAQ service DTOs onto the frozen contracts in
``src/core/contracts/knowledge.py`` and builds the export documents. The
service layer already returns the wire ``FAQEntry`` / ``FAQEntryListResponse``
shapes, so most of this module is envelope construction plus the two
export renderers (CSV and JSON), which mirror the upstream export format:

- CSV columns follow the FAQ import template so an export can be edited
  and re-imported unchanged.
- JSON emits an array of ``FAQEntryPayload``-compatible objects.

The FAQ container resolver is here too: the FAQ service requires the
caller to pass the FAQ knowledge container id for an entry write, so the
views resolve it from the knowledge base's documents.
"""
# ruff: noqa: RUF001  # Chinese user-facing messages use fullwidth punctuation.

from __future__ import annotations

import csv
import io
import json

from pydantic import BaseModel, ConfigDict

from src.common.exception import ValidationError
from src.core.contracts.knowledge import (
    FAQEntry,
    FAQEntryListResponse,
    FAQImportTaskProgress,
)
from src.core.knowledge.documents.service.knowledge_service import KnowledgeService
from src.core.knowledge.documents.types import KNOWLEDGE_TYPE_FAQ
from src.core.knowledge.faq.import_parser import IMPORT_HEADERS, VALUE_SEPARATOR
from src.core.knowledge.faq.service.faq_service import FAQService
from src.core.knowledge.faq.types import ANSWER_STRATEGY_ALL

#: Export-page size while collecting every entry of a knowledge base.
_EXPORT_PAGE_SIZE = 1000


class FAQEntryEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` — single-entry responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: FAQEntry


class FAQEntryListEnvelope(BaseModel):
    """``{"success": true, "data": {total, page, page_size, data}}`` — list responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: FAQEntryListResponse


class FAQImportProgressEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` — import-progress responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: FAQImportTaskProgress


class DeleteFAQEntriesResponse(BaseModel):
    """``{"success": true}`` — batch-delete ack response."""

    model_config = ConfigDict(frozen=True)

    success: bool


def entry_envelope(entry: FAQEntry) -> FAQEntryEnvelope:
    """Wrap one entry in the success envelope."""
    return FAQEntryEnvelope(success=True, data=entry)


def list_envelope(response: FAQEntryListResponse) -> FAQEntryListEnvelope:
    """Wrap one page of entries in the success envelope."""
    return FAQEntryListEnvelope(success=True, data=response)


def import_progress_envelope(progress: FAQImportTaskProgress) -> FAQImportProgressEnvelope:
    """Wrap one import-progress record in the success envelope."""
    return FAQImportProgressEnvelope(success=True, data=progress)


def delete_ack() -> DeleteFAQEntriesResponse:
    """Build the batch-delete acknowledgement."""
    return DeleteFAQEntriesResponse(success=True)


async def resolve_faq_knowledge_id(
    knowledge_service: KnowledgeService,
    *,
    tenant_id: int,
    knowledge_base_id: str,
) -> str:
    """Resolve the FAQ container knowledge id of a knowledge base.

    FAQ entries belong to a FAQ-type knowledge document inside the
    knowledge base; the FAQ service requires that container id for an
    entry write. The newest FAQ document is used, matching the upstream
    ``ensureFAQKnowledge`` lookup.
    """
    documents = await knowledge_service.list_documents(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
    )
    for document in documents:
        if document.type == KNOWLEDGE_TYPE_FAQ:
            return document.id
    raise ValidationError(
        code="faq.knowledge_container_missing",
        message="知识库不存在 FAQ 容器，无法写入 FAQ 条目",
    )


async def collect_all_entries(
    faq_service: FAQService,
    *,
    tenant_id: int,
    knowledge_base_id: str,
) -> list[FAQEntry]:
    """Return every entry of the knowledge base, paging through the service.

    The FAQ service exposes paged listing only, so the export walker
    pages with a fixed page size until the total is reached.
    """
    collected: list[FAQEntry] = []
    while True:
        page = await faq_service.list_entries(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            limit=_EXPORT_PAGE_SIZE,
            offset=len(collected),
        )
        collected.extend(page.data)
        if len(collected) >= page.total:
            return collected


def build_export_csv(entries: list[FAQEntry]) -> bytes:
    """Render entries as the FAQ import-template CSV (UTF-8, no BOM).

    The header and column order match the import template so the bytes
    can be fed straight back into the import endpoint.
    """
    output = io.StringIO()
    writer = csv.writer(output, dialect="excel")
    writer.writerow(list(IMPORT_HEADERS))
    for entry in entries:
        writer.writerow(
            [
                entry.tag_name or "",
                entry.standard_question,
                VALUE_SEPARATOR.join(entry.similar_questions),
                VALUE_SEPARATOR.join(entry.negative_questions),
                VALUE_SEPARATOR.join(entry.answers),
                _bool_token(entry.answer_strategy == ANSWER_STRATEGY_ALL),
                _bool_token(not entry.is_enabled),
                _bool_token(not entry.is_recommended),
            ]
        )
    return output.getvalue().encode("utf-8")


def build_export_json(entries: list[FAQEntry]) -> bytes:
    """Render entries as a JSON array compatible with the entry payload.

    Each object carries the ``FAQEntryPayload`` field set so an export can
    be edited and re-imported via the batch path.
    """
    payload = [
        {
            "id": entry.id,
            "tag_name": entry.tag_name,
            "standard_question": entry.standard_question,
            "similar_questions": entry.similar_questions,
            "negative_questions": entry.negative_questions,
            "answers": entry.answers,
            "answer_strategy": entry.answer_strategy,
            "is_enabled": entry.is_enabled,
            "is_recommended": entry.is_recommended,
        }
        for entry in entries
    ]
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _bool_token(value: bool) -> str:
    """Render a boolean toggle the way the import template expects."""
    return "TRUE" if value else "FALSE"


__all__ = [
    "DeleteFAQEntriesResponse",
    "FAQEntryEnvelope",
    "FAQEntryListEnvelope",
    "FAQImportProgressEnvelope",
    "build_export_csv",
    "build_export_json",
    "collect_all_entries",
    "delete_ack",
    "entry_envelope",
    "import_progress_envelope",
    "list_envelope",
    "resolve_faq_knowledge_id",
]
