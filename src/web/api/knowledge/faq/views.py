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
views resolve it from the knowledge base's documents. A first write on a
new FAQ knowledge base creates that container; otherwise the manager's
create and import buttons only toast.
"""
# Chinese user-facing messages use fullwidth punctuation.

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from json import JSONDecodeError

from fastapi import Request, UploadFile
from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError as PydanticValidationError
from starlette.datastructures import FormData
from starlette.datastructures import UploadFile as StarletteUploadFile

from src.common.exception import ValidationError
from src.common.json import JsonObject, JsonValue
from src.core.contracts.knowledge import (
    FAQBatchUpsertPayload,
    FAQEntry,
    FAQEntryListResponse,
    FAQImportTaskProgress,
)
from src.core.knowledge.documents.faq_import import FAQ_BATCH_MODE_APPEND
from src.core.knowledge.documents.service.knowledge_service import KnowledgeService
from src.core.knowledge.documents.types import (
    KNOWLEDGE_TYPE_FAQ,
    PARSE_STATUS_COMPLETED,
)
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


class FAQSearchEnvelope(BaseModel):
    """``{"success": true, "data": [...]}`` — search hits, not a paged list."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: list[FAQEntry]


class DeleteFAQEntriesResponse(BaseModel):
    """``{"success": true}`` — batch-delete ack response."""

    model_config = ConfigDict(frozen=True)

    success: bool


class FAQAckResponse(BaseModel):
    """``{"success": true}`` — batch tag / field ack."""

    model_config = ConfigDict(frozen=True)

    success: bool


@dataclass(frozen=True)
class MultipartFAQImport:
    """Parsed multipart file import; CSV / Excel bytes only."""

    file_data: bytes
    filename: str
    mode: str
    dry_run: bool


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


def search_envelope(entries: list[FAQEntry]) -> FAQSearchEnvelope:
    """Wrap search hits as a bare list in ``data``."""
    return FAQSearchEnvelope(success=True, data=entries)


def mutation_ack() -> FAQAckResponse:
    """Build the batch tag / field acknowledgement."""
    return FAQAckResponse(success=True)


def inlined_json_schema(model: type[BaseModel]) -> JsonObject:
    """Return a JSON Schema with ``$defs`` refs expanded.

    ``openapi-typescript`` cannot resolve Pydantic ``#/$defs/...`` refs
    when they are pasted into a path-level requestBody.
    """
    raw: JsonValue = model.model_json_schema()
    if not isinstance(raw, dict):
        return {}
    defs_raw = raw.pop("$defs", {})
    defs: JsonObject = defs_raw if isinstance(defs_raw, dict) else {}
    expanded = _expand_schema_refs(raw, defs)
    return expanded if isinstance(expanded, dict) else {}


def _expand_schema_refs(node: JsonValue, defs: JsonObject) -> JsonValue:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            replacement = defs.get(ref.rsplit("/", 1)[-1])
            if replacement is not None:
                return _expand_schema_refs(replacement, defs)
        expanded: JsonObject = {}
        for key, value in node.items():
            expanded[str(key)] = _expand_schema_refs(value, defs)
        return expanded
    if isinstance(node, list):
        return [_expand_schema_refs(item, defs) for item in node]
    return node


def is_json_content_type(content_type: str) -> bool:
    """Whether the request is the SPA JSON upsert, not a file upload."""
    media_type = content_type.split(";", 1)[0].strip().lower()
    return media_type == "application/json"


async def read_json_upsert_payload(request: Request) -> FAQBatchUpsertPayload:
    """Parse ``POST /entries`` JSON into the frozen upsert contract."""
    try:
        raw: JsonValue = await request.json()
    except JSONDecodeError as exc:
        raise ValidationError(
            code="faq.invalid_json",
            message="FAQ 批量写入请求不是合法 JSON",
        ) from exc
    try:
        return FAQBatchUpsertPayload.model_validate(raw)
    except PydanticValidationError as exc:
        raise ValidationError(
            code="faq.invalid_upsert_payload",
            message="FAQ 批量写入请求不合法",
        ) from exc


async def read_multipart_import(request: Request) -> MultipartFAQImport:
    """Read the CSV / Excel file part and the mode / dry-run switches."""
    form = await request.form()
    try:
        return await _multipart_from_form(form)
    finally:
        await form.close()


async def _multipart_from_form(form: FormData) -> MultipartFAQImport:
    upload = form.get("file")
    if not isinstance(upload, (UploadFile, StarletteUploadFile)):
        raise ValidationError(
            code="faq.import_file_required",
            message="请上传 FAQ 导入文件",
        )
    file_data = await upload.read()
    return MultipartFAQImport(
        file_data=file_data,
        filename=upload.filename or "",
        mode=_form_str(form.get("mode"), FAQ_BATCH_MODE_APPEND),
        dry_run=_form_bool(form.get("dry_run"), default=False),
    )


def _form_str(value: str | UploadFile | StarletteUploadFile | None, default: str) -> str:
    if value is None or value == "":
        return default
    if isinstance(value, str):
        return value
    raise ValidationError(
        code="faq.invalid_import_mode",
        message="模式仅支持 append 或 replace",
    )


def _form_bool(
    value: str | bool | UploadFile | StarletteUploadFile | None, *, default: bool
) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    raise ValidationError(
        code="faq.invalid_dry_run",
        message="dry_run 仅支持 true 或 false",
    )


async def _create_faq_container(
    knowledge_service: KnowledgeService,
    *,
    tenant_id: int,
    knowledge_base_id: str,
) -> str:
    """Insert the FAQ knowledge row a first entry write needs."""
    created = await knowledge_service.create_document(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        type=KNOWLEDGE_TYPE_FAQ,
        title="FAQ",
        source="faq",
        parse_status=PARSE_STATUS_COMPLETED,
    )
    return created.id


async def resolve_faq_knowledge_id(
    knowledge_service: KnowledgeService,
    *,
    tenant_id: int,
    knowledge_base_id: str,
) -> str:
    """Resolve the FAQ container knowledge id of a knowledge base.

    FAQ entries belong to a FAQ-type knowledge document inside the
    knowledge base. The newest FAQ document is used. A knowledge base
    with none gets a completed container so the first write can persist.
    """
    documents = await knowledge_service.list_documents(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
    )
    for document in documents:
        if document.type == KNOWLEDGE_TYPE_FAQ:
            return document.id
    return await _create_faq_container(
        knowledge_service,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
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
    "FAQAckResponse",
    "FAQEntryEnvelope",
    "FAQEntryListEnvelope",
    "FAQImportProgressEnvelope",
    "FAQSearchEnvelope",
    "MultipartFAQImport",
    "build_export_csv",
    "build_export_json",
    "collect_all_entries",
    "delete_ack",
    "entry_envelope",
    "import_progress_envelope",
    "inlined_json_schema",
    "is_json_content_type",
    "list_envelope",
    "mutation_ack",
    "read_json_upsert_payload",
    "read_multipart_import",
    "resolve_faq_knowledge_id",
    "search_envelope",
]
