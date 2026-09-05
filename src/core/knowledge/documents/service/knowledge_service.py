"""Document service — CRUD, list, get, count and status queries.

Request-scoped service over the merged ``KnowledgeRepository``. Every
method resolves a ``(tenant_id, knowledge_base_id)`` scope from explicit
arguments (the request already carries the principal tenant), so one
workspace can never reach another's rows.

Behaviour mirrors the upstream knowledge service for the document
lifecycle:

- ``create_document`` stamps the service defaults (new UUID id,
  ``pending`` parse status, empty custom metadata) and persists the row.
- ``update_document`` applies only the non-empty mutable fields, and
  validates ``custom_metadata`` against the upstream field rules before
  storing it wholesale.
- ``get_document`` raises on an absent row; ``get_documents`` drops
  absent ids silently, matching the upstream single vs batch read split.
- ``delete_document`` / ``delete_documents`` soft-delete idempotently
  and report whether / how many rows were affected.

Tag-based listing and tag assignment belong to the tag-relation layer,
not this repository, so a non-empty tag filter is rejected here rather
than silently dropped.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from src.common.exception import NotFoundError, ValidationError
from src.common.json import BindParams, JsonObject, JsonValue
from src.common.pagination import PaginationResponse
from src.core.contracts.knowledge import Knowledge
from src.core.knowledge.documents.create_common import (
    MANUAL_CONTENT_MAX_LENGTH,
    clean_markdown,
    validate_knowledge_tags,
)
from src.core.knowledge.documents.create_manual import (
    normalize_manual_status,
    reject_manual_publish,
)
from src.core.knowledge.documents.types import (
    CHANNEL_WEB,
    KNOWLEDGE_TYPE_MANUAL,
    PARSE_STATUS_PENDING,
    SUMMARY_STATUS_NONE,
    DocumentListFilter,
)
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.dao.knowledge_tag_repository import TagRepository
from src.db.models.knowledge import Document

_NOT_FOUND_CODE = "knowledge.not_found"

# Pagination cap mirrors ``src.common.pagination.Pagination.page_size``
# and the upstream handler's ``maxListPageSize = 100`` constant.
_MAX_PAGE_SIZE = 100
_MAX_SEARCH_FILE_TYPES = 20

# ``custom_metadata`` field rules mirror the upstream update validation.
_MAX_CUSTOM_METADATA_FIELDS = 20
_MAX_CUSTOM_METADATA_KEY_LENGTH = 64
_MAX_CUSTOM_METADATA_VALUE_LENGTH = 1000

_ENABLE_STATUS_ENABLED = "enabled"


def _require_tenant_id(tenant_id: int) -> None:
    """Reject a non-positive tenant id at the service boundary."""
    if not isinstance(tenant_id, int) or tenant_id <= 0:
        raise ValidationError(
            code="knowledge.tenant_required",
            message="tenant ID is required",
        )


def _require_non_empty(value: str, *, code: str, message: str) -> None:
    """Reject a blank string at the service boundary."""
    if not value.strip():
        raise ValidationError(code=code, message=message)


def _require_document_id(id: str) -> None:
    """Reject a blank document id at the service boundary."""
    if not id.strip():
        raise ValidationError(
            code="knowledge.id_required",
            message="document ID is required",
        )


def _is_custom_metadata_scalar(value: JsonValue) -> bool:
    """Accept the scalar value kinds allowed in custom metadata.

    Both ``int`` and ``float`` are accepted: the upstream rule is framed
    on JSON numbers, which decode to a single number type.
    """
    return value is None or isinstance(value, (bool, int, float, str))


def _to_knowledge(row: Document, *, knowledge_base_name: str | None = None) -> Knowledge:
    """Project a persisted ``documents`` row onto the wire shape.

    Storage-only columns (``pending_subtasks_count``,
    ``custom_metadata``, ``last_faq_import_result``) stay off the wire.
    ``knowledge_base_name`` is filled only by cross-KB search.
    """
    return Knowledge(
        id=row.id,
        tenant_id=row.tenant_id,
        knowledge_base_id=row.knowledge_base_id,
        type=row.type,
        title=row.title,
        description=row.description,
        source=row.source,
        channel=row.channel,
        summary_status=row.summary_status,
        parse_status=row.parse_status,
        enable_status=row.enable_status,
        embedding_model_id=row.embedding_model_id,
        file_name=row.file_name,
        file_type=row.file_type,
        file_size=row.file_size,
        file_hash=row.file_hash,
        file_path=row.file_path,
        storage_size=row.storage_size,
        metadata=row.metadata,
        created_at=row.created_at,
        updated_at=row.updated_at,
        processed_at=row.processed_at,
        error_message=row.error_message,
        deleted_at=row.deleted_at,
        knowledge_base_name=knowledge_base_name,
    )


def _require_search_page(
    *,
    keyword: str,
    recent: bool,
    offset: int,
    limit: int,
    file_types: list[str],
) -> str:
    """Validate search paging and return the trimmed keyword."""
    trimmed = keyword.strip()
    if not trimmed and not recent:
        raise ValidationError(
            code="knowledge.search_keyword_required",
            message="keyword is required unless recent=true",
        )
    if offset < 0:
        raise ValidationError(
            code="knowledge.invalid_offset",
            message="offset must be >= 0",
        )
    if limit < 1 or limit > _MAX_PAGE_SIZE:
        raise ValidationError(
            code="knowledge.invalid_page_size",
            message="limit must be between 1 and 100",
        )
    if len(file_types) > _MAX_SEARCH_FILE_TYPES:
        raise ValidationError(
            code="knowledge.too_many_file_types",
            message=f"too many file_types (max {_MAX_SEARCH_FILE_TYPES})",
        )
    return trimmed


def _build_document_row(
    *,
    tenant_id: int,
    knowledge_base_id: str,
    type: str,
    title: str,
    source: str,
    channel: str,
    parse_status: str,
    summary_status: str,
    enable_status: str,
    description: str | None,
    file_name: str | None,
    file_type: str | None,
    file_size: int | None,
    file_hash: str | None,
    file_path: str | None,
    storage_size: int,
    metadata: JsonObject | None,
    custom_metadata: JsonObject | None,
    now: datetime,
) -> Document:
    """Build a new ``documents`` row with the service-assigned defaults.

    The id is a fresh UUID (assigned at the service edge, mirroring the
    upstream create-time default) and an empty custom-metadata map is
    stamped so the persisted JSONB value is never ``NULL``.
    """
    return Document(
        id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        type=type,
        title=title,
        description=description,
        source=source,
        channel=channel,
        parse_status=parse_status,
        pending_subtasks_count=0,
        summary_status=summary_status,
        enable_status=enable_status,
        embedding_model_id=None,
        file_name=file_name,
        file_type=file_type,
        file_size=file_size,
        file_hash=file_hash,
        file_path=file_path,
        storage_size=storage_size,
        metadata=metadata,
        custom_metadata=custom_metadata if custom_metadata is not None else {},
        last_faq_import_result=None,
        created_at=now,
        updated_at=now,
        processed_at=None,
        error_message=None,
        deleted_at=None,
    )


def _manual_update_changes(
    row: Document,
    *,
    content: str | None,
    status: str | None,
    process_config: JsonObject | None,
) -> BindParams:
    """Patch manual metadata. File documents reject these fields."""
    if content is None and status is None and process_config is None:
        return {}
    if row.type != KNOWLEDGE_TYPE_MANUAL:
        raise ValidationError(
            code="knowledge.manual_fields_unsupported",
            message="content and status apply only to manual documents",
        )
    metadata: JsonObject = dict(row.metadata or {})
    if content is not None:
        clean = clean_markdown(content)
        if not clean.strip():
            raise ValidationError(
                code="knowledge.content_required",
                message="内容不能为空",
            )
        if len(clean) > MANUAL_CONTENT_MAX_LENGTH:
            raise ValidationError(
                code="knowledge.content_too_long",
                message=f"内容长度超出限制（最多{MANUAL_CONTENT_MAX_LENGTH}个字符）",
            )
        metadata["content"] = clean
        raw_version = metadata.get("version")
        metadata["version"] = raw_version + 1 if isinstance(raw_version, int) else 1
    normalized_status = normalize_manual_status(status) if status is not None else None
    if normalized_status is not None:
        reject_manual_publish(normalized_status)
        metadata["status"] = normalized_status
    if process_config is not None:
        metadata["process_overrides"] = process_config
    metadata["updated_at"] = datetime.now(UTC).isoformat()
    return {"metadata": metadata}


@runtime_checkable
class SummaryRefresher(Protocol):
    """Generates a document summary and returns the updated row."""

    async def refresh(self, *, tenant_id: int, knowledge_id: str) -> Knowledge: ...


class KnowledgeService:
    """Stateless document service, constructed per request."""

    def __init__(
        self,
        *,
        knowledge_repo: KnowledgeRepository,
        summary_refresher: SummaryRefresher | None = None,
        tag_repo: TagRepository | None = None,
    ) -> None:
        self._knowledge_repo = knowledge_repo
        self._summary_refresher = summary_refresher
        self._tag_repo = tag_repo

    # ── Create ──────────────────────────────────────────────────────

    async def create_document(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        type: str,
        title: str,
        source: str,
        channel: str = CHANNEL_WEB,
        parse_status: str = PARSE_STATUS_PENDING,
        summary_status: str = SUMMARY_STATUS_NONE,
        enable_status: str = _ENABLE_STATUS_ENABLED,
        description: str | None = None,
        file_name: str | None = None,
        file_type: str | None = None,
        file_size: int | None = None,
        file_hash: str | None = None,
        file_path: str | None = None,
        storage_size: int = 0,
        metadata: JsonObject | None = None,
        custom_metadata: JsonObject | None = None,
    ) -> Knowledge:
        """Insert a new document and return its wire shape.

        Mirrors the upstream generic create: the caller supplies the
        domain fields and the service stamps the defaults (fresh UUID id,
        ``pending`` parse status, empty custom metadata) on its behalf.
        """
        _require_tenant_id(tenant_id)
        _require_non_empty(
            knowledge_base_id,
            code="knowledge.kb_required",
            message="knowledge base ID is required",
        )
        _require_non_empty(type, code="knowledge.type_required", message="type is required")
        _require_non_empty(title, code="knowledge.title_required", message="title is required")
        _require_non_empty(source, code="knowledge.source_required", message="source is required")
        now = datetime.now(UTC)
        row = _build_document_row(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            type=type,
            title=title,
            source=source,
            channel=channel,
            parse_status=parse_status,
            summary_status=summary_status,
            enable_status=enable_status,
            description=description,
            file_name=file_name,
            file_type=file_type,
            file_size=file_size,
            file_hash=file_hash,
            file_path=file_path,
            storage_size=storage_size,
            metadata=metadata,
            custom_metadata=custom_metadata,
            now=now,
        )
        persisted = await self._knowledge_repo.create(row)
        return _to_knowledge(persisted)

    # ── Read ────────────────────────────────────────────────────────

    async def get_document(self, *, tenant_id: int, id: str) -> Knowledge:
        """Return one document within the tenant scope, or raise."""
        _require_tenant_id(tenant_id)
        _require_document_id(id)
        row = await self._knowledge_repo.get_by_id(tenant_id, id)
        if row is None:
            raise NotFoundError(
                code=_NOT_FOUND_CODE,
                message="knowledge not found",
            )
        return _to_knowledge(row)

    async def get_document_by_id_only(self, *, id: str) -> Knowledge | None:
        """Resolve one document by id alone, with no tenant scope.

        Used for permission resolution. Returns ``None`` when the row is
        absent or soft-deleted, mirroring the upstream single-id lookup.
        """
        _require_document_id(id)
        row = await self._knowledge_repo.get_by_id_only(id)
        if row is None:
            return None
        return _to_knowledge(row)

    async def get_documents(self, *, tenant_id: int, ids: list[str]) -> list[Knowledge]:
        """Return the live documents whose id is in ``ids``.

        Empty ``ids`` returns an empty list without touching the database;
        absent ids are dropped rather than raising, mirroring the
        upstream batch-read semantics.
        """
        _require_tenant_id(tenant_id)
        if not ids:
            return []
        rows = await self._knowledge_repo.get_batch(tenant_id, ids)
        return [_to_knowledge(row) for row in rows]

    async def list_documents(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
    ) -> list[Knowledge]:
        """Return every live document of the knowledge base, newest first."""
        _require_tenant_id(tenant_id)
        _require_non_empty(
            knowledge_base_id,
            code="knowledge.kb_required",
            message="knowledge base ID is required",
        )
        rows = await self._knowledge_repo.list_by_knowledge_base(tenant_id, knowledge_base_id)
        return [_to_knowledge(row) for row in rows]

    async def list_documents_paged(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        page: int = 1,
        page_size: int = 20,
        list_filter: DocumentListFilter | None = None,
    ) -> PaginationResponse[Knowledge]:
        """Return one page of the knowledge base's documents plus the total.

        ``page`` / ``page_size`` mirror the pagination contract (page
        starts at 1, page size capped at 100). A non-empty ``tag_ids``
        filter is rejected: tag-based listing belongs to the
        tag-relation layer, and silently ignoring the dimension would
        mislead callers.
        """
        _require_tenant_id(tenant_id)
        _require_non_empty(
            knowledge_base_id,
            code="knowledge.kb_required",
            message="knowledge base ID is required",
        )
        if page < 1:
            raise ValidationError(
                code="knowledge.invalid_page",
                message="page must be >= 1",
            )
        if page_size < 1 or page_size > _MAX_PAGE_SIZE:
            raise ValidationError(
                code="knowledge.invalid_page_size",
                message="page_size must be between 1 and 100",
            )
        doc_filter = list_filter or DocumentListFilter()
        if doc_filter.tag_ids:
            raise ValidationError(
                code="knowledge.tag_filter_unsupported",
                message="tag filtering is handled by the tag-relation layer",
            )
        rows, total = await self._knowledge_repo.list_paged_by_knowledge_base(
            tenant_id,
            knowledge_base_id,
            limit=page_size,
            offset=(page - 1) * page_size,
            keyword=doc_filter.keyword,
            file_type=doc_filter.file_type,
            parse_status=doc_filter.parse_status,
            source=doc_filter.source,
            updated_from=doc_filter.updated_from,
            updated_to=doc_filter.updated_to,
        )
        return PaginationResponse(
            total=total,
            page=page,
            page_size=page_size,
            data=[_to_knowledge(row) for row in rows],
        )

    async def search_documents(
        self,
        *,
        tenant_id: int,
        keyword: str,
        offset: int,
        limit: int,
        file_types: list[str],
        recent: bool,
    ) -> tuple[list[Knowledge], int]:
        """Search live documents across document-type knowledge bases."""
        _require_tenant_id(tenant_id)
        trimmed = _require_search_page(
            keyword=keyword,
            recent=recent,
            offset=offset,
            limit=limit,
            file_types=file_types,
        )
        pairs, total = await self._knowledge_repo.search_across_document_kbs(
            tenant_id,
            keyword=trimmed,
            offset=offset,
            limit=limit,
            file_types=file_types,
        )
        items = [_to_knowledge(row, knowledge_base_name=name) for row, name in pairs]
        return items, total

    # ── Counts ──────────────────────────────────────────────────────

    async def count_documents(self, *, tenant_id: int, knowledge_base_id: str) -> int:
        """Count live documents in a knowledge base."""
        _require_tenant_id(tenant_id)
        _require_non_empty(
            knowledge_base_id,
            code="knowledge.kb_required",
            message="knowledge base ID is required",
        )
        return await self._knowledge_repo.count_by_knowledge_base(tenant_id, knowledge_base_id)

    async def count_documents_by_status(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        parse_statuses: list[str],
    ) -> int:
        """Count documents whose parse status is any of ``parse_statuses``.

        An empty status list counts zero without touching the database,
        mirroring the upstream status-count semantics.
        """
        _require_tenant_id(tenant_id)
        _require_non_empty(
            knowledge_base_id,
            code="knowledge.kb_required",
            message="knowledge base ID is required",
        )
        return await self._knowledge_repo.count_by_status(
            tenant_id,
            knowledge_base_id,
            parse_statuses,
        )

    # ── Update ──────────────────────────────────────────────────────

    async def update_document(
        self,
        *,
        tenant_id: int,
        id: str,
        title: str | None = None,
        description: str | None = None,
        custom_metadata: JsonObject | None = None,
        content: str | None = None,
        status: str | None = None,
        process_config: JsonObject | None = None,
        tag_ids: list[str] | None = None,
    ) -> Knowledge:
        """Patch mutable fields. ``tag_ids=[]`` clears bindings; omit to leave them."""
        _require_tenant_id(tenant_id)
        _require_document_id(id)
        row = await self._knowledge_repo.get_by_id(tenant_id, id)
        if row is None:
            raise NotFoundError(
                code=_NOT_FOUND_CODE,
                message="knowledge not found",
            )
        changes: BindParams = {}
        if title is not None and title != "":
            changes["title"] = title
        if description is not None and description != "":
            changes["description"] = description
        if custom_metadata is not None:
            self._validate_custom_metadata(custom_metadata)
            changes["custom_metadata"] = custom_metadata
        changes.update(
            _manual_update_changes(
                row, content=content, status=status, process_config=process_config
            )
        )
        if tag_ids is not None:
            await self._bind_document_tags(
                tenant_id=tenant_id,
                knowledge_base_id=row.knowledge_base_id,
                knowledge_id=row.id,
                tag_ids=tag_ids,
            )
        if not changes:
            return _to_knowledge(row)
        updated = row.model_copy(update={**changes, "updated_at": datetime.now(UTC)})
        persisted = await self._knowledge_repo.update(updated)
        return _to_knowledge(persisted)

    async def _bind_document_tags(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        knowledge_id: str,
        tag_ids: list[str],
    ) -> None:
        if self._tag_repo is None:
            raise ValidationError(
                code="knowledge.tag_repo_required",
                message="tag_ids requires a tag repository",
            )
        validated = await validate_knowledge_tags(
            tenant_id=tenant_id,
            kb_id=knowledge_base_id,
            tag_ids=tag_ids,
            tag_repo=self._tag_repo,
        )
        await self._tag_repo.set_knowledge_tags(
            knowledge_id=knowledge_id,
            tag_ids=validated,
        )

    async def request_summary_refresh(self, *, tenant_id: int, id: str) -> Knowledge:
        """Generate the document summary and return the updated row.

        A missing refresher is a configuration error, not a queued job.
        The drawer polls ``summary_status``; a pending row with no worker
        would spin forever.
        """
        _require_tenant_id(tenant_id)
        _require_document_id(id)
        if self._summary_refresher is None:
            raise ValidationError(
                code="knowledge.summary_generation_unavailable",
                message="summary generation is not available",
            )
        return await self._summary_refresher.refresh(
            tenant_id=tenant_id,
            knowledge_id=id,
        )

    # ── Delete ──────────────────────────────────────────────────────

    async def delete_document(self, *, tenant_id: int, id: str) -> bool:
        """Soft-delete one document; return whether a live row was removed.

        Idempotent: an unknown or already-deleted id reports ``False``
        rather than raising, mirroring the upstream delete semantics.
        """
        _require_tenant_id(tenant_id)
        _require_document_id(id)
        return await self._knowledge_repo.soft_delete(
            tenant_id=tenant_id,
            id=id,
            now=datetime.now(UTC),
        )

    async def delete_documents(self, *, tenant_id: int, ids: list[str]) -> int:
        """Soft-delete a batch of documents; return the number removed.

        Blank ids are dropped before the query; an empty list reports
        zero without touching the database.
        """
        _require_tenant_id(tenant_id)
        clean_ids = [value for value in ids if value.strip()]
        if not clean_ids:
            return 0
        return await self._knowledge_repo.soft_delete_list(
            tenant_id=tenant_id,
            ids=clean_ids,
            now=datetime.now(UTC),
        )

    # ── Internal helpers ────────────────────────────────────────────

    @staticmethod
    def _validate_custom_metadata(custom_metadata: JsonObject) -> None:
        """Enforce the upstream custom-metadata field rules.

        Custom metadata must be a JSON object of at most 20 scalar
        fields, each with a non-blank key of at most 64 characters whose
        serialised value is at most 1000 characters.
        """
        if not isinstance(custom_metadata, dict):
            raise ValidationError(
                code="knowledge.invalid_custom_metadata",
                message="custom_metadata must be a JSON object",
            )
        if len(custom_metadata) > _MAX_CUSTOM_METADATA_FIELDS:
            raise ValidationError(
                code="knowledge.custom_metadata_too_many_fields",
                message="custom_metadata supports at most 20 fields",
            )
        for key, value in custom_metadata.items():
            if (
                not key.strip()
                or len(key) > _MAX_CUSTOM_METADATA_KEY_LENGTH
                or len(str(value)) > _MAX_CUSTOM_METADATA_VALUE_LENGTH
            ):
                raise ValidationError(
                    code="knowledge.invalid_custom_metadata_field",
                    message=f'invalid custom_metadata field "{key}"',
                )
            if not _is_custom_metadata_scalar(value):
                raise ValidationError(
                    code="knowledge.invalid_custom_metadata_value",
                    message=(
                        f'custom_metadata field "{key}" must be a string, number, boolean, or null'
                    ),
                )


__all__ = ["KnowledgeService"]
