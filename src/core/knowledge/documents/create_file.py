"""Knowledge-from-file creation: upload -> storage -> document create -> async dispatch.

Standalone orchestration over the merged document / knowledge-base /
storage / tag dependencies. The web layer composes this module later;
nothing here modifies the merged services.

Flow mirrors the upstream ``CreateKnowledgeFromFile`` semantics in order:

1. Resolve the display file name (custom name wins over the original).
2. Reject video uploads, missing / FAQ knowledge bases, and
   unconfigured storage before any hashing or persistence.
3. Reject an unimportable file type, hash the content, and run the
   duplicate gate (existing rows win, with ``created_at`` refreshed).
4. Enforce the storage-quota gate and the safe-file-name rule.
5. Resolve the effective process config (multimodal, question
   generation) and enforce the VLM / ASR prerequisites for the type.
6. Persist the bytes through the resolved file service, then insert the
   ``documents`` row (the saved object is removed if the insert fails).
7. Attach tags, then dispatch the async document-process task. Dispatch
   is best-effort: a failure marks the row ``failed`` but still returns
   the created knowledge so callers can retry it.

Deferred seams (neutral wording): audit recording, data-table summary
task fan-out, and the deep process-override validation against the
worker's parser/model registry land with those layers; the dispatch
seam here carries the task payload and id.

Error messages surfaced to callers are intentionally Chinese: they match
the upstream strings the frontend already renders. RUF001 (ambiguous
full-width punctuation) is suppressed file-wide for that reason.
"""

from __future__ import annotations

import contextlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from src.ai.storage.base import FileService, FileUpload
from src.common.exception import ConflictError, ValidationError
from src.common.json import JsonObject
from src.core.contracts.knowledge import Knowledge
from src.core.knowledge.documents.types import (
    KNOWLEDGE_TYPE_FAQ,
    PARSE_STATUS_FAILED,
    PARSE_STATUS_PENDING,
    SUMMARY_STATUS_NONE,
)
from src.core.knowledge.documents.upload_pipeline import (
    METADATA_PROCESS_OVERRIDES_KEY,
    DocumentProcessPayload,
    DocumentTaskDispatcher,
    calculate_file_hash,
    check_file_knowledge_exists,
    default_channel,
    file_type_of,
    is_valid_file_type,
    is_video_type,
    resolve_effective_process_config,
    validate_file_name,
    validate_import_file_type,
    validate_media_prerequisites,
)
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.tags.service.tag_service import TagService
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.knowledge import Document

# Knowledge row type for file imports.
KNOWLEDGE_TYPE_FILE = "file"

# Fresh rows start disabled; the parse worker flips them on.
_ENABLE_STATUS_DISABLED = "disabled"

# Error message stamped when the async task could not be dispatched.
_ENQUEUE_FAILED_MESSAGE = "Failed to enqueue processing task"

_DUPLICATE_FILE_CODE = "knowledge.duplicate_file"
_STORAGE_ENGINE_REQUIRED_CODE = "knowledge.storage_engine_required"


def _require_tenant_id(tenant_id: int) -> None:
    """Reject a non-positive tenant id at the orchestration boundary."""
    if not isinstance(tenant_id, int) or tenant_id <= 0:
        raise ValidationError(
            code="knowledge.tenant_required",
            message="tenant ID is required",
        )


@dataclass(frozen=True)
class TenantStorageInfo:
    """Storage accounting used by the quota gate (0 disables the check)."""

    storage_quota: int = 0
    storage_used: int = 0


@runtime_checkable
class StorageResolver(Protocol):
    """Resolve the file service for a knowledge base and tenant.

    Returns ``None`` when no storage engine is configured, which the
    orchestration turns into the storage-configured error.
    """

    async def resolve_file_service(
        self, *, knowledge_base_id: str, tenant_id: int
    ) -> FileService | None:
        """Return the storage file service, or ``None`` if unconfigured."""
        ...


class _BufferedUpload:
    """Re-readable ``FileUpload`` backed by in-memory bytes.

    The orchestration reads the content once for hashing; the storage
    adapter then reads it again from this wrapper, so the object stays
    re-readable regardless of how the underlying upload spools.
    """

    def __init__(self, *, filename: str, content_type: str, data: bytes) -> None:
        self.filename = filename
        self.content_type = content_type
        self.size = len(data)
        self._data = data

    async def read(self) -> bytes:
        """Return the whole buffered payload."""
        return self._data


async def create_knowledge_from_file(
    *,
    tenant_id: int,
    knowledge_base_id: str,
    file: FileUpload,
    knowledge_repo: KnowledgeRepository,
    kb_service: KBService,
    storage_resolver: StorageResolver | None = None,
    file_service: FileService | None = None,
    tag_service: TagService | None = None,
    dispatcher: DocumentTaskDispatcher | None = None,
    metadata: JsonObject | None = None,
    enable_multimodel: bool | None = None,
    custom_file_name: str | None = None,
    tag_ids: list[str] | None = None,
    channel: str = "",
    process_overrides: JsonObject | None = None,
    tenant_storage: TenantStorageInfo | None = None,
    language: str = "",
) -> Knowledge:
    """Create a knowledge entry from an uploaded file.

    ``knowledge_repo`` / ``kb_service`` are required dependencies;
    ``storage_resolver`` or ``file_service`` supply the storage leg;
    ``dispatcher`` carries the async document-process task. Returns the
    created knowledge in its wire shape. Raises ``ValidationError`` /
    ``ConflictError`` / ``NotFoundError`` from ``src.common.exception``
    for the invalid-input / duplicate / missing-knowledge-base cases.
    """
    _require_tenant_id(tenant_id)

    # 1. Resolve the display file name (custom name wins over original).
    file_name = custom_file_name if custom_file_name and custom_file_name.strip() else file.filename
    file_type = file_type_of(file_name)

    # 2. Video uploads are rejected before any lookup or hashing.
    if is_video_type(file_type):
        raise ValidationError(
            code="knowledge.video_not_supported",
            message="暂不支持上传视频文件",
        )

    # 3. The knowledge base must exist and accept file uploads.
    kb = await kb_service.get_knowledge_base_by_id(knowledge_base_id=knowledge_base_id)
    if kb.type == KNOWLEDGE_TYPE_FAQ:
        raise ValidationError(
            code="knowledge.faq_file_upload_unsupported",
            message="FAQ 知识库不支持文件上传，请使用 FAQ 导入功能",
        )

    # 4. Storage must be configured before any persistence.
    resolved_file_service = file_service
    if storage_resolver is not None:
        resolved_file_service = await storage_resolver.resolve_file_service(
            knowledge_base_id=knowledge_base_id,
            tenant_id=tenant_id,
        )
    if resolved_file_service is None:
        raise ValidationError(
            code=_STORAGE_ENGINE_REQUIRED_CODE,
            message="请先为知识库选择存储引擎，再上传内容。请前往知识库设置页面进行配置。",
        )

    # 5. Reject an unimportable type (distinct from the shared gate so
    #    this path keeps its own error identity).
    if not is_valid_file_type(file_name):
        raise ValidationError(
            code="knowledge.unsupported_file_type",
            message="unsupported file type",
        )

    # 6. Hash the content and run the duplicate gate.
    data = await file.read()
    file_hash = calculate_file_hash(data)
    now = datetime.now(UTC)
    existing = await check_file_knowledge_exists(
        knowledge_repo,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        file_name=file_name,
        file_type=file_type,
        file_size=len(data),
        file_hash=file_hash,
    )
    if existing is not None:
        await knowledge_repo.update_columns(existing.id, {"created_at": now})
        raise ConflictError(
            code=_DUPLICATE_FILE_CODE,
            message=f"File already exists: {existing.file_name or existing.title}",
            details={"knowledge_id": existing.id},
        )

    # 7. Storage-quota gate (skipped when no accounting is supplied).
    if (
        tenant_storage is not None
        and tenant_storage.storage_quota > 0
        and tenant_storage.storage_used >= tenant_storage.storage_quota
    ):
        raise ValidationError(
            code="knowledge.storage_quota_exceeded",
            message="Storage quota exceeded",
        )

    # 8. Safe-file-name rule and metadata assembly.
    safe_file_name = validate_file_name(file_name)
    stored_metadata = metadata if metadata else None
    if process_overrides:
        stored_metadata = dict(metadata or {})
        stored_metadata[METADATA_PROCESS_OVERRIDES_KEY] = process_overrides

    # 9. Effective process config + model prerequisites for the type.
    normalized_type = validate_import_file_type(file_type)
    validate_media_prerequisites(
        file_type=normalized_type,
        vlm_config=kb.vlm_config,
        asr_config=kb.asr_config,
    )
    eff = resolve_effective_process_config(
        chunking_config=kb.chunking_config,
        question_generation_config=kb.question_generation_config,
        enable_multimodel=enable_multimodel,
        process_overrides=process_overrides,
    )

    # 10. Persist the bytes, then insert the document row.
    knowledge_id = str(uuid.uuid4())
    file_path = await resolved_file_service.save_file(
        file=_BufferedUpload(
            filename=safe_file_name,
            content_type=file.content_type,
            data=data,
        ),
        tenant_id=tenant_id,
        knowledge_id=knowledge_id,
    )
    row = Document(
        id=knowledge_id,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        type=KNOWLEDGE_TYPE_FILE,
        title=safe_file_name,
        description=None,
        source="",
        channel=default_channel(channel),
        parse_status=PARSE_STATUS_PENDING,
        pending_subtasks_count=0,
        summary_status=SUMMARY_STATUS_NONE,
        enable_status=_ENABLE_STATUS_DISABLED,
        embedding_model_id=kb.embedding_model_id or None,
        file_name=safe_file_name,
        file_type=normalized_type,
        file_size=len(data),
        file_hash=file_hash,
        file_path=file_path,
        storage_size=0,
        metadata=stored_metadata,
        custom_metadata={},
        last_faq_import_result=None,
        created_at=now,
        updated_at=now,
        processed_at=None,
        error_message=None,
        deleted_at=None,
    )
    try:
        persisted = await knowledge_repo.create(row)
    except Exception:
        # Best-effort cleanup of the saved object when the row insert
        # fails; the original error is the one surfaced.
        with contextlib.suppress(Exception):
            await resolved_file_service.delete_file(file_path)
        raise

    # 11. Attach tags when a tag service is wired.
    if tag_service is not None and tag_ids:
        await tag_service.set_knowledge_tags(
            knowledge_id=persisted.id,
            tag_ids=[value for value in tag_ids if value],
        )

    # 12. Dispatch the async document-process task (best-effort).
    if dispatcher is not None:
        payload = DocumentProcessPayload(
            tenant_id=tenant_id,
            knowledge_id=persisted.id,
            knowledge_base_id=knowledge_base_id,
            file_path=file_path,
            file_name=safe_file_name,
            file_type=normalized_type,
            enable_multimodel=eff.enable_multimodel,
            enable_question_generation=eff.enable_question_generation,
            question_count=eff.question_count,
            language=language,
        )
        try:
            await dispatcher.dispatch(payload=payload)
        except Exception:
            # Dispatch failure is non-fatal: the file is already stored
            # and the row is returned so callers can retry it.
            await knowledge_repo.update_columns(
                persisted.id,
                {
                    "parse_status": PARSE_STATUS_FAILED,
                    "error_message": _ENQUEUE_FAILED_MESSAGE,
                },
            )

    return Knowledge.model_validate(persisted.model_dump())


__all__ = [
    "KNOWLEDGE_TYPE_FILE",
    "StorageResolver",
    "TenantStorageInfo",
    "create_knowledge_from_file",
]
