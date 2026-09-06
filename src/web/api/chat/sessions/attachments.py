"""Session-attachment HTTP handlers.

Decorators stay on ``router.py``. Bytes are persisted on the tenant
default store, then the metadata row is marked ``ready`` so the
paperclip stops polling. Parse into the prompt is a later bind.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, UploadFile
from fastapi.responses import StreamingResponse

from src.ai.storage.base import FileService
from src.app_context import request_context
from src.common.exception import NotFoundError, StorageBackendError, ValidationError
from src.core.chat.sessions.service.session_service import SessionService
from src.core.knowledge.documents.temporary_document import (
    TemporaryDocumentCreateOptions,
    TemporaryDocumentInfo,
    TemporaryDocumentService,
    validate_upload,
)
from src.web.api.chat.sessions.views import (
    DeleteSessionResponse,
    TemporaryAttachmentEnvelope,
    TemporaryAttachmentListEnvelope,
    attachment_envelope,
    attachment_list_envelope,
    delete_session_response,
)
from src.web.api.files.router import content_type_for_storage_path, resolve_file_service_for_path
from src.web.deps.session import SessionDep

_DELETE_MESSAGE = "Attachment deleted successfully"


def _context_tenant_id() -> int:
    """Read the active workspace id, or raise a typed scope error."""
    raw = request_context.get_tenant_id()
    try:
        tenant_id = int(raw) if raw else 0
    except (TypeError, ValueError):
        tenant_id = 0
    if tenant_id <= 0:
        raise ValidationError(
            code="temporary_document.invalid_scope",
            message="invalid attachment scope",
        )
    return tenant_id


async def get_session_attachment_storage(session: SessionDep) -> FileService:
    """Resolve the tenant default store. Missing backend is a typed 404."""
    service = await resolve_file_service_for_path(session, _context_tenant_id(), "")
    if service is None:
        raise NotFoundError(
            code="temporary_document.storage_unavailable",
            message="file service unavailable",
        )
    return service


SessionAttachmentStorageDep = Annotated[
    FileService,
    Depends(get_session_attachment_storage),
]


def _create_options(
    agent_source_tenant_id: str | None,
    parser_engine: str | None,
) -> TemporaryDocumentCreateOptions:
    """Map optional paperclip form fields onto persist options."""
    resource_tenant_id = 0
    raw = (agent_source_tenant_id or "").strip()
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = 0
        if parsed > 0:
            resource_tenant_id = parsed
    return TemporaryDocumentCreateOptions(
        resource_tenant_id=resource_tenant_id,
        parser_engine=(parser_engine or "").strip(),
    )


async def _persist_bytes(
    storage: FileService,
    *,
    tenant_id: int,
    file_name: str,
    data: bytes,
) -> str:
    """Write upload bytes to the temporary store."""
    try:
        return await storage.save_bytes(
            data=data,
            tenant_id=tenant_id,
            file_name=file_name,
            temp=True,
        )
    except StorageBackendError:
        raise
    except Exception as exc:
        raise StorageBackendError(
            code="temporary_document.storage_failed",
            message="failed to persist attachment bytes",
        ) from exc


def _require_attachment(
    row: TemporaryDocumentInfo | None,
    attachment_id: str,
) -> TemporaryDocumentInfo:
    """Return the row or raise a typed missing-attachment error."""
    if row is None:
        raise NotFoundError(
            code="temporary_document.not_found",
            message=f"attachment {attachment_id} not found",
        )
    return row


async def upload_session_attachment(
    *,
    session_id: str,
    file: UploadFile,
    agent_source_tenant_id: str | None,
    parser_engine: str | None,
    session_service: SessionService,
    temp_docs: TemporaryDocumentService,
    storage: FileService,
) -> TemporaryAttachmentEnvelope:
    """Persist bytes, record metadata, and promote the row to ``ready``."""
    session = await session_service.get(session_id)
    data = await file.read()
    safe_name = validate_upload(
        tenant_id=session.tenant_id,
        session_id=session.id,
        file_name=file.filename or "",
        file_size=len(data),
    )
    resource_ref = await _persist_bytes(
        storage,
        tenant_id=session.tenant_id,
        file_name=safe_name,
        data=data,
    )
    created = await temp_docs.create(
        tenant_id=session.tenant_id,
        session_id=session.id,
        resource_ref=resource_ref,
        file_name=safe_name,
        mime_type=(file.content_type or "").strip(),
        file_size=len(data),
        options=_create_options(agent_source_tenant_id, parser_engine),
    )
    ready = await temp_docs.mark_ready(tenant_id=session.tenant_id, document_id=created.id)
    return attachment_envelope(ready if ready is not None else created)


async def get_session_attachment(
    *,
    session_id: str,
    attachment_id: str,
    session_service: SessionService,
    temp_docs: TemporaryDocumentService,
) -> TemporaryAttachmentEnvelope:
    """Return one attachment scoped to the owned session."""
    session = await session_service.get(session_id)
    row = await temp_docs.get(
        tenant_id=session.tenant_id,
        session_id=session.id,
        document_id=attachment_id,
    )
    return attachment_envelope(_require_attachment(row, attachment_id))


async def list_session_attachments(
    *,
    session_id: str,
    session_service: SessionService,
    temp_docs: TemporaryDocumentService,
) -> TemporaryAttachmentListEnvelope:
    """Return every live attachment of the owned session."""
    session = await session_service.get(session_id)
    rows = await temp_docs.list(tenant_id=session.tenant_id, session_id=session.id)
    return attachment_list_envelope(rows)


async def delete_session_attachment(
    *,
    session_id: str,
    attachment_id: str,
    session_service: SessionService,
    temp_docs: TemporaryDocumentService,
) -> DeleteSessionResponse:
    """Soft-delete one attachment of the owned session."""
    session = await session_service.get(session_id)
    deleted = await temp_docs.delete(
        tenant_id=session.tenant_id,
        session_id=session.id,
        document_id=attachment_id,
    )
    if not deleted:
        raise NotFoundError(
            code="temporary_document.not_found",
            message=f"attachment {attachment_id} not found",
        )
    return delete_session_response(_DELETE_MESSAGE)


async def preview_session_attachment(
    *,
    session_id: str,
    attachment_id: str,
    session_service: SessionService,
    temp_docs: TemporaryDocumentService,
    storage: FileService,
) -> StreamingResponse:
    """Stream the stored bytes for an owned session attachment."""
    session = await session_service.get(session_id)
    row = _require_attachment(
        await temp_docs.get(
            tenant_id=session.tenant_id,
            session_id=session.id,
            document_id=attachment_id,
        ),
        attachment_id,
    )
    try:
        stream = await storage.get_file(row.resource_ref)
    except Exception as exc:
        raise NotFoundError(
            code="temporary_document.file_not_found",
            message="file not found",
        ) from exc
    media_type = row.mime_type.strip() or content_type_for_storage_path(row.resource_ref)
    safe_name = row.file_name.replace('"', "")
    return StreamingResponse(
        stream,
        media_type=media_type,
        headers={
            "Content-Disposition": f'inline; filename="{safe_name}"',
            "Cache-Control": "private, max-age=3600",
            "X-Content-Type-Options": "nosniff",
        },
    )


__all__ = [
    "SessionAttachmentStorageDep",
    "delete_session_attachment",
    "get_session_attachment",
    "get_session_attachment_storage",
    "list_session_attachments",
    "preview_session_attachment",
    "upload_session_attachment",
]
