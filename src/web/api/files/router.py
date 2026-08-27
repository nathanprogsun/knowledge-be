"""File-proxy HTTP endpoints.

Two surfaces, mirroring the upstream contract:

- ``GET /files?file_path=<provider://...>`` — tenant-scoped raw storage
  proxy, mounted at the app root (not under ``/api/v1``) so nginx and
  the dev proxy can forward ``/files`` unchanged.
- ``GET /api/v1/knowledge-bases/{id}/files?file_path=...`` — KB-scoped
  proxy used by shared knowledge bases (cross-tenant image rendering).

Both authenticate the caller, validate that the requested object path
belongs to the tenant that owns it, resolve the configured storage
backend for that tenant, and stream the stored bytes back.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.ai.storage.base import (
    FileService,
    content_type_for_ext,
    parse_storage_backend_path,
    parse_tenant_id_from_storage_path,
)
from src.ai.storage.factory import new_file_service_from_storage_config
from src.common.exception import NotFoundError, PermissionDeniedError, ValidationError
from src.core.infra.storage_backends.factory import build_storage_backend_service
from src.core.infra.storage_backends.types import StorageBackendConfigInfo
from src.web.deps import AuthDep, RoleViewerDep
from src.web.deps.knowledge_bases import KBServiceDep
from src.web.deps.session import SessionDep

bare_files_router = APIRouter(prefix="/files", tags=["files"])

kb_files_router = APIRouter(prefix="/knowledge-bases", tags=["files"])


class _ResolvableBackendConfig(StorageBackendConfigInfo):
    """Storage-config view carrying the blank provider default the factory reads."""

    default_provider: str = ""


def _require_tenant_id(request: Request) -> int:
    """Return the authenticated caller's active tenant id (must be positive)."""
    tenant_id = int(request.state.tenant_id or 0)
    if tenant_id <= 0:
        raise PermissionDeniedError(
            code="files.no_tenant",
            message="unauthorized: workspace context missing",
        )
    return tenant_id


def _validate_file_path(file_path: str) -> str:
    """Trim + validate the ``file_path`` query parameter."""
    path = (file_path or "").strip()
    if not path:
        raise ValidationError(
            code="files.missing_path",
            message="missing required parameter: file_path",
        )
    if ".." in path:
        raise ValidationError(
            code="files.invalid_path",
            message="invalid file path",
        )
    return path


def _content_type_for_path(path: str) -> str:
    """Pick a safe content type from the file extension."""
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    return content_type_for_ext(ext) if ext else "application/octet-stream"


async def _resolve_file_service(
    session: AsyncSession,
    tenant_id: int,
    file_path: str,
) -> FileService | None:
    """Resolve the storage file service for the tenant + object path.

    An explicit ``backend://<id>/`` segment wins; otherwise the
    workspace default backend is used. ``None`` means no storage
    backend is registered for this tenant.
    """
    storage_service = build_storage_backend_service(session)
    backend_id = ""
    parsed = parse_storage_backend_path(file_path)
    if parsed is not None:
        backend_id = parsed[0]
    info = await storage_service.resolve_backend(
        tenant_id=tenant_id,
        backend_id=backend_id,
    )
    if info is None:
        return None
    config = _ResolvableBackendConfig(**info.config.model_dump())
    return new_file_service_from_storage_config(info.provider, config)[0]


async def _stream_file(service: FileService, path: str) -> StreamingResponse:
    """Stream a stored object with a safe content type + shared cache header."""
    stream = await service.get_file(path)
    return StreamingResponse(
        stream,
        media_type=_content_type_for_path(path),
        headers={"Cache-Control": "public, max-age=86400"},
    )


@bare_files_router.get("")
async def serve_file(
    _auth: AuthDep,
    request: Request,
    session: SessionDep,
    file_path: str = Query(...),
) -> StreamingResponse:
    """Serve a stored object by storage path, tenant-scoped.

    The requested path must belong to the caller's active tenant (its
    first numeric provider segment is the tenant id); cross-tenant
    paths are rejected with 403.
    """
    path = _validate_file_path(file_path)
    tenant_id = _require_tenant_id(request)
    path_tenant_id = parse_tenant_id_from_storage_path(path)
    if path_tenant_id and path_tenant_id != tenant_id:
        raise PermissionDeniedError(
            code="files.forbidden",
            message="forbidden: file path not accessible",
        )
    service = await _resolve_file_service(session, tenant_id, path)
    if service is None:
        raise NotFoundError(
            code="files.unavailable",
            message="file service unavailable",
        )
    try:
        return await _stream_file(service, path)
    except Exception as exc:
        raise NotFoundError(
            code="files.not_found",
            message="file not found",
        ) from exc


@kb_files_router.get("/{kb_id}/files")
async def serve_kb_file(
    _auth: AuthDep,
    _role: RoleViewerDep,
    kb_service: KBServiceDep,
    request: Request,
    session: SessionDep,
    kb_id: str,
    file_path: str = Query(...),
) -> StreamingResponse:
    """Serve a knowledge-base-scoped stored object.

    The owner tenant of the knowledge base is authoritative: the object
    path must belong to that tenant, and the file is fetched through the
    owner tenant's storage config. This keeps shared-KB images reachable
    by any tenant the KB is shared with.
    """
    path = _validate_file_path(file_path)
    _require_tenant_id(request)
    kb = await kb_service.get_knowledge_base_by_id(knowledge_base_id=kb_id)
    owner_tenant_id = kb.tenant_id
    path_tenant_id = parse_tenant_id_from_storage_path(path)
    if path_tenant_id and path_tenant_id != owner_tenant_id:
        raise PermissionDeniedError(
            code="files.forbidden",
            message="forbidden: file path not accessible",
        )
    service = await _resolve_file_service(session, owner_tenant_id, path)
    if service is None:
        raise NotFoundError(
            code="files.unavailable",
            message="file service unavailable",
        )
    try:
        return await _stream_file(service, path)
    except Exception as exc:
        raise NotFoundError(
            code="files.not_found",
            message="file not found",
        ) from exc


__all__ = [
    "bare_files_router",
    "kb_files_router",
]
