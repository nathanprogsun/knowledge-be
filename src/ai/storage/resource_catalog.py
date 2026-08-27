"""Resource-catalog decorator for a physical file service.

``ResourceCatalogFileService`` keeps provider drivers physical-path-only
while exposing stable ``resource://`` references to the application
layer. Every persisted object is registered with the injected catalog
(ownership, mime kind, size, content hash), reads/writes resolve the
reference back to its physical path, and ``APP_EXTERNAL_URL``-backed
references return short-lived access tokens.
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
from dataclasses import dataclass
from typing import BinaryIO, Protocol

from loguru import logger

from src.ai.storage.base import FileService, FileUpload
from src.common.exception import StorageBackendError

# Access-grant TTL for externally served resource references (2 hours).
_RESOURCE_GRANT_TTL_SECONDS = 2 * 3600


@dataclass(frozen=True)
class ResourceRegistration:
    """One physical object's identity at registration time."""

    kind: str
    mime_type: str
    original_name: str
    size: int
    content_hash: str
    temporary: bool


class ResourceCatalog(Protocol):
    """Maps public resource references to internal storage locations."""

    async def register(
        self, *, tenant_id: int, physical_path: str, meta: ResourceRegistration
    ) -> str:
        """Persist a physical object and return its stable reference."""
        ...

    async def resolve_path(self, value: str) -> tuple[str, object | None]:
        """Return ``(physical_path, resource_or_none)`` for ``value``."""
        ...

    async def bind(self, reference: str, owner_type: str, owner_id: str, relation: str) -> None:
        """Attach ``reference`` to an owning entity."""
        ...

    async def mark_deleted(self, reference: str) -> None:
        """Mark a resource deleted (soft delete)."""
        ...

    async def create_access_grant(self, reference: str, ttl_seconds: int) -> str:
        """Issue a short-lived access token for ``reference``."""
        ...


def resource_kind(name: str) -> tuple[str, str]:
    """Classify a file name into ``(kind, mime_type)``.

    Kind is ``file`` unless the mime type starts with ``image/``,
    ``audio/`` or ``video/``.
    """
    mime_type, _ = mimetypes.guess_type(name)
    mime_type = mime_type or ""
    kind = "file"
    if mime_type.startswith("image/"):
        kind = "image"
    elif mime_type.startswith("audio/"):
        kind = "audio"
    elif mime_type.startswith("video/"):
        kind = "video"
    return kind, mime_type


class ResourceCatalogFileService:
    """Decorates a physical ``FileService`` with resource registration."""

    def __init__(self, *, inner: FileService, catalog: ResourceCatalog) -> None:
        self._inner = inner
        self._catalog = catalog
        self._external_url = os.environ.get("APP_EXTERNAL_URL", "").strip().rstrip("/")

    async def check_connectivity(self) -> None:
        """Delegate to the inner service."""
        await self._inner.check_connectivity()

    # ── File operations ─────────────────────────────────────────────

    async def save_file(self, *, file: FileUpload, tenant_id: int, knowledge_id: str) -> str:
        """Save, register, then bind to the knowledge base when given."""
        physical = await self._inner.save_file(
            file=file, tenant_id=tenant_id, knowledge_id=knowledge_id
        )
        ref = await self._register(
            physical=physical,
            tenant_id=tenant_id,
            name=file.filename,
            size=file.size,
            temporary=False,
            content_hash="",
        )
        if knowledge_id:
            await self._bind_knowledge(ref, knowledge_id)
        return ref

    async def save_bytes(self, *, data: bytes, tenant_id: int, file_name: str, temp: bool) -> str:
        """Save raw bytes and register them (with a content hash)."""
        physical = await self._inner.save_bytes(
            data=data, tenant_id=tenant_id, file_name=file_name, temp=temp
        )
        content_hash = hashlib.sha256(data).hexdigest()
        return await self._register(
            physical=physical,
            tenant_id=tenant_id,
            name=file_name,
            size=len(data),
            temporary=temp,
            content_hash=content_hash,
        )

    async def get_file(self, file_path: str) -> BinaryIO:
        """Resolve the reference and delegate to the inner service."""
        physical, _ = await self._resolve(file_path)
        return await self._inner.get_file(physical)

    async def get_file_url(self, file_path: str) -> str:
        """A download URL for the object.

        Registered resources with an external URL configured return a
        short-lived ``/r/{token}`` URL; everything else delegates to the
        inner service's physical URL.
        """
        physical, is_resource = await self._resolve(file_path)
        if is_resource and self._external_url:
            token = await self._catalog.create_access_grant(file_path, _RESOURCE_GRANT_TTL_SECONDS)
            return f"{self._external_url}/r/{token}"
        return await self._inner.get_file_url(physical)

    async def delete_file(self, file_path: str) -> None:
        """Delete the physical object, then soft-delete the resource."""
        physical, is_resource = await self._resolve(file_path)
        await self._inner.delete_file(physical)
        if is_resource:
            await self._catalog.mark_deleted(file_path)

    async def copy_file(self, src_path: str, tenant_id: int, knowledge_id: str) -> str:
        """Copy the resolved object and register the new resource."""
        physical, _ = await self._resolve(src_path)
        copied = await self._inner.copy_file(
            src_path=physical, tenant_id=tenant_id, knowledge_id=knowledge_id
        )
        ref = await self._register(
            physical=copied,
            tenant_id=tenant_id,
            name=os.path.basename(physical),
            size=0,
            temporary=False,
            content_hash="",
        )
        if knowledge_id:
            await self._bind_knowledge(ref, knowledge_id)
        return ref

    # ── Internals ───────────────────────────────────────────────────

    async def _register(
        self,
        *,
        physical: str,
        tenant_id: int,
        name: str,
        size: int,
        temporary: bool,
        content_hash: str,
    ) -> str:
        kind, mime_type = resource_kind(name)
        try:
            return await self._catalog.register(
                tenant_id=tenant_id,
                physical_path=physical,
                meta=ResourceRegistration(
                    kind=kind,
                    mime_type=mime_type,
                    original_name=os.path.basename(name),
                    size=size,
                    content_hash=content_hash,
                    temporary=temporary,
                ),
            )
        except Exception as exc:
            await self._best_effort_delete(physical)
            raise StorageBackendError(
                code="storage_backend.register_failed",
                message="register stored resource failed",
                details={"reason": str(exc)},
            ) from exc

    async def _bind_knowledge(self, ref: str, knowledge_id: str) -> None:
        try:
            await self._catalog.bind(ref, "knowledge", knowledge_id, "source_file")
        except Exception as exc:
            await self._best_effort_delete(ref)
            raise StorageBackendError(
                code="storage_backend.bind_failed",
                message="bind stored resource failed",
                details={"reason": str(exc)},
            ) from exc

    async def _resolve(self, value: str) -> tuple[str, bool]:
        physical, resource = await self._catalog.resolve_path(value)
        return physical, resource is not None

    async def _best_effort_delete(self, reference: str) -> None:
        """Remove a freshly registered object after a later step failed.

        Mirrors the upstream best-effort cleanup: the primary error is
        already being raised, so a failed cleanup only logs.
        """
        try:
            await self.delete_file(reference)
        except Exception:
            logger.warning("resource catalog cleanup delete failed for {!r}", reference)


def new_resource_catalog_file_service(
    inner: FileService | None, catalog: ResourceCatalog | None
) -> FileService | None:
    """Wrap ``inner`` with the catalog decorator.

    Returns ``inner`` unchanged when either argument is missing, so a
    deployment without a resource catalog still gets a working physical
    file service.
    """
    if inner is None or catalog is None:
        return inner
    return ResourceCatalogFileService(inner=inner, catalog=catalog)


__all__ = [
    "ResourceCatalog",
    "ResourceCatalogFileService",
    "ResourceRegistration",
    "new_resource_catalog_file_service",
    "resource_kind",
]
