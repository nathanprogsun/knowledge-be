"""Backend-instance scoping decorator for a file service.

``BackendScopedFileService`` makes the concrete storage instance part of
every newly persisted path (``storage://<backendID>/<provider>://...``)
while delegating actual I/O to the underlying provider driver. Reads and
deletes unwrap the instance prefix first, refusing a path that names a
different backend.
"""

from __future__ import annotations

import time
import urllib.parse
from typing import BinaryIO, Final

from src.ai.storage.base import (
    FileService,
    FileUpload,
    build_storage_backend_path,
    parse_storage_backend_path,
    parse_tenant_id_from_storage_path,
    sign_file_url,
)
from src.common.exception import StorageBackendError

# URL path the local backend emits for presigned downloads.
_PRESIGNED_URL_PATH: Final = "/api/v1/files/presigned"


class BackendScopedFileService:
    """Decorates a ``FileService`` with a storage-backend instance id."""

    def __init__(self, *, backend_id: str, inner: FileService) -> None:
        self._backend_id = backend_id.strip()
        self._inner = inner

    async def check_connectivity(self) -> None:
        """Delegate to the inner service."""
        await self._inner.check_connectivity()

    async def save_file(
        self, *, file: FileUpload, tenant_id: int, knowledge_id: str
    ) -> str:
        """Save via the inner service and wrap the returned path."""
        path = await self._inner.save_file(
            file=file, tenant_id=tenant_id, knowledge_id=knowledge_id
        )
        return self._wrap(path)

    async def save_bytes(
        self, *, data: bytes, tenant_id: int, file_name: str, temp: bool
    ) -> str:
        """Save bytes via the inner service and wrap the returned path."""
        path = await self._inner.save_bytes(
            data=data, tenant_id=tenant_id, file_name=file_name, temp=temp
        )
        return self._wrap(path)

    async def get_file(self, file_path: str) -> BinaryIO:
        """Unwrap the path, then delegate to the inner service."""
        inner_path = self._unwrap(file_path)
        return await self._inner.get_file(inner_path)

    async def get_file_url(self, file_path: str) -> str:
        """Unwrap the path and delegate URL generation.

        When the inner service returns a presigned URL for the unwrapped
        path, it is re-signed for the scoped path so the file proxy
        resolves the exact local instance instead of falling back to
        another backend of the same provider.
        """
        inner_path = self._unwrap(file_path)
        result = await self._inner.get_file_url(inner_path)
        scoped = self._wrap(inner_path)
        if result == inner_path:
            return scoped
        re_signed = self._re_sign_presigned(result, inner_path, scoped)
        return re_signed if re_signed is not None else result

    async def delete_file(self, file_path: str) -> None:
        """Unwrap the path, then delegate to the inner service."""
        inner_path = self._unwrap(file_path)
        await self._inner.delete_file(inner_path)

    async def copy_file(self, src_path: str, tenant_id: int, knowledge_id: str) -> str:
        """Unwrap the source, copy via the inner service, wrap the result."""
        inner_path = self._unwrap(src_path)
        result = await self._inner.copy_file(
            src_path=inner_path, tenant_id=tenant_id, knowledge_id=knowledge_id
        )
        return self._wrap(result)

    # ── Wrapping ────────────────────────────────────────────────────

    def _wrap(self, path: str) -> str:
        return build_storage_backend_path(self._backend_id, path)

    def _unwrap(self, path: str) -> str:
        parsed = parse_storage_backend_path(path)
        if parsed is None:
            return path
        backend_id, inner = parsed
        if backend_id != self._backend_id:
            raise StorageBackendError(
                code="storage_backend.backend_mismatch",
                message=f"storage backend mismatch: got {backend_id}, want {self._backend_id}",
            )
        return inner

    def _re_sign_presigned(self, result: str, inner_path: str, scoped: str) -> str | None:
        """Re-sign a local presigned URL for the scoped path.

        Returns ``None`` when ``result`` is not a presigned URL for
        ``inner_path`` (the caller returns it unchanged).
        """
        parsed = urllib.parse.urlsplit(result)
        if not parsed.path.endswith(_PRESIGNED_URL_PATH):
            return None
        query = urllib.parse.parse_qs(parsed.query)
        if query.get("file_path", [""])[0] != inner_path:
            return None
        base_path = parsed.path[: -len(_PRESIGNED_URL_PATH)]
        base_url = f"{parsed.scheme}://{parsed.netloc}{base_path}"
        ttl = 0
        expires = query.get("expires", [""])[0]
        if expires.isdigit():
            ttl = int(expires) - int(time.time())
        try:
            return sign_file_url(
                base_url=base_url,
                file_path=scoped,
                tenant_id=parse_tenant_id_from_storage_path(scoped),
                ttl=ttl,
            )
        except StorageBackendError:
            return None


__all__ = ["BackendScopedFileService"]
