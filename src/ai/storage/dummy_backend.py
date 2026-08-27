"""No-op storage adapter for deployments without file storage.

``DummyFileService`` satisfies the full ``FileService`` interface but
performs no I/O: uploads return random ``dummy://`` paths, deletes
succeed silently and copies return the source unchanged. Useful for
testing and for running the platform with file storage disabled.
"""

from __future__ import annotations

import uuid
from typing import BinaryIO, Final

from loguru import logger

from src.ai.storage.base import FileUpload
from src.common.exception import StorageBackendError

DUMMY_SCHEME: Final = "dummy://"


class DummyFileService:
    """A FileService that never touches a real storage backend."""

    async def check_connectivity(self) -> None:
        """Always succeeds for the dummy service."""
        return

    async def save_file(self, *, file: FileUpload, tenant_id: int, knowledge_id: str) -> str:
        """Pretend to save the file, returning a random ``dummy://`` path."""
        return f"{DUMMY_SCHEME}{tenant_id}/{uuid.uuid4()}"

    async def save_bytes(self, *, data: bytes, tenant_id: int, file_name: str, temp: bool) -> str:
        """Pretend to save the bytes, returning a random ``dummy://`` path."""
        return f"{DUMMY_SCHEME}{tenant_id}/{uuid.uuid4()}"

    async def get_file(self, file_path: str) -> BinaryIO:
        """The dummy service stores nothing, so reads are unsupported."""
        raise StorageBackendError(
            code="storage_backend.not_implemented",
            message="get file is not implemented for the dummy storage backend",
        )

    async def get_file_url(self, file_path: str) -> str:
        """Return the path unchanged as its "download URL"."""
        return file_path

    async def delete_file(self, file_path: str) -> None:
        """No-op — there is nothing to delete."""
        return

    async def copy_file(self, src_path: str, tenant_id: int, knowledge_id: str) -> str:
        """No-op copy: log a warning and return the source unchanged."""
        logger.warning(
            "[dummy] copy_file no-op: returning source path {!r} unchanged "
            "(no real copy performed)",
            src_path,
        )
        return src_path


__all__ = ["DUMMY_SCHEME", "DummyFileService"]
