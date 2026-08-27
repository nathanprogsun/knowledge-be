"""Local-filesystem storage adapter (full file interface).

The probe stats the base directory and requires it to be an existing
directory. The service creates the directory (under a base-dir guard)
before probing, so a first-time local backend passes without a manual
mkdir.

The file operations mirror the upstream local file service: objects are
stored under ``baseDir/{tenantID}/{knowledgeID}/`` (``save_file``) or
``baseDir/{tenantID}/exports/`` (``save_bytes``) and addressed as
``local://{relative_path}``. ``get_file``/``delete_file``/``copy_file``
accept provider paths, absolute paths and legacy relative paths, always
resolved under the base-dir guard so a crafted path cannot escape.

Blocking ``os`` calls run on a worker thread — the ``ai`` layer obeys the
same async-purity rule as the rest of the codebase.
"""

from __future__ import annotations

import asyncio
import os
import posixpath
import stat
import time
from pathlib import Path
from typing import BinaryIO, Final

from src.ai.storage.base import (
    FileUpload,
    parse_tenant_id_from_storage_path,
    safe_file_name,
    sign_file_url,
)
from src.ai.storage.errors import CrossBackendCopyError
from src.common.exception import StorageBackendError, ValidationError

# ``LOCAL_STORAGE_BASE_DIR`` with this fallback.
LOCAL_STORAGE_BASE_DIR_ENV: Final = "LOCAL_STORAGE_BASE_DIR"
DEFAULT_LOCAL_BASE_DIR: Final = "/data/files"

# Provider scheme this backend returns for stored objects.
LOCAL_SCHEME: Final = "local://"

# External-URL env consumed by ``get_file_url`` presigning.
APP_EXTERNAL_URL_ENV: Final = "APP_EXTERNAL_URL"

# Directory permissions used to create the storage tree (0o755).
_DIR_MODE: Final = 0o755


def local_base_dir() -> str:
    """Return the configured local storage base directory."""
    configured = os.environ.get(LOCAL_STORAGE_BASE_DIR_ENV, "").strip()
    return configured or DEFAULT_LOCAL_BASE_DIR


def safe_path_under_base(base_dir: str, candidate: str) -> str:
    """Resolve ``candidate`` and require it to stay under ``base_dir``.

    Mirrors ``SafePathUnderBase``: symlinks and ``..`` segments are
    normalised away before the containment test, so a crafted
    ``path_prefix`` cannot escape the storage root.
    """
    base = Path(base_dir).expanduser()
    resolved_base = Path(posixpath.normpath(str(base)))
    resolved = Path(posixpath.normpath(str(Path(candidate).expanduser())))
    if resolved != resolved_base and resolved_base not in resolved.parents:
        raise ValidationError(
            code="storage_backend.path_escapes_base",
            message="resolved storage path escapes the configured base directory",
        )
    return str(resolved)


class LocalStorageAdapter:
    """Full file service on a local-filesystem backend."""

    def __init__(
        self,
        *,
        path_prefix: str = "",
        base_dir: str | None = None,
        external_url: str | None = None,
    ) -> None:
        resolved = base_dir if base_dir is not None else local_base_dir()
        prefix = path_prefix.strip().strip("/\\")
        if prefix:
            candidate = posixpath.join(resolved, prefix)
            # The prefix is baked into the base dir so every operation
            # (probe, save, read, delete) shares one storage root.
            resolved = safe_path_under_base(resolved, candidate)
        self._base_dir = resolved
        configured = (
            external_url if external_url is not None else os.environ.get(APP_EXTERNAL_URL_ENV, "")
        )
        self._external_url = configured.strip().rstrip("/")

    @property
    def directory(self) -> str:
        """The absolute directory this backend writes into."""
        return self._base_dir

    async def ensure_directory(self) -> None:
        """Create the backend's directory tree if it does not exist.

        The upstream ``Test`` creates the tree before probing, so an
        unconfigured deployment's first local backend is usable
        immediately.
        """
        await _mkdir(self.directory)

    async def check_connectivity(self) -> None:
        """Require the backend's directory to exist and be a directory."""
        target = self.directory
        stat_result = await asyncio.to_thread(_stat_kind, target)
        if stat_result is None:
            raise StorageBackendError(
                code="storage_backend.directory_not_accessible",
                message="storage directory not accessible",
            )
        if not stat_result:
            raise StorageBackendError(
                code="storage_backend.not_a_directory",
                message="storage path is not a directory",
            )

    # ── File operations ─────────────────────────────────────────────

    async def save_file(self, *, file: FileUpload, tenant_id: int, knowledge_id: str) -> str:
        """Store an uploaded file under ``{tenant}/{knowledge}/``.

        Returns the ``local://`` relative path of the new object.
        """
        directory = posixpath.join(self._base_dir, str(tenant_id), knowledge_id)
        safe_path_under_base(self._base_dir, directory)
        await _mkdir(directory)

        ext = os.path.splitext(file.filename)[1]
        filename = f"{time.time_ns()}{ext}"
        file_path = posixpath.join(directory, filename)
        data = await file.read()
        try:
            await asyncio.to_thread(_write_bytes, file_path, data)
        except OSError as exc:
            raise StorageBackendError(
                code="storage_backend.write_failed",
                message="failed to save file",
                details={"reason": exc.strerror or type(exc).__name__},
            ) from exc
        return LOCAL_SCHEME + self._relative(file_path)

    async def save_bytes(self, *, data: bytes, tenant_id: int, file_name: str, temp: bool) -> str:
        """Persist raw bytes under ``{tenant}/exports/``.

        ``temp`` is ignored — a local filesystem has no auto-expiring
        store, so the bytes land in the same layout either way.
        """
        safe_name = safe_file_name(file_name)
        directory = posixpath.join(self._base_dir, str(tenant_id), "exports")
        safe_path_under_base(self._base_dir, directory)
        await _mkdir(directory)

        ext = os.path.splitext(safe_name)[1]
        base_name = safe_name[: len(safe_name) - len(ext)] if ext else safe_name
        unique = f"{base_name}_{time.time_ns()}{ext}"
        file_path = posixpath.join(directory, unique)
        try:
            await asyncio.to_thread(_write_bytes, file_path, data)
        except OSError as exc:
            raise StorageBackendError(
                code="storage_backend.write_failed",
                message="failed to write file",
                details={"reason": exc.strerror or type(exc).__name__},
            ) from exc
        return LOCAL_SCHEME + self._relative(file_path)

    async def get_file(self, file_path: str) -> BinaryIO:
        """Open a stored object for reading.

        Accepts ``local://`` paths, absolute paths under the base dir and
        legacy relative paths. The caller owns the returned handle.
        """
        candidate = self._normalize_path_for_base(file_path)
        resolved = safe_path_under_base(self._base_dir, candidate)
        try:
            return await asyncio.to_thread(_open_binary, resolved)
        except OSError as exc:
            raise StorageBackendError(
                code="storage_backend.open_failed",
                message="failed to open file",
                details={"reason": exc.strerror or type(exc).__name__},
            ) from exc

    async def delete_file(self, file_path: str) -> None:
        """Remove a stored object, guarding against path traversal."""
        candidate = self._normalize_path_for_base(file_path)
        resolved = safe_path_under_base(self._base_dir, candidate)
        try:
            await asyncio.to_thread(os.remove, resolved)
        except OSError as exc:
            raise StorageBackendError(
                code="storage_backend.delete_failed",
                message="failed to delete file",
                details={"reason": exc.strerror or type(exc).__name__},
            ) from exc

    async def copy_file(self, src_path: str, tenant_id: int, knowledge_id: str) -> str:
        """Copy a local object to a new knowledge-owned object.

        The copy is byte-for-byte (no hardlink), so deleting the source
        never affects it. A source with a foreign provider scheme raises
        ``CrossBackendCopyError``.
        """
        if "://" in src_path and not src_path.startswith(LOCAL_SCHEME):
            raise CrossBackendCopyError(message=f"local file service cannot copy {src_path!r}")
        src_candidate = self._normalize_path_for_base(src_path)
        src_resolved = safe_path_under_base(self._base_dir, src_candidate)

        directory = posixpath.join(self._base_dir, str(tenant_id), knowledge_id)
        safe_path_under_base(self._base_dir, directory)
        await _mkdir(directory)

        ext = os.path.splitext(src_path)[1]
        filename = f"{time.time_ns()}{ext}"
        dst_path = posixpath.join(directory, filename)
        try:
            data = await asyncio.to_thread(_read_bytes, src_resolved)
            await asyncio.to_thread(_write_bytes, dst_path, data)
        except OSError as exc:
            raise StorageBackendError(
                code="storage_backend.copy_failed",
                message="failed to copy file",
                details={"reason": exc.strerror or type(exc).__name__},
            ) from exc
        return LOCAL_SCHEME + self._relative(dst_path)

    async def get_file_url(self, file_path: str) -> str:
        """Return a download URL for the object.

        With ``APP_EXTERNAL_URL`` configured the result is an HMAC-signed
        URL served by the file proxy; otherwise the ``local://`` path is
        returned unchanged (backward-compatible).
        """
        normalized = self._to_scheme_path(file_path)
        if not self._external_url:
            return normalized
        tenant_id = parse_tenant_id_from_storage_path(normalized)
        try:
            return sign_file_url(
                base_url=self._external_url, file_path=normalized, tenant_id=tenant_id
            )
        except StorageBackendError:
            return normalized

    # ── Path normalisation ──────────────────────────────────────────

    def _relative(self, file_path: str) -> str:
        rel = os.path.relpath(file_path, self._base_dir)
        return posixpath.normpath(rel)

    def _to_scheme_path(self, file_path: str) -> str:
        """Convert an absolute/relative path into ``local://`` form."""
        if file_path.startswith(LOCAL_SCHEME):
            return file_path
        try:
            rel = os.path.relpath(file_path, self._base_dir)
        except ValueError:
            return file_path
        return LOCAL_SCHEME + posixpath.normpath(rel)

    def _normalize_path_for_base(self, file_path: str) -> str:
        """Map legacy path forms onto an absolute path under the base dir.

        - ``local://tenant/..`` → ``baseDir/tenant/..``
        - absolute path → unchanged
        - path under the base dir → joined with the base dir
        - legacy relative with the base prefix (``data/files/..``) → the
          duplicated base segment is stripped before joining
        """
        if file_path.startswith(LOCAL_SCHEME):
            rel = file_path[len(LOCAL_SCHEME) :]
            return posixpath.join(self._base_dir, rel)

        clean = os.path.normpath(file_path.strip())
        if clean in (".", ""):
            return clean
        if os.path.isabs(clean):
            return clean

        base_clean = os.path.normpath(self._base_dir)
        base_no_slash = base_clean.strip(os.sep)
        clean_no_dot = clean[2:] if clean.startswith("." + os.sep) else clean
        if base_no_slash and clean_no_dot.startswith(base_no_slash + os.sep):
            clean_no_dot = clean_no_dot[len(base_no_slash) + 1 :]
        return posixpath.join(base_clean, clean_no_dot)


async def _mkdir(target: str) -> None:
    """Create ``target`` (with parents) or raise ``StorageBackendError``."""
    try:
        await asyncio.to_thread(lambda: os.makedirs(target, mode=_DIR_MODE, exist_ok=True))
    except OSError as exc:
        raise StorageBackendError(
            code="storage_backend.mkdir_failed",
            message="create local storage directory failed",
            details={"reason": exc.strerror or type(exc).__name__},
        ) from exc


def _stat_kind(path: str) -> bool | None:
    """Classify ``path``: ``True`` directory, ``False`` exists but is not a
    directory, ``None`` cannot be stat'ed."""
    try:
        stat_result = Path(path).stat()
    except OSError:
        return None
    return stat.S_ISDIR(stat_result.st_mode)


def _write_bytes(path: str, data: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(data)


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def _open_binary(path: str) -> BinaryIO:
    return open(path, "rb")


__all__ = [
    "APP_EXTERNAL_URL_ENV",
    "DEFAULT_LOCAL_BASE_DIR",
    "LOCAL_SCHEME",
    "LOCAL_STORAGE_BASE_DIR_ENV",
    "LocalStorageAdapter",
    "local_base_dir",
    "safe_path_under_base",
]
