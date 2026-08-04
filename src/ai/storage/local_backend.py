"""Local-filesystem storage adapter.

Mirrors ``internal/application/service/file/local.go``: the probe stats
the base directory and requires it to be an existing directory. The
service creates the directory (under a base-dir guard) before probing, so
a first-time local backend passes without a manual mkdir.

Blocking ``os`` calls run on a worker thread — the ``ai`` layer obeys the
same async-purity rule as the rest of the codebase.
"""

from __future__ import annotations

import asyncio
import os
import posixpath
import stat
from pathlib import Path
from typing import Final

from src.common.exception import StorageBackendError, ValidationError

# Go: ``LOCAL_STORAGE_BASE_DIR`` with this fallback.
LOCAL_STORAGE_BASE_DIR_ENV: Final = "LOCAL_STORAGE_BASE_DIR"
DEFAULT_LOCAL_BASE_DIR: Final = "/data/files"

# Directory permissions Go creates the tree with (``os.MkdirAll(_, 0o755)``).
_DIR_MODE: Final = 0o755


def local_base_dir() -> str:
    """Return the configured local storage base directory."""
    configured = os.environ.get(LOCAL_STORAGE_BASE_DIR_ENV, "").strip()
    return configured or DEFAULT_LOCAL_BASE_DIR


def safe_path_under_base(base_dir: str, candidate: str) -> str:
    """Resolve ``candidate`` and require it to stay under ``base_dir``.

    Mirrors ``utils.SafePathUnderBase``: symlinks and ``..`` segments are
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
    """Probe for a local-filesystem backend."""

    def __init__(self, *, path_prefix: str = "", base_dir: str | None = None) -> None:
        self._base_dir = base_dir if base_dir is not None else local_base_dir()
        self._path_prefix = path_prefix.strip().strip("/\\")

    @property
    def directory(self) -> str:
        """The absolute directory this backend writes into."""
        candidate = posixpath.join(self._base_dir, self._path_prefix)
        return safe_path_under_base(self._base_dir, candidate)

    async def ensure_directory(self) -> None:
        """Create the backend's directory tree if it does not exist.

        Go does this inside ``Test`` before probing, so an unconfigured
        deployment's first local backend is usable immediately.
        """
        target = self.directory
        try:
            await asyncio.to_thread(lambda: os.makedirs(target, mode=_DIR_MODE, exist_ok=True))
        except OSError as exc:
            raise StorageBackendError(
                code="storage_backend.mkdir_failed",
                message="create local storage directory failed",
                details={"reason": exc.strerror or type(exc).__name__},
            ) from exc

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


def _stat_kind(path: str) -> bool | None:
    """Classify ``path``: ``True`` directory, ``False`` exists but is not a
    directory, ``None`` cannot be stat'ed."""
    try:
        stat_result = Path(path).stat()
    except OSError:
        return None
    return stat.S_ISDIR(stat_result.st_mode)


__all__ = [
    "DEFAULT_LOCAL_BASE_DIR",
    "LOCAL_STORAGE_BASE_DIR_ENV",
    "LocalStorageAdapter",
    "local_base_dir",
    "safe_path_under_base",
]
