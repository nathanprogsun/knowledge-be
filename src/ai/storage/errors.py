"""Storage file-service error types.

``CrossBackendCopyError`` is raised by every backend's ``copy_file`` when
the source path belongs to a different storage provider. Same-provider
copies are server-side; streaming a copy across providers is
intentionally not implemented yet.
"""

from __future__ import annotations

from src.common.exception import StorageBackendError


class CrossBackendCopyError(StorageBackendError):
    """The copy source belongs to another storage provider."""

    code = "storage_backend.cross_backend_copy"
    message = "cross-backend copy not supported"


# Sentinel name kept for parity with the upstream ``ErrCrossBackendCopy``.
ErrCrossBackendCopy = CrossBackendCopyError


__all__ = ["CrossBackendCopyError", "ErrCrossBackendCopy"]
