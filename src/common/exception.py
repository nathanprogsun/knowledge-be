"""Application error hierarchy — single contract for all raised domain errors.

Rules:
- `core`, `db`, `ai`, `workers` MUST raise `ApplicationError` subclasses only.
- `web` translates `ApplicationError` subclasses to HTTP status via one
  exception handler (added in a later PR).
- Each subclass carries `code` (stable string for clients), `message`
  (human-readable), and optional `details`.

Naming note: `PermissionError` collides with the Python builtin, so we
expose `PermissionDeniedError` in the standard hierarchy.
"""

from __future__ import annotations


class ApplicationError(Exception):
    """Base for all domain-raised errors."""

    code: str = "internal_error"
    message: str = "Internal error"
    details: dict[str, object] | None = None

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, object] | None = None,
    ) -> None:
        if message is not None:
            self.message = message
        if details is not None:
            self.details = details
        super().__init__(self.message)


class NotFoundError(ApplicationError):
    code = "not_found"
    message = "Resource not found"


class ConflictError(ApplicationError):
    code = "conflict"
    message = "Resource conflict"


class ValidationError(ApplicationError):
    code = "validation_error"
    message = "Validation failed"


class PermissionDeniedError(ApplicationError):
    code = "permission_denied"
    message = "Permission denied"


class UnauthorizedError(ApplicationError):
    code = "unauthorized"
    message = "Unauthorized"


class ExternalServiceError(ApplicationError):
    code = "external_service_error"
    message = "External service error"


class AIProviderError(ExternalServiceError):
    code = "ai_provider_error"
    message = "AI provider error"


class VectorStoreError(ExternalServiceError):
    code = "vector_store_error"
    message = "Vector store error"


class StorageBackendError(ExternalServiceError):
    code = "storage_backend_error"
    message = "Storage backend error"


class DataError(ApplicationError):
    code = "data_error"
    message = "Data error"


__all__ = [
    "AIProviderError",
    "ApplicationError",
    "ConflictError",
    "DataError",
    "ExternalServiceError",
    "NotFoundError",
    "PermissionDeniedError",
    "StorageBackendError",
    "UnauthorizedError",
    "ValidationError",
    "VectorStoreError",
]
