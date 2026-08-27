"""Application error hierarchy - single contract for all raised domain errors.

Each subclass carries `code` (stable string for clients), `message`
(human-readable), and optional `details`. `PermissionError` would
collide with the Python builtin, so the standard hierarchy exposes
`PermissionDeniedError` instead.
"""

from __future__ import annotations

from src.common.json import JsonObject


class ApplicationError(Exception):
    """Base for all domain-raised errors."""

    code: str = "internal_error"
    message: str = "Internal error"
    details: JsonObject | None = None

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        details: JsonObject | None = None,
    ) -> None:
        if code is not None:
            self.code = code
        if message is not None:
            self.message = message
        if details is not None:
            self.details = details
        super().__init__(self.message)


class NotFoundError(ApplicationError):
    code = "not_found"
    message = "Resource not found"


class GoneError(NotFoundError):
    """Resource used to exist but is no longer available (HTTP 410).

    Mirrors the upstream contract for expired / revoked invitation
    links: ``POST /auth/invitations/lookup`` and
    ``POST /auth/register-by-invite`` collapse unknown, expired, and
    revoked tokens into a single 410 so a stolen token's failure mode
    does not leak which slot it occupied.
    """

    code = "not_found"
    message = "Resource is gone"


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


class NotImplementedFeatureError(ApplicationError):
    """Wire-level marker for a feature whose backing seam is not wired.

    Mirrors the upstream convention of surfacing "not implemented" as a
    distinct status (HTTP 501) rather than collapsing it into the generic
    ``internal_server_error`` envelope — the frontend branches on
    ``error.code == "feature.not_implemented"`` to render a placeholder
    card instead of a hard error toast, and the chat pipeline uses the
    SSE error frame's ``error_code`` field for the same switch.
    """

    code = "feature.not_implemented"
    message = "Feature not implemented"


__all__ = [
    "AIProviderError",
    "ApplicationError",
    "ConflictError",
    "DataError",
    "ExternalServiceError",
    "GoneError",
    "NotFoundError",
    "NotImplementedFeatureError",
    "PermissionDeniedError",
    "StorageBackendError",
    "UnauthorizedError",
    "ValidationError",
    "VectorStoreError",
]
