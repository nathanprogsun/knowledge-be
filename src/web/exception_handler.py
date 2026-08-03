"""Exception handler mapping ``ApplicationError`` subclasses to HTTP status.

Per AGENTS.md §6.6, ``web`` is the ONLY layer that knows about HTTP status.
Services raise ``ApplicationError`` subclasses; this single handler converts
them to JSON responses in the ``ErrorResponse`` wire shape
(see ``src/core/contracts/shared.py``).

The MRO walk lets subclasses of ``ExternalServiceError`` (e.g.
``AIProviderError``, ``StorageBackendError``) inherit their parent's status
without each appearing in the map explicitly.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.common.exception import (
    ApplicationError,
    ConflictError,
    DataError,
    ExternalServiceError,
    NotFoundError,
    PermissionDeniedError,
    UnauthorizedError,
    ValidationError,
)
from src.core.contracts.shared import ErrorDetail, ErrorResponse

_STATUS_BY_TYPE: dict[type[ApplicationError], int] = {
    NotFoundError: 404,
    ConflictError: 409,
    ValidationError: 422,
    PermissionDeniedError: 403,
    UnauthorizedError: 401,
    ExternalServiceError: 502,
    DataError: 500,
    ApplicationError: 500,
}


def _status_for(error: ApplicationError) -> int:
    """Resolve the HTTP status for ``error`` by walking its MRO."""
    for cls in type(error).__mro__:
        if cls in _STATUS_BY_TYPE:
            return _STATUS_BY_TYPE[cls]
    return 500


def _to_error_response(error: ApplicationError) -> ErrorResponse:
    return ErrorResponse(
        success=False,
        error=ErrorDetail(
            code=error.code,
            message=error.message,
            details=str(error.details) if error.details else None,
        ),
    )


async def application_error_handler(
    request: Request,
    exc: ApplicationError,
) -> JSONResponse:
    """Convert an ``ApplicationError`` into a JSON ``ErrorResponse``."""
    return JSONResponse(
        status_code=_status_for(exc),
        content=_to_error_response(exc).model_dump(mode="json"),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register the ``ApplicationError`` handler on the FastAPI app."""
    app.add_exception_handler(ApplicationError, application_error_handler)  # type: ignore[arg-type]


__all__ = ["application_error_handler", "register_exception_handlers"]
