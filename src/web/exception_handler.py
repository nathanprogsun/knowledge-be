"""Exception handler mapping ``ApplicationError`` subclasses to HTTP status.

The web layer is the only layer that maps errors to HTTP status. The
MRO walk lets subclasses inherit their parent's status without each
appearing in the map explicitly.

A fallback handler wraps every uncaught exception in the standard error
envelope (``{"success": false, "error": {"code", "message"}}``) so that
clients can rely on the response shape regardless of the error class.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.ai.mcp_transport.errors import OAuthRequiredError
from src.app_logging import logger
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

#: Code used when an uncaught exception reaches the fallback handler.
_UNCAUGHT_EXCEPTION_CODE = "internal_server_error"
#: Public-facing message for the fallback envelope (the underlying
#: exception is logged but never exposed to the client).
_UNCAUGHT_EXCEPTION_MESSAGE = "Internal server error"

_STATUS_BY_TYPE: dict[type[ApplicationError], int] = {
    NotFoundError: 404,
    ConflictError: 409,
    ValidationError: 422,
    PermissionDeniedError: 403,
    UnauthorizedError: 401,
    OAuthRequiredError: 401,
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


async def uncaught_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Wrap any uncaught exception in the standard error envelope.

    Logs the exception with stack trace at ERROR (server-side only) and
    returns a generic 500 to the client. Mirrors the structured shape of
    ``application_error_handler`` so clients can rely on a single error
    response schema.
    """
    logger.exception(
        "uncaught exception serving %s %s: %s",
        request.method,
        request.url.path,
        exc,
    )
    envelope = ErrorResponse(
        success=False,
        error=ErrorDetail(
            code=_UNCAUGHT_EXCEPTION_CODE,
            message=_UNCAUGHT_EXCEPTION_MESSAGE,
        ),
    )
    return JSONResponse(
        status_code=500,
        content=envelope.model_dump(mode="json"),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register the ``ApplicationError`` handler on the FastAPI app."""
    app.add_exception_handler(ApplicationError, application_error_handler)  # type: ignore[arg-type]
    # Catch-all fallback must be registered last so the specific
    # ``ApplicationError`` handler still wins for known business errors.
    app.add_exception_handler(Exception, uncaught_exception_handler)  # type: ignore[arg-type]


__all__ = [
    "application_error_handler",
    "register_exception_handlers",
    "uncaught_exception_handler",
]
