"""Web exception handler invariants.

Pins the contract that every error response — business ``ApplicationError``
or an uncaught exception — uses the standard envelope
``{"success": false, "error": {"code", "message"}}``.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from src.common.exception import NotFoundError, ValidationError
from src.core.contracts.shared import ErrorDetail, ErrorResponse
from src.web.exception_handler import (
    application_error_handler,
    register_exception_handlers,
    uncaught_exception_handler,
)


def _build_app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/raise_application_error")
    async def raise_application_error() -> None:
        raise NotFoundError(code="demo.not_found", message="not here")

    @app.get("/raise_validation_error")
    async def raise_validation_error() -> None:
        raise ValidationError(code="demo.bad_input", message="bad input")

    @app.get("/raise_uncaught")
    async def raise_uncaught() -> None:
        raise RuntimeError("boom")

    @app.get("/raise_http_exception")
    async def raise_http_exception() -> None:
        raise HTTPException(status_code=418, detail="teapot")

    class _Echo(BaseModel):
        name: str
        age: int

    @app.post("/echo")
    async def echo(body: _Echo) -> dict[str, str]:
        return {"name": body.name}

    return app


def test_application_error_returns_envelope() -> None:
    client = TestClient(_build_app(), raise_server_exceptions=False)
    response = client.get("/raise_application_error")
    assert response.status_code == 404
    body = response.json()
    assert body == ErrorResponse(
        success=False,
        error=ErrorDetail(code="demo.not_found", message="not here"),
    ).model_dump(mode="json")


def test_validation_error_status_422() -> None:
    client = TestClient(_build_app(), raise_server_exceptions=False)
    response = client.get("/raise_validation_error")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "demo.bad_input"


def test_request_validation_error_uses_app_envelope() -> None:
    """FastAPI body validation is wrapped in the standard envelope (PR-146 B3).

    Without the explicit handler, FastAPI emits ``{"detail": [...]}`` for
    malformed bodies. We override to keep the contract single-shape so
    the frontend does not have to branch on envelope type.
    """
    client = TestClient(_build_app(), raise_server_exceptions=False)
    response = client.post("/echo", json={"age": "not-an-int"})
    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "request.validation_error"
    assert body["error"]["message"] == "Request validation failed"
    # Per-field Pydantic diagnostics are preserved in ``details``.
    assert isinstance(body["error"]["details"], list)
    assert body["error"]["details"], "details must carry the per-field errors"


def test_uncaught_exception_returns_envelope_with_status_500() -> None:
    client = TestClient(_build_app(), raise_server_exceptions=False)
    response = client.get("/raise_uncaught")
    assert response.status_code == 500
    body = response.json()
    # Envelope shape (success=false, error.code, error.message).
    assert body["success"] is False
    assert "error" in body
    assert body["error"]["code"] == "internal_server_error"
    assert body["error"]["message"] == "Internal server error"
    # Internal details must not leak.
    assert "boom" not in str(body)


def test_http_exception_passthrough() -> None:
    """FastAPI's built-in ``HTTPException`` keeps its default 4xx shape.

    The fallback handler must not intercept ``HTTPException``; FastAPI's
    own handler still applies so framework-level responses (raise from
    dependencies, 405, etc.) stay recognizable.
    """
    client = TestClient(_build_app(), raise_server_exceptions=False)
    response = client.get("/raise_http_exception")
    # FastAPI's default handler is invoked before our fallback because it
    # is registered earlier. The status code is preserved.
    assert response.status_code == 418


def test_envelope_model_round_trip() -> None:
    """``ErrorResponse`` keeps producing the expected JSON shape."""
    payload = ErrorResponse(
        success=False,
        error=ErrorDetail(code="x.y", message="m", details=None),
    ).model_dump(mode="json")
    assert payload == {"success": False, "error": {"code": "x.y", "message": "m", "details": None}}


def test_handlers_exported() -> None:
    """Both handlers must remain importable for direct use in tests."""
    assert callable(application_error_handler)
    assert callable(uncaught_exception_handler)
