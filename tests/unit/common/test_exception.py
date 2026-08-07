"""Unit tests for :mod:`src.common.exception`.

Mirrors the salestech-be ``tests/unit/common/test_exception.py`` path.
Covers the ``ApplicationError`` base defaults, constructor overrides,
the subclass ``code`` / ``message`` contract, and the MRO of the
external-service error family.
"""

from __future__ import annotations

import pytest

from src.common.exception import (
    AIProviderError,
    ApplicationError,
    ConflictError,
    DataError,
    ExternalServiceError,
    NotFoundError,
    PermissionDeniedError,
    StorageBackendError,
    UnauthorizedError,
    ValidationError,
    VectorStoreError,
)

# ── Base defaults ───────────────────────────────────────────────────


def test_application_error_defaults() -> None:
    err = ApplicationError()
    assert err.code == "internal_error"
    assert err.message == "Internal error"
    assert err.details is None


def test_application_error_is_exception() -> None:
    assert isinstance(ApplicationError(), Exception)


def test_application_error_message_override() -> None:
    err = ApplicationError("boom")
    assert err.message == "boom"
    assert str(err) == "boom"


def test_application_error_code_override() -> None:
    assert ApplicationError(code="custom_code").code == "custom_code"


def test_application_error_details_override() -> None:
    err = ApplicationError(details={"key": "value"})
    assert err.details == {"key": "value"}


def test_application_error_overrides_preserve_other_defaults() -> None:
    err = ApplicationError("msg", code="c", details={"k": 1})
    assert err.message == "msg"
    assert err.code == "c"
    assert err.details == {"k": 1}


# ── Subclass code/message contract ───────────────────────────────────


_SUBCLASS_CONTRACT = [
    (NotFoundError, "not_found", "Resource not found"),
    (ConflictError, "conflict", "Resource conflict"),
    (ValidationError, "validation_error", "Validation failed"),
    (PermissionDeniedError, "permission_denied", "Permission denied"),
    (UnauthorizedError, "unauthorized", "Unauthorized"),
    (ExternalServiceError, "external_service_error", "External service error"),
    (AIProviderError, "ai_provider_error", "AI provider error"),
    (VectorStoreError, "vector_store_error", "Vector store error"),
    (StorageBackendError, "storage_backend_error", "Storage backend error"),
    (DataError, "data_error", "Data error"),
]


@pytest.mark.parametrize(
    ("cls", "code", "message"),
    _SUBCLASS_CONTRACT,
    ids=[c[0].__name__ for c in _SUBCLASS_CONTRACT],
)
def test_subclass_contract(cls: type, code: str, message: str) -> None:
    err = cls()
    assert err.code == code
    assert err.message == message
    assert isinstance(err, ApplicationError)


def test_subclass_message_override_keeps_code() -> None:
    err = NotFoundError("missing thing")
    assert err.message == "missing thing"
    assert err.code == "not_found"


# ── MRO ─────────────────────────────────────────────────────────────


def test_ai_provider_error_is_external_service_error() -> None:
    err = AIProviderError()
    assert isinstance(err, ExternalServiceError)
    assert isinstance(err, ApplicationError)


def test_external_service_subclasses_distinct_codes() -> None:
    codes = {
        AIProviderError().code,
        VectorStoreError().code,
        StorageBackendError().code,
    }
    assert codes == {
        "ai_provider_error",
        "vector_store_error",
        "storage_backend_error",
    }


# ── Builtin collision avoidance ─────────────────────────────────────


def test_permission_denied_not_builtin_permission_error() -> None:
    # The hierarchy exposes ``PermissionDeniedError`` precisely because a
    # ``PermissionError`` subclass would collide with the Python builtin.
    err = PermissionDeniedError()
    assert not isinstance(err, PermissionError)
    assert isinstance(err, ApplicationError)
