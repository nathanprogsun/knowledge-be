"""Internal DTOs for the `models` domain.

Service-output projection (not the HTTP wire shape). The wire shape is
the frozen ``Model`` in ``src/core/contracts/infra.py``; ``map_from_db``
performs the boundary translation from the storage ``Model`` row to a
typed DTO.

The DTO keeps every column that the service layer needs. Sensitive
credential fields (``parameters.api_key``, ``parameters.app_secret``)
are redacted at this boundary so they never cross out of the service
layer in plaintext -- mirroring ``dto.NewModelResponse`` on the Go
side, which omits the secret fields altogether. The wire layer
(``views.py``) then translates the placeholder into the visible
``"sk-***"`` form the UI already expects.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from src.core.contracts.infra import ModelParameters
from src.db.models.infra.model import Model

# Placeholder substituted for any stored credential at the service
# boundary. Kept identical to
# ``src/core/infra/storage_backends/types.REDACTED_SECRET_PLACEHOLDER``
# and the Go ``internal/types/secret.go::RedactedSecretPlaceholder`` so
# a UI that already special-cases the value stays compatible.
REDACTED_SECRET_PLACEHOLDER: str = "***"

# Credential-bearing fields on ``parameters`` that must never leak past
# the service boundary. ``api_key`` / ``app_secret`` are the two
# provider-supplied secrets (Go ``ModelParameters.APIKey`` /
# ``AppSecret``).
_SENSITIVE_PARAMETER_FIELDS: frozenset[str] = frozenset({"api_key", "app_secret"})


class ModelInfo(BaseModel):
    """Service-side projection of a `models` row."""

    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: int
    name: str
    display_name: str | None = Field(default=None)
    type: str
    source: str
    description: str | None = Field(default=None)
    parameters: ModelParameters
    is_default: bool = False
    is_builtin: bool = False
    managed_by: str | None = Field(default=None)
    status: str | None = Field(default="active")
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = Field(default=None)

    @classmethod
    def map_from_db(cls, db: Model) -> Self:
        """Project a storage ``Model`` row to the service DTO.

        Hydrates ``parameters`` from the JSON column, deep-copying the
        blob so the credential-bearing fields (``api_key``,
        ``app_secret``) can be substituted with
        ``REDACTED_SECRET_PLACEHOLDER`` before the row leaves the
        service boundary. Empty credentials stay empty so the wire
        layer's ``credentials`` map can distinguish "set (hidden)"
        from "not set" without an extra flag -- mirroring Go's
        ``dto.NewModelResponse``, which omits the secret fields
        entirely; here a placeholder string is returned so a buggy
        caller that bypasses ``views.py`` still cannot leak the raw
        value.
        """
        parameters = dict(db.parameters)
        for field_name in _SENSITIVE_PARAMETER_FIELDS:
            if parameters.get(field_name):
                parameters[field_name] = REDACTED_SECRET_PLACEHOLDER
        return cls.model_validate(
            {
                "id": db.id,
                "tenant_id": db.tenant_id,
                "name": db.name,
                "display_name": db.display_name,
                "type": db.type,
                "source": db.source,
                "description": db.description,
                "parameters": parameters,
                "is_default": db.is_default,
                "is_builtin": db.is_builtin,
                "managed_by": db.managed_by,
                "status": db.status,
                "created_at": db.created_at,
                "updated_at": db.updated_at,
                "deleted_at": db.deleted_at,
            }
        )


__all__ = ["ModelInfo", "REDACTED_SECRET_PLACEHOLDER"]
