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

import json
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

# PR-30.6c H2: storage columns that must not cross into the
# service-output projection per AGENTS.md §9. ``deleted_at`` is the
# soft-delete tombstone; the service layer treats a missing row as the
# only delete signal.
MODEL_EXCLUDE_COLUMNS: frozenset[str] = frozenset({"deleted_at"})


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

        Hydrates ``parameters`` from the JSON column and redacts
        ``api_key`` / ``app_secret`` so they never cross the service
        boundary in plaintext.

        - ``MODEL_EXCLUDE_COLUMNS`` (frozen per §9) drops
          ``deleted_at`` before ``model_validate``.
        - ``parameters`` is decoded from a raw JSON string when the
          storage layer persists it as text (SQLite path) so the
          downstream layer never has to handle the unparsed blob.
        - Sensitive parameter values are substituted with
          ``REDACTED_SECRET_PLACEHOLDER`` so a buggy caller that
          bypasses the wire-layer masking still cannot leak the raw
          credential.
        """
        record = db.model_dump(exclude=MODEL_EXCLUDE_COLUMNS)
        parameters = record.get("parameters")
        if isinstance(parameters, str):
            try:
                parameters = json.loads(parameters)
            except json.JSONDecodeError:
                parameters = {}
        if not isinstance(parameters, dict):
            parameters = {}
        for field_name in _SENSITIVE_PARAMETER_FIELDS:
            if parameters.get(field_name):
                parameters[field_name] = REDACTED_SECRET_PLACEHOLDER
        record["parameters"] = parameters
        return cls.model_validate(record)


__all__ = ["MODEL_EXCLUDE_COLUMNS", "ModelInfo", "REDACTED_SECRET_PLACEHOLDER"]
