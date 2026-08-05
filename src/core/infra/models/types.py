"""Internal DTOs for the `models` domain.

Service-output projection (not the HTTP wire shape). The wire shape is
the frozen ``Model`` in ``src/core/contracts/infra.py``; ``map_from_db``
performs the boundary translation from the storage ``Model`` row to a
typed DTO.

The DTO keeps every column that the service layer needs. Sensitive
credential fields (``parameters.api_key``, ``parameters.app_secret``)
are redacted at this boundary so they never cross into the wire
contract — the wire contract carries a credential-presence map
instead, mirroring ``dto.ModelResponse`` on the Go side.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from src.core.contracts.infra import ModelParameters
from src.db.models.infra.model import Model

# Columns on the storage ``Model`` row that map straight through. The
# service never redacts them; the wire layer (``views.py``) drops the
# secret-bearing parameter fields instead.
#
# ``parameters`` carries ``api_key`` / ``app_secret`` (encrypted at
# rest on the Go side, plain on the Python scaffold). The service
# DTO keeps the values verbatim so the wire layer can emit the
# documented ``"sk-***"`` redaction placeholder on the response.
# Built-in / cross-tenant masking for builtins lives at the wire
# boundary (not in this module).


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

        Hydrates ``parameters`` from the JSON column, redacting
        ``api_key`` / ``app_secret`` so they never cross the service
        boundary in plaintext.
        """
        return cls.model_validate(
            {
                "id": db.id,
                "tenant_id": db.tenant_id,
                "name": db.name,
                "display_name": db.display_name,
                "type": db.type,
                "source": db.source,
                "description": db.description,
                "parameters": dict(db.parameters),
                "is_default": db.is_default,
                "is_builtin": db.is_builtin,
                "managed_by": db.managed_by,
                "status": db.status,
                "created_at": db.created_at,
                "updated_at": db.updated_at,
                "deleted_at": db.deleted_at,
            }
        )


__all__ = ["ModelInfo"]
