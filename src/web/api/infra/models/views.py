"""Wire-shape conversion for the model endpoints.

``ModelInfo`` is the service-side projection of a ``models`` row; the
wire shape is the frozen ``Model`` in
``src/core/contracts/infra.py``. ``model_info_to_contract`` performs
the boundary translation: it re-emits the storage row onto the wire
contract, omitting the credential-bearing ``parameters`` fields.

The wire ``Model`` carries the raw ``ModelParameters`` (which still
includes ``api_key`` / ``app_secret`` in its field set). This module
explicitly strips those two fields when projecting so a response can
never carry plaintext credentials — mirrors
``dto.NewModelResponse`` on the Go side.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.common.json import JsonValue
from src.core.contracts.infra import (
    CredentialFieldMetadata,
    Model,
    ModelParameters,
    ProviderTypeMeta,
)
from src.core.infra.models.types import ModelInfo


class ModelEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` - single-model responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: Model


class ModelListEnvelope(BaseModel):
    """``{"success": true, "data": [...]}`` - list responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: list[Model]


class ProviderListEnvelope(BaseModel):
    """``{"success": true, "data": [...]}`` - provider metadata responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: list[ProviderTypeMeta]


class DeleteModelResponse(BaseModel):
    """``{"success": true, "message": "..."}`` - simple ack response."""

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str


class ModelDebugEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` - debug call responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: dict[str, JsonValue]


def _redact_credential(value: str | None) -> str:
    """Return the wire placeholder for a stored credential.

    Go's ``dto.NewModelResponse`` emits ``"sk-***"`` when an
    ``api_key`` is set and ``""`` when it is empty — the ``sk-`` prefix
    is preserved so the UI can tell at a glance whether a key is
    configured without revealing its value.
    """
    if not value:
        return ""
    return "sk-***"


def _parameters_for_wire(parameters: ModelParameters) -> ModelParameters:
    """Strip ``api_key`` / ``app_secret`` for the wire response.

    Mirrors Go's ``dto.ModelParametersDTO`` which masks both fields by
    construction. We rebuild a fresh ``ModelParameters`` with the two
    sensitive fields replaced by the wire placeholder.
    """
    return parameters.model_copy(
        update={
            "api_key": _redact_credential(parameters.api_key),
            "app_secret": _redact_credential(parameters.app_secret),
        }
    )


def model_info_to_contract(info: ModelInfo) -> Model:
    """Project the service DTO onto the frozen wire contract."""
    parameters = info.parameters
    return Model(
        id=info.id,
        tenant_id=info.tenant_id,
        name=info.name,
        display_name=info.display_name,
        type=info.type,
        source=info.source,
        description=info.description,
        parameters=_parameters_for_wire(parameters),
        is_default=info.is_default,
        is_builtin=info.is_builtin,
        status=info.status,
        created_at=info.created_at,
        updated_at=info.updated_at,
        # Go emits the per-field "configured?" map on the response so the
        # UI can render credential presence without ever seeing values.
        credentials={
            "api_key": CredentialFieldMetadata(configured=bool(parameters.api_key)),
            "app_secret": CredentialFieldMetadata(configured=bool(parameters.app_secret)),
        },
    )


def model_envelope(info: ModelInfo) -> ModelEnvelope:
    """Wrap one model in the success envelope."""
    return ModelEnvelope(success=True, data=model_info_to_contract(info))


def model_list_envelope(infos: list[ModelInfo]) -> ModelListEnvelope:
    """Wrap a list of models in the success envelope."""
    return ModelListEnvelope(
        success=True,
        data=[model_info_to_contract(info) for info in infos],
    )


def provider_list_envelope(providers: list[ProviderTypeMeta]) -> ProviderListEnvelope:
    """Wrap a list of provider metadata in the success envelope."""
    return ProviderListEnvelope(success=True, data=providers)


__all__ = [
    "DeleteModelResponse",
    "ModelDebugEnvelope",
    "ModelEnvelope",
    "ModelListEnvelope",
    "ProviderListEnvelope",
    "model_envelope",
    "model_info_to_contract",
    "model_list_envelope",
    "provider_list_envelope",
]
