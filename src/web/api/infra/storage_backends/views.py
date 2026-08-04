"""Wire-shape conversion for the storage-backend endpoints.

The frozen contracts in ``src/core/contracts/infra.py`` define the shapes;
this module only projects the service DTOs onto them and wraps them in the
``{"success": true, ...}`` envelopes the Go handler returns.

Credentials are already masked by the service (``NewStorageBackendResponse``
in Go does the same), so no redaction happens here.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.core.contracts.infra import (
    StorageBackend,
    StorageBackendConfig,
    StorageBackendListResponse,
)
from src.core.infra.storage_backends.types import (
    StorageBackendConfigInfo,
    StorageBackendInfo,
    StorageBackendListResult,
    StorageConnectivityResult,
)


class StorageBackendEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` — single-backend responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: StorageBackend


class StorageProviderTypesEnvelope(BaseModel):
    """``{"success": true, "data": ["local", ...]}`` — allowed provider names."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: list[str]


class StorageConnectivityEnvelope(BaseModel):
    """``{"success": bool, "error": "..."}`` — connectivity probe result.

    A failed probe is a 200 with ``success=false``; the status stays 200 so
    the UI renders the message inline instead of treating it as a request
    error (Go's ``TestRaw`` / ``TestByID``).
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    error: str | None = None


class StorageBackendAckEnvelope(BaseModel):
    """``{"success": true}`` — delete / set-default acknowledgement."""

    model_config = ConfigDict(frozen=True)

    success: bool


def config_info_to_contract(config: StorageBackendConfigInfo) -> StorageBackendConfig:
    """Project the typed config onto the frozen wire contract.

    Empty strings become ``None``: the Go struct tags every config field
    ``omitempty``, so a zero value is absent from the JSON rather than
    present-and-blank.
    """
    return StorageBackendConfig(
        mode=config.mode or None,
        endpoint=config.endpoint or None,
        region=config.region or None,
        access_key_id=config.access_key_id or None,
        secret_access_key=config.secret_access_key or None,
        bucket_name=config.bucket_name or None,
        path_prefix=config.path_prefix or None,
        app_id=config.app_id or None,
        use_ssl=config.use_ssl,
        force_path_style=config.force_path_style,
        use_temp_bucket=config.use_temp_bucket,
        temp_bucket_name=config.temp_bucket_name or None,
        temp_region=config.temp_region or None,
    )


def config_from_contract(config: StorageBackendConfig | None) -> StorageBackendConfigInfo:
    """Hydrate the service-side config from a request body.

    Absent fields fall back to the Go zero values (blank string / false),
    which is what the per-provider validation is written against.
    """
    if config is None:
        return StorageBackendConfigInfo()
    return StorageBackendConfigInfo(
        mode=config.mode or "",
        endpoint=config.endpoint or "",
        region=config.region or "",
        access_key_id=config.access_key_id or "",
        secret_access_key=config.secret_access_key or "",
        bucket_name=config.bucket_name or "",
        path_prefix=config.path_prefix or "",
        app_id=config.app_id or "",
        use_ssl=bool(config.use_ssl),
        force_path_style=bool(config.force_path_style),
        use_temp_bucket=bool(config.use_temp_bucket),
        temp_bucket_name=config.temp_bucket_name or "",
        temp_region=config.temp_region or "",
    )


def backend_info_to_contract(info: StorageBackendInfo) -> StorageBackend:
    """Project the service DTO onto the frozen wire contract."""
    return StorageBackend(
        id=info.id,
        tenant_id=info.tenant_id,
        name=info.name,
        provider=info.provider,
        config=config_info_to_contract(info.config),
        source=info.source,
        status=info.status,
        legacy_alias=info.legacy_alias,
        created_at=info.created_at,
        updated_at=info.updated_at,
        deleted_at=None,
    )


def backend_envelope(info: StorageBackendInfo) -> StorageBackendEnvelope:
    """Wrap one backend in the success envelope."""
    return StorageBackendEnvelope(success=True, data=backend_info_to_contract(info))


def backend_list_response(result: StorageBackendListResult) -> StorageBackendListResponse:
    """Wrap a workspace's backends + default id in the list contract."""
    return StorageBackendListResponse(
        success=True,
        data=[backend_info_to_contract(info) for info in result.backends],
        default_storage_backend_id=result.default_storage_backend_id,
    )


def connectivity_envelope(result: StorageConnectivityResult) -> StorageConnectivityEnvelope:
    """Wrap a probe outcome; the error message is already sanitized."""
    return StorageConnectivityEnvelope(success=result.success, error=result.error)


__all__ = [
    "StorageBackendAckEnvelope",
    "StorageBackendEnvelope",
    "StorageConnectivityEnvelope",
    "StorageProviderTypesEnvelope",
    "backend_envelope",
    "backend_info_to_contract",
    "backend_list_response",
    "config_from_contract",
    "config_info_to_contract",
    "connectivity_envelope",
]
