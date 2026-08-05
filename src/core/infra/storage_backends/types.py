"""Internal DTOs for the `infra.storage_backends` domain.

Service-side projections of a `storage_backends` row plus the typed view
of its JSON ``config`` column. Mirrors
``internal/types/storagebackend.go``: the field names, the provider
allow-list, the per-provider required-field validation, the secret
masking / merge rules, and the location key that decides which physical
destination a row points at.
"""

from __future__ import annotations

import os
import posixpath
from datetime import datetime
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field

from src.common.exception import ValidationError
from src.common.json import JsonObject
from src.db.models.storage_backend import (
    STORAGE_BACKEND_STATUS_ACTIVE,
    STORAGE_BACKEND_STATUS_DISABLED,
    StorageBackend,
)

# Canonical provider names in display order — ``storageallowlist.supported``.
SUPPORTED_PROVIDERS: Final[tuple[str, ...]] = (
    "local",
    "minio",
    "cos",
    "tos",
    "s3",
    "oss",
    "ks3",
    "obs",
)

# ``STORAGE_ALLOW_LIST`` narrows the supported set for a deployment.
STORAGE_ALLOW_LIST_ENV: Final = "STORAGE_ALLOW_LIST"

# Separators accepted inside ``STORAGE_ALLOW_LIST`` (Go's ``FieldsFunc``).
_ALLOW_LIST_SEPARATORS: Final = ",;|\n\t "

# Placeholder a masked secret is replaced with. Sending it back on an
# update means "keep the stored value" — Go's
# ``types.RedactedSecretPlaceholder`` (internal/types/secret.go).
REDACTED_SECRET_PLACEHOLDER: Final = "***"

# Storage-only column of a `storage_backends` row.
_BACKEND_EXCLUDE_COLUMNS: frozenset[str] = frozenset({"deleted_at"})


def allowed_providers() -> tuple[str, ...]:
    """Return the providers permitted by ``STORAGE_ALLOW_LIST``.

    An unset or blank value allows every supported provider; otherwise the
    env value is intersected with the supported set, preserving canonical
    display order. Mirrors ``storageallowlist.AllowedList``.
    """
    raw = os.environ.get(STORAGE_ALLOW_LIST_ENV, "").strip()
    if not raw:
        return SUPPORTED_PROVIDERS
    requested: set[str] = set()
    token = ""
    for char in raw:
        if char in _ALLOW_LIST_SEPARATORS:
            if token:
                requested.add(token.strip().lower())
            token = ""
        else:
            token += char
    if token:
        requested.add(token.strip().lower())
    return tuple(p for p in SUPPORTED_PROVIDERS if p in requested)


def is_provider_allowed(provider: str) -> bool:
    """True when ``provider`` is permitted. An empty name is allowed.

    Matches ``storageallowlist.IsAllowed`` — the empty case exists so a
    caller that has not chosen a provider yet is not rejected here.
    """
    normalized = provider.strip().lower()
    if not normalized:
        return True
    return normalized in allowed_providers()


class StorageBackendConfigInfo(BaseModel):
    """Typed view of the ``config`` JSON column.

    Normalized union of provider-specific settings.
    ``access_key_id``/``secret_access_key`` map to COS ``SecretID``/
    ``SecretKey`` and to the access/secret pair of S3-compatible
    providers.
    """

    model_config = ConfigDict(frozen=True)

    mode: str = ""
    endpoint: str = ""
    region: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""
    bucket_name: str = ""
    path_prefix: str = ""
    app_id: str = ""
    use_ssl: bool = False
    force_path_style: bool = False
    use_temp_bucket: bool = False
    temp_bucket_name: str = ""
    temp_region: str = ""

    @classmethod
    def from_json(cls, raw: JsonObject | str | None) -> StorageBackendConfigInfo:
        """Build from the column value, tolerating ``None`` and raw text."""
        if raw is None or raw == "":
            return cls()
        if isinstance(raw, str):
            return cls.model_validate_json(raw)
        # Drop nulls: the Go struct uses zero values, not nullable fields.
        return cls.model_validate({k: v for k, v in raw.items() if v is not None})

    def to_json(self) -> JsonObject:
        """Serialise for the JSONB column."""
        return dict(self.model_dump())

    def mask_sensitive_fields(self) -> StorageBackendConfigInfo:
        """Return a copy with both credentials replaced by the placeholder.

        Empty credentials stay empty so the UI can tell "never set" from
        "set but hidden". Mirrors ``MaskSensitiveFields``.
        """
        updates: dict[str, str] = {}
        if self.access_key_id:
            updates["access_key_id"] = REDACTED_SECRET_PLACEHOLDER
        if self.secret_access_key:
            updates["secret_access_key"] = REDACTED_SECRET_PLACEHOLDER
        return self.model_copy(update=updates)

    def merge_secrets(self, existing: StorageBackendConfigInfo) -> StorageBackendConfigInfo:
        """Restore stored credentials where the incoming value is redacted.

        Mirrors ``MergeSecrets`` + ``PreserveIfRedacted``: the placeholder
        (or a blank) means "unchanged", anything else is a rotation.
        """
        return self.model_copy(
            update={
                "access_key_id": _preserve_if_redacted(self.access_key_id, existing.access_key_id),
                "secret_access_key": _preserve_if_redacted(
                    self.secret_access_key, existing.secret_access_key
                ),
            }
        )

    def location_key(self, provider: str) -> str:
        """Identify the physical destination this config points at.

        Credentials deliberately do not participate, so they can be
        rotated without changing object identity. Mirrors ``LocationKey``.
        """
        mode = self.mode.strip()
        if provider == "minio" and not mode:
            mode = "remote"
        return "|".join(
            [
                provider,
                mode,
                self.endpoint.strip(),
                self.region.strip(),
                self.bucket_name.strip(),
                self.path_prefix.strip().strip("/"),
            ]
        )

    def validate_for_provider(self, provider: str) -> None:
        """Raise ``ValidationError`` when a required field is missing.

        Mirrors ``ValidateForProvider``: ``path_prefix`` must be relative
        and traversal-free for every provider; the required credential set
        then varies by provider (``local`` needs nothing, ``minio`` in
        docker mode inherits the process env, ``cos`` is region-based, and
        everything else is endpoint-based).
        """
        self._validate_path_prefix()
        if provider == "local":
            return
        if provider == "minio":
            self._validate_minio()
            return
        if provider == "cos":
            self._require_all(("region", "access_key_id", "secret_access_key", "bucket_name"))
            return
        self._require_all(
            ("endpoint", "region", "access_key_id", "secret_access_key", "bucket_name")
        )

    # ── Validation internals ────────────────────────────────────────

    def _validate_path_prefix(self) -> None:
        prefix = self.path_prefix.strip().replace("\\", "/")
        clean = posixpath.normpath(prefix) if prefix else "."
        if prefix.startswith("/") or clean == ".." or clean.startswith("../"):
            raise ValidationError(
                code="storage_backend.invalid_path_prefix",
                message="path_prefix must be a relative path without parent traversal",
            )

    def _validate_minio(self) -> None:
        # ``docker`` mode reads endpoint + credentials from the process
        # environment, so only the bucket is required on the row.
        mode = self.mode or "remote"
        if mode != "docker":
            self._require_all(("endpoint", "access_key_id", "secret_access_key"))
        self._require_all(("bucket_name",))

    def _require_all(self, names: tuple[str, ...]) -> None:
        for name in names:
            value = getattr(self, name)
            if not str(value).strip():
                raise ValidationError(
                    code="storage_backend.missing_config_field",
                    message=f"{name} is required",
                )


def _preserve_if_redacted(incoming: str, existing: str) -> str:
    """Return ``existing`` when ``incoming`` is blank or the placeholder."""
    if not incoming or incoming == REDACTED_SECRET_PLACEHOLDER:
        return existing
    return incoming


class StorageBackendInfo(BaseModel):
    """Service-side projection of a `storage_backends` row.

    Drops the soft-delete marker (every read filters it) and hydrates the
    typed config from the JSON column.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: int
    name: str
    provider: str
    config: StorageBackendConfigInfo = Field(default_factory=StorageBackendConfigInfo)
    source: str
    status: str
    legacy_alias: bool = False
    created_at: datetime
    updated_at: datetime

    @classmethod
    def map_from_db(cls, db: StorageBackend) -> Self:
        """Project a storage row to the service DTO."""
        record = db.model_dump(exclude=set(_BACKEND_EXCLUDE_COLUMNS))
        record["config"] = StorageBackendConfigInfo.from_json(db.config)
        return cls.model_validate(record)

    def masked(self) -> Self:
        """Return a copy with credentials masked — ``NewStorageBackendResponse``."""
        return self.model_copy(update={"config": self.config.mask_sensitive_fields()})

    @property
    def is_active(self) -> bool:
        """True when the row's status is ``active``."""
        return self.status == STORAGE_BACKEND_STATUS_ACTIVE

    @property
    def is_disabled(self) -> bool:
        """True when the row's status is ``disabled``."""
        return self.status == STORAGE_BACKEND_STATUS_DISABLED


class StorageBackendListResult(BaseModel):
    """A workspace's backends plus its default-backend pointer.

    The list endpoint returns both in one payload
    (``{"data": [...], "default_storage_backend_id": ...}``), so the
    service returns them together rather than making the router issue a
    second call.
    """

    model_config = ConfigDict(frozen=True)

    backends: list[StorageBackendInfo] = Field(default_factory=list)
    default_storage_backend_id: str | None = None


class StorageConnectivityResult(BaseModel):
    """Outcome of a connectivity probe.

    A failed probe is a 200 response with ``success=false`` and a
    sanitized message, so the failure travels as data rather than as an
    exception (``TestRaw`` / ``TestByID``).
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    error: str | None = None


__all__ = [
    "REDACTED_SECRET_PLACEHOLDER",
    "STORAGE_ALLOW_LIST_ENV",
    "SUPPORTED_PROVIDERS",
    "StorageBackendConfigInfo",
    "StorageBackendInfo",
    "StorageBackendListResult",
    "StorageConnectivityResult",
    "allowed_providers",
    "is_provider_allowed",
]
