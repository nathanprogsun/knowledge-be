"""File-service factory dispatching on the storage provider name.

``new_file_service_from_storage_config`` builds a provider-specific
``FileService`` from the normalized storage config (the ai-layer view of
a storage-backend row's ``config`` column). The provider may be blank, in
which case the config's ``default_provider`` is used.

Validation mirrors the upstream factory: each provider requires its
credential set, blank values fall back to the process environment, and
unknown providers are refused with ``StorageBackendError``.
"""

from __future__ import annotations

import os
import posixpath
from typing import Final, Protocol

from src.ai.storage.base import FileService
from src.ai.storage.cos_backend import CosStorageAdapter
from src.ai.storage.ks3_backend import KS3FileService
from src.ai.storage.local_backend import (
    DEFAULT_LOCAL_BASE_DIR,
    LocalStorageAdapter,
    safe_path_under_base,
)
from src.ai.storage.minio_backend import MINIO_MODE_REMOTE, MinioStorageAdapter
from src.ai.storage.obs_backend import ObsStorageAdapter
from src.ai.storage.oss_backend import OssFileService
from src.ai.storage.s3_backend import S3StorageAdapter
from src.ai.storage.tos_backend import TosFileService
from src.common.exception import StorageBackendError, ValidationError

# Fallback base dir when neither the argument nor the env provides one.
_LOCAL_BASE_DIR_ENV: Final = "LOCAL_STORAGE_BASE_DIR"

# Default object-key prefix applied by the S3-shaped providers.
_DEFAULT_S3_PREFIX: Final = "weknora/"

# Default COS prefix (no trailing slash; COS joins with ``/``).
_DEFAULT_COS_PREFIX: Final = "weknora"

# Default OBS region when the config leaves it blank.
_DEFAULT_OBS_REGION: Final = "cn-north-4"


class StorageConfig(Protocol):
    """Structural view of the normalized storage config.

    ``core``'s ``StorageBackendConfigInfo`` (a frozen pydantic model)
    satisfies this protocol; tests can build a plain frozen dataclass
    with the same fields.
    """

    @property
    def default_provider(self) -> str: ...

    @property
    def mode(self) -> str: ...

    @property
    def endpoint(self) -> str: ...

    @property
    def region(self) -> str: ...

    @property
    def access_key_id(self) -> str: ...

    @property
    def secret_access_key(self) -> str: ...

    @property
    def bucket_name(self) -> str: ...

    @property
    def path_prefix(self) -> str: ...

    @property
    def app_id(self) -> str: ...

    @property
    def use_ssl(self) -> bool: ...

    @property
    def force_path_style(self) -> bool: ...

    @property
    def use_temp_bucket(self) -> bool: ...

    @property
    def temp_bucket_name(self) -> str: ...

    @property
    def temp_region(self) -> str: ...


def new_file_service_from_storage_config(
    provider: str, config: StorageConfig | None = None, local_base_dir: str | None = None
) -> tuple[FileService, str]:
    """Build the provider's file service and return it with the provider name.

    ``provider`` is normalised (trimmed + lowercased); a blank value
    falls back to ``config.default_provider``. Raises
    ``StorageBackendError`` for a missing or unsupported provider or an
    incomplete credential set.
    """
    name = provider.strip().lower()
    if not name and config is not None:
        name = str(getattr(config, "default_provider", "") or "").strip().lower()
    if not name:
        raise StorageBackendError(
            code="storage_backend.empty_provider", message="empty provider"
        )

    base_dir = (local_base_dir or "").strip()
    if not base_dir:
        base_dir = os.environ.get(_LOCAL_BASE_DIR_ENV, "").strip()
    if not base_dir:
        base_dir = DEFAULT_LOCAL_BASE_DIR

    if name == "local":
        return _build_local(config, base_dir), name
    if name == "minio":
        return _build_minio(config), name
    if name == "cos":
        return _build_cos(config), name
    if name == "tos":
        return _build_tos(config), name
    if name == "s3":
        return _build_s3(config), name
    if name == "obs":
        return _build_obs(config), name
    if name == "oss":
        return _build_oss(config), name
    if name == "ks3":
        return _build_ks3(config), name
    raise StorageBackendError(
        code="storage_backend.unsupported_provider",
        message=f"unsupported provider {name!r}",
    )


# ── Per-provider builders ──────────────────────────────────────────────


def _build_local(config: StorageConfig | None, base_dir: str) -> FileService:
    prefix = _field(config, "path_prefix").strip().strip("/\\")
    if prefix:
        candidate = posixpath.join(base_dir, prefix)
        try:
            safe_path_under_base(base_dir, candidate)
        except ValidationError:
            # The upstream factory keeps the plain base dir when the
            # prefix would escape it.
            prefix = ""
    external_url = os.environ.get("APP_EXTERNAL_URL", "").strip()
    return LocalStorageAdapter(
        path_prefix=prefix, base_dir=base_dir, external_url=external_url
    )


def _build_minio(config: StorageConfig | None) -> FileService:
    if config is None:
        raise _incomplete("minio")
    mode = _field(config, "mode") or MINIO_MODE_REMOTE
    if mode == MINIO_MODE_REMOTE:
        endpoint = _field(config, "endpoint")
        access_key_id = _field(config, "access_key_id")
        secret_access_key = _field(config, "secret_access_key")
    else:
        endpoint = os.environ.get("MINIO_ENDPOINT", "").strip()
        access_key_id = os.environ.get("MINIO_ACCESS_KEY_ID", "").strip()
        secret_access_key = os.environ.get("MINIO_SECRET_ACCESS_KEY", "").strip()
    bucket_name = _field(config, "bucket_name") or os.environ.get("MINIO_BUCKET_NAME", "").strip()
    if not endpoint or not access_key_id or not secret_access_key or not bucket_name:
        raise _incomplete("minio")
    return MinioStorageAdapter(
        endpoint=endpoint,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        bucket_name=bucket_name,
        use_ssl=_bool(config, "use_ssl"),
        region=_field(config, "region"),
        mode=mode,
    )


def _build_cos(config: StorageConfig | None) -> FileService:
    _require(config, "cos", ("region", "access_key_id", "secret_access_key", "bucket_name"))
    assert config is not None
    path_prefix = _field(config, "path_prefix") or _DEFAULT_COS_PREFIX
    return CosStorageAdapter(
        region=_field(config, "region"),
        access_key_id=_field(config, "access_key_id"),
        secret_access_key=_field(config, "secret_access_key"),
        bucket_name=_field(config, "bucket_name"),
        app_id=_field(config, "app_id"),
        path_prefix=path_prefix,
        temp_bucket_name=_field(config, "temp_bucket_name"),
        temp_region=_field(config, "temp_region"),
    )


def _build_tos(config: StorageConfig | None) -> FileService:
    _require(config, "tos", ("endpoint", "region", "access_key_id", "secret_access_key", "bucket_name"))
    assert config is not None
    return TosFileService(
        endpoint=_field(config, "endpoint"),
        region=_field(config, "region"),
        access_key=_field(config, "access_key_id"),
        secret_key=_field(config, "secret_access_key"),
        bucket_name=_field(config, "bucket_name"),
        path_prefix=_field(config, "path_prefix"),
        temp_bucket_name=_field(config, "temp_bucket_name"),
        temp_region=_field(config, "temp_region"),
    )


def _build_s3(config: StorageConfig | None) -> FileService:
    _require(config, "s3", ("endpoint", "region", "access_key_id", "secret_access_key", "bucket_name"))
    assert config is not None
    return S3StorageAdapter(
        endpoint=_field(config, "endpoint"),
        region=_field(config, "region"),
        access_key_id=_field(config, "access_key_id"),
        secret_access_key=_field(config, "secret_access_key"),
        bucket_name=_field(config, "bucket_name"),
        use_ssl=_bool(config, "use_ssl"),
        force_path_style=_bool(config, "force_path_style"),
        path_prefix=_field(config, "path_prefix") or _DEFAULT_S3_PREFIX,
        provider_label="S3",
    )


def _build_obs(config: StorageConfig | None) -> FileService:
    endpoint = _field(config, "endpoint") or os.environ.get("OBS_ENDPOINT", "").strip()
    region = _field(config, "region") or os.environ.get("OBS_REGION", "").strip()
    access_key_id = _field(config, "access_key_id") or os.environ.get("OBS_ACCESS_KEY", "").strip()
    secret_access_key = _field(config, "secret_access_key") or os.environ.get("OBS_SECRET_KEY", "").strip()
    bucket_name = _field(config, "bucket_name") or os.environ.get("OBS_BUCKET_NAME", "").strip()
    path_prefix = _field(config, "path_prefix") or os.environ.get("OBS_PATH_PREFIX", "").strip()
    if not path_prefix:
        path_prefix = _DEFAULT_S3_PREFIX
    if not endpoint or not access_key_id or not secret_access_key or not bucket_name:
        raise _incomplete("obs")
    if not region:
        region = _DEFAULT_OBS_REGION
    return ObsStorageAdapter(
        endpoint=endpoint,
        region=region,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        bucket_name=bucket_name,
        use_ssl=_bool(config, "use_ssl") if config is not None else True,
        path_prefix=path_prefix,
    )


def _build_oss(config: StorageConfig | None) -> FileService:
    _require(config, "oss", ("endpoint", "region", "access_key_id", "secret_access_key", "bucket_name"))
    assert config is not None
    path_prefix = _field(config, "path_prefix") or _DEFAULT_S3_PREFIX
    if _bool(config, "use_temp_bucket") and _field(config, "temp_bucket_name"):
        return OssFileService(
            endpoint=_field(config, "endpoint"),
            region=_field(config, "region"),
            access_key=_field(config, "access_key_id"),
            secret_key=_field(config, "secret_access_key"),
            bucket_name=_field(config, "bucket_name"),
            path_prefix=path_prefix,
            temp_bucket_name=_field(config, "temp_bucket_name"),
            temp_region=_field(config, "temp_region"),
        )
    return OssFileService(
        endpoint=_field(config, "endpoint"),
        region=_field(config, "region"),
        access_key=_field(config, "access_key_id"),
        secret_key=_field(config, "secret_access_key"),
        bucket_name=_field(config, "bucket_name"),
        path_prefix=path_prefix,
    )


def _build_ks3(config: StorageConfig | None) -> FileService:
    _require(config, "ks3", ("endpoint", "region", "access_key_id", "secret_access_key", "bucket_name"))
    assert config is not None
    return KS3FileService(
        endpoint=_field(config, "endpoint"),
        region=_field(config, "region"),
        access_key=_field(config, "access_key_id"),
        secret_key=_field(config, "secret_access_key"),
        bucket_name=_field(config, "bucket_name"),
        path_prefix=_field(config, "path_prefix") or _DEFAULT_S3_PREFIX,
    )


# ── Config access helpers ──────────────────────────────────────────────


def _field(config: StorageConfig | None, name: str) -> str:
    """Read a string field, tolerating a missing or null value."""
    if config is None:
        return ""
    value = getattr(config, name, "") or ""
    return str(value).strip()


def _bool(config: StorageConfig | None, name: str) -> bool:
    """Read a boolean field, defaulting to ``False``."""
    if config is None:
        return False
    value = getattr(config, name, False) or False
    return bool(value)


def _require(config: StorageConfig | None, provider: str, names: tuple[str, ...]) -> None:
    """Raise ``StorageBackendError`` when any required field is blank."""
    if config is None or any(not _field(config, name) for name in names):
        raise _incomplete(provider)


def _incomplete(provider: str) -> StorageBackendError:
    return StorageBackendError(
        code="storage_backend.incomplete_config", message=f"incomplete {provider} config"
    )


__all__ = ["new_file_service_from_storage_config"]
