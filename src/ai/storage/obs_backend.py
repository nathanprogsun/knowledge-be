"""Huawei OBS storage adapter.

Mirrors ``internal/application/service/file/obs.go`` +
``CheckObsConnectivity``: OBS speaks the S3 API through a custom endpoint
resolver and is probed path-style (Go passes ``UsePathStyle: true``), so
the adapter fixes that flag rather than reading it from the row.
"""

from __future__ import annotations

from typing import Final

from src.ai.storage.base import head_bucket, normalize_endpoint
from src.common.exception import StorageBackendError

PROVIDER_OBS: Final = "obs"

# Go's OBS client is constructed with ``UsePathStyle: true``.
_FORCE_PATH_STYLE: Final = True


class ObsStorageAdapter:
    """Probe for a Huawei Cloud OBS backend."""

    def __init__(
        self,
        *,
        endpoint: str,
        region: str,
        access_key_id: str,
        secret_access_key: str,
        bucket_name: str,
        use_ssl: bool = True,
    ) -> None:
        self._endpoint_url = normalize_endpoint(endpoint, use_ssl=use_ssl)
        self._region = region
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._bucket_name = bucket_name

    async def check_connectivity(self) -> None:
        """Signed path-style ``HEAD`` on the configured bucket."""
        if not self._endpoint_url:
            raise StorageBackendError(
                code="storage_backend.endpoint_required",
                message="OBS connectivity check requires an endpoint",
            )
        await head_bucket(
            endpoint_url=self._endpoint_url,
            bucket_name=self._bucket_name,
            region=self._region,
            access_key_id=self._access_key_id,
            secret_access_key=self._secret_access_key,
            force_path_style=_FORCE_PATH_STYLE,
            provider_label="OBS",
        )


__all__ = ["PROVIDER_OBS", "ObsStorageAdapter"]
