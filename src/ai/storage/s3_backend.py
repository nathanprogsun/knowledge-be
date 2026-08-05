"""S3 storage adapter.

Mirrors ``internal/application/service/file/s3.go`` +
``CheckS3ConnectivityWithOptions``: a signed ``HeadBucket`` against the
configured endpoint, honouring the row's ``force_path_style`` flag. The
same code path serves the S3-compatible providers that need nothing beyond
endpoint + region + path-style selection (``tos``, ``oss``, ``ks3``), so
they are constructed from this adapter with a different label.
"""

from __future__ import annotations

from typing import Final

from src.ai.storage.base import head_bucket, normalize_endpoint
from src.common.exception import StorageBackendError

PROVIDER_S3: Final = "s3"

# Region AWS SigV4 falls back to when the row leaves it blank.
DEFAULT_S3_REGION: Final = "us-east-1"


class S3StorageAdapter:
    """Probe for AWS S3 and endpoint-compatible object stores."""

    def __init__(
        self,
        *,
        endpoint: str,
        region: str,
        access_key_id: str,
        secret_access_key: str,
        bucket_name: str,
        use_ssl: bool = True,
        force_path_style: bool = False,
        provider_label: str = "S3",
    ) -> None:
        self._endpoint_url = normalize_endpoint(endpoint, use_ssl=use_ssl)
        self._region = region or DEFAULT_S3_REGION
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._bucket_name = bucket_name
        self._force_path_style = force_path_style
        self._provider_label = provider_label

    async def check_connectivity(self) -> None:
        """Signed ``HEAD`` on the bucket — 2xx means reachable + authorized."""
        if not self._endpoint_url:
            raise StorageBackendError(
                code="storage_backend.endpoint_required",
                message=f"{self._provider_label} connectivity check requires an endpoint",
            )
        await head_bucket(
            endpoint_url=self._endpoint_url,
            bucket_name=self._bucket_name,
            region=self._region,
            access_key_id=self._access_key_id,
            secret_access_key=self._secret_access_key,
            force_path_style=self._force_path_style,
            provider_label=self._provider_label,
        )


__all__ = ["DEFAULT_S3_REGION", "PROVIDER_S3", "S3StorageAdapter"]
