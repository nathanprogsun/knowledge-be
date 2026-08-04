"""Tencent COS storage adapter.

Mirrors ``internal/application/service/file/cos.go`` +
``CheckCosConnectivity``: COS is addressed by region rather than by a
user-supplied endpoint, so the host is derived as
``{bucket}[-{app_id}].cos.{region}.myqcloud.com`` and probed with a signed
``HEAD``. ``access_key_id`` / ``secret_access_key`` carry the COS
``SecretID`` / ``SecretKey`` pair (the normalized config union renames
them; the values are the same).
"""

from __future__ import annotations

from typing import Final

from src.ai.storage.base import head_bucket
from src.common.exception import StorageBackendError

PROVIDER_COS: Final = "cos"

# COS buckets are always addressed virtual-host style on this suffix.
_COS_SERVICE_HOST_TEMPLATE: Final = "https://cos.{region}.myqcloud.com"


class CosStorageAdapter:
    """Probe for a Tencent Cloud COS backend."""

    def __init__(
        self,
        *,
        region: str,
        access_key_id: str,
        secret_access_key: str,
        bucket_name: str,
        app_id: str = "",
    ) -> None:
        self._region = region.strip()
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._bucket_name = _qualified_bucket(bucket_name, app_id)
        # Region-derived service host; the bucket becomes the leading DNS
        # label, giving ``{bucket}.cos.{region}.myqcloud.com``.
        self._endpoint_url = _COS_SERVICE_HOST_TEMPLATE.format(region=self._region)

    async def check_connectivity(self) -> None:
        """Signed ``HEAD`` on the region-derived bucket host."""
        if not self._region:
            raise StorageBackendError(
                code="storage_backend.region_required",
                message="COS connectivity check requires a region",
            )
        await head_bucket(
            endpoint_url=self._endpoint_url,
            bucket_name=self._bucket_name,
            region=self._region,
            access_key_id=self._access_key_id,
            secret_access_key=self._secret_access_key,
            force_path_style=False,
            provider_label="COS",
        )


def _qualified_bucket(bucket_name: str, app_id: str) -> str:
    """Append the COS app id when the bucket name lacks the suffix.

    COS bucket names are globally ``{name}-{appid}``; the UI accepts either
    form, so the suffix is added only when missing.
    """
    name = bucket_name.strip()
    suffix = app_id.strip()
    if not suffix or not name or name.endswith(f"-{suffix}"):
        return name
    return f"{name}-{suffix}"


__all__ = ["PROVIDER_COS", "CosStorageAdapter"]
