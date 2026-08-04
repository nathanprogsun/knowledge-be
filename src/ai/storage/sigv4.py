"""AWS Signature V4 request signing for S3-compatible storage probes.

Extracted so the MinIO / S3 / OBS adapters share one implementation of
the signing algorithm rather than three copies. Pure computation: no I/O,
no network, no dependency on any other layer.

Only what a connectivity probe needs is implemented — a bodyless request
(``HEAD``/``GET``) with an empty-payload hash. Multipart uploads and
streaming signatures are out of scope.
"""

from __future__ import annotations

import hashlib
import hmac
import urllib.parse
from datetime import UTC, datetime
from typing import Final

# SHA-256 of the empty string — the payload hash of every bodyless request.
EMPTY_PAYLOAD_SHA256: Final = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

_ALGORITHM: Final = "AWS4-HMAC-SHA256"
_SERVICE: Final = "s3"
_REQUEST_TYPE: Final = "aws4_request"


def _sign(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def _signing_key(secret_access_key: str, date_stamp: str, region: str) -> bytes:
    """Derive the date/region/service-scoped signing key."""
    date_key = _sign(f"AWS4{secret_access_key}".encode(), date_stamp)
    region_key = _sign(date_key, region)
    service_key = _sign(region_key, _SERVICE)
    return _sign(service_key, _REQUEST_TYPE)


def canonical_uri(path: str) -> str:
    """Percent-encode ``path`` per SigV4 rules, preserving ``/``."""
    if not path.startswith("/"):
        path = "/" + path
    return urllib.parse.quote(path, safe="/~")


def sign_request(
    *,
    method: str,
    host: str,
    path: str,
    region: str,
    access_key_id: str,
    secret_access_key: str,
    now: datetime | None = None,
) -> dict[str, str]:
    """Return the signed headers for a bodyless S3-compatible request.

    ``host`` is the ``Host`` header value (with port when non-default) and
    ``path`` the absolute request path. The returned mapping carries
    ``Host``, ``x-amz-date``, ``x-amz-content-sha256`` and
    ``Authorization``, and is passed straight to the HTTP client.
    """
    stamp = now or datetime.now(UTC)
    amz_date = stamp.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = stamp.strftime("%Y%m%d")

    signed_headers = "host;x-amz-content-sha256;x-amz-date"
    canonical_headers = (
        f"host:{host}\nx-amz-content-sha256:{EMPTY_PAYLOAD_SHA256}\nx-amz-date:{amz_date}\n"
    )
    canonical_request = "\n".join(
        [
            method.upper(),
            canonical_uri(path),
            "",  # no query string on a bucket probe
            canonical_headers,
            signed_headers,
            EMPTY_PAYLOAD_SHA256,
        ]
    )
    scope = f"{date_stamp}/{region}/{_SERVICE}/{_REQUEST_TYPE}"
    string_to_sign = "\n".join(
        [
            _ALGORITHM,
            amz_date,
            scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )
    signature = hmac.new(
        _signing_key(secret_access_key, date_stamp, region),
        string_to_sign.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    authorization = (
        f"{_ALGORITHM} Credential={access_key_id}/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    return {
        "Host": host,
        "x-amz-date": amz_date,
        "x-amz-content-sha256": EMPTY_PAYLOAD_SHA256,
        "Authorization": authorization,
    }


__all__ = ["EMPTY_PAYLOAD_SHA256", "canonical_uri", "sign_request"]
