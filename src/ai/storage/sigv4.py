"""AWS Signature V4 request signing for S3-compatible storage.

Extracted so the MinIO / S3 / OBS / TOS / OSS / KS3 / COS adapters share
one implementation of the signing algorithm rather than a copy per
provider. Pure computation: no I/O, no network, no dependency on any
other layer.

Supports the connectivity probe (a bodyless ``HEAD``/``GET`` with an
empty-payload hash) plus the file operations (``PUT``/``GET``/``DELETE``
with an optional body payload) and presigned GET URLs. Multipart uploads
and streaming signatures remain out of scope.
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


def _payload_hash(payload: bytes | None) -> str:
    """Return the payload hash — empty-payload hash when ``payload`` is None."""
    if payload is None:
        return EMPTY_PAYLOAD_SHA256
    return hashlib.sha256(payload).hexdigest()


def sign_request(
    *,
    method: str,
    host: str,
    path: str,
    region: str,
    access_key_id: str,
    secret_access_key: str,
    now: datetime | None = None,
    payload: bytes | None = None,
) -> dict[str, str]:
    """Return the signed headers for an S3-compatible request.

    ``host`` is the ``Host`` header value (with port when non-default),
    ``path`` the absolute request path. ``payload`` hashes the request
    body; omitting it signs the empty-payload hash (bodyless probes).
    The returned mapping carries ``Host``, ``x-amz-date``,
    ``x-amz-content-sha256`` and ``Authorization``, and is passed
    straight to the HTTP client.
    """
    stamp = now or datetime.now(UTC)
    amz_date = stamp.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = stamp.strftime("%Y%m%d")
    body_hash = _payload_hash(payload)

    signed_headers = "host;x-amz-content-sha256;x-amz-date"
    canonical_headers = f"host:{host}\nx-amz-content-sha256:{body_hash}\nx-amz-date:{amz_date}\n"
    canonical_request = "\n".join(
        [
            method.upper(),
            canonical_uri(path),
            "",  # no query string on object requests
            canonical_headers,
            signed_headers,
            body_hash,
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
        "x-amz-content-sha256": body_hash,
        "Authorization": authorization,
    }


def presign_get_url(
    *,
    scheme: str,
    host: str,
    path: str,
    region: str,
    access_key_id: str,
    secret_access_key: str,
    expires_seconds: int = 86400,
    now: datetime | None = None,
) -> str:
    """Build a SigV4-presigned GET URL valid for ``expires_seconds``.

    The signature covers the canonical URI plus the ``X-Amz-*`` query
    parameters, so the URL works without a credential exchange on the
    download side. Used for ``GetFileURL`` on the S3-compatible
    backends.
    """
    stamp = now or datetime.now(UTC)
    amz_date = stamp.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = stamp.strftime("%Y%m%d")
    scope = f"{date_stamp}/{region}/{_SERVICE}/{_REQUEST_TYPE}"

    params = [
        ("X-Amz-Algorithm", _ALGORITHM),
        ("X-Amz-Credential", f"{access_key_id}/{scope}"),
        ("X-Amz-Date", amz_date),
        ("X-Amz-Expires", str(expires_seconds)),
        ("X-Amz-SignedHeaders", "host"),
    ]
    canonical_query = "&".join(
        f"{urllib.parse.quote(k, safe='')}={urllib.parse.quote(v, safe='')}" for k, v in params
    )
    canonical_request = "\n".join(
        [
            "GET",
            canonical_uri(path),
            canonical_query,
            f"host:{host}\n",
            "host",
            "UNSIGNED-PAYLOAD",
        ]
    )
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
    signed_url = f"{scheme}://{host}{canonical_uri(path)}"
    return f"{signed_url}?{canonical_query}&X-Amz-Signature={signature}"


__all__ = [
    "EMPTY_PAYLOAD_SHA256",
    "canonical_uri",
    "presign_get_url",
    "sign_request",
]
