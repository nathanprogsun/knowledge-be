"""Cloud credential service — save + status.

Two operations:

- ``save_credentials`` — validate that both ``app_id`` and
  ``app_secret`` are present, probe the upstream ``/api/v1/health``
  endpoint with signed headers, then persist the pair into the
  workspace's ``credentials`` JSONB column under ``cloud``.
  No model rows are created (the upstream contract: "仅保存
  APPID/APPSECRET 凭证, 不自动创建模型").
- ``check_status`` — report whether the workspace has usable
  credentials and whether they need re-entering because the stored
  ``app_secret`` is still an ``enc:v1:`` blob (the AES key rotated, so
  the row-load decryption silently kept the ciphertext).

The upstream ``/api/v1/health`` is a liveness probe, so a 200 only
proves reachability. We keep the same semantics (401/403 → invalid
credentials, other non-200 → bad status, transport failure →
unreachable) rather than inventing a stricter check.
"""

from __future__ import annotations

import hashlib
import secrets
import string
import time
from typing import Final
from urllib.parse import quote

import httpx

from src.common.exception import ExternalServiceError, NotFoundError, ValidationError
from src.common.json import JsonObject
from src.core.contracts.infra import CloudStatusResponse
from src.db.dao.tenants_repository import TenantRepository

# Hard-coded upstream entry point. Path segments are appended by each caller.
KB_CLOUD_BASE_URL: Final = "https://kb.weixin.qq.com"

# The credential probe path on the upstream service.
_HEALTH_PATH: Final = "/api/v1/health"

# Per-call HTTP timeout for the verification request.
_VERIFY_TIMEOUT_SECONDS: Final = 10.0

# Prefix that marks an AES-256-GCM-encrypted secret blob at rest.
ENC_PREFIX: Final = "enc:v1:"

# Key of the provider object inside the ``credentials`` JSONB column
# (the public wire tag for this provider slot).
_CREDENTIALS_KEY: Final = "cloud"

# Nonce alphabet + length used by the upstream signing scheme.
_NONCE_CHARS: Final = string.ascii_lowercase + string.ascii_uppercase + string.digits
_NONCE_LENGTH: Final = 16

# Empty-body placeholder used for the body MD5 when signing a GET.
_EMPTY_BODY_JSON: Final = "{}"

# RFC3986 unreserved set — everything else is percent-encoded.
_RFC3986_SAFE: Final = "-_.~"

# Reason string surfaced when the stored secret could not be decrypted.
# The fullwidth punctuation is part of the user-facing copy.
_REINIT_REASON: Final = (
    "Cloud 凭证解密失败（服务重启后加密密钥已变更），请重新填写 APPID 和 APPSECRET"
)


def _md5_hex(value: str) -> str:
    """Hex MD5 of ``value`` (required by the upstream signing scheme).

    MD5 here is a protocol requirement of the upstream signing scheme,
    not a security primitive of ours.
    """
    return hashlib.md5(value.encode("utf-8"), usedforsecurity=False).hexdigest()


def _generate_nonce(length: int = _NONCE_LENGTH) -> str:
    """Random alphanumeric nonce in the upstream signing scheme."""
    return "".join(secrets.choice(_NONCE_CHARS) for _ in range(length))


def _rfc3986_encode(value: str) -> str:
    """Percent-encode all but ``A-Za-z0-9-_.~`` (RFC3986 unreserved set)."""
    return quote(value, safe=_RFC3986_SAFE)


def sign_request_headers(
    *,
    app_id: str,
    app_secret: str,
    request_id: str,
    body_json: str = "",
) -> dict[str, str]:
    """Build the signed Cloud request headers.

    Sorts the six signing params by key, joins as
    ``rfc3986(k)=rfc3986(v)`` with ``&``, then MD5s the result.
    ``app_secret`` carries the upstream API key.
    """
    timestamp = str(int(time.time()))
    nonce = _generate_nonce()
    body_md5 = _md5_hex(body_json or _EMPTY_BODY_JSON)
    params = {
        "x-appid": app_id,
        "x-api-key": app_secret,
        "x-request-id": request_id,
        "x-timestamp": timestamp,
        "x-nonce": nonce,
        "body": body_md5,
    }
    joined = "&".join(
        f"{_rfc3986_encode(key)}={_rfc3986_encode(params[key])}" for key in sorted(params)
    )
    return {
        "X-APPID": app_id,
        "X-API-Key": app_secret,
        "X-Request-ID": request_id,
        "X-Timestamp": timestamp,
        "X-Nonce": nonce,
        "X-Signature": _md5_hex(joined),
    }


def is_kb_cloud_doc_reader_addr(addr: str) -> bool:
    """True when ``addr`` is the Cloud docreader endpoint.

    Trailing slashes are ignored on both sides of the comparison.
    """
    normalized = addr.strip().rstrip("/")
    expected = KB_CLOUD_BASE_URL.rstrip("/") + "/api/v1/doc/reader"
    return normalized == expected


class CloudService:
    """Cloud credential persistence + status, constructed per request."""

    def __init__(
        self,
        *,
        tenants_repo: TenantRepository,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Bind the request-scoped tenant repository.

        ``http_client`` is optional: when omitted, ``save_credentials``
        opens a short-lived client for the single verification call.
        Tests inject a client with a mock transport.
        """
        self._tenants_repo = tenants_repo
        self._http_client = http_client

    # ── Save ────────────────────────────────────────────────────────

    async def save_credentials(
        self,
        *,
        tenant_id: int,
        app_id: str,
        app_secret: str,
    ) -> None:
        """Verify then persist the workspace's Cloud credentials.

        Raises ``ValidationError`` when either field is blank,
        ``ExternalServiceError`` when verification fails, and
        ``NotFoundError`` when the workspace does not exist.
        """
        if not app_id:
            raise ValidationError(
                code="cloud.app_id_required",
                message="app_id is required",
            )
        if not app_secret:
            raise ValidationError(
                code="cloud.app_secret_required",
                message="app_secret is required",
            )

        await self._verify_credentials(app_id=app_id, app_secret=app_secret)
        await self._update_tenant_credentials(
            tenant_id=tenant_id,
            app_id=app_id,
            app_secret=app_secret,
        )

    async def _verify_credentials(self, *, app_id: str, app_secret: str) -> None:
        """GET ``/api/v1/health`` with signed headers.

        401/403 means the pair is rejected; any other non-200 is an
        unexpected upstream status; a transport error means the service
        is unreachable. All three raise ``ExternalServiceError`` with a
        ``credential verification failed: ...`` message.
        """
        health_url = KB_CLOUD_BASE_URL.rstrip("/") + _HEALTH_PATH
        request_id = f"verify-{time.time_ns()}"
        headers = sign_request_headers(
            app_id=app_id,
            app_secret=app_secret,
            request_id=request_id,
            body_json=_EMPTY_BODY_JSON,
        )
        try:
            response = await self._get(health_url, headers=headers)
        except httpx.HTTPError as exc:
            raise ExternalServiceError(
                code="cloud.service_unreachable",
                message=f"credential verification failed: service unreachable: {exc}",
            ) from exc

        if response.status_code in (401, 403):
            raise ExternalServiceError(
                code="cloud.invalid_credentials",
                message=(
                    "credential verification failed: invalid APPID or APPSECRET "
                    f"(HTTP {response.status_code})"
                ),
            )
        if response.status_code != 200:
            raise ExternalServiceError(
                code="cloud.verification_failed",
                message=(
                    "credential verification failed: invalid response status code: "
                    f"{response.status_code}"
                ),
            )

    async def _get(self, url: str, *, headers: dict[str, str]) -> httpx.Response:
        """Issue the verification GET on the injected or an ad-hoc client."""
        if self._http_client is not None:
            return await self._http_client.get(url, headers=headers)
        async with httpx.AsyncClient(timeout=_VERIFY_TIMEOUT_SECONDS) as client:
            return await client.get(url, headers=headers)

    async def _update_tenant_credentials(
        self,
        *,
        tenant_id: int,
        app_id: str,
        app_secret: str,
    ) -> None:
        """Merge the provider object into the workspace ``credentials`` column.

        Other providers already present in the JSONB object are
        preserved — only the Cloud provider slot is replaced.
        """
        tenant = await self._tenants_repo.find_by_id(tenant_id)
        credentials: JsonObject = dict(tenant.credentials or {})
        credentials[_CREDENTIALS_KEY] = {"app_id": app_id, "app_secret": app_secret}
        updated = await self._tenants_repo.update_by_primary_key(
            {"id": tenant_id},
            {"credentials": credentials},
        )
        if updated is None:
            raise NotFoundError(
                code="tenant.not_found",
                message=f"Tenant {tenant_id} not found",
            )

    # ── Status ──────────────────────────────────────────────────────

    async def check_status(self, *, tenant_id: int) -> CloudStatusResponse:
        """Report whether the workspace's credentials are usable.

        A missing workspace and a missing credential block both
        resolve to ``{has_models: false, needs_reinit: false}`` so the
        status endpoint never 404s a fresh workspace.
        """
        try:
            tenant = await self._tenants_repo.find_by_id(tenant_id)
        except NotFoundError:
            return CloudStatusResponse(has_models=False, needs_reinit=False)

        credentials = _read_credentials(tenant.credentials)
        if credentials is None:
            return CloudStatusResponse(has_models=False, needs_reinit=False)

        # A stored secret still carrying the enc:v1: prefix means the row
        # loaded but decryption did not happen (rotated / missing key).
        if credentials[1].startswith(ENC_PREFIX):
            return CloudStatusResponse(
                has_models=True,
                needs_reinit=True,
                reason=_REINIT_REASON,
            )
        return CloudStatusResponse(has_models=True, needs_reinit=False)


def _read_credentials(raw: JsonObject | None) -> tuple[str, str] | None:
    """Extract ``(app_id, app_secret)`` from the ``credentials`` column.

    An absent block, or one with either field empty, counts as
    "not configured" (``None``).
    """
    if not raw:
        return None
    block = raw.get(_CREDENTIALS_KEY)
    if not isinstance(block, dict):
        return None
    app_id = block.get("app_id")
    app_secret = block.get("app_secret")
    if not isinstance(app_id, str) or not isinstance(app_secret, str):
        return None
    if not app_id or not app_secret:
        return None
    return app_id, app_secret


__all__ = [
    "ENC_PREFIX",
    "KB_CLOUD_BASE_URL",
    "CloudService",
    "is_kb_cloud_doc_reader_addr",
    "sign_request_headers",
]
