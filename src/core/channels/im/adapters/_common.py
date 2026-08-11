"""Shared primitives for the IM platform adapters.

Cross-cutting helpers the per-platform adapter modules build on:
credential access, HTTP client construction, signature primitives,
case-insensitive header lookup, JSON payload field access, and
outbound-endpoint URL validation.

The URL validation mirrors the upstream contract: outbound endpoints
must use an explicit scheme and webhook targets must match the
configured host suffix, so a compromised credential cannot point
traffic at an arbitrary internal host.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import time
import urllib.parse
from typing import NoReturn

import httpx

from src.common.exception import ExternalServiceError, ValidationError
from src.common.json import JsonObject, JsonValue

# Default timeout for platform API calls. Adapters that interact with a
# customer-hosted server (Mattermost, Yunzhijia) accept a credential
# override; fixed public endpoints keep this default.
HTTP_TIMEOUT_SECONDS: float = 15.0

# Truncation bound for error detail text embedded in raised messages.
_ERROR_DETAIL_LIMIT = 200


# ── Credential access ──────────────────────────────────────────────────


def string_credential(credentials: JsonObject, key: str) -> str:
    """Read ``key`` as a string, mirroring the bot-identity coercion.

    Booleans are rejected and numbers are rendered without a decimal
    point, matching the identity derivation in the channel service.
    """
    value = credentials.get(key)
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return f"{value:.0f}"
    return ""


def bool_credential(credentials: JsonObject, key: str) -> bool:
    """Read ``key`` as a boolean (bool, ``"true"``/``"1"``/``"yes"``, or non-zero number)."""
    value = credentials.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    if isinstance(value, (int, float)):
        return value != 0
    return False


def int_credential(credentials: JsonObject, key: str, default: int) -> int:
    """Read ``key`` as a positive integer, falling back to ``default``."""
    value = credentials.get(key)
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value if value > 0 else default
    if isinstance(value, float):
        return int(value) if value > 0 else default
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return default
        return parsed if parsed > 0 else default
    return default


# ── HTTP client construction ──────────────────────────────────────────


def build_http_client(
    *,
    timeout: float = HTTP_TIMEOUT_SECONDS,
    transport: httpx.BaseTransport | None = None,
) -> httpx.Client:
    """Return a sync ``httpx.Client`` with a bounded timeout.

    ``transport`` lets tests inject ``httpx.MockTransport``. Redirects
    stay disabled (the httpx default) so outbound calls never follow a
    redirect off the configured endpoint.
    """
    if transport is not None:
        return httpx.Client(timeout=timeout, transport=transport)
    return httpx.Client(timeout=timeout)


# ── Signature primitives ──────────────────────────────────────────────


def hmac_sha256_base64(secret: str, message: str) -> str:
    """Return ``base64(HMAC-SHA256(secret, message))``."""
    digest = hmac.new(
        secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256
    ).digest()
    return base64.b64encode(digest).decode("ascii")


def hmac_sha256_hex(secret: str, message: str) -> str:
    """Return the lowercase hex HMAC-SHA256 of ``message`` under ``secret``."""
    digest = hmac.new(
        secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256
    ).digest()
    return digest.hex()


def hmac_sha1_base64(secret: str, message: str) -> str:
    """Return ``base64(HMAC-SHA1(secret, message))``."""
    digest = hmac.new(
        secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha1
    ).digest()
    return base64.b64encode(digest).decode("ascii")


def constant_time_equals(left: str, right: str) -> bool:
    """Constant-time string comparison (``hmac.compare_digest``)."""
    return hmac.compare_digest(left, right)


# ── Header and payload access ─────────────────────────────────────────


def header_value(headers: JsonObject, name: str) -> str:
    """Case-insensitive header lookup; coerces values to ``str``."""
    target = name.lower()
    for key, value in headers.items():
        if str(key).lower() == target:
            return str(value)
    return ""


def payload_string(payload: JsonObject, key: str) -> str:
    """Read ``key`` from a parsed JSON object as a string."""
    value = payload.get(key)
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return f"{value:.0f}"
    return ""


def payload_int(payload: JsonObject, key: str) -> int:
    """Read ``key`` from a parsed JSON object as an integer."""
    value = payload.get(key)
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def payload_dict(payload: JsonObject, key: str) -> JsonObject:
    """Read ``key`` as a nested JSON object, else ``{}``."""
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def payload_list(payload: JsonObject, key: str) -> list[JsonValue]:
    """Read ``key`` as a JSON list, else ``[]``."""
    value = payload.get(key)
    return value if isinstance(value, list) else []


# ── Timestamp tolerance ───────────────────────────────────────────────


def timestamp_is_fresh(headers: JsonObject, name: str, tolerance_seconds: int) -> bool:
    """Return whether the named header carries a timestamp near ``now``.

    The header value is interpreted as epoch seconds (Slack contract).
    """
    raw = header_value(headers, name)
    if not raw:
        return False
    try:
        ts = int(raw)
    except ValueError:
        return False
    return abs(time.time() - ts) <= tolerance_seconds


def timestamp_ms_is_fresh(headers: JsonObject, name: str, tolerance_seconds: int) -> bool:
    """Return whether the named header carries an epoch-milliseconds timestamp.

    Used by platforms that sign with millisecond timestamps (DingTalk).
    """
    raw = header_value(headers, name)
    if not raw:
        return False
    try:
        ts = int(raw)
    except ValueError:
        return False
    return abs(time.time() * 1000 - ts) <= tolerance_seconds * 1000


# ── Outbound-endpoint URL validation ──────────────────────────────────


def validate_http_endpoint(raw_url: str) -> urllib.parse.ParseResult:
    """Validate a user-supplied http(s) endpoint.

    Rejects non-http(s) schemes, missing hosts, embedded credentials,
    and localhost / literal-IP hosts. Does not require a host suffix,
    so private Mattermost / QQ deployments remain supported.
    """
    parsed = urllib.parse.urlparse(raw_url.strip())
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValidationError(
            code="im.invalid_endpoint",
            message=f"endpoint must use http(s), got: {scheme or '(none)'}",
        )
    if parsed.username is not None or parsed.password is not None:
        raise ValidationError(
            code="im.invalid_endpoint",
            message="endpoint must not embed credentials",
        )
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise ValidationError(code="im.invalid_endpoint", message="endpoint has no host")
    if host == "localhost" or host.endswith(".localhost") or _is_ip_literal(host):
        raise ValidationError(
            code="im.invalid_endpoint",
            message="endpoint host must be a DNS name",
        )
    return parsed


def validate_https_host_suffix(
    raw_url: str, allowed_host_suffix: str
) -> urllib.parse.ParseResult:
    """Validate an HTTPS webhook endpoint against a required host suffix."""
    parsed = validate_http_endpoint(raw_url)
    if parsed.scheme.lower() != "https":
        raise ValidationError(code="im.invalid_endpoint", message="endpoint must use https")
    suffix = allowed_host_suffix.strip().strip(".").lower()
    if not suffix:
        raise ValidationError(
            code="im.invalid_endpoint",
            message="allowed host suffix is required",
        )
    host = (parsed.hostname or "").lower().rstrip(".")
    if host != suffix and not host.endswith("." + suffix):
        raise ValidationError(
            code="im.invalid_endpoint",
            message=f"endpoint host {host!r} does not match allowed suffix {suffix!r}",
        )
    return parsed


def _is_ip_literal(host: str) -> bool:
    """Return whether ``host`` is a literal IP address (v4 or v6)."""
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


# ── Error helpers ─────────────────────────────────────────────────────


def send_error(platform: str, action: str, detail: str) -> NoReturn:
    """Raise ``ExternalServiceError`` for an outbound platform call failure."""
    raise ExternalServiceError(
        code="im.send_failed",
        message=f"{platform} {action} failed: {detail[:_ERROR_DETAIL_LIMIT]}",
    )


def assert_http_ok(resp: httpx.Response, *, platform: str, action: str) -> None:
    """Raise when ``resp`` carries a non-2xx status."""
    if 200 <= resp.status_code < 300:
        return
    send_error(platform, action, f"HTTP {resp.status_code} {resp.text[:200]}")


def assert_http_ok_strict(resp: httpx.Response, *, platform: str, action: str) -> None:
    """Raise unless ``resp`` carries exactly ``200`` (mirrors the wire contract)."""
    if resp.status_code == 200:
        return
    send_error(platform, action, f"HTTP {resp.status_code} {resp.text[:200]}")


__all__ = [
    "HTTP_TIMEOUT_SECONDS",
    "assert_http_ok",
    "assert_http_ok_strict",
    "bool_credential",
    "build_http_client",
    "constant_time_equals",
    "header_value",
    "hmac_sha1_base64",
    "hmac_sha256_base64",
    "hmac_sha256_hex",
    "int_credential",
    "payload_dict",
    "payload_int",
    "payload_list",
    "payload_string",
    "send_error",
    "string_credential",
    "timestamp_is_fresh",
    "timestamp_ms_is_fresh",
    "validate_http_endpoint",
    "validate_https_host_suffix",
]
