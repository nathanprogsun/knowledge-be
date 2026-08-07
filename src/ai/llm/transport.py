"""LLM call timeouts and an SSRF-protected HTTP client.

The timeout helpers mirror the reference transport: they are a fallback used
only so a hung provider request cannot block a worker forever. The chat /
stream callers wrap their round-trip in ``asyncio.timeout``; a caller that has
already imposed its own deadline naturally wins because asyncio timeouts are
scoped.

The SSRF layer validates both the base URL / endpoint at parse time
(:func:`validate_url_for_ssrf`) and the resolved host IPs at the connection
phase (:class:`_SSRFSafeAsyncTransport`) to defend against DNS-rebinding.
Restricted hostnames, private / link-local / multicast addresses, IP-like
hostnames and sensitive service ports are rejected; ``SSRF_WHITELIST``
exempts trusted hosts, matching the reference semantics.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import re
import socket
import urllib.parse
from contextlib import AbstractAsyncContextManager

import httpx

from src.common.exception import AIProviderError

# ── Configuration ─────────────────────────────────────────────────────

LLM_CHAT_TIMEOUT_ENV = "WEKNORA_LLM_CHAT_TIMEOUT_SECONDS"
LLM_STREAM_TIMEOUT_ENV = "WEKNORA_LLM_STREAM_TIMEOUT_SECONDS"

# Fallback deadlines (seconds), honored only when the caller set none.
DEFAULT_CHAT_TIMEOUT_SECONDS = 300.0
DEFAULT_STREAM_TIMEOUT_SECONDS = 600.0

SSRF_WHITELIST_ENV = "SSRF_WHITELIST"
_MAX_URL_LENGTH = 2048

_RESTRICTED_HOSTNAMES = frozenset(
    {
        "localhost",
        "127.0.0.1",
        "::1",
        "0.0.0.0",
        "metadata.google.internal",
        "metadata.tencentyun.com",
        "metadata.aws.internal",
        "host.docker.internal",
        "gateway.docker.internal",
        "kubernetes.docker.internal",
        "kubernetes",
        "kubernetes.default",
        "kubernetes.default.svc",
        "kubernetes.default.svc.cluster.local",
    }
)

_RESTRICTED_HOST_SUFFIXES = (
    ".local",
    ".localhost",
    ".internal",
    ".corp",
    ".lan",
    ".home",
    ".localdomain",
    ".svc.cluster.local",
    ".pod.cluster.local",
)

_RESTRICTED_IPV4_NETWORKS = tuple(
    ipaddress.ip_network(cidr)
    for cidr in (
        "100.64.0.0/10",
        "198.18.0.0/15",
        "198.51.100.0/24",
        "203.0.113.0/24",
        "192.0.0.0/24",
        "192.0.2.0/24",
        "0.0.0.0/8",
        "240.0.0.0/4",
        "255.255.255.255/32",
        "172.17.0.0/16",
        "172.18.0.0/16",
        "172.19.0.0/16",
        "172.20.0.0/16",
    )
)

_BLOCKED_PORTS = frozenset(
    {"22", "23", "25", "445", "3389", "5432", "3306", "6379", "27017", "9200", "2379", "2380", "8500", "4001"}
)

# Headers users may never override: they are owned by provider auth/signing
# or by the SSE flow, and overwriting them would break the call.
_RESERVED_HEADER_KEYS = frozenset(
    {
        "authorization",
        "api-key",
        "x-api-key",
        "x-goog-api-key",
        "content-type",
        "content-length",
        "accept-encoding",
        "host",
        "connection",
        "transfer-encoding",
    }
)

_IP_LIKE_PATTERNS = (
    re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"),
    re.compile(r"^\d{8,10}$"),
    re.compile(r"^0[0-7]+\."),
    re.compile(r"(?i)^0x[0-9a-f]+\."),
    re.compile(r"(?i)^0x[0-9a-f]{6,8}$"),
    re.compile(r"(?i)^[0-9a-f:]+::[0-9a-f:]*$"),
    re.compile(r"(?i)^[0-9a-f]{1,4}(:[0-9a-f]{1,4}){7}$"),
    re.compile(r"(?i)^::ffff:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"),
    re.compile(r"(?i)^\[[0-9a-f:]+\]$"),
)


class SSRFValidationError(AIProviderError):
    """Raised when a URL / host fails an SSRF safety check."""

    code = "ssrf_validation_error"


# ── Timeout helpers ───────────────────────────────────────────────────


def env_duration_seconds(key: str, fallback: float) -> float:
    """Read a seconds-valued environment variable, falling back on parse errors."""
    value = os.environ.get(key, "").strip()
    if not value:
        return fallback
    try:
        seconds = int(value)
    except ValueError:
        return fallback
    if seconds <= 0:
        return fallback
    return float(seconds)


def with_llm_timeout(duration: float) -> AbstractAsyncContextManager[asyncio.Timeout]:
    """Return an async context manager enforcing a fallback deadline.

    The reference implementation only attaches the default when the caller set
    no deadline; asyncio's scoped timeout achieves the same effect because a
    shorter enclosing deadline fires first and cancels this scope.
    """
    return asyncio.timeout(duration)


# ── SSRF validation ───────────────────────────────────────────────────


def is_ssrf_whitelisted(hostname: str) -> bool:
    """Check ``SSRF_WHITELIST`` (exact / wildcard / CIDR) membership."""
    raw = os.environ.get(SSRF_WHITELIST_ENV, "")
    entries = [entry.strip() for entry in raw.split(",") if entry.strip()]
    if not entries:
        return False
    lower = hostname.lower()
    for entry in entries:
        entry_lower = entry.lower()
        if entry_lower == lower:
            return True
        if entry_lower.startswith("*."):
            suffix = entry_lower[1:]  # ".example.com"
            if lower.endswith(suffix) or lower == suffix[1:]:
                return True
        if "/" in entry_lower:
            try:
                network = ipaddress.ip_network(entry_lower, strict=False)
                ip = ipaddress.ip_address(hostname)
            except ValueError:
                continue
            if ip in network:
                return True
    return False


def is_restricted_ip(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> tuple[bool, str]:
    """Return ``(restricted, reason)`` for a resolved address."""
    if ip.is_private:
        return True, "private IP address"
    if ip.is_loopback:
        return True, "loopback address"
    if ip.is_link_local:
        return True, "link-local address"
    if ip.is_multicast:
        return True, "multicast address"
    if ip.is_unspecified:
        return True, "unspecified address"
    if isinstance(ip, ipaddress.IPv4Address):
        for network in _RESTRICTED_IPV4_NETWORKS:
            if ip in network:
                return True, f"restricted range {network}"
    return False, ""


def _is_ip_like_hostname(hostname: str) -> bool:
    """True for octal / hex / decimal encodings that evade ``ip_address``."""
    return any(pattern.match(hostname) for pattern in _IP_LIKE_PATTERNS)


def _resolve_and_check_host(hostname: str) -> None:
    """Resolve ``hostname`` and reject it when any address is restricted."""
    try:
        infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise SSRFValidationError(
            f"DNS resolution failed for hostname {hostname}: cannot verify if it resolves to safe IP"
        ) from exc
    for _family, _type, _proto, _canonname, sockaddr in infos:
        raw_ip = sockaddr[0]
        try:
            ip = ipaddress.ip_address(raw_ip)
        except ValueError:
            continue
        restricted, reason = is_restricted_ip(ip)
        if restricted:
            raise SSRFValidationError(
                f"hostname {hostname} resolves to restricted IP {raw_ip}: {reason}"
            )


def validate_url_for_ssrf(raw_url: str) -> None:
    """Validate a caller-supplied URL against SSRF protections.

    ``raw_url`` may be a full URL or a bare host; a missing scheme is treated
    as ``https://``. Returns ``None`` on success, raises
    :class:`SSRFValidationError` otherwise.
    """
    if not raw_url:
        return
    if len(raw_url) > _MAX_URL_LENGTH:
        raise SSRFValidationError("URL exceeds maximum length")
    normalized = raw_url if "://" in raw_url else f"https://{raw_url}"
    try:
        parsed = urllib.parse.urlsplit(normalized)
        port = parsed.port
    except ValueError as exc:
        raise SSRFValidationError(f"invalid URL: {exc}") from exc

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise SSRFValidationError(f"invalid scheme: {scheme} (only http/https allowed)")

    hostname = parsed.hostname or ""
    if not hostname:
        raise SSRFValidationError("URL has no hostname")

    # Whitelisted hosts skip the heavy checks, matching the reference flow.
    if is_ssrf_whitelisted(hostname):
        return

    hostname_lower = hostname.lower()
    if hostname_lower in _RESTRICTED_HOSTNAMES:
        raise SSRFValidationError(f"hostname {hostname} is restricted")
    for suffix in _RESTRICTED_HOST_SUFFIXES:
        if hostname_lower.endswith(suffix):
            raise SSRFValidationError(f"hostname suffix {suffix} is restricted")

    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        raise SSRFValidationError(
            "direct IP address access is not allowed, use domain name or add to SSRF_WHITELIST"
        )

    if _is_ip_like_hostname(hostname):
        raise SSRFValidationError("IP-like hostname format is not allowed")

    _resolve_and_check_host(hostname)

    if port is not None and str(port) in _BLOCKED_PORTS:
        raise SSRFValidationError(f"port {port} is blocked for security reasons")


def _validate_resolved_host_ips(host: str) -> None:
    """Connection-phase DNS-rebinding defense (see SSRFSafeDialContext)."""
    if not host:
        return
    hostname_lower = host.lower()
    if hostname_lower in _RESTRICTED_HOSTNAMES:
        raise SSRFValidationError(f"connection blocked: hostname {host} is restricted")
    for suffix in _RESTRICTED_HOST_SUFFIXES:
        if hostname_lower.endswith(suffix):
            raise SSRFValidationError(
                f"connection blocked: hostname suffix {suffix} is restricted"
            )
    if is_ssrf_whitelisted(host):
        return
    _resolve_and_check_host(host)


# ── SSRF-safe HTTP client ─────────────────────────────────────────────


class _SSRFSafeAsyncTransport(httpx.AsyncBaseTransport):
    """Validates resolved host IPs before the inner transport dials."""

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        _validate_resolved_host_ips(request.url.host)
        return await self._inner.handle_async_request(request)


def build_ssrf_safe_client() -> httpx.AsyncClient:
    """Build a shared LLM HTTP client with SSRF-safe dialing.

    Per-request deadlines are enforced by the caller's ``asyncio.timeout``
    scope rather than by the client, so streaming calls are not prematurely
    terminated.
    """
    return httpx.AsyncClient(
        transport=_SSRFSafeAsyncTransport(httpx.AsyncHTTPTransport()),
        timeout=httpx.Timeout(None),
    )


# ── Custom header plumbing ────────────────────────────────────────────


def is_reserved_header(key: str) -> bool:
    """True when ``key`` may not be overridden by user custom headers."""
    return key.strip().lower() in _RESERVED_HEADER_KEYS


def apply_custom_headers(
    headers: dict[str, str], custom: dict[str, str] | None
) -> dict[str, str]:
    """Return ``headers`` with ``custom`` applied, skipping reserved keys."""
    if not custom:
        return headers
    merged = dict(headers)
    for key, value in custom.items():
        name = key.strip()
        if not name or is_reserved_header(name):
            continue
        merged[name] = value
    return merged


__all__ = [
    "DEFAULT_CHAT_TIMEOUT_SECONDS",
    "DEFAULT_STREAM_TIMEOUT_SECONDS",
    "LLM_CHAT_TIMEOUT_ENV",
    "LLM_STREAM_TIMEOUT_ENV",
    "SSRFValidationError",
    "apply_custom_headers",
    "build_ssrf_safe_client",
    "env_duration_seconds",
    "is_reserved_header",
    "is_restricted_ip",
    "is_ssrf_whitelisted",
    "validate_url_for_ssrf",
    "with_llm_timeout",
]
