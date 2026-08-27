"""Outbound HTTP client construction and SSRF-safe URL validation.

Every provider builds its ``httpx.Client`` through :func:`build_http_client`,
which applies the same proxy policy as the upstream search client helper:
an explicit ``proxy_url`` is validated for SSRF safety and used when
given; otherwise the environment proxy configuration applies (the
standard "proxy fallback" — an unset proxy_url means "use the
deployment's environment proxy").

:func:`validate_url_for_ssrf` is the synchronous counterpart of the
shared async guard used by the connector layer. It enforces the same
scheme / hostname / whitelist / IP-literal rules; the async guard's DNS
resolution step is intentionally not repeated here because the
providers construct clients synchronously and connect through httpx,
which resolves at request time.
"""

from __future__ import annotations

import ipaddress
import os
import urllib.parse

import httpx

from src.common.exception import ValidationError

_MAX_URL_LENGTH = 2048

_RESTRICTED_HOSTNAMES: frozenset[str] = frozenset(
    {
        "localhost",
        "metadata",
        "metadata.google.internal",
        "ip6-localhost",
        "ip6-loopback",
    }
)

_RESTRICTED_SUFFIXES: tuple[str, ...] = (
    ".internal",
    ".local",
    ".localhost",
    ".home",
    ".lan",
    ".intra",
)

_ERROR_CODE = "web_search_provider.ssrf_blocked"


def _load_ssrf_whitelist() -> tuple[
    set[str], list[str], list[ipaddress.IPv4Network | ipaddress.IPv6Network]
]:
    """Parse ``SSRF_WHITELIST`` into (exact_hosts, suffixes, cidr_nets).

    Entries are comma-separated; ``*.example.com`` is a suffix,
    ``10.0.0.0/8`` a CIDR, and a bare host / IP an exact match. The env
    is re-read on every call so tests can mutate it.
    """
    raw = os.environ.get("SSRF_WHITELIST", "")
    exact: set[str] = set()
    suffixes: list[str] = []
    cidrs: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for token in raw.split(","):
        entry = token.strip().lower()
        if not entry:
            continue
        if entry.startswith("*."):
            suffixes.append(entry[1:])
            continue
        if "/" in entry:
            try:
                cidrs.append(ipaddress.ip_network(entry, strict=False))
            except ValueError:
                continue
            continue
        exact.add(entry)
    return exact, suffixes, cidrs


def _is_ssrf_whitelisted(hostname_lower: str) -> bool:
    exact, suffixes, cidrs = _load_ssrf_whitelist()
    if hostname_lower in exact:
        return True
    for suffix in suffixes:
        if hostname_lower.endswith(suffix) or hostname_lower == suffix[1:]:
            return True
    try:
        ip = ipaddress.ip_address(hostname_lower)
    except ValueError:
        ip = None
    if ip is not None:
        for net in cidrs:
            if ip in net:
                return True
    return False


def _is_ip_like_hostname(hostname: str) -> bool:
    """Detect decimal / hex / octal IP obfuscation forms."""
    if hostname.isdigit():
        try:
            ipaddress.IPv4Address(int(hostname))
            return True
        except ValueError:
            pass
    parts = hostname.split(".")
    if len(parts) != 4:
        return False
    octets: list[int] = []
    for part in parts:
        try:
            if part.lower().startswith("0x"):
                octets.append(int(part, 16))
            elif len(part) > 1 and part.startswith("0") and part.isdigit():
                octets.append(int(part, 8))
            else:
                octets.append(int(part, 10))
        except ValueError:
            return False
    return all(0 <= o <= 255 for o in octets)


def validate_url_for_ssrf(raw_url: str) -> str:
    """Validate ``raw_url`` for SSRF safety; return the trimmed URL.

    Mirrors the upstream ``ValidateURLForSSRF`` semantics: empty URLs
    pass (callers that require a non-empty value validate separately),
    whitelisted hosts skip the heavy checks, and http(s)-only, no
    restricted hostname, no direct / obfuscated IP literal. Raises
    ``ValidationError`` with ``web_search_provider.ssrf_blocked``.
    """
    url = raw_url.strip()
    if not url:
        return url
    if len(url) > _MAX_URL_LENGTH:
        raise ValidationError("URL exceeds maximum length", code=_ERROR_CODE)
    if "://" not in url:
        url = "https://" + url
    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValidationError(
            f"invalid scheme: {scheme or '(none)'} (only http/https allowed)",
            code=_ERROR_CODE,
        )
    hostname = parsed.hostname or ""
    if not hostname:
        raise ValidationError("URL has no hostname", code=_ERROR_CODE)
    hostname_lower = hostname.lower()
    if _is_ssrf_whitelisted(hostname_lower):
        return url
    if hostname_lower in _RESTRICTED_HOSTNAMES:
        raise ValidationError(f"hostname {hostname} is restricted", code=_ERROR_CODE)
    for suffix in _RESTRICTED_SUFFIXES:
        if hostname_lower.endswith(suffix):
            raise ValidationError(f"hostname suffix {suffix} is restricted", code=_ERROR_CODE)
    try:
        ipaddress.ip_address(hostname)
        raise ValidationError(
            "direct IP address access is not allowed; use a domain or add to SSRF_WHITELIST",
            code=_ERROR_CODE,
        )
    except ValueError:
        pass
    if _is_ip_like_hostname(hostname_lower):
        raise ValidationError("IP-like hostname format is not allowed", code=_ERROR_CODE)
    return url


def build_http_client(*, timeout: float, proxy_url: str = "") -> httpx.Client:
    """Build the ``httpx.Client`` an outbound web-search provider uses.

    Proxy policy: an explicit ``proxy_url`` (SSRF-validated) wins; an
    empty ``proxy_url`` falls back to the environment proxy
    configuration. Redirects are not auto-followed — the provider
    endpoints are hardcoded API URLs and a redirect is surfaced as a
    status error rather than silently following a potentially
    untrusted target.
    """
    proxy = proxy_url.strip()
    if proxy:
        validate_url_for_ssrf(proxy)
        return httpx.Client(timeout=timeout, proxy=proxy, trust_env=False)
    return httpx.Client(timeout=timeout)


__all__ = [
    "build_http_client",
    "validate_url_for_ssrf",
]
