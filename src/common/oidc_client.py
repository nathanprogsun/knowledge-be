"""OIDC HTTP client - discovery, code exchange, userinfo fetch.

Validates provider URLs at call time to avoid SSRF; decodes id_token
claims without signature verification. Returns own result dataclasses.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import ipaddress
import json
import os
import socket
import urllib.parse
from dataclasses import dataclass
from typing import ClassVar, cast

import httpx

from src.common.exception import ApplicationError, ExternalServiceError, ValidationError
from src.common.json import JsonValue
from src.common.oidc_types import OIDCClaimsDict

# ── Result dataclasses ──────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class OIDCDiscoveryDocument:
    """Subset of the OIDC discovery document this client consumes."""

    authorization_endpoint: str
    token_endpoint: str
    user_info_endpoint: str


@dataclass(frozen=True, slots=True)
class OIDCTokenResponse:
    """Token endpoint response. ``access_token`` and ``id_token`` may both
    be present; at least one is required (enforced by the caller)."""

    access_token: str
    id_token: str
    token_type: str


@dataclass(frozen=True, slots=True)
class OIDCUserInfoClaims:
    """Resolved provider userinfo. ``claims`` is the raw merged claim dict
    (id_token + userinfo endpoint); ``subject`` / ``username`` / ``email``
    are the projected fields the service consumes."""

    subject: str
    username: str
    email: str
    claims: OIDCClaimsDict


# ── SSRF guard ───────────────────────────────────────────────────────

_RESTRICTED_HOSTNAMES: frozenset[str] = frozenset(
    {
        "localhost",
        "metadata",
        "metadata.google.internal",
        "ip6-localhost",
        "ip6-loopback",
    }
)

# HTTP headers stripped on cross-host redirects (the standard
# `stripRedirectSensitiveHeaders` set in the OIDC security helper).
# Connector tokens must not leak to a third party because the IdP (or
# any intermediary) can return a redirect to attacker-controlled host.
_REDIRECT_STRIPPED_HEADERS: frozenset[str] = frozenset(
    {
        "authorization",
        "cookie",
        "x-auth-token",
        "x-api-key",
        "api-key",
    }
)

# Toggles `httpx`'s own redirect-following off. Re-validated manually
# on each hop, exactly like Go's `newSSRFCheckRedirect`.
_MAX_REDIRECTS = 10

_RESTRICTED_SUFFIXES: tuple[str, ...] = (
    ".internal",
    ".local",
    ".localhost",
    ".home",
    ".lan",
    ".intra",
)

_MAX_URL_LENGTH = 2048


def _load_ssrf_whitelist() -> tuple[
    set[str], list[str], list[ipaddress.IPv4Network | ipaddress.IPv6Network]
]:
    """Parse ``SSRF_WHITELIST`` into (exact_hosts, suffixes, cidr_nets).

    Entries are comma-separated. ``*.example.com`` -> suffix; ``10.0.0.0/8``
    -> CIDR; bare host/IP -> exact. Whitespace is trimmed; empty entries
    are dropped. Re-read every call so tests can mutate the env.
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
            suffixes.append(entry[1:])  # keep leading '.' for endswith
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
    # IP literal whitelist / CIDR match.
    try:
        ip = ipaddress.ip_address(hostname_lower)
    except ValueError:
        ip = None
    if ip is not None:
        for net in cidrs:
            if ip in net:
                return True
    return False


def _is_restricted_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _is_ip_like_hostname(hostname: str) -> bool:
    """Detect decimal / hex / octal IP obfuscation a browser would resolve.

    Covers the common forms: bare decimal (``2130706433``), dotted hex
    (``0x7f.0.0.1``), dotted octal (``0177.0.0.1``). Not exhaustive - the
    main SSRF vector (direct dotted-decimal) is caught by
    ``ipaddress.ip_address``.
    """
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


async def validate_ssrf_safe_url(raw_url: str) -> None:
    """Raise ``ValidationError`` if ``raw_url`` is not SSRF-safe.

    Mirrors the upstream ``isSSRFSafeURL`` contract: empty URL is
    rejected, DNS resolution is fail-closed (unknown host cannot be
    proven safe), and every redirect target is re-validated
    separately by the HTTP client.
    """
    if not raw_url:
        raise ValidationError("URL is empty", code="oidc.ssrf_blocked")
    if len(raw_url) > _MAX_URL_LENGTH:
        raise ValidationError("URL exceeds maximum length", code="oidc.ssrf_blocked")
    parsed = urllib.parse.urlparse(raw_url)
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValidationError(
            f"invalid scheme: {scheme or '(none)'} (only http/https allowed)",
            code="oidc.ssrf_blocked",
        )
    hostname = parsed.hostname or ""
    if not hostname:
        raise ValidationError("URL has no hostname", code="oidc.ssrf_blocked")
    hostname_lower = hostname.lower()
    if _is_ssrf_whitelisted(hostname_lower):
        return
    if hostname_lower in _RESTRICTED_HOSTNAMES:
        raise ValidationError(f"hostname {hostname} is restricted", code="oidc.ssrf_blocked")
    for suffix in _RESTRICTED_SUFFIXES:
        if hostname_lower.endswith(suffix):
            raise ValidationError(
                f"hostname suffix {suffix} is restricted", code="oidc.ssrf_blocked"
            )
    # Direct IP literal.
    try:
        ipaddress.ip_address(hostname)
        raise ValidationError(
            "direct IP address access is not allowed; use a domain or add to SSRF_WHITELIST",
            code="oidc.ssrf_blocked",
        )
    except ValueError:
        pass
    if _is_ip_like_hostname(hostname_lower):
        raise ValidationError("IP-like hostname format is not allowed", code="oidc.ssrf_blocked")
    # DNS resolution - must not land on a restricted IP. Fail-closed:
    # if we cannot resolve the hostname, we cannot prove it is safe.
    infos = await asyncio.to_thread(_resolve_host, hostname)
    for resolved in infos:
        if _is_restricted_ip(resolved):
            raise ValidationError(
                f"hostname {hostname} resolves to restricted IP {resolved}",
                code="oidc.ssrf_blocked",
            )


def _resolve_host(hostname: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Resolve ``hostname`` to IP address objects.

    Raises ``ValidationError`` on resolution failure - mirrors Go's
    ``net.LookupIP`` failure path (fail-closed).
    """
    try:
        entries = socket.getaddrinfo(hostname, None)
    except OSError as exc:
        raise ValidationError(
            f"DNS resolution failed for hostname {hostname}: cannot verify it is safe",
            code="oidc.ssrf_blocked",
        ) from exc
    ips: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    seen: set[str] = set()
    for entry in entries:
        addr_raw = entry[4][0]
        if not isinstance(addr_raw, str):
            continue
        if addr_raw in seen:
            continue
        seen.add(addr_raw)
        try:
            ips.append(ipaddress.ip_address(addr_raw))
        except ValueError:
            continue
    return ips


# ── JWT claim decoding ──────────────────────────────────────────────


def decode_jwt_claims_unverified(id_token: str) -> OIDCClaimsDict:
    """Decode the payload claims of ``id_token`` without signature check.

    Split on ``.``, base64url-decode the payload segment, JSON-parse.
    Returns an empty dict on any malformed input (the caller treats
    missing claims as a soft failure).
    """
    parts = id_token.split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1]
    # base64.urlsafe_b64decode requires padding to length 4.
    padding = "=" * (-len(payload) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload + padding)
        decoded = json.loads(raw)
    except (ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(decoded, dict):
        return {}
    return decoded


# ── HTTP client ─────────────────────────────────────────────────────


def _default_client(
    transport: httpx.AsyncBaseTransport | None,
    timeout: float,
) -> httpx.AsyncClient:
    """Construct the underlying httpx client (no redirect following).

    Redirects are intentionally disabled at the client level: every hop
    is re-validated manually by :meth:`OidcClient._send`.
    """
    if transport is not None:
        return httpx.AsyncClient(transport=transport, timeout=timeout, follow_redirects=False)
    return httpx.AsyncClient(timeout=timeout, follow_redirects=False)


def _same_http_origin(a: httpx.URL, b: httpx.URL) -> bool:
    return a.scheme.lower() == b.scheme.lower() and a.host.lower() == b.host.lower()


def _strip_redirect_sensitive_headers(request: httpx.Request) -> None:
    for header in _REDIRECT_STRIPPED_HEADERS:
        if header in request.headers:
            del request.headers[header]


class OidcClient:
    """Async OIDC HTTP client (discovery + code exchange + userinfo).

    Stateless apart from the underlying ``httpx.AsyncClient``; safe to
    share as a singleton once constructed, or instantiate per request.
    Tests inject an ``httpx.MockTransport`` via ``transport=``.
    """

    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._client = _default_client(transport, timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _send(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        data: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Send with manual redirect handling + SSRF re-validation per hop.

        Aligned with Go's ``newSSRFCheckRedirect``: cap redirects at
        ``_MAX_REDIRECTS``, strip connector credentials on cross-host
        hops, and re-run ``validate_ssrf_safe_url`` on every Location
        before sending.
        """
        original = httpx.Request(method, url, headers=headers, data=data)
        for hop in range(_MAX_REDIRECTS + 1):
            await validate_ssrf_safe_url(str(original.url))
            try:
                response = await self._client.send(original)
            except httpx.HTTPError as exc:
                raise ExternalServiceError(
                    f"OIDC request failed: {exc}",
                    code="oidc.exchange_failed",
                ) from exc
            if not response.is_redirect:
                return response
            if hop >= _MAX_REDIRECTS:
                raise ExternalServiceError(
                    f"stopped after {_MAX_REDIRECTS} redirects",
                    code="oidc.redirect_blocked",
                )
            location = response.headers.get("Location")
            if not location:
                raise ExternalServiceError(
                    "redirect response missing Location header",
                    code="oidc.redirect_blocked",
                )
            next_url = urllib.parse.urljoin(str(original.url), location)
            next_request = httpx.Request(method, next_url, headers=original.headers)
            if not _same_http_origin(original.url, next_request.url):
                _strip_redirect_sensitive_headers(next_request)
            original = next_request
        raise ExternalServiceError(
            f"stopped after {_MAX_REDIRECTS} redirects",
            code="oidc.redirect_blocked",
        )

    # ── Discovery ────────────────────────────────────────────────────

    async def discover_endpoints(self, discovery_url: str) -> OIDCDiscoveryDocument:
        """Fetch ``.well-known/openid-configuration`` and return endpoints."""
        response = await self._send("GET", discovery_url, headers={"Accept": "application/json"})
        body = self._read_json(response, label="discovery")
        authorization = _as_str(body.get("authorization_endpoint"))
        token = _as_str(body.get("token_endpoint"))
        userinfo = _as_str(body.get("userinfo_endpoint"))
        if not authorization or not token:
            raise ExternalServiceError(
                "OIDC discovery document missing required endpoints",
                code="oidc.discovery_failed",
            )
        return OIDCDiscoveryDocument(
            authorization_endpoint=authorization,
            token_endpoint=token,
            user_info_endpoint=userinfo,
        )

    # ── Code exchange ───────────────────────────────────────────────

    async def exchange_code(
        self,
        *,
        token_endpoint: str,
        client_id: str,
        client_secret: str,
        code: str,
        redirect_uri: str,
    ) -> OIDCTokenResponse:
        """POST the authorization-code grant and return the token response."""
        form: dict[str, str] = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "client_secret": client_secret,
        }
        response = await self._send(
            "POST",
            token_endpoint,
            data=form,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
        )
        body = self._read_json(response, label="token exchange")
        access_token = _as_str(body.get("access_token"))
        id_token = _as_str(body.get("id_token"))
        if not access_token and not id_token:
            raise ExternalServiceError(
                "OIDC token response missing access_token and id_token",
                code="oidc.exchange_failed",
            )
        return OIDCTokenResponse(
            access_token=access_token,
            id_token=id_token,
            token_type=_as_str(body.get("token_type")),
        )

    # ── Userinfo ────────────────────────────────────────────────────

    async def fetch_userinfo(
        self,
        *,
        userinfo_endpoint: str,
        access_token: str,
    ) -> OIDCClaimsDict:
        """GET the userinfo endpoint with a bearer token, return claims."""
        response = await self._send(
            "GET",
            userinfo_endpoint,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            },
        )
        return self._read_json(response, label="userinfo")

    # ── Userinfo projection ──────────────────────────────────────────

    async def resolve_userinfo(
        self,
        *,
        user_info_endpoint: str,
        access_token: str,
        id_token: str,
        username_claim: str,
        email_claim: str,
    ) -> OIDCUserInfoClaims:
        """Merge id_token + userinfo claims, project ``subject``/``username``/``email``.

        Userinfo fetch is best-effort; on failure id_token claims suffice.
        Raises ``ExternalServiceError`` if no claims are available.
        """
        claims: dict[str, JsonValue] = {}
        if id_token:
            claims.update(cast(dict[str, JsonValue], decode_jwt_claims_unverified(id_token)))
        if user_info_endpoint and access_token:
            with contextlib.suppress(ApplicationError):
                claims.update(
                    cast(
                        dict[str, JsonValue],
                        await self.fetch_userinfo(
                            userinfo_endpoint=user_info_endpoint,
                            access_token=access_token,
                        ),
                    )
                )
        if not claims:
            raise ExternalServiceError(
                "OIDC provider returned no usable claims",
                code="oidc.userinfo_failed",
            )
        subject = _as_str(claims.get("sub"))
        username = (
            _as_str(claims.get(username_claim))
            or _as_str(claims.get("preferred_username"))
            or _as_str(claims.get("name"))
        )
        email = _as_str(claims.get(email_claim))
        if not username and email:
            username = email.split("@", 1)[0]
        return OIDCUserInfoClaims(
            subject=subject,
            username=username,
            email=email,
            claims=claims,
        )

    # ── Shared response handling ────────────────────────────────────

    _ERROR_CODE_BY_LABEL: ClassVar[dict[str, str]] = {
        "discovery": "oidc.discovery_failed",
        "token exchange": "oidc.exchange_failed",
        "userinfo": "oidc.userinfo_failed",
    }

    def _read_json(self, response: httpx.Response, *, label: str) -> dict[str, JsonValue]:
        """Read + JSON-decode a 2xx response; raise ``ExternalServiceError`` otherwise."""
        code = self._ERROR_CODE_BY_LABEL.get(label, "oidc.exchange_failed")
        if response.status_code < 200 or response.status_code >= 300:
            snippet = response.text[:2048].strip()
            raise ExternalServiceError(
                f"OIDC {label} request failed: status={response.status_code} body={snippet}",
                code=code,
            )
        try:
            decoded = response.json()
        except json.JSONDecodeError as exc:
            raise ExternalServiceError(
                f"failed to decode OIDC {label} JSON: {exc}",
                code=code,
            ) from exc
        if not isinstance(decoded, dict):
            raise ExternalServiceError(
                f"OIDC {label} response is not a JSON object",
                code=code,
            )
        # response.json() yields Any; the dict-isinstance check above plus
        # the JSON grammar (object keys are strings, values are JSON
        # values) justify the narrowing cast.
        return cast(dict[str, JsonValue], decoded)


def _as_str(value: JsonValue) -> str:
    """Coerce a claim value to a trimmed string.

    Accepts ``str`` or ``None``; rejects every other type with
    ``TypeError`` so claim misconfiguration (e.g. ``email_claim``
    pointing at an array such as ``groups``) fails fast instead of
    silently producing a Python ``repr`` string like ``"['a', 'b']"``.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    raise TypeError(f"expected str or None, got {type(value).__name__}: {value!r}")


__all__ = [
    "OIDCDiscoveryDocument",
    "OIDCTokenResponse",
    "OIDCUserInfoClaims",
    "OidcClient",
    "decode_jwt_claims_unverified",
    "validate_ssrf_safe_url",
]
