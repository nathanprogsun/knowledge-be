"""OIDC HTTP client - discovery, code exchange, userinfo fetch.

Mirrors the network-side of the upstream OIDC flow in
``internal/application/service/user.go`` (``populateOIDCEndpoints`` /
``exchangeOIDCCode`` / ``fetchOIDCUserInfo`` / ``decodeJWTClaims``).

This module lives in the ``ai`` layer: it imports only ``common`` and
``util`` (AGENTS.md §1) and returns its own result dataclasses
(``OIDCDiscoveryDocument`` / ``OIDCTokenResponse`` / ``OIDCUserInfoClaims``).
The ``core`` layer (``OidcService``) maps these onto domain DTOs - the
``ai`` layer never references ``core`` or ``db`` types.

SSRF hardening
--------------

Every endpoint URL is validated by :func:`validate_ssrf_safe_url` before
the request is dispatched - porting the upstream ``ValidateURLForSSRF``
core: scheme must be http/https, no IP literals, no restricted
hostnames/suffixes (``localhost``, ``.internal``, ``.local`` …), and DNS
resolution must not land on private/loopback/link-local/reserved IPs.
The ``SSRF_WHITELIST`` env var (exact host / ``*.suffix`` / IP / CIDR)
bypasses the heavy checks. DNS resolution runs in a thread
(``asyncio.to_thread``) so the event loop is not blocked.

Known gap vs. the upstream: the Go ``NewSSRFSafeHTTPClient`` pins the
resolved IP via a custom ``net.Dialer`` to defeat DNS rebinding across
the validation→request window; httpx has no equivalent transport hook,
so this client validates at call time only. Acceptable for PR-4; the
rebinding hardening is tracked as a follow-up.

id_token signature
------------------

``decode_jwt_claims_unverified`` base64-decodes the payload **without**
verifying the provider signature, matching the upstream
``decodeJWTClaims``. Proper id_token verification (JWKS) is a follow-up;
this keeps behaviour aligned with the Go source.
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
from typing import TypeAlias

import httpx

from src.common.exception import ApplicationError, ExternalServiceError, ValidationError

# ── JSON value type ──────────────────────────────────────────────────
# Recursive alias so claim dicts never need ``Any`` / ``object``
# (AGENTS.md §1). The string forward refs resolve through the module
# namespace; ``from __future__ import annotations`` is not required for
# ``TypeAlias`` runtime evaluation but keeps mypy happy with the cycle.

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


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
    claims: dict[str, JsonValue]


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
    ``ipaddress.ip_address`` upstream.
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

    Mirrors the upstream ``ValidateURLForSSRF`` + ``isSSRFSafeURL`` core.
    DNS resolution runs in a worker thread; everything else is cheap.
    """
    if not raw_url:
        return
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
    # DNS resolution - must not land on a restricted IP.
    infos = await asyncio.to_thread(_resolve_host, hostname)
    for resolved in infos:
        if _is_restricted_ip(resolved):
            raise ValidationError(
                f"hostname {hostname} resolves to restricted IP {resolved}",
                code="oidc.ssrf_blocked",
            )


def _resolve_host(hostname: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Resolve ``hostname`` to IP address objects (empty list on failure)."""
    try:
        entries = socket.getaddrinfo(hostname, None)
    except OSError:
        return []
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


def decode_jwt_claims_unverified(id_token: str) -> dict[str, JsonValue]:
    """Decode the payload claims of ``id_token`` without signature check.

    Mirrors the upstream ``decodeJWTClaims``: split on ``.``, base64url-
    decode the payload segment, JSON-parse. Returns an empty dict on any
    malformed input (the caller treats missing claims as a soft failure).
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
    """Construct the underlying httpx client (no redirect following)."""
    if transport is not None:
        return httpx.AsyncClient(transport=transport, timeout=timeout)
    return httpx.AsyncClient(timeout=timeout)


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

    # ── Discovery ────────────────────────────────────────────────────

    async def discover_endpoints(self, discovery_url: str) -> OIDCDiscoveryDocument:
        """Fetch ``.well-known/openid-configuration`` and return endpoints."""
        await validate_ssrf_safe_url(discovery_url)
        try:
            response = await self._client.get(discovery_url, headers={"Accept": "application/json"})
        except httpx.HTTPError as exc:
            raise ExternalServiceError(
                f"failed to load OIDC discovery document: {exc}",
                code="oidc.discovery_failed",
            ) from exc
        body = await self._read_json(response, label="discovery")
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
        await validate_ssrf_safe_url(token_endpoint)
        form: dict[str, str] = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "client_secret": client_secret,
        }
        try:
            response = await self._client.post(
                token_endpoint,
                data=form,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
            )
        except httpx.HTTPError as exc:
            raise ExternalServiceError(
                f"failed to exchange OIDC code: {exc}",
                code="oidc.exchange_failed",
            ) from exc
        body = await self._read_json(response, label="token exchange")
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
    ) -> dict[str, JsonValue]:
        """GET the userinfo endpoint with a bearer token, return claims."""
        await validate_ssrf_safe_url(userinfo_endpoint)
        try:
            response = await self._client.get(
                userinfo_endpoint,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json",
                },
            )
        except httpx.HTTPError as exc:
            raise ExternalServiceError(
                f"failed to fetch OIDC userinfo: {exc}",
                code="oidc.userinfo_failed",
            ) from exc
        return await self._read_json(response, label="userinfo")

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

        Mirrors the upstream ``resolveOIDCUserInfo``: id_token claims are
        decoded first; the userinfo endpoint (when configured and an access
        token is present) is fetched and merged on top. The userinfo fetch
        is best-effort - on failure the id_token claims alone suffice
        (matching the upstream warn-and-continue). If neither source yields
        any claims, ``ExternalServiceError`` is raised rather than silently
        returning an empty projection.
        """
        claims: dict[str, JsonValue] = {}
        if id_token:
            claims.update(decode_jwt_claims_unverified(id_token))
        if user_info_endpoint and access_token:
            # Best-effort: any ApplicationError from fetch_userinfo (HTTP
            # failure via ExternalServiceError, or SSRF block via
            # ValidationError) falls back to id_token claims - matching the
            # upstream warn-and-continue which catches any fetch error.
            with contextlib.suppress(ApplicationError):
                claims.update(
                    await self.fetch_userinfo(
                        userinfo_endpoint=user_info_endpoint,
                        access_token=access_token,
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

    async def _read_json(self, response: httpx.Response, *, label: str) -> dict[str, JsonValue]:
        """Read + JSON-decode a 2xx response; raise ``ExternalServiceError`` otherwise."""
        if response.status_code < 200 or response.status_code >= 300:
            snippet = response.text[:2048].strip()
            raise ExternalServiceError(
                f"OIDC {label} request failed: status={response.status_code} body={snippet}",
                code="oidc.exchange_failed" if label != "discovery" else "oidc.discovery_failed",
            )
        try:
            decoded = response.json()
        except json.JSONDecodeError as exc:
            raise ExternalServiceError(
                f"failed to decode OIDC {label} JSON: {exc}",
                code="oidc.exchange_failed" if label != "discovery" else "oidc.discovery_failed",
            ) from exc
        if not isinstance(decoded, dict):
            raise ExternalServiceError(
                f"OIDC {label} response is not a JSON object",
                code="oidc.exchange_failed" if label != "discovery" else "oidc.discovery_failed",
            )
        return decoded


def _as_str(value: JsonValue) -> str:
    """Coerce a claim value to a trimmed string (mirrors ``extractClaimAsString``)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


__all__ = [
    "OIDCDiscoveryDocument",
    "OIDCTokenResponse",
    "OIDCUserInfoClaims",
    "OidcClient",
    "decode_jwt_claims_unverified",
    "validate_ssrf_safe_url",
]
