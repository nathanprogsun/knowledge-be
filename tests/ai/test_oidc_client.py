"""Unit tests for ``src.ai.oidc_client``.

Happy-path HTTP calls use ``httpx.MockTransport`` so no real network is
hit; the discovery/exchange/userinfo URLs live under ``idp.example.com``
which is added to ``SSRF_WHITELIST`` so the SSRF guard's DNS step is
skipped (deterministic, no network). SSRF rejection cases use IP
literals / restricted hostnames which are blocked before any DNS
lookup.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping

import httpx
import pytest

from src.ai.oidc_client import (
    OidcClient,
    OIDCDiscoveryDocument,
    OIDCTokenResponse,
    decode_jwt_claims_unverified,
    validate_ssrf_safe_url,
)
from src.common.exception import ExternalServiceError, ValidationError

_DISCOVERY_URL = "https://idp.example.com/.well-known/openid-configuration"
_AUTHORIZE = "https://idp.example.com/authorize"
_TOKEN = "https://idp.example.com/token"
_USERINFO = "https://idp.example.com/userinfo"


@pytest.fixture(autouse=True)
def _whitelist_idp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass SSRF DNS checks for the test IdP host."""
    monkeypatch.setenv("SSRF_WHITELIST", "idp.example.com")


def _make_client(handler: object) -> OidcClient:
    return OidcClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def _b64url_json(payload: Mapping[str, object]) -> str:
    """base64url-encode a JSON payload (for fake id_token construction)."""
    raw = json.dumps(payload).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


# ── Discovery ────────────────────────────────────────────────────────


async def test_discover_endpoints_parses_document() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == _DISCOVERY_URL
        return httpx.Response(
            200,
            json={
                "authorization_endpoint": _AUTHORIZE,
                "token_endpoint": _TOKEN,
                "userinfo_endpoint": _USERINFO,
            },
        )

    client = _make_client(handler)
    doc = await client.discover_endpoints(_DISCOVERY_URL)
    assert isinstance(doc, OIDCDiscoveryDocument)
    assert doc.authorization_endpoint == _AUTHORIZE
    assert doc.token_endpoint == _TOKEN
    assert doc.user_info_endpoint == _USERINFO


async def test_discover_endpoints_non_2xx_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = _make_client(handler)
    with pytest.raises(ExternalServiceError, match="discovery"):
        await client.discover_endpoints(_DISCOVERY_URL)


async def test_discover_endpoints_missing_required_fields_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # No authorization_endpoint / token_endpoint.
        return httpx.Response(200, json={"userinfo_endpoint": _USERINFO})

    client = _make_client(handler)
    with pytest.raises(ExternalServiceError, match="missing required endpoints"):
        await client.discover_endpoints(_DISCOVERY_URL)


# ── Code exchange ────────────────────────────────────────────────────


async def test_exchange_code_posts_form_and_parses_token() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == _TOKEN
        # The form is urlencoded in the request body.
        body = request.content.decode("utf-8")
        for pair in body.split("&"):
            key, _, value = pair.partition("=")
            captured[key] = value
        return httpx.Response(
            200,
            json={
                "access_token": "at-123",
                "id_token": "idt-456",
                "token_type": "Bearer",
            },
        )

    client = _make_client(handler)
    token = await client.exchange_code(
        token_endpoint=_TOKEN,
        client_id="cid",
        client_secret="sec",
        code="authcode",
        redirect_uri="https://app.example.com/cb",
    )
    assert isinstance(token, OIDCTokenResponse)
    assert token.access_token == "at-123"
    assert token.id_token == "idt-456"
    assert token.token_type == "Bearer"
    assert captured["grant_type"] == "authorization_code"
    assert captured["code"] == "authcode"
    assert captured["client_id"] == "cid"
    assert captured["client_secret"] == "sec"


async def test_exchange_code_missing_tokens_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"token_type": "Bearer"})

    client = _make_client(handler)
    with pytest.raises(ExternalServiceError, match="missing access_token and id_token"):
        await client.exchange_code(
            token_endpoint=_TOKEN,
            client_id="cid",
            client_secret="sec",
            code="c",
            redirect_uri="https://app.example.com/cb",
        )


async def test_exchange_code_non_2xx_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="bad code")

    client = _make_client(handler)
    with pytest.raises(ExternalServiceError, match="status=401"):
        await client.exchange_code(
            token_endpoint=_TOKEN,
            client_id="cid",
            client_secret="sec",
            code="c",
            redirect_uri="https://app.example.com/cb",
        )


# ── Userinfo ────────────────────────────────────────────────────────


async def test_fetch_userinfo_sends_bearer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.headers["Authorization"] == "Bearer tok"
        return httpx.Response(200, json={"sub": "s1", "email": "x@example.com"})

    client = _make_client(handler)
    claims = await client.fetch_userinfo(userinfo_endpoint=_USERINFO, access_token="tok")
    assert claims["email"] == "x@example.com"


# ── resolve_userinfo projection ──────────────────────────────────────


async def test_resolve_userinfo_merges_id_token_and_endpoint() -> None:
    id_token = _b64url_json({"sub": "sub-1", "name": "Alice", "email": "alice@example.com"})

    # The userinfo endpoint adds a preferred_username; id_token provides the
    # mapped ``name`` claim.
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer at"
        return httpx.Response(
            200,
            json={"sub": "sub-1", "preferred_username": "alice", "email": "alice@example.com"},
        )

    client = _make_client(handler)
    info = await client.resolve_userinfo(
        user_info_endpoint=_USERINFO,
        access_token="at",
        id_token=f"header.{id_token}.sig",
        username_claim="name",
        email_claim="email",
    )
    assert info.subject == "sub-1"
    assert info.username == "Alice"
    assert info.email == "alice@example.com"
    assert info.claims["preferred_username"] == "alice"


async def test_resolve_userinfo_falls_back_to_id_token_only() -> None:
    # Userinfo endpoint returns 500; id_token claims must still suffice.
    id_token = _b64url_json({"sub": "sub-2", "name": "Bob", "email": "bob@example.com"})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="down")

    client = _make_client(handler)
    info = await client.resolve_userinfo(
        user_info_endpoint=_USERINFO,
        access_token="at",
        id_token=f"header.{id_token}.sig",
        username_claim="name",
        email_claim="email",
    )
    assert info.username == "Bob"
    assert info.email == "bob@example.com"


async def test_resolve_userinfo_falls_back_when_userinfo_ssrf_blocked() -> None:
    # SSRF-blocked userinfo endpoint (IP literal) raises ValidationError
    # before any HTTP call. suppress(ApplicationError) must catch it so the
    # login falls back to id_token claims instead of aborting.
    id_token = _b64url_json({"sub": "sub-4", "name": "Dana", "email": "dana@example.com"})

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("userinfo HTTP should not be attempted; SSRF blocks first")

    client = _make_client(handler)
    info = await client.resolve_userinfo(
        user_info_endpoint="http://127.0.0.1/userinfo",
        access_token="at",
        id_token=f"header.{id_token}.sig",
        username_claim="name",
        email_claim="email",
    )
    assert info.username == "Dana"
    assert info.email == "dana@example.com"


async def test_resolve_userinfo_no_claims_raises() -> None:
    client = _make_client(lambda request: httpx.Response(200, json={}))
    with pytest.raises(ExternalServiceError, match="no usable claims"):
        await client.resolve_userinfo(
            user_info_endpoint="",
            access_token="",
            id_token="",
            username_claim="name",
            email_claim="email",
        )


async def test_resolve_userinfo_username_falls_back_to_email_local_part() -> None:
    id_token = _b64url_json({"sub": "sub-3", "email": "carol@example.com"})
    client = _make_client(lambda request: httpx.Response(200, json={}))
    info = await client.resolve_userinfo(
        user_info_endpoint="",
        access_token="",
        id_token=f"header.{id_token}.sig",
        username_claim="name",
        email_claim="email",
    )
    assert info.username == "carol"


# ── decode_jwt_claims_unverified ─────────────────────────────────────


def test_decode_jwt_claims_unverified_parses_payload() -> None:
    payload = {"sub": "123", "email": "x@example.com", "name": "Alice"}
    token = f"header.{_b64url_json(payload)}.signature"
    claims = decode_jwt_claims_unverified(token)
    assert claims["sub"] == "123"
    assert claims["email"] == "x@example.com"


def test_decode_jwt_claims_unverified_malformed_returns_empty() -> None:
    assert decode_jwt_claims_unverified("not-a-jwt") == {}
    assert decode_jwt_claims_unverified("") == {}


# ── _as_str hardening ────────────────────────────────────────────────


def test_as_str_returns_trimmed_string() -> None:
    from src.ai.oidc_client import _as_str

    assert _as_str("hello") == "hello"
    assert _as_str("  hello  ") == "hello"
    assert _as_str("") == ""


def test_as_str_returns_empty_for_none() -> None:
    from src.ai.oidc_client import _as_str

    assert _as_str(None) == ""


@pytest.mark.parametrize(
    "value",
    [
        42,
        3.14,
        True,
        False,
        [],
        {},
        {"key": "val"},
        ["a", "b"],
    ],
)
def test_as_str_rejects_non_string_non_none(value: object) -> None:
    from src.ai.oidc_client import _as_str

    with pytest.raises(TypeError) as exc_info:
        _as_str(value)
    # Error message must mention the actual type so misconfiguration is debuggable.
    assert type(value).__name__ in str(exc_info.value)


# ── SSRF guard ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/path",
        "http://localhost/path",
        "http://169.254.169.254/latest/meta-data",
        "http://0.0.0.0/",
        "https://10.0.0.1/",
        "https://192.168.1.1/",
        "ftp://idp.example.com/",
        "https://host.internal/",
        "https://svc.local/",
    ],
)
async def test_validate_ssrf_rejects(url: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        await validate_ssrf_safe_url(url)
    assert exc_info.value.code == "oidc.ssrf_blocked"


async def test_validate_ssrf_whitelist_bypasses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SSRF_WHITELIST", "idp.example.com")
    # Whitelisted host -> no DNS, no rejection.
    await validate_ssrf_safe_url("https://idp.example.com/authorize")


async def test_validate_ssrf_empty_url_is_noop() -> None:
    await validate_ssrf_safe_url("")
