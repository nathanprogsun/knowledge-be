"""Unit tests for the SSRF-safe URL validator and HTTP client builder.

The validator is exercised directly with a curated set of valid /
invalid URLs (loopback, RFC1918, restricted suffixes, obfuscated IP
forms, scheme and length rules) and via the ``SSRF_WHITELIST``
environment variable. The client builder is checked for both proxy
and proxy-less paths and for the no-redirect default.
"""

from __future__ import annotations

import httpx
import pytest
from pytest import MonkeyPatch

from src.ai.web_search_clients.proxy import (
    build_http_client,
    validate_url_for_ssrf,
)
from src.common.exception import ValidationError

_MAX_URL_LENGTH = 2048


# ── Happy path ────────────────────────────────────────────────────────


def test_empty_url_returns_empty() -> None:
    assert validate_url_for_ssrf("") == ""
    assert validate_url_for_ssrf("   ") == ""


def test_https_url_passes_through() -> None:
    assert validate_url_for_ssrf("https://example.com/proxy") == "https://example.com/proxy"


def test_http_url_passes_through() -> None:
    assert validate_url_for_ssrf("http://example.com:8080/proxy") == "http://example.com:8080/proxy"


def test_url_without_scheme_gets_https() -> None:
    assert validate_url_for_ssrf("example.com/proxy") == "https://example.com/proxy"


def test_url_is_stripped() -> None:
    assert validate_url_for_ssrf("  https://example.com/proxy  ") == "https://example.com/proxy"


# ── Length ────────────────────────────────────────────────────────────


def test_url_length_limit() -> None:
    too_long = "https://example.com/" + "a" * _MAX_URL_LENGTH
    with pytest.raises(ValidationError) as excinfo:
        validate_url_for_ssrf(too_long)
    assert excinfo.value.code == "web_search_provider.ssrf_blocked"


def test_url_at_length_limit_passes() -> None:
    # A URL exactly at the cap should still be accepted.
    base = "https://example.com/"
    url = base + "a" * (_MAX_URL_LENGTH - len(base))
    assert validate_url_for_ssrf(url) == url


# ── Scheme ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com",
        "file:///etc/passwd",
        "gopher://example.com",
        "ssh://example.com",
    ],
)
def test_invalid_scheme_rejected(url: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        validate_url_for_ssrf(url)
    assert excinfo.value.code == "web_search_provider.ssrf_blocked"


def test_url_without_hostname_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        validate_url_for_ssrf("https:///path")
    assert excinfo.value.code == "web_search_provider.ssrf_blocked"


# ── Restricted hostnames ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost",
        "http://localhost:8080/path",
        "http://LOCALHOST",
        "http://metadata",
        "http://metadata.google.internal",
        "http://ip6-localhost",
        "http://ip6-loopback",
    ],
)
def test_restricted_hostnames_rejected(url: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        validate_url_for_ssrf(url)
    assert excinfo.value.code == "web_search_provider.ssrf_blocked"


# ── Restricted suffixes ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        "http://api.internal/path",
        "http://printer.local",
        "http://printer.localhost",
        "http://nas.home",
        "http://dev.lan",
        "http://corp.intra",
    ],
)
def test_restricted_suffixes_rejected(url: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        validate_url_for_ssrf(url)
    assert excinfo.value.code == "web_search_provider.ssrf_blocked"


# ── Direct IP literals ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1",
        "http://10.0.0.5:8080",
        "http://192.168.1.1",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]",
        "http://[2001:db8::1]",
    ],
)
def test_direct_ip_literal_rejected(url: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        validate_url_for_ssrf(url)
    assert excinfo.value.code == "web_search_provider.ssrf_blocked"


# ── IP-like hostname obfuscation ──────────────────────────────────────


def test_decimal_ip_like_hostname_rejected() -> None:
    # 2130706433 == 127.0.0.1 in decimal form. The validator's
    # ``isdigit()`` fast path catches a pure decimal that fits IPv4.
    with pytest.raises(ValidationError) as excinfo:
        validate_url_for_ssrf("http://2130706433")
    assert excinfo.value.code == "web_search_provider.ssrf_blocked"


def test_octal_ip_like_hostname_rejected() -> None:
    # 0177.0.0.1 == 127.0.0.1 in octal form.
    with pytest.raises(ValidationError) as excinfo:
        validate_url_for_ssrf("http://0177.0.0.1")
    assert excinfo.value.code == "web_search_provider.ssrf_blocked"


def test_hex_octet_obfuscation_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        validate_url_for_ssrf("http://0x7f.0.0.1")
    assert excinfo.value.code == "web_search_provider.ssrf_blocked"


def test_non_ip_like_string_passes_ip_like_check() -> None:
    # A plain alphanumeric hostname is not IP-like.
    assert validate_url_for_ssrf("https://proxy.example.com") == "https://proxy.example.com"


# ── Whitelist ─────────────────────────────────────────────────────────


def test_whitelist_exact_host_skips_heavy_checks(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("SSRF_WHITELIST", "127.0.0.1,localhost")
    assert validate_url_for_ssrf("http://127.0.0.1:3128") == "http://127.0.0.1:3128"


def test_whitelist_suffix_skips_heavy_checks(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("SSRF_WHITELIST", "*.trusted.example.com")
    assert (
        validate_url_for_ssrf("http://api.trusted.example.com") == "http://api.trusted.example.com"
    )


def test_whitelist_cidr_skips_heavy_checks(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("SSRF_WHITELIST", "10.0.0.0/8")
    assert validate_url_for_ssrf("http://10.5.6.7:8080") == "http://10.5.6.7:8080"


def test_whitelist_with_invalid_cidr_is_ignored(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("SSRF_WHITELIST", "not-a-cidr,example.com")
    # example.com is whitelisted (bare host), the bad CIDR is silently
    # dropped.
    assert validate_url_for_ssrf("https://example.com/x") == "https://example.com/x"


def test_unsetting_whitelist_still_blocks_loopback(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.delenv("SSRF_WHITELIST", raising=False)
    with pytest.raises(ValidationError):
        validate_url_for_ssrf("http://127.0.0.1")


# ── Redirect-style usage ─────────────────────────────────────────────


def test_redirect_target_validated_like_any_other_url(
    monkeypatch: MonkeyPatch,
) -> None:
    """Redirect targets follow the same validation rules."""
    monkeypatch.delenv("SSRF_WHITELIST", raising=False)
    # A redirect that points to a private IP must be rejected.
    with pytest.raises(ValidationError):
        validate_url_for_ssrf("http://10.0.0.5/redirect-target")
    # A public redirect target is accepted.
    assert (
        validate_url_for_ssrf("https://example.com/redirect-target")
        == "https://example.com/redirect-target"
    )


def test_redirect_to_restricted_hostname_rejected(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.delenv("SSRF_WHITELIST", raising=False)
    with pytest.raises(ValidationError):
        validate_url_for_ssrf("http://metadata.google.internal/")


# ── build_http_client ─────────────────────────────────────────────────


def test_build_http_client_without_proxy_returns_default_client() -> None:
    client = build_http_client(timeout=5.0)
    try:
        assert client.timeout.connect == 5.0
        # Default httpx clients do not follow redirects.
        assert client.follow_redirects is False
        # Environment proxy is honored when no explicit proxy is given.
        assert client.trust_env is True
    finally:
        client.close()


def test_build_http_client_with_valid_proxy_sets_explicit_proxy() -> None:
    client = build_http_client(
        timeout=10.0,
        proxy_url="https://proxy.example.com:3128",
    )
    try:
        assert client.timeout.connect == 10.0
        # An explicit proxy disables trust_env so the two paths can't
        # both apply.
        assert client.trust_env is False
        # Redirects stay disabled.
        assert client.follow_redirects is False
    finally:
        client.close()


def test_build_http_client_rejects_invalid_proxy() -> None:
    with pytest.raises(ValidationError) as excinfo:
        build_http_client(timeout=5.0, proxy_url="http://127.0.0.1:3128")
    assert excinfo.value.code == "web_search_provider.ssrf_blocked"


def test_build_http_client_accepts_whitelisted_loopback(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("SSRF_WHITELIST", "127.0.0.1")
    client = build_http_client(timeout=5.0, proxy_url="http://127.0.0.1:3128")
    try:
        assert isinstance(client, httpx.Client)
        assert client.trust_env is False
    finally:
        client.close()


def test_build_http_client_whitespace_proxy_treated_as_empty() -> None:
    client = build_http_client(timeout=5.0, proxy_url="   ")
    try:
        # An empty / whitespace proxy falls back to env proxy.
        assert client.trust_env is True
    finally:
        client.close()
