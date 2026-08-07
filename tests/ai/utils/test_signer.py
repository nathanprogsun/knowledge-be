"""Tests for the managed-cloud request signer.

Deterministic assertions use fixed ``timestamp`` / ``nonce`` overrides;
the algorithm under test is re-derived in the tests so the expected
headers and signature are verified independently.
"""

from __future__ import annotations

import hashlib

from src.ai.utils.signer import _generate_nonce, _md5_hex, _rfc3986_encode, sign, sign_request

_APP_ID = "app-123"
_API_KEY = "key-456"
_REQUEST_ID = "req-0001"
_BODY = '{"model": "qwen-max"}'
_TIMESTAMP = "1700000000"
_NONCE = "abcDEF0123456789"


def _expected_signature(
    *,
    body: str,
    timestamp: str,
    nonce: str,
) -> str:
    params = {
        "body": _md5_hex(body if body else "{}"),
        "x-api-key": _API_KEY,
        "x-appid": _APP_ID,
        "x-nonce": nonce,
        "x-request-id": _REQUEST_ID,
        "x-timestamp": timestamp,
    }
    parts = [f"{_rfc3986_encode(k)}={_rfc3986_encode(v)}" for k, v in sorted(params.items())]
    return _md5_hex("&".join(parts))


def test_sign_request_returns_all_six_headers() -> None:
    headers = sign_request(
        _APP_ID,
        _API_KEY,
        _REQUEST_ID,
        _BODY,
        timestamp=_TIMESTAMP,
        nonce=_NONCE,
    )
    assert set(headers) == {
        "X-APPID",
        "X-API-Key",
        "X-Request-ID",
        "X-Timestamp",
        "X-Nonce",
        "X-Signature",
    }
    assert headers["X-APPID"] == _APP_ID
    assert headers["X-API-Key"] == _API_KEY
    assert headers["X-Request-ID"] == _REQUEST_ID
    assert headers["X-Timestamp"] == _TIMESTAMP
    assert headers["X-Nonce"] == _NONCE


def test_signature_is_md5_over_sorted_encoded_params() -> None:
    headers = sign_request(
        _APP_ID,
        _API_KEY,
        _REQUEST_ID,
        _BODY,
        timestamp=_TIMESTAMP,
        nonce=_NONCE,
    )
    assert headers["X-Signature"] == _expected_signature(
        body=_BODY,
        timestamp=_TIMESTAMP,
        nonce=_NONCE,
    )


def test_sign_is_alias_for_sign_request() -> None:
    assert sign is sign_request


def test_empty_body_is_hashed_as_empty_object() -> None:
    headers = sign_request(
        _APP_ID,
        _API_KEY,
        _REQUEST_ID,
        "",
        timestamp=_TIMESTAMP,
        nonce=_NONCE,
    )
    assert headers["X-Signature"] == _expected_signature(
        body="{}",
        timestamp=_TIMESTAMP,
        nonce=_NONCE,
    )


def test_different_body_produces_different_signature() -> None:
    first = sign_request(
        _APP_ID,
        _API_KEY,
        _REQUEST_ID,
        _BODY,
        timestamp=_TIMESTAMP,
        nonce=_NONCE,
    )
    second = sign_request(
        _APP_ID,
        _API_KEY,
        _REQUEST_ID,
        '{"model": "other"}',
        timestamp=_TIMESTAMP,
        nonce=_NONCE,
    )
    assert first["X-Signature"] != second["X-Signature"]


def test_generated_timestamp_and_nonce_are_fresh() -> None:
    first = sign_request(_APP_ID, _API_KEY, _REQUEST_ID, _BODY)
    second = sign_request(_APP_ID, _API_KEY, _REQUEST_ID, _BODY)
    # Nonce is 16 alphanumeric characters and differs between calls.
    assert len(first["X-Nonce"]) == 16
    assert first["X-Nonce"].isalnum()
    assert first["X-Nonce"] != second["X-Nonce"]
    # Timestamp looks like Unix seconds.
    assert first["X-Timestamp"].isdigit()


def test_rfc3986_encode_preserves_unreserved_characters() -> None:
    assert _rfc3986_encode("ABCabc012-_.~") == "ABCabc012-_.~"


def test_rfc3986_encode_percent_encodes_other_characters() -> None:
    assert _rfc3986_encode("a b&c") == "a%20b%26c"


def test_rfc3986_encode_uses_rune_value_for_non_ascii() -> None:
    # Matches the upstream rune-based encoder (code point value, not UTF-8
    # byte sequence).
    assert _rfc3986_encode("中") == "%4E2D"


def test_generate_nonce_length() -> None:
    assert len(_generate_nonce(16)) == 16
    assert len(_generate_nonce(8)) == 8
    assert _generate_nonce(16).isalnum()


def test_md5_hex_is_lowercase() -> None:
    assert _md5_hex("abc") == hashlib.md5(b"abc").hexdigest()
