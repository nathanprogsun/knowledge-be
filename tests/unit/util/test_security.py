"""Unit tests for :mod:`src.util.security` (password hashing + JWT).

Covers bcrypt hash/verify, access + refresh token round-trips, expiry
enforcement, and tamper detection. Token secrets use the module's
internal ``_secret()`` cache; each round-trip is internally consistent
so no explicit key material is required.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.util.security import (
    TokenError,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)

# ── Password hashing ─────────────────────────────────────────────────


def test_hash_and_verify_password_round_trip() -> None:
    digest = hash_password("s3cret!")
    assert verify_password("s3cret!", digest) is True


def test_verify_password_wrong_returns_false() -> None:
    digest = hash_password("s3cret!")
    assert verify_password("nope", digest) is False


def test_verify_password_invalid_hash_returns_false() -> None:
    assert verify_password("s3cret!", "not-a-bcrypt-digest") is False


def test_hash_password_handles_unicode() -> None:
    digest = hash_password("pässwörd")
    assert verify_password("pässwörd", digest) is True


def test_hash_password_is_non_deterministic() -> None:
    assert hash_password("s3cret!") != hash_password("s3cret!")


# ── Access tokens ───────────────────────────────────────────────────


def test_access_token_round_trip() -> None:
    token, expires_at = create_access_token(user_id="usr-1", email="u@example.com", tenant_id=42)
    claims = decode_token(token)
    assert claims["user_id"] == "usr-1"
    assert claims["email"] == "u@example.com"
    assert claims["tenant_id"] == 42
    assert claims["type"] == "access"
    assert claims["exp"] == int(expires_at.timestamp())


def test_access_token_expiry_is_in_future() -> None:
    _, expires_at = create_access_token(user_id="usr-1", email="u@example.com", tenant_id=None)
    assert expires_at > datetime.now(UTC)


def test_decode_expired_token_raises() -> None:
    token, _ = create_access_token(
        user_id="usr-1",
        email="u@example.com",
        tenant_id=1,
        ttl=timedelta(seconds=-1),
    )
    with pytest.raises(TokenError):
        decode_token(token)


def test_decode_tampered_token_raises() -> None:
    token, _ = create_access_token(user_id="usr-1", email="u@example.com", tenant_id=1)
    # Corrupt the first char of the signature segment. The first char of a
    # base64url segment is always significant (only the last can land in
    # padding-dropped bits), so this reliably invalidates the HMAC without
    # depending on the per-process random signing secret.
    header, payload, signature = token.split(".")
    signature = ("A" if signature[0] != "A" else "B") + signature[1:]
    tampered = f"{header}.{payload}.{signature}"
    with pytest.raises(TokenError):
        decode_token(tampered)


def test_decode_garbage_raises() -> None:
    with pytest.raises(TokenError):
        decode_token("not.a.real.token")


# ── Refresh tokens ──────────────────────────────────────────────────


def test_refresh_token_round_trip() -> None:
    token, expires_at = create_refresh_token(user_id="usr-1")
    claims = decode_token(token)
    assert claims["user_id"] == "usr-1"
    assert claims["type"] == "refresh"
    assert "tenant_id" not in claims
    assert claims["exp"] == int(expires_at.timestamp())


def test_refresh_token_expiry_is_in_future() -> None:
    _, expires_at = create_refresh_token(user_id="usr-1")
    assert expires_at > datetime.now(UTC)
