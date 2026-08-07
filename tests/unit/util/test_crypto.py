"""Unit tests for :mod:`src.util.crypto` (AES-256-GCM credential encryption).

Covers the ``enc:v1:`` wire format, round-trip, tamper / wrong-key
failure, legacy-plaintext pass-through, and the lenient decrypt path.
``get_aes_key`` is patched where the settings layer would otherwise be
exercised, so no real ``SYSTEM_AES_KEY`` is required.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest
from cryptography.exceptions import InvalidTag

from src.util.crypto import (
    ENC_PREFIX,
    decrypt_aesgcm,
    decrypt_stored_secret_lenient,
    encrypt_aesgcm,
    get_aes_key,
)

# AES-256 requires a raw 32-byte key.
_KEY = b"0123456789abcdef0123456789abcdef"


# ── Round-trip ───────────────────────────────────────────────────────


def test_encrypt_decrypt_round_trip() -> None:
    cipher = encrypt_aesgcm("secret-value", _KEY)
    assert decrypt_aesgcm(cipher, _KEY) == "secret-value"


def test_encrypt_produces_enc_prefix() -> None:
    cipher = encrypt_aesgcm("secret-value", _KEY)
    assert cipher.startswith(ENC_PREFIX)


def test_encrypt_is_non_deterministic() -> None:
    a = encrypt_aesgcm("secret-value", _KEY)
    b = encrypt_aesgcm("secret-value", _KEY)
    assert a != b  # random nonce per call


# ── Pass-through cases ───────────────────────────────────────────────


def test_encrypt_empty_returns_empty() -> None:
    assert encrypt_aesgcm("", _KEY) == ""


def test_encrypt_already_encrypted_passes_through() -> None:
    already = ENC_PREFIX + "deadbeef"
    assert encrypt_aesgcm(already, _KEY) == already


def test_decrypt_legacy_plaintext_passes_through() -> None:
    assert decrypt_aesgcm("plain-legacy", _KEY) == "plain-legacy"
    assert decrypt_aesgcm("", _KEY) == ""


# ── Failure modes ────────────────────────────────────────────────────


def test_decrypt_tampered_raises_invalid_tag() -> None:
    cipher = encrypt_aesgcm("secret-value", _KEY)
    payload = cipher[len(ENC_PREFIX) :]
    data = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
    # Flip a byte in the ciphertext (past the 12-byte nonce) so the GCM
    # auth tag always fails, regardless of the random nonce. (Flipping a
    # base64 char is unreliable: the last char can land in the padding
    # bits that b64decode drops, leaving the ciphertext untouched.)
    tampered_data = data[:13] + bytes([data[13] ^ 0xFF]) + data[14:]
    tampered_payload = base64.urlsafe_b64encode(tampered_data).rstrip(b"=").decode("ascii")
    with pytest.raises(InvalidTag):
        decrypt_aesgcm(ENC_PREFIX + tampered_payload, _KEY)


def test_decrypt_wrong_key_raises_invalid_tag() -> None:
    cipher = encrypt_aesgcm("secret-value", _KEY)
    other = b"fedcba9876543210fedcba9876543210"  # different 32-byte key
    with pytest.raises(InvalidTag):
        decrypt_aesgcm(cipher, other)


def test_decrypt_short_payload_raises_value_error() -> None:
    # The nonce is 12 bytes; a payload shorter than that is malformed.
    short_blob = base64.urlsafe_b64encode(b"short").rstrip(b"=").decode("ascii")
    with pytest.raises(ValueError, match="too short"):
        decrypt_aesgcm(ENC_PREFIX + short_blob, _KEY)


# ── get_aes_key ──────────────────────────────────────────────────────


def test_get_aes_key_returns_32_byte_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.util.crypto.get_settings",
        lambda: SimpleNamespace(system_aes_key="a" * 32),
    )
    assert get_aes_key() == b"a" * 32


def test_get_aes_key_returns_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.util.crypto.get_settings",
        lambda: SimpleNamespace(system_aes_key=""),
    )
    assert get_aes_key() is None


def test_get_aes_key_returns_none_when_wrong_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.util.crypto.get_settings",
        lambda: SimpleNamespace(system_aes_key="too-short"),
    )
    assert get_aes_key() is None


# ── decrypt_stored_secret_lenient ───────────────────────────────────


def test_lenient_decrypt_success(monkeypatch: pytest.MonkeyPatch) -> None:
    cipher = encrypt_aesgcm("secret-value", _KEY)
    monkeypatch.setattr("src.util.crypto.get_aes_key", lambda: _KEY)
    assert decrypt_stored_secret_lenient(cipher) == ("secret-value", True)


def test_lenient_decrypt_legacy_plaintext() -> None:
    assert decrypt_stored_secret_lenient("plain-legacy") == ("plain-legacy", True)
    assert decrypt_stored_secret_lenient("") == ("", True)


def test_lenient_decrypt_missing_key_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cipher = encrypt_aesgcm("secret-value", _KEY)
    monkeypatch.setattr("src.util.crypto.get_aes_key", lambda: None)
    assert decrypt_stored_secret_lenient(cipher) == ("", False)
