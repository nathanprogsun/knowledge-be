"""AES-256-GCM credential encryption.

Format:

- ``enc:v1:`` prefix + ``base64.RawURLEncoding(nonce || ciphertext)``.
- ``SYSTEM_AES_KEY`` is a raw 32-byte key (not hex); a key of any other
  length is treated as unset.
- Values without the ``enc:v1:`` prefix are legacy plaintext and pass
  through untouched, so historical rows keep working without a
  migration step.
- Decryption failures (rotated/missing key) surface as ``(value, False)``
  via :func:`decrypt_stored_secret_lenient` so callers can blank the
  field and stay visible.
"""

from __future__ import annotations

import base64
import secrets

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from src.settings import get_settings

ENC_PREFIX = "enc:v1:"

# AES-256-GCM nonce size (12 bytes, the GCM standard).
_NONCE_SIZE = 12


def get_aes_key() -> bytes | None:
    """Return the 32-byte AES key from ``SYSTEM_AES_KEY``, or ``None``.

    Mirrors Go ``GetAESKey``: the env value is used verbatim (raw
    bytes, not hex) and only when it is exactly 32 bytes long.
    """
    raw = get_settings().system_aes_key
    if not raw:
        return None
    key = raw.encode("utf-8")
    if len(key) != 32:
        return None
    return key


def encrypt_aesgcm(plaintext: str, key: bytes) -> str:
    """Encrypt ``plaintext`` with AES-256-GCM, Go wire format.

    Empty input or an already-encrypted value is returned unchanged
    (mirrors Go ``EncryptAESGCM``).
    """
    if not plaintext or plaintext.startswith(ENC_PREFIX):
        return plaintext
    nonce = secrets.token_bytes(_NONCE_SIZE)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext.encode("utf-8"), None)
    combined = nonce + ciphertext
    return ENC_PREFIX + base64.urlsafe_b64encode(combined).rstrip(b"=").decode("ascii")


def decrypt_aesgcm(encrypted: str, key: bytes) -> str:
    """Decrypt an ``enc:v1:`` blob; legacy plaintext passes through.

    Raises ``ValueError`` on malformed or tampered ciphertext.
    """
    if not encrypted or not encrypted.startswith(ENC_PREFIX):
        return encrypted
    payload = encrypted[len(ENC_PREFIX) :]
    data = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
    if len(data) < _NONCE_SIZE:
        raise ValueError("invalid encrypted data: too short")
    nonce, ciphertext = data[:_NONCE_SIZE], data[_NONCE_SIZE:]
    plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
    return plaintext.decode("utf-8")


def decrypt_stored_secret_lenient(encrypted: str) -> tuple[str, bool]:
    """Decrypt a stored secret without failing the load.

    Returns ``(plaintext, True)`` on success (including legacy
    plaintext), ``("", False)`` when the value carries the ``enc:v1:``
    prefix but cannot be decrypted (missing/rotated key) — the caller
    blanks the field so the row stays visible.
    """
    if not encrypted or not encrypted.startswith(ENC_PREFIX):
        return encrypted, True
    key = get_aes_key()
    if key is None:
        return "", False
    try:
        return decrypt_aesgcm(encrypted, key), True
    except (ValueError, OSError):
        return "", False


__all__ = [
    "ENC_PREFIX",
    "decrypt_aesgcm",
    "decrypt_stored_secret_lenient",
    "encrypt_aesgcm",
    "get_aes_key",
]
