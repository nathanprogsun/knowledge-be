"""Shared helpers for the IM platform adapters.

Small pure utilities the concrete adapters reuse: PKCS#7-unpadded
AES-CBC decryption, JSON int coercion, credential extraction, and the
no-op stop callable webhook adapters return from ``connect``.

Deliberately platform-agnostic — platform-specific key derivation and
payload framing live in the individual adapters.
"""

from __future__ import annotations

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from src.common.json import JsonObject, JsonValue

#: AES block size in bytes (also the PKCS#7 padding block).
_BLOCK_SIZE = 16


def pkcs7_unpad(padded: bytes) -> bytes:
    """Strip PKCS#7 padding from ``padded``; raise ``ValueError`` when invalid."""
    if not padded:
        raise ValueError("empty plaintext")
    pad_len = padded[-1]
    if pad_len < 1 or pad_len > _BLOCK_SIZE or pad_len > len(padded):
        raise ValueError("invalid padding")
    if padded[-pad_len:] != bytes([pad_len]) * pad_len:
        raise ValueError("invalid padding")
    return padded[:-pad_len]


def aes_cbc_decrypt(key: bytes, iv: bytes, ciphertext: bytes) -> bytes:
    """AES-CBC decrypt ``ciphertext`` and strip PKCS#7 padding.

    Raises ``ValueError`` when the ciphertext is not a multiple of the
    block size or the padding is invalid. CBC provides no authenticity
    guarantee; callers still verify the platform token / signature
    around the decrypted payload.
    """
    decryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()
    return pkcs7_unpad(padded)


def json_int(value: JsonValue) -> int:
    """Coerce a JSON scalar to an ``int`` for API error-code comparisons.

    Booleans are coerced to ``1`` / ``0`` (JSON ``true`` is a number in
    most platform responses); ``None`` and non-numeric values become
    ``-1`` so the caller can treat them as an unexpected/absent code.
    """
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return -1


def credential_string(credentials: JsonObject, key: str) -> str:
    """Read ``key`` from ``credentials`` as a string.

    Booleans are explicitly rejected so ``True``/``False`` never
    round-trip into an adapter credential; numbers are rendered without
    a decimal point, matching the platform identity fields.
    """
    value = credentials.get(key)
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return f"{value:.0f}"
    return ""


def noop_stop() -> None:
    """No-op teardown returned by webhook adapters' ``connect``.

    Webhook mode holds no persistent connection, so the supervisor's
    stop contract is satisfied with a callable that does nothing.
    """


__all__ = [
    "aes_cbc_decrypt",
    "credential_string",
    "json_int",
    "noop_stop",
    "pkcs7_unpad",
]
