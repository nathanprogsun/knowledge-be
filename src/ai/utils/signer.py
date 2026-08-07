"""Request signing for the managed cloud chat service.

Mirrors the upstream ``Sign`` helper. The signature is an MD5 over the
sorted, RFC 3986-encoded ``key=value`` pairs of ``{x-appid, x-api-key,
x-request-id, x-timestamp, x-nonce, body}``, where ``body`` is the MD5 of
the request-body JSON (``"{}"`` when the body is empty). The result is a
set of request headers with the uppercase ``X-*`` names the service
expects.

``timestamp`` and ``nonce`` are accepted as keyword arguments for
deterministic testing; production callers omit them and the values are
generated exactly as upstream does (current Unix seconds and 16 random
alphanumeric characters).
"""

from __future__ import annotations

import hashlib
import random
import time

_NONCE_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_NONCE_LENGTH = 16
_UNRESERVED = frozenset("-_.~")


def _md5_hex(value: str) -> str:
    """Lowercase hex MD5 digest of ``value`` (Go ``md5Hex``)."""
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def _generate_nonce(length: int) -> str:
    """Random alphanumeric nonce of ``length`` characters (Go ``generateNonce``)."""
    return "".join(random.choice(_NONCE_CHARS) for _ in range(length))


def _rfc3986_encode(value: str) -> str:
    """Percent-encode ``value`` per RFC 3986, preserving ``A-Z a-z 0-9 - _ . ~``.

    Mirrors the upstream rune-based encoder: every other code point is
    emitted as ``%`` followed by its uppercase hex value.
    """
    out: list[str] = []
    for ch in value:
        is_ascii_alnum = "A" <= ch <= "Z" or "a" <= ch <= "z" or "0" <= ch <= "9"
        if is_ascii_alnum or ch in _UNRESERVED:
            out.append(ch)
        else:
            out.append(f"%{ord(ch):02X}")
    return "".join(out)


def sign_request(
    app_id: str,
    api_key: str,
    request_id: str,
    body_json: str = "{}",
    *,
    timestamp: str | None = None,
    nonce: str | None = None,
) -> dict[str, str]:
    """Generate the signed ``X-*`` headers for one request.

    Args:
        app_id: The upstream application id.
        api_key: The upstream API key (currently carried by the app-secret
            field).
        request_id: A per-request unique id (UUID string).
        body_json: The request-body JSON; an empty string is treated as
            ``"{}"``.
        timestamp: Optional Unix-seconds string override (tests).
        nonce: Optional nonce override (tests).
    """
    timestamp_value = timestamp if timestamp is not None else str(int(time.time()))
    nonce_value = nonce if nonce is not None else _generate_nonce(_NONCE_LENGTH)

    body_for_hash = body_json if body_json else "{}"
    body_md5 = _md5_hex(body_for_hash)

    params = {
        "x-appid": app_id,
        "x-api-key": api_key,
        "x-request-id": request_id,
        "x-timestamp": timestamp_value,
        "x-nonce": nonce_value,
        "body": body_md5,
    }

    parts = [
        f"{_rfc3986_encode(key)}={_rfc3986_encode(params[key])}"
        for key in sorted(params)
    ]
    signature = _md5_hex("&".join(parts))

    return {
        "X-APPID": app_id,
        "X-API-Key": api_key,
        "X-Request-ID": request_id,
        "X-Timestamp": timestamp_value,
        "X-Nonce": nonce_value,
        "X-Signature": signature,
    }


# Go-facing alias: the upstream helper is named ``Sign``.
sign = sign_request


__all__ = ["_generate_nonce", "_md5_hex", "_rfc3986_encode", "sign", "sign_request"]
