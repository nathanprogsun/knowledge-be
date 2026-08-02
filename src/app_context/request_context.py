"""Request-scoped context using contextvars.

Carries tenant_id, user_id, request_id, and locale across async boundaries
without module globals. Middleware sets values on request entry; deeper
code reads them via the `get_*` accessors.
"""

from __future__ import annotations

from contextvars import ContextVar, Token

_tenant_id: ContextVar[str | None] = ContextVar("tenant_id", default=None)
_user_id: ContextVar[str | None] = ContextVar("user_id", default=None)
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_locale: ContextVar[str] = ContextVar("locale", default="en-US")


def set_tenant_id(value: str) -> Token[str | None]:
    return _tenant_id.set(value)


def get_tenant_id() -> str | None:
    return _tenant_id.get()


def set_user_id(value: str) -> Token[str | None]:
    return _user_id.set(value)


def get_user_id() -> str | None:
    return _user_id.get()


def set_request_id(value: str) -> Token[str | None]:
    return _request_id.set(value)


def get_request_id() -> str | None:
    return _request_id.get()


def set_locale(value: str) -> Token[str]:
    return _locale.set(value)


def get_locale() -> str:
    return _locale.get()


def reset(
    token_tenant: Token[str | None],
    token_user: Token[str | None],
    token_request: Token[str | None],
    token_locale: Token[str],
) -> None:
    """Restore all four contextvars to their previous values.

    Middleware should capture tokens on entry and call this in a finally.
    """
    _tenant_id.reset(token_tenant)
    _user_id.reset(token_user)
    _request_id.reset(token_request)
    _locale.reset(token_locale)


__all__ = [
    "get_locale",
    "get_request_id",
    "get_tenant_id",
    "get_user_id",
    "reset",
    "set_locale",
    "set_request_id",
    "set_tenant_id",
    "set_user_id",
]
