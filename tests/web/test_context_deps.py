"""Unit tests for the FastAPI context deps in ``src.web.deps.context``.

The deps read the authenticated principal from ``request.state`` and
return typed values. Tests assert the typed signatures and the
int-coercion default behaviour.
"""

from __future__ import annotations

from fastapi import Request

from src.web.deps.context import (
    get_api_key_scope_dep,
    get_is_system_admin_dep,
    get_tenant_id_dep,
    get_tenant_role_dep,
    get_user_id_dep,
    get_user_info_dep,
)


def _make_request(state: dict[str, object]) -> Request:
    """Build a minimal Starlette ``Request`` with the given state dict."""
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/test",
        "headers": [],
        "query_string": b"",
    }
    request = Request(scope)
    for k, v in state.items():
        setattr(request.state, k, v)
    return request


def test_get_tenant_id_dep_returns_int() -> None:
    request = _make_request({"tenant_id": "42"})
    assert get_tenant_id_dep(request) == 42


def test_get_tenant_id_dep_defaults_to_zero() -> None:
    request = _make_request({})
    assert get_tenant_id_dep(request) == 0


def test_get_tenant_role_dep_returns_str() -> None:
    request = _make_request({"tenant_role": "owner"})
    assert get_tenant_role_dep(request) == "owner"


def test_get_tenant_role_dep_defaults_to_empty() -> None:
    request = _make_request({})
    assert get_tenant_role_dep(request) == ""


def test_get_is_system_admin_dep_returns_bool() -> None:
    request = _make_request({"is_system_admin": True})
    assert get_is_system_admin_dep(request) is True


def test_get_user_id_dep_returns_user_id_when_set() -> None:
    request = _make_request({"user_info": {"id": "u-1"}})
    assert get_user_id_dep(request) == "u-1"


def test_get_user_id_dep_returns_none_when_unset() -> None:
    request = _make_request({})
    assert get_user_id_dep(request) is None


def test_get_user_info_dep_returns_dict_or_none() -> None:
    request = _make_request({"user_info": {"id": "u-1"}})
    assert get_user_info_dep(request) == {"id": "u-1"}

    empty = _make_request({})
    assert get_user_info_dep(empty) is None


def test_get_api_key_scope_dep_returns_none_by_default() -> None:
    request = _make_request({})
    assert get_api_key_scope_dep(request) is None
