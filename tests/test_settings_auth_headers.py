"""Tests for the auth header name constants on `Settings`."""

from __future__ import annotations

from src.settings import Settings


def test_auth_header_user_id_default() -> None:
    assert Settings().auth_header_user_id == "X-User-Id"


def test_auth_header_tenant_id_default() -> None:
    assert Settings().auth_header_tenant_id == "X-Tenant-ID"


def test_auth_header_roles_default() -> None:
    assert Settings().auth_header_roles == "X-Roles"
