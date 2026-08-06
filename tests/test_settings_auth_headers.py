"""Tests for the auth header name constants on `Settings`."""

from __future__ import annotations

from src.settings import Settings


def test_auth_header_user_id_default() -> None:
    assert Settings().auth_header_user_id == "x-knowledge-user-id"


def test_auth_header_tenant_id_default() -> None:
    assert Settings().auth_header_tenant_id == "x-knowledge-tenant-id"


def test_auth_header_roles_default() -> None:
    assert Settings().auth_header_roles == "x-knowledge-roles"
