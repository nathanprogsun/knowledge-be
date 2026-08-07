"""Sync wrappers around the ``system/admin`` router endpoints.

Endpoint names match the handler function names in
``src/web/api/system/router.py``.
"""

from __future__ import annotations

from typing import Any

import httpx

from tests.integration.web.api.util.api_client import APITestClient


def list_settings(client: APITestClient) -> httpx.Response:
    """``GET /system/admin/settings`` - every known setting (System Admin)."""
    return client.get(endpoint_name="list_settings")


def get_setting(client: APITestClient, key: str) -> httpx.Response:
    """``GET /system/admin/settings/{key}`` - one setting by key (System Admin)."""
    return client.get(
        endpoint_name="get_setting",
        path_params={"key": key},
    )


def update_setting(
    client: APITestClient,
    key: str,
    *,
    body: dict[str, Any],
) -> httpx.Response:
    """``PUT /system/admin/settings/{key}`` - set value (System Admin)."""
    return client.put(
        endpoint_name="update_setting",
        path_params={"key": key},
        json=body,
    )


def reset_setting(client: APITestClient, key: str) -> httpx.Response:
    """``DELETE /system/admin/settings/{key}`` - reset value (System Admin)."""
    return client.delete(
        endpoint_name="reset_setting",
        path_params={"key": key},
    )


def list_system_audit_log(
    client: APITestClient,
    *,
    after_id: int = 0,
    limit: int = 50,
    action: str | None = None,
    outcome: str | None = None,
    actor: str | None = None,
) -> httpx.Response:
    """``GET /system/admin/audit-log`` - system-scope feed (System Admin)."""
    params: dict[str, Any] = {"after_id": after_id, "limit": limit}
    if action is not None:
        params["action"] = action
    if outcome is not None:
        params["outcome"] = outcome
    if actor is not None:
        params["actor"] = actor
    return client.get(endpoint_name="list_system_audit_log", params=params)


__all__ = [
    "get_setting",
    "list_settings",
    "list_system_audit_log",
    "reset_setting",
    "update_setting",
]
