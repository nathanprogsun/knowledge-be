"""Sync wrappers around the ``mcp-services`` router endpoints.

Endpoint names match the handler function names in
``src/web/api/infra/mcp_services/router.py``. The ``sse`` and
``messages`` paths used by the live-MCP HTTP transport are not in
this module because they live outside the JSON RPC surface; see the
module-level test (``test_live_endpoints_controller.py``).
"""

from __future__ import annotations

from typing import Any

import httpx

from tests.integration.web.api.util.api_client import APITestClient


def create_mcp_service(client: APITestClient, *, body: dict[str, Any]) -> httpx.Response:
    """``POST /mcp-services`` - register one MCP service (Admin)."""
    return client.post(endpoint_name="create_mcp_service", json=body)


def list_mcp_services(client: APITestClient) -> httpx.Response:
    """``GET /mcp-services`` - list the workspace's services (Viewer)."""
    return client.get(endpoint_name="list_mcp_services")


def get_mcp_service(client: APITestClient, service_id: str) -> httpx.Response:
    """``GET /mcp-services/{service_id}`` - fetch one service (Viewer)."""
    return client.get(
        endpoint_name="get_mcp_service",
        path_params={"service_id": service_id},
    )


def update_mcp_service(
    client: APITestClient,
    service_id: str,
    *,
    body: dict[str, Any],
) -> httpx.Response:
    """``PUT /mcp-services/{service_id}`` - patch one service (Admin)."""
    return client.put(
        endpoint_name="update_mcp_service",
        path_params={"service_id": service_id},
        json=body,
    )


def delete_mcp_service(client: APITestClient, service_id: str) -> httpx.Response:
    """``DELETE /mcp-services/{service_id}`` - soft-delete one service (Admin)."""
    return client.delete(
        endpoint_name="delete_mcp_service",
        path_params={"service_id": service_id},
    )


def test_mcp_service(client: APITestClient, service_id: str) -> httpx.Response:
    """``POST /mcp-services/{service_id}/test`` - connectivity probe (Admin)."""
    return client.post(
        endpoint_name="test_mcp_service",
        path_params={"service_id": service_id},
    )


def list_mcp_service_tools(client: APITestClient, service_id: str) -> httpx.Response:
    """``GET /mcp-services/{service_id}/tools`` - discovered tools (Viewer)."""
    return client.get(
        endpoint_name="list_mcp_service_tools",
        path_params={"service_id": service_id},
    )


def list_mcp_service_resources(client: APITestClient, service_id: str) -> httpx.Response:
    """``GET /mcp-services/{service_id}/resources`` - discovered resources (Viewer)."""
    return client.get(
        endpoint_name="list_mcp_service_resources",
        path_params={"service_id": service_id},
    )


def list_mcp_tool_approvals(client: APITestClient, service_id: str) -> httpx.Response:
    """``GET /mcp-services/{service_id}/tool-approvals`` - approval overrides (Viewer)."""
    return client.get(
        endpoint_name="list_mcp_tool_approvals",
        path_params={"service_id": service_id},
    )


def set_mcp_tool_approval(
    client: APITestClient,
    service_id: str,
    tool_name: str,
    *,
    body: dict[str, Any],
) -> httpx.Response:
    """``PUT /mcp-services/{service_id}/tool-approvals/{tool_name}`` (Admin)."""
    return client.put(
        endpoint_name="set_mcp_tool_approval",
        path_params={"service_id": service_id, "tool_name": tool_name},
        json=body,
    )


def start_oauth_authorization(
    client: APITestClient,
    service_id: str,
    *,
    body: dict[str, Any],
) -> httpx.Response:
    """``POST /mcp-services/{service_id}/oauth/authorize-url`` - begin OAuth (Viewer)."""
    return client.post(
        endpoint_name="start_oauth_authorization",
        path_params={"service_id": service_id},
        json=body,
    )


def get_oauth_status(client: APITestClient, service_id: str) -> httpx.Response:
    """``GET /mcp-services/{service_id}/oauth/status`` (Viewer)."""
    return client.get(
        endpoint_name="get_oauth_status",
        path_params={"service_id": service_id},
    )


def revoke_oauth_token(client: APITestClient, service_id: str) -> httpx.Response:
    """``DELETE /mcp-services/{service_id}/oauth/token`` (Viewer)."""
    return client.delete(
        endpoint_name="revoke_oauth_token",
        path_params={"service_id": service_id},
    )


__all__ = [
    "create_mcp_service",
    "delete_mcp_service",
    "get_mcp_service",
    "get_oauth_status",
    "list_mcp_service_resources",
    "list_mcp_service_tools",
    "list_mcp_services",
    "list_mcp_tool_approvals",
    "revoke_oauth_token",
    "set_mcp_tool_approval",
    "start_oauth_authorization",
    "test_mcp_service",
    "update_mcp_service",
]
