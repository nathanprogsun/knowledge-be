"""Sync wrappers around the ``datasource`` router endpoints.

Each function takes the ``APITestClient`` plus the path / body / query
arguments the corresponding FastAPI handler declares. Endpoint names
match the handler function names in ``src/web/api/infra/datasources/router.py``
so ``app.url_path_for(...)`` resolves the URL.
"""

from __future__ import annotations

from typing import Any

import httpx

from tests.integration.web.api.util.api_client import APITestClient


def list_connector_types(client: APITestClient) -> httpx.Response:
    """``GET /datasource/types`` - public connector catalogue (Viewer)."""
    return client.get(endpoint_name="list_connector_types")


def validate_credentials(client: APITestClient, *, body: dict[str, Any]) -> httpx.Response:
    """``POST /datasource/validate-credentials`` - test raw credentials (Admin)."""
    return client.post(endpoint_name="validate_credentials", json=body)


def create_datasource(client: APITestClient, *, body: dict[str, Any]) -> httpx.Response:
    """``POST /datasource`` - create one data source (Admin)."""
    return client.post(endpoint_name="create_datasource", json=body)


def list_datasources(
    client: APITestClient,
    *,
    kb_id: str = "",
) -> httpx.Response:
    """``GET /datasource`` - list a knowledge base's data sources (Viewer)."""
    params: dict[str, Any] = {}
    if kb_id:
        params["kb_id"] = kb_id
    return client.get(endpoint_name="list_datasources", params=params)


def get_datasource(client: APITestClient, datasource_id: str) -> httpx.Response:
    """``GET /datasource/{id}`` - fetch one data source (Viewer)."""
    return client.get(
        endpoint_name="get_datasource",
        path_params={"id": datasource_id},
    )


def update_datasource(
    client: APITestClient,
    datasource_id: str,
    *,
    body: dict[str, Any],
) -> httpx.Response:
    """``PUT /datasource/{id}`` - update mutable fields (Admin)."""
    return client.put(
        endpoint_name="update_datasource",
        path_params={"id": datasource_id},
        json=body,
    )


def delete_datasource(client: APITestClient, datasource_id: str) -> httpx.Response:
    """``DELETE /datasource/{id}`` - soft-delete one data source (Admin)."""
    return client.delete(
        endpoint_name="delete_datasource",
        path_params={"id": datasource_id},
    )


def validate_connection(client: APITestClient, datasource_id: str) -> httpx.Response:
    """``POST /datasource/{id}/validate`` - test stored connection (Admin)."""
    return client.post(
        endpoint_name="validate_connection",
        path_params={"id": datasource_id},
    )


def list_available_resources(
    client: APITestClient,
    datasource_id: str,
    *,
    parent_id: str = "",
) -> httpx.Response:
    """``GET /datasource/{id}/resources`` - browse external resources (Admin)."""
    params: dict[str, Any] = {}
    if parent_id:
        params["parent_id"] = parent_id
    return client.get(
        endpoint_name="list_available_resources",
        path_params={"id": datasource_id},
        params=params,
    )


def resolve_resource_ancestors(
    client: APITestClient,
    datasource_id: str,
    *,
    body: dict[str, Any],
) -> httpx.Response:
    """``POST /datasource/{id}/resource-ancestors`` - resolve picker ancestors (Admin)."""
    return client.post(
        endpoint_name="resolve_resource_ancestors",
        path_params={"id": datasource_id},
        json=body,
    )


def manual_sync(client: APITestClient, datasource_id: str) -> httpx.Response:
    """``POST /datasource/{id}/sync`` - trigger immediate sync (Admin)."""
    return client.post(
        endpoint_name="manual_sync",
        path_params={"id": datasource_id},
    )


def pause_datasource(client: APITestClient, datasource_id: str) -> httpx.Response:
    """``POST /datasource/{id}/pause`` - pause one data source (Admin)."""
    return client.post(
        endpoint_name="pause_datasource",
        path_params={"id": datasource_id},
    )


def resume_datasource(client: APITestClient, datasource_id: str) -> httpx.Response:
    """``POST /datasource/{id}/resume`` - resume paused source (Admin)."""
    return client.post(
        endpoint_name="resume_datasource",
        path_params={"id": datasource_id},
    )


def list_sync_logs(client: APITestClient, datasource_id: str) -> httpx.Response:
    """``GET /datasource/{id}/logs`` - list this source's sync history (Viewer)."""
    return client.get(
        endpoint_name="list_sync_logs",
        path_params={"id": datasource_id},
    )


def get_sync_log(client: APITestClient, log_id: str) -> httpx.Response:
    """``GET /datasource/logs/{log_id}`` - fetch one sync-log entry (Viewer)."""
    return client.get(
        endpoint_name="get_sync_log",
        path_params={"log_id": log_id},
    )


__all__ = [
    "create_datasource",
    "delete_datasource",
    "get_datasource",
    "get_sync_log",
    "list_available_resources",
    "list_connector_types",
    "list_datasources",
    "list_sync_logs",
    "manual_sync",
    "pause_datasource",
    "resolve_resource_ancestors",
    "resume_datasource",
    "update_datasource",
    "validate_connection",
    "validate_credentials",
]
