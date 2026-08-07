"""Sync wrappers around the ``storage-backends`` router endpoints.

Endpoint names match the handler function names in
``src/web/api/infra/storage_backends/router.py``.
"""

from __future__ import annotations

from typing import Any

import httpx

from tests.integration.web.api.util.api_client import APITestClient


def list_storage_provider_types(client: APITestClient) -> httpx.Response:
    """``GET /storage-backends/types`` - provider catalogue (Viewer)."""
    return client.get(endpoint_name="list_storage_provider_types")


def test_storage_backend_config(client: APITestClient, *, body: dict[str, Any]) -> httpx.Response:
    """``POST /storage-backends/test`` - probe raw config (Admin)."""
    return client.post(endpoint_name="test_storage_backend_config", json=body)


def create_storage_backend(client: APITestClient, *, body: dict[str, Any]) -> httpx.Response:
    """``POST /storage-backends`` - register a backend (Admin)."""
    return client.post(endpoint_name="create_storage_backend", json=body)


def list_storage_backends(client: APITestClient) -> httpx.Response:
    """``GET /storage-backends`` - list workspace backends (Viewer)."""
    return client.get(endpoint_name="list_storage_backends")


def get_storage_backend(client: APITestClient, backend_id: str) -> httpx.Response:
    """``GET /storage-backends/{id}`` - fetch one backend (Viewer)."""
    return client.get(
        endpoint_name="get_storage_backend",
        path_params={"id": backend_id},
    )


def update_storage_backend(
    client: APITestClient,
    backend_id: str,
    *,
    body: dict[str, Any],
) -> httpx.Response:
    """``PUT /storage-backends/{id}`` - update mutable fields (Admin)."""
    return client.put(
        endpoint_name="update_storage_backend",
        path_params={"id": backend_id},
        json=body,
    )


def delete_storage_backend(client: APITestClient, backend_id: str) -> httpx.Response:
    """``DELETE /storage-backends/{id}`` - soft-delete one backend (Admin)."""
    return client.delete(
        endpoint_name="delete_storage_backend",
        path_params={"id": backend_id},
    )


def test_storage_backend_by_id(client: APITestClient, backend_id: str) -> httpx.Response:
    """``POST /storage-backends/{id}/test`` - probe stored backend (Admin)."""
    return client.post(
        endpoint_name="test_storage_backend_by_id",
        path_params={"id": backend_id},
    )


def set_default_storage_backend(client: APITestClient, backend_id: str) -> httpx.Response:
    """``PUT /storage-backends/{id}/default`` - mark default backend (Admin)."""
    return client.put(
        endpoint_name="set_default_storage_backend",
        path_params={"id": backend_id},
    )


__all__ = [
    "create_storage_backend",
    "delete_storage_backend",
    "get_storage_backend",
    "list_storage_backends",
    "list_storage_provider_types",
    "set_default_storage_backend",
    "test_storage_backend_by_id",
    "test_storage_backend_config",
    "update_storage_backend",
]
