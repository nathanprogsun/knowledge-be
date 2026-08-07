"""Sync wrappers around the ``vector-stores`` router endpoints.

Endpoint names match the handler function names in
``src/web/api/infra/vector_stores/router.py``.
"""

from __future__ import annotations

from typing import Any

import httpx

from tests.integration.web.api.util.api_client import APITestClient


def list_vector_store_types(client: APITestClient) -> httpx.Response:
    """``GET /vector-stores/types`` - registry metadata (Viewer)."""
    return client.get(endpoint_name="list_vector_store_types")


def test_vector_store_raw(client: APITestClient, *, body: dict[str, Any]) -> httpx.Response:
    """``POST /vector-stores/test`` - probe raw config (Admin)."""
    return client.post(endpoint_name="test_vector_store_raw", json=body)


def create_vector_store(client: APITestClient, *, body: dict[str, Any]) -> httpx.Response:
    """``POST /vector-stores`` - register a store (Admin)."""
    return client.post(endpoint_name="create_vector_store", json=body)


def list_vector_stores(client: APITestClient) -> httpx.Response:
    """``GET /vector-stores`` - list workspace + env stores (Viewer)."""
    return client.get(endpoint_name="list_vector_stores")


def get_vector_store(client: APITestClient, store_id: str) -> httpx.Response:
    """``GET /vector-stores/{store_id}`` - fetch one store (Viewer)."""
    return client.get(
        endpoint_name="get_vector_store",
        path_params={"store_id": store_id},
    )


def update_vector_store(
    client: APITestClient,
    store_id: str,
    *,
    body: dict[str, Any],
) -> httpx.Response:
    """``PUT /vector-stores/{store_id}`` - rename one store (Admin)."""
    return client.put(
        endpoint_name="update_vector_store",
        path_params={"store_id": store_id},
        json=body,
    )


def delete_vector_store(client: APITestClient, store_id: str) -> httpx.Response:
    """``DELETE /vector-stores/{store_id}`` - soft-delete one store (Admin)."""
    return client.delete(
        endpoint_name="delete_vector_store",
        path_params={"store_id": store_id},
    )


def test_vector_store_by_id(client: APITestClient, store_id: str) -> httpx.Response:
    """``POST /vector-stores/{store_id}/test`` - probe stored store (Admin)."""
    return client.post(
        endpoint_name="test_vector_store_by_id",
        path_params={"store_id": store_id},
    )


__all__ = [
    "create_vector_store",
    "delete_vector_store",
    "get_vector_store",
    "list_vector_store_types",
    "list_vector_stores",
    "test_vector_store_by_id",
    "test_vector_store_raw",
    "update_vector_store",
]
