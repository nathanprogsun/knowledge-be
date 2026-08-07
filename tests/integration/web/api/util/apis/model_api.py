"""Sync wrappers around the ``models`` router endpoints.

Endpoint names match the handler function names in
``src/web/api/infra/models/router.py``.
"""

from __future__ import annotations

from typing import Any

import httpx

from tests.integration.web.api.util.api_client import APITestClient


def list_model_providers(
    client: APITestClient,
    *,
    model_type: str | None = None,
) -> httpx.Response:
    """``GET /models/providers`` - static provider catalog (Viewer)."""
    params: dict[str, Any] = {}
    if model_type is not None:
        params["model_type"] = model_type
    return client.get(endpoint_name="list_model_providers", params=params)


def create_model(client: APITestClient, *, body: dict[str, Any]) -> httpx.Response:
    """``POST /models`` - register a model (Admin)."""
    return client.post(endpoint_name="create_model", json=body)


def list_models(
    client: APITestClient,
    *,
    type: str | None = None,
    source: str | None = None,
    include_builtin: bool = True,
) -> httpx.Response:
    """``GET /models`` - list workspace models (Viewer)."""
    params: dict[str, Any] = {"include_builtin": str(include_builtin).lower()}
    if type is not None:
        params["type"] = type
    if source is not None:
        params["source"] = source
    return client.get(endpoint_name="list_models", params=params)


def get_model(client: APITestClient, model_id: str) -> httpx.Response:
    """``GET /models/{model_id}`` - fetch one model (Viewer)."""
    return client.get(
        endpoint_name="get_model",
        path_params={"model_id": model_id},
    )


def update_model(
    client: APITestClient,
    model_id: str,
    *,
    body: dict[str, Any],
) -> httpx.Response:
    """``PUT /models/{model_id}`` - update mutable fields (Admin)."""
    return client.put(
        endpoint_name="update_model",
        path_params={"model_id": model_id},
        json=body,
    )


def delete_model(client: APITestClient, model_id: str) -> httpx.Response:
    """``DELETE /models/{model_id}`` - delete a model (Admin)."""
    return client.delete(
        endpoint_name="delete_model",
        path_params={"model_id": model_id},
    )


def debug_model(
    client: APITestClient,
    model_id: str,
    *,
    input: str = "",
    options: str = "",
    documents: str = "",
) -> httpx.Response:
    """``POST /models/{model_id}/debug`` - end-to-end probe (Admin).

    The handler accepts ``multipart/form-data`` (Go parity), so this
    helper uses the underlying ``TestClient`` form-encoded path. The
    APITestClient is only used to resolve the URL.
    """
    url = client.client.app.url_path_for("debug_model", model_id=model_id)
    return client.client.post(
        url,
        data={"input": input, "options": options, "documents": documents},
    )


__all__ = [
    "create_model",
    "debug_model",
    "delete_model",
    "get_model",
    "list_model_providers",
    "list_models",
    "update_model",
]
