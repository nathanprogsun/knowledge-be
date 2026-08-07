"""Sync wrappers around the ``initialization`` router endpoints.

Endpoint names match the handler function names in
``src/web/api/infra/initialization/router.py`` so the APITestClient
resolves them through ``app.url_path_for(...)``.
"""

from __future__ import annotations

from typing import Any

import httpx

from tests.integration.web.api.util.api_client import APITestClient


def check_ollama_status(client: APITestClient) -> httpx.Response:
    """``GET /initialization/ollama/status`` - probe local Ollama (Viewer)."""
    return client.get(endpoint_name="check_ollama_status")


def list_ollama_models(client: APITestClient) -> httpx.Response:
    """``GET /initialization/ollama/models`` - list installed models (Viewer)."""
    return client.get(endpoint_name="list_ollama_models")


def check_ollama_models(client: APITestClient, *, body: dict[str, Any]) -> httpx.Response:
    """``POST /initialization/ollama/models/check`` - per-name presence (Admin)."""
    return client.post(endpoint_name="check_ollama_models", json=body)


def download_ollama_model(client: APITestClient, *, body: dict[str, Any]) -> httpx.Response:
    """``POST /initialization/ollama/models/download`` - start async pull (Admin)."""
    return client.post(endpoint_name="download_ollama_model", json=body)


def get_download_progress(client: APITestClient, task_id: str) -> httpx.Response:
    """``GET /initialization/ollama/download/progress/{task_id}`` (Viewer)."""
    return client.get(
        endpoint_name="get_download_progress",
        path_params={"task_id": task_id},
    )


def list_download_tasks(client: APITestClient) -> httpx.Response:
    """``GET /initialization/ollama/download/tasks`` - every pull task (Viewer)."""
    return client.get(endpoint_name="list_download_tasks")


def check_remote_model(client: APITestClient, *, body: dict[str, Any]) -> httpx.Response:
    """``POST /initialization/remote/check`` - probe remote chat model (Admin)."""
    return client.post(endpoint_name="check_remote_model", json=body)


def test_embedding_model(client: APITestClient, *, body: dict[str, Any]) -> httpx.Response:
    """``POST /initialization/embedding/test`` - probe embedding model (Admin)."""
    return client.post(endpoint_name="test_embedding_model", json=body)


def check_rerank_model(client: APITestClient, *, body: dict[str, Any]) -> httpx.Response:
    """``POST /initialization/rerank/check`` - probe rerank endpoint (Admin)."""
    return client.post(endpoint_name="check_rerank_model", json=body)


def check_asr_model(client: APITestClient, *, body: dict[str, Any]) -> httpx.Response:
    """``POST /initialization/asr/check`` - probe ASR endpoint (Admin)."""
    return client.post(endpoint_name="check_asr_model", json=body)


__all__ = [
    "check_asr_model",
    "check_ollama_models",
    "check_ollama_status",
    "check_remote_model",
    "check_rerank_model",
    "download_ollama_model",
    "get_download_progress",
    "list_download_tasks",
    "list_ollama_models",
    "test_embedding_model",
]
