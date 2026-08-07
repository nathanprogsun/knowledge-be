"""Sync wrappers around the live-MCP HTTP transport endpoints.

The ``/sse``, ``/messages/``, and ``/mcp`` paths live outside the
JSON-RPC ``/api/v1/mcp-services`` router — they are mounted by the
live MCP connection manager — so they are exposed here rather than in
``mcp_api.py`` for clarity.
"""

from __future__ import annotations

from typing import Any

import httpx
from fastapi.testclient import TestClient

from tests.integration.web.api.util.api_client import APITestClient


def open_sse(
    client: APITestClient,
    *,
    workspace_id: str | int | None = None,
    knowledge_base_id: str | int | None = None,
    service_id: str | None = None,
) -> httpx.Response:
    """``GET /sse`` - open the server-sent-events stream.

    ``sse`` accepts query parameters for routing per workspace. Pass
    them as ``query_params`` on the underlying ``TestClient.get``
    directly when in doubt; this helper passes only the keys it knows
    about and lets the rest default.
    """
    params: dict[str, Any] = {}
    if workspace_id is not None:
        params["workspace_id"] = str(workspace_id)
    if knowledge_base_id is not None:
        params["knowledge_base_id"] = str(knowledge_base_id)
    if service_id is not None:
        params["service_id"] = service_id
    return client.client.get("/sse", params=params)


def post_mcp_message(
    test_client: TestClient,
    *,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """``POST /messages/`` - JSON-RPC transport mirror.

    Bypasses the APITestClient because this endpoint is registered
    outside the FastAPI app's normal ``url_path_for`` namespace.
    """
    return test_client.post("/messages/", json=payload, headers=headers)


def post_mcp_rpc(
    test_client: TestClient,
    *,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """``POST /mcp`` - JSON-RPC facade mirror.

    Same caveat as ``post_mcp_message``: the path is registered outside
    the standard ``url_path_for`` namespace.
    """
    return test_client.post("/mcp", json=payload, headers=headers)


__all__ = [
    "open_sse",
    "post_mcp_message",
    "post_mcp_rpc",
]
