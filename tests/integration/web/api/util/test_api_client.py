"""Smoke tests for the :class:`APITestClient` wrapper.

The tests build a tiny FastAPI app inline (an ``/echo`` endpoint that
reflects the ``x-knowledge-user-id`` and ``x-knowledge-tenant-id``
headers back as a typed ``Echo`` body) and exercise both ``get`` and
``post`` against it. The point is to prove the wrapper:

- Resolves endpoint names through ``app.url_path_for``.
- Forwards ``default_headers`` on every request.
- Parses the response into a Pydantic model when ``response_type`` is
  supplied.
- Returns raw JSON when no ``response_type`` is supplied.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from fastapi import FastAPI, Header
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel

from tests.integration.web.api.util.api_client import APITestClient

app = FastAPI()


class Echo(BaseModel):
    user_id: UUID
    tenant_id: UUID


@app.get("/echo", response_model=Echo)
async def echo(
    x_knowledge_user_id: str = Header(...),
    x_knowledge_tenant_id: str = Header(...),
) -> Echo:
    return Echo(
        user_id=UUID(x_knowledge_user_id),
        tenant_id=UUID(x_knowledge_tenant_id),
    )


class EchoBody(BaseModel):
    note: str


@app.post("/echo", response_model=Echo)
async def echo_post(
    body: EchoBody,
    x_knowledge_user_id: str = Header(...),
    x_knowledge_tenant_id: str = Header(...),
) -> Echo:
    return Echo(
        user_id=UUID(x_knowledge_user_id),
        tenant_id=UUID(x_knowledge_tenant_id),
    )


def _client() -> tuple[APITestClient, AsyncClient]:
    user_id = uuid4()
    tenant_id = uuid4()
    raw = AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    )
    api = APITestClient(
        client=raw,
        default_user_id=user_id,
        default_organization_id=tenant_id,
        default_headers={
            "x-knowledge-user-id": str(user_id),
            "x-knowledge-tenant-id": str(tenant_id),
        },
    )
    return api, raw


async def test_get_via_endpoint_name() -> None:
    api, raw = _client()
    try:
        result = await api.get(endpoint_name="echo", response_type=Echo)
        assert result.user_id == UUID(api.default_headers["x-knowledge-user-id"])
        assert result.tenant_id == UUID(api.default_headers["x-knowledge-tenant-id"])
    finally:
        await raw.aclose()


async def test_get_returns_raw_dict_without_response_type() -> None:
    api, raw = _client()
    try:
        result = await api.get(endpoint_name="echo")
        assert isinstance(result, dict)
        assert "user_id" in result
    finally:
        await raw.aclose()


async def test_post_via_endpoint_name() -> None:
    api, raw = _client()
    try:
        result = await api.post(
            endpoint_name="echo_post",
            request_body=EchoBody(note="hi"),
            response_type=Echo,
        )
        assert result.user_id == UUID(api.default_headers["x-knowledge-user-id"])
    finally:
        await raw.aclose()
