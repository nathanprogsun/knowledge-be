"""Tests for the sync APITestClient wrapper.

The tests build a tiny FastAPI app inline (endpoints that reflect the
``X-User-Id/X-Tenant-ID/X-Roles`` headers and path params back as typed bodies) and
exercise all five HTTP methods through the sync APITestClient wrapper.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

import pytest
from fastapi import FastAPI, Header
from fastapi.testclient import TestClient
from pydantic import BaseModel

from tests.integration.web.api.util.api_client import APITestClient, APITestError


class EchoBody(BaseModel):
    note: str


class Echo(BaseModel):
    user_id: str
    tenant_id: str
    note: str | None = None


def _build_app() -> FastAPI:
    app = FastAPI()

    @app.get("/echo", response_model=Echo, name="echo")
    def echo(
        x_user_id: str = Header(...),
        x_tenant_id: str = Header(...),
    ) -> Echo:
        return Echo(user_id=x_user_id, tenant_id=x_tenant_id)

    @app.post("/echo", response_model=Echo, name="echo_post")
    def echo_post(
        body: EchoBody,
        x_user_id: str = Header(...),
        x_tenant_id: str = Header(...),
    ) -> Echo:
        return Echo(
            user_id=x_user_id,
            tenant_id=x_tenant_id,
            note=body.note,
        )

    @app.put("/items/{item_id}", response_model=Echo, name="update_item")
    def update_item(
        item_id: int,
        body: EchoBody,
        x_user_id: str = Header(...),
        x_tenant_id: str = Header(...),
    ) -> Echo:
        return Echo(
            user_id=x_user_id,
            tenant_id=x_tenant_id,
            note=f"{body.note}:{item_id}",
        )

    @app.patch("/items/{item_id}", response_model=Echo, name="patch_item")
    def patch_item(
        item_id: int,
        body: EchoBody,
        x_user_id: str = Header(...),
        x_tenant_id: str = Header(...),
    ) -> Echo:
        return Echo(
            user_id=x_user_id,
            tenant_id=x_tenant_id,
            note=f"{body.note}:{item_id}",
        )

    @app.delete("/items/{item_id}", name="delete_item")
    def delete_item(
        item_id: int,
        x_user_id: str = Header(...),
        x_tenant_id: str = Header(...),
    ) -> dict[str, Any]:
        return {"deleted": item_id, "user_id": x_user_id}

    @app.get("/search", name="search")
    def search(
        q: str,
        x_user_id: str = Header(...),
        x_tenant_id: str = Header(...),
    ) -> dict[str, Any]:
        return {"q": q, "user_id": x_user_id}

    @app.get("/fail", name="fail", status_code=400)
    def fail() -> dict[str, str]:
        return {"detail": "bad request"}

    return app


_DEFAULT_USER_ID = str(uuid4())
_DEFAULT_TENANT_ID = 42


@contextmanager
def _make_api(
    *,
    user_id: str = _DEFAULT_USER_ID,
    tenant_id: int = _DEFAULT_TENANT_ID,
) -> Iterator[APITestClient]:
    """Yield an APITestClient wrapping a TestClient with X-User-Id/X-Tenant-ID/X-Roles headers."""
    app = _build_app()
    with TestClient(app=app) as tc:
        tc.headers.update(
            {
                "X-User-Id": user_id,
                "X-Tenant-ID": str(tenant_id),
            }
        )
        yield APITestClient(
            client=tc,
            default_user_id=user_id,
            default_organization_id=tenant_id,
        )


def test_get_resolves_endpoint_name_and_returns_response() -> None:
    with _make_api() as api:
        response = api.get("echo")

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == api.default_user_id
        assert data["tenant_id"] == str(api.default_organization_id)


def test_get_forwards_query_params() -> None:
    with _make_api() as api:
        response = api.get("search", params={"q": "hello"})

        assert response.status_code == 200
        assert response.json()["q"] == "hello"


def test_post_sends_json_body() -> None:
    with _make_api() as api:
        response = api.post("echo_post", json={"note": "hello"})

        assert response.status_code == 200
        assert response.json()["note"] == "hello"


def test_put_resolves_path_params_and_sends_body() -> None:
    with _make_api() as api:
        response = api.put("update_item", path_params={"item_id": 7}, json={"note": "x"})

        assert response.status_code == 200
        assert response.json()["note"] == "x:7"


def test_patch_resolves_path_params_and_sends_body() -> None:
    with _make_api() as api:
        response = api.patch("patch_item", path_params={"item_id": 9}, json={"note": "y"})

        assert response.status_code == 200
        assert response.json()["note"] == "y:9"


def test_delete_resolves_path_params() -> None:
    with _make_api() as api:
        response = api.delete("delete_item", path_params={"item_id": 3})

        assert response.status_code == 200
        body = response.json()
        assert body["deleted"] == 3
        assert body["user_id"] == api.default_user_id


def test_error_response_raises_api_test_error() -> None:
    with _make_api() as api, pytest.raises(APITestError):
        api.get("fail")


def test_per_request_headers_override_client_defaults() -> None:
    with _make_api() as api:
        override_user = str(uuid4())
        response = api.get("echo", headers={"X-User-Id": override_user})

        assert response.status_code == 200
        assert response.json()["user_id"] == override_user


def test_default_metadata_fields_are_stored() -> None:
    user_id = str(uuid4())
    org_id = 99
    with _make_api(user_id=user_id, tenant_id=org_id) as api:
        assert api.default_user_id == user_id
        assert api.default_organization_id == org_id
