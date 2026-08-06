"""HTTP client wrapper used by the web-layer integration tests.

``APITestClient`` is a thin wrapper around :class:`httpx.AsyncClient`
that:

- Resolves endpoint names (``"create_tenant"``) to URLs via the
  wrapped ``FastAPI`` app's URL router (``app.url_path_for``).
- Carries a default user / workspace id pair and the matching
  ``x-knowledge-*-id`` headers so every test starts with a valid
  request context without per-test boilerplate.
- Validates responses against an optional Pydantic model and raises
  :class:`APITestError` on any 4xx / 5xx response.

This is a first cut. The point is to have a working scaffold; the
full web-test migration in a later commit may add pagination helpers,
header overrides, and per-request auth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeVar
from uuid import UUID

import httpx
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class APITestError(Exception):
    """Raised when a request through :class:`APITestClient` fails."""


@dataclass
class APITestClient:
    """Typed wrapper around :class:`httpx.AsyncClient`."""

    client: httpx.AsyncClient
    default_user_id: UUID
    default_organization_id: UUID
    default_headers: dict[str, str] = field(default_factory=dict)

    def _url(self, endpoint_name: str, path_params: dict | None) -> str:
        """Resolve an endpoint name to a URL through the FastAPI app router."""
        transport = self.client._transport  # type: ignore[attr-defined]
        app = getattr(transport, "app", None)
        if app is None:
            raise APITestError(
                "APITestClient requires an ASGITransport with a .app attribute"
            )
        return app.url_path_for(endpoint_name, **(path_params or {}))

    @staticmethod
    def _parse(r: httpx.Response, response_type: type[T] | None) -> Any:
        data = r.json()
        if response_type is None:
            return data
        return response_type.model_validate(data)

    async def get(
        self,
        *,
        endpoint_name: str,
        path_params: dict | None = None,
        params: dict | None = None,
        response_type: type[T] | None = None,
    ) -> Any:
        url = self._url(endpoint_name, path_params)
        r = await self.client.get(url, params=params, headers=self.default_headers)
        if r.status_code >= 400:
            raise APITestError(f"GET {url} -> {r.status_code}: {r.text}")
        return self._parse(r, response_type)

    async def post(
        self,
        *,
        endpoint_name: str,
        path_params: dict | None = None,
        request_body: BaseModel | dict | None = None,
        response_type: type[T] | None = None,
    ) -> Any:
        url = self._url(endpoint_name, path_params)
        if isinstance(request_body, BaseModel):
            json_body: Any = request_body.model_dump(mode="json")
        elif request_body is None:
            json_body = None
        else:
            json_body = request_body
        r = await self.client.post(
            url,
            json=json_body,
            headers=self.default_headers,
        )
        if r.status_code >= 400:
            raise APITestError(f"POST {url} -> {r.status_code}: {r.text}")
        return self._parse(r, response_type)

    async def put(
        self,
        *,
        endpoint_name: str,
        path_params: dict | None = None,
        request_body: BaseModel | dict | None = None,
        response_type: type[T] | None = None,
    ) -> Any:
        url = self._url(endpoint_name, path_params)
        if isinstance(request_body, BaseModel):
            json_body: Any = request_body.model_dump(mode="json")
        elif request_body is None:
            json_body = None
        else:
            json_body = request_body
        r = await self.client.put(
            url,
            json=json_body,
            headers=self.default_headers,
        )
        if r.status_code >= 400:
            raise APITestError(f"PUT {url} -> {r.status_code}: {r.text}")
        return self._parse(r, response_type)

    async def patch(
        self,
        *,
        endpoint_name: str,
        path_params: dict | None = None,
        request_body: BaseModel | dict | None = None,
        response_type: type[T] | None = None,
    ) -> Any:
        url = self._url(endpoint_name, path_params)
        if isinstance(request_body, BaseModel):
            json_body: Any = request_body.model_dump(mode="json")
        elif request_body is None:
            json_body = None
        else:
            json_body = request_body
        r = await self.client.patch(
            url,
            json=json_body,
            headers=self.default_headers,
        )
        if r.status_code >= 400:
            raise APITestError(f"PATCH {url} -> {r.status_code}: {r.text}")
        return self._parse(r, response_type)

    async def delete(
        self,
        *,
        endpoint_name: str,
        path_params: dict | None = None,
        response_type: type[T] | None = None,
    ) -> Any:
        url = self._url(endpoint_name, path_params)
        r = await self.client.delete(url, headers=self.default_headers)
        if r.status_code >= 400:
            raise APITestError(f"DELETE {url} -> {r.status_code}: {r.text}")
        return self._parse(r, response_type)


__all__ = ["APITestClient", "APITestError"]
