"""Sync HTTP client wrapper used by the web-layer integration tests.

``APITestClient`` is a thin wrapper around
:class:`fastapi.testclient.TestClient` that:

- Resolves endpoint names (``"create_tenant"``) to URLs via the wrapped
  FastAPI app's URL router (``app.url_path_for``).
- Carries a default user / organization id pair as metadata so tests
  can reference the authenticated principal without re-parsing headers.
- Raises :class:`APITestError` on any 4xx / 5xx response.

The wrapped ``TestClient`` owns the ``x-knowledge-*`` header trio on its
own ``headers`` dict; per-request header overrides are merged on top by
passing the ``headers`` keyword arg to each method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi.testclient import TestClient
from httpx import Response


class APITestError(Exception):
    """Raised when a sync request through APITestClient returns a 4xx/5xx."""


@dataclass
class APITestClient:
    """Typed sync wrapper around :class:`fastapi.testclient.TestClient`.

    The wrapped ``TestClient`` carries the ``x-knowledge-*`` header trio
    on its own ``headers`` dict; ``default_user_id`` and
    ``default_organization_id`` are metadata fields so tests can reference
    the authenticated principal.
    """

    client: TestClient
    default_user_id: str | None = None
    default_organization_id: int | None = None

    def _url(self, endpoint_name: str, path_params: dict[str, Any] | None) -> str:
        """Resolve an endpoint name to a URL through the FastAPI app router."""
        return self.client.app.url_path_for(endpoint_name, **(path_params or {}))

    def get(
        self,
        endpoint_name: str,
        *,
        path_params: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Response:
        url = self._url(endpoint_name, path_params)
        r = self.client.get(url, params=params, headers=headers)
        if r.status_code >= 400:
            raise APITestError(f"GET {url} -> {r.status_code}: {r.text}")
        return r

    def post(
        self,
        endpoint_name: str,
        *,
        json: dict[str, Any] | None = None,
        path_params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Response:
        url = self._url(endpoint_name, path_params)
        r = self.client.post(url, json=json, headers=headers)
        if r.status_code >= 400:
            raise APITestError(f"POST {url} -> {r.status_code}: {r.text}")
        return r

    def put(
        self,
        endpoint_name: str,
        *,
        json: dict[str, Any] | None = None,
        path_params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Response:
        url = self._url(endpoint_name, path_params)
        r = self.client.put(url, json=json, headers=headers)
        if r.status_code >= 400:
            raise APITestError(f"PUT {url} -> {r.status_code}: {r.text}")
        return r

    def patch(
        self,
        endpoint_name: str,
        *,
        json: dict[str, Any] | None = None,
        path_params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Response:
        url = self._url(endpoint_name, path_params)
        r = self.client.patch(url, json=json, headers=headers)
        if r.status_code >= 400:
            raise APITestError(f"PATCH {url} -> {r.status_code}: {r.text}")
        return r

    def delete(
        self,
        endpoint_name: str,
        *,
        path_params: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Response:
        url = self._url(endpoint_name, path_params)
        r = self.client.delete(url, params=params, headers=headers)
        if r.status_code >= 400:
            raise APITestError(f"DELETE {url} -> {r.status_code}: {r.text}")
        return r


__all__ = ["APITestClient", "APITestError"]
