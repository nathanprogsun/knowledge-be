"""Domain-facade over :class:`APITestClient`.

Each method maps one logical operation to a single HTTP request. The
facade lets test code read like a workflow (``create_tenant``,
``create_storage_backend``) instead of a sequence of URL and JSON
plumbing calls.

This file is a first cut. Only a handful of operations are wired up
here; later commits add the rest as the web-layer integration tests
migrate from raw ``httpx`` calls to this facade.
"""

from __future__ import annotations

from src.core.contracts.infra import (
    CreateStorageBackendRequest,
    StorageBackend,
)
from src.core.contracts.tenants import (
    CreateTenantRequest,
    Tenant,
    TenantList,
)

from tests.integration.web.api.util.api_client import APITestClient


class CommonAPIClient:
    """A small facade exposing the most common cross-domain operations."""

    def __init__(self, api: APITestClient) -> None:
        self.api = api

    # ── tenants ───────────────────────────────────────────────────────

    async def create_tenant(self, request: CreateTenantRequest) -> Tenant:
        return await self.api.post(
            endpoint_name="create_tenant",
            request_body=request,
            response_type=Tenant,
        )

    async def list_tenants(self) -> TenantList:
        return await self.api.get(
            endpoint_name="list_my_tenants",
            response_type=TenantList,
        )

    async def list_all_tenants(self) -> TenantList:
        return await self.api.get(
            endpoint_name="list_all_tenants",
            response_type=TenantList,
        )

    # ── storage backends ──────────────────────────────────────────────

    async def create_storage_backend(
        self, request: CreateStorageBackendRequest
    ) -> StorageBackend:
        return await self.api.post(
            endpoint_name="create_storage_backend",
            request_body=request,
            response_type=StorageBackend,
        )


__all__ = ["CommonAPIClient"]
