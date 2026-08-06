"""Web-layer tests for the tenant router.

Exercises the router over HTTP via ``httpx.AsyncClient`` with
``get_tenant_service`` overridden to use a real ``TenantService`` backed
by the shared in-memory fake repository, so the full web -> service
path runs without a database.

Uses the shared ``web_app`` fixture (header-based auth) and applies
the service dep override on it; the real ``require_auth`` dep resolves
the principal via the ``x-knowledge-*`` header trio.

Note on tenant-id duality: the URL path uses literal ``/tenants/{id}``
paths (the fake repos key on the URL id, e.g. ``7``), while the
``x-knowledge-tenant-id`` header carries ``1`` (the integration seed).
The header-based RBAC gate is a no-op shim, so the two coexist
without conflict — the path id drives the repo lookup and the header
id is only used to populate ``request.state``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from src.core.tenants.member_service import TenantMemberService
from src.core.tenants.service import TenantService
from src.db.models.tenants.tenants import DEFAULT_STORAGE_QUOTA_BYTES, Tenant
from src.web.deps import get_tenant_member_service, get_tenant_service
from tests.integration.web.conftest import web_app, web_authed_client  # noqa: F401
from tests.unit.fakes.tenant_members import FakeTenantMemberRepository
from tests.unit.fakes.tenants import FakeTenantRepository

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def repo() -> FakeTenantRepository:
    return FakeTenantRepository()


@pytest.fixture(autouse=True)
def _override_services(
    web_app: FastAPI,  # noqa: ARG001 - resolved from the parent conftest
    repo: FakeTenantRepository,
) -> FastAPI:
    """Override the tenant service deps on the shared web app (autouse)."""
    web_app.dependency_overrides[get_tenant_service] = lambda: TenantService(
        tenants_repo=repo,  # type: ignore[arg-type]
    )
    web_app.dependency_overrides[get_tenant_member_service] = (
        lambda: TenantMemberService(
            members_repo=FakeTenantMemberRepository(),  # type: ignore[arg-type]
        )
    )
    return web_app


async def _seed(
    repo: FakeTenantRepository,
    *,
    name: str = "acme",
    description: str | None = "acme workspace",
    created_at: datetime = _NOW,
    **columns: object,
) -> Tenant:
    return await repo.insert(
        Tenant.model_validate(
            {
                "name": name,
                "description": description,
                "created_at": created_at,
                "updated_at": created_at,
                **columns,
            }
        )
    )


# ── POST /tenants ───────────────────────────────────────────────────


async def test_create_tenant_returns_201_with_envelope(
    web_authed_client: AsyncClient,
    repo: FakeTenantRepository,
) -> None:
    resp = await web_authed_client.post(
        "/tenants", json={"name": "acme", "description": "the workspace"}
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["name"] == "acme"
    assert body["data"]["status"] == "active"
    assert body["data"]["id"] in repo.rows


async def test_create_tenant_applies_default_quota(web_authed_client: AsyncClient) -> None:
    resp = await web_authed_client.post("/tenants", json={"name": "acme"})

    assert resp.json()["data"]["storage_quota"] == DEFAULT_STORAGE_QUOTA_BYTES


async def test_create_tenant_stores_retriever_engines(
    web_authed_client: AsyncClient,
    repo: FakeTenantRepository,
) -> None:
    resp = await web_authed_client.post(
        "/tenants",
        json={
            "name": "acme",
            "retriever_engines": {
                "engines": [
                    {"retriever_type": "keywords", "retriever_engine_type": "postgres"},
                ]
            },
        },
    )

    engines = resp.json()["data"]["retriever_engines"]["engines"]
    assert engines == [{"retriever_type": "keywords", "retriever_engine_type": "postgres"}]


async def test_create_tenant_rejects_blank_name(web_authed_client: AsyncClient) -> None:
    resp = await web_authed_client.post("/tenants", json={"name": "   "})

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "tenant.name_required"


async def test_create_tenant_requires_a_name(web_authed_client: AsyncClient) -> None:
    resp = await web_authed_client.post("/tenants", json={})

    assert resp.status_code == 422


# ── GET /tenants/{id} ───────────────────────────────────────────────


async def test_get_tenant_returns_envelope(
    web_authed_client: AsyncClient,
    repo: FakeTenantRepository,
) -> None:
    stored = await _seed(repo)

    resp = await web_authed_client.get(f"/tenants/{stored.id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["id"] == stored.id
    assert body["data"]["description"] == "acme workspace"


async def test_get_tenant_missing_returns_404(web_authed_client: AsyncClient) -> None:
    resp = await web_authed_client.get("/tenants/4242")

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "tenant.not_found"


async def test_get_tenant_rejects_non_numeric_id(web_authed_client: AsyncClient) -> None:
    resp = await web_authed_client.get("/tenants/not-a-number")

    assert resp.status_code == 422


async def test_get_tenant_redacts_secret_config_blobs(
    web_authed_client: AsyncClient,
    repo: FakeTenantRepository,
) -> None:
    stored = await _seed(
        repo,
        credentials={"acme_cloud": {"app_id": "a", "app_secret": "s3cret"}},
        web_search_config={"api_key": "s3cret"},
        parser_engine_config={"mineru_api_key": "s3cret"},
        storage_engine_config={"minio": {"secret_access_key": "s3cret"}},
    )

    data = (await web_authed_client.get(f"/tenants/{stored.id}")).json()["data"]

    assert data["credentials"] is None
    assert data["web_search_config"] is None
    assert data["parser_engine_config"] is None
    assert data["storage_engine_config"] is None


async def test_get_tenant_keeps_non_secret_config_blobs(
    web_authed_client: AsyncClient,
    repo: FakeTenantRepository,
) -> None:
    stored = await _seed(
        repo,
        context_config={"max_tokens": 100},
        retrieval_config={"rerank_top_k": 5},
    )

    data = (await web_authed_client.get(f"/tenants/{stored.id}")).json()["data"]

    assert data["context_config"] == {"max_tokens": 100}
    assert data["retrieval_config"] == {"rerank_top_k": 5}


# ── GET /tenants/all ────────────────────────────────────────────────


async def test_list_all_tenants_returns_items_newest_first(
    web_authed_client: AsyncClient,
    repo: FakeTenantRepository,
) -> None:
    older = await _seed(repo, name="older", created_at=_NOW)
    newer = await _seed(repo, name="newer", created_at=_NOW + timedelta(days=1))

    body = (await web_authed_client.get("/tenants/all")).json()

    assert [item["id"] for item in body["data"]["items"]] == [newer.id, older.id]
    assert body["data"]["total"] is None


async def test_list_all_tenants_excludes_deleted(
    web_authed_client: AsyncClient,
    repo: FakeTenantRepository,
) -> None:
    live = await _seed(repo, name="live")
    gone = await _seed(repo, name="gone")
    await web_authed_client.delete(f"/tenants/{gone.id}")

    body = (await web_authed_client.get("/tenants/all")).json()

    assert [item["id"] for item in body["data"]["items"]] == [live.id]


# ── GET /tenants/search ─────────────────────────────────────────────


async def test_search_returns_page_with_total(
    web_authed_client: AsyncClient,
    repo: FakeTenantRepository,
) -> None:
    for index in range(5):
        await _seed(repo, name=f"t{index}", created_at=_NOW + timedelta(hours=index))

    body = (
        await web_authed_client.get(
            "/tenants/search", params={"page": 2, "page_size": 2}
        )
    ).json()

    assert body["data"]["total"] == 5
    assert body["data"]["page"] == 2
    assert body["data"]["page_size"] == 2
    assert [item["name"] for item in body["data"]["items"]] == ["t2", "t1"]


async def test_search_filters_by_keyword(
    web_authed_client: AsyncClient,
    repo: FakeTenantRepository,
) -> None:
    match = await _seed(repo, name="alpha", description=None)
    await _seed(repo, name="beta", description=None)

    body = (
        await web_authed_client.get("/tenants/search", params={"keyword": "alpha"})
    ).json()

    assert [item["id"] for item in body["data"]["items"]] == [match.id]


async def test_search_filters_by_tenant_id(
    web_authed_client: AsyncClient,
    repo: FakeTenantRepository,
) -> None:
    wanted = await _seed(repo, name="wanted", description=None)
    await _seed(repo, name="other", description=None)

    body = (
        await web_authed_client.get("/tenants/search", params={"tenant_id": wanted.id})
    ).json()

    assert [item["id"] for item in body["data"]["items"]] == [wanted.id]


@pytest.mark.parametrize(
    ("page", "page_size", "expected_page", "expected_page_size"),
    [
        (0, 10, 1, 10),
        (-3, 10, 1, 10),
        (1, 0, 1, 20),
        (1, 5000, 1, 100),
    ],
)
async def test_search_clamps_paging_params(
    web_authed_client: AsyncClient,
    page: int,
    page_size: int,
    expected_page: int,
    expected_page_size: int,
) -> None:
    body = (
        await web_authed_client.get(
            "/tenants/search", params={"page": page, "page_size": page_size}
        )
    ).json()

    assert body["data"]["page"] == expected_page
    assert body["data"]["page_size"] == expected_page_size


# ── PUT /tenants/{id} ───────────────────────────────────────────────


async def test_update_tenant_patches_name_and_description(
    web_authed_client: AsyncClient,
    repo: FakeTenantRepository,
) -> None:
    stored = await _seed(repo, name="acme", description="original")

    body = (
        await web_authed_client.put(
            f"/tenants/{stored.id}",
            json={"name": "acme corp", "description": "patched"},
        )
    ).json()

    assert body["data"]["name"] == "acme corp"
    assert body["data"]["description"] == "patched"


async def test_update_tenant_ignores_privileged_columns(
    web_authed_client: AsyncClient,
    repo: FakeTenantRepository,
) -> None:
    stored = await _seed(repo, name="acme")

    body = (
        await web_authed_client.put(
            f"/tenants/{stored.id}",
            json={"name": "acme", "status": "suspended", "storage_quota": 1},
        )
    ).json()

    assert body["data"]["status"] == "active"
    assert body["data"]["storage_quota"] == DEFAULT_STORAGE_QUOTA_BYTES
    assert repo.rows[stored.id].status == "active"


async def test_update_tenant_rejects_blank_name(
    web_authed_client: AsyncClient,
    repo: FakeTenantRepository,
) -> None:
    stored = await _seed(repo, name="acme")

    resp = await web_authed_client.put(f"/tenants/{stored.id}", json={"name": "  "})

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "tenant.name_required"


async def test_update_tenant_missing_returns_404(web_authed_client: AsyncClient) -> None:
    resp = await web_authed_client.put("/tenants/4242", json={"name": "acme"})

    assert resp.status_code == 404


# ── DELETE /tenants/{id} ────────────────────────────────────────────


async def test_delete_tenant_soft_deletes_and_reports_success(
    web_authed_client: AsyncClient,
    repo: FakeTenantRepository,
) -> None:
    stored = await _seed(repo)

    resp = await web_authed_client.delete(f"/tenants/{stored.id}")

    assert resp.status_code == 200
    assert resp.json() == {"success": True, "message": "Workspace deleted successfully"}
    assert repo.rows[stored.id].deleted_at is not None


async def test_delete_tenant_is_idempotent(
    web_authed_client: AsyncClient,
    repo: FakeTenantRepository,
) -> None:
    stored = await _seed(repo)
    await web_authed_client.delete(f"/tenants/{stored.id}")

    resp = await web_authed_client.delete(f"/tenants/{stored.id}")

    assert resp.status_code == 200
    assert resp.json()["success"] is True


async def test_delete_unknown_tenant_still_returns_200(web_authed_client: AsyncClient) -> None:
    resp = await web_authed_client.delete("/tenants/4242")

    assert resp.status_code == 200


__all__ = []
