"""Web-layer tests for the data-source router.

Per AGENTS.md §9, web routers are tested via ``httpx.AsyncClient``
against the app: the full HTTP path (routing, serialization, exception
mapping) with the service dependency overridden by a service-backed fake
so no database is involved.

The router is mounted on a purpose-built app rather than ``create_app()``
because app assembly (``lifespan.py``) is wired in a later checkpoint;
mounting it here keeps this suite honest about the routes themselves.

The load-bearing checks:

1. All 15 endpoints exist under the paths and methods Go registers.
2. Every endpoint declares the auth gate plus the role gate upstream uses
   (asserted structurally, so a dropped guard fails the suite rather than
   silently opening a credential-bearing route).
3. Credentials never appear in any response body; ``credentials`` carries
   presence only.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.core.infra.datasources.connector_base import ConnectorRegistry
from src.core.infra.datasources.service.datasource_service import DataSourceService
from src.core.system.audit_service import AuditLogService
from src.db.models.datasource import DataSource, SyncLog
from src.web.api.infra.datasources.router import router
from src.web.deps.infra_datasources import get_datasource_service
from src.web.deps.rbac import make_role_dep, require_role_dep
from src.web.exception_handler import register_exception_handlers
from src.web.middleware.auth import require_auth
from tests.fakes.auth_gates import override_auth_gates
from tests.fakes.datasources import (
    FakeAuditRepo,
    FakeDataSourceRepo,
    FakeSyncLogRepo,
    RecordingIngestor,
    StubConnector,
    unreachable_error,
)

TENANT_ID = 1
KB_ID = "kb-1"
NOW = datetime(2026, 4, 1, tzinfo=UTC)


# ── App wiring ───────────────────────────────────────────────────────


@pytest.fixture
def ds_repo() -> FakeDataSourceRepo:
    return FakeDataSourceRepo()


@pytest.fixture
def sync_log_repo() -> FakeSyncLogRepo:
    return FakeSyncLogRepo()


@pytest.fixture
def connector() -> StubConnector:
    return StubConnector("notion")


@pytest.fixture
def service(
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
    connector: StubConnector,
) -> DataSourceService:
    registry = ConnectorRegistry()
    registry.register(connector)
    return DataSourceService(
        ds_repo=ds_repo,  # type: ignore[arg-type]
        sync_log_repo=sync_log_repo,  # type: ignore[arg-type]
        connector_registry=registry,
        audit_service=AuditLogService(audit_repo=FakeAuditRepo()),  # type: ignore[arg-type]
    )


@pytest.fixture
def app(service: DataSourceService) -> FastAPI:
    application = FastAPI()
    register_exception_handlers(application)
    application.include_router(router)
    override_auth_gates(application)
    application.dependency_overrides[get_datasource_service] = lambda: service
    return application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _row(
    *,
    id: str = "ds-1",
    tenant_id: int = TENANT_ID,
    status: str = "active",
    config: dict[str, object] | None = None,
) -> DataSource:
    return DataSource(
        id=id,
        tenant_id=tenant_id,
        knowledge_base_id=KB_ID,
        name="my source",
        type="notion",
        config=config if config is not None else {"resource_ids": ["r-1"]},  # type: ignore[arg-type]
        status=status,
        created_at=NOW,
        updated_at=NOW,
    )


def _log(*, id: str = "log-1", data_source_id: str = "ds-1", status: str = "success") -> SyncLog:
    return SyncLog(
        id=id,
        data_source_id=data_source_id,
        tenant_id=TENANT_ID,
        status=status,
        started_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )


# ── Route inventory + permission gates ───────────────────────────────

# Go's RegisterDataSourceRoutes, verbatim.
EXPECTED_ROUTES: set[tuple[str, str]] = {
    ("GET", "/datasource/types"),
    ("POST", "/datasource/validate-credentials"),
    ("POST", "/datasource"),
    ("GET", "/datasource"),
    ("GET", "/datasource/{id}"),
    ("PUT", "/datasource/{id}"),
    ("DELETE", "/datasource/{id}"),
    ("POST", "/datasource/{id}/validate"),
    ("GET", "/datasource/{id}/resources"),
    ("POST", "/datasource/{id}/resource-ancestors"),
    ("POST", "/datasource/{id}/sync"),
    ("POST", "/datasource/{id}/pause"),
    ("POST", "/datasource/{id}/resume"),
    ("GET", "/datasource/{id}/logs"),
    ("GET", "/datasource/logs/{log_id}"),
}

# Reads are Viewer+; everything touching credentials or content is Admin+.
EXPECTED_ROLES: dict[tuple[str, str], str] = {
    ("GET", "/datasource/types"): "viewer",
    ("POST", "/datasource/validate-credentials"): "admin",
    ("POST", "/datasource"): "admin",
    ("GET", "/datasource"): "viewer",
    ("GET", "/datasource/{id}"): "viewer",
    ("PUT", "/datasource/{id}"): "admin",
    ("DELETE", "/datasource/{id}"): "admin",
    ("POST", "/datasource/{id}/validate"): "admin",
    ("GET", "/datasource/{id}/resources"): "admin",
    ("POST", "/datasource/{id}/resource-ancestors"): "admin",
    ("POST", "/datasource/{id}/sync"): "admin",
    ("POST", "/datasource/{id}/pause"): "admin",
    ("POST", "/datasource/{id}/resume"): "admin",
    ("GET", "/datasource/{id}/logs"): "viewer",
    ("GET", "/datasource/logs/{log_id}"): "viewer",
}


def _declared_routes() -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for route in router.routes:
        methods: set[str] = getattr(route, "methods", set()) or set()
        path = getattr(route, "path", "")
        for method in methods:
            found.add((method, path))
    return found


def test_router_declares_exactly_the_upstream_routes() -> None:
    assert _declared_routes() == EXPECTED_ROUTES


def test_every_endpoint_declares_the_auth_gate() -> None:
    # A missing AuthDep would expose an endpoint that reads external
    # credentials, so this is asserted structurally rather than by probing.
    for route in router.routes:
        deps = [d.call for d in getattr(route, "dependant", None).dependencies]  # type: ignore[union-attr]
        assert require_auth in deps, f"{route.path} is missing AuthDep"  # type: ignore[attr-defined]


def test_every_endpoint_declares_the_expected_role_gate() -> None:
    viewer_dep = make_role_dep("viewer")
    admin_dep = make_role_dep("admin")
    # make_role_dep returns a fresh closure per call, so identity cannot be
    # compared; the closed-over min_role is the observable.
    assert viewer_dep is not admin_dep

    for route in router.routes:
        path = getattr(route, "path", "")
        methods: set[str] = getattr(route, "methods", set()) or set()
        dependant = getattr(route, "dependant", None)
        assert dependant is not None
        roles: set[str] = set()
        for dep in dependant.dependencies:
            closure = getattr(dep.call, "__closure__", None)
            wrapped = getattr(dep.call, "__wrapped__", None)
            if closure is None and wrapped is None:
                continue
            for cell in closure or ():
                if isinstance(cell.cell_contents, str):
                    roles.add(cell.cell_contents)
        for method in methods:
            expected = EXPECTED_ROLES[(method, path)]
            assert expected in roles, f"{method} {path} expected role gate {expected}, got {roles}"


def test_role_gate_helper_is_the_shared_rbac_dependency() -> None:
    # Guards must come from web.deps.rbac, not a local reimplementation.
    dep = make_role_dep("admin")
    assert dep.__module__ == require_role_dep.__module__


# ── GET /datasources/types ───────────────────────────────────────────


async def test_list_types_returns_all_connectors_sorted(client: AsyncClient) -> None:
    resp = await client.get("/datasource/types")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 13
    priorities = [c["priority"] for c in body]
    assert priorities == sorted(priorities)
    assert {"type", "name", "priority", "auth_type", "capabilities"} <= set(body[0])


# ── POST /datasources/validate-credentials ───────────────────────────


async def test_validate_credentials_returns_connected(client: AsyncClient) -> None:
    resp = await client.post(
        "/datasource/validate-credentials",
        json={"type": "notion", "credentials": {"api_key": "k"}},
    )

    assert resp.status_code == 200
    assert resp.json() == {"status": "connected"}


async def test_validate_credentials_unknown_type_returns_404(client: AsyncClient) -> None:
    resp = await client.post(
        "/datasource/validate-credentials",
        json={"type": "nope", "credentials": {}},
    )

    assert resp.status_code == 404


async def test_validate_credentials_requires_body_fields(client: AsyncClient) -> None:
    resp = await client.post("/datasource/validate-credentials", json={"type": "notion"})

    assert resp.status_code == 422


async def test_validate_credentials_upstream_failure_returns_502(
    client: AsyncClient,
    connector: StubConnector,
) -> None:
    connector.validate_error = unreachable_error()

    resp = await client.post(
        "/datasource/validate-credentials",
        json={"type": "notion", "credentials": {"api_key": "bad"}},
    )

    assert resp.status_code == 502


# ── POST /datasources ────────────────────────────────────────────────


async def test_create_returns_201_and_entity(client: AsyncClient) -> None:
    resp = await client.post(
        "/datasource",
        json={"knowledge_base_id": KB_ID, "name": "notion sync", "type": "notion"},
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "notion sync"
    assert body["type"] == "notion"
    assert body["tenant_id"] == TENANT_ID
    assert body["status"] == "active"


async def test_create_response_reports_credential_presence_only(client: AsyncClient) -> None:
    resp = await client.post(
        "/datasource",
        json={
            "knowledge_base_id": KB_ID,
            "name": "notion sync",
            "type": "notion",
            "config": {"credentials": {"api_key": "super-secret"}},
        },
    )

    assert resp.status_code == 201
    assert "super-secret" not in resp.text
    assert resp.json()["credentials"] == {"credentials": {"configured": True}}
    assert "credentials" not in resp.json()["config"]


async def test_create_rejects_unknown_type_with_404(client: AsyncClient) -> None:
    resp = await client.post(
        "/datasource",
        json={"knowledge_base_id": KB_ID, "name": "x", "type": "nope"},
    )

    assert resp.status_code == 404


async def test_create_rejects_missing_required_fields(client: AsyncClient) -> None:
    resp = await client.post("/datasource", json={"name": "x"})

    assert resp.status_code == 422


# ── GET /datasources ─────────────────────────────────────────────────


async def test_list_returns_kb_sources(
    client: AsyncClient,
    ds_repo: FakeDataSourceRepo,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row

    resp = await client.get("/datasource", params={"kb_id": KB_ID})

    assert resp.status_code == 200
    assert [d["id"] for d in resp.json()] == ["ds-1"]


async def test_list_excludes_other_tenants(
    client: AsyncClient,
    ds_repo: FakeDataSourceRepo,
) -> None:
    mine = _row(id="ds-mine")
    theirs = _row(id="ds-theirs", tenant_id=99)
    ds_repo.rows[mine.id] = mine
    ds_repo.rows[theirs.id] = theirs

    resp = await client.get("/datasource", params={"kb_id": KB_ID})

    assert [d["id"] for d in resp.json()] == ["ds-mine"]


async def test_list_without_kb_id_returns_422(client: AsyncClient) -> None:
    resp = await client.get("/datasource")

    assert resp.status_code == 422


# ── GET /datasources/{id} ────────────────────────────────────────────


async def test_get_returns_entity_with_latest_sync_log(
    client: AsyncClient,
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row
    ds_repo.items_synced[row.id] = 11
    sync_log_repo.rows["log-1"] = _log()

    resp = await client.get(f"/datasource/{row.id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["total_items_synced"] == 11
    assert body["latest_sync_log"]["id"] == "log-1"


async def test_get_missing_returns_404(client: AsyncClient) -> None:
    resp = await client.get("/datasource/nope")

    assert resp.status_code == 404


async def test_get_cross_tenant_returns_404(
    client: AsyncClient,
    ds_repo: FakeDataSourceRepo,
) -> None:
    row = _row(tenant_id=99)
    ds_repo.rows[row.id] = row

    resp = await client.get(f"/datasource/{row.id}")

    # 404 not 403: a 403 would confirm the id exists.
    assert resp.status_code == 404


# ── PUT /datasources/{id} ────────────────────────────────────────────


async def test_update_patches_name(
    client: AsyncClient,
    ds_repo: FakeDataSourceRepo,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row

    resp = await client.put(f"/datasource/{row.id}", json={"name": "renamed"})

    assert resp.status_code == 200
    assert resp.json()["name"] == "renamed"


async def test_update_cannot_overwrite_credentials(
    client: AsyncClient,
    ds_repo: FakeDataSourceRepo,
) -> None:
    row = _row(config={"credentials": {"api_key": "original"}})
    ds_repo.rows[row.id] = row

    resp = await client.put(
        f"/datasource/{row.id}",
        json={"config": {"credentials": {"api_key": "attacker"}}},
    )

    assert resp.status_code == 200
    stored = ds_repo.rows[row.id].config
    assert stored is not None
    assert stored["credentials"] == {"api_key": "original"}


async def test_update_missing_returns_404(client: AsyncClient) -> None:
    resp = await client.put("/datasource/nope", json={"name": "x"})

    assert resp.status_code == 404


# ── DELETE /datasources/{id} ─────────────────────────────────────────


async def test_delete_returns_204_and_soft_deletes(
    client: AsyncClient,
    ds_repo: FakeDataSourceRepo,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row

    resp = await client.delete(f"/datasource/{row.id}")

    assert resp.status_code == 204
    assert ds_repo.rows[row.id].deleted_at is not None


async def test_delete_missing_returns_404(client: AsyncClient) -> None:
    resp = await client.delete("/datasource/nope")

    assert resp.status_code == 404


# ── POST /datasources/{id}/validate ──────────────────────────────────


async def test_validate_connection_returns_connected(
    client: AsyncClient,
    ds_repo: FakeDataSourceRepo,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row

    resp = await client.post(f"/datasource/{row.id}/validate")

    assert resp.status_code == 200
    assert resp.json() == {"status": "connected"}


async def test_validate_connection_failure_records_error_state(
    client: AsyncClient,
    ds_repo: FakeDataSourceRepo,
    connector: StubConnector,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row
    connector.validate_error = unreachable_error("token expired")

    resp = await client.post(f"/datasource/{row.id}/validate")

    assert resp.status_code == 502
    assert ds_repo.rows[row.id].status == "error"
    assert ds_repo.rows[row.id].error_message == "token expired"


# ── GET /datasources/{id}/resources ──────────────────────────────────


async def test_list_resources_returns_connector_output(
    client: AsyncClient,
    ds_repo: FakeDataSourceRepo,
    connector: StubConnector,
) -> None:
    from src.core.infra.datasources.types import Resource

    row = _row()
    ds_repo.rows[row.id] = row
    connector.resources = [
        Resource(external_id="page-1", name="Page", type="page", has_children=True)
    ]

    resp = await client.get(f"/datasource/{row.id}/resources")

    assert resp.status_code == 200
    body = resp.json()
    assert body[0]["external_id"] == "page-1"
    assert body[0]["has_children"] is True


async def test_list_resources_forwards_parent_id(
    client: AsyncClient,
    ds_repo: FakeDataSourceRepo,
    connector: StubConnector,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row

    resp = await client.get(f"/datasource/{row.id}/resources", params={"parent_id": "root-9"})

    assert resp.status_code == 200
    assert connector.list_resources_calls == ["root-9"]


# ── POST /datasources/{id}/resource-ancestors ────────────────────────


async def test_resolve_ancestors_returns_ancestor_list(
    client: AsyncClient,
    ds_repo: FakeDataSourceRepo,
    connector: StubConnector,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row
    connector.ancestors = ["root-1", "mid-2"]

    resp = await client.post(
        f"/datasource/{row.id}/resource-ancestors",
        json={"resource_ids": ["leaf-3"]},
    )

    assert resp.status_code == 200
    assert resp.json() == {"ancestors": ["root-1", "mid-2"]}


async def test_resolve_ancestors_empty_request_returns_empty(
    client: AsyncClient,
    ds_repo: FakeDataSourceRepo,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row

    resp = await client.post(
        f"/datasource/{row.id}/resource-ancestors",
        json={"resource_ids": []},
    )

    assert resp.status_code == 200
    assert resp.json() == {"ancestors": []}


# ── POST /datasources/{id}/sync ──────────────────────────────────────


async def test_manual_sync_returns_running_log(
    client: AsyncClient,
    ds_repo: FakeDataSourceRepo,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row

    resp = await client.post(f"/datasource/{row.id}/sync")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "running"
    assert body["data_source_id"] == row.id


async def test_manual_sync_on_unsyncable_source_returns_422(
    client: AsyncClient,
    ds_repo: FakeDataSourceRepo,
) -> None:
    row = _row(status="deleted")
    ds_repo.rows[row.id] = row

    resp = await client.post(f"/datasource/{row.id}/sync")

    assert resp.status_code == 422


# ── pause / resume ───────────────────────────────────────────────────


async def test_pause_returns_paused(
    client: AsyncClient,
    ds_repo: FakeDataSourceRepo,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row

    resp = await client.post(f"/datasource/{row.id}/pause")

    assert resp.status_code == 200
    assert resp.json() == {"status": "paused"}
    assert ds_repo.rows[row.id].status == "paused"


async def test_resume_returns_active(
    client: AsyncClient,
    ds_repo: FakeDataSourceRepo,
) -> None:
    row = _row(status="paused")
    ds_repo.rows[row.id] = row

    resp = await client.post(f"/datasource/{row.id}/resume")

    assert resp.status_code == 200
    assert resp.json() == {"status": "active"}
    assert ds_repo.rows[row.id].status == "active"


# ── sync logs ────────────────────────────────────────────────────────


async def test_list_sync_logs_returns_history(
    client: AsyncClient,
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row
    sync_log_repo.rows["log-1"] = _log()

    resp = await client.get(f"/datasource/{row.id}/logs")

    assert resp.status_code == 200
    assert [entry["id"] for entry in resp.json()] == ["log-1"]


async def test_list_sync_logs_rejects_oversized_limit(
    client: AsyncClient,
    ds_repo: FakeDataSourceRepo,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row

    resp = await client.get(f"/datasource/{row.id}/logs", params={"limit": 5000})

    assert resp.status_code == 422


async def test_get_sync_log_returns_entry(
    client: AsyncClient,
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
) -> None:
    row = _row()
    ds_repo.rows[row.id] = row
    sync_log_repo.rows["log-1"] = _log()

    resp = await client.get("/datasource/logs/log-1")

    assert resp.status_code == 200
    assert resp.json()["id"] == "log-1"


async def test_get_sync_log_missing_returns_404(client: AsyncClient) -> None:
    resp = await client.get("/datasource/logs/nope")

    assert resp.status_code == 404


async def test_sync_log_route_is_not_shadowed_by_id_route(
    client: AsyncClient,
    ds_repo: FakeDataSourceRepo,
) -> None:
    # "logs" must never be captured as a data-source id.
    row = _row(id="logs")
    ds_repo.rows[row.id] = row

    resp = await client.get("/datasource/logs/nope")

    assert resp.status_code == 404
    assert "sync log" in resp.text.lower()


# ── ingestion tally surfaced over HTTP ───────────────────────────────


async def test_sync_result_tally_visible_through_log_endpoint(
    client: AsyncClient,
    service: DataSourceService,
    ds_repo: FakeDataSourceRepo,
    sync_log_repo: FakeSyncLogRepo,
    connector: StubConnector,
) -> None:
    from src.core.infra.datasources.types import FetchedItem

    row = _row()
    ds_repo.rows[row.id] = row
    connector.items = [
        FetchedItem(external_id="a", title="A"),
        FetchedItem(external_id="b", title="B"),
    ]
    service._ingestor = RecordingIngestor(updates={"b"})

    opened = await client.post(f"/datasource/{row.id}/sync")
    log_id = opened.json()["id"]
    await service.process_sync(data_source_id=row.id, sync_log_id=log_id)

    resp = await client.get(f"/datasource/logs/{log_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["items_created"] == 1
    assert body["items_updated"] == 1
