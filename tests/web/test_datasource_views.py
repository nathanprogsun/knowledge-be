"""Web-layer tests for the data-source router.

Exercises the router over HTTP via ``httpx.AsyncClient`` against the
app: the full HTTP path (routing, serialization, exception mapping)
with the service dependency overridden by a service-backed
``AsyncMock(spec=...)`` repository so no database is involved. The
non-repository doubles (``StubConnector``, ``RecordingIngestor``)
remain as protocol doubles for the connector / ingestor seams.

Uses the shared ``web_app`` fixture (header-based auth) and applies
the service dep override on it; the real ``require_auth`` dep resolves
the principal via the ``x-knowledge-*`` header trio.

The load-bearing checks:

1. All 15 endpoints exist under the paths and methods Go registers.
2. Every endpoint declares the auth gate plus the role gate upstream uses
   (asserted structurally, so a dropped guard fails the suite rather than
   silently opening a credential-bearing route).
3. Credentials never appear in any response body; ``credentials`` carries
   presence only.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from src.core.infra.datasources.connector_base import ConnectorRegistry
from src.core.infra.datasources.service.datasource_service import DataSourceService
from src.core.system.audit_service import AuditLogService
from src.db.dao.audit_log_repository import AuditLogRepository
from src.db.dao.datasource_repository import DataSourceRepository, SyncLogRepository
from src.db.models.datasource import DataSource, SyncLog
from src.db.models.system.audit_log import AuditLog
from src.web.api.infra.datasources.router import router
from src.web.deps.infra_datasources import get_datasource_service
from src.web.deps.rbac import make_role_dep, require_role_dep
from src.web.middleware.auth import require_auth
from tests.integration.web.conftest import web_app, web_authed_client  # noqa: F401
from tests.unit.fakes.datasources import (  # type: ignore[attr-defined]
    RecordingIngestor,
    StubConnector,
    unreachable_error,
)

TENANT_ID = 1
KB_ID = "kb-1"
NOW = datetime(2026, 4, 1, tzinfo=UTC)


# ── App wiring ───────────────────────────────────────────────────────


@pytest.fixture
def ds_repo() -> AsyncMock:
    """``AsyncMock(spec=DataSourceRepository)`` with stateful closures."""
    repo = AsyncMock(spec=DataSourceRepository)
    rows: dict[str, DataSource] = {}
    items_synced: dict[str, int] = {}

    async def _create(row: DataSource) -> DataSource:
        rows[row.id] = row
        return row

    async def _find_by_id_or_none(id: str) -> DataSource | None:
        row = rows.get(id)
        if row is None or row.deleted_at is not None:
            return None
        return row

    async def _find_by_knowledge_base(
        knowledge_base_id: str,
    ) -> list[DataSource]:
        out = [
            r
            for r in rows.values()
            if r.knowledge_base_id == knowledge_base_id and r.deleted_at is None
        ]
        return sorted(out, key=lambda r: r.created_at, reverse=True)

    async def _update(row: DataSource) -> DataSource:
        existing = rows.get(row.id)
        if existing is None:
            from src.common.exception import ValidationError

            raise ValidationError(code="db.not_found", message="row missing")
        # Immutable columns are preserved exactly as the real repo does.
        persisted = row.model_copy(
            update={
                "tenant_id": existing.tenant_id,
                "knowledge_base_id": existing.knowledge_base_id,
                "created_at": existing.created_at,
            }
        )
        rows[row.id] = persisted
        return persisted

    async def _soft_delete(*, id: str, now: datetime) -> bool:
        existing = rows.get(id)
        if existing is None or existing.deleted_at is not None:
            return False
        rows[id] = existing.model_copy(update={"deleted_at": now, "updated_at": now})
        return True

    async def _count_items_synced(data_source_id: str) -> int:
        return items_synced.get(data_source_id, 0)

    repo.create.side_effect = _create
    repo.find_by_id_or_none.side_effect = _find_by_id_or_none
    repo.find_by_knowledge_base.side_effect = _find_by_knowledge_base
    repo.update.side_effect = _update
    repo.soft_delete.side_effect = _soft_delete
    repo.count_items_synced.side_effect = _count_items_synced
    repo._rows = rows  # type: ignore[attr-defined]
    repo._items_synced = items_synced  # type: ignore[attr-defined]
    return repo


@pytest.fixture
def sync_log_repo() -> AsyncMock:
    """``AsyncMock(spec=SyncLogRepository)`` with stateful closures."""
    repo = AsyncMock(spec=SyncLogRepository)
    rows: dict[str, SyncLog] = {}

    async def _create(row: SyncLog) -> SyncLog:
        rows[row.id] = row
        return row

    async def _find_by_id_or_none(id: str) -> SyncLog | None:
        return rows.get(id)

    async def _find_by_data_source(
        data_source_id: str, *, limit: int, offset: int
    ) -> list[SyncLog]:
        out = [r for r in rows.values() if r.data_source_id == data_source_id]
        out = sorted(out, key=lambda r: r.started_at, reverse=True)
        return out[offset : offset + limit]

    async def _find_latest(data_source_id: str) -> SyncLog | None:
        out = [r for r in rows.values() if r.data_source_id == data_source_id]
        if not out:
            return None
        return max(out, key=lambda r: r.started_at)

    async def _update(row: SyncLog) -> SyncLog:
        rows[row.id] = row
        return row

    async def _cancel_pending(*, data_source_id: str, now: datetime) -> int:
        count = 0
        for log_id, row in list(rows.items()):
            if row.data_source_id == data_source_id and row.status == "running":
                rows[log_id] = row.model_copy(
                    update={
                        "status": "canceled",
                        "finished_at": now,
                        "updated_at": now,
                    }
                )
                count += 1
        return count

    repo.create.side_effect = _create
    repo.find_by_id_or_none.side_effect = _find_by_id_or_none
    repo.find_by_data_source.side_effect = _find_by_data_source
    repo.find_latest.side_effect = _find_latest
    repo.update.side_effect = _update
    repo.cancel_pending_by_data_source.side_effect = _cancel_pending
    repo._rows = rows  # type: ignore[attr-defined]
    return repo


@pytest.fixture
def audit_repo() -> AsyncMock:
    """``AsyncMock(spec=AuditLogRepository)`` for the audit pipeline."""
    repo = AsyncMock(spec=AuditLogRepository)
    audit_rows: list[AuditLog] = []
    counter = [0]

    async def _create(entry: AuditLog) -> AuditLog:
        counter[0] += 1
        persisted = entry.model_copy(update={"id": counter[0]})
        audit_rows.append(persisted)
        return persisted

    repo.create.side_effect = _create
    return repo


@pytest.fixture
def connector() -> StubConnector:
    return StubConnector("notion")


@pytest.fixture
def service(
    ds_repo: AsyncMock,
    sync_log_repo: AsyncMock,
    audit_repo: AsyncMock,
    connector: StubConnector,
) -> DataSourceService:
    registry = ConnectorRegistry()
    registry.register(connector)
    return DataSourceService(
        ds_repo=ds_repo,
        sync_log_repo=sync_log_repo,
        connector_registry=registry,
        audit_service=AuditLogService(audit_repo=audit_repo),
    )


@pytest.fixture
def app(
    request: pytest.FixtureRequest,  # noqa: ARG001 - explicit fixture-param
    web_app: FastAPI,  # noqa: ARG001 - resolved from the parent conftest
    service: DataSourceService,
) -> FastAPI:
    """Override ``get_datasource_service`` on the shared web app."""
    web_app.dependency_overrides[get_datasource_service] = lambda: service
    return web_app


@pytest.fixture
def client(app: FastAPI, web_authed_client: AsyncClient) -> AsyncClient:  # noqa: ARG001
    """Alias ``web_authed_client``; depending on ``app`` forces the
    dep-override fixture to run before the test executes."""
    return web_authed_client


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
    ds_repo: AsyncMock,
) -> None:
    row = _row()
    ds_repo._rows[row.id] = row  # type: ignore[attr-defined]

    resp = await client.get("/datasource", params={"kb_id": KB_ID})

    assert resp.status_code == 200
    assert [d["id"] for d in resp.json()] == ["ds-1"]


async def test_list_excludes_other_tenants(
    client: AsyncClient,
    ds_repo: AsyncMock,
) -> None:
    mine = _row(id="ds-mine")
    theirs = _row(id="ds-theirs", tenant_id=99)
    rows = ds_repo._rows  # type: ignore[attr-defined]
    rows[mine.id] = mine
    rows[theirs.id] = theirs

    resp = await client.get("/datasource", params={"kb_id": KB_ID})

    assert [d["id"] for d in resp.json()] == ["ds-mine"]


async def test_list_without_kb_id_returns_422(client: AsyncClient) -> None:
    resp = await client.get("/datasource")

    assert resp.status_code == 422


# ── GET /datasources/{id} ────────────────────────────────────────────


async def test_get_returns_entity_with_latest_sync_log(
    client: AsyncClient,
    ds_repo: AsyncMock,
    sync_log_repo: AsyncMock,
) -> None:
    row = _row()
    ds_repo._rows[row.id] = row  # type: ignore[attr-defined]
    ds_repo._items_synced[row.id] = 11  # type: ignore[attr-defined]
    sync_log_repo._rows["log-1"] = _log()  # type: ignore[attr-defined]

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
    ds_repo: AsyncMock,
) -> None:
    row = _row(tenant_id=99)
    ds_repo._rows[row.id] = row  # type: ignore[attr-defined]

    resp = await client.get(f"/datasource/{row.id}")

    # 404 not 403: a 403 would confirm the id exists.
    assert resp.status_code == 404


# ── PUT /datasources/{id} ────────────────────────────────────────────


async def test_update_patches_name(
    client: AsyncClient,
    ds_repo: AsyncMock,
) -> None:
    row = _row()
    ds_repo._rows[row.id] = row  # type: ignore[attr-defined]

    resp = await client.put(f"/datasource/{row.id}", json={"name": "renamed"})

    assert resp.status_code == 200
    assert resp.json()["name"] == "renamed"


async def test_update_cannot_overwrite_credentials(
    client: AsyncClient,
    ds_repo: AsyncMock,
) -> None:
    row = _row(config={"credentials": {"api_key": "original"}})
    ds_repo._rows[row.id] = row  # type: ignore[attr-defined]

    resp = await client.put(
        f"/datasource/{row.id}",
        json={"config": {"credentials": {"api_key": "attacker"}}},
    )

    assert resp.status_code == 200
    stored = ds_repo._rows[row.id].config  # type: ignore[attr-defined]
    assert stored is not None
    assert stored["credentials"] == {"api_key": "original"}


async def test_update_missing_returns_404(client: AsyncClient) -> None:
    resp = await client.put("/datasource/nope", json={"name": "x"})

    assert resp.status_code == 404


# ── DELETE /datasources/{id} ─────────────────────────────────────────


async def test_delete_returns_204_and_soft_deletes(
    client: AsyncClient,
    ds_repo: AsyncMock,
) -> None:
    row = _row()
    ds_repo._rows[row.id] = row  # type: ignore[attr-defined]

    resp = await client.delete(f"/datasource/{row.id}")

    assert resp.status_code == 204
    assert ds_repo._rows[row.id].deleted_at is not None  # type: ignore[attr-defined]


async def test_delete_missing_returns_404(client: AsyncClient) -> None:
    resp = await client.delete("/datasource/nope")

    assert resp.status_code == 404


# ── POST /datasources/{id}/validate ──────────────────────────────────


async def test_validate_connection_returns_connected(
    client: AsyncClient,
    ds_repo: AsyncMock,
) -> None:
    row = _row()
    ds_repo._rows[row.id] = row  # type: ignore[attr-defined]

    resp = await client.post(f"/datasource/{row.id}/validate")

    assert resp.status_code == 200
    assert resp.json() == {"status": "connected"}


async def test_validate_connection_failure_records_error_state(
    client: AsyncClient,
    ds_repo: AsyncMock,
    connector: StubConnector,
) -> None:
    row = _row()
    ds_repo._rows[row.id] = row  # type: ignore[attr-defined]
    connector.validate_error = unreachable_error("token expired")

    resp = await client.post(f"/datasource/{row.id}/validate")

    assert resp.status_code == 502
    stored = ds_repo._rows[row.id]  # type: ignore[attr-defined]
    assert stored.status == "error"
    assert stored.error_message == "token expired"


# ── GET /datasources/{id}/resources ──────────────────────────────────


async def test_list_resources_returns_connector_output(
    client: AsyncClient,
    ds_repo: AsyncMock,
    connector: StubConnector,
) -> None:
    from src.core.infra.datasources.types import Resource

    row = _row()
    ds_repo._rows[row.id] = row  # type: ignore[attr-defined]
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
    ds_repo: AsyncMock,
    connector: StubConnector,
) -> None:
    row = _row()
    ds_repo._rows[row.id] = row  # type: ignore[attr-defined]

    resp = await client.get(f"/datasource/{row.id}/resources", params={"parent_id": "root-9"})

    assert resp.status_code == 200
    assert connector.list_resources_calls == ["root-9"]


# ── POST /datasources/{id}/resource-ancestors ────────────────────────


async def test_resolve_ancestors_returns_ancestor_list(
    client: AsyncClient,
    ds_repo: AsyncMock,
    connector: StubConnector,
) -> None:
    row = _row()
    ds_repo._rows[row.id] = row  # type: ignore[attr-defined]
    connector.ancestors = ["root-1", "mid-2"]

    resp = await client.post(
        f"/datasource/{row.id}/resource-ancestors",
        json={"resource_ids": ["leaf-3"]},
    )

    assert resp.status_code == 200
    assert resp.json() == {"ancestors": ["root-1", "mid-2"]}


async def test_resolve_ancestors_empty_request_returns_empty(
    client: AsyncClient,
    ds_repo: AsyncMock,
) -> None:
    row = _row()
    ds_repo._rows[row.id] = row  # type: ignore[attr-defined]

    resp = await client.post(
        f"/datasource/{row.id}/resource-ancestors",
        json={"resource_ids": []},
    )

    assert resp.status_code == 200
    assert resp.json() == {"ancestors": []}


# ── POST /datasources/{id}/sync ──────────────────────────────────────


async def test_manual_sync_returns_running_log(
    client: AsyncClient,
    ds_repo: AsyncMock,
) -> None:
    row = _row()
    ds_repo._rows[row.id] = row  # type: ignore[attr-defined]

    resp = await client.post(f"/datasource/{row.id}/sync")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "running"
    assert body["data_source_id"] == row.id


async def test_manual_sync_on_unsyncable_source_returns_422(
    client: AsyncClient,
    ds_repo: AsyncMock,
) -> None:
    row = _row(status="deleted")
    ds_repo._rows[row.id] = row  # type: ignore[attr-defined]

    resp = await client.post(f"/datasource/{row.id}/sync")

    assert resp.status_code == 422


# ── pause / resume ───────────────────────────────────────────────────


async def test_pause_returns_paused(
    client: AsyncClient,
    ds_repo: AsyncMock,
) -> None:
    row = _row()
    ds_repo._rows[row.id] = row  # type: ignore[attr-defined]

    resp = await client.post(f"/datasource/{row.id}/pause")

    assert resp.status_code == 200
    assert resp.json() == {"status": "paused"}
    assert ds_repo._rows[row.id].status == "paused"  # type: ignore[attr-defined]


async def test_resume_returns_active(
    client: AsyncClient,
    ds_repo: AsyncMock,
) -> None:
    row = _row(status="paused")
    ds_repo._rows[row.id] = row  # type: ignore[attr-defined]

    resp = await client.post(f"/datasource/{row.id}/resume")

    assert resp.status_code == 200
    assert resp.json() == {"status": "active"}
    assert ds_repo._rows[row.id].status == "active"  # type: ignore[attr-defined]


# ── sync logs ────────────────────────────────────────────────────────


async def test_list_sync_logs_returns_history(
    client: AsyncClient,
    ds_repo: AsyncMock,
    sync_log_repo: AsyncMock,
) -> None:
    row = _row()
    ds_repo._rows[row.id] = row  # type: ignore[attr-defined]
    sync_log_repo._rows["log-1"] = _log()  # type: ignore[attr-defined]

    resp = await client.get(f"/datasource/{row.id}/logs")

    assert resp.status_code == 200
    assert [entry["id"] for entry in resp.json()] == ["log-1"]


async def test_list_sync_logs_rejects_oversized_limit(
    client: AsyncClient,
    ds_repo: AsyncMock,
) -> None:
    row = _row()
    ds_repo._rows[row.id] = row  # type: ignore[attr-defined]

    resp = await client.get(f"/datasource/{row.id}/logs", params={"limit": 5000})

    assert resp.status_code == 422


async def test_get_sync_log_returns_entry(
    client: AsyncClient,
    ds_repo: AsyncMock,
    sync_log_repo: AsyncMock,
) -> None:
    row = _row()
    ds_repo._rows[row.id] = row  # type: ignore[attr-defined]
    sync_log_repo._rows["log-1"] = _log()  # type: ignore[attr-defined]

    resp = await client.get("/datasource/logs/log-1")

    assert resp.status_code == 200
    assert resp.json()["id"] == "log-1"


async def test_get_sync_log_missing_returns_404(client: AsyncClient) -> None:
    resp = await client.get("/datasource/logs/nope")

    assert resp.status_code == 404


async def test_sync_log_route_is_not_shadowed_by_id_route(
    client: AsyncClient,
    ds_repo: AsyncMock,
) -> None:
    # "logs" must never be captured as a data-source id.
    row = _row(id="logs")
    ds_repo._rows[row.id] = row  # type: ignore[attr-defined]

    resp = await client.get("/datasource/logs/nope")

    assert resp.status_code == 404
    assert "sync log" in resp.text.lower()


# ── ingestion tally surfaced over HTTP ───────────────────────────────


async def test_sync_result_tally_visible_through_log_endpoint(
    client: AsyncClient,
    service: DataSourceService,
    ds_repo: AsyncMock,
    connector: StubConnector,
) -> None:
    from src.core.infra.datasources.types import FetchedItem

    row = _row()
    ds_repo._rows[row.id] = row  # type: ignore[attr-defined]
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