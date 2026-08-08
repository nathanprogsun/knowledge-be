"""Web-layer tests for the knowledge document routers.

Exercises both routers (``/knowledge-bases/{id}/knowledge`` and
``/knowledge``) over HTTP via ``TestClient`` against the app: the full
HTTP path (routing, serialization, exception mapping) with the two
document dependency factories overridden by ``AsyncMock(spec=...)``
services so no database is involved.

Uses the shared ``web_app`` fixture (header-based auth) and applies the
dependency overrides on it; the real ``require_auth`` dep resolves the
principal via the ``x-knowledge-*`` header trio.

The load-bearing checks:

1. All routes exist under the paths and methods the upstream handler
   registers (both KB-scoped creates and per-document routes).
2. Every endpoint declares the auth gate plus the role gate upstream
   uses (reads Viewer+, mutations Contributor+).
3. Success payloads carry the ``success``/``data`` envelope; business
   errors map to the standard error envelope with the expected status.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.common.exception import NotFoundError, ValidationError
from src.common.pagination import PaginationResponse
from src.core.contracts.knowledge import Knowledge
from src.core.knowledge.documents.documents_orchestrator import (
    KnowledgeDocumentsOrchestrator,
)
from src.core.knowledge.documents.service.knowledge_service import KnowledgeService
from src.web.api.knowledge.documents.router import (
    documents_router,
    kb_documents_router,
)
from src.web.deps.knowledge_documents import (
    get_documents_orchestrator,
    get_knowledge_service,
)
from src.web.deps.rbac import make_role_dep
from src.web.middleware.auth import require_auth

TENANT_ID = 1
KB_ID = "kb-1"
NOW = datetime(2026, 4, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _bind_tenant_id_to_admin(
    admin_user: tuple[int, int],
) -> None:
    """Rewrite the module-level ``TENANT_ID`` to the minted admin tenant.

    Per-test conftest mints a fresh ``tenant_id``; this rebind keeps the
    helper closures (which build rows keyed by ``TENANT_ID``) aligned
    with the principal the authed client presents.
    """
    global TENANT_ID
    TENANT_ID = admin_user[1]


# ── App wiring ───────────────────────────────────────────────────────


@pytest.fixture
def knowledge_service() -> AsyncMock:
    """``AsyncMock(spec=KnowledgeService)`` for the CRUD surface."""
    return AsyncMock(spec=KnowledgeService)


@pytest.fixture
def orchestrator() -> AsyncMock:
    """``AsyncMock(spec=KnowledgeDocumentsOrchestrator)`` for lifecycle ops."""
    return AsyncMock(spec=KnowledgeDocumentsOrchestrator)


@pytest.fixture
def app(
    request: pytest.FixtureRequest,
    web_app: FastAPI,
    knowledge_service: AsyncMock,
    orchestrator: AsyncMock,
) -> FastAPI:
    """Override both document dependency factories on the shared web app."""
    web_app.dependency_overrides[get_knowledge_service] = lambda: knowledge_service
    web_app.dependency_overrides[get_documents_orchestrator] = lambda: orchestrator
    return web_app


@pytest.fixture
def client(app: FastAPI, web_authed_client: TestClient) -> TestClient:
    """Alias ``web_authed_client``; depending on ``app`` forces the
    dep-override fixture to run before the test executes."""
    return web_authed_client


def _knowledge(
    *,
    id: str = "kn-1",
    tenant_id: int | None = None,
    kb_id: str = KB_ID,
    title: str = "doc",
    parse_status: str = "completed",
) -> Knowledge:
    if tenant_id is None:
        tenant_id = TENANT_ID
    return Knowledge(
        id=id,
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        type="file",
        title=title,
        description=None,
        source="manual",
        channel="web",
        tag_id=None,
        summary_status="none",
        parse_status=parse_status,
        enable_status="enabled",
        embedding_model_id=None,
        file_name=None,
        file_type=None,
        file_size=None,
        file_hash=None,
        file_path=None,
        storage_size=None,
        metadata=None,
        created_at=NOW,
        updated_at=NOW,
        processed_at=None,
        error_message=None,
        deleted_at=None,
    )


# ── Route inventory + permission gates ───────────────────────────────

# The upstream document route table, verbatim.
EXPECTED_ROUTES: set[tuple[str, str]] = {
    ("POST", "/knowledge-bases/{id}/knowledge/file"),
    ("POST", "/knowledge-bases/{id}/knowledge/url"),
    ("POST", "/knowledge-bases/{id}/knowledge/passage"),
    ("POST", "/knowledge-bases/{id}/knowledge/manual"),
    ("GET", "/knowledge-bases/{id}/knowledge"),
    ("GET", "/knowledge/{id}"),
    ("PUT", "/knowledge/{id}"),
    ("DELETE", "/knowledge/{id}"),
    ("POST", "/knowledge/{id}/reparse"),
    ("POST", "/knowledge/{id}/cancel-parse"),
    ("POST", "/knowledge/{id}/clone"),
    ("POST", "/knowledge/move"),
    ("GET", "/knowledge/move/progress/{task_id}"),
}

# Reads are Viewer+; every content mutation is Contributor+.
EXPECTED_ROLES: dict[tuple[str, str], str] = {
    ("POST", "/knowledge-bases/{id}/knowledge/file"): "contributor",
    ("POST", "/knowledge-bases/{id}/knowledge/url"): "contributor",
    ("POST", "/knowledge-bases/{id}/knowledge/passage"): "contributor",
    ("POST", "/knowledge-bases/{id}/knowledge/manual"): "contributor",
    ("GET", "/knowledge-bases/{id}/knowledge"): "viewer",
    ("GET", "/knowledge/{id}"): "viewer",
    ("PUT", "/knowledge/{id}"): "contributor",
    ("DELETE", "/knowledge/{id}"): "contributor",
    ("POST", "/knowledge/{id}/reparse"): "contributor",
    ("POST", "/knowledge/{id}/cancel-parse"): "contributor",
    ("POST", "/knowledge/{id}/clone"): "contributor",
    ("POST", "/knowledge/move"): "contributor",
    ("GET", "/knowledge/move/progress/{task_id}"): "viewer",
}


def _declared_routes() -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for router in (kb_documents_router, documents_router):
        for route in router.routes:
            methods: set[str] = getattr(route, "methods", set()) or set()
            path = getattr(route, "path", "")
            for method in methods:
                found.add((method, path))
    return found


def test_router_declares_exactly_the_upstream_routes() -> None:
    assert _declared_routes() == EXPECTED_ROUTES


def test_every_endpoint_declares_the_auth_gate() -> None:
    for router in (kb_documents_router, documents_router):
        for route in router.routes:
            deps = [d.call for d in getattr(route, "dependant", None).dependencies]  # type: ignore[union-attr]
            assert require_auth in deps, f"{route.path} is missing AuthDep"  # type: ignore[attr-defined]


def test_every_endpoint_declares_the_expected_role_gate() -> None:
    for router in (kb_documents_router, documents_router):
        for route in router.routes:
            path = getattr(route, "path", "")
            methods: set[str] = getattr(route, "methods", set()) or set()
            dependant = getattr(route, "dependant", None)
            assert dependant is not None
            roles: set[str] = set()
            for dep in dependant.dependencies:
                closure = getattr(dep.call, "__closure__", None)
                for cell in closure or ():
                    if isinstance(cell.cell_contents, str):
                        roles.add(cell.cell_contents)
            for method in methods:
                expected = EXPECTED_ROLES[(method, path)]
                assert expected in roles, f"{method} {path} expected role gate {expected}, got {roles}"


def test_role_gate_helper_is_the_shared_rbac_dependency() -> None:
    # Guards must come from web.deps.rbac, not a local reimplementation.
    dep = make_role_dep("contributor")
    assert dep.__module__.endswith("web.deps.rbac")


# ── POST /knowledge-bases/{id}/knowledge/file ─────────────────────────


async def test_create_file_returns_document(
    client: TestClient,
    orchestrator: AsyncMock,
) -> None:
    orchestrator.create_from_file.return_value = _knowledge(id="kn-file")

    resp = client.post(
        f"/knowledge-bases/{KB_ID}/knowledge/file",
        files={"file": ("a.txt", b"file content", "text/plain")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["id"] == "kn-file"
    assert orchestrator.create_from_file.await_count == 1


async def test_create_file_forwards_form_fields(
    client: TestClient,
    orchestrator: AsyncMock,
) -> None:
    orchestrator.create_from_file.return_value = _knowledge()

    client.post(
        f"/knowledge-bases/{KB_ID}/knowledge/file",
        files={"file": ("a.txt", b"data", "text/plain")},
        data={
            "file_name": "custom.txt",
            "tag_ids": "tag-1,tag-2",
            "enable_multimodel": "true",
            "channel": "api",
        },
    )

    call = orchestrator.create_from_file.await_args.kwargs
    assert call["custom_file_name"] == "custom.txt"
    assert call["tag_ids"] == ["tag-1", "tag-2"]
    assert call["enable_multimodel"] is True
    assert call["channel"] == "api"


async def test_create_file_rejects_missing_file(client: TestClient) -> None:
    resp = client.post(f"/knowledge-bases/{KB_ID}/knowledge/file")

    assert resp.status_code == 422


# ── POST /knowledge-bases/{id}/knowledge/url ──────────────────────────


async def test_create_url_returns_201_document(
    client: TestClient,
    orchestrator: AsyncMock,
) -> None:
    orchestrator.create_from_url.return_value = _knowledge(id="kn-url")

    resp = client.post(
        f"/knowledge-bases/{KB_ID}/knowledge/url",
        json={"url": "https://example.com/page"},
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["id"] == "kn-url"


async def test_create_url_requires_url(client: TestClient) -> None:
    resp = client.post(f"/knowledge-bases/{KB_ID}/knowledge/url", json={})

    assert resp.status_code == 422


# ── POST /knowledge-bases/{id}/knowledge/passage ──────────────────────


async def test_create_passage_returns_201_document(
    client: TestClient,
    orchestrator: AsyncMock,
) -> None:
    orchestrator.create_from_passage.return_value = _knowledge(id="kn-passage")

    resp = client.post(
        f"/knowledge-bases/{KB_ID}/knowledge/passage",
        json={"passages": ["first", "second"], "sync": True},
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["id"] == "kn-passage"
    assert orchestrator.create_from_passage.await_args.kwargs["sync"] is True


async def test_create_passage_rejects_empty_list(client: TestClient) -> None:
    resp = client.post(
        f"/knowledge-bases/{KB_ID}/knowledge/passage",
        json={"passages": []},
    )

    assert resp.status_code == 422


# ── POST /knowledge-bases/{id}/knowledge/manual ───────────────────────


async def test_create_manual_returns_document(
    client: TestClient,
    orchestrator: AsyncMock,
    default_create_document_request: dict[str, object],
) -> None:
    orchestrator.create_from_manual.return_value = _knowledge(id="kn-manual")

    resp = client.post(
        f"/knowledge-bases/{KB_ID}/knowledge/manual",
        json=default_create_document_request,
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["id"] == "kn-manual"


async def test_create_manual_requires_title_and_content(client: TestClient) -> None:
    resp = client.post(f"/knowledge-bases/{KB_ID}/knowledge/manual", json={})

    assert resp.status_code == 422


# ── GET /knowledge-bases/{id}/knowledge ───────────────────────────────


async def test_list_returns_paged_envelope(
    client: TestClient,
    knowledge_service: AsyncMock,
) -> None:
    knowledge_service.list_documents_paged.return_value = PaginationResponse(
        total=1,
        page=1,
        page_size=20,
        data=[_knowledge()],
    )

    resp = client.get(f"/knowledge-bases/{KB_ID}/knowledge")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["total"] == 1
    assert body["page"] == 1
    assert body["page_size"] == 20
    assert body["data"][0]["id"] == "kn-1"


async def test_list_rejects_unparseable_time(
    client: TestClient,
    knowledge_service: AsyncMock,
) -> None:
    resp = client.get(
        f"/knowledge-bases/{KB_ID}/knowledge",
        params={"start_time": "not-a-date"},
    )

    assert resp.status_code == 422


# ── GET /knowledge/{id} ───────────────────────────────────────────────


async def test_get_returns_document(
    client: TestClient,
    knowledge_service: AsyncMock,
) -> None:
    knowledge_service.get_document.return_value = _knowledge()

    resp = client.get("/knowledge/kn-1")

    assert resp.status_code == 200
    assert resp.json()["data"]["id"] == "kn-1"


async def test_get_missing_returns_404(
    client: TestClient,
    knowledge_service: AsyncMock,
) -> None:
    knowledge_service.get_document.side_effect = NotFoundError(
        code="knowledge.not_found", message="knowledge not found"
    )

    resp = client.get("/knowledge/nope")

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "knowledge.not_found"


async def test_get_cross_tenant_returns_404(
    client: TestClient,
    knowledge_service: AsyncMock,
) -> None:
    # Out-of-scope ids read as 404 so the id space is not enumerable.
    knowledge_service.get_document.side_effect = NotFoundError(
        code="knowledge.not_found", message="knowledge not found"
    )

    resp = client.get("/knowledge/kn-theirs")

    assert resp.status_code == 404


# ── PUT /knowledge/{id} ───────────────────────────────────────────────


async def test_update_patches_title(
    client: TestClient,
    knowledge_service: AsyncMock,
) -> None:
    knowledge_service.update_document.return_value = _knowledge(title="renamed")

    resp = client.put("/knowledge/kn-1", json={"title": "renamed"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["message"] == "Knowledge updated successfully"
    assert body["data"]["title"] == "renamed"


async def test_update_missing_returns_404(
    client: TestClient,
    knowledge_service: AsyncMock,
) -> None:
    knowledge_service.update_document.side_effect = NotFoundError(
        code="knowledge.not_found", message="knowledge not found"
    )

    resp = client.put("/knowledge/nope", json={"title": "x"})

    assert resp.status_code == 404


# ── DELETE /knowledge/{id} ────────────────────────────────────────────


async def test_delete_returns_ack(
    client: TestClient,
    orchestrator: AsyncMock,
) -> None:
    orchestrator.delete.return_value = True

    resp = client.delete("/knowledge/kn-1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"] == {"deleted": True}


async def test_delete_missing_returns_404(
    client: TestClient,
    orchestrator: AsyncMock,
) -> None:
    orchestrator.delete.side_effect = NotFoundError(
        code="knowledge.not_found", message="knowledge not found"
    )

    resp = client.delete("/knowledge/nope")

    assert resp.status_code == 404


# ── POST /knowledge/{id}/reparse ──────────────────────────────────────


async def test_reparse_returns_submitted_ack(
    client: TestClient,
    orchestrator: AsyncMock,
) -> None:
    orchestrator.reparse.return_value = _knowledge(parse_status="pending")

    resp = client.post("/knowledge/kn-1/reparse")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["message"] == "Knowledge reparse task submitted"
    assert body["data"]["parse_status"] == "pending"


async def test_reparse_forwards_process_config(
    client: TestClient,
    orchestrator: AsyncMock,
) -> None:
    orchestrator.reparse.return_value = _knowledge(parse_status="pending")

    client.post(
        "/knowledge/kn-1/reparse",
        json={"process_config": {"enable_multimodel": True}},
    )

    call = orchestrator.reparse.await_args.kwargs
    assert call["process_overrides"] == {"enable_multimodel": True}


async def test_reparse_missing_returns_404(
    client: TestClient,
    orchestrator: AsyncMock,
) -> None:
    orchestrator.reparse.side_effect = NotFoundError(
        code="knowledge.not_found", message="knowledge not found"
    )

    resp = client.post("/knowledge/nope/reparse")

    assert resp.status_code == 404


# ── POST /knowledge/{id}/cancel-parse ─────────────────────────────────


async def test_cancel_parse_returns_cancelled_ack(
    client: TestClient,
    orchestrator: AsyncMock,
) -> None:
    orchestrator.cancel_parse.return_value = _knowledge(parse_status="cancelled")

    resp = client.post("/knowledge/kn-1/cancel-parse")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["message"] == "Knowledge parse cancelled"
    assert body["data"]["parse_status"] == "cancelled"


async def test_cancel_parse_finished_returns_422(
    client: TestClient,
    orchestrator: AsyncMock,
) -> None:
    orchestrator.cancel_parse.side_effect = ValidationError(
        code="knowledge.parse_not_cancellable", message="parse already finished"
    )

    resp = client.post("/knowledge/kn-1/cancel-parse")

    assert resp.status_code == 422


# ── POST /knowledge/{id}/clone ────────────────────────────────────────


async def test_clone_returns_cloned_document(
    client: TestClient,
    orchestrator: AsyncMock,
) -> None:
    orchestrator.clone.return_value = _knowledge(id="kn-clone", kb_id="kb-2")

    resp = client.post("/knowledge/kn-1/clone", json={"target_kb_id": "kb-2"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["id"] == "kn-clone"
    assert body["data"]["knowledge_base_id"] == "kb-2"


async def test_clone_not_completed_returns_422(
    client: TestClient,
    orchestrator: AsyncMock,
) -> None:
    orchestrator.clone.return_value = None

    resp = client.post("/knowledge/kn-1/clone", json={"target_kb_id": "kb-2"})

    assert resp.status_code == 422


# ── POST /knowledge/move ──────────────────────────────────────────────


async def test_move_returns_task_response(
    client: TestClient,
    orchestrator: AsyncMock,
) -> None:
    orchestrator.move.return_value = _knowledge(kb_id="kb-2")

    resp = client.post(
        "/knowledge/move",
        json={
            "knowledge_ids": ["kn-1"],
            "source_kb_id": KB_ID,
            "target_kb_id": "kb-2",
            "mode": "reparse",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    data = body["data"]
    assert data["knowledge_count"] == 1
    assert data["source_kb_id"] == KB_ID
    assert data["target_kb_id"] == "kb-2"
    assert data["message"] == "Knowledge move task started"
    assert data["task_id"].startswith("kg_move_")
    assert str(TENANT_ID) in data["task_id"].split("_")


async def test_move_same_kb_returns_422(client: TestClient) -> None:
    resp = client.post(
        "/knowledge/move",
        json={
            "knowledge_ids": ["kn-1"],
            "source_kb_id": KB_ID,
            "target_kb_id": KB_ID,
            "mode": "reparse",
        },
    )

    assert resp.status_code == 422


async def test_move_empty_ids_returns_422(client: TestClient) -> None:
    resp = client.post(
        "/knowledge/move",
        json={
            "knowledge_ids": [],
            "source_kb_id": KB_ID,
            "target_kb_id": "kb-2",
            "mode": "reparse",
        },
    )

    assert resp.status_code == 422


# ── GET /knowledge/move/progress/{task_id} ────────────────────────────


def _task_id(tenant_id: int) -> str:
    return f"kg_move_{tenant_id}_1700000000000_aabbccdd_{KB_ID}"


async def test_move_progress_malformed_task_id_returns_422(
    client: TestClient,
) -> None:
    resp = client.get("/knowledge/move/progress/not-a-task")

    assert resp.status_code == 422


async def test_move_progress_cross_tenant_hidden_as_404(
    client: TestClient,
) -> None:
    resp = client.get(f"/knowledge/move/progress/{_task_id(TENANT_ID + 1)}")

    # A cross-workspace task reads as not-found so the task space is not
    # enumerable.
    assert resp.status_code == 404


async def test_move_progress_same_tenant_no_record_returns_404(
    client: TestClient,
) -> None:
    # The tenant guard passes; progress records land with the async task
    # infrastructure, so a well-formed task resolves to no record.
    resp = client.get(f"/knowledge/move/progress/{_task_id(TENANT_ID)}")

    assert resp.status_code == 404
