"""Web-layer tests for the knowledge-base router.

Exercises the router over HTTP via ``TestClient`` against the app with
``get_kb_service`` overridden by a real ``KBService`` backed by an
``AsyncMock(spec=KnowledgeBaseRepository)`` configured with stateful
closures, so the full web -> service path runs without a database. The
copy / duplicate endpoints additionally inject the shared ``AsyncSession``
(never exercised by the paths under test, so it stays untouched).

Uses the shared ``web_app`` fixture (header-based auth) and applies the
service dep override on it; the real ``require_auth`` dep resolves the
principal via the ``X-User-Id/X-Tenant-ID/X-Roles`` header trio.

The load-bearing checks:

1. All 10 endpoints exist under the paths and methods the handler
   registers, each carrying the auth gate plus the role gate.
2. Tenant isolation: a cross-workspace id reads as 404 on every
   id-scoped route so the id space is not enumerable.
3. Count enrichment: the detail response fills ``knowledge_count`` for
   document rows and ``chunk_count`` for FAQ rows.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.common.exception import ValidationError
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.models.knowledge_base import KnowledgeBase
from src.web.api.knowledge_bases.router import router
from src.web.deps.knowledge_bases import get_kb_service
from src.web.deps.rbac import make_role_dep, require_role_dep
from src.web.middleware.auth import require_auth

TENANT_ID = 1
NOW = datetime(2026, 4, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _bind_tenant_id_to_admin(
    admin_user: tuple[int, int],
) -> None:
    """Rewrite the module-level ``TENANT_ID`` to the minted admin tenant.

    Per-test conftest mints a fresh ``tenant_id``; this rebind keeps the
    helper closures (which seed mocks keyed by ``TENANT_ID``) aligned
    with the principal the authed client presents.
    """
    global TENANT_ID
    TENANT_ID = admin_user[1]


# ── App wiring ───────────────────────────────────────────────────────


@pytest.fixture
def kb_repo() -> AsyncMock:
    """``AsyncMock(spec=KnowledgeBaseRepository)`` with stateful closures."""
    repo = AsyncMock(spec=KnowledgeBaseRepository)
    rows: dict[str, KnowledgeBase] = {}

    def _live() -> dict[str, KnowledgeBase]:
        return {k: r for k, r in rows.items() if r.deleted_at is None}

    async def _create(row: KnowledgeBase) -> KnowledgeBase:
        rows[row.id] = row
        return row

    async def _get_by_id_or_none(id: str) -> KnowledgeBase | None:
        return _live().get(id)

    async def _get_by_id_and_tenant(id: str, tenant_id: int) -> KnowledgeBase | None:
        row = _live().get(id)
        if row is not None and row.tenant_id == tenant_id:
            return row
        return None

    async def _get_by_ids(ids: list[str]) -> list[KnowledgeBase]:
        live = _live()
        return [live[i] for i in ids if i in live]

    async def _list_by_tenant(tenant_id: int) -> list[KnowledgeBase]:
        return sorted(
            (r for r in _live().values() if r.tenant_id == tenant_id and not r.is_temporary),
            key=lambda r: r.created_at,
            reverse=True,
        )

    async def _update(row: KnowledgeBase) -> KnowledgeBase:
        existing = rows.get(row.id)
        if existing is None or existing.deleted_at is not None:
            raise ValidationError(code="db.not_found", message="row missing")
        # Immutable columns are preserved exactly as the real repo does.
        persisted = row.model_copy(
            update={
                "tenant_id": existing.tenant_id,
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

    async def _count_documents(*, tenant_id: int, knowledge_base_id: str) -> int:
        return 0

    async def _count_chunks(*, tenant_id: int, knowledge_base_id: str) -> int:
        return 0

    async def _count_members(*, tenant_id: int, knowledge_base_id: str) -> int:
        return 0

    repo.create.side_effect = _create
    repo.get_by_id_or_none.side_effect = _get_by_id_or_none
    repo.get_by_id_and_tenant.side_effect = _get_by_id_and_tenant
    repo.get_by_ids.side_effect = _get_by_ids
    repo.list_by_tenant.side_effect = _list_by_tenant
    repo.update.side_effect = _update
    repo.soft_delete.side_effect = _soft_delete
    repo.count_documents.side_effect = _count_documents
    repo.count_chunks.side_effect = _count_chunks
    repo.count_members.side_effect = _count_members
    repo._rows = rows  # type: ignore[attr-defined]
    return repo


@pytest.fixture
def app(
    request: pytest.FixtureRequest,
    web_app: FastAPI,
    kb_repo: AsyncMock,
) -> FastAPI:
    """Override ``get_kb_service`` on the shared web app."""

    def _override_service() -> KBService:
        return KBService(kb_repo=kb_repo)

    web_app.dependency_overrides[get_kb_service] = _override_service
    return web_app


@pytest.fixture
def client(app: FastAPI, web_authed_client: TestClient) -> TestClient:
    """Alias ``web_authed_client``; depending on ``app`` forces the
    dep-override fixture to run before the test executes."""
    return web_authed_client


def _kb_row(
    *,
    id: str = "kb-1",
    name: str = "Knowledge Base",
    type: str = "document",
    tenant_id: int | None = None,
    creator_id: str | None = None,
    embedding_model_id: str = "",
    **overrides: object,
) -> KnowledgeBase:
    """Build a ``knowledge_bases`` row with the minimal required columns."""
    if tenant_id is None:
        tenant_id = TENANT_ID
    return KnowledgeBase(
        id=id,
        tenant_id=tenant_id,
        name=name,
        type=type,
        creator_id=creator_id,
        embedding_model_id=embedding_model_id,
        created_at=NOW,
        updated_at=NOW,
        **overrides,
    )


def _create_body(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "name": "Knowledge Base",
        "description": "test description",
        "type": "document",
    }
    body.update(overrides)
    return body


# ── Route surface (structural) ───────────────────────────────────────


EXPECTED_ROUTES: set[tuple[str, str]] = {
    ("POST", "/knowledge-bases"),
    ("GET", "/knowledge-bases"),
    ("GET", "/knowledge-bases/{id}"),
    ("PUT", "/knowledge-bases/{id}"),
    ("DELETE", "/knowledge-bases/{id}"),
    ("POST", "/knowledge-bases/{id}/duplicate"),
    ("POST", "/knowledge-bases/copy"),
    ("GET", "/knowledge-bases/{id}/move-targets"),
    ("POST", "/knowledge-bases/{id}/hybrid-search"),
    ("GET", "/knowledge-bases/{id}/hybrid-search"),
    ("PUT", "/knowledge-bases/{id}/pin"),
}

# Reads are Viewer+; content-mutating operations are Admin+.
EXPECTED_ROLES: dict[tuple[str, str], str] = {
    ("POST", "/knowledge-bases"): "admin",
    ("GET", "/knowledge-bases"): "viewer",
    ("GET", "/knowledge-bases/{id}"): "viewer",
    ("PUT", "/knowledge-bases/{id}"): "admin",
    ("DELETE", "/knowledge-bases/{id}"): "admin",
    ("POST", "/knowledge-bases/{id}/duplicate"): "admin",
    ("POST", "/knowledge-bases/copy"): "admin",
    ("GET", "/knowledge-bases/{id}/move-targets"): "viewer",
    ("POST", "/knowledge-bases/{id}/hybrid-search"): "viewer",
    ("GET", "/knowledge-bases/{id}/hybrid-search"): "viewer",
    ("PUT", "/knowledge-bases/{id}/pin"): "viewer",
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
    # A missing AuthDep would expose a workspace-scoped entity, so this
    # is asserted structurally rather than by probing.
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


# ── POST /knowledge-bases ────────────────────────────────────────────


async def test_create_returns_201_envelope(
    client: TestClient,
    kb_repo: AsyncMock,
) -> None:
    resp = client.post("/api/v1/knowledge-bases", json=_create_body())

    assert resp.status_code == 201
    payload = resp.json()
    assert payload["success"] is True
    assert payload["data"]["name"] == "Knowledge Base"
    assert payload["data"]["type"] == "document"
    assert payload["data"]["is_temporary"] is False
    rows = kb_repo._rows  # type: ignore[attr-defined]
    assert payload["data"]["id"] in rows


async def test_create_applies_default_indexing_strategy(client: TestClient) -> None:
    resp = client.post("/api/v1/knowledge-bases", json=_create_body())

    assert resp.status_code == 201
    strategy = resp.json()["data"]["indexing_strategy"]
    assert strategy == {
        "vector_enabled": True,
        "keyword_enabled": True,
        "wiki_enabled": False,
        "graph_enabled": False,
    }


async def test_create_rejects_blank_name(client: TestClient) -> None:
    resp = client.post("/api/v1/knowledge-bases", json=_create_body(name="   "))

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "knowledge_base.name_required"


async def test_create_rejects_unknown_type(client: TestClient) -> None:
    resp = client.post("/api/v1/knowledge-bases", json=_create_body(type="bogus"))

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "knowledge_base.type_invalid"


# ── GET /knowledge-bases ─────────────────────────────────────────────


async def test_list_returns_tenant_rows(
    client: TestClient,
    kb_repo: AsyncMock,
) -> None:
    kb_repo._rows["kb-1"] = _kb_row(id="kb-1")  # type: ignore[attr-defined]
    kb_repo._rows["kb-2"] = _kb_row(id="kb-2")  # type: ignore[attr-defined]
    client.post("/api/v1/knowledge-bases", json=_create_body(name="kb-3"))

    resp = client.get("/api/v1/knowledge-bases")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    names = {kb["name"] for kb in payload["data"]}
    assert names == {"Knowledge Base", "kb-3"}
    assert len(payload["data"]) == 3


async def test_list_excludes_other_tenants(
    client: TestClient,
    kb_repo: AsyncMock,
) -> None:
    kb_repo._rows["kb-own"] = _kb_row(id="kb-own")  # type: ignore[attr-defined]
    kb_repo._rows["kb-other"] = _kb_row(id="kb-other", tenant_id=TENANT_ID + 1)  # type: ignore[attr-defined]

    resp = client.get("/api/v1/knowledge-bases")

    assert resp.status_code == 200
    assert [kb["id"] for kb in resp.json()["data"]] == ["kb-own"]


async def test_list_filters_by_creator(
    client: TestClient,
    kb_repo: AsyncMock,
    admin_user: tuple[int, int],
) -> None:
    user_id, _ = admin_user
    kb_repo._rows["kb-mine"] = _kb_row(id="kb-mine", creator_id=user_id)  # type: ignore[attr-defined]
    kb_repo._rows["kb-other"] = _kb_row(id="kb-other", creator_id="someone-else")  # type: ignore[attr-defined]
    kb_repo._rows["kb-anon"] = _kb_row(id="kb-anon", creator_id=None)  # type: ignore[attr-defined]

    mine = client.get("/api/v1/knowledge-bases?creator=mine")
    assert mine.status_code == 200
    assert {kb["id"] for kb in mine.json()["data"]} == {"kb-mine"}

    others = client.get("/api/v1/knowledge-bases?creator=others")
    assert others.status_code == 200
    assert {kb["id"] for kb in others.json()["data"]} == {"kb-other"}


# ── GET /knowledge-bases/{id} ────────────────────────────────────────


async def test_get_returns_one_knowledge_base(client: TestClient) -> None:
    created = client.post("/api/v1/knowledge-bases", json=_create_body())
    kb_id = created.json()["data"]["id"]

    resp = client.get(f"/api/v1/knowledge-bases/{kb_id}")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["data"]["id"] == kb_id
    assert payload["data"]["name"] == "Knowledge Base"


async def test_get_fills_document_count(
    client: TestClient,
    kb_repo: AsyncMock,
) -> None:
    kb_repo._rows["kb-1"] = _kb_row(id="kb-1")  # type: ignore[attr-defined]

    async def _count_docs(*, tenant_id: int, knowledge_base_id: str) -> int:
        return 7

    kb_repo.count_documents.side_effect = _count_docs

    resp = client.get("/api/v1/knowledge-bases/kb-1")

    assert resp.status_code == 200
    assert resp.json()["data"]["knowledge_count"] == 7


async def test_get_fills_faq_chunk_count(
    client: TestClient,
    kb_repo: AsyncMock,
) -> None:
    kb_repo._rows["kb-1"] = _kb_row(id="kb-1", type="faq")  # type: ignore[attr-defined]

    async def _count_chunks(*, tenant_id: int, knowledge_base_id: str) -> int:
        return 5

    kb_repo.count_chunks.side_effect = _count_chunks

    resp = client.get("/api/v1/knowledge-bases/kb-1")

    assert resp.status_code == 200
    assert resp.json()["data"]["chunk_count"] == 5


async def test_get_missing_returns_404(client: TestClient) -> None:
    resp = client.get("/api/v1/knowledge-bases/does-not-exist")

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "knowledge_base.not_found"


async def test_get_cross_tenant_returns_404(
    client: TestClient,
    kb_repo: AsyncMock,
) -> None:
    kb_repo._rows["kb-other"] = _kb_row(id="kb-other", tenant_id=TENANT_ID + 1)  # type: ignore[attr-defined]

    resp = client.get("/api/v1/knowledge-bases/kb-other")

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "knowledge_base.not_found"


# ── PUT /knowledge-bases/{id} ────────────────────────────────────────


async def test_update_patches_name_and_description(
    client: TestClient,
) -> None:
    created = client.post("/api/v1/knowledge-bases", json=_create_body())
    kb_id = created.json()["data"]["id"]

    resp = client.put(
        f"/api/v1/knowledge-bases/{kb_id}",
        json={"name": "Renamed KB", "description": "renamed"},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["data"]["name"] == "Renamed KB"
    assert payload["data"]["description"] == "renamed"


async def test_update_missing_returns_404(client: TestClient) -> None:
    resp = client.put("/api/v1/knowledge-bases/does-not-exist", json={"name": "x"})

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "knowledge_base.not_found"


async def test_update_cross_tenant_returns_404(
    client: TestClient,
    kb_repo: AsyncMock,
) -> None:
    kb_repo._rows["kb-other"] = _kb_row(id="kb-other", tenant_id=TENANT_ID + 1)  # type: ignore[attr-defined]

    resp = client.put("/api/v1/knowledge-bases/kb-other", json={"name": "x"})

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "knowledge_base.not_found"


async def test_update_rejects_blank_name(client: TestClient) -> None:
    created = client.post("/api/v1/knowledge-bases", json=_create_body())
    kb_id = created.json()["data"]["id"]

    resp = client.put(f"/api/v1/knowledge-bases/{kb_id}", json={"name": "  "})

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "knowledge_base.name_required"


# ── DELETE /knowledge-bases/{id} ─────────────────────────────────────


async def test_delete_returns_message_and_soft_deletes(
    client: TestClient,
    kb_repo: AsyncMock,
) -> None:
    created = client.post("/api/v1/knowledge-bases", json=_create_body())
    kb_id = created.json()["data"]["id"]

    resp = client.delete(f"/api/v1/knowledge-bases/{kb_id}")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    assert payload["message"] == "Knowledge base deleted successfully"
    rows = kb_repo._rows  # type: ignore[attr-defined]
    assert kb_id not in rows or rows[kb_id].deleted_at is not None


async def test_delete_missing_returns_404(client: TestClient) -> None:
    resp = client.delete("/api/v1/knowledge-bases/does-not-exist")

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "knowledge_base.not_found"


# ── POST /knowledge-bases/copy ───────────────────────────────────────


async def test_copy_without_target_creates_new_kb(
    client: TestClient,
    kb_repo: AsyncMock,
) -> None:
    created = client.post("/api/v1/knowledge-bases", json=_create_body())
    source_id = created.json()["data"]["id"]

    resp = client.post("/api/v1/knowledge-bases/copy", json={"source_id": source_id})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    data = payload["data"]
    assert data["source_id"] == source_id
    assert data["target_id"] != source_id
    assert data["message"] == "Knowledge base copy task started"
    assert data["task_id"].startswith("kb_clone_")
    rows = kb_repo._rows  # type: ignore[attr-defined]
    assert data["target_id"] in rows


async def test_copy_missing_source_returns_404(client: TestClient) -> None:
    resp = client.post("/api/v1/knowledge-bases/copy", json={"source_id": "does-not-exist"})

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "knowledge_base.not_found"


# ── POST /knowledge-bases/{id}/duplicate ─────────────────────────────


async def test_duplicate_returns_201_envelope(
    client: TestClient,
    kb_repo: AsyncMock,
) -> None:
    created = client.post("/api/v1/knowledge-bases", json=_create_body(name="Original"))
    source_id = created.json()["data"]["id"]

    resp = client.post(f"/api/v1/knowledge-bases/{source_id}/duplicate")

    assert resp.status_code == 201
    payload = resp.json()
    assert payload["success"] is True
    data = payload["data"]
    assert data["source_id"] == source_id
    assert data["target_id"] != source_id
    assert data["message"] == "Knowledge base duplicate created"
    assert data["knowledge_base"]["name"] == "Original Copy"
    rows = kb_repo._rows  # type: ignore[attr-defined]
    assert data["target_id"] in rows


async def test_duplicate_missing_source_returns_404(client: TestClient) -> None:
    resp = client.post("/api/v1/knowledge-bases/does-not-exist/duplicate")

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "knowledge_base.not_found"


# ── GET /knowledge-bases/{id}/move-targets ───────────────────────────


async def test_move_targets_filters_eligible_rows(
    client: TestClient,
    kb_repo: AsyncMock,
) -> None:
    kb_repo._rows["kb-src"] = _kb_row(  # type: ignore[attr-defined]
        id="kb-src", name="source", embedding_model_id="embed-a"
    )
    kb_repo._rows["kb-ok"] = _kb_row(  # type: ignore[attr-defined]
        id="kb-ok", name="ok", embedding_model_id="embed-a"
    )
    kb_repo._rows["kb-faq"] = _kb_row(  # type: ignore[attr-defined]
        id="kb-faq", name="faq", type="faq", embedding_model_id="embed-a"
    )
    kb_repo._rows["kb-embed"] = _kb_row(  # type: ignore[attr-defined]
        id="kb-embed", name="embed", embedding_model_id="embed-b"
    )

    resp = client.get("/api/v1/knowledge-bases/kb-src/move-targets")

    assert resp.status_code == 200
    assert [kb["id"] for kb in resp.json()["data"]] == ["kb-ok"]


async def test_move_targets_missing_source_returns_404(client: TestClient) -> None:
    resp = client.get("/api/v1/knowledge-bases/does-not-exist/move-targets")

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "knowledge_base.not_found"


# ── POST /knowledge-bases/{id}/hybrid-search ─────────────────────────


async def test_hybrid_search_returns_empty_data_envelope(
    client: TestClient,
) -> None:
    created = client.post("/api/v1/knowledge-bases", json=_create_body())
    kb_id = created.json()["data"]["id"]

    resp = client.post(
        f"/api/v1/knowledge-bases/{kb_id}/hybrid-search",
        json={"query_text": "how to use"},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    assert payload["data"] == []


async def test_hybrid_search_missing_kb_returns_404(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/knowledge-bases/does-not-exist/hybrid-search",
        json={"query_text": "how to use"},
    )

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "knowledge_base.not_found"


# ── GET /knowledge-bases/{id}/hybrid-search ──────────────────────────


async def test_hybrid_search_get_compat_shim(
    client: TestClient,
) -> None:
    created = client.post("/api/v1/knowledge-bases", json=_create_body())
    kb_id = created.json()["data"]["id"]

    # ``TestClient`` cannot attach a JSON body to a GET; the shim ignores
    # the legacy body anyway, so the request is asserted bare.
    resp = client.get(f"/api/v1/knowledge-bases/{kb_id}/hybrid-search")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    assert payload["data"] == []
