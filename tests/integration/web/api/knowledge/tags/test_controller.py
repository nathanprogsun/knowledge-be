"""Web-layer tests for the knowledge-tag router.

Exercises the router over HTTP via ``TestClient`` against the app:
the full HTTP path (routing, serialization, exception mapping) with
the tag-service dependency overridden by a real ``TagService`` built
on stateful ``AsyncMock(spec=...)`` repositories, so no database is
involved.

Uses the shared ``web_app`` fixture (header-based auth) and applies
the tag-service dep override on it; the real ``require_auth`` dep
resolves the principal via the ``x-knowledge-*`` header trio.

The load-bearing checks:

1. All four endpoints exist under the paths and methods upstream
   registers (``/knowledge-bases/{id}/tags`` CRUD).
2. Every endpoint declares the auth gate plus the role gate (reads
   Viewer+, mutations Contributor+), asserted structurally so a
   dropped guard fails the suite.
3. Success payloads match the envelope convention
   (``{"success": true, "data": ...}``) and carry no ``seq_id``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.core.knowledge.tags.service.tag_service import TagService
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.dao.knowledge_tag_repository import TagReferenceCounts, TagRepository
from src.db.models.knowledge_base import KnowledgeBase
from src.db.models.knowledge_tag import KnowledgeTag
from src.web.api.knowledge.tags.router import router
from src.web.deps.knowledge_tags import get_tag_service
from src.web.deps.rbac import make_role_dep, require_role_dep
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
    helper closures (which seed mocks keyed by ``TENANT_ID``) aligned
    with the principal the authed client presents.
    """
    global TENANT_ID
    TENANT_ID = admin_user[1]


# ── App wiring ───────────────────────────────────────────────────────


def _make_tag_repo() -> AsyncMock:
    """``AsyncMock(spec=TagRepository)`` with stateful closures."""
    repo = AsyncMock(spec=TagRepository)
    rows: dict[str, KnowledgeTag] = {}
    refs: dict[str, TagReferenceCounts] = {}

    async def _create(row: KnowledgeTag) -> KnowledgeTag:
        rows[row.id] = row
        return row

    async def _update(row: KnowledgeTag) -> KnowledgeTag:
        rows[row.id] = row
        return row

    async def _get_by_id(tenant_id: int, id: str) -> KnowledgeTag | None:
        row = rows.get(id)
        if row is None or row.tenant_id != tenant_id:
            return None
        return row

    async def _get_by_seq_id(tenant_id: int, seq_id: int) -> KnowledgeTag | None:
        for row in rows.values():
            if row.seq_id == seq_id and row.tenant_id == tenant_id:
                return row
        return None

    async def _get_by_name(
        tenant_id: int,
        knowledge_base_id: str,
        name: str,
    ) -> KnowledgeTag | None:
        for row in rows.values():
            if (
                row.tenant_id == tenant_id
                and row.knowledge_base_id == knowledge_base_id
                and row.name == name
            ):
                return row
        return None

    async def _list_by_kb(
        *,
        tenant_id: int,
        knowledge_base_id: str,
        page: int = 1,
        page_size: int = 20,
        keyword: str = "",
    ) -> tuple[list[KnowledgeTag], int]:
        matches = [
            row
            for row in rows.values()
            if row.tenant_id == tenant_id
            and row.knowledge_base_id == knowledge_base_id
            and (not keyword or keyword in row.name)
        ]
        matches.sort(key=lambda r: (r.sort_order, -r.created_at.timestamp(), -r.seq_id))
        offset = (page - 1) * page_size
        return matches[offset : offset + page_size], len(matches)

    async def _count_references(
        *,
        tenant_id: int,
        knowledge_base_id: str,
        tag_id: str,
    ) -> TagReferenceCounts:
        return refs.get(tag_id, TagReferenceCounts(0, 0))

    async def _batch_count_references(
        *,
        tenant_id: int,
        knowledge_base_id: str,
        tag_ids: list[str],
    ) -> dict[str, TagReferenceCounts]:
        return {tag_id: TagReferenceCounts(0, 0) for tag_id in tag_ids}

    async def _delete(*, tenant_id: int, id: str) -> bool:
        row = rows.get(id)
        if row is None or row.tenant_id != tenant_id:
            return False
        del rows[id]
        return True

    repo.create.side_effect = _create
    repo.update.side_effect = _update
    repo.get_by_id.side_effect = _get_by_id
    repo.get_by_seq_id.side_effect = _get_by_seq_id
    repo.get_by_name.side_effect = _get_by_name
    repo.list_by_kb.side_effect = _list_by_kb
    repo.count_references.side_effect = _count_references
    repo.batch_count_references.side_effect = _batch_count_references
    repo.delete.side_effect = _delete
    repo._rows = rows  # type: ignore[attr-defined]
    repo._refs = refs  # type: ignore[attr-defined]
    return repo


def _make_kb_repo() -> AsyncMock:
    """``AsyncMock(spec=KnowledgeBaseRepository)`` keyed by (id, tenant)."""
    repo = AsyncMock(spec=KnowledgeBaseRepository)
    kbs: dict[tuple[str, int], KnowledgeBase] = {}

    async def _get_by_id_and_tenant(id: str, tenant_id: int) -> KnowledgeBase | None:
        return kbs.get((id, tenant_id))

    repo.get_by_id_and_tenant.side_effect = _get_by_id_and_tenant
    repo._kbs = kbs  # type: ignore[attr-defined]
    return repo


@pytest.fixture
def tag_repo() -> AsyncMock:
    return _make_tag_repo()


@pytest.fixture
def kb_repo() -> AsyncMock:
    return _make_kb_repo()


@pytest.fixture
def owned_kb(kb_repo: AsyncMock) -> KnowledgeBase:
    """Seed the KB row owned by the authed tenant."""
    row = _kb_row()
    kb_repo._kbs[(row.id, row.tenant_id)] = row  # type: ignore[attr-defined]
    return row


@pytest.fixture
def service(tag_repo: AsyncMock, kb_repo: AsyncMock) -> TagService:
    """Real ``TagService`` over the stateful repo mocks."""
    return TagService(tag_repo=tag_repo, kb_repo=kb_repo)


@pytest.fixture
def app(
    web_app: FastAPI,
    service: TagService,
) -> FastAPI:
    """Override ``get_tag_service`` on the shared web app."""
    web_app.dependency_overrides[get_tag_service] = lambda: service
    return web_app


@pytest.fixture
def client(app: FastAPI, web_authed_client: TestClient) -> TestClient:
    """Alias ``web_authed_client``; depending on ``app`` forces the
    dep-override fixture to run before the test executes."""
    return web_authed_client


def _tag_row(
    *,
    tag_id: str = "tag-abc",
    tenant_id: int | None = None,
    knowledge_base_id: str = KB_ID,
    name: str = "infrastructure",
    color: str | None = "#ff0000",
    sort_order: int = 3,
    seq_id: int = 10000001,
) -> KnowledgeTag:
    # ``tenant_id`` default is resolved at call time so the
    # ``_bind_tenant_id_to_admin`` autouse fixture's rebind is honoured.
    if tenant_id is None:
        tenant_id = TENANT_ID
    return KnowledgeTag(
        id=tag_id,
        seq_id=seq_id,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        name=name,
        color=color,
        sort_order=sort_order,
        created_at=NOW,
        updated_at=NOW,
    )


def _kb_row(*, kb_id: str = KB_ID, tenant_id: int | None = None) -> KnowledgeBase:
    if tenant_id is None:
        tenant_id = TENANT_ID
    return KnowledgeBase(
        id=kb_id,
        name="infra-kb",
        type="document",
        is_temporary=False,
        tenant_id=tenant_id,
        created_at=NOW,
        updated_at=NOW,
    )


# ── Route inventory + permission gates ───────────────────────────────

EXPECTED_ROUTES: set[tuple[str, str]] = {
    ("GET", "/knowledge-bases/{id}/tags"),
    ("POST", "/knowledge-bases/{id}/tags"),
    ("PUT", "/knowledge-bases/{id}/tags/{tag_id}"),
    ("DELETE", "/knowledge-bases/{id}/tags/{tag_id}"),
}

# Reads are Viewer+; mutations are Contributor+.
EXPECTED_ROLES: dict[tuple[str, str], str] = {
    ("GET", "/knowledge-bases/{id}/tags"): "viewer",
    ("POST", "/knowledge-bases/{id}/tags"): "contributor",
    ("PUT", "/knowledge-bases/{id}/tags/{tag_id}"): "contributor",
    ("DELETE", "/knowledge-bases/{id}/tags/{tag_id}"): "contributor",
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
    for route in router.routes:
        deps = [d.call for d in getattr(route, "dependant", None).dependencies]  # type: ignore[union-attr]
        assert require_auth in deps, f"{route.path} is missing AuthDep"  # type: ignore[attr-defined]


def test_every_endpoint_declares_the_expected_role_gate() -> None:
    viewer_dep = make_role_dep("viewer")
    contributor_dep = make_role_dep("contributor")
    assert viewer_dep is not contributor_dep

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
    dep = make_role_dep("contributor")
    assert dep.__module__ == require_role_dep.__module__


# ── GET /knowledge-bases/{id}/tags ───────────────────────────────────


async def test_list_returns_enveloped_page(
    client: TestClient,
    tag_repo: AsyncMock,
    owned_kb: KnowledgeBase,
) -> None:
    tag_repo._rows["tag-abc"] = _tag_row()  # type: ignore[attr-defined]

    resp = client.get(f"/knowledge-bases/{owned_kb.id}/tags")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    data = body["data"]
    assert data["total"] == 1
    assert data["page"] == 1
    assert data["page_size"] == 20
    assert data["data"][0]["id"] == "tag-abc"
    assert data["data"][0]["name"] == "infrastructure"
    assert data["data"][0]["knowledge_count"] == 0
    assert data["data"][0]["chunk_count"] == 0
    assert "seq_id" not in data["data"][0]


async def test_list_filters_by_keyword(
    client: TestClient,
    tag_repo: AsyncMock,
    owned_kb: KnowledgeBase,
) -> None:
    rows = tag_repo._rows  # type: ignore[attr-defined]
    rows["t1"] = _tag_row(tag_id="t1", name="networking")
    rows["t2"] = _tag_row(tag_id="t2", name="storage")

    resp = client.get(
        f"/knowledge-bases/{owned_kb.id}/tags",
        params={"keyword": "network"},
    )

    assert resp.status_code == 200
    assert [t["name"] for t in resp.json()["data"]["data"]] == ["networking"]


async def test_list_empty_kb_returns_empty_page(
    client: TestClient,
    owned_kb: KnowledgeBase,
) -> None:
    resp = client.get(f"/knowledge-bases/{owned_kb.id}/tags")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["data"] == []
    assert body["data"]["total"] == 0


async def test_list_cross_tenant_kb_returns_404(client: TestClient) -> None:
    # No KB row is seeded for the authed tenant, so the id is invisible.
    resp = client.get("/knowledge-bases/kb-other/tags")

    assert resp.status_code == 404


async def test_list_rejects_invalid_pagination(client: TestClient) -> None:
    resp = client.get(
        f"/knowledge-bases/{KB_ID}/tags",
        params={"page": 0},
    )

    assert resp.status_code == 422


# ── POST /knowledge-bases/{id}/tags ──────────────────────────────────


async def test_create_returns_enveloped_tag(
    client: TestClient,
    owned_kb: KnowledgeBase,
    default_create_tag_request: dict[str, object],
) -> None:
    resp = client.post(
        f"/knowledge-bases/{owned_kb.id}/tags",
        json=default_create_tag_request,
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    data = body["data"]
    assert data["name"] == "placeholder"
    assert data["tenant_id"] == TENANT_ID
    assert data["knowledge_base_id"] == owned_kb.id
    assert data["sort_order"] == 0
    assert "seq_id" not in data


async def test_create_persists_color_and_sort_order(
    client: TestClient,
    tag_repo: AsyncMock,
    owned_kb: KnowledgeBase,
) -> None:
    resp = client.post(
        f"/knowledge-bases/{owned_kb.id}/tags",
        json={"name": "networking", "color": "#00ff00", "sort_order": 2},
    )

    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["name"] == "networking"
    assert data["color"] == "#00ff00"
    assert data["sort_order"] == 2
    stored = next(iter(tag_repo._rows.values()))  # type: ignore[attr-defined]
    assert stored.name == "networking"
    assert stored.color == "#00ff00"
    assert stored.sort_order == 2


async def test_create_duplicate_name_returns_conflict(
    client: TestClient,
    tag_repo: AsyncMock,
    owned_kb: KnowledgeBase,
) -> None:
    tag_repo._rows["existing"] = _tag_row(tag_id="existing", name="placeholder")  # type: ignore[attr-defined]

    resp = client.post(
        f"/knowledge-bases/{owned_kb.id}/tags",
        json={"name": "placeholder"},
    )

    assert resp.status_code == 409


async def test_create_unknown_kb_returns_404(client: TestClient) -> None:
    resp = client.post("/knowledge-bases/kb-other/tags", json={"name": "x"})

    assert resp.status_code == 404


async def test_create_rejects_missing_name(client: TestClient) -> None:
    resp = client.post(f"/knowledge-bases/{KB_ID}/tags", json={})

    assert resp.status_code == 422


# ── PUT /knowledge-bases/{id}/tags/{tag_id} ──────────────────────────


async def test_update_patches_name_by_uuid(
    client: TestClient,
    tag_repo: AsyncMock,
    owned_kb: KnowledgeBase,
) -> None:
    tag_repo._rows["tag-abc"] = _tag_row()  # type: ignore[attr-defined]

    resp = client.put(
        f"/knowledge-bases/{owned_kb.id}/tags/tag-abc",
        json={"name": "renamed"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["name"] == "renamed"
    assert tag_repo._rows["tag-abc"].name == "renamed"  # type: ignore[attr-defined]


async def test_update_by_seq_id_resolves_to_uuid(
    client: TestClient,
    tag_repo: AsyncMock,
    owned_kb: KnowledgeBase,
) -> None:
    tag_repo._rows["tag-abc"] = _tag_row(seq_id=10000001)  # type: ignore[attr-defined]

    resp = client.put(
        f"/knowledge-bases/{owned_kb.id}/tags/10000001",
        json={"color": "#0000ff"},
    )

    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["id"] == "tag-abc"
    assert data["color"] == "#0000ff"


async def test_update_missing_returns_404(client: TestClient) -> None:
    resp = client.put(
        f"/knowledge-bases/{KB_ID}/tags/nope",
        json={"name": "x"},
    )

    assert resp.status_code == 404


async def test_update_unknown_seq_id_returns_404(client: TestClient) -> None:
    resp = client.put(
        f"/knowledge-bases/{KB_ID}/tags/99999999",
        json={"name": "x"},
    )

    assert resp.status_code == 404


async def test_update_cross_tenant_tag_returns_404(
    client: TestClient,
    tag_repo: AsyncMock,
) -> None:
    tag_repo._rows["theirs"] = _tag_row(tag_id="theirs", tenant_id=99)  # type: ignore[attr-defined]

    resp = client.put(
        f"/knowledge-bases/{KB_ID}/tags/theirs",
        json={"name": "x"},
    )

    assert resp.status_code == 404


# ── DELETE /knowledge-bases/{id}/tags/{tag_id} ───────────────────────


async def test_delete_returns_success_ack_and_removes(
    client: TestClient,
    tag_repo: AsyncMock,
    owned_kb: KnowledgeBase,
) -> None:
    tag_repo._rows["tag-abc"] = _tag_row()  # type: ignore[attr-defined]

    resp = client.delete(f"/knowledge-bases/{owned_kb.id}/tags/tag-abc")

    assert resp.status_code == 200
    assert resp.json() == {"success": True}
    assert "tag-abc" not in tag_repo._rows  # type: ignore[attr-defined]


async def test_delete_referenced_tag_without_force_returns_422(
    client: TestClient,
    tag_repo: AsyncMock,
    owned_kb: KnowledgeBase,
) -> None:
    tag_repo._rows["tag-abc"] = _tag_row()  # type: ignore[attr-defined]
    tag_repo._refs["tag-abc"] = TagReferenceCounts(knowledge_count=2, chunk_count=0)  # type: ignore[attr-defined]

    resp = client.delete(f"/knowledge-bases/{owned_kb.id}/tags/tag-abc")

    assert resp.status_code == 422


async def test_delete_referenced_tag_with_force_succeeds(
    client: TestClient,
    tag_repo: AsyncMock,
    owned_kb: KnowledgeBase,
) -> None:
    tag_repo._rows["tag-abc"] = _tag_row()  # type: ignore[attr-defined]
    tag_repo._refs["tag-abc"] = TagReferenceCounts(knowledge_count=2, chunk_count=0)  # type: ignore[attr-defined]

    resp = client.delete(
        f"/knowledge-bases/{owned_kb.id}/tags/tag-abc",
        params={"force": "true"},
    )

    assert resp.status_code == 200
    assert resp.json() == {"success": True}


async def test_delete_content_only_keeps_tag(
    client: TestClient,
    tag_repo: AsyncMock,
    owned_kb: KnowledgeBase,
) -> None:
    tag_repo._rows["tag-abc"] = _tag_row()  # type: ignore[attr-defined]

    resp = client.delete(
        f"/knowledge-bases/{owned_kb.id}/tags/tag-abc",
        params={"content_only": "true"},
    )

    assert resp.status_code == 200
    assert "tag-abc" in tag_repo._rows  # type: ignore[attr-defined]


async def test_delete_with_exclude_ids_keeps_tag(
    client: TestClient,
    tag_repo: AsyncMock,
    owned_kb: KnowledgeBase,
) -> None:
    tag_repo._rows["tag-abc"] = _tag_row()  # type: ignore[attr-defined]

    resp = client.request(
        "DELETE",
        f"/knowledge-bases/{owned_kb.id}/tags/tag-abc",
        json={"exclude_ids": [1, 2]},
    )

    assert resp.status_code == 200
    assert "tag-abc" in tag_repo._rows  # type: ignore[attr-defined]


async def test_delete_missing_returns_404(client: TestClient) -> None:
    resp = client.delete(f"/knowledge-bases/{KB_ID}/tags/nope")

    assert resp.status_code == 404
