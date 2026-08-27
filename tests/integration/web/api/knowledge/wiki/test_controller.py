"""Web-layer tests for the wiki router.

Exercises the router over HTTP via ``TestClient`` with the wiki service
deps overridden to real services backed by ``AsyncMock(spec=...)``
repositories configured with stateful closures — the full web -> service
path runs without a database.

Uses the shared ``web_app`` fixture (header-based auth) and applies the
service dep overrides on it; the real ``require_auth`` dep resolves the
principal via the ``X-User-Id/X-Tenant-ID/X-Roles`` header trio.

The load-bearing checks:

1. All 17 endpoints exist under the paths and methods the upstream
   handler registers.
2. Every endpoint declares the auth gate plus the role gate upstream
   uses (reads Viewer+, mutations Admin+).
3. Success payloads use the ``{"success": true, "data": ...}`` envelope
   and errors flow through the standard error envelope.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.common.exception import ConflictError, NotFoundError
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.wiki.folders import WikiFolderService
from src.core.knowledge.wiki.lint_service import WikiLintService
from src.core.knowledge.wiki.page_service import WikiPageService
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.dao.wiki_page_repository import WikiFolderRepository, WikiPageRepository
from src.db.models.knowledge_base import KnowledgeBase
from src.db.models.wiki_page import WikiFolder, WikiIndexEntry, WikiPage, WikiPageLite
from src.web.api.knowledge.wiki.router import router
from src.web.deps.knowledge_wiki import (
    get_kb_service,
    get_wiki_folder_service,
    get_wiki_lint_service,
    get_wiki_page_service,
)
from src.web.middleware.auth import require_auth

KB_ID = "kb-1"
NOW = datetime(2026, 5, 1, tzinfo=UTC)

# Rebound per test to the freshly-minted admin tenant by the autouse
# fixture below; page rows and assertions key off it.
TENANT_ID = 1


@pytest.fixture(autouse=True)
def _bind_tenant_id_to_admin(admin_user: tuple[int, int]) -> None:
    """Rebind ``TENANT_ID`` to the minted admin tenant for the current test."""
    global TENANT_ID
    TENANT_ID = admin_user[1]


# ── App wiring ───────────────────────────────────────────────────────


@pytest.fixture
def kb_repo() -> AsyncMock:
    """``AsyncMock(spec=KnowledgeBaseRepository)`` returning a wiki KB."""
    repo = AsyncMock(spec=KnowledgeBaseRepository)
    rows: dict[str, KnowledgeBase] = {KB_ID: _kb_row()}

    async def _get_by_id_or_none(id: str) -> KnowledgeBase | None:
        return rows.get(id)

    repo.get_by_id_or_none.side_effect = _get_by_id_or_none
    repo._rows = rows  # type: ignore[attr-defined]
    return repo


@pytest.fixture
def page_repo() -> AsyncMock:
    """``AsyncMock(spec=WikiPageRepository)`` with stateful closures."""
    repo = AsyncMock(spec=WikiPageRepository)
    rows: dict[str, WikiPage] = {}

    def _live() -> dict[str, WikiPage]:
        return {pid: r for pid, r in rows.items() if r.deleted_at is None}

    def _for_kb() -> list[WikiPage]:
        return [r for r in _live().values() if r.knowledge_base_id == KB_ID]

    async def _create(row: WikiPage) -> WikiPage:
        rows[row.id] = row
        return row

    async def _get_by_slug_or_none(*, knowledge_base_id: str, slug: str) -> WikiPage | None:
        for page in _live().values():
            if page.knowledge_base_id == knowledge_base_id and page.slug == slug:
                return page
        return None

    async def _get_by_id_or_none(id: str) -> WikiPage | None:
        return _live().get(id)

    async def _update(*, row: WikiPage, now: datetime) -> WikiPage:
        existing = rows.get(row.id)
        if existing is None or existing.deleted_at is not None:
            raise NotFoundError(code="wiki.page_not_found", message="page missing")
        if existing.version != row.version:
            raise ConflictError(code="wiki.page_conflict", message="version mismatch")
        updated = row.model_copy(update={"version": row.version + 1, "updated_at": now})
        rows[row.id] = updated
        return updated

    async def _update_meta(*, row: WikiPage, now: datetime) -> WikiPage:
        existing = rows.get(row.id)
        if existing is None or existing.deleted_at is not None:
            raise NotFoundError(code="wiki.page_not_found", message="page missing")
        updated = row.model_copy(update={"updated_at": now})
        rows[row.id] = updated
        return updated

    async def _soft_delete_by_slug(*, knowledge_base_id: str, slug: str, now: datetime) -> bool:
        for pid, page in rows.items():
            if (
                page.knowledge_base_id == knowledge_base_id
                and page.slug == slug
                and page.deleted_at is None
            ):
                rows[pid] = page.model_copy(update={"deleted_at": now, "updated_at": now})
                return True
        return False

    async def _list_pages(
        *,
        knowledge_base_id: str,
        page_types: list[str] | None = None,
        status: str = "",
        query: str = "",
        folder_id: str | None = None,
        category_depth: int | None = None,
        category_path: list[str] | None = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "",
        sort_order: str = "desc",
    ) -> tuple[list[WikiPage], int]:
        candidates = [r for r in _for_kb() if r.knowledge_base_id == knowledge_base_id]
        if page_types:
            candidates = [r for r in candidates if r.page_type in page_types]
        if status:
            candidates = [r for r in candidates if r.status == status]
        if query:
            needle = query.lower()
            candidates = [
                r
                for r in candidates
                if needle in r.title.lower()
                or needle in r.content.lower()
                or needle in r.summary.lower()
            ]
        if folder_id is not None:
            candidates = [r for r in candidates if r.folder_id == folder_id]
        candidates.sort(key=lambda r: r.updated_at, reverse=True)
        total = len(candidates)
        page = max(1, page)
        page_size = page_size if page_size >= 1 else 20
        window = candidates[(page - 1) * page_size : page * page_size]
        return window, total

    async def _list_all(*, knowledge_base_id: str) -> list[WikiPage]:
        return [r for r in _for_kb() if r.knowledge_base_id == knowledge_base_id]

    async def _list_all_slugs(*, knowledge_base_id: str) -> list[str]:
        return [r.slug for r in _for_kb() if r.knowledge_base_id == knowledge_base_id]

    async def _search(*, knowledge_base_id: str, query: str, limit: int = 10) -> list[WikiPage]:
        needle = query.lower()
        hits = [r for r in _for_kb() if needle in r.title.lower() or needle in r.content.lower()]
        return hits[:limit]

    async def _count_by_type(*, knowledge_base_id: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for page in _for_kb():
            counts[page.page_type] = counts.get(page.page_type, 0) + 1
        return counts

    async def _count_orphans(*, knowledge_base_id: str) -> int:
        return sum(1 for page in _for_kb() if not page.in_links)

    async def _list_by_type_light(
        *,
        knowledge_base_id: str,
        page_type: str,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[WikiIndexEntry], int]:
        entries = [
            WikiIndexEntry(
                slug=page.slug,
                title=page.title,
                summary=page.summary,
                parent_slug=page.parent_slug,
                category_path=list(page.category_path),
                wiki_path=page.wiki_path,
                depth=page.depth,
                sort_order=page.sort_order,
            )
            for page in _for_kb()
            if page.page_type == page_type and page.status != "archived"
        ]
        entries.sort(key=lambda e: e.wiki_path)
        total = len(entries)
        return entries[offset : offset + limit], total

    async def _list_pages_cursor(
        *, knowledge_base_id: str, cursor: str = "", limit: int = 100
    ) -> tuple[list[WikiPage], str]:
        ordered = sorted(_for_kb(), key=lambda r: r.id)
        start = int(cursor) if cursor else 0
        window = ordered[start : start + limit]
        next_cursor = str(start + limit) if len(window) == limit else ""
        return window, next_cursor

    async def _list_by_slugs(
        *, knowledge_base_id: str, slugs: list[str]
    ) -> dict[str, WikiPageLite]:
        out: dict[str, WikiPageLite] = {}
        for page in _for_kb():
            if page.slug in slugs:
                out[page.slug] = WikiPageLite(
                    slug=page.slug,
                    title=page.title,
                    page_type=page.page_type,
                    status=page.status,
                )
        return out

    async def _exists_slugs(*, knowledge_base_id: str, slugs: list[str]) -> dict[str, bool]:
        live_slugs = {page.slug for page in _for_kb()}
        return {slug: slug in live_slugs for slug in slugs}

    async def _count_pages_by_folder(
        *, knowledge_base_id: str, page_types: list[str] | None = None
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for page in _for_kb():
            if page_types and page.page_type not in page_types:
                continue
            counts[page.folder_id] = counts.get(page.folder_id, 0) + 1
        return counts

    async def _list_pages_by_folder_ids(
        *, knowledge_base_id: str, folder_ids: list[str]
    ) -> list[WikiPage]:
        return [page for page in _for_kb() if page.folder_id in folder_ids]

    repo.create.side_effect = _create
    repo.get_by_slug_or_none.side_effect = _get_by_slug_or_none
    repo.get_by_id_or_none.side_effect = _get_by_id_or_none
    repo.update.side_effect = _update
    repo.update_meta.side_effect = _update_meta
    repo.soft_delete_by_slug.side_effect = _soft_delete_by_slug
    repo.list_pages.side_effect = _list_pages
    repo.list_all.side_effect = _list_all
    repo.list_all_slugs.side_effect = _list_all_slugs
    repo.search.side_effect = _search
    repo.count_by_type.side_effect = _count_by_type
    repo.count_orphans.side_effect = _count_orphans
    repo.list_by_type_light.side_effect = _list_by_type_light
    repo.list_pages_cursor.side_effect = _list_pages_cursor
    repo.list_by_slugs.side_effect = _list_by_slugs
    repo.exists_slugs.side_effect = _exists_slugs
    repo.count_pages_by_folder.side_effect = _count_pages_by_folder
    repo.list_pages_by_folder_ids.side_effect = _list_pages_by_folder_ids
    repo._rows = rows  # type: ignore[attr-defined]
    return repo


@pytest.fixture
def folder_repo() -> AsyncMock:
    """``AsyncMock(spec=WikiFolderRepository)`` with stateful closures."""
    repo = AsyncMock(spec=WikiFolderRepository)
    rows: dict[str, WikiFolder] = {}

    def _live() -> dict[str, WikiFolder]:
        return {fid: r for fid, r in rows.items() if r.deleted_at is None}

    async def _get_by_id_or_none(*, knowledge_base_id: str, id: str) -> WikiFolder | None:
        return _live().get(id)

    async def _get_child_by_name_or_none(
        *, knowledge_base_id: str, parent_id: str, name: str
    ) -> WikiFolder | None:
        for folder in _live().values():
            if (
                folder.knowledge_base_id == knowledge_base_id
                and folder.parent_id == parent_id
                and folder.name == name
            ):
                return folder
        return None

    async def _create(row: WikiFolder) -> WikiFolder:
        rows[row.id] = row
        return row

    async def _update(*, row: WikiFolder, now: datetime) -> WikiFolder:
        existing = rows.get(row.id)
        if existing is None or existing.deleted_at is not None:
            raise NotFoundError(code="wiki.folder_not_found", message="folder missing")
        updated = row.model_copy(update={"updated_at": now})
        rows[row.id] = updated
        return updated

    async def _delete(*, knowledge_base_id: str, id: str, now: datetime) -> None:
        existing = rows.get(id)
        if existing is None or existing.deleted_at is not None:
            raise NotFoundError(code="wiki.folder_not_found", message="folder missing")
        rows[id] = existing.model_copy(update={"deleted_at": now, "updated_at": now})

    async def _list_all(*, knowledge_base_id: str) -> list[WikiFolder]:
        return [
            r
            for r in rows.values()
            if r.knowledge_base_id == knowledge_base_id and r.deleted_at is None
        ]

    repo.get_by_id_or_none.side_effect = _get_by_id_or_none
    repo.get_child_by_name_or_none.side_effect = _get_child_by_name_or_none
    repo.create.side_effect = _create
    repo.update.side_effect = _update
    repo.delete.side_effect = _delete
    repo.list_all.side_effect = _list_all
    repo._rows = rows  # type: ignore[attr-defined]
    return repo


@pytest.fixture
def kb_service(kb_repo: AsyncMock) -> KBService:
    return KBService(kb_repo=kb_repo)


@pytest.fixture
def wiki_service(
    page_repo: AsyncMock,
    folder_repo: AsyncMock,
) -> WikiPageService:
    return WikiPageService(page_repo=page_repo, folder_repo=folder_repo)


@pytest.fixture
def folder_service(
    folder_repo: AsyncMock,
    page_repo: AsyncMock,
) -> WikiFolderService:
    return WikiFolderService(folder_repo=folder_repo, page_repo=page_repo)


@pytest.fixture
def lint_service(
    wiki_service: WikiPageService,
    kb_service: KBService,
) -> WikiLintService:
    return WikiLintService(wiki_service=wiki_service, kb_service=kb_service)


@pytest.fixture
def app(
    request: pytest.FixtureRequest,
    web_app: FastAPI,
    kb_service: KBService,
    wiki_service: WikiPageService,
    folder_service: WikiFolderService,
    lint_service: WikiLintService,
) -> FastAPI:
    """Override the wiki service deps on the shared web app."""
    web_app.dependency_overrides[get_kb_service] = lambda: kb_service
    web_app.dependency_overrides[get_wiki_page_service] = lambda: wiki_service
    web_app.dependency_overrides[get_wiki_folder_service] = lambda: folder_service
    web_app.dependency_overrides[get_wiki_lint_service] = lambda: lint_service
    return web_app


@pytest.fixture
def client(app: FastAPI, web_authed_client: TestClient) -> TestClient:
    """Alias ``web_authed_client``; depending on ``app`` forces the
    dep-override fixture to run before the test executes."""
    return web_authed_client


def _kb_row() -> KnowledgeBase:
    return KnowledgeBase(
        id=KB_ID,
        name="Wiki KB",
        type="wiki",
        tenant_id=TENANT_ID,
        indexing_strategy={"wiki_enabled": True},
        created_at=NOW,
        updated_at=NOW,
    )


def _page(**overrides: object) -> WikiPage:
    defaults: dict[str, object] = {
        "id": "page-1",
        "tenant_id": TENANT_ID,
        "knowledge_base_id": KB_ID,
        "slug": "entity/acme-corp",
        "title": "Acme Corp",
        "page_type": "entity",
        "status": "published",
        "content": "# Acme Corp\n\nA fictional company.",
        "version": 1,
        "created_at": NOW,
        "updated_at": NOW,
    }
    defaults.update(overrides)
    return WikiPage(**defaults)  # type: ignore[arg-type]


def _folder(**overrides: object) -> WikiFolder:
    defaults: dict[str, object] = {
        "id": "folder-1",
        "tenant_id": TENANT_ID,
        "knowledge_base_id": KB_ID,
        "parent_id": "",
        "name": "Products",
        "path": "Products",
        "depth": 1,
        "created_at": NOW,
        "updated_at": NOW,
    }
    defaults.update(overrides)
    return WikiFolder(**defaults)  # type: ignore[arg-type]


# ── Route inventory + permission gates ───────────────────────────────

EXPECTED_ROUTES: set[tuple[str, str]] = {
    ("GET", "/api/v1/knowledgebase/{kb_id}/wiki/pages"),
    ("POST", "/api/v1/knowledgebase/{kb_id}/wiki/pages"),
    ("GET", "/api/v1/knowledgebase/{kb_id}/wiki/pages/{slug:path}"),
    ("PUT", "/api/v1/knowledgebase/{kb_id}/wiki/pages/{slug:path}"),
    ("DELETE", "/api/v1/knowledgebase/{kb_id}/wiki/pages/{slug:path}"),
    ("PUT", "/api/v1/knowledgebase/{kb_id}/wiki/move-page"),
    ("GET", "/api/v1/knowledgebase/{kb_id}/wiki/folders"),
    ("POST", "/api/v1/knowledgebase/{kb_id}/wiki/folders"),
    ("PUT", "/api/v1/knowledgebase/{kb_id}/wiki/folders/{folder_id}"),
    ("DELETE", "/api/v1/knowledgebase/{kb_id}/wiki/folders/{folder_id}"),
    ("GET", "/api/v1/knowledgebase/{kb_id}/wiki/index"),
    ("GET", "/api/v1/knowledgebase/{kb_id}/wiki/graph"),
    ("GET", "/api/v1/knowledgebase/{kb_id}/wiki/stats"),
    ("GET", "/api/v1/knowledgebase/{kb_id}/wiki/search"),
    ("POST", "/api/v1/knowledgebase/{kb_id}/wiki/rebuild-links"),
    ("GET", "/api/v1/knowledgebase/{kb_id}/wiki/lint"),
    ("POST", "/api/v1/knowledgebase/{kb_id}/wiki/auto-fix"),
}

# Reads are Viewer+; content mutations and maintenance are Admin+.
EXPECTED_ROLES: dict[tuple[str, str], str] = {
    ("GET", "/api/v1/knowledgebase/{kb_id}/wiki/pages"): "viewer",
    ("POST", "/api/v1/knowledgebase/{kb_id}/wiki/pages"): "admin",
    ("GET", "/api/v1/knowledgebase/{kb_id}/wiki/pages/{slug:path}"): "viewer",
    ("PUT", "/api/v1/knowledgebase/{kb_id}/wiki/pages/{slug:path}"): "admin",
    ("DELETE", "/api/v1/knowledgebase/{kb_id}/wiki/pages/{slug:path}"): "admin",
    ("PUT", "/api/v1/knowledgebase/{kb_id}/wiki/move-page"): "admin",
    ("GET", "/api/v1/knowledgebase/{kb_id}/wiki/folders"): "viewer",
    ("POST", "/api/v1/knowledgebase/{kb_id}/wiki/folders"): "admin",
    ("PUT", "/api/v1/knowledgebase/{kb_id}/wiki/folders/{folder_id}"): "admin",
    ("DELETE", "/api/v1/knowledgebase/{kb_id}/wiki/folders/{folder_id}"): "admin",
    ("GET", "/api/v1/knowledgebase/{kb_id}/wiki/index"): "viewer",
    ("GET", "/api/v1/knowledgebase/{kb_id}/wiki/graph"): "viewer",
    ("GET", "/api/v1/knowledgebase/{kb_id}/wiki/stats"): "viewer",
    ("GET", "/api/v1/knowledgebase/{kb_id}/wiki/search"): "viewer",
    ("POST", "/api/v1/knowledgebase/{kb_id}/wiki/rebuild-links"): "admin",
    ("GET", "/api/v1/knowledgebase/{kb_id}/wiki/lint"): "viewer",
    ("POST", "/api/v1/knowledgebase/{kb_id}/wiki/auto-fix"): "admin",
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


# ── KB gate ──────────────────────────────────────────────────────────


async def test_missing_kb_returns_404(client: TestClient, kb_repo: AsyncMock) -> None:
    kb_repo._rows.pop("kb-1", None)  # type: ignore[attr-defined]

    resp = client.get("/api/v1/knowledgebase/nope/wiki/pages")

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "knowledge_base.not_found"


async def test_non_wiki_kb_returns_422(client: TestClient, kb_repo: AsyncMock) -> None:
    row = _kb_row()
    kb_repo._rows[KB_ID] = row.model_copy(  # type: ignore[attr-defined]
        update={"indexing_strategy": {"wiki_enabled": False}}
    )

    resp = client.get(f"/api/v1/knowledgebase/{KB_ID}/wiki/pages")

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "wiki.kb_wiki_not_enabled"


# ── GET /wiki/pages ──────────────────────────────────────────────────


async def test_list_pages_returns_envelope(client: TestClient, page_repo: AsyncMock) -> None:
    page_repo._rows["page-1"] = _page()  # type: ignore[attr-defined]
    page_repo._rows["page-2"] = _page(  # type: ignore[attr-defined]
        id="page-2", slug="concept/rag", title="RAG", page_type="concept"
    )

    resp = client.get(f"/api/v1/knowledgebase/{KB_ID}/wiki/pages")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["total"] == 2
    assert [p["slug"] for p in body["data"]["pages"]] == [
        "entity/acme-corp",
        "concept/rag",
    ]
    assert body["data"]["page"] == 1
    assert body["data"]["page_size"] == 20
    assert body["data"]["total_pages"] == 1


async def test_list_pages_filters_by_status(client: TestClient, page_repo: AsyncMock) -> None:
    page_repo._rows["page-1"] = _page()  # type: ignore[attr-defined]
    page_repo._rows["page-2"] = _page(  # type: ignore[attr-defined]
        id="page-2", slug="concept/rag", title="RAG", page_type="concept", status="archived"
    )

    resp = client.get(f"/api/v1/knowledgebase/{KB_ID}/wiki/pages", params={"status": "archived"})

    assert resp.status_code == 200
    assert [p["slug"] for p in resp.json()["data"]["pages"]] == ["concept/rag"]


# ── POST /wiki/pages ─────────────────────────────────────────────────


async def test_create_page_returns_201(
    client: TestClient,
    page_repo: AsyncMock,
    default_create_wiki_request: dict[str, object],
) -> None:
    resp = client.post(
        f"/api/v1/knowledgebase/{KB_ID}/wiki/pages",
        json=default_create_wiki_request,
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["success"] is True
    data = body["data"]
    assert data["slug"] == "entity/acme-corp"
    assert data["page_type"] == "entity"
    assert data["status"] == "published"
    assert data["tenant_id"] == TENANT_ID
    assert data["knowledge_base_id"] == KB_ID
    rows = page_repo._rows  # type: ignore[attr-defined]
    assert data["id"] in rows


async def test_create_page_rejects_invalid_page_type(
    client: TestClient,
    default_create_wiki_request: dict[str, object],
) -> None:
    body = dict(default_create_wiki_request, page_type="bogus")

    resp = client.post(f"/api/v1/knowledgebase/{KB_ID}/wiki/pages", json=body)

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "wiki.page_invalid_type"


async def test_create_page_requires_slug(
    client: TestClient,
    default_create_wiki_request: dict[str, object],
) -> None:
    body = {k: v for k, v in default_create_wiki_request.items() if k != "slug"}

    resp = client.post(f"/api/v1/knowledgebase/{KB_ID}/wiki/pages", json=body)

    assert resp.status_code == 422


# ── GET /wiki/pages/{slug} ───────────────────────────────────────────


async def test_get_page_returns_page(client: TestClient, page_repo: AsyncMock) -> None:
    page_repo._rows["page-1"] = _page()  # type: ignore[attr-defined]

    resp = client.get(f"/api/v1/knowledgebase/{KB_ID}/wiki/pages/entity/acme-corp")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["slug"] == "entity/acme-corp"
    assert body["data"]["title"] == "Acme Corp"


async def test_get_page_missing_returns_404(client: TestClient) -> None:
    resp = client.get(f"/api/v1/knowledgebase/{KB_ID}/wiki/pages/entity/nope")

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "wiki.page_not_found"


# ── PUT /wiki/pages/{slug} ───────────────────────────────────────────


async def test_update_page_patches_fields(
    client: TestClient,
    page_repo: AsyncMock,
) -> None:
    page_repo._rows["page-1"] = _page()  # type: ignore[attr-defined]

    resp = client.put(
        f"/api/v1/knowledgebase/{KB_ID}/wiki/pages/entity/acme-corp",
        json={"title": "Acme Renamed"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["title"] == "Acme Renamed"
    assert body["data"]["version"] == 2


async def test_update_page_version_conflict_returns_409(
    client: TestClient,
    page_repo: AsyncMock,
) -> None:
    page_repo._rows["page-1"] = _page()  # type: ignore[attr-defined]

    resp = client.put(
        f"/api/v1/knowledgebase/{KB_ID}/wiki/pages/entity/acme-corp",
        json={"title": "x", "version": 99},
    )

    assert resp.status_code == 409
    error = resp.json()["error"]
    assert error["code"] == "wiki.page_conflict"
    assert "current_version" in error["details"]


async def test_update_page_missing_returns_404(client: TestClient) -> None:
    resp = client.put(
        f"/api/v1/knowledgebase/{KB_ID}/wiki/pages/entity/nope",
        json={"title": "x"},
    )

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "wiki.page_not_found"


# ── DELETE /wiki/pages/{slug} ────────────────────────────────────────


async def test_delete_page_returns_204(client: TestClient, page_repo: AsyncMock) -> None:
    page_repo._rows["page-1"] = _page()  # type: ignore[attr-defined]

    resp = client.delete(f"/api/v1/knowledgebase/{KB_ID}/wiki/pages/entity/acme-corp")

    assert resp.status_code == 204
    assert page_repo._rows["page-1"].deleted_at is not None  # type: ignore[attr-defined]


async def test_delete_page_missing_returns_404(client: TestClient) -> None:
    resp = client.delete(f"/api/v1/knowledgebase/{KB_ID}/wiki/pages/entity/nope")

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "wiki.page_not_found"


# ── PUT /wiki/move-page ──────────────────────────────────────────────


async def test_move_page_returns_page(
    client: TestClient,
    page_repo: AsyncMock,
    folder_repo: AsyncMock,
) -> None:
    page_repo._rows["page-1"] = _page()  # type: ignore[attr-defined]
    folder_repo._rows["folder-9"] = _folder(  # type: ignore[attr-defined]
        id="folder-9", name="Research", path="Research", depth=1
    )

    resp = client.put(
        f"/api/v1/knowledgebase/{KB_ID}/wiki/move-page",
        json={"slug": "entity/acme-corp", "folder_id": "folder-9"},
    )

    assert resp.status_code == 200
    assert resp.json()["data"]["slug"] == "entity/acme-corp"
    assert page_repo._rows["page-1"].folder_id == "folder-9"  # type: ignore[attr-defined]


async def test_move_page_requires_slug(client: TestClient) -> None:
    resp = client.put(
        f"/api/v1/knowledgebase/{KB_ID}/wiki/move-page",
        json={"slug": "   "},
    )

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "wiki.page_slug_required"


# ── GET /wiki/folders ────────────────────────────────────────────────


async def test_list_folders_returns_nodes(
    client: TestClient,
    folder_repo: AsyncMock,
    page_repo: AsyncMock,
) -> None:
    folder_repo._rows["folder-1"] = _folder()  # type: ignore[attr-defined]
    page_repo._rows["page-1"] = _page(folder_id="folder-1")  # type: ignore[attr-defined]

    resp = client.get(f"/api/v1/knowledgebase/{KB_ID}/wiki/folders")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["parent_id"] == ""
    folders = body["data"]["folders"]
    assert len(folders) == 1
    assert folders[0]["name"] == "Products"
    assert folders[0]["page_count"] == 1
    assert folders[0]["has_children"] is False


# ── POST /wiki/folders ───────────────────────────────────────────────


async def test_create_folder_returns_201(client: TestClient, folder_repo: AsyncMock) -> None:
    resp = client.post(
        f"/api/v1/knowledgebase/{KB_ID}/wiki/folders",
        json={"name": "Research"},
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["name"] == "Research"
    assert body["data"]["tenant_id"] == TENANT_ID


async def test_create_folder_conflict_returns_409(
    client: TestClient,
    folder_repo: AsyncMock,
) -> None:
    folder_repo._rows["folder-1"] = _folder(name="Research", path="Research")  # type: ignore[attr-defined]

    resp = client.post(
        f"/api/v1/knowledgebase/{KB_ID}/wiki/folders",
        json={"name": "Research"},
    )

    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "wiki.folder_conflict"


# ── PUT /wiki/folders/{folder_id} ────────────────────────────────────


async def test_update_folder_renames(client: TestClient, folder_repo: AsyncMock) -> None:
    folder_repo._rows["folder-1"] = _folder()  # type: ignore[attr-defined]

    resp = client.put(
        f"/api/v1/knowledgebase/{KB_ID}/wiki/folders/folder-1",
        json={"name": "Products v2"},
    )

    assert resp.status_code == 200
    assert resp.json()["data"]["name"] == "Products v2"


# ── DELETE /wiki/folders/{folder_id} ─────────────────────────────────


async def test_delete_folder_returns_204(client: TestClient, folder_repo: AsyncMock) -> None:
    folder_repo._rows["folder-1"] = _folder()  # type: ignore[attr-defined]

    resp = client.delete(f"/api/v1/knowledgebase/{KB_ID}/wiki/folders/folder-1")

    assert resp.status_code == 204
    assert folder_repo._rows["folder-1"].deleted_at is not None  # type: ignore[attr-defined]


async def test_delete_folder_missing_returns_404(client: TestClient) -> None:
    resp = client.delete(f"/api/v1/knowledgebase/{KB_ID}/wiki/folders/nope")

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "wiki.folder_not_found"


# ── GET /wiki/index ──────────────────────────────────────────────────


async def test_get_index_returns_groups(
    client: TestClient,
    page_repo: AsyncMock,
) -> None:
    page_repo._rows["page-1"] = _page()  # type: ignore[attr-defined]
    page_repo._rows["page-2"] = _page(  # type: ignore[attr-defined]
        id="page-2", slug="concept/rag", title="RAG", page_type="concept"
    )

    resp = client.get(f"/api/v1/knowledgebase/{KB_ID}/wiki/index")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    data = body["data"]
    assert data["version"] == 1
    groups = {g["type"]: g for g in data["groups"]}
    assert groups["entity"]["total"] == 1
    assert groups["concept"]["total"] == 1
    assert groups["entity"]["items"][0]["slug"] == "entity/acme-corp"


# ── GET /wiki/graph ──────────────────────────────────────────────────


async def test_get_graph_overview_returns_nodes(
    client: TestClient,
    page_repo: AsyncMock,
) -> None:
    page_repo._rows["page-1"] = _page(  # type: ignore[attr-defined]
        out_links=["concept/rag"]
    )
    page_repo._rows["page-2"] = _page(  # type: ignore[attr-defined]
        id="page-2",
        slug="concept/rag",
        title="RAG",
        page_type="concept",
        in_links=["entity/acme-corp"],
    )

    resp = client.get(f"/api/v1/knowledgebase/{KB_ID}/wiki/graph")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    data = body["data"]
    assert data["meta"]["mode"] == "overview"
    assert {n["slug"] for n in data["nodes"]} == {"entity/acme-corp", "concept/rag"}
    assert {"source": "entity/acme-corp", "target": "concept/rag"} in data["edges"]


async def test_get_graph_invalid_mode_returns_422(client: TestClient) -> None:
    resp = client.get(f"/api/v1/knowledgebase/{KB_ID}/wiki/graph", params={"mode": "bogus"})

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "wiki.graph_invalid_mode"


async def test_get_graph_ego_requires_center(client: TestClient) -> None:
    resp = client.get(f"/api/v1/knowledgebase/{KB_ID}/wiki/graph", params={"mode": "ego"})

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "wiki.graph_center_required"


# ── GET /wiki/stats ──────────────────────────────────────────────────


async def test_get_stats_returns_totals(
    client: TestClient,
    page_repo: AsyncMock,
) -> None:
    page_repo._rows["page-1"] = _page()  # type: ignore[attr-defined]
    page_repo._rows["page-2"] = _page(  # type: ignore[attr-defined]
        id="page-2", slug="concept/rag", title="RAG", page_type="concept"
    )

    resp = client.get(f"/api/v1/knowledgebase/{KB_ID}/wiki/stats")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["total_pages"] == 2
    assert body["data"]["pages_by_type"] == {"entity": 1, "concept": 1}


# ── GET /wiki/search ─────────────────────────────────────────────────


async def test_search_requires_q(client: TestClient) -> None:
    resp = client.get(f"/api/v1/knowledgebase/{KB_ID}/wiki/search")

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "wiki.search_query_required"


async def test_search_returns_pages(
    client: TestClient,
    page_repo: AsyncMock,
) -> None:
    page_repo._rows["page-1"] = _page()  # type: ignore[attr-defined]
    page_repo._rows["page-2"] = _page(  # type: ignore[attr-defined]
        id="page-2", slug="concept/rag", title="RAG", page_type="concept"
    )

    resp = client.get(f"/api/v1/knowledgebase/{KB_ID}/wiki/search", params={"q": "RAG"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert [p["slug"] for p in body["data"]["pages"]] == ["concept/rag"]


# ── POST /wiki/rebuild-links ─────────────────────────────────────────


async def test_rebuild_links_returns_message(
    client: TestClient,
    page_repo: AsyncMock,
) -> None:
    page_repo._rows["page-1"] = _page()  # type: ignore[attr-defined]

    resp = client.post(f"/api/v1/knowledgebase/{KB_ID}/wiki/rebuild-links")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["message"] == "Links rebuilt successfully"


# ── GET /wiki/lint ───────────────────────────────────────────────────


async def test_lint_returns_report(
    client: TestClient,
    page_repo: AsyncMock,
) -> None:
    # A healthy page: long enough body and an inbound link so the lint
    # pass reports zero issues and a perfect health score.
    page_repo._rows["page-1"] = _page(  # type: ignore[attr-defined]
        content=(
            "# Acme Corp\n\n"
            "Acme Corporation is a fictional company that appears in "
            "many cartoons and films. It is known for manufacturing "
            "products that never work as advertised."
        ),
        in_links=["concept/company"],
    )

    resp = client.get(f"/api/v1/knowledgebase/{KB_ID}/wiki/lint")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["knowledge_base_id"] == KB_ID
    assert body["data"]["health_score"] == 100
    assert body["data"]["summary"] == "Wiki is healthy! No issues found."


# ── POST /wiki/auto-fix ──────────────────────────────────────────────


async def test_auto_fix_returns_fixed(
    client: TestClient,
    page_repo: AsyncMock,
) -> None:
    # A healthy page leaves nothing for the fixer to repair.
    page_repo._rows["page-1"] = _page(  # type: ignore[attr-defined]
        content=(
            "# Acme Corp\n\n"
            "Acme Corporation is a fictional company that appears in "
            "many cartoons and films. It is known for manufacturing "
            "products that never work as advertised."
        ),
        in_links=["concept/company"],
    )

    resp = client.post(f"/api/v1/knowledgebase/{KB_ID}/wiki/auto-fix")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["fixed"] == 0
