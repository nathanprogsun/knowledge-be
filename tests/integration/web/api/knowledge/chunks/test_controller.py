"""Web-layer tests for the chunk router.

Exercises the router over HTTP via ``TestClient`` against the app: the
full HTTP path (routing, serialization, exception mapping) with the
chunk service dependencies overridden by ``AsyncMock(spec=...)`` doubles
so no database is involved. Uses the shared ``web_app`` fixture
(header-based auth) and applies the service dep overrides on it.

The load-bearing checks:

1. All 10 endpoints exist under the paths and methods upstream registers.
2. Every endpoint declares the auth gate plus the role gate upstream uses
   (asserted structurally, so a dropped guard fails the suite rather than
   silently opening a credential-bearing route).
3. Mutations verify chunk ownership against the URL ``:knowledge_id`` and
   answer a mismatch with 403 rather than leaking the chunk.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.common.exception import ConflictError, NotFoundError
from src.core.knowledge.chunks.revisions import ChunkRevisionInfo, ChunkRevisionService
from src.core.knowledge.chunks.service.chunk_service import ChunkService
from src.db.models.chunk import Chunk
from src.web.api.knowledge.chunks.router import router
from src.web.deps.chunks import get_chunk_revision_service, get_chunk_service
from src.web.deps.rbac import make_role_dep, require_role_dep
from src.web.middleware.auth import require_auth

TENANT_ID = 1
KB_ID = "kb-1"
KNOWLEDGE_ID = "knowledge-1"
CHUNK_ID = "chunk-1"
NOW = datetime(2026, 4, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _bind_tenant_id_to_admin(
    admin_user: tuple[int, int],
) -> None:
    """Rewrite the module-level ``TENANT_ID`` to the minted admin tenant.

    Per-test conftest mints a fresh ``tenant_id``; this rebind keeps the
    helper closures (which seed rows keyed by ``TENANT_ID``) aligned with
    the principal the authed client presents.
    """
    global TENANT_ID
    TENANT_ID = admin_user[1]


# ── Service doubles ───────────────────────────────────────────────────


@pytest.fixture
def chunk_service() -> AsyncMock:
    """``AsyncMock(spec=ChunkService)`` with stateful closures."""
    service = AsyncMock(spec=ChunkService)
    rows: dict[str, Chunk] = {}

    async def _get_by_id(*, tenant_id: int, id: str) -> Chunk:
        row = rows.get(id)
        if row is None or row.tenant_id != tenant_id or row.deleted_at is not None:
            raise NotFoundError(code="chunk.not_found", message=f"chunk {id} not found")
        return row

    async def _get_by_id_only(*, id: str) -> Chunk:
        row = rows.get(id)
        if row is None or row.deleted_at is not None:
            raise NotFoundError(code="chunk.not_found", message=f"chunk {id} not found")
        return row

    async def _list_by_knowledge_id(
        *,
        tenant_id: int,
        knowledge_id: str,
    ) -> list[Chunk]:
        out = [
            r
            for r in rows.values()
            if r.tenant_id == tenant_id
            and r.knowledge_id == knowledge_id
            and r.deleted_at is None
        ]
        return sorted(out, key=lambda r: r.chunk_index)

    async def _update_document_chunk(
        *,
        tenant_id: int,
        chunk_id: str,
        content: str | None = None,
        is_enabled: bool | None = None,
        expected_revision: int | None = None,
        last_editor_id: str = "",
    ) -> Chunk:
        current = await _get_by_id(tenant_id=tenant_id, id=chunk_id)
        if expected_revision is not None and expected_revision != current.content_revision:
            raise ConflictError(
                code="chunk.revision_conflict",
                message="chunk changed since the expected revision",
            )
        updated = current.model_copy(
            update={
                "content": content if content is not None else current.content,
                "is_enabled": is_enabled if is_enabled is not None else current.is_enabled,
                "content_revision": current.content_revision + 1,
                "last_editor_id": last_editor_id,
                "updated_at": NOW,
            }
        )
        rows[chunk_id] = updated
        return updated

    async def _delete_chunk(*, tenant_id: int, id: str) -> bool:
        row = rows.get(id)
        if row is None or row.tenant_id != tenant_id or row.deleted_at is not None:
            return False
        rows[id] = row.model_copy(update={"deleted_at": NOW, "updated_at": NOW})
        return True

    async def _delete_by_knowledge(*, tenant_id: int, knowledge_id: str) -> int:
        count = 0
        for cid, row in list(rows.items()):
            if (
                row.tenant_id == tenant_id
                and row.knowledge_id == knowledge_id
                and row.deleted_at is None
            ):
                rows[cid] = row.model_copy(update={"deleted_at": NOW, "updated_at": NOW})
                count += 1
        return count

    async def _update_chunk(*, chunk: Chunk) -> Chunk:
        rows[chunk.id] = chunk
        return chunk

    service.get_chunk_by_id.side_effect = _get_by_id
    service.get_chunk_by_id_only.side_effect = _get_by_id_only
    service.list_chunks_by_knowledge_id.side_effect = _list_by_knowledge_id
    service.update_document_chunk.side_effect = _update_document_chunk
    service.delete_chunk.side_effect = _delete_chunk
    service.delete_chunks_by_knowledge_id.side_effect = _delete_by_knowledge
    service.update_chunk.side_effect = _update_chunk
    service._rows = rows  # type: ignore[attr-defined]
    return service


@pytest.fixture
def revision_service() -> AsyncMock:
    """``AsyncMock(spec=ChunkRevisionService)`` with stateful closures."""
    service = AsyncMock(spec=ChunkRevisionService)
    revisions: dict[tuple[str, int], ChunkRevisionInfo] = {}

    async def _list_revisions(*, tenant_id: int, chunk_id: str) -> list[ChunkRevisionInfo]:
        out = [
            r
            for (cid, _revision), r in revisions.items()
            if cid == chunk_id and r.tenant_id == tenant_id
        ]
        return sorted(out, key=lambda r: r.revision, reverse=True)

    async def _get_revision(
        *,
        tenant_id: int,
        chunk_id: str,
        revision: int,
    ) -> ChunkRevisionInfo:
        row = revisions.get((chunk_id, revision))
        if row is None or row.tenant_id != tenant_id:
            raise NotFoundError(
                code="chunk.revision_not_found",
                message=f"chunk revision {revision} not found",
            )
        return row

    service.list_revisions.side_effect = _list_revisions
    service.get_revision.side_effect = _get_revision
    service._revisions = revisions  # type: ignore[attr-defined]
    return service


@pytest.fixture
def app(
    request: pytest.FixtureRequest,
    web_app: FastAPI,
    chunk_service: AsyncMock,
    revision_service: AsyncMock,
) -> FastAPI:
    """Override the chunk service deps on the shared web app."""
    web_app.dependency_overrides[get_chunk_service] = lambda: chunk_service
    web_app.dependency_overrides[get_chunk_revision_service] = lambda: revision_service
    return web_app


@pytest.fixture
def client(app: FastAPI, web_authed_client: TestClient) -> TestClient:
    """Alias ``web_authed_client``; depending on ``app`` forces the
    dep-override fixture to run before the test executes."""
    return web_authed_client


def _chunk(
    *,
    id: str = CHUNK_ID,
    tenant_id: int | None = None,
    knowledge_id: str = KNOWLEDGE_ID,
    content: str = "original content",
    chunk_index: int = 0,
    chunk_type: str = "text",
    content_revision: int = 0,
    is_enabled: bool = True,
    metadata: dict[str, object] | None = None,
) -> Chunk:
    # ``tenant_id`` default is resolved at call time so the
    # ``_bind_tenant_id_to_admin`` autouse fixture's rebind is honoured.
    if tenant_id is None:
        tenant_id = TENANT_ID
    return Chunk(
        id=id,
        tenant_id=tenant_id,
        knowledge_base_id=KB_ID,
        knowledge_id=knowledge_id,
        content=content,
        chunk_index=chunk_index,
        is_enabled=is_enabled,
        start_at=0,
        end_at=len(content),
        chunk_type=chunk_type,
        content_revision=content_revision,
        metadata=metadata,  # type: ignore[arg-type]
        created_at=NOW,
        updated_at=NOW,
    )


def _revision(
    *,
    revision: int,
    chunk_id: str = CHUNK_ID,
    content: str = "snapshot content",
) -> ChunkRevisionInfo:
    return ChunkRevisionInfo(
        id=f"rev-{revision}",
        tenant_id=TENANT_ID,
        knowledge_base_id=KB_ID,
        knowledge_id=KNOWLEDGE_ID,
        chunk_id=chunk_id,
        revision=revision,
        content=content,
        is_enabled=True,
        editor_id="user-1",
        edit_source="user",
        edited_at=NOW,
        created_at=NOW,
    )


# ── Route inventory + permission gates ───────────────────────────────

# Upstream's RegisterChunkRoutes, verbatim.
EXPECTED_ROUTES: set[tuple[str, str]] = {
    ("GET", "/chunks/{knowledge_id}"),
    ("GET", "/chunks/by-id/{id}"),
    ("GET", "/chunks/{knowledge_id}/{id}/revisions"),
    ("DELETE", "/chunks/{knowledge_id}/{id}"),
    ("DELETE", "/chunks/{knowledge_id}"),
    ("PUT", "/chunks/{knowledge_id}/{id}"),
    ("POST", "/chunks/{knowledge_id}/{id}/revert"),
    ("DELETE", "/chunks/by-id/{id}/questions"),
    ("PUT", "/chunks/by-id/{id}/questions"),
    ("POST", "/chunks/by-id/{id}/questions/regenerate"),
}

# Reads are Viewer+; every mutation is Admin+ (closest role-level
# approximation of the upstream KB-owner-or-Admin gate).
EXPECTED_ROLES: dict[tuple[str, str], str] = {
    ("GET", "/chunks/{knowledge_id}"): "viewer",
    ("GET", "/chunks/by-id/{id}"): "viewer",
    ("GET", "/chunks/{knowledge_id}/{id}/revisions"): "viewer",
    ("DELETE", "/chunks/{knowledge_id}/{id}"): "admin",
    ("DELETE", "/chunks/{knowledge_id}"): "admin",
    ("PUT", "/chunks/{knowledge_id}/{id}"): "admin",
    ("POST", "/chunks/{knowledge_id}/{id}/revert"): "admin",
    ("DELETE", "/chunks/by-id/{id}/questions"): "admin",
    ("PUT", "/chunks/by-id/{id}/questions"): "admin",
    ("POST", "/chunks/by-id/{id}/questions/regenerate"): "admin",
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
    # A missing AuthDep would expose an endpoint that reads or mutates
    # workspace chunks, so this is asserted structurally.
    for route in router.routes:
        deps = [d.call for d in getattr(route, "dependant", None).dependencies]  # type: ignore[union-attr]
        assert require_auth in deps, f"{route.path} is missing AuthDep"  # type: ignore[attr-defined]


def test_every_endpoint_declares_the_expected_role_gate() -> None:
    viewer_dep = make_role_dep("viewer")
    admin_dep = make_role_dep("admin")
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


# ── GET /chunks/by-id/{id} ───────────────────────────────────────────


async def test_get_chunk_by_id_only_returns_chunk(
    client: TestClient,
    chunk_service: AsyncMock,
) -> None:
    chunk_service._rows[CHUNK_ID] = _chunk()  # type: ignore[attr-defined]

    resp = client.get(f"/chunks/by-id/{CHUNK_ID}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["id"] == CHUNK_ID
    assert body["data"]["knowledge_id"] == KNOWLEDGE_ID
    assert body["data"]["content"] == "original content"


async def test_get_chunk_by_id_only_missing_returns_404(client: TestClient) -> None:
    resp = client.get("/chunks/by-id/nope")

    assert resp.status_code == 404
    assert resp.json()["success"] is False


# ── GET /chunks/{knowledge_id} ───────────────────────────────────────


async def test_list_chunks_returns_paged_envelope(
    client: TestClient,
    chunk_service: AsyncMock,
) -> None:
    for i in range(3):
        chunk_service._rows[f"c{i}"] = _chunk(  # type: ignore[attr-defined]
            id=f"c{i}",
            content=f"content {i}",
            chunk_index=i,
        )

    resp = client.get(f"/chunks/{KNOWLEDGE_ID}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["total"] == 3
    assert body["page"] == 1
    assert body["page_size"] == 10
    assert [c["id"] for c in body["data"]] == ["c0", "c1", "c2"]


async def test_list_chunks_respects_page_bounds(
    client: TestClient,
    chunk_service: AsyncMock,
) -> None:
    for i in range(5):
        chunk_service._rows[f"c{i}"] = _chunk(  # type: ignore[attr-defined]
            id=f"c{i}",
            content=f"content {i}",
            chunk_index=i,
        )

    resp = client.get(f"/chunks/{KNOWLEDGE_ID}", params={"page": 2, "page_size": 2})

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 5
    assert body["page"] == 2
    assert body["page_size"] == 2
    assert [c["id"] for c in body["data"]] == ["c2", "c3"]


async def test_list_chunks_clamps_page_size(
    client: TestClient,
    chunk_service: AsyncMock,
) -> None:
    chunk_service._rows[CHUNK_ID] = _chunk()  # type: ignore[attr-defined]

    resp = client.get(f"/chunks/{KNOWLEDGE_ID}", params={"page_size": 500})

    assert resp.status_code == 200
    assert resp.json()["page_size"] == 100


async def test_list_chunks_filters_chunk_type(
    client: TestClient,
    chunk_service: AsyncMock,
) -> None:
    chunk_service._rows["text-1"] = _chunk(id="text-1", chunk_type="text")  # type: ignore[attr-defined]
    chunk_service._rows["img-1"] = _chunk(  # type: ignore[attr-defined]
        id="img-1",
        chunk_type="image_caption",
    )

    resp = client.get(f"/chunks/{KNOWLEDGE_ID}", params={"chunk_type": "image_caption"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["data"][0]["id"] == "img-1"


async def test_list_chunks_excludes_other_tenants(
    client: TestClient,
    chunk_service: AsyncMock,
) -> None:
    chunk_service._rows["mine"] = _chunk(id="mine")  # type: ignore[attr-defined]
    chunk_service._rows["theirs"] = _chunk(id="theirs", tenant_id=99)  # type: ignore[attr-defined]

    resp = client.get(f"/chunks/{KNOWLEDGE_ID}")

    assert resp.status_code == 200
    assert [c["id"] for c in resp.json()["data"]] == ["mine"]


# ── GET /chunks/{knowledge_id}/{id}/revisions ────────────────────────


async def test_list_revisions_returns_history_newest_first(
    client: TestClient,
    chunk_service: AsyncMock,
    revision_service: AsyncMock,
) -> None:
    chunk_service._rows[CHUNK_ID] = _chunk()  # type: ignore[attr-defined]
    revision_service._revisions[(CHUNK_ID, 1)] = _revision(revision=1)  # type: ignore[attr-defined]
    revision_service._revisions[(CHUNK_ID, 2)] = _revision(revision=2)  # type: ignore[attr-defined]

    resp = client.get(f"/chunks/{KNOWLEDGE_ID}/{CHUNK_ID}/revisions")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert [r["revision"] for r in body["data"]] == [2, 1]


async def test_list_revisions_unknown_chunk_returns_404(client: TestClient) -> None:
    resp = client.get(f"/chunks/{KNOWLEDGE_ID}/nope/revisions")

    assert resp.status_code == 404


async def test_list_revisions_foreign_knowledge_returns_403(
    client: TestClient,
    chunk_service: AsyncMock,
) -> None:
    chunk_service._rows[CHUNK_ID] = _chunk(knowledge_id="other-knowledge")  # type: ignore[attr-defined]

    resp = client.get(f"/chunks/{KNOWLEDGE_ID}/{CHUNK_ID}/revisions")

    assert resp.status_code == 403


# ── PUT /chunks/{knowledge_id}/{id} ──────────────────────────────────


async def test_update_chunk_returns_updated(
    client: TestClient,
    chunk_service: AsyncMock,
    default_create_chunk_request: dict[str, object],
) -> None:
    chunk_service._rows[CHUNK_ID] = _chunk()  # type: ignore[attr-defined]
    body = dict(default_create_chunk_request)
    body["content"] = "updated body"

    resp = client.put(f"/chunks/{KNOWLEDGE_ID}/{CHUNK_ID}", json=body)

    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["id"] == CHUNK_ID
    assert data["content"] == "updated body"


async def test_update_chunk_empty_body_keeps_values(
    client: TestClient,
    chunk_service: AsyncMock,
) -> None:
    chunk_service._rows[CHUNK_ID] = _chunk()  # type: ignore[attr-defined]

    resp = client.put(f"/chunks/{KNOWLEDGE_ID}/{CHUNK_ID}", json={})

    assert resp.status_code == 200
    assert resp.json()["data"]["content"] == "original content"


async def test_update_chunk_revision_conflict_returns_409(
    client: TestClient,
    chunk_service: AsyncMock,
) -> None:
    chunk_service._rows[CHUNK_ID] = _chunk(content_revision=2)  # type: ignore[attr-defined]

    resp = client.put(
        f"/chunks/{KNOWLEDGE_ID}/{CHUNK_ID}",
        json={"content": "x", "expected_revision": 1},
    )

    assert resp.status_code == 409


async def test_update_chunk_missing_returns_404(client: TestClient) -> None:
    resp = client.put(f"/chunks/{KNOWLEDGE_ID}/nope", json={"content": "x"})

    assert resp.status_code == 404


async def test_update_chunk_foreign_knowledge_returns_403(
    client: TestClient,
    chunk_service: AsyncMock,
) -> None:
    chunk_service._rows[CHUNK_ID] = _chunk(knowledge_id="other-knowledge")  # type: ignore[attr-defined]

    resp = client.put(f"/chunks/{KNOWLEDGE_ID}/{CHUNK_ID}", json={"content": "x"})

    assert resp.status_code == 403


# ── DELETE /chunks/{knowledge_id}/{id} ───────────────────────────────


async def test_delete_chunk_returns_ack_and_soft_deletes(
    client: TestClient,
    chunk_service: AsyncMock,
) -> None:
    chunk_service._rows[CHUNK_ID] = _chunk()  # type: ignore[attr-defined]

    resp = client.delete(f"/chunks/{KNOWLEDGE_ID}/{CHUNK_ID}")

    assert resp.status_code == 200
    assert resp.json() == {"success": True, "message": "Chunk deleted"}
    assert chunk_service._rows[CHUNK_ID].deleted_at is not None  # type: ignore[attr-defined]


async def test_delete_chunk_missing_returns_404(client: TestClient) -> None:
    resp = client.delete(f"/chunks/{KNOWLEDGE_ID}/nope")

    assert resp.status_code == 404


# ── DELETE /chunks/{knowledge_id} ────────────────────────────────────


async def test_delete_chunks_by_knowledge_returns_ack(
    client: TestClient,
    chunk_service: AsyncMock,
) -> None:
    chunk_service._rows["c1"] = _chunk(id="c1")  # type: ignore[attr-defined]
    chunk_service._rows["c2"] = _chunk(id="c2")  # type: ignore[attr-defined]

    resp = client.delete(f"/chunks/{KNOWLEDGE_ID}")

    assert resp.status_code == 200
    assert resp.json() == {"success": True, "message": "All chunks under knowledge deleted"}


# ── POST /chunks/{knowledge_id}/{id}/revert ──────────────────────────


async def test_revert_chunk_replays_snapshot(
    client: TestClient,
    chunk_service: AsyncMock,
    revision_service: AsyncMock,
) -> None:
    chunk_service._rows[CHUNK_ID] = _chunk(content_revision=2)  # type: ignore[attr-defined]
    revision_service._revisions[(CHUNK_ID, 1)] = _revision(  # type: ignore[attr-defined]
        revision=1,
        content="snapshot content",
    )

    resp = client.post(f"/chunks/{KNOWLEDGE_ID}/{CHUNK_ID}/revert", json={"revision": 1})

    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["id"] == CHUNK_ID
    assert data["content"] == "snapshot content"


async def test_revert_chunk_negative_revision_returns_422(
    client: TestClient,
    chunk_service: AsyncMock,
) -> None:
    chunk_service._rows[CHUNK_ID] = _chunk()  # type: ignore[attr-defined]

    resp = client.post(f"/chunks/{KNOWLEDGE_ID}/{CHUNK_ID}/revert", json={"revision": -1})

    assert resp.status_code == 422


async def test_revert_chunk_unknown_revision_returns_404(
    client: TestClient,
    chunk_service: AsyncMock,
) -> None:
    chunk_service._rows[CHUNK_ID] = _chunk(content_revision=1)  # type: ignore[attr-defined]

    resp = client.post(f"/chunks/{KNOWLEDGE_ID}/{CHUNK_ID}/revert", json={"revision": 9})

    assert resp.status_code == 404


async def test_revert_chunk_foreign_knowledge_returns_403(
    client: TestClient,
    chunk_service: AsyncMock,
) -> None:
    chunk_service._rows[CHUNK_ID] = _chunk(knowledge_id="other-knowledge")  # type: ignore[attr-defined]

    resp = client.post(f"/chunks/{KNOWLEDGE_ID}/{CHUNK_ID}/revert", json={"revision": 0})

    assert resp.status_code == 403


# ── PUT /chunks/by-id/{id}/questions ─────────────────────────────────


async def test_upsert_question_appends_new(
    client: TestClient,
    chunk_service: AsyncMock,
) -> None:
    chunk_service._rows[CHUNK_ID] = _chunk()  # type: ignore[attr-defined]

    resp = client.put(
        f"/chunks/by-id/{CHUNK_ID}/questions",
        json={"question": "What is a chunk?"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["question"] == "What is a chunk?"
    assert body["data"]["id"]
    stored = chunk_service._rows[CHUNK_ID]  # type: ignore[attr-defined]
    assert stored.metadata is not None
    assert stored.metadata["generated_questions"][0]["question"] == "What is a chunk?"  # type: ignore[index]


async def test_upsert_question_replaces_known_id(
    client: TestClient,
    chunk_service: AsyncMock,
) -> None:
    chunk_service._rows[CHUNK_ID] = _chunk(  # type: ignore[attr-defined]
        metadata={
            "generated_questions": [
                {"id": "q-1", "question": "old", "content_revision": 0}
            ],
            "generated_questions_revision": 0,
        }
    )

    resp = client.put(
        f"/chunks/by-id/{CHUNK_ID}/questions",
        json={"question_id": "q-1", "question": "new"},
    )

    assert resp.status_code == 200
    assert resp.json()["data"]["question"] == "new"
    stored = chunk_service._rows[CHUNK_ID]  # type: ignore[attr-defined]
    assert stored.metadata is not None
    assert stored.metadata["generated_questions"][0]["question"] == "new"  # type: ignore[index]


async def test_upsert_question_blank_text_returns_422(
    client: TestClient,
    chunk_service: AsyncMock,
) -> None:
    chunk_service._rows[CHUNK_ID] = _chunk()  # type: ignore[attr-defined]

    resp = client.put(f"/chunks/by-id/{CHUNK_ID}/questions", json={"question": "   "})

    assert resp.status_code == 422


async def test_upsert_question_unknown_id_returns_422(
    client: TestClient,
    chunk_service: AsyncMock,
) -> None:
    chunk_service._rows[CHUNK_ID] = _chunk(  # type: ignore[attr-defined]
        metadata={
            "generated_questions": [{"id": "q-1", "question": "old"}],
            "generated_questions_revision": 0,
        }
    )

    resp = client.put(
        f"/chunks/by-id/{CHUNK_ID}/questions",
        json={"question_id": "q-unknown", "question": "x"},
    )

    assert resp.status_code == 422


async def test_upsert_question_missing_chunk_returns_404(client: TestClient) -> None:
    resp = client.put("/chunks/by-id/nope/questions", json={"question": "x"})

    assert resp.status_code == 404


# ── DELETE /chunks/by-id/{id}/questions ──────────────────────────────


async def test_delete_question_returns_ack_and_removes(
    client: TestClient,
    chunk_service: AsyncMock,
) -> None:
    chunk_service._rows[CHUNK_ID] = _chunk(  # type: ignore[attr-defined]
        metadata={
            "generated_questions": [{"id": "q-1", "question": "q"}],
            "generated_questions_revision": 0,
        }
    )

    resp = client.request(
        "DELETE",
        f"/chunks/by-id/{CHUNK_ID}/questions",
        json={"question_id": "q-1"},
    )

    assert resp.status_code == 200
    assert resp.json() == {"success": True, "message": "Generated question deleted"}
    stored = chunk_service._rows[CHUNK_ID]  # type: ignore[attr-defined]
    # The empty question list is omitted from the persisted metadata
    # (mirrors the upstream ``omitempty`` serialization).
    assert stored.metadata is not None
    assert stored.metadata.get("generated_questions") in (None, [])


async def test_delete_question_no_questions_returns_422(
    client: TestClient,
    chunk_service: AsyncMock,
) -> None:
    chunk_service._rows[CHUNK_ID] = _chunk()  # type: ignore[attr-defined]

    resp = client.request(
        "DELETE",
        f"/chunks/by-id/{CHUNK_ID}/questions",
        json={"question_id": "q-1"},
    )

    assert resp.status_code == 422


async def test_delete_question_unknown_id_returns_422(
    client: TestClient,
    chunk_service: AsyncMock,
) -> None:
    chunk_service._rows[CHUNK_ID] = _chunk(  # type: ignore[attr-defined]
        metadata={
            "generated_questions": [{"id": "q-1", "question": "q"}],
            "generated_questions_revision": 0,
        }
    )

    resp = client.request(
        "DELETE",
        f"/chunks/by-id/{CHUNK_ID}/questions",
        json={"question_id": "q-unknown"},
    )

    assert resp.status_code == 422


async def test_delete_question_missing_chunk_returns_404(client: TestClient) -> None:
    resp = client.request(
        "DELETE",
        "/chunks/by-id/nope/questions",
        json={"question_id": "q-1"},
    )

    assert resp.status_code == 404


# ── POST /chunks/by-id/{id}/questions/regenerate ─────────────────────


async def test_regenerate_questions_returns_empty_list(
    client: TestClient,
    chunk_service: AsyncMock,
) -> None:
    chunk_service._rows[CHUNK_ID] = _chunk()  # type: ignore[attr-defined]

    resp = client.post(f"/chunks/by-id/{CHUNK_ID}/questions/regenerate")

    assert resp.status_code == 200
    assert resp.json() == {"success": True, "data": []}


async def test_regenerate_questions_non_text_returns_422(
    client: TestClient,
    chunk_service: AsyncMock,
) -> None:
    chunk_service._rows[CHUNK_ID] = _chunk(chunk_type="image_caption")  # type: ignore[attr-defined]

    resp = client.post(f"/chunks/by-id/{CHUNK_ID}/questions/regenerate")

    assert resp.status_code == 422


async def test_regenerate_questions_missing_chunk_returns_404(client: TestClient) -> None:
    resp = client.post("/chunks/by-id/nope/questions/regenerate")

    assert resp.status_code == 404
