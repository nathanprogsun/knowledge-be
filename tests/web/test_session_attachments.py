"""HTTP tests for session attachment routes.

Standalone app: auth and role gates are no-oped, services are
``AsyncMock`` doubles, and no database is opened.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from io import BytesIO
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.params import Depends
from fastapi.testclient import TestClient

from src.app_context import request_context
from src.common.exception import NotFoundError
from src.core.knowledge.documents.temporary_document import TemporaryDocumentInfo
from src.db.models.session import Session as SessionRow
from src.web.api.chat.sessions.attachments import get_session_attachment_storage
from src.web.api.chat.sessions.router import router as sessions_router
from src.web.deps import RoleContributorDep, RoleViewerDep
from src.web.deps.chat_sessions import get_session_service
from src.web.deps.knowledge_documents import get_temporary_document_service
from src.web.exception_handler import register_exception_handlers
from src.web.middleware.auth import require_auth

_NOW = datetime.now(UTC)


def _session_row(*, session_id: str = "sess-1", tenant_id: int = 7) -> SessionRow:
    return SessionRow(
        id=session_id,
        tenant_id=tenant_id,
        title="title",
        description="desc",
        user_id="u-1",
        is_pinned=False,
        pinned_at=None,
        created_at=_NOW,
        updated_at=_NOW,
        deleted_at=None,
    )


def _attachment_info(
    *, attachment_id: str = "att-1", status: str = "ready"
) -> TemporaryDocumentInfo:
    return TemporaryDocumentInfo(
        id=attachment_id,
        tenant_id=7,
        session_id="sess-1",
        resource_ref="local://tmp/note.md",
        file_name="note.md",
        file_type=".md",
        file_size=5,
        mime_type="text/markdown",
        status=status,
        token_count=0,
        chunk_count=0,
        expires_at=_NOW + timedelta(hours=24),
        processing_options={},
        created_at=_NOW,
        updated_at=_NOW,
    )


def _noop_role_gates(app: FastAPI) -> None:
    def _noop() -> None:
        return None

    for dep in (RoleViewerDep, RoleContributorDep):
        for metadata in getattr(dep, "__metadata__", ()):
            if isinstance(metadata, Depends) and metadata.dependency is not None:
                app.dependency_overrides[metadata.dependency] = _noop


def _build_app(
    *,
    session_service: object,
    temp_docs: object,
    storage: object,
) -> FastAPI:
    async def _noop_auth() -> None:
        request_context.set_tenant_id("7")
        request_context.set_user_id("u-1")
        return

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(sessions_router, prefix="/api/v1")
    app.dependency_overrides[require_auth] = _noop_auth
    _noop_role_gates(app)
    app.dependency_overrides[get_session_service] = lambda: session_service
    app.dependency_overrides[get_temporary_document_service] = lambda: temp_docs
    app.dependency_overrides[get_session_attachment_storage] = lambda: storage
    return app


def _client(app: FastAPI) -> TestClient:
    return TestClient(app)


def test_upload_persists_bytes_and_returns_ready() -> None:
    sessions = AsyncMock()
    sessions.get = AsyncMock(return_value=_session_row())
    created = _attachment_info(status="uploaded")
    ready = _attachment_info(status="ready")
    temp_docs = AsyncMock()
    temp_docs.create = AsyncMock(return_value=created)
    temp_docs.mark_ready = AsyncMock(return_value=ready)
    storage = AsyncMock()
    storage.save_bytes = AsyncMock(return_value="local://tmp/note.md")
    app = _build_app(session_service=sessions, temp_docs=temp_docs, storage=storage)

    with _client(app) as client:
        response = client.post(
            "/api/v1/sessions/sess-1/attachments",
            files={"file": ("note.md", b"hello", "text/markdown")},
        )

    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True
    assert body["data"]["id"] == "att-1"
    assert body["data"]["status"] == "ready"
    assert body["data"]["file_name"] == "note.md"
    storage.save_bytes.assert_awaited_once()
    assert storage.save_bytes.await_args.kwargs["temp"] is True
    temp_docs.mark_ready.assert_awaited_once()
    sessions.get.assert_awaited_once_with("sess-1")


def test_upload_other_session_is_404() -> None:
    sessions = AsyncMock()
    sessions.get = AsyncMock(side_effect=NotFoundError(code="session.not_found", message="gone"))
    storage = AsyncMock()
    app = _build_app(session_service=sessions, temp_docs=AsyncMock(), storage=storage)

    with _client(app) as client:
        response = client.post(
            "/api/v1/sessions/other/attachments",
            files={"file": ("note.md", b"hello", "text/markdown")},
        )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session.not_found"
    storage.save_bytes.assert_not_called()


def test_upload_rejects_unsupported_type() -> None:
    sessions = AsyncMock()
    sessions.get = AsyncMock(return_value=_session_row())
    storage = AsyncMock()
    app = _build_app(session_service=sessions, temp_docs=AsyncMock(), storage=storage)

    with _client(app) as client:
        response = client.post(
            "/api/v1/sessions/sess-1/attachments",
            files={"file": ("evil.exe", b"hello", "application/octet-stream")},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "temporary_document.unsupported_file_type"
    storage.save_bytes.assert_not_called()


def test_upload_rejects_oversize(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.core.knowledge.documents.temporary_document.max_upload_bytes",
        lambda: 4,
    )
    sessions = AsyncMock()
    sessions.get = AsyncMock(return_value=_session_row())
    storage = AsyncMock()
    app = _build_app(session_service=sessions, temp_docs=AsyncMock(), storage=storage)

    with _client(app) as client:
        response = client.post(
            "/api/v1/sessions/sess-1/attachments",
            files={"file": ("note.md", b"hello", "text/markdown")},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "temporary_document.invalid_file_size"
    storage.save_bytes.assert_not_called()


def test_get_missing_row_is_typed_404() -> None:
    sessions = AsyncMock()
    sessions.get = AsyncMock(return_value=_session_row())
    temp_docs = AsyncMock()
    temp_docs.get = AsyncMock(return_value=None)
    app = _build_app(session_service=sessions, temp_docs=temp_docs, storage=AsyncMock())

    with _client(app) as client:
        response = client.get("/api/v1/sessions/sess-1/attachments/missing")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "temporary_document.not_found"


def test_get_returns_envelope() -> None:
    sessions = AsyncMock()
    sessions.get = AsyncMock(return_value=_session_row())
    temp_docs = AsyncMock()
    temp_docs.get = AsyncMock(return_value=_attachment_info())
    app = _build_app(session_service=sessions, temp_docs=temp_docs, storage=AsyncMock())

    with _client(app) as client:
        response = client.get("/api/v1/sessions/sess-1/attachments/att-1")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["id"] == "att-1"
    assert body["data"]["session_id"] == "sess-1"
    temp_docs.get.assert_awaited_once()


def test_list_returns_success_data_array() -> None:
    sessions = AsyncMock()
    sessions.get = AsyncMock(return_value=_session_row())
    temp_docs = AsyncMock()
    temp_docs.list = AsyncMock(
        return_value=[_attachment_info(), _attachment_info(attachment_id="att-2")]
    )
    app = _build_app(session_service=sessions, temp_docs=temp_docs, storage=AsyncMock())

    with _client(app) as client:
        response = client.get("/api/v1/sessions/sess-1/attachments")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert [item["id"] for item in body["data"]] == ["att-1", "att-2"]


def test_delete_missing_row_is_typed_404() -> None:
    sessions = AsyncMock()
    sessions.get = AsyncMock(return_value=_session_row())
    temp_docs = AsyncMock()
    temp_docs.delete = AsyncMock(return_value=False)
    app = _build_app(session_service=sessions, temp_docs=temp_docs, storage=AsyncMock())

    with _client(app) as client:
        response = client.delete("/api/v1/sessions/sess-1/attachments/missing")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "temporary_document.not_found"


def test_delete_returns_success_message() -> None:
    sessions = AsyncMock()
    sessions.get = AsyncMock(return_value=_session_row())
    temp_docs = AsyncMock()
    temp_docs.delete = AsyncMock(return_value=True)
    app = _build_app(session_service=sessions, temp_docs=temp_docs, storage=AsyncMock())

    with _client(app) as client:
        response = client.delete("/api/v1/sessions/sess-1/attachments/att-1")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert "deleted" in body["message"].lower()


def test_preview_streams_stored_bytes() -> None:
    sessions = AsyncMock()
    sessions.get = AsyncMock(return_value=_session_row())
    temp_docs = AsyncMock()
    temp_docs.get = AsyncMock(return_value=_attachment_info())
    storage = AsyncMock()
    storage.get_file = AsyncMock(return_value=BytesIO(b"hello"))
    app = _build_app(session_service=sessions, temp_docs=temp_docs, storage=storage)

    with _client(app) as client:
        response = client.get("/api/v1/sessions/sess-1/attachments/att-1/preview")

    assert response.status_code == 200
    assert response.content == b"hello"
    storage.get_file.assert_awaited_once_with("local://tmp/note.md")


def test_preview_missing_file_is_typed_404() -> None:
    sessions = AsyncMock()
    sessions.get = AsyncMock(return_value=_session_row())
    temp_docs = AsyncMock()
    temp_docs.get = AsyncMock(return_value=_attachment_info())
    storage = AsyncMock()
    storage.get_file = AsyncMock(side_effect=FileNotFoundError("gone"))
    app = _build_app(session_service=sessions, temp_docs=temp_docs, storage=storage)

    with _client(app) as client:
        response = client.get("/api/v1/sessions/sess-1/attachments/att-1/preview")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "temporary_document.file_not_found"


async def test_storage_dep_missing_backend_is_typed() -> None:
    request_context.set_tenant_id("7")
    with (
        patch(
            "src.web.api.chat.sessions.attachments.resolve_file_service_for_path",
            AsyncMock(return_value=None),
        ),
        pytest.raises(NotFoundError) as err,
    ):
        await get_session_attachment_storage(AsyncMock())
    assert err.value.code == "temporary_document.storage_unavailable"


def test_upload_and_delete_require_contributor() -> None:
    expected: dict[tuple[str, str], str] = {
        ("POST", "/sessions/{session_id}/attachments"): "contributor",
        ("GET", "/sessions/{session_id}/attachments"): "viewer",
        ("GET", "/sessions/{session_id}/attachments/{attachment_id}"): "viewer",
        ("GET", "/sessions/{session_id}/attachments/{attachment_id}/preview"): "viewer",
        ("DELETE", "/sessions/{session_id}/attachments/{attachment_id}"): "contributor",
    }
    for route in sessions_router.routes:
        path = getattr(route, "path", "")
        methods: set[str] = getattr(route, "methods", set()) or set()
        dependant = getattr(route, "dependant", None)
        if dependant is None:
            continue
        roles: set[str] = set()
        for dep in dependant.dependencies:
            closure = getattr(dep.call, "__closure__", None)
            for cell in closure or ():
                if isinstance(cell.cell_contents, str):
                    roles.add(cell.cell_contents)
        for method in methods:
            key = (method, path)
            if key in expected:
                assert expected[key] in roles, f"{method} {path} roles={roles}"
