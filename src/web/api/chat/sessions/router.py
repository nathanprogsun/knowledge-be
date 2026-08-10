"""Session HTTP endpoints - create, list, get, update, delete, pin.

Maps the session route family onto the request-scoped
``SessionService``:

==============  ====================================  ====================
Method          Path                                   Handler
==============  ====================================  ====================
``POST   ``    ``/sessions``                           create
``GET    ``    ``/sessions``                           list (paged)
``GET    ``    ``/sessions/{session_id}``              get
``PUT    ``    ``/sessions/{session_id}``              update
``DELETE ``    ``/sessions/{session_id}``              delete
``DELETE ``    ``/sessions/{session_id}/messages``     clear messages
``DELETE ``    ``/sessions/batch``                     batch delete
``POST   ``    ``/sessions/{session_id}/pin``          pin
``DELETE ``    ``/sessions/{session_id}/pin``          unpin
==============  ====================================  ====================

Sessions are per-user chat state (Viewer+ surface, mirroring the
upstream RBAC wiring); the service enforces the owner scope on every
read and write.

The static ``/sessions/batch`` path is declared before the
``/{session_id}``-shaped routes so a literal segment is never
captured as a session id.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Query

from src.common.exception import NotFoundError, ValidationError
from src.core.chat.sessions.service.session_service import SessionListQuery
from src.core.contracts.sessions import (
    BatchDeleteSessionsRequest,
    CreateSessionRequest,
    UpdateSessionRequest,
)
from src.db.models.session import Session as SessionRow
from src.web.api.chat.sessions.views import (
    DeleteSessionResponse,
    PinSessionEnvelope,
    SessionEnvelope,
    SessionListEnvelope,
    delete_session_response,
    session_envelope,
    session_list_envelope,
)
from src.web.deps import AuthDep, RoleViewerDep
from src.web.deps.chat_sessions import (
    MessageContextDep,
    MessageServiceDep,
    SessionServiceDep,
)

router = APIRouter(prefix="/sessions", tags=["sessions"])

_DEFAULT_PAGE = 1
_DEFAULT_PAGE_SIZE = 20
_MAX_PAGE_SIZE = 1000

_DELETE_MESSAGE = "Session deleted successfully"
_DELETE_ALL_MESSAGE = "All sessions deleted successfully"
_BATCH_DELETE_MESSAGE = "Sessions deleted successfully"
_CLEAR_MESSAGES_MESSAGE = "Session messages cleared successfully"


def _require_session_id(session_id: str) -> str:
    """Reject an empty session id (upstream rejects with a 400)."""
    if not session_id or not session_id.strip():
        raise ValidationError(
            code="session.id_required",
            message="Session ID is empty",
        )
    return session_id.strip()


def _now() -> datetime:
    """UTC ``now`` for constructing session rows the service re-stamps."""
    return datetime.now(UTC)


def _clamp_page(page: int) -> int:
    """Coerce a page value onto ``[1, inf)`` like the upstream clamp."""
    return page if page >= 1 else _DEFAULT_PAGE


def _clamp_page_size(page_size: int) -> int:
    """Coerce a page size onto ``[1, 1000]`` like the upstream clamp."""
    if page_size < 1:
        return _DEFAULT_PAGE_SIZE
    return min(page_size, _MAX_PAGE_SIZE)


@router.post("", response_model=SessionEnvelope, status_code=201)
async def create_session(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    body: CreateSessionRequest,
    session_service: SessionServiceDep,
) -> SessionEnvelope:
    """Create a session in the caller's workspace.

    The tenant and owner come from the request context; the caller
    supplies only the display fields (``title`` / ``description``).
    """
    row = await session_service.create(
        SessionRow(
            id="",
            tenant_id=session_service.tenant_id,
            title=body.title,
            description=body.description,
            user_id=session_service.user_id,
            created_at=_now(),
            updated_at=_now(),
        )
    )
    return session_envelope(row)


@router.delete("/batch", response_model=DeleteSessionResponse)
async def batch_delete_sessions(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    body: BatchDeleteSessionsRequest,
    session_service: SessionServiceDep,
) -> DeleteSessionResponse:
    """Delete the listed sessions, or every session when ``delete_all``.

    Mirrors the upstream batch delete: ``delete_all=true`` clears the
    caller's whole session list; otherwise ``ids`` must be non-empty
    (rejected with a 400) and each id is soft-deleted.
    """
    if body.delete_all:
        await session_service.delete_all()
        return delete_session_response(_DELETE_ALL_MESSAGE)

    ids = [sid.strip() for sid in (body.ids or []) if sid and sid.strip()]
    if not ids:
        raise ValidationError(
            code="session.ids_required",
            message="ids are required when delete_all is false",
        )
    deleted = await session_service.batch_delete(ids)
    if deleted == 0:
        raise NotFoundError(
            code="session.not_found",
            message="no visible sessions found for batch delete",
        )
    return delete_session_response(_BATCH_DELETE_MESSAGE)


@router.get("/{session_id}", response_model=SessionEnvelope)
async def get_session(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    session_id: str,
    session_service: SessionServiceDep,
) -> SessionEnvelope:
    """Return one session visible to the caller (404 when absent)."""
    row = await session_service.get(_require_session_id(session_id))
    return session_envelope(row)


@router.get("", response_model=SessionListEnvelope)
async def list_sessions(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    session_service: SessionServiceDep,
    page: int = Query(default=_DEFAULT_PAGE),
    page_size: int = Query(default=_DEFAULT_PAGE_SIZE),
    keyword: str | None = Query(default=None),
    source: str | None = Query(default=None),
    agent_id: str | None = Query(default=None),
) -> SessionListEnvelope:
    """Return a paged session list for the caller's workspace.

    ``keyword`` filters titles with a case-insensitive match. The
    ``source`` / ``agent_id`` filters are accepted but the channel
    mapping that backs them is a deferred seam; this build returns the
    owner-scoped page.
    """
    page = _clamp_page(page)
    page_size = _clamp_page_size(page_size)
    result = await session_service.list_with_filters(
        SessionListQuery(
            keyword=(keyword or "").strip(),
            page=page,
            page_size=page_size,
            source=source or "",
            agent_id=agent_id or "",
        )
    )
    return session_list_envelope(
        list(result.data),
        total=result.total,
        page=result.page,
        page_size=result.page_size,
    )


@router.put("/{session_id}", response_model=SessionEnvelope)
async def update_session(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    session_id: str,
    body: UpdateSessionRequest,
    session_service: SessionServiceDep,
) -> SessionEnvelope:
    """Update a session's display fields, then return the stored row."""
    updated = await session_service.update(
        SessionRow(
            id=_require_session_id(session_id),
            tenant_id=session_service.tenant_id,
            title=body.title,
            description=body.description,
            created_at=_now(),
            updated_at=_now(),
        )
    )
    return session_envelope(updated)


@router.delete("/{session_id}", response_model=DeleteSessionResponse)
async def delete_session(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    session_id: str,
    session_service: SessionServiceDep,
) -> DeleteSessionResponse:
    """Soft-delete one session (404 when absent or not owned)."""
    deleted = await session_service.delete(_require_session_id(session_id))
    if not deleted:
        raise NotFoundError(
            code="session.not_found",
            message=f"session {session_id} not found",
        )
    return delete_session_response(_DELETE_MESSAGE)


@router.delete("/{session_id}/messages", response_model=DeleteSessionResponse)
async def clear_session_messages(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    session_id: str,
    ctx: MessageContextDep,
    message_service: MessageServiceDep,
) -> DeleteSessionResponse:
    """Clear every message of a session; the session itself is kept."""
    await message_service.clear_session_messages(
        ctx,
        _require_session_id(session_id),
    )
    return delete_session_response(_CLEAR_MESSAGES_MESSAGE)


@router.post("/{session_id}/pin", response_model=PinSessionEnvelope)
async def pin_session(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    session_id: str,
    session_service: SessionServiceDep,
) -> PinSessionEnvelope:
    """Pin a session for the caller (404 when absent or not owned)."""
    await _set_pinned(session_service, session_id, True)
    return PinSessionEnvelope(success=True, is_pinned=True)


@router.delete("/{session_id}/pin", response_model=PinSessionEnvelope)
async def unpin_session(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    session_id: str,
    session_service: SessionServiceDep,
) -> PinSessionEnvelope:
    """Unpin a session for the caller (404 when absent or not owned)."""
    await _set_pinned(session_service, session_id, False)
    return PinSessionEnvelope(success=True, is_pinned=False)


async def _set_pinned(
    session_service: SessionServiceDep,
    session_id: str,
    pinned: bool,
) -> None:
    """Toggle the pin state; a zero-row result reads as 404."""
    affected = await session_service.set_pinned(
        _require_session_id(session_id),
        pinned,
    )
    if not affected:
        raise NotFoundError(
            code="session.not_found",
            message=f"session {session_id} not found",
        )


__all__ = ["router"]
