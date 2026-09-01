"""Wire-shape conversion for the session endpoints.

Projects the session-domain row (``db.models.session.Session``) onto
the frozen contract shapes in ``src/core/contracts/sessions.py`` and
wraps them in the success envelope. The IM-origin fields on the
contract are joined at read time upstream; this build has no channel
mapper, so they stay ``None`` (the contract types them nullable).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.core.contracts.sessions import Session
from src.db.models.session import Session as SessionRow


class SessionEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` - single-session responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: Session


class SessionListEnvelope(BaseModel):
    """``{"success": true, "data": [...], "total": n, "page": n, "page_size": n}``.

    The list payload is a flat array at ``data`` with the paging
    counters as envelope siblings, matching the upstream contract.
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    data: list[Session]
    total: int
    page: int
    page_size: int


class DeleteSessionResponse(BaseModel):
    """``{"success": true, "message": "..."}`` - delete acknowledgement."""

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str


class PinSessionEnvelope(BaseModel):
    """``{"success": true, "is_pinned": bool}`` - pin toggle response."""

    model_config = ConfigDict(frozen=True)

    success: bool
    is_pinned: bool


def session_to_contract(row: SessionRow) -> Session:
    """Project a session row onto the frozen wire contract.

    The IM-origin fields (``im_platform`` etc.) are filled by a
    channel mapper at read time in the upstream; without a channel
    mapping this build emits them as ``None``.
    """
    return Session(
        id=row.id,
        title=row.title,
        description=row.description,
        tenant_id=row.tenant_id,
        user_id=row.user_id,
        is_pinned=row.is_pinned,
        pinned_at=row.pinned_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
        deleted_at=row.deleted_at,
    )


def session_envelope(row: SessionRow) -> SessionEnvelope:
    """Wrap one session in the success envelope."""
    return SessionEnvelope(success=True, data=session_to_contract(row))


def session_list_envelope(
    rows: list[SessionRow],
    *,
    total: int,
    page: int,
    page_size: int,
) -> SessionListEnvelope:
    """Wrap a paged session list in the flat success envelope."""
    return SessionListEnvelope(
        success=True,
        data=[session_to_contract(row) for row in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


def delete_session_response(message: str) -> DeleteSessionResponse:
    """Wrap a delete acknowledgement."""
    return DeleteSessionResponse(success=True, message=message)


__all__ = [
    "DeleteSessionResponse",
    "PinSessionEnvelope",
    "SessionEnvelope",
    "SessionListEnvelope",
    "delete_session_response",
    "session_envelope",
    "session_list_envelope",
    "session_to_contract",
]
