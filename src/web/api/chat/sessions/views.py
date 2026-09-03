"""Wire-shape conversion for the session endpoints.

Wraps the service-side ``SessionInfo`` projection into the frozen
contract shapes in ``src/core/contracts/sessions.py`` plus the success
envelope. The IM-origin fields on the contract are joined at read time
upstream; this build has no channel mapper, so they stay ``None`` (the
contract types them nullable).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.core.chat.sessions.types import SessionInfo
from src.core.contracts.sessions import Session


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


def session_to_contract(info: SessionInfo) -> Session:
    """Project the service-side session DTO onto the frozen wire contract.

    The IM-origin fields (``im_platform`` etc.) are filled by a
    channel mapper at read time in the upstream; without a channel
    mapping this build emits them as ``None``.
    """
    return Session(
        id=info.id,
        title=info.title,
        description=info.description,
        tenant_id=info.tenant_id,
        user_id=info.user_id,
        is_pinned=info.is_pinned,
        pinned_at=info.pinned_at,
        created_at=info.created_at,
        updated_at=info.updated_at,
        deleted_at=info.deleted_at,
    )


def session_envelope(info: SessionInfo) -> SessionEnvelope:
    """Wrap one session in the success envelope."""
    return SessionEnvelope(success=True, data=session_to_contract(info))


def session_list_envelope(
    rows: list[SessionInfo],
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
