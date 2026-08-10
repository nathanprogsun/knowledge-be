"""Chat-domain FastAPI dependency factories.

One-line forwarder to ``src.core.chat.factory``: the request-scoped
``ChatService`` is assembled per request on the shared ``AsyncSession``,
with the tenant / user / request ids read from the request-context
store. ``web`` never imports ``db``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.app_context import request_context
from src.common.exception import ValidationError
from src.core.chat.factory import build_chat_service
from src.core.chat.service import ChatService
from src.web.deps.session import SessionDep


def get_chat_service(session: SessionDep) -> ChatService:
    """Build a per-request ``ChatService`` on the shared session."""
    return build_chat_service(
        session,
        tenant_id=_require_context_tenant(),
        user_id=request_context.get_user_id() or "",
        request_id=request_context.get_request_id() or "",
    )


def _require_context_tenant() -> int:
    """Return the current tenant id from request context, or raise."""
    raw = request_context.get_tenant_id()
    if raw is None or raw == "":
        raise ValidationError(
            code="chat.tenant_context_missing",
            message="No active workspace in request context",
        )
    try:
        return int(raw)
    except ValueError as exc:
        raise ValidationError(
            code="chat.tenant_context_invalid",
            message="Active workspace id is invalid",
        ) from exc


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]


__all__ = [
    "ChatServiceDep",
    "get_chat_service",
]
