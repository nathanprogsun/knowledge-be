"""Session / message / suggestion service dependency factories.

Each request-scoped service is built on the shared ``AsyncSession``,
following the ``src.web.deps.chat`` pattern: repositories and services
are assembled fresh per request via the core factories so ``web``
never imports ``db``.

The message service reads ``tenant_id`` off the pipeline ``Context``
at call time, so the message factory needs no tenant up front; the
context dependency resolves it from the request store. For API-key
callers ``request_context.get_user_id()`` is ``None`` and the session
service requires a non-empty owner scope, so the helper falls back to
a synthetic ``tenant:<id>`` owner — keeping the scope non-empty while
the session repository's owner branch still matches the tenant-scoped
behaviour the upstream uses for non-Web principals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends

from src.app_context import request_context
from src.common.exception import ValidationError
from src.core.chat.messages.factory import build_message_service
from src.core.chat.messages.service.message_service import MessageServiceImpl
from src.core.chat.messages.suggestion_service import MessageSuggestionService
from src.core.chat.pipeline.types import Context
from src.core.chat.sessions.factory import build_session_service
from src.core.chat.sessions.service.session_service import SessionService
from src.web.deps.session import SessionDep


@dataclass(frozen=True, slots=True)
class _RequestContext:
    """Minimal pipeline ``Context`` carrying the active workspace id.

    The message service reads ``tenant_id`` off the context for
    session-existence checks; the other fields are protocol-narrow
    placeholders the not-yet-wired chat-history KB seams never look
    at.
    """

    tenant_id: int


def _resolve_context_tenant() -> int:
    """Return the active workspace id from request context, or raise."""
    raw = request_context.get_tenant_id()
    if raw is None or raw == "":
        raise ValidationError(
            code="chat.tenant_context_missing",
            message="No active workspace in request context",
        )
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            code="chat.tenant_context_invalid",
            message="Active workspace id is invalid",
        ) from exc


def _resolve_owner_id() -> str:
    """Return the caller's owner id for the session scope.

    Web-console and embed visitors carry a real user id. API-key
    callers have no principal user; the session service requires a
    non-empty owner scope, so the helper falls back to a synthetic
    tenant-scoped id derived from the tenant id. This keeps the
    session repository's owner-scope branch consistent (the
    repository treats empty owners as tenant-wide).
    """
    raw = request_context.get_user_id()
    if raw:
        return raw
    tenant = request_context.get_tenant_id() or ""
    return f"tenant:{tenant}"


def get_session_service(session: SessionDep) -> SessionService:
    """Build a per-request :class:`SessionService` on the shared session."""
    return build_session_service(
        session,
        tenant_id=_resolve_context_tenant(),
        user_id=_resolve_owner_id(),
    )


def get_message_service(session: SessionDep) -> MessageServiceImpl:
    """Build a per-request :class:`MessageServiceImpl` on the shared session."""
    return build_message_service(session)


def get_message_context() -> Context:
    """Return the pipeline ``Context`` the message service reads.

    The message service consumes ``ctx.tenant_id`` for its
    session-existence checks. Exposed as a dependency so the router
    layer threads the value through without re-reading the context
    store itself.
    """
    return _RequestContext(tenant_id=_resolve_context_tenant())


def get_message_suggestion_service() -> MessageSuggestionService:
    """Return the per-request suggestion service (stateless stub).

    The full suggestion-generation pipeline lands in a later PR;
    today the service surface exists so the wire shape and routing
    can be exercised against a stable interface.
    """
    return MessageSuggestionService()


SessionServiceDep = Annotated[SessionService, Depends(get_session_service)]
MessageServiceDep = Annotated[MessageServiceImpl, Depends(get_message_service)]
MessageContextDep = Annotated[Context, Depends(get_message_context)]
MessageSuggestionServiceDep = Annotated[
    MessageSuggestionService, Depends(get_message_suggestion_service)
]


__all__ = [
    "MessageContextDep",
    "MessageServiceDep",
    "MessageSuggestionServiceDep",
    "SessionServiceDep",
    "get_message_context",
    "get_message_service",
    "get_message_suggestion_service",
    "get_session_service",
]
