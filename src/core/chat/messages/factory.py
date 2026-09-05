"""Message-domain request-scoped service factory.

Assembles a per-request :class:`MessageServiceImpl` on the shared
``AsyncSession``, following the ``src.core.chat.sessions.factory``
pattern: repositories and services are built fresh for every request;
``web`` never imports ``db``.

The constructor takes the message and session repositories so the
service can scope its reads to a tenant (via ``session_repo``) and
its writes/reads to a session (via ``message_repo``). The optional
vector-searcher, chat-history-config, and indexer seams stay at
their defaults; the retrieval engine and KB-index lifecycle land
in a later change.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.chat.messages.service.message_service import MessageServiceImpl
from src.core.chat.messages.suggestion_service import MessageSuggestionService
from src.core.infra.models.factory import build_chat_model_service
from src.db.dao.message_repository import MessageRepository
from src.db.dao.message_suggestion_repository import MessageSuggestionRepository
from src.db.dao.session_repository import SessionRepository


def build_message_service(session: AsyncSession) -> MessageServiceImpl:
    """Build a per-request :class:`MessageServiceImpl` on ``session``.

    Wires the message and session repositories. The request context
    (``tenant_id`` / ``user_id``) is read from the pipeline ``Context``
    at call time, so the factory does not need them up front.
    """
    return MessageServiceImpl(
        message_repo=MessageRepository(session),
        session_repo=SessionRepository(session),
    )


def build_message_suggestion_service(
    session: AsyncSession,
    *,
    tenant_id: int,
) -> MessageSuggestionService:
    """Build the follow-up generator on the shared session."""
    return MessageSuggestionService(
        tenant_id=tenant_id,
        messages=MessageRepository(session),
        suggestions=MessageSuggestionRepository(session),
        chat_models=build_chat_model_service(session),
    )


__all__ = ["build_message_service", "build_message_suggestion_service"]
