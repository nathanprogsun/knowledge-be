"""Session-domain request-scoped service factory.

Assembles a per-request :class:`SessionService` on the shared
``AsyncSession``, following the ``src.core.chat.factory`` pattern:
repositories and services are built fresh for every request; ``web``
never imports ``db``.

The chat factory and title generator are left as ``None`` by the
default factory — the QA handler wires the real chat-factory seam
when title generation is actually requested. The QA handler obtains
its own :class:`SessionService` via
:meth:`build_session_service_with_title` so the title path can
exercise the heavy seams against the live model registry.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.chat.sessions.service.session_service import (
    ChatFactoryLike,
    SessionService,
)
from src.core.chat.sessions.stop import StopStreamService
from src.core.chat.sessions.title_gen import TitleGenerator
from src.core.chat.stream.manager import StreamManager
from src.db.dao.message_repository import MessageRepository
from src.db.dao.session_repository import SessionRepository


def build_session_service(
    session: AsyncSession,
    *,
    tenant_id: int,
    user_id: str,
) -> SessionService:
    """Build a per-request :class:`SessionService` on ``session``.

    The chat factory and title generator stay at their defaults
    (``None`` / :class:`TitleGenerator`); title generation is wired
    by the QA handler through
    :meth:`build_session_service_with_title`.
    """
    return SessionService(
        tenant_id=tenant_id,
        user_id=user_id,
        session_repo=SessionRepository(session),
        message_repo=MessageRepository(session),
    )


def build_session_service_with_title(
    session: AsyncSession,
    *,
    tenant_id: int,
    user_id: str,
    chat_factory: ChatFactoryLike,
    title_generator: TitleGenerator | None = None,
) -> SessionService:
    """Build a per-request service that can generate titles.

    The chat factory is the only non-trivial seam; it resolves the
    active model and returns a chat client. The default title
    generator is the stock :class:`TitleGenerator`; callers may
    inject a custom one for tests or specialised prompts.
    """
    return SessionService(
        tenant_id=tenant_id,
        user_id=user_id,
        session_repo=SessionRepository(session),
        message_repo=MessageRepository(session),
        title_generator=title_generator or TitleGenerator(),
        chat_factory=chat_factory,
    )


def build_stop_stream_service(stream_manager: StreamManager) -> StopStreamService:
    """Build a stop facade on the shared process-local cancel store."""
    return StopStreamService(stream_manager=stream_manager)


__all__ = [
    "build_session_service",
    "build_session_service_with_title",
    "build_stop_stream_service",
]
