"""Session-domain service surface.

Exposes the public ``SessionService`` class. The per-request factory
lives in ``src.core.chat.sessions.factory`` so web / QA handlers
can construct a service on the shared ``AsyncSession`` without
importing ``db`` directly.
"""

from src.core.chat.sessions.service.session_service import (
    ChatFactoryLike,
    SessionListQuery,
    SessionMessageReader,
    SessionService,
)

__all__ = [
    "ChatFactoryLike",
    "SessionListQuery",
    "SessionMessageReader",
    "SessionService",
]
