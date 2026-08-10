"""Message service subpackage — request-scoped CRUD, search, and KB indexing."""

from __future__ import annotations

from src.core.chat.messages.service.message_service import (
    CHAT_HISTORY_KB_STATS_DISABLED,
    ChatHistoryConfigProvider,
    ChatHistoryKBStats,
    MessageSearchGroupItem,
    MessageSearchParams,
    MessageSearchResult,
    MessageSearchResultItem,
    MessageService,
    MessageServiceImpl,
    MessageVectorSearcher,
    MessageWithSession,
)
from src.core.chat.messages.types import MessageSearchMode

__all__ = [
    "CHAT_HISTORY_KB_STATS_DISABLED",
    "ChatHistoryConfigProvider",
    "ChatHistoryKBStats",
    "MessageSearchGroupItem",
    "MessageSearchMode",
    "MessageSearchParams",
    "MessageSearchResult",
    "MessageSearchResultItem",
    "MessageService",
    "MessageServiceImpl",
    "MessageVectorSearcher",
    "MessageWithSession",
]