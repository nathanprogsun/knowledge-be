"""Chat message domain: vocabulary, persistence, and services."""

from __future__ import annotations

from src.core.chat.messages.index_to_kb import (
    DefaultMessageIndexer,
    KnowledgePassageCreator,
    MessageIndexer,
    build_passage,
    strip_think_tags,
)
from src.core.chat.messages.service import (
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
from src.core.chat.messages.suggestion_service import (
    SUGGESTION_EVENT_CLICK,
    SUGGESTION_EVENT_DISMISS,
    SUGGESTION_EVENT_IMPRESSION,
    SUGGESTION_EVENT_REGENERATE,
    SUGGESTION_PLACEMENT_AFTER_ANSWER,
    SUGGESTION_STATUS_FAILED,
    SUGGESTION_STATUS_GENERATING,
    SUGGESTION_STATUS_READY,
    SUGGESTION_STATUS_SUPPRESSED,
    MessageSuggestionService,
)
from src.core.chat.messages.types import (
    MESSAGE_ROLES,
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_USER,
    MessageSearchMode,
)

__all__ = [
    "CHAT_HISTORY_KB_STATS_DISABLED",
    "ChatHistoryConfigProvider",
    "ChatHistoryKBStats",
    "DefaultMessageIndexer",
    "KnowledgePassageCreator",
    "MESSAGE_ROLES",
    "MessageIndexer",
    "MessageSearchGroupItem",
    "MessageSearchMode",
    "MessageSearchParams",
    "MessageSearchResult",
    "MessageSearchResultItem",
    "MessageService",
    "MessageServiceImpl",
    "MessageSuggestionService",
    "MessageVectorSearcher",
    "MessageWithSession",
    "ROLE_ASSISTANT",
    "ROLE_SYSTEM",
    "ROLE_USER",
    "SUGGESTION_EVENT_CLICK",
    "SUGGESTION_EVENT_DISMISS",
    "SUGGESTION_EVENT_IMPRESSION",
    "SUGGESTION_EVENT_REGENERATE",
    "SUGGESTION_PLACEMENT_AFTER_ANSWER",
    "SUGGESTION_STATUS_FAILED",
    "SUGGESTION_STATUS_GENERATING",
    "SUGGESTION_STATUS_READY",
    "SUGGESTION_STATUS_SUPPRESSED",
    "build_passage",
    "strip_think_tags",
]