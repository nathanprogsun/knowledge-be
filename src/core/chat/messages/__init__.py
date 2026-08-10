"""Chat message domain: vocabulary, persistence, and services."""

from __future__ import annotations

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
    "MESSAGE_ROLES",
    "ROLE_ASSISTANT",
    "ROLE_SYSTEM",
    "ROLE_USER",
    "MessageSearchMode",
    "SUGGESTION_EVENT_CLICK",
    "SUGGESTION_EVENT_DISMISS",
    "SUGGESTION_EVENT_IMPRESSION",
    "SUGGESTION_EVENT_REGENERATE",
    "SUGGESTION_PLACEMENT_AFTER_ANSWER",
    "SUGGESTION_STATUS_FAILED",
    "SUGGESTION_STATUS_GENERATING",
    "SUGGESTION_STATUS_READY",
    "SUGGESTION_STATUS_SUPPRESSED",
    "MessageSuggestionService",
]
