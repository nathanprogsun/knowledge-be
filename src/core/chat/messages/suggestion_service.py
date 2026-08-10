"""Message suggestion service — surface and vocabulary (stub).

The full generation pipeline (LLM-driven follow-up question generation,
knowledge-based candidates, suppression rules) lands in a later PR.
This module carries the service surface and re-exports the domain
vocabulary so downstream callers can depend on the interface.
"""

from __future__ import annotations

from src.db.models.message_suggestion import (
    SUGGESTION_EVENT_CLICK,
    SUGGESTION_EVENT_DISMISS,
    SUGGESTION_EVENT_IMPRESSION,
    SUGGESTION_EVENT_REGENERATE,
    SUGGESTION_PLACEMENT_AFTER_ANSWER,
    SUGGESTION_STATUS_FAILED,
    SUGGESTION_STATUS_GENERATING,
    SUGGESTION_STATUS_READY,
    SUGGESTION_STATUS_SUPPRESSED,
    MessageSuggestionSet,
)


class MessageSuggestionService:
    """Follow-up suggestion generation and analytics (stub).

    Method signatures mirror the upstream service surface; the
    implementations are wired in a later PR.
    """

    async def ensure_follow_ups(
        self,
        *,
        session_id: str,
        assistant_message_id: str,
        regenerate: bool,
    ) -> MessageSuggestionSet:
        """Generate (or return the cached) follow-up suggestions."""
        raise NotImplementedError

    async def get_follow_ups(
        self,
        *,
        session_id: str,
        assistant_message_id: str,
    ) -> MessageSuggestionSet | None:
        """Return the cached follow-up suggestions for an assistant message."""
        raise NotImplementedError

    async def record_event(
        self,
        *,
        session_id: str,
        suggestion_set_id: str,
        question_id: str,
        event_type: str,
    ) -> None:
        """Record a product-analytics event for a suggestion question."""
        raise NotImplementedError

    async def validate_attribution(
        self,
        *,
        session_id: str,
        query: str,
        suggestion_set_id: str,
        question_id: str,
    ) -> None:
        """Validate that a follow-up click attribution matches a ready set."""
        raise NotImplementedError


__all__ = [
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
    "MessageSuggestionSet",
]
