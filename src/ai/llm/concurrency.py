"""Background concurrency governor for chat clients.

Model-provider budgets are the real bottleneck shared by every LLM-backed
background stage, all targeting the same model. This governor caps concurrent
calls per model at the client layer — the one place that sees all task types —
instead of at the queue layer. Only background tasks are throttled (see
``limiter.background_task_context``); interactive chat passes straight
through, so a document-ingestion storm cannot exhaust the provider yet
user-facing latency is never gated behind the semaphore.

The wrapper is the outermost decorator, so the slot is held only around the
actual provider round-trip and the wait time is excluded from inner timing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from src.ai.llm.limiter import gate_named_n
from src.ai.llm.types import Chat, ChatOptions, ChatResponse, Message, StreamResponse


class ConcurrencyChat:
    """Throttles background LLM calls through a per-model distributed semaphore."""

    def __init__(self, inner: Chat, limit: int) -> None:
        self._inner = inner
        # This model's configured per-model background cap; 0 falls back to
        # the process-wide default (see limiter.gate_named_n).
        self._limit = limit

    def get_model_name(self) -> str:
        return self._inner.get_model_name()

    def get_model_id(self) -> str:
        return self._inner.get_model_id()

    async def chat(
        self, messages: list[Message], opts: ChatOptions | None = None
    ) -> ChatResponse:
        release = await gate_named_n(
            self._inner.get_model_id(), self._inner.get_model_name(), self._limit
        )
        try:
            return await self._inner.chat(messages, opts)
        finally:
            release()

    async def chat_stream(
        self, messages: list[Message], opts: ChatOptions | None = None
    ) -> AsyncIterator[StreamResponse]:
        release = await gate_named_n(
            self._inner.get_model_id(), self._inner.get_model_name(), self._limit
        )
        try:
            async for response in self._inner.chat_stream(messages, opts):
                yield response
        finally:
            release()


def wrap_chat_concurrency(chat: Chat, limit: int) -> Chat:
    """Install the background concurrency governor as the outermost decorator."""
    if chat is None:
        return chat
    return ConcurrencyChat(chat, limit)


__all__ = ["ConcurrencyChat", "wrap_chat_concurrency"]
