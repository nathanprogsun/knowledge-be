"""LLM-powered memory consolidation for agent conversations.

When the context window grows too large, ``MemoryConsolidator`` summarizes the
older history into a compact system message that preserves key facts and tool
results, while keeping the current turn intact. If the summarization LLM fails
after repeated attempts, the older messages are archived as raw text instead.

The module operates on plain ``AgentMessage`` lists and reaches the LLM only
through an injectable ``chat`` seam, so it stays independent of any particular
chat client or engine wiring.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeAlias

from src.common.exception import AIProviderError
from src.core.agents.engine.token_est import AgentMessage, TokenEstimator

#: Ratio of the context window that triggers consolidation (0.5 = half full).
DEFAULT_CONSOLIDATION_THRESHOLD = 0.5
#: Maximum number of LLM summarization attempts before raw-archive fallback.
MAX_CONSOLIDATION_ATTEMPTS = 3
#: Fraction of the threshold aimed for when sizing the preserved history.
_SUMMARY_TARGET_RATIO = 0.6
#: Token reserve left for the inserted summary message.
_SUMMARY_RESERVE_TOKENS = 500

#: Injectable chat seam: takes the summarization prompt messages and returns
#: the summary text. The injected implementation applies the low-temperature
#: sampling appropriate for factual summarization.
ConsolidationChatFn: TypeAlias = Callable[[list[AgentMessage]], Awaitable[str]]

#: System prompt prepended to every summarization request.
CONSOLIDATION_SYSTEM_PROMPT = (
    "You are a conversation summarizer. "
    "Your task is to create a concise but comprehensive summary "
    "of a conversation between a user and an AI assistant.\n\n"
    "The summary should:\n"
    "- Be written in the same language as the original conversation\n"
    "- Preserve all key facts, numbers, and specific details\n"
    "- Include the outcomes of any tool executions\n"
    "- Note any errors or issues encountered\n"
    "- Be structured with clear sections if the conversation covered multiple topics\n"
    "- Be concise — aim for 30% or less of the original length\n\n"
    "Output only the summary, no preamble or explanation."
)


def _truncate_for_prompt(text: str, max_len: int) -> str:
    """Truncate ``text`` to ``max_len`` code points for use in prompts."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


class MemoryConsolidator:
    """Compress agent conversation history using LLM summarization.

    ``chat`` is the injectable summarization seam (see ``ConsolidationChatFn``).
    ``max_context_tokens`` is the total context-window budget; ``threshold`` is
    the ratio (0-1) at which consolidation triggers (0 = use the default 0.5).
    """

    def __init__(
        self,
        chat: ConsolidationChatFn,
        estimator: TokenEstimator,
        max_context_tokens: int,
        threshold: float = 0.0,
    ) -> None:
        if threshold <= 0 or threshold >= 1:
            threshold = DEFAULT_CONSOLIDATION_THRESHOLD
        self._chat = chat
        self._estimator = estimator
        self._max_tokens = max_context_tokens
        self._threshold = threshold

    def should_consolidate(self, current_tokens: int) -> bool:
        """Return whether the given token estimate triggers consolidation.

        The estimate should come from the model API's ``Usage`` when available.
        """
        if self._max_tokens <= 0:
            return False
        trigger_at = int(float(self._max_tokens) * self._threshold)
        return current_tokens > trigger_at

    async def consolidate(self, messages: list[AgentMessage]) -> list[AgentMessage]:
        """Summarize older messages and return a compressed message array.

        Preserves the system prompt (first message), the current turn (the last
        user query and all subsequent assistant / tool messages), and recent
        history that fits within the token budget. Older history is replaced by
        a summary system message; on LLM failure after ``MAX_CONSOLIDATION_ATTEMPTS``
        the older history is archived as raw text instead.
        """
        if len(messages) <= 3:
            return messages

        system_msg = messages[0]

        # The current user query — the last message with role "user".
        last_user_idx = 0
        for i in range(len(messages) - 1, 0, -1):
            if messages[i].role == "user":
                last_user_idx = i
                break
        if last_user_idx <= 1:
            return messages

        history = messages[1:last_user_idx]
        tail = messages[last_user_idx:]

        if len(history) < 2:
            return messages

        target_tokens = int(float(self._max_tokens) * self._threshold * _SUMMARY_TARGET_RATIO)

        tail_tokens = sum(self._estimator.estimate_message(m) for m in tail)
        keep_from_end = self._find_keep_boundary(history, target_tokens, system_msg, tail_tokens)

        if keep_from_end >= len(history):
            return messages

        to_consolidate = history[: len(history) - keep_from_end]
        to_keep = history[len(history) - keep_from_end :]

        try:
            summary = await self._summarize_with_retry(to_consolidate)
        except AIProviderError:
            summary = self._raw_archive(to_consolidate)

        summary_msg = AgentMessage(
            role="system",
            content=(
                f"[Memory Summary - {len(to_consolidate)} earlier messages consolidated]\n\n"
                f"{summary}"
            ),
        )

        return [system_msg, summary_msg, *to_keep, *tail]

    def _find_keep_boundary(
        self,
        history: list[AgentMessage],
        target_tokens: int,
        system_msg: AgentMessage,
        tail_tokens: int,
    ) -> int:
        """Return how many messages from the end of ``history`` to keep.

        Respects tool_call / tool_result boundaries: a tool result group (the
        trailing tool messages plus the preceding assistant with ``tool_calls``)
        is kept or dropped as a unit. ``tail_tokens`` is the token cost of the
        current-turn tail that is always preserved.
        """
        budget = (
            target_tokens
            - self._estimator.estimate_message(system_msg)
            - tail_tokens
            - _SUMMARY_RESERVE_TOKENS
        )

        if budget <= 0:
            return 0

        tokens = 0
        keep_count = 0
        i = len(history) - 1

        while i >= 0:
            msg = history[i]
            msg_tokens = self._estimator.estimate_message(msg)

            if msg.role == "tool":
                group_tokens = msg_tokens
                group_size = 1
                j = i - 1
                while j >= 0 and history[j].role == "tool":
                    group_tokens += self._estimator.estimate_message(history[j])
                    group_size += 1
                    j -= 1
                if j >= 0 and history[j].role == "assistant":
                    group_tokens += self._estimator.estimate_message(history[j])
                    group_size += 1

                if tokens + group_tokens > budget:
                    break
                tokens += group_tokens
                keep_count += group_size
                i -= group_size
            else:
                if tokens + msg_tokens > budget:
                    break
                tokens += msg_tokens
                keep_count += 1
                i -= 1

        return keep_count

    async def _summarize_with_retry(self, messages: list[AgentMessage]) -> str:
        """Attempt LLM summarization with up to ``MAX_CONSOLIDATION_ATTEMPTS``."""
        prompt = self._build_consolidation_prompt(messages)
        last_error: Exception | None = None

        for _attempt in range(1, MAX_CONSOLIDATION_ATTEMPTS + 1):
            try:
                summary = await self._chat(
                    [
                        AgentMessage(role="system", content=CONSOLIDATION_SYSTEM_PROMPT),
                        AgentMessage(role="user", content=prompt),
                    ]
                )
            except Exception as exc:  # retry any provider failure
                last_error = exc
                continue
            if summary != "":
                return summary
            last_error = AIProviderError(
                code="agent.memory.empty_summary",
                message="empty response from LLM",
            )

        raise AIProviderError(
            code="agent.memory_consolidation_failed",
            message=f"summarization failed after {MAX_CONSOLIDATION_ATTEMPTS} attempts",
            details={"last_error": str(last_error)} if last_error is not None else None,
        )

    def _build_consolidation_prompt(self, messages: list[AgentMessage]) -> str:
        """Build the prompt asking the LLM to summarize ``messages``."""
        parts = [
            "Summarize the following conversation history, preserving:\n",
            "1. Key facts and decisions made\n",
            "2. Tool execution results and their outcomes\n",
            "3. User's original intent and requirements\n",
            "4. Any errors encountered and how they were resolved\n\n",
            "Conversation to summarize:\n\n",
        ]
        for msg in messages:
            if msg.role == "user":
                parts.append(f"**User**: {_truncate_for_prompt(msg.content, 2000)}\n\n")
            elif msg.role == "assistant":
                if msg.tool_calls:
                    names = ", ".join(tc.function.name for tc in msg.tool_calls)
                    parts.append(
                        f"**Assistant** [called tools: {names}]: "
                        f"{_truncate_for_prompt(msg.content, 1000)}\n\n"
                    )
                else:
                    parts.append(f"**Assistant**: {_truncate_for_prompt(msg.content, 2000)}\n\n")
            elif msg.role == "tool":
                parts.append(
                    f"**Tool [{msg.name}]**: {_truncate_for_prompt(msg.content, 1000)}\n\n"
                )
        return "".join(parts)

    def _raw_archive(self, messages: list[AgentMessage]) -> str:
        """Create a simple text dump of messages as fallback when the LLM fails."""
        parts = ["Raw conversation archive (LLM summarization unavailable):\n\n"]
        for msg in messages:
            content = _truncate_for_prompt(msg.content, 500)
            if msg.role == "user":
                parts.append(f"- User: {content}\n")
            elif msg.role == "assistant":
                if msg.tool_calls:
                    names = ",".join(tc.function.name for tc in msg.tool_calls)
                    parts.append(f"- Assistant [tools: {names}]: {content}\n")
                else:
                    parts.append(f"- Assistant: {content}\n")
            elif msg.role == "tool":
                parts.append(f"- Tool[{msg.name}]: {content}\n")
        return "".join(parts)


__all__ = [
    "CONSOLIDATION_SYSTEM_PROMPT",
    "DEFAULT_CONSOLIDATION_THRESHOLD",
    "MAX_CONSOLIDATION_ATTEMPTS",
    "ConsolidationChatFn",
    "MemoryConsolidator",
]
