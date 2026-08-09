"""Token estimation and context-budget compression for agent conversations.

``TokenEstimator`` counts tokens for messages and strings. The authoritative
count for a full context is the model API's ``Usage`` field; this estimator is
a supplementary approximation used in two scenarios:

1. Delta estimation — new messages (assistant reply + tool results) are
   appended between LLM calls, and the engine needs the token cost of those
   deltas to decide whether context compression is needed without an extra API
   round-trip.
2. First-round fallback — on the very first round of a session no prior
   ``Usage`` is available, so the estimator provides a full estimate.

The default encoder is a deterministic, dependency-free character heuristic
that approximates ``cl100k_base`` BPE behaviour: CJK / Hangul / kana code
points weigh ~2 tokens each while latin text averages ~4 characters per token.
The estimate only needs to be close enough to trigger compression at roughly
the right time — over- or under-estimating by a small margin is corrected by
the next API call. Callers that need exact counts inject a real tokenizer
through the ``encoder`` seam (e.g. ``tiktoken.get_encoding("cl100k_base")``
wrapped to return a count).

``compress_context`` trims older history to bring a context back under the
budget threshold, preserving the system prompt, the current turn, and
tool_call / tool_result pairs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeAlias

#: Per-message fixed token overhead charged by OpenAI-shaped chat APIs.
PER_MESSAGE_OVERHEAD = 3
#: Extra tokens appended once per full conversation.
PER_CONVERSATION_TAIL = 3
#: Ratio of the context window at which ``compress_context`` triggers.
DEFAULT_CONTEXT_THRESHOLD_RATIO = 0.8

#: Approximate tokens spent on each wide-script code point by the heuristic.
_WIDE_CHAR_TOKENS = 2
#: Average latin characters per token in ``cl100k_base``.
_LATIN_CHARS_PER_TOKEN = 4

#: Code-point ranges treated as wide scripts (Han, kana, Hangul, CJK symbols
#: and punctuation, fullwidth forms). cl100k_base spends multiple tokens per
#: code point on these.
_WIDE_RANGES: tuple[tuple[int, int], ...] = (
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x2E80, 0x2EFF),  # CJK Radicals Supplement
    (0x3000, 0x303F),  # CJK Symbols and Punctuation
    (0x3040, 0x30FF),  # Hiragana, Katakana
    (0x3130, 0x318F),  # Hangul Compatibility Jamo
    (0x31C0, 0x31EF),  # CJK Strokes
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xA960, 0xA97F),  # Hangul Jamo Extended-A
    (0xAC00, 0xD7AF),  # Hangul Syllables
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0xFE30, 0xFE4F),  # CJK Compatibility Forms
    (0xFF00, 0xFFEF),  # Halfwidth and Fullwidth Forms
    (0x20000, 0x2FA1F),  # CJK Unified Ideographs Extensions B-F
)


@dataclass(frozen=True)
class AgentFunctionCall:
    """Function details attached to a tool call."""

    name: str
    arguments: str = ""


@dataclass(frozen=True)
class AgentToolCall:
    """A tool call embedded in an assistant message."""

    id: str = ""
    type: str = "function"
    function: AgentFunctionCall = field(default_factory=lambda: AgentFunctionCall(name=""))


@dataclass(frozen=True)
class AgentMessage:
    """One chat message in an agent conversation.

    ``tool_calls`` is a tuple of ``AgentToolCall`` for assistant messages;
    ``tool_call_id`` / ``name`` identify the originating call on tool results.
    """

    role: str
    content: str = ""
    name: str = ""
    tool_call_id: str = ""
    tool_calls: tuple[AgentToolCall, ...] = ()


#: Injectable token-count seam: maps a string to its token count.
TokenEncoder: TypeAlias = Callable[[str], int]


def _is_wide_char(cp: int) -> bool:
    """Return whether ``cp`` falls inside any wide-script range."""
    return any(start <= cp <= end for start, end in _WIDE_RANGES)


def _heuristic_token_count(text: str) -> int:
    """Approximate ``cl100k_base`` token count without a real tokenizer."""
    wide = sum(1 for ch in text if _is_wide_char(ord(ch)))
    other = len(text) - wide
    return wide * _WIDE_CHAR_TOKENS + (other + _LATIN_CHARS_PER_TOKEN - 1) // _LATIN_CHARS_PER_TOKEN


class TokenEstimator:
    """Counts tokens for messages and strings.

    ``encoder`` is an optional seam mapping a string to a token count (e.g. a
    real BPE tokenizer). When provided, its output is trusted unless it raises,
    in which case the character heuristic is used as a fallback — mirroring the
    upstream behaviour of degrading gracefully on tokenizer failure. When
    ``encoder`` is absent, the heuristic is used directly.
    """

    def __init__(self, encoder: TokenEncoder | None = None) -> None:
        self._encoder = encoder

    def estimate_string(self, text: str) -> int:
        """Return the estimated token count for a single string."""
        if not text:
            return 0
        if self._encoder is not None:
            try:
                return max(self._encoder(text), 0)
            except Exception:
                return (len(text) + _LATIN_CHARS_PER_TOKEN - 1) // _LATIN_CHARS_PER_TOKEN
        return _heuristic_token_count(text)

    def estimate_message(self, message: AgentMessage) -> int:
        """Return the estimated token count for a single message."""
        tokens = PER_MESSAGE_OVERHEAD
        tokens += self.estimate_string(message.role)
        tokens += self.estimate_string(message.content)
        tokens += self.estimate_string(message.name)
        for call in message.tool_calls:
            tokens += self.estimate_string(call.function.name)
            tokens += self.estimate_string(call.function.arguments)
            tokens += 4
        return tokens

    def estimate_messages(self, messages: list[AgentMessage]) -> int:
        """Return the estimated token count for a list of messages.

        Prefer the API ``Usage`` for the full context and this method only for
        deltas between calls.
        """
        return sum(self.estimate_message(m) for m in messages) + PER_CONVERSATION_TAIL


def group_tool_messages(messages: list[AgentMessage]) -> list[list[AgentMessage]]:
    """Group messages into logical units so tool pairs are never split.

    - An assistant message carrying ``tool_calls`` plus its following tool
      result messages form one group.
    - Any other message (user, assistant without tool_calls) is its own group.
    """
    groups: list[list[AgentMessage]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.role == "assistant" and msg.tool_calls:
            group = [msg]
            i += 1
            while i < len(messages) and messages[i].role == "tool":
                group.append(messages[i])
                i += 1
            groups.append(group)
        else:
            groups.append([msg])
            i += 1
    return groups


def compress_context(
    messages: list[AgentMessage],
    estimator: TokenEstimator,
    max_tokens: int,
    current_tokens: int,
) -> list[AgentMessage]:
    """Trim older history to bring a context below the budget threshold.

    Preserves the system prompt (first message), the current turn (the last
    user query and everything after it), and tool_call / tool_result pairs
    (they are grouped and removed as a unit). ``current_tokens`` is the
    caller's best estimate of the current context size.
    """
    if max_tokens <= 0 or len(messages) <= 2:
        return messages

    threshold = int(float(max_tokens) * DEFAULT_CONTEXT_THRESHOLD_RATIO)
    if current_tokens <= threshold:
        return messages

    system_msg = messages[0]

    # The current user query — the last message with role "user".
    last_user_idx = len(messages) - 1
    for i in range(len(messages) - 1, 0, -1):
        if messages[i].role == "user":
            last_user_idx = i
            break

    history = messages[1:last_user_idx]
    tail = messages[last_user_idx:]

    if not history:
        return messages

    groups = group_tool_messages(history)

    tokens_to_free = current_tokens - threshold
    freed = 0
    remove_up_to = 0

    for i, group in enumerate(groups):
        group_tokens = sum(estimator.estimate_message(m) for m in group)
        freed += group_tokens
        remove_up_to = i + 1
        if freed >= tokens_to_free:
            break

    remaining: list[AgentMessage] = [system_msg]
    for group in groups[remove_up_to:]:
        remaining.extend(group)
    remaining.extend(tail)
    return remaining


__all__ = [
    "DEFAULT_CONTEXT_THRESHOLD_RATIO",
    "PER_CONVERSATION_TAIL",
    "PER_MESSAGE_OVERHEAD",
    "AgentFunctionCall",
    "AgentMessage",
    "AgentToolCall",
    "TokenEncoder",
    "TokenEstimator",
    "compress_context",
    "group_tool_messages",
]
