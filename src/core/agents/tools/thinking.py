"""Thinking tools: sequential reasoning, think-block stripping, think streaming.

:class:`SequentialThinkingTool` is a stateful agent tool that records a
chain-of-thought step and reports progress (thought number, total estimate,
branch list, unfinished-step flag) back to the model. Each invocation
appends to the tool's in-memory thought history and branch registry so the
model can revise, branch, or extend its reasoning across turns.

The module also ports the two think-tag helpers used when a model embeds
``<think>...</think>`` reasoning directly in its ``content`` field instead
of a separate reasoning channel:

- :func:`strip_think_blocks` — remove completed think blocks from an
  accumulated string (also trimming the leftover whitespace);
- :class:`ThinkStreamSplitter` — the streaming counterpart that separates
  thinking text from answer text chunk-by-chunk, buffering tag fragments
  that straddle two chunks until the next ``feed``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import cast

from src.ai.embedding.base import Context
from src.common.json import JsonObject, JsonValue
from src.core.agents.tools.base import TOOL_THINKING, ToolResult

logger = logging.getLogger(__name__)

#: Inline reasoning markers some models embed in their ``content`` field.
THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"

#: Matches ``<think>...</think>`` blocks; ``(?s)`` makes ``.`` match newlines.
_THINK_BLOCK_RE = re.compile(r"(?s)<think>.*?</think>")

_THINKING_DESCRIPTION = """A detailed tool for dynamic and reflective problem-solving through thoughts.

This tool helps analyze problems through a flexible thinking process that can adapt and evolve.

Each thought can build on, question, or revise previous insights as understanding deepens.

## When to Use This Tool

- Breaking down complex problems into steps
- Planning and design with room for revision
- Analysis that might need course correction
- Problems where the full scope might not be clear initially
- Problems that require a multi-step solution
- Tasks that need to maintain context over multiple steps
- Situations where irrelevant information needs to be filtered out

## Key Features

- You can adjust total_thoughts up or down as you progress
- You can question or revise previous thoughts
- You can add more thoughts even after reaching what seemed like the end
- You can express uncertainty and explore alternative approaches
- Not every thought needs to build linearly - you can branch or backtrack
- Generates a solution hypothesis
- Verifies the hypothesis based on the Chain of Thought steps
- Repeats the process until satisfied
- When your thinking is complete, deliver your answer by writing it as your plain reply and stopping (no further tool calls). NEVER include the final answer directly in a thought.

## Parameters Explained

- **thought**: Your current thinking step, which can include:
  * Regular analytical steps
  * Revisions of previous thoughts
  * Questions about previous decisions
  * Realizations about needing more analysis
  * Changes in approach
  * Hypothesis generation
  * Hypothesis verification

  **CRITICAL - User-Friendly Thinking**: Write your thoughts in natural, user-friendly language. NEVER mention tool names (like "grep_chunks", "knowledge_search", "web_search", etc.) in your thinking process. Instead, describe your actions in plain language:
  - ❌ BAD: "I'll use grep_chunks to search for keywords, then knowledge_search for semantic understanding"
  - ✅ GOOD: "I'll start by searching for key terms in the knowledge base, then explore related concepts"
  - ❌ BAD: "After grep_chunks returns results, I'll use knowledge_search"
  - ✅ GOOD: "After finding relevant documents, I'll search for semantically related content"

  Write thinking as if explaining your reasoning to a user, not documenting technical steps. Focus on WHAT you're trying to find and WHY, not HOW (which tools you'll use).

- **next_thought_needed**: True if you need more thinking, even if at what seemed like the end
- **thought_number**: Current number in sequence (can go beyond initial total if needed)
- **total_thoughts**: Current estimate of thoughts needed (can be adjusted up/down)
- **is_revision**: A boolean indicating if this thought revises previous thinking
- **revises_thought**: If is_revision is true, which thought number is being reconsidered
- **branch_from_thought**: If branching, which thought number is the branching point
- **branch_id**: Identifier for the current branch (if any)
- **needs_more_thoughts**: If reaching end but realizing more thoughts needed

## Best Practices

1. Start with an initial estimate of needed thoughts, but be ready to adjust
2. Feel free to question or revise previous thoughts
3. Don't hesitate to add more thoughts if needed, even at the "end"
4. Express uncertainty when present
5. Mark thoughts that revise previous thinking or branch into new paths
6. Ignore information that is irrelevant to the current step
7. Generate a solution hypothesis when appropriate
8. Verify the hypothesis based on the Chain of Thought steps
9. Repeat the process until satisfied with the solution
10. Only set next_thought_needed to false when truly done and a satisfactory answer is reached
11. NEVER include the final answer in the thought content. When thinking is complete, deliver the final answer by writing it as your plain reply and stopping (no further tool calls)"""

_THINKING_SCHEMA: JsonObject = {
    "type": "object",
    "properties": {
        "thought": {
            "type": "string",
            "description": (
                "Your current thinking step. Write in natural, user-friendly language. "
                'NEVER mention tool names (like "grep_chunks", "knowledge_search", '
                '"web_search", etc.). Instead, describe actions in plain language '
                '(e.g., "I\'ll search for key terms" instead of "I\'ll use grep_chunks"). '
                "Focus on WHAT you're trying to find and WHY, not HOW (which tools you'll use)."
            ),
        },
        "next_thought_needed": {
            "type": "boolean",
            "description": "Whether another thought step is needed",
        },
        "thought_number": {
            "type": "integer",
            "description": "Current thought number (numeric value, e.g., 1, 2, 3)",
            "minimum": 1,
        },
        "total_thoughts": {
            "type": "integer",
            "description": "Estimated total thoughts needed (numeric value, e.g., 5, 10)",
            "minimum": 1,
        },
        "is_revision": {
            "type": "boolean",
            "description": "Whether this revises previous thinking",
        },
        "revises_thought": {
            "type": "integer",
            "description": "Which thought is being reconsidered",
            "minimum": 1,
        },
        "branch_from_thought": {
            "type": "integer",
            "description": "Branching point thought number",
            "minimum": 1,
        },
        "branch_id": {
            "type": "string",
            "description": "Branch identifier",
        },
        "needs_more_thoughts": {
            "type": "boolean",
            "description": "If more thoughts are needed",
        },
    },
    "required": ["thought", "next_thought_needed", "thought_number", "total_thoughts"],
}


@dataclass(frozen=True, slots=True)
class SequentialThinkingInput:
    """Parsed input for the sequential thinking tool."""

    thought: str = ""
    next_thought_needed: bool = False
    thought_number: int = 0
    total_thoughts: int = 0
    is_revision: bool = False
    revises_thought: int | None = None
    branch_from_thought: int | None = None
    branch_id: str = ""
    needs_more_thoughts: bool = False

    @classmethod
    def from_json(cls, raw: JsonObject) -> SequentialThinkingInput:
        """Build the input from a parsed JSON object, applying defaults."""
        return cls(
            thought=_as_str(raw.get("thought")),
            next_thought_needed=_as_bool(raw.get("next_thought_needed")),
            thought_number=_as_int(raw.get("thought_number")),
            total_thoughts=_as_int(raw.get("total_thoughts")),
            is_revision=_as_bool(raw.get("is_revision")),
            revises_thought=_as_int_or_none(raw.get("revises_thought")),
            branch_from_thought=_as_int_or_none(raw.get("branch_from_thought")),
            branch_id=_as_str(raw.get("branch_id")),
            needs_more_thoughts=_as_bool(raw.get("needs_more_thoughts")),
        )


@dataclass(frozen=True, slots=True)
class _ThinkingState:
    """Immutable accumulated state of one thinking session."""

    thought_history: tuple[SequentialThinkingInput, ...] = ()
    branches: dict[str, tuple[SequentialThinkingInput, ...]] = field(default_factory=dict)

    def record(self, item: SequentialThinkingInput) -> _ThinkingState:
        """Return a new state with ``item`` appended to history and its branch."""
        history = (*self.thought_history, item)
        branches = self.branches
        if item.branch_from_thought is not None and item.branch_id:
            existing = branches.get(item.branch_id, ())
            branches = {**branches, item.branch_id: (*existing, item)}
        return _ThinkingState(thought_history=history, branches=branches)


class SequentialThinkingTool:
    """Dynamic, reflective problem-solving tool with per-session state."""

    def __init__(self) -> None:
        self._state = _ThinkingState()

    def name(self) -> str:
        return TOOL_THINKING

    def description(self) -> str:
        return _THINKING_DESCRIPTION

    def parameters(self) -> str:
        return json.dumps(_THINKING_SCHEMA, ensure_ascii=False)

    async def execute(self, ctx: Context, args: str) -> ToolResult:
        """Record one thought step and report progress to the model."""
        del ctx
        input_, parse_error = _parse_thinking_args(args)
        if parse_error is not None:
            return ToolResult(success=False, error=parse_error)

        validation_error = _validate_thought(input_)
        if validation_error is not None:
            logger.error("[Tool][SequentialThinking] Validation failed: %s", validation_error)
            return ToolResult(success=False, error=validation_error)

        # Adjust totalThoughts if thoughtNumber exceeds it.
        if input_.thought_number > input_.total_thoughts:
            input_ = _with_total_thoughts(input_, input_.thought_number)

        self._state = self._state.record(input_)

        logger.debug("[Tool][SequentialThinking] %s", input_.thought)

        branch_keys = sorted(self._state.branches)
        incomplete = (
            input_.next_thought_needed
            or input_.needs_more_thoughts
            or input_.thought_number < input_.total_thoughts
        )

        response_data: JsonObject = {
            "thought_number": input_.thought_number,
            "total_thoughts": input_.total_thoughts,
            "next_thought_needed": input_.next_thought_needed,
            "branches": cast("list[JsonValue]", branch_keys),
            "thought_history_length": len(self._state.thought_history),
            "display_type": "thinking",
            "thought": input_.thought,
            "incomplete_steps": incomplete,
        }

        logger.info(
            "[Tool][SequentialThinking] Execute completed - Thought %d/%d",
            input_.thought_number,
            input_.total_thoughts,
        )

        output_msg = "Thought process recorded"
        if incomplete:
            output_msg = (
                "Thought process recorded - unfinished steps remain, continue exploring "
                "and calling tools"
            )
        return ToolResult(success=True, output=output_msg, data=response_data)


def _validate_thought(data: SequentialThinkingInput) -> str | None:
    """Return an error message for invalid thought data, or ``None``."""
    if data.thought == "":
        return "invalid thought: must be a non-empty string"
    if data.thought_number < 1:
        return "invalid thoughtNumber: must be >= 1"
    if data.total_thoughts < 1:
        return "invalid totalThoughts: must be >= 1"
    return None


def _with_total_thoughts(data: SequentialThinkingInput, total: int) -> SequentialThinkingInput:
    """Return ``data`` with ``total_thoughts`` raised to ``total``."""
    return SequentialThinkingInput(
        thought=data.thought,
        next_thought_needed=data.next_thought_needed,
        thought_number=data.thought_number,
        total_thoughts=total,
        is_revision=data.is_revision,
        revises_thought=data.revises_thought,
        branch_from_thought=data.branch_from_thought,
        branch_id=data.branch_id,
        needs_more_thoughts=data.needs_more_thoughts,
    )


def _parse_thinking_args(args: str) -> tuple[SequentialThinkingInput, str | None]:
    """Parse tool args; ``(input, error_message)`` with exactly one set."""
    try:
        parsed = json.loads(args)
    except json.JSONDecodeError as exc:
        return SequentialThinkingInput(), f"Failed to parse args: {exc}"
    if not isinstance(parsed, dict):
        return SequentialThinkingInput(), "Failed to parse args: expected a JSON object"
    return SequentialThinkingInput.from_json(cast(JsonObject, parsed)), None


# ── Think-tag helpers ──────────────────────────────────────────────────


def strip_think_blocks(content: str) -> str:
    """Remove ``<think>...</think>`` blocks, trimming leftover whitespace."""
    if content == "":
        return ""
    cleaned = _THINK_BLOCK_RE.sub("", content)
    return _trim_whitespace(cleaned)


def _trim_whitespace(text: str) -> str:
    return text.strip(" \n\r\t")


class ThinkStreamSplitter:
    """Split inline ``<think>...</think>`` reasoning from answer text live.

    Incremental counterpart of :func:`strip_think_blocks`: each :meth:`feed`
    returns the portions now unambiguously thinking text and answer text.
    Tag fragments straddling two chunks (e.g. ``<thi`` then ``nk>``) are
    buffered until the next feed. Call :meth:`flush` at end-of-stream to
    drain any buffered remainder; an unterminated ``<think>`` block is
    treated as thinking text.

    Not safe for concurrent use; create one splitter per stream.
    """

    def __init__(self) -> None:
        self._in_think = False
        self._pending = ""

    def feed(self, s: str) -> tuple[str, str]:
        """Consume one chunk; return ``(thinking, answer)`` text so far.

        Either return value may be empty. Bytes that could still be part of a
        tag spanning into the next chunk are buffered internally.
        """
        if s == "":
            return "", ""
        self._pending += s

        think_parts: list[str] = []
        answer_parts: list[str] = []
        while True:
            if self._in_think:
                close_idx = self._pending.find(THINK_CLOSE_TAG)
                if close_idx >= 0:
                    think_parts.append(self._pending[:close_idx])
                    self._pending = self._pending[close_idx + len(THINK_CLOSE_TAG) :]
                    self._in_think = False
                    continue
                safe, hold = hold_back_partial_tag(self._pending, THINK_CLOSE_TAG)
                think_parts.append(safe)
                self._pending = hold
                return "".join(think_parts), "".join(answer_parts)

            open_idx = self._pending.find(THINK_OPEN_TAG)
            if open_idx >= 0:
                answer_parts.append(self._pending[:open_idx])
                self._pending = self._pending[open_idx + len(THINK_OPEN_TAG) :]
                self._in_think = True
                continue
            safe, hold = hold_back_partial_tag(self._pending, THINK_OPEN_TAG)
            answer_parts.append(safe)
            self._pending = hold
            return "".join(think_parts), "".join(answer_parts)

    def flush(self) -> tuple[str, str]:
        """Drain any buffered remainder at end-of-stream."""
        rest = self._pending
        self._pending = ""
        if rest == "":
            return "", ""
        if self._in_think:
            return rest, ""
        return "", rest


def hold_back_partial_tag(text: str, tag: str) -> tuple[str, str]:
    """Split ``text`` into the safe-to-emit prefix and a held tag fragment.

    The trailing suffix is held back when it is a proper prefix of ``tag``
    (and so might complete into a real tag on the next chunk). When no such
    prefix matches, the whole string is safe and ``hold`` is empty.
    """
    max_k = len(tag) - 1
    if max_k > len(text):
        max_k = len(text)
    for k in range(max_k, 0, -1):
        if text.endswith(tag[:k]):
            return text[: len(text) - k], text[len(text) - k :]
    return text, ""


def _as_str(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


def _as_bool(value: JsonValue) -> bool:
    return value is True


def _as_int(value: JsonValue) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _as_int_or_none(value: JsonValue) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


__all__ = [
    "THINK_CLOSE_TAG",
    "THINK_OPEN_TAG",
    "SequentialThinkingInput",
    "SequentialThinkingTool",
    "ThinkStreamSplitter",
    "hold_back_partial_tag",
    "strip_think_blocks",
]
