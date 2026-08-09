"""Output budget and truncation for tool results.

The registry publishes its output ceiling into the execution scope so
budget-aware tools can shape batched results themselves; a fallback
truncation keeps the head and tail of any oversized blob. All limits are
counted in code points (matching rune semantics) so CJK text is treated
fairly.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

#: Default maximum tool output (code points). Outputs exceeding this are
#: truncated preserving the head and tail.
DEFAULT_MAX_TOOL_OUTPUT = 24000

#: Head/tail split when truncating (70% head, 30% tail).
_HEAD_RATIO = 0.7

#: Code-point budget reserved for the truncation marker itself.
_TRUNCATION_MARKER_RESERVE = 200

_output_budget_var: ContextVar[int] = ContextVar("agent_tool_output_budget", default=0)


@contextmanager
def with_output_budget(max_chars: int) -> Iterator[None]:
    """Publish ``max_chars`` as the output ceiling for the enclosing block.

    A non-positive ceiling leaves the scope unchanged (callers that execute
    a tool directly fall back to the default).
    """
    if max_chars <= 0:
        yield
        return
    token = _output_budget_var.set(max_chars)
    try:
        yield
    finally:
        _output_budget_var.reset(token)


def output_budget() -> int:
    """Return the published ceiling, falling back to the default."""
    budget = _output_budget_var.get()
    return budget if budget > 0 else DEFAULT_MAX_TOOL_OUTPUT


def truncate_tool_output(output: str, max_chars: int) -> str:
    """Truncate ``output`` to ``max_chars`` code points, keeping head + tail.

    Preserves the first 70% and the last 30% of the usable budget with a
    truncation marker between them, so headers / summaries at the start and
    conclusions at the end survive. Returns the input unchanged when it is
    already within the limit.
    """
    rune_count = len(output)
    if max_chars <= 0 or rune_count <= max_chars:
        return output

    usable = max_chars - _TRUNCATION_MARKER_RESERVE
    if usable <= 0:
        return output[:max_chars]

    head_size = int(usable * _HEAD_RATIO)
    tail_size = usable - head_size
    if tail_size <= 0:
        tail_size = 0

    marker = (
        f"\n\n... [output truncated: {rune_count} → {max_chars} chars, "
        f"showing first {head_size} + last {tail_size}] ...\n\n"
    )
    if tail_size == 0:
        return output[:head_size] + marker
    return output[:head_size] + marker + output[-tail_size:]


def split_budget_fairly(total: int, sizes: list[int]) -> list[int]:
    """Max-min fair (water-filling) allocation of ``total`` across sizes.

    Entries smaller than an equal share keep their full size and donate the
    slack to the larger entries, so a batched result degrades by trimming
    its biggest records rather than dropping whole ones. The returned caps
    never exceed the corresponding size and their sum never exceeds
    ``total``.
    """
    caps = [0] * len(sizes)
    if not sizes or total <= 0:
        return caps
    settled = [False] * len(sizes)
    remaining = total
    unsettled = len(sizes)
    while unsettled > 0:
        share = remaining // unsettled
        if share <= 0:
            break
        progressed = False
        for i, size in enumerate(sizes):
            if settled[i] or size > share:
                continue
            caps[i] = size
            settled[i] = True
            remaining -= size
            unsettled -= 1
            progressed = True
        if not progressed:
            # Every unsettled entry exceeds the fair share, so the remainder
            # splits evenly and the allocation is complete.
            for i in range(len(sizes)):
                if not settled[i]:
                    caps[i] = share
            break
    return caps


__all__ = [
    "DEFAULT_MAX_TOOL_OUTPUT",
    "output_budget",
    "split_budget_fairly",
    "truncate_tool_output",
    "with_output_budget",
]
