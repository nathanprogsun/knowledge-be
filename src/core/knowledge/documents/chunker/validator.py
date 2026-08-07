"""Chunk-set validation for the adaptive text chunker.

Inspects a tier's output and decides whether it is good enough to ship or
whether the strategy chain should fall through to the next tier. The
validator is intentionally permissive: a single "obviously broken" output is
rejected, but plausible-looking variation is accepted so we don't oscillate
between tiers.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.knowledge.documents.chunker.splitter import Chunk


@dataclass(frozen=True)
class ValidationResult:
    """Captures the verdict and reason for a chunk-set."""

    ok: bool
    reason: str = ""


def validate_chunks(chunks: list[Chunk], total_chars: int, chunk_size: int) -> ValidationResult:
    """Check whether ``chunks`` form a usable result for the given document.

    Returns OK when no broken-output indicator triggers.
    """
    if not chunks:
        return ValidationResult(ok=False, reason="no chunks produced")

    # A single chunk for a document much larger than chunk_size means the
    # strategy did not actually split — fail so the next tier runs.
    if len(chunks) == 1 and total_chars > 2 * chunk_size:
        return ValidationResult(ok=False, reason="single chunk for large document")

    # Compute size statistics.
    max_len = 0
    for chunk in chunks:
        length = len(chunk.content)
        if length > max_len:
            max_len = length

    # All but the last chunk should carry meaningful content. The last chunk
    # may be tiny because tail residue is normal.
    tiny_count = 0
    for i, chunk in enumerate(chunks):
        if i == len(chunks) - 1:
            continue
        if len(chunk.content) < 50:
            tiny_count += 1
    if tiny_count > len(chunks) // 4 and tiny_count > 2:
        return ValidationResult(ok=False, reason="too many tiny chunks")

    # Reject when no chunk reached at least 25% of the target — the splitter
    # is fragmenting too aggressively to be useful.
    if max_len < chunk_size // 4 and total_chars > chunk_size:
        return ValidationResult(ok=False, reason="all chunks far below target size")

    # Sanity check on absolute upper bound. Anything past 2x chunk_size is a
    # red flag — the splitter ignored its size budget.
    if max_len > 2 * chunk_size and chunk_size > 0:
        return ValidationResult(ok=False, reason="chunk exceeds 2x target size")

    return ValidationResult(ok=True)


__all__ = ["ValidationResult", "validate_chunks"]
