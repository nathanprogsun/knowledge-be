"""Markdown heading nesting tracking for the adaptive text chunker.

The heading-aware splitter uses this to prepend a breadcrumb like
``# Top > ## Section > ### Subsection`` to each chunk. Conceptually similar
to the table header tracker but a Markdown heading has no explicit end — it
is ended by the next heading of equal or shallower depth, so a level-stack
models it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from src.core.knowledge.documents.chunker.patterns import MARKDOWN_HEADING_PATTERN


@dataclass(frozen=True)
class HeadingHierarchy:
    """Stack of active Markdown headings indexed by level (1..6).

    Pushing a level-N heading pops every entry of level >= N because the
    previous siblings/descendants are no longer in scope. Instances are
    immutable: :meth:`observe` returns the updated hierarchy.
    """

    # stack[i] holds the heading text for level i+1 (so stack[0] = H1).
    # Entries beyond the deepest active level are empty strings.
    stack: tuple[str, ...] = ("", "", "", "", "", "")
    depth: int = 0  # current deepest active level (0 if no active heading)

    def observe(self, line: str) -> tuple[int, str, HeadingHierarchy]:
        """Parse ``line`` and update the hierarchy if it is a Markdown heading.

        Returns ``(level, heading_text, new_state)``; ``level == 0`` and an
        empty heading text when the line is not a heading. Lines that look
        like headings inside fenced code blocks are NOT detected here —
        callers must avoid feeding code-block content to ``observe``.
        """
        m = MARKDOWN_HEADING_PATTERN.search(line)
        if m is None:
            return 0, "", self
        level = len(m.group(1))
        if level < 1 or level > 6:
            return 0, "", self
        heading = m.group(2).strip()
        # Replace this level and clear deeper ones — siblings/descendants of
        # the previous heading at this level are no longer in scope.
        new_stack = list(self.stack)
        new_stack[level - 1] = heading
        for i in range(level, 6):
            new_stack[i] = ""
        if level > self.depth:
            new_depth = level
        else:
            # Recompute depth: it might shrink if we just pushed a shallower
            # heading.
            new_depth = 0
            for i in range(6):
                if new_stack[i] != "":
                    new_depth = i + 1
        updated = replace(self, stack=tuple(new_stack), depth=new_depth)
        return level, heading, updated

    def breadcrumb(self) -> str:
        """Current heading path joined by `` > ``, e.g. ``Chapter 1 > Section 2``.

        Returns "" when no headings are active.
        """
        if self.depth == 0:
            return ""
        parts = [self.stack[i] for i in range(self.depth) if self.stack[i] != ""]
        return " > ".join(parts)

    def breadcrumb_with_hashes(self) -> str:
        """Path with the original ``#`` prefixes for embedding as a context header.

        Example: ``"# Chapter 1\\n## Section 2\\n### Subsection a"``.
        """
        if self.depth == 0:
            return ""
        lines: list[str] = []
        for i in range(self.depth):
            if self.stack[i] == "":
                continue
            lines.append("#" * (i + 1) + " " + self.stack[i])
        return "\n".join(lines)


__all__ = ["HeadingHierarchy"]
