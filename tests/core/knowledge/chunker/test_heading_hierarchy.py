"""Unit tests for the Markdown heading hierarchy tracker.

Verifies the level-stack semantics: linear nesting builds a breadcrumb,
sibling and top-level headings pop deeper entries, non-heading lines are
ignored, and the ``#``-prefixed breadcrumb form is produced correctly.
"""

from __future__ import annotations

from src.core.knowledge.documents.chunker.heading_hierarchy import HeadingHierarchy


def _observe(h: HeadingHierarchy, *lines: str) -> HeadingHierarchy:
    for line in lines:
        _, _, h = h.observe(line)
    return h


class TestHeadingHierarchy:
    def test_builds_breadcrumb_for_linear_nesting(self) -> None:
        # Arrange
        h = _observe(HeadingHierarchy(), "# Chapter 1", "## Section 1.1", "### Subsection 1.1.1")

        # Act / Assert
        assert h.breadcrumb() == "Chapter 1 > Section 1.1 > Subsection 1.1.1"

    def test_pops_deeper_entries_on_sibling_heading(self) -> None:
        # Arrange: a sibling H2 ends the H3 that was in scope.
        h = _observe(
            HeadingHierarchy(),
            "# Chapter 1",
            "## Section 1.1",
            "### Subsection 1.1.1",
            "## Section 1.2",
        )

        # Act / Assert
        assert h.breadcrumb() == "Chapter 1 > Section 1.2"

    def test_pops_all_entries_on_new_top_level(self) -> None:
        # Arrange
        h = _observe(HeadingHierarchy(), "# Chapter 1", "## Section A", "### Sub", "# Chapter 2")

        # Act
        breadcrumb = h.breadcrumb()

        # Assert
        assert breadcrumb == "Chapter 2"
        assert h.depth == 1

    def test_ignores_non_heading_lines(self) -> None:
        # Arrange
        h = _observe(HeadingHierarchy(), "# Title")

        # Act: a plain paragraph must not register.
        level, text, _ = h.observe("just a paragraph")

        # Assert
        assert level == 0
        assert text == ""
        assert h.breadcrumb() == "Title"

    def test_produces_hashes_breadcrumb(self) -> None:
        # Arrange
        h = _observe(HeadingHierarchy(), "# A", "## B")

        # Act / Assert
        assert h.breadcrumb_with_hashes() == "# A\n## B"

    def test_empty_state(self) -> None:
        # Arrange / Act
        h = HeadingHierarchy()

        # Assert
        assert h.breadcrumb() == ""
        assert h.breadcrumb_with_hashes() == ""
        assert h.depth == 0

    def test_handles_skipped_levels(self) -> None:
        # Arrange: document jumps from H1 directly to H3.
        h = _observe(HeadingHierarchy(), "# Top", "### Deep")

        # Act / Assert
        assert h.breadcrumb() == "Top > Deep"

    def test_observe_returns_updated_state(self) -> None:
        # Arrange
        h = HeadingHierarchy()

        # Act
        level, text, updated = h.observe("# Chapter 1")

        # Assert: the call reports the heading and returns a new state while
        # the original instance is untouched (immutability).
        assert level == 1
        assert text == "Chapter 1"
        assert updated.breadcrumb() == "Chapter 1"
        assert h.breadcrumb() == ""
