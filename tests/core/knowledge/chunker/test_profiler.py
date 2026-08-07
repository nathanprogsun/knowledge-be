"""Unit tests for document profiling and tier selection.

Verifies the structural counters (Markdown headings, numbered sections,
chapter markers, form feeds, code/table detection), line statistics, the
dominant-heading-level rule with its fallback, and the strategy-chain
selection for heading-rich, heuristic and unstructured documents.
"""

from __future__ import annotations

from src.core.knowledge.documents.chunker.profiler import (
    StrategyTier,
    profile_document,
    select_strategy,
)


class TestProfileDocument:
    def test_empty_document_has_zero_stats(self) -> None:
        # Arrange / Act
        profile = profile_document("")

        # Assert
        assert profile.total_chars == 0
        assert profile.total_lines == 0
        assert profile.heading_density() == 0.0

    def test_counts_markdown_headings_by_level(self) -> None:
        # Arrange
        doc = """# Title
Some intro text here.

## Section 1
Body of section 1.

## Section 2
Body of section 2.

### Subsection 2.1
Detail.

## Section 3
More body."""

        # Act
        profile = profile_document(doc)

        # Assert
        assert profile.md_heading_counts[1] == 1
        assert profile.md_heading_counts[2] == 3
        assert profile.md_heading_counts[3] == 1
        assert profile.md_heading_total == 5
        assert profile.dominant_heading_level() == 2  # H2 reaches 3 occurrences

    def test_dominant_level_falls_back_to_deepest_present(self) -> None:
        # Arrange: no level reaches 3 occurrences.
        doc = "# Single H1\n## H2 a\n## H2 b\n"

        # Act / Assert
        assert profile_document(doc).dominant_heading_level() == 2

    def test_counts_numbered_sections(self) -> None:
        # Arrange
        doc = """1. Introduction
text

2. Methodology
text

3. Results
text"""

        # Act / Assert
        assert profile_document(doc).numbered_section_count >= 3

    def test_counts_german_chapters(self) -> None:
        # Arrange
        doc = "Kapitel 1: Einführung\n\nText\n\nKapitel 2: Hauptteil\n\nText"

        # Act / Assert
        assert profile_document(doc).german_chapter_count == 2

    def test_counts_chinese_chapters(self) -> None:
        # Arrange
        doc = "第一章 引言\n\n内容\n\n第二章 方法\n\n内容"

        # Act / Assert
        assert profile_document(doc).chinese_chapter_count == 2

    def test_counts_form_feeds(self) -> None:
        # Arrange
        doc = "page 1 content\f\npage 2 content\f\npage 3 content"

        # Act / Assert
        assert profile_document(doc).form_feed_count == 2

    def test_detects_fenced_code_block(self) -> None:
        # Arrange
        doc = "Some prose.\n\n```go\nfunc main() {}\n```\n\nMore prose."

        # Act / Assert
        assert profile_document(doc).has_code is True

    def test_detects_table(self) -> None:
        # Arrange
        doc = "Intro.\n\n| col a | col b |\n| --- | --- |\n| 1 | 2 |\n"

        # Act / Assert
        assert profile_document(doc).has_tables is True

    def test_computes_line_statistics(self) -> None:
        # Arrange
        doc = "short\nthis is a longer line of text\nanother line here"

        # Act
        profile = profile_document(doc)

        # Assert
        assert profile.total_lines == 3
        assert profile.avg_line_len > 0

    def test_counts_excessive_blank_paragraph_breaks(self) -> None:
        # Arrange: one run of 3 newlines inside the document.
        doc = "a\nb\n\n\nc"

        # Act / Assert
        assert profile_document(doc).blank_paragraph_breaks == 1

    def test_heuristic_marker_total_sums_markers(self) -> None:
        # Arrange: 2 numbered sections + 1 visual separator = 3.
        doc = "1. One\n---\n2. Two\n"

        # Act / Assert
        assert profile_document(doc).heuristic_marker_total() == 3


class TestSelectStrategy:
    def test_picks_heading_tier_for_heading_doc(self) -> None:
        # Arrange
        doc = "# A\nbody\n## B\nbody\n## C\nbody\n## D\nbody"
        profile = profile_document(doc)

        # Act
        chain = select_strategy(profile)

        # Assert
        assert chain[0] == StrategyTier.HEADING

    def test_picks_heuristic_tier_for_chapter_doc(self) -> None:
        # Arrange: no Markdown headings, but German chapter markers.
        doc = "Kapitel 1: Foo\nbody body body\n\nKapitel 2: Bar\nbody body body\n\n"
        profile = profile_document(doc)

        # Act
        chain = select_strategy(profile)

        # Assert: heading tier is skipped entirely.
        assert chain[0] == StrategyTier.HEURISTIC

    def test_picks_legacy_for_unstructured_doc(self) -> None:
        # Arrange
        doc = "just a paragraph of plain text without any structure indicators at all here"
        profile = profile_document(doc)

        # Act
        chain = select_strategy(profile)

        # Assert
        assert chain[0] == StrategyTier.LEGACY

    def test_always_falls_back_to_legacy(self) -> None:
        # Arrange / Act: every chain must end with the legacy safety net.
        for doc in ["", "simple", "# H1\nbody"]:
            chain = select_strategy(profile_document(doc))

            # Assert
            assert chain[-1] == StrategyTier.LEGACY

    def test_none_profile_yields_legacy_only(self) -> None:
        # Act / Assert
        assert select_strategy(None) == [StrategyTier.LEGACY]
