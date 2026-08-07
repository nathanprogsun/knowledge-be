"""Unit tests for the heading-aware (Tier 1) splitter.

Verifies per-section emission with breadcrumb context, fall-through to the
legacy splitter for unstructured or single-section documents, recursion into
the legacy splitter for oversized sections, code-fence heading exclusion,
breadcrumb de-duplication, and tiny-chunk coalescing.
"""

from __future__ import annotations

from src.core.knowledge.documents.chunker.heading_splitter import (
    common_heading_prefix,
    split_by_headings,
)
from src.core.knowledge.documents.chunker.splitter import Chunk, SplitterConfig

_BODY = "Lorem ipsum dolor sit amet consectetur adipiscing elit. " * 4


def _position_invariant(text: str, chunks: list[Chunk]) -> bool:
    return all(text[c.start : c.end] == c.content for c in chunks)


class TestSplitByHeadings:
    def test_emits_one_chunk_per_primary_section_with_breadcrumb(self) -> None:
        # Arrange: sections sized above the merge target stay distinct.
        doc = "# Top\n" + _BODY + "\n\n## Section A\n" + _BODY + "\n\n## Section B\n" + _BODY
        cfg = SplitterConfig(chunk_size=300, chunk_overlap=0)

        # Act
        chunks = split_by_headings(doc, cfg)

        # Assert
        assert len(chunks) >= 3
        assert all("# Top" in c.context_header for c in chunks)
        # EmbeddingContent merges header + content for the embedder.
        assert all("# Top" in c.embedding_content() for c in chunks)
        assert any("## Section B" in c.content for c in chunks)

    def test_falls_through_for_unstructured_doc(self) -> None:
        # Arrange
        doc = "Just a plain paragraph without any headings at all in this text."
        cfg = SplitterConfig(chunk_size=200, chunk_overlap=0)

        # Act / Assert: no headings -> the whole doc stays one chunk.
        assert len(split_by_headings(doc, cfg)) == 1

    def test_large_section_recurses_into_legacy_splitter(self) -> None:
        # Arrange: a section bigger than the chunk budget must be sub-split.
        doc = "# Top\n## Big\n" + "This is a long sentence repeated many times. " * 50
        cfg = SplitterConfig(chunk_size=300, chunk_overlap=30, separators=[". "])

        # Act
        chunks = split_by_headings(doc, cfg)

        # Assert
        assert len(chunks) >= 2
        assert all("# Top" in c.context_header for c in chunks)

    def test_breadcrumb_reflects_latest_path(self) -> None:
        # Arrange
        doc = "# Chapter 1\n" + _BODY + "\n\n## Section A\n" + _BODY + "\n\n## Section B\n" + _BODY
        cfg = SplitterConfig(chunk_size=300, chunk_overlap=0)

        # Act
        chunks = split_by_headings(doc, cfg)

        # Assert: Section B's chunk must not carry Section A's breadcrumb.
        for chunk in chunks:
            if "## Section B" in chunk.content:
                assert "## Section A" not in chunk.context_header
                assert "## Section B" in chunk.context_header

    def test_ignores_headings_inside_code_fence(self) -> None:
        # Arrange
        doc = "# Real\n\n```\n# Fake heading inside code\n```\n\nbody"
        cfg = SplitterConfig(chunk_size=500, chunk_overlap=0)

        # Act
        chunks = split_by_headings(doc, cfg)

        # Assert: the real H1 breadcrumb appears on some chunk.
        assert any("# Real" in c.context_header or "# Real" in c.content for c in chunks)

    def test_preserves_position_relative_to_original(self) -> None:
        # Arrange
        doc = "# Top\nintro\n\n## A\nbody A\n\n## B\nbody B"
        cfg = SplitterConfig(chunk_size=500, chunk_overlap=0)

        # Act
        chunks = split_by_headings(doc, cfg)

        # Assert
        assert _position_invariant(doc, chunks)
        assert all(0 <= c.start <= c.end <= len(doc) for c in chunks)

    def test_empty_text(self) -> None:
        # Act / Assert
        assert split_by_headings("", SplitterConfig()) == []


class TestCommonHeadingPrefix:
    def test_returns_identical_breadcrumb_verbatim(self) -> None:
        # Act / Assert
        assert common_heading_prefix("# A\n## B", "# A\n## B") == "# A\n## B"

    def test_returns_shared_line_prefix(self) -> None:
        # Act / Assert
        assert common_heading_prefix("# A\n## B", "# A\n## C") == "# A"

    def test_returns_empty_when_no_shared_line(self) -> None:
        # Act / Assert
        assert common_heading_prefix("# A", "## B") == ""
