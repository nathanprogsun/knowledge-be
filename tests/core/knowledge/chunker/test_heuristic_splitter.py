"""Unit tests for the heuristic (Tier 2) splitter.

Verifies candidate boundary detection (form feeds, numbered sections, chapter
markers, all-caps headings, visual separators, page footers, blank blocks),
priority de-duplication, protected-region filtering, and the greedy
bin-packing that produces the final chunks.
"""

from __future__ import annotations

from src.core.knowledge.documents.chunker.heuristic_splitter import (
    Boundary,
    apply_overlap_aligned,
    drop_bounds_inside_spans,
    find_heuristic_boundaries,
    split_by_heuristics,
)
from src.core.knowledge.documents.chunker.splitter import Span, SplitterConfig


class TestFindHeuristicBoundaries:
    def test_detects_german_chapter_markers(self) -> None:
        # Arrange
        doc = "Kapitel 1: Einleitung\nbody\n\nKapitel 2: Hauptteil\nbody"

        # Act
        bounds = find_heuristic_boundaries(doc, [])

        # Assert: one boundary per chapter at the chapter line start.
        assert len(bounds) == 2
        assert bounds[0].rune_start == 0
        assert doc[bounds[1].rune_start :].startswith("Kapitel 2")

    def test_detects_form_feeds(self) -> None:
        # Arrange / Act
        bounds = find_heuristic_boundaries("page1\fpage2\fpage3", [])

        # Assert: the strongest single-character boundary.
        assert [b.rune_start for b in bounds] == [5, 11]
        assert all(b.priority == 100 for b in bounds)

    def test_detects_all_caps_headings(self) -> None:
        # Arrange / Act
        bounds = find_heuristic_boundaries(
            "INTRODUCTION\n\nSOME TEXT HERE\n\nMETHODS\nmore text", []
        )

        # Assert
        assert len(bounds) == 3
        assert bounds[0].rune_start == 0

    def test_detects_numbered_sections(self) -> None:
        # Arrange / Act
        bounds = find_heuristic_boundaries("1. Introduction\ntext\n\n2. Methodology\ntext", [])

        # Assert
        assert len(bounds) == 2
        assert bounds[0].rune_start == 0

    def test_detects_blank_block_boundaries(self) -> None:
        # Arrange: three consecutive newlines denote a hard section break.
        doc = "part one\n\n\npart two"
        bounds = find_heuristic_boundaries(doc, [])

        # Act / Assert: boundary lands right after the blank run.
        assert len(bounds) == 1
        assert doc[bounds[0].rune_start :] == "part two"

    def test_dedupes_offsets_keeping_highest_priority(self) -> None:
        # Arrange: a chapter line that also matches the numbered pattern
        # occupies the same offset — only the higher priority survives.
        doc = "Kapitel 1: Einführung\nbody"
        bounds = find_heuristic_boundaries(doc, [])

        # Assert
        starts = [b.rune_start for b in bounds]
        assert len(starts) == len(set(starts))

    def test_returns_empty_for_plain_text(self) -> None:
        # Act / Assert
        assert find_heuristic_boundaries("plain text without any markers here", []) == []


class TestDropBoundsInsideSpans:
    def test_drops_boundaries_strictly_inside_protected_span(self) -> None:
        # Arrange: a protected span [5, 20); a boundary inside it and one at its edge.
        bounds = [Boundary(rune_start=2), Boundary(rune_start=10), Boundary(rune_start=20)]
        spans = [Span(start=5, end=20)]

        # Act
        kept = drop_bounds_inside_spans(bounds, spans)

        # Assert: the interior boundary is removed, edge boundary is kept.
        assert [b.rune_start for b in kept] == [2, 20]

    def test_returns_input_when_no_spans(self) -> None:
        # Arrange / Act
        bounds = [Boundary(rune_start=3)]
        kept = drop_bounds_inside_spans(bounds, [])

        # Assert
        assert kept == bounds


class TestApplyOverlapAligned:
    def test_snaps_to_nearest_boundary_in_window(self) -> None:
        # Arrange: boundaries at 20 and 40 inside the overlap window.
        bounds = [
            Boundary(rune_start=0),
            Boundary(rune_start=20),
            Boundary(rune_start=40),
            Boundary(rune_start=60),
        ]

        # Act: target = 60 - 25 = 35; nearest boundary within [10, 60) is 40.
        start = apply_overlap_aligned("x" * 60, cur_end=60, overlap=25, bounds=bounds)

        # Assert
        assert start == 40

    def test_returns_zero_overlap_when_no_boundary(self) -> None:
        # Act / Assert
        assert apply_overlap_aligned("x" * 100, cur_end=100, overlap=0, bounds=[]) == 100

    def test_falls_back_to_previous_newline(self) -> None:
        # Arrange: no boundary inside the window, but a newline at index 30.
        text = "x" * 30 + "\n" + "y" * 70
        bounds = [Boundary(rune_start=0)]

        # Act: target = 100 - 40 = 60; scan back to newline at 30.
        start = apply_overlap_aligned(text, cur_end=100, overlap=40, bounds=bounds)

        # Assert
        assert start == 31


class TestSplitByHeuristics:
    def test_bin_packs_blocks_into_bounded_chunks(self) -> None:
        # Arrange: two German chapters, each larger than the budget.
        doc = (
            "Kapitel 1: Einleitung\n"
            + "Beispieltext. " * 30
            + "\n\n"
            + "Kapitel 2: Hauptteil\n"
            + "Mehr Text. " * 30
        )
        cfg = SplitterConfig(chunk_size=300, chunk_overlap=30)

        # Act
        chunks = split_by_heuristics(doc, cfg)

        # Assert
        assert len(chunks) >= 2
        for chunk in chunks:
            assert doc[chunk.start : chunk.end] == chunk.content

    def test_falls_through_when_no_boundaries_found(self) -> None:
        # Arrange
        doc = "Just a short paragraph without structural markers."
        cfg = SplitterConfig(chunk_size=100, chunk_overlap=0)

        # Act / Assert: falls through to the legacy splitter.
        assert len(split_by_heuristics(doc, cfg)) == 1

    def test_short_document_delegates_to_legacy(self) -> None:
        # Arrange: doc fits within the budget, so no boundary scan is needed.
        doc = "Kapitel 1: Einleitung\nshort body"
        cfg = SplitterConfig(chunk_size=500, chunk_overlap=0)

        # Act
        chunks = split_by_heuristics(doc, cfg)

        # Assert: exactly one chunk with the full content.
        assert len(chunks) == 1
        assert chunks[0].content == doc

    def test_empty_text(self) -> None:
        # Act / Assert
        assert split_by_heuristics("", SplitterConfig()) == []

    def test_keeps_form_feed_separated_pages(self) -> None:
        # Arrange: a long document partitioned by form feeds.
        doc = (
            "page one content. " * 8
            + "\f"
            + "page two content. " * 8
            + "\f"
            + "page three content. " * 8
        )
        cfg = SplitterConfig(chunk_size=50, chunk_overlap=0)

        # Act
        chunks = split_by_heuristics(doc, cfg)

        # Assert: form feeds partition the document.
        assert len(chunks) >= 2
        assert all(doc[c.start : c.end] == c.content for c in chunks)
