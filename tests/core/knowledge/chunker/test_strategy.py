"""Unit tests for the strategy resolver and public entry points.

Covers tier selection per strategy hint, validator-driven fall-through,
config defaulting with token-limit clamping, breadcrumb merging, diagnostics
tracing, and two-level parent/child splitting.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.core.knowledge.documents.chunker.profiler import StrategyTier
from src.core.knowledge.documents.chunker.splitter import SplitterConfig, split_text
from src.core.knowledge.documents.chunker.strategy import (
    STRATEGY_AUTO,
    STRATEGY_HEADING,
    STRATEGY_LEGACY,
    ensure_defaults,
    merge_breadcrumbs,
    split,
    split_parent_child,
    split_with_diagnostics,
)


class TestSplit:
    def test_empty_text_returns_no_chunks(self) -> None:
        # Act / Assert
        assert split("", SplitterConfig()) == []

    def test_legacy_strategy_matches_split_text(self) -> None:
        # Arrange
        text = "Hello world.\n\n" * 30
        cfg = SplitterConfig(
            chunk_size=100, chunk_overlap=20, separators=["\n\n"], strategy=STRATEGY_LEGACY
        )

        # Act
        a = split(text, cfg)
        b = split_text(text, cfg)

        # Assert
        assert [c.content for c in a] == [c.content for c in b]

    def test_empty_strategy_equals_legacy(self) -> None:
        # Arrange
        text = "Sentence one. Sentence two.\n" * 20
        cfg = SplitterConfig(chunk_size=80, chunk_overlap=10)

        # Act
        a = split(text, cfg)
        b = split(text, replace(cfg, strategy=STRATEGY_LEGACY))

        # Assert
        assert [c.content for c in a] == [c.content for c in b]

    def test_heading_strategy_keeps_distinct_top_level_headings(self) -> None:
        # Arrange
        doc = "# Intro\nshort intro.\n\n# Usage\nshort usage.\n\n# FAQ\nshort faq."
        cfg = SplitterConfig(chunk_size=500, chunk_overlap=0, strategy=STRATEGY_HEADING)

        # Act
        chunks = split(doc, cfg)

        # Assert: one chunk per top-level heading, each carrying its heading.
        assert len(chunks) == 3
        for chunk, heading in zip(chunks, ["# Intro", "# Usage", "# FAQ"], strict=True):
            assert heading in chunk.content

    def test_preserves_position_invariant_across_tiers(self) -> None:
        # Arrange: one document per adaptive tier.
        cases = {
            "heading-tier": "# Top\nintro paragraph here.\n\n## Section A\nbody A here.\n\n## Section B\nbody B here.\n\n## Section C\nbody C.",
            "heuristic-tier": "Kapitel 1: Einleitung\n"
            + "Beispieltext. " * 50
            + "\n\nKapitel 2: Hauptteil\n"
            + "Mehr Text. " * 50,
            "recursive-tier": "plain prose without structure. " * 100,
        }
        cfg = SplitterConfig(
            chunk_size=300,
            chunk_overlap=30,
            separators=["\n\n", "\n", "。", ". "],
            strategy=STRATEGY_AUTO,
        )

        for name, doc in cases.items():
            # Act
            chunks = split(doc, cfg)

            # Assert: End-Start == len(Content) and runes[Start:End] == Content.
            assert chunks, f"{name}: expected chunks"
            for chunk in chunks:
                assert chunk.end - chunk.start == len(chunk.content), name
                assert doc[chunk.start : chunk.end] == chunk.content, name

    def test_falls_through_when_tier_output_is_rejected(self) -> None:
        # Arrange: many tiny distinct sections trip the validator's tiny-chunk
        # rule, so the chain falls through to the legacy tier.
        doc = "".join(f"# Section {i}\nbody {i}\n\n" for i in range(20))
        cfg = SplitterConfig(chunk_size=200, chunk_overlap=0, strategy=STRATEGY_HEADING)

        # Act
        chunks = split(doc, cfg)

        # Assert: legacy re-packs the sections into a valid chunk-set.
        assert 0 < len(chunks) < 20


class TestSplitWithDiagnostics:
    def test_reports_selected_tier_for_auto_strategy(self) -> None:
        # Arrange
        doc = "# A\nbody\n## B\nbody\n## C\nbody\n## D\nbody"
        cfg = SplitterConfig(chunk_size=200, chunk_overlap=20, strategy=STRATEGY_AUTO)

        # Act
        _, diag = split_with_diagnostics(doc, cfg)

        # Assert: heading tier wins and the chain includes the legacy fallback.
        assert diag.selected_tier == StrategyTier.HEADING
        assert diag.tier_chain == (StrategyTier.HEADING, StrategyTier.LEGACY)
        assert diag.profile is not None

    def test_reports_legacy_for_explicit_legacy_strategy(self) -> None:
        # Arrange / Act
        _, diag = split_with_diagnostics("plain text", SplitterConfig(strategy=STRATEGY_LEGACY))

        # Assert
        assert diag.selected_tier == StrategyTier.LEGACY

    def test_empty_text_yields_legacy_diagnostics(self) -> None:
        # Act
        chunks, diag = split_with_diagnostics("", SplitterConfig())

        # Assert: an empty trace never carries an unset tier.
        assert chunks == []
        assert diag.selected_tier == StrategyTier.LEGACY

    def test_records_rejected_tiers_with_reasons(self) -> None:
        # Arrange: many tiny sections -> the heading tier's output is rejected
        # by the validator, then legacy produces the final chunk-set.
        doc = "".join(f"# Section {i}\nbody {i}\n\n" for i in range(20))
        cfg = SplitterConfig(chunk_size=200, chunk_overlap=0, strategy=STRATEGY_HEADING)

        # Act
        chunks, diag = split_with_diagnostics(doc, cfg)

        # Assert
        assert len(diag.rejected) == 1
        assert diag.rejected[0].tier == StrategyTier.HEADING
        assert diag.rejected[0].reason == "too many tiny chunks"
        assert diag.selected_tier == StrategyTier.LEGACY
        assert chunks


class TestEnsureDefaults:
    def test_fills_zero_config_with_defaults(self) -> None:
        # Arrange / Act
        cfg = ensure_defaults(SplitterConfig())

        # Assert
        assert cfg.chunk_size == 512
        assert cfg.chunk_overlap == 80
        assert cfg.separators == ["\n\n", "\n", "。"]

    def test_token_limit_clamps_chunk_size(self) -> None:
        # Arrange: a huge chunk size plus a small token limit.
        cfg = ensure_defaults(SplitterConfig(chunk_size=10000, token_limit=100, languages=["en"]))

        # Act / Assert: 100 tokens * 4 chars/token * 0.9 ~ 360 chars.
        assert cfg.chunk_size < 1000
        assert cfg.chunk_overlap < cfg.chunk_size

    def test_token_limit_chinese_budget_is_tighter(self) -> None:
        # Arrange
        en = ensure_defaults(SplitterConfig(token_limit=200, languages=["en"]))
        zh = ensure_defaults(SplitterConfig(token_limit=200, languages=["zh"]))

        # Assert: Chinese consumes more chars per token, so the budget is smaller.
        assert zh.chunk_size < en.chunk_size

    def test_no_token_limit_keeps_chunk_size(self) -> None:
        # Arrange / Act
        cfg = ensure_defaults(SplitterConfig(chunk_size=800))

        # Assert
        assert cfg.chunk_size == 800

    def test_caps_overlap_at_half_of_chunk_size(self) -> None:
        # Arrange: pathological overlap of 400 with chunk size 500.
        cfg = ensure_defaults(SplitterConfig(chunk_size=500, chunk_overlap=400))

        # Act / Assert
        assert cfg.chunk_overlap == 250


class TestMergeBreadcrumbs:
    @pytest.mark.parametrize(
        ("parent", "child", "want"),
        [
            ("", "## Sub", "## Sub"),
            ("# Top", "", "# Top"),
            ("", "", ""),
            ("# Top", "## Other", "# Top\n## Other"),
            ("# Top\n## A", "## A\n### A1", "# Top\n## A\n### A1"),
            ("# Top", "# Top", "# Top"),
            ("# Top\n## A", "  ## A  \n### A1", "# Top\n## A\n### A1"),
        ],
    )
    def test_merges_and_deduplicates_seam(self, parent: str, child: str, want: str) -> None:
        # Act / Assert
        assert merge_breadcrumbs(parent, child) == want


class TestSplitParentChild:
    def test_auto_strategy_enriches_child_breadcrumbs(self) -> None:
        # Arrange: child splitter re-detects sub-headings inside a parent.
        body = "Lorem ipsum dolor sit amet. " * 40
        doc = "# Chapter\n" + body + "\n\n## Section A\n" + body + "\n\n## Section B\n" + body
        seps = ["\n\n", "\n", ". "]
        parent_cfg = SplitterConfig(
            chunk_size=800, chunk_overlap=80, strategy=STRATEGY_AUTO, separators=seps
        )
        child_cfg = SplitterConfig(
            chunk_size=200, chunk_overlap=20, strategy=STRATEGY_AUTO, separators=seps
        )

        # Act
        result = split_parent_child(doc, parent_cfg, child_cfg)

        # Assert: at least one child carries a sub-heading breadcrumb.
        assert result.children
        assert any("## Section" in child.chunk.context_header for child in result.children)

    def test_no_duplicate_breadcrumb_lines(self) -> None:
        # Arrange
        body = "Lorem ipsum dolor sit amet. " * 40
        doc = "# Chapter\n" + body + "\n\n## Section A\n" + body
        seps = ["\n\n", "\n", ". "]
        result = split_parent_child(
            doc,
            parent_cfg=SplitterConfig(
                chunk_size=800, chunk_overlap=80, strategy=STRATEGY_AUTO, separators=seps
            ),
            child_cfg=SplitterConfig(
                chunk_size=200, chunk_overlap=20, strategy=STRATEGY_AUTO, separators=seps
            ),
        )

        # Act / Assert
        for child in result.children:
            lines = child.chunk.context_header.split("\n")
            for i in range(1, len(lines)):
                assert not (lines[i].strip() != "" and lines[i] == lines[i - 1]), (
                    child.chunk.context_header
                )

    def test_legacy_strategy_produces_valid_parent_links(self) -> None:
        # Arrange
        text = "This is a sentence. Another one.\n\n" * 50
        result = split_parent_child(
            text,
            parent_cfg=SplitterConfig(chunk_size=400, chunk_overlap=40, strategy=STRATEGY_LEGACY),
            child_cfg=SplitterConfig(chunk_size=100, chunk_overlap=20, strategy=STRATEGY_LEGACY),
        )

        # Act / Assert
        assert result.children
        for child in result.children:
            assert child.parent_index < len(result.parents)

    def test_empty_text(self) -> None:
        # Act / Assert
        result = split_parent_child("", SplitterConfig(), SplitterConfig())
        assert result.parents == []
        assert result.children == []
