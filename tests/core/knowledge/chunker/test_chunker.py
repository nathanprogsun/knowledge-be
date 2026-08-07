"""Integration tests for the adaptive text chunker.

Exercises the public package API end-to-end: the auto strategy resolves the
correct tier for heading-rich, heuristic and plain documents; table header
context is preserved across chunk boundaries; and the public exports expose
the expected surface.
"""

from __future__ import annotations

from src.core.knowledge.documents.chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    Chunk,
    LangEnglish,
    SplitterConfig,
    approx_token_count,
    default_config,
    split,
    split_text,
    validate_chunks,
)

_TABLE_HEADER = "| 姓名 | 年龄 | 城市 |\n| --- | --- | --- |\n"


class TestAdaptivePipeline:
    def test_auto_strategy_selects_heading_tier_for_markdown_doc(self) -> None:
        # Arrange: a heading-structured document.
        doc = "# Top\nbody\n## A\nbody\n## B\nbody\n## C\nbody"
        cfg = SplitterConfig(chunk_size=200, chunk_overlap=20, strategy="auto")

        # Act
        chunks = split(doc, cfg)

        # Assert: heading-aware chunks carry breadcrumb context.
        assert chunks
        assert any(c.context_header for c in chunks)
        assert all(doc[c.start : c.end] == c.content for c in chunks)

    def test_auto_strategy_selects_heuristic_tier_for_chapter_doc(self) -> None:
        # Arrange: German chapter markers but no Markdown headings.
        doc = (
            "Kapitel 1: Einleitung\n"
            + "Beispieltext. " * 40
            + "\n\nKapitel 2: Hauptteil\n"
            + "Mehr Text. " * 40
        )
        cfg = SplitterConfig(chunk_size=200, chunk_overlap=20, strategy="auto")

        # Act
        chunks = split(doc, cfg)

        # Assert
        assert len(chunks) >= 2
        assert all(doc[c.start : c.end] == c.content for c in chunks)

    def test_auto_strategy_uses_legacy_for_plain_doc(self) -> None:
        # Arrange
        doc = "plain prose without any structure. " * 20
        cfg = SplitterConfig(chunk_size=150, chunk_overlap=10, strategy="auto")

        # Act
        chunks = split(doc, cfg)

        # Assert
        assert chunks
        assert all(c.context_header == "" for c in chunks)

    def test_output_passes_validation(self) -> None:
        # Arrange
        doc = "# One\nshort\n## A\n" + "body text here. " * 20 + "\n## B\n" + "more body. " * 20
        cfg = SplitterConfig(chunk_size=300, chunk_overlap=30, strategy="auto")

        # Act
        chunks = split(doc, cfg)

        # Assert: the returned chunk-set is validator-approved.
        assert validate_chunks(chunks, total_chars=len(doc), chunk_size=300).ok is True


class TestTableHeaderContext:
    def test_header_prepended_to_later_table_chunks(self) -> None:
        # Arrange: a table spanning many chunks.
        text = (
            "前面的文字\n\n"
            + _TABLE_HEADER
            + "".join(f"| 用户{i} | {i} | 城市{i} |\n" for i in range(60))
            + "\n后面的文字"
        )
        cfg = SplitterConfig(chunk_size=60, chunk_overlap=5, separators=["\n\n", "\n"])

        # Act
        chunks = split_text(text, cfg)

        # Assert: every data chunk carries the prepended header.
        prepends = 0
        for chunk in chunks:
            if "| 用户" in chunk.content:
                assert chunk.content.startswith(_TABLE_HEADER), (
                    f"chunk missing header:\n{chunk.content[:80]}"
                )
                prepends += 1
        assert prepends > 0

    def test_table_header_does_not_leak_into_later_tables(self) -> None:
        # Arrange: two tables separated by a paragraph break.
        text = (
            _TABLE_HEADER
            + "".join(f"| 用户{i} | {i} |\n" for i in range(40))
            + "\n中间的文字\n\n"
            + "| 项目 | 状态 |\n| --- | --- |\n"
            + "".join(f"| X{i} | 完成 |\n" for i in range(40))
        )
        cfg = SplitterConfig(chunk_size=50, chunk_overlap=5, separators=["\n\n", "\n"])

        # Act
        chunks = split_text(text, cfg)

        # Assert: a chunk holding rows of table 2 never carries table 1's header.
        for chunk in chunks:
            if "| X" in chunk.content:
                assert "| 姓名" not in chunk.content
                if "| 项目 |" not in chunk.content:
                    assert chunk.content.startswith("| 项目 | 状态 |\n"), chunk.content[:80]


class TestPublicSurface:
    def test_default_config_matches_documented_constants(self) -> None:
        # Arrange / Act
        cfg = default_config()

        # Assert
        assert cfg.chunk_size == DEFAULT_CHUNK_SIZE == 512
        assert cfg.chunk_overlap == DEFAULT_CHUNK_OVERLAP == 80
        assert cfg.separators == ["\n\n", "\n", "。"]

    def test_embedding_content_merges_header_and_trimmed_body(self) -> None:
        # Arrange
        chunk = Chunk(content="  body text  ", context_header="# Top")

        # Act
        embedded = chunk.embedding_content()

        # Assert
        assert embedded == "# Top\n\nbody text"

    def test_embedding_content_without_header_trims_only(self) -> None:
        # Arrange / Act
        embedded = Chunk(content="  plain body  ").embedding_content()

        # Assert
        assert embedded == "plain body"

    def test_token_estimator_is_accessible_from_package(self) -> None:
        # Arrange / Act
        count = approx_token_count("The quick brown fox", LangEnglish)

        # Assert
        assert count > 0
