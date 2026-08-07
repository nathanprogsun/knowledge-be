"""Unit tests for the legacy recursive text splitter.

Covers separator-based splitting, protected-region atomicity, rune-offset
position tracking, the semantic overlap boundary logic, two-level parent/
child splitting, and image-reference extraction.
"""

from __future__ import annotations

import pytest

from src.core.knowledge.documents.chunker.splitter import (
    SplitterConfig,
    extract_image_refs,
    find_semantic_overlap_boundary,
    find_semantic_overlap_boundary_ending_at_or_after,
    split_by_separators,
    split_text,
    split_text_parent_child,
    unwrap_linked_images,
)


class TestSplitText:
    def test_splits_ascii_document_into_bounded_chunks(self) -> None:
        # Arrange
        text = "Hello world.\n\n" * 30
        cfg = SplitterConfig(chunk_size=100, chunk_overlap=20, separators=["\n\n"])

        # Act
        chunks = split_text(text, cfg)

        # Assert
        assert len(chunks) == 5

    def test_empty_text_returns_no_chunks(self) -> None:
        # Act / Assert
        assert split_text("", SplitterConfig()) == []

    def test_single_char_chinese(self) -> None:
        # Arrange: a single CJK character must not crash the splitter.
        chunks = split_text("中", SplitterConfig(chunk_size=10, chunk_overlap=0))

        # Act / Assert
        assert len(chunks) == 1
        assert chunks[0].content == "中"

    def test_chinese_offsets_are_rune_based(self) -> None:
        # Arrange
        text = "这是一段中文文本，用于测试分块。" * 8  # noqa: RUF001
        cfg = SplitterConfig(chunk_size=20, chunk_overlap=5, separators=["\n\n", "\n", "。"])
        chunks = split_text(text, cfg)

        # Assert: every chunk slices back to the original text exactly.
        for chunk in chunks:
            assert chunk.end - chunk.start == len(chunk.content)
            assert text[chunk.start : chunk.end] == chunk.content

    def test_protected_code_block_stays_atomic(self) -> None:
        # Arrange: the fenced block is far larger than chunk_size.
        text = 'intro\n\n```python\nprint("hello")\nprint("world")\n```\n\noutro'
        cfg = SplitterConfig(chunk_size=15, chunk_overlap=0, separators=["\n\n", "\n"])

        # Act
        chunks = split_text(text, cfg)

        # Assert: the code fence is never split.
        assert len(chunks) == 3
        assert chunks[1].content.startswith("```python")
        assert chunks[1].content.endswith("```")

    def test_laplace_math_block_stays_atomic(self) -> None:
        # Arrange
        text = "intro $$x^2 + y^2 = z^2$$ outro"
        cfg = SplitterConfig(chunk_size=10, chunk_overlap=0, separators=[" "])

        # Act
        chunks = split_text(text, cfg)

        # Assert: the $$...$$ block survives intact.
        assert any("$$x^2 + y^2 = z^2$$" in chunk.content for chunk in chunks)

    def test_keeps_separator_within_unit(self) -> None:
        # Arrange / Act
        pieces = split_by_separators("a\n\nb\n\nc", ["\n\n"], chunk_size=2)

        # Assert: separators are retained so reconstruction is lossless.
        assert "".join(pieces) == "a\n\nb\n\nc"

    def test_position_invariant_for_table_documents(self) -> None:
        # Arrange: a Markdown table spanning many chunks.
        text = "| 姓名 | 年龄 |\n| --- | --- |\n" + "".join(
            f"| 用户{i} | {i} |\n" for i in range(200)
        )
        cfg = SplitterConfig(chunk_size=40, chunk_overlap=5, separators=["\n\n", "\n"])
        chunks = split_text(text, cfg)

        # Act / Assert: table rows remain contiguous and reconstruct exactly.
        # Chunks may carry a synthetic prepended header (a zero-width source
        # unit), in which case the source slice is a suffix of the content.
        assert len(chunks) > 1
        for chunk in chunks:
            original = text[chunk.start : chunk.end]
            assert chunk.content == original or chunk.content.endswith(original)


class TestSemanticOverlapBoundary:
    @pytest.mark.parametrize(
        ("text", "want"),
        [
            ("第一句。第二句\n普通换行后的内容\n\n最后一段", "最后一段"),
            ("第一句。第二句。第三句", "第二句。第三句"),
            ("第一段。\r\n\r\n第二段", "第二段"),
            ("第一句。第一行\n第二行", "第二行"),
            ("第一句。第一行\r\n第二行", "第二行"),
            ("第一句？第二句", "第二句"),  # noqa: RUF001
            ("第一句！第二句", "第二句"),  # noqa: RUF001
            ("First sentence. Second sentence", "Second sentence"),
            ("Question? Answer", "Answer"),
            ("Warning! Continue", "Continue"),
        ],
    )
    def test_returns_tail_after_selected_boundary(self, text: str, want: str) -> None:
        # Act
        end, ok = find_semantic_overlap_boundary(text)

        # Assert
        assert ok is True
        assert text[end:] == want

    @pytest.mark.parametrize(
        "text",
        [
            "没有任何语义分隔符的连续文本",
            "只有，逗号；分号：冒号",  # noqa: RUF001
            "version1.2 remains one unit",
            "address 192.168.1.1 remains one unit",
            "see https://ex.com?q=1&foo=bar",
        ],
    )
    def test_finds_no_boundary_without_separators(self, text: str) -> None:
        # Act
        _, ok = find_semantic_overlap_boundary(text)

        # Assert
        assert ok is False

    @pytest.mark.parametrize(
        "text",
        [
            '代码是 `fmt.Println("hello. world")` 后续内容',
            '代码块 ```go\nfmt.Println("hello. world")\n``` 后续内容',
            "公式 $$x. y$$ 后续内容",
        ],
    )
    def test_ignores_protected_content(self, text: str) -> None:
        # Act: a "." inside code/math is protected, so no boundary is found.
        _, ok = find_semantic_overlap_boundary(text)

        # Assert
        assert ok is False

    def test_filters_eligibility_before_priority(self) -> None:
        # Arrange: the first sentence is ineligible (before min_end).
        end, ok = find_semantic_overlap_boundary_ending_at_or_after("a。bc。XYZ", min_end=4)

        # Assert: the eligible sentence wins.
        assert ok is True
        assert "a。bc。XYZ"[end:] == "XYZ"

    def test_ineligible_paragraph_does_not_outrank_eligible_sentence(self) -> None:
        # Act
        end, ok = find_semantic_overlap_boundary_ending_at_or_after("\n\nx? tail", min_end=3)

        # Assert
        assert ok is True
        assert "\n\nx? tail"[end:] == "tail"


class TestSplitTextParentChild:
    def test_children_are_globally_sequenced_and_offset_shifted(self) -> None:
        # Arrange
        text = "Paragraph one. " * 100
        parent_cfg = SplitterConfig(chunk_size=400, chunk_overlap=40)
        child_cfg = SplitterConfig(chunk_size=100, chunk_overlap=20)

        # Act
        result = split_text_parent_child(text, parent_cfg, child_cfg)

        # Assert
        assert result.children
        seqs = [child.chunk.seq for child in result.children]
        assert seqs == list(range(len(seqs)))
        for child in result.children:
            assert text[child.chunk.start : child.chunk.end] == child.chunk.content

    def test_parent_index_links_children_to_parents(self) -> None:
        # Arrange
        text = "Sentence. " * 200
        result = split_text_parent_child(
            text,
            parent_cfg=SplitterConfig(chunk_size=400, chunk_overlap=40),
            child_cfg=SplitterConfig(chunk_size=100, chunk_overlap=20),
        )

        # Act / Assert: every valid parent index resolves within parents.
        for child in result.children:
            if child.parent_index >= 0:
                assert child.parent_index < len(result.parents)

    def test_empty_text(self) -> None:
        # Act / Assert
        result = split_text_parent_child("", SplitterConfig(), SplitterConfig())
        assert result.parents == []
        assert result.children == []


class TestImageRefs:
    def test_extracts_flat_image_references(self) -> None:
        # Arrange
        text = "Before ![alt](https://example.com/a.png) after"

        # Act
        refs = extract_image_refs(text)

        # Assert
        assert len(refs) == 1
        assert refs[0].original_ref == "https://example.com/a.png"
        assert refs[0].alt_text == "alt"

    def test_captures_balanced_parentheses_in_url(self) -> None:
        # Arrange / Act
        refs = extract_image_refs("![c](https://e.com/item_(abc)/1)")

        # Assert
        assert refs[0].original_ref == "https://e.com/item_(abc)/1"

    def test_unwraps_linked_images(self) -> None:
        # Arrange / Act
        result = unwrap_linked_images("before [![alt](img.png)](https://link.example) after")

        # Assert
        assert result == "before ![alt](img.png) after"
