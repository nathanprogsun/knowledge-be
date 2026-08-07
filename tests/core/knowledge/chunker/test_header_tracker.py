"""Unit tests for the table header tracker.

Covers the header lifecycle (start / stay active / end), the rewrite of
empty column-name rows (``||``) from the first data row, column-width
mismatch handling, and the paragraph-break pending resolution that prevents
one table's header from leaking into a later table.
"""

from __future__ import annotations

from src.core.knowledge.documents.chunker.header_tracker import HeaderTracker


class TestHeaderTrackerLifecycle:
    def test_no_headers_before_a_table(self) -> None:
        # Arrange / Act
        tracker = HeaderTracker()
        tracker.update("Some regular text")

        # Assert
        assert tracker.get_headers() == ""

    def test_activates_on_table_header_unit(self) -> None:
        # Arrange / Act
        tracker = HeaderTracker()
        tracker.update("| A | B |\n| --- | --- |\n")

        # Assert
        assert tracker.get_headers() != ""

    def test_stays_active_during_table_rows(self) -> None:
        # Arrange
        tracker = HeaderTracker()
        tracker.update("| A | B |\n| --- | --- |\n")

        # Act: a data row must keep the header active.
        tracker.update("| 1 | 2 |\n")

        # Assert
        assert tracker.get_headers() != ""

    def test_ends_on_empty_line(self) -> None:
        # Arrange
        tracker = HeaderTracker()
        tracker.update("| A | B |\n| --- | --- |\n")
        tracker.update("| 1 | 2 |\n")

        # Act
        tracker.update("\n")

        # Assert
        assert tracker.get_headers() == ""

    def test_tracks_new_table_after_previous_ended(self) -> None:
        # Arrange
        tracker = HeaderTracker()
        tracker.update("| A | B |\n| --- | --- |\n")
        tracker.update("\n")

        # Act
        tracker.update("| X | Y |\n| --- | --- |\n")

        # Assert
        assert tracker.get_headers() != ""


class TestHeaderTrackerEmptyHeaderRow:
    def test_rewrites_empty_column_names_from_first_data_row(self) -> None:
        # Arrange: MarkItDown-style table with an empty header row.
        tracker = HeaderTracker()
        tracker.update("||\n| --- | --- | --- |\n")

        # Act: the first data row becomes the real column names.
        tracker.update("| 测试用例 ID | 测试模块 | 备注 |\n")
        header = tracker.get_headers()

        # Assert
        assert "测试用例 ID" in header
        assert "||" not in header
        assert "---" in header
        # Column names must come before the separator.
        assert header.index("测试用例 ID") < header.index("---")

    def test_does_not_absorb_subsequent_data_rows(self) -> None:
        # Arrange
        tracker = HeaderTracker()
        tracker.update("||\n| --- | --- | --- |\n")
        tracker.update("| 测试用例 ID | 测试模块 | 备注 |\n")

        # Act
        tracker.update("| TC-001 | 模块A | 备注1 |\n")
        header = tracker.get_headers()

        # Assert
        assert "TC-001" not in header

    def test_normal_header_is_not_extended(self) -> None:
        # Arrange: proper column names in the header row.
        tracker = HeaderTracker()
        tracker.update("| 姓名 | 年龄 |\n| --- | --- |\n")

        # Act: a data row must not be absorbed.
        tracker.update("| 张三 | 25 |\n")
        header = tracker.get_headers()

        # Assert
        assert "张三" not in header


class TestHeaderTrackerColumnMismatch:
    def test_ends_table_when_column_count_differs(self) -> None:
        # Arrange: a 4-column header.
        tracker = HeaderTracker()
        tracker.update("| Name | Game | Fame | Blame |\n| --- | --- | --- | --- |\n")
        assert tracker.get_headers() != ""

        # Act: a 2-column row starts a different table.
        tracker.update("| Sinple | Table |\n")

        # Assert
        assert tracker.get_headers() == ""

    def test_keeps_header_when_column_count_matches(self) -> None:
        # Arrange
        tracker = HeaderTracker()
        tracker.update("| Name | Game |\n| --- | --- |\n")

        # Act
        tracker.update("| Russell | Football |\n")

        # Assert
        assert tracker.get_headers() != ""


class TestHeaderTrackerParagraphBreak:
    def test_paragraph_break_sets_pending_then_clears_on_next_table_row(self) -> None:
        # Arrange
        tracker = HeaderTracker()
        tracker.update("| Name | Game | Fame | Blame |\n| --- | --- | --- | --- |\n")

        # Act: a row ending with \n\n must not clear the header yet.
        tracker.update("| Russell Wilson | Football | High | Tacky uniform |\n\n")

        # Assert: header still active, break is pending.
        assert tracker.get_headers() != ""
        assert tracker.pending_table_break is True

        # Act: the next unit is a new table row.
        tracker.update("| Sinple | Table |\n")

        # Assert: the previous header is cleared and a flush is signalled.
        assert tracker.get_headers() == ""
        assert tracker.header_ended_this_unit is True

    def test_paragraph_break_with_plain_text_clears_without_flush_signal(self) -> None:
        # Arrange
        tracker = HeaderTracker()
        tracker.update("| A | B |\n| --- | --- |\n")
        tracker.update("| 1 | 2 |\n\n")

        # Act: a non-table unit after the break.
        tracker.update("following text")

        # Assert
        assert tracker.get_headers() == ""
