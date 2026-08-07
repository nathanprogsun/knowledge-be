"""Unit tests for chunk-set validation.

Covers every rejection path (empty output, single chunk for a large
document, too many tiny chunks, all chunks far below target, chunk above 2x
the size budget) plus the permissive cases the validator must accept.
"""

from __future__ import annotations

from src.core.knowledge.documents.chunker.splitter import Chunk
from src.core.knowledge.documents.chunker.validator import validate_chunks


def _chunks(*lengths: int) -> list[Chunk]:
    return [Chunk(content="a" * length) for length in lengths]


class TestValidationRejections:
    def test_rejects_empty_chunk_set(self) -> None:
        # Arrange / Act
        result = validate_chunks([], total_chars=1000, chunk_size=500)

        # Assert
        assert result.ok is False
        assert result.reason == "no chunks produced"

    def test_rejects_single_chunk_for_large_document(self) -> None:
        # Arrange: one chunk that is 10x the target -> splitter never split.
        result = validate_chunks(_chunks(5000), total_chars=5000, chunk_size=500)

        # Act / Assert
        assert result.ok is False
        assert result.reason == "single chunk for large document"

    def test_rejects_chunk_exceeding_two_times_target(self) -> None:
        # Arrange: second chunk 5x the size budget.
        result = validate_chunks(_chunks(100, 5000), total_chars=5100, chunk_size=1000)

        # Act / Assert
        assert result.ok is False
        assert result.reason == "chunk exceeds 2x target size"

    def test_rejects_too_many_tiny_chunks(self) -> None:
        # Arrange: 8 tiny chunks plus a tail; 8 > 9//4=2 and 8 > 2.
        result = validate_chunks(
            _chunks(10, 10, 10, 10, 10, 10, 10, 10, 5), total_chars=85, chunk_size=512
        )

        # Act / Assert
        assert result.ok is False
        assert result.reason == "too many tiny chunks"

    def test_rejects_when_all_chunks_far_below_target(self) -> None:
        # Arrange: max chunk is 1/5 of target while the doc exceeds the budget.
        result = validate_chunks(
            _chunks(100, 100, 100, 100, 100, 100), total_chars=600, chunk_size=512
        )

        # Act / Assert
        assert result.ok is False
        assert result.reason == "all chunks far below target size"


class TestValidationAcceptance:
    def test_accepts_reasonable_output(self) -> None:
        # Arrange: three chunks near the target size.
        result = validate_chunks(_chunks(480, 510, 460), total_chars=1500, chunk_size=512)

        # Act / Assert
        assert result.ok is True
        assert result.reason == ""

    def test_tolerates_tiny_last_chunk(self) -> None:
        # Arrange: the tail residue is exempt from the tiny-chunk rule.
        result = validate_chunks(_chunks(480, 510, 4), total_chars=994, chunk_size=512)

        # Act / Assert
        assert result.ok is True

    def test_accepts_single_chunk_within_budget(self) -> None:
        # Arrange: one chunk whose doc fits the budget -> not the "no split" case.
        result = validate_chunks(_chunks(400), total_chars=400, chunk_size=512)

        # Act / Assert
        assert result.ok is True

    def test_accepts_few_tiny_chunks_below_threshold(self) -> None:
        # Arrange: 2 tiny chunks among 6; 2 > 6//4=1 holds but 2 > 2 is false.
        result = validate_chunks(
            _chunks(480, 480, 480, 10, 10, 10), total_chars=1470, chunk_size=512
        )

        # Act / Assert
        assert result.ok is True
