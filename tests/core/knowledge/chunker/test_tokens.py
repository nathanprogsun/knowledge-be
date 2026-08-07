"""Unit tests for token counting and language detection.

Ports the reference token-estimator cases: per-language chars-per-token
ratios, conservative rounding, the CJK/Latin/mixed language classifier, and
the token-limit-to-char-budget conversion with its 10% safety margin.
"""

from __future__ import annotations

from src.core.knowledge.documents.chunker.tokens import (
    LangChinese,
    LangEnglish,
    LangGerman,
    LangMixed,
    approx_token_count,
    chars_for_token_limit,
    detect_language,
)


class TestApproxTokenCount:
    def test_returns_english_estimate_for_english_sentence(self) -> None:
        # Arrange: 45 ASCII chars at 4.0 chars/token ~ 11 tokens.
        text = "The quick brown fox jumps over the lazy dog."

        # Act / Assert
        assert 9 <= approx_token_count(text, LangEnglish) <= 13

    def test_returns_chinese_estimate_for_chinese_sentence(self) -> None:
        # Arrange: 18 runes at 1.7 chars/token ~ 11 tokens.
        text = "这是一段中文测试内容用于检验分词估算"

        # Act / Assert
        assert 9 <= approx_token_count(text, LangChinese) <= 12

    def test_returns_zero_for_empty_string(self) -> None:
        assert approx_token_count("", LangEnglish) == 0

    def test_returns_positive_for_unknown_language(self) -> None:
        # Act: unknown language falls back to the mixed ratio.
        assert approx_token_count("Hello world hello world", "xx") > 0

    def test_rounds_half_up(self) -> None:
        # Arrange: 2 runes at 4 chars/token = 0.5 -> clamped to 1.
        assert approx_token_count("ab", LangEnglish) == 1

    def test_never_returns_zero_for_nonempty_rune_len(self) -> None:
        # Arrange: 1 rune at 4 chars/token = 0.25 -> minimum of 1.
        assert approx_token_count("a", LangEnglish) == 1


class TestDetectLanguage:
    def test_english_for_ascii_prose(self) -> None:
        assert detect_language("The quick brown fox jumps over the lazy dog.") == LangEnglish

    def test_german_for_umlaut_text(self) -> None:
        assert (
            detect_language("Der schnelle braune Fuchs springt über den faulen Hund.") == LangGerman
        )

    def test_german_for_stopwords_without_umlauts(self) -> None:
        # Arrange: no umlauts but plenty of German function words.
        assert detect_language("Das ist ein Test und nicht mit Umlauten.") == LangGerman

    def test_chinese_for_cjk_only(self) -> None:
        assert detect_language("这是一段中文测试内容") == LangChinese

    def test_mixed_for_dual_script_text(self) -> None:
        assert detect_language("This 这是 mixed 测试 content with 多语言 inside") == LangMixed

    def test_mixed_for_empty_string(self) -> None:
        assert detect_language("") == LangMixed

    def test_mixed_for_no_latin_or_cjk(self) -> None:
        # Arrange: punctuation-only text has neither script -> mixed.
        assert detect_language("!!! ??? ---") == LangMixed


class TestCharsForTokenLimit:
    def test_applies_safety_margin(self) -> None:
        # Arrange: 1000 tokens * 4 chars/token * 0.9 = 3600.
        assert 3500 <= chars_for_token_limit(1000, LangEnglish) <= 3700

    def test_zero_tokens_gives_zero_chars(self) -> None:
        assert chars_for_token_limit(0, LangEnglish) == 0

    def test_unknown_language_falls_back_to_mixed(self) -> None:
        assert chars_for_token_limit(100, "xx") == int(100 * 3.0 * 0.9)
