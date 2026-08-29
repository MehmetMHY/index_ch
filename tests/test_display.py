"""Tests for retrieve/display.py: truncate, preview, chat_preview, epoch fns,
timestamp formatting, time filter labels, format_help."""

import pytest

from retrieve.display import (
    truncate,
    preview,
    chat_preview,
    filename_epoch,
    chat_epoch,
    format_timestamp,
    format_list_timestamp,
    time_filter_label,
    time_filter_desc,
    format_help,
    DANGLING_WORDS,
    TIME_LABELS,
)
from retrieve.state import Session
from config import TOP_K, TIME_RANGES


class TestTruncate:
    def test_under_limit_no_ellipsis(self):
        assert truncate("short text", 100) == "short text"

    def test_exact_limit_no_ellipsis(self):
        text = "a" * 50
        assert truncate(text, 50) == text

    def test_sentence_cut(self):
        text = "This is a sentence. This is another sentence that goes on."
        result = truncate(text, 50)
        assert result.endswith("...")
        assert "This is a sentence." in result

    def test_word_cut(self):
        text = "word " * 30  # 150 chars
        result = truncate(text, 50)
        assert result.endswith("...")
        assert not result.endswith(" ...")

    def test_dangling_word_dropped(self):
        # construct text where word cut ends on a dangling word
        text = "The best approach is to work with a"
        result = truncate(text, 30)
        assert (
            not result.rstrip(".").rstrip("...").split()[-1].lower() in DANGLING_WORDS
        )

    def test_multiple_dangling_words_popped(self):
        text = "going to the store and then to the" + " x" * 50
        result = truncate(text, 40)
        # the last word of the truncated result should NOT be a dangling word
        words = result.rstrip(".").rstrip(".").rstrip().split()
        if words:
            assert words[-1].strip(",;:()").lower() not in DANGLING_WORDS

    def test_no_spaces(self):
        text = "a" * 200
        result = truncate(text, 50)
        assert result.endswith("...")

    def test_empty(self):
        assert truncate("", 100) == ""

    def test_always_ends_with_ellipsis_when_truncated(self):
        text = "x" * 200
        result = truncate(text, 50)
        assert "..." in result


class TestPreview:
    def test_empty(self):
        assert preview("") == ""
        assert preview(None) == ""

    def test_strips_headers(self):
        text = "# Title\n\n## Subtitle\n\nReal content here."
        result = preview(text)
        assert "Title" not in result or "Real content" in result
        assert "Real content" in result

    def test_strips_horizontal_rules(self):
        text = "---\n\n===\n\nReal paragraph."
        result = preview(text)
        assert "Real paragraph" in result

    def test_strips_bold_and_code(self):
        text = "This is **bold** and `code` here."
        result = preview(text)
        assert "**" not in result
        assert "`" not in result

    def test_preserves_underscores(self):
        text = "some_function_name is useful."
        result = preview(text)
        assert "some_function_name" in result

    def test_truncation(self):
        text = "word " * 200
        result = preview(text, limit=50)
        assert result.endswith("...")
        assert len(result) <= 55  # some slack for ellipsis

    def test_multi_line_paragraph_joined(self):
        text = "Line one.\nLine two.\n\nSecond paragraph."
        result = preview(text)
        assert "Line one." in result
        assert "Line two." in result
        assert "Second paragraph" not in result

    def test_only_headers_fallback(self):
        text = "# Title\n## Sub"
        result = preview(text)
        # should not crash; may return joined text
        assert isinstance(result, str)


class TestChatPreview:
    def test_short_summary_present(self):
        info = {"short_summary": "A short summary.", "summary": "Long summary."}
        result = chat_preview(info)
        assert "A short summary" in result

    def test_short_summary_missing_fallback(self):
        info = {"short_summary": None, "summary": "Long detailed summary."}
        result = chat_preview(info)
        assert "Long detailed summary" in result

    def test_short_summary_whitespace_falls_back(self):
        info = {"short_summary": "   \n  ", "summary": "Real summary."}
        result = chat_preview(info)
        assert "Real summary" in result

    def test_short_summary_normalized_whitespace(self):
        info = {"short_summary": "  extra   spaces  here  ", "summary": "x"}
        result = chat_preview(info)
        assert "extra spaces here" in result


class TestFilenameEpoch:
    def test_typical(self):
        assert filename_epoch("ch_session_1700000000.json") == 1700000000

    def test_no_digits(self):
        assert filename_epoch("ch_session_no_epoch.json") is None

    def test_short_digits_ignored(self):
        assert filename_epoch("ch_123.json") is None  # only 3 digits

    def test_with_path(self):
        assert filename_epoch("/some/dir/ch_session_1700000000.json") == 1700000000

    def test_multiple_digit_runs(self):
        assert filename_epoch("ch_session_1700000000_v2.json") == 1700000000


class TestChatEpoch:
    def test_last_message_epoch_present(self):
        info = {
            "last_message_epoch": 1700000500,
            "file_path": "ch_session_1700000000.json",
        }
        assert chat_epoch(info) == 1700000500

    def test_falls_back_to_filename(self):
        info = {"last_message_epoch": None, "file_path": "ch_session_1700000000.json"}
        assert chat_epoch(info) == 1700000000

    def test_zero_falls_back(self):
        info = {"last_message_epoch": 0, "file_path": "ch_session_1700000000.json"}
        assert chat_epoch(info) == 1700000000

    def test_both_none(self):
        info = {"last_message_epoch": None, "file_path": "no_epoch.json"}
        assert chat_epoch(info) is None


class TestFormatTimestamp:
    def test_none(self):
        assert format_timestamp(None) == ""

    def test_zero(self):
        assert format_timestamp(0) == ""

    def test_valid_epoch(self):
        ts = format_timestamp(1700000000)
        assert "UTC" in ts
        assert "2023" in ts  # 1700000000 is Nov 14, 2023

    def test_negative(self):
        # very large negative overflows
        assert format_timestamp(-99999999999999) == ""

    def test_huge_overflow(self):
        assert format_timestamp(10**18) == ""


class TestFormatListTimestamp:
    def test_none(self):
        assert format_list_timestamp(None) == ""

    def test_zero(self):
        assert format_list_timestamp(0) == ""

    def test_valid_epoch(self):
        ts = format_list_timestamp(1700000000)
        assert "Z" in ts
        assert "\u2022" in ts  # bullet character

    def test_overflow(self):
        assert format_list_timestamp(10**18) == ""


class TestTimeFilterLabel:
    def test_none(self):
        assert time_filter_label(None) is None

    def test_tuple(self):
        assert time_filter_label((100, 200)) == "custom"

    def test_str(self):
        assert time_filter_label("1d") == "1d"


class TestTimeFilterDesc:
    def test_none(self):
        assert time_filter_desc(None) == "all time"

    def test_tuple(self):
        desc = time_filter_desc((1700000000, 1700001000))
        assert "to" in desc
        assert "UTC" in desc

    def test_str(self):
        assert time_filter_desc("1d") == TIME_LABELS["1d"]

    def test_unknown_str_raises(self):
        with pytest.raises(KeyError):
            time_filter_desc("nonexistent")


class TestFormatHelp:
    def test_renders_all_fields(self, sample_session):
        result = format_help(sample_session)
        assert "rerank:" in result
        assert "expansion:" in result
        assert "archived:" in result
        assert "time:" in result
        assert "results:" in result

    def test_toggles(self, sample_session):
        sample_session.do_rerank = False
        sample_session.do_expand = False
        sample_session.show_archived = True
        result = format_help(sample_session)
        assert "rerank:" in result and "off" in result
        assert "expansion:" in result and "off" in result
        assert "archived:" in result and "shown" in result
