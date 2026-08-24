"""Tests for retrieve/pickers.py: resolve_pick and resolve_picks."""

import pytest
from unittest.mock import patch

from retrieve.pickers import resolve_pick, resolve_picks


@pytest.fixture
def last_results():
    return [(10, 3), (20, 2), (30, 1)]


@pytest.fixture
def meta():
    return {
        10: {
            "file_path": "/tmp/chat1.json",
            "summary": "s1",
            "short_summary": "ss1",
            "last_message_epoch": 100,
            "archived": False,
        },
        20: {
            "file_path": "/tmp/chat2.json",
            "summary": "s2",
            "short_summary": "ss2",
            "last_message_epoch": 200,
            "archived": False,
        },
        30: {
            "file_path": "/tmp/chat3.json",
            "summary": "s3",
            "short_summary": "ss3",
            "last_message_epoch": 300,
            "archived": False,
        },
    }


class TestResolvePick:
    def test_empty_results(self, last_results, meta, capsys):
        result = resolve_pick([], [], meta, "/view")
        assert result is None
        assert "No results" in capsys.readouterr().out

    def test_valid_index_first(self, last_results, meta):
        result = resolve_pick(["1"], last_results, meta, "/view")
        assert result == 10

    def test_valid_index_last(self, last_results, meta):
        result = resolve_pick(["3"], last_results, meta, "/view")
        assert result == 30

    def test_index_zero_out_of_range(self, last_results, meta, capsys):
        result = resolve_pick(["0"], last_results, meta, "/view")
        assert result is None
        assert "between" in capsys.readouterr().out

    def test_index_too_large(self, last_results, meta, capsys):
        result = resolve_pick(["4"], last_results, meta, "/view")
        assert result is None
        assert "between" in capsys.readouterr().out

    def test_non_numeric(self, last_results, meta, capsys):
        result = resolve_pick(["abc"], last_results, meta, "/view")
        assert result is None
        assert "Usage" in capsys.readouterr().out

    def test_no_args_delegates_to_fzf(self, last_results, meta):
        with patch("retrieve.pickers.pick_with_fzf", return_value=42):
            result = resolve_pick([], last_results, meta, "/view")
            assert result == 42


class TestResolvePicks:
    def test_empty_results(self, last_results, meta, capsys):
        result = resolve_picks([], [], meta, "/dump")
        assert result == []

    def test_multiple_valid(self, last_results, meta):
        result = resolve_picks(["1", "3"], last_results, meta, "/dump")
        assert result == [10, 30]

    def test_dedup_preserves_order(self, last_results, meta):
        result = resolve_picks(["1", "1", "2", "1"], last_results, meta, "/dump")
        assert result == [10, 20]

    def test_non_numeric_returns_empty(self, last_results, meta):
        result = resolve_picks(["1", "abc", "2"], last_results, meta, "/dump")
        assert result == []

    def test_out_of_range_returns_empty(self, last_results, meta):
        result = resolve_picks(["1", "5"], last_results, meta, "/dump")
        assert result == []

    def test_no_args_delegates_to_fzf(self, last_results, meta):
        with patch("retrieve.pickers.pick_many_with_fzf", return_value=[10, 20]):
            result = resolve_picks([], last_results, meta, "/dump")
            assert result == [10, 20]

    def test_boundary_one(self, last_results, meta):
        result = resolve_picks(["1"], last_results, meta, "/dump")
        assert result == [10]

    def test_boundary_last(self, last_results, meta):
        result = resolve_picks(["3"], last_results, meta, "/dump")
        assert result == [30]
