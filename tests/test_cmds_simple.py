"""Tests for retrieve/cmds/simple.py: parse_time_token, handle_len, handle_time."""

import pytest
from unittest.mock import patch

from retrieve.cmds.simple import (
    parse_time_token,
    handle_len,
    handle_time,
    handle_run,
    handle_view,
    handle_copy,
    RESULT_LEN_MIN,
    RESULT_LEN_MAX,
    LEN_USAGE,
)
from retrieve.state import Session
from config import TIME_RANGES, TOP_K


class TestParseTimeToken:
    def test_all(self):
        assert parse_time_token("all") == ("set", None)

    def test_off(self):
        assert parse_time_token("off") == ("set", None)

    def test_none(self):
        assert parse_time_token("none") == ("set", None)

    def test_custom(self):
        assert parse_time_token("custom") == ("custom", None)

    def test_custom_uppercase(self):
        assert parse_time_token("CUSTOM") == ("custom", None)

    def test_all_time_ranges(self):
        for key in TIME_RANGES:
            assert parse_time_token(key) == ("set", key)

    def test_time_ranges_uppercase(self):
        assert parse_time_token("1D") == ("set", "1d")

    def test_unknown(self):
        assert parse_time_token("garbage") == ("error", None)

    def test_empty(self):
        assert parse_time_token("") == ("error", None)


class TestHandleLen:
    def test_no_args_prints_current(self, sample_session, capsys):
        original = sample_session.result_len
        handle_len(sample_session, [])
        assert sample_session.result_len == original  # no mutation
        out = capsys.readouterr().out
        assert str(original) in out

    def test_non_numeric_no_mutation(self, sample_session, capsys):
        original = sample_session.result_len
        handle_len(sample_session, ["abc"])
        assert sample_session.result_len == original
        assert "Usage" in capsys.readouterr().out

    def test_zero_out_of_range(self, sample_session, capsys):
        original = sample_session.result_len
        handle_len(sample_session, ["0"])
        assert sample_session.result_len == original
        assert "Usage" in capsys.readouterr().out

    def test_min_boundary(self, sample_session, capsys):
        handle_len(sample_session, ["1"])
        assert sample_session.result_len == 1

    def test_max_boundary(self, sample_session, capsys):
        handle_len(sample_session, ["25"])
        assert sample_session.result_len == 25

    def test_above_max(self, sample_session, capsys):
        original = sample_session.result_len
        handle_len(sample_session, ["26"])
        assert sample_session.result_len == original
        assert "Usage" in capsys.readouterr().out

    def test_negative(self, sample_session, capsys):
        original = sample_session.result_len
        handle_len(sample_session, ["-1"])
        assert sample_session.result_len == original

    def test_valid_sets_value(self, sample_session, capsys):
        handle_len(sample_session, ["10"])
        assert sample_session.result_len == 10
        assert "set to 10" in capsys.readouterr().out

    def test_constants(self):
        assert RESULT_LEN_MIN == 1
        assert RESULT_LEN_MAX == 25


class TestHandleTime:
    def test_valid_token_sets_filter(self, sample_session, capsys):
        handle_time(sample_session, ["1d"])
        assert sample_session.time_filter == "1d"
        assert "Time filter set" in capsys.readouterr().out

    def test_all_clears_filter(self, sample_session, capsys):
        sample_session.time_filter = "1w"
        handle_time(sample_session, ["all"])
        assert sample_session.time_filter is None

    def test_error_no_mutation(self, sample_session, capsys):
        original = sample_session.time_filter
        handle_time(sample_session, ["garbage"])
        assert sample_session.time_filter == original
        assert "Usage" in capsys.readouterr().out

    def test_cancel_no_mutation(self, sample_session):
        original = sample_session.time_filter
        with patch(
            "retrieve.cmds.simple.pick_time_with_fzf", return_value=("cancel", None)
        ):
            handle_time(sample_session, [])
            assert sample_session.time_filter == original

    def test_custom_returns_none_no_mutation(self, sample_session):
        original = sample_session.time_filter
        with patch(
            "retrieve.cmds.simple.pick_time_with_fzf", return_value=("custom", None)
        ):
            with patch("retrieve.cmds.simple.run_custom_picker", return_value=None):
                handle_time(sample_session, [])
                assert sample_session.time_filter == original

    def test_custom_sets_tuple(self, sample_session, capsys):
        with patch(
            "retrieve.cmds.simple.pick_time_with_fzf", return_value=("custom", None)
        ):
            with patch(
                "retrieve.cmds.simple.run_custom_picker", return_value=(100.0, 200.0)
            ):
                handle_time(sample_session, [])
                assert sample_session.time_filter == (100.0, 200.0)

    def test_fzf_set_path(self, sample_session, capsys):
        with patch(
            "retrieve.cmds.simple.pick_time_with_fzf", return_value=("set", "1w")
        ):
            handle_time(sample_session, [])
            assert sample_session.time_filter == "1w"


class TestHandleRun:
    def test_no_pick(self, sample_session):
        with patch("retrieve.cmds.simple.resolve_pick", return_value=None):
            assert handle_run(sample_session, ["1"]) is True

    def test_missing_ch(self, sample_session):
        with patch("retrieve.cmds.simple.resolve_pick", return_value=1), patch(
            "retrieve.cmds.simple.shutil.which", return_value=None
        ):
            assert handle_run(sample_session, ["1"]) is True

    def test_cancel_confirm(self, sample_session):
        with patch("retrieve.cmds.simple.resolve_pick", return_value=1), patch(
            "retrieve.cmds.simple.shutil.which", return_value="/usr/bin/ch"
        ), patch("retrieve.cmds.simple.confirm_return", return_value=None), patch(
            "retrieve.cmds.simple.run_chat"
        ) as mock_run_chat:
            assert handle_run(sample_session, ["1"]) is True
            mock_run_chat.assert_not_called()

    def test_no_exits_repl(self, sample_session):
        with patch("retrieve.cmds.simple.resolve_pick", return_value=1), patch(
            "retrieve.cmds.simple.shutil.which", return_value="/usr/bin/ch"
        ), patch("retrieve.cmds.simple.confirm_return", return_value=False), patch(
            "retrieve.cmds.simple.run_chat"
        ) as mock_run_chat:
            assert handle_run(sample_session, ["1"]) is False
            mock_run_chat.assert_called_once_with(
                1, sample_session.meta, reprint_prompt=False
            )

    def test_yes_stays_in_repl(self, sample_session):
        with patch("retrieve.cmds.simple.resolve_pick", return_value=1), patch(
            "retrieve.cmds.simple.shutil.which", return_value="/usr/bin/ch"
        ), patch("retrieve.cmds.simple.confirm_return", return_value=True), patch(
            "retrieve.cmds.simple.run_chat"
        ) as mock_run_chat:
            assert handle_run(sample_session, ["1"]) is True
            mock_run_chat.assert_called_once_with(
                1, sample_session.meta, reprint_prompt=True
            )
