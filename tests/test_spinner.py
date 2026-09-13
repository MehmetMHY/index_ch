"""Tests for retrieve/spinner.py."""

import sys
from unittest.mock import MagicMock, patch

from retrieve.spinner import (
    Spinner,
    start_startup_spinner,
    stop_startup_spinner,
    erase_wrapped_input,
)


class TestStopStartupSpinner:
    def test_idempotent_when_none(self):
        # should not raise even if no spinner was started
        stop_startup_spinner()

    def test_stops_if_running(self):
        start_startup_spinner("test")
        stop_startup_spinner()
        stop_startup_spinner()  # double-stop is safe


class TestSpinnerExit:
    def test_width_empty_message(self):
        spinner = Spinner("")
        # when message is empty, width for clearing is 1
        assert len(spinner.message) == 0

    def test_width_with_message(self):
        spinner = Spinner("Loading")
        assert spinner.message == "Loading"

    def test_context_manager_no_tty(self):
        # when stdout is not a TTY, the spinner thread should not start
        spinner = Spinner("test")
        with patch.object(sys, "stdout") as mock_stdout:
            mock_stdout.isatty.return_value = False
            with spinner:
                pass
            # should not have started a thread
            assert spinner._thread is None


class TestEraseWrappedInput:
    def _capture(self, cols, prompt, text):
        writes = []
        mock_stdout = MagicMock()
        mock_stdout.isatty.return_value = True
        mock_stdout.write = lambda s: writes.append(s)
        mock_stdout.flush = lambda: None
        with patch.object(sys, "stdout", mock_stdout):
            with patch("retrieve.spinner.shutil.get_terminal_size") as ts:
                ts.return_value = MagicMock(columns=cols)
                erase_wrapped_input(prompt, text)
        return "".join(writes)

    def test_no_tty_is_noop(self):
        with patch.object(sys, "stdout") as mock_stdout:
            mock_stdout.isatty.return_value = False
            erase_wrapped_input("> ", "hello")
            mock_stdout.write.assert_not_called()

    def test_short_input_no_erase(self):
        # single-line input: leave the prompt+query visible (no escape written)
        out = self._capture(80, "> ", "hello")
        assert out == ""

    def test_wrapped_input_erases(self):
        # prompt(2) + 50 chars on a 40-col terminal => 2 lines => erase
        out = self._capture(40, "> ", "x" * 50)
        assert "\033[2A" in out
        assert "\033[J" in out

    def test_ansi_in_prompt_ignored_for_width(self):
        # color codes must not count toward visible width
        prompt = "\033[1m\033[34m> \033[0m"
        out = self._capture(40, prompt, "x" * 50)
        assert "\033[2A" in out
