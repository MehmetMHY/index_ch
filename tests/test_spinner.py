"""Tests for retrieve/spinner.py."""

import sys
from unittest.mock import MagicMock, patch

from retrieve.spinner import Spinner, start_startup_spinner, stop_startup_spinner


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
