"""Tests for retrieve/cli.py: _drain_stdin and setup KeyboardInterrupt."""

import io
import sys
from unittest.mock import patch

import pytest

from retrieve.cli import _drain_stdin


class TestDrainStdin:
    def test_noop_when_not_a_tty(self):
        # stdin is a real file/pipe in the test runner, isatty() is False
        # so _drain_stdin must return without touching termios
        with patch("termios.tcflush") as mock_flush:
            _drain_stdin()
            mock_flush.assert_not_called()

    def test_flushes_when_tty(self):
        fake_stdin = io.StringIO("")
        fake_stdin.isatty = lambda: True
        fake_stdin.fileno = lambda: 0
        with patch.object(sys, "stdin", fake_stdin), patch(
            "termios.tcflush"
        ) as mock_flush, patch("termios.TCIFLUSH", 0, create=True):
            _drain_stdin()
            mock_flush.assert_called_once_with(0, 0)

    def test_swallows_tcflush_error(self):
        fake_stdin = io.TextIOWrapper(io.BytesIO(b""))
        fake_stdin.isatty = lambda: True
        fake_stdin.fileno = lambda: 0
        # tcflush raises OSError on a non-terminal fd; _drain_stdin must swallow it
        with patch.object(sys, "stdin", fake_stdin), patch(
            "termios.tcflush", side_effect=OSError("bad fd")
        ), patch("termios.TCIFLUSH", 0, create=True):
            _drain_stdin()


class TestSetupInterrupt:
    def test_ctrl_c_during_fts_load_exits_cleanly(self, monkeypatch):
        # ^C during ensure_fts/load_vectors must not traceback; it prints a
        # newline and returns 0
        import retrieve.cli as cli

        conn = cli.get_connection()
        monkeypatch.setattr(cli, "get_connection", lambda: conn)
        monkeypatch.setattr(cli, "backfill_message_epochs", lambda c: None)
        monkeypatch.setattr(cli, "backfill_archived", lambda c: None)
        monkeypatch.setattr(cli, "stop_startup_spinner", lambda: None)
        monkeypatch.setattr(cli, "_drain_stdin", lambda: None)
        monkeypatch.setattr(cli, "warm_connections", lambda: None)
        monkeypatch.setattr(
            cli, "ensure_fts", lambda c: (_ for _ in ()).throw(KeyboardInterrupt())
        )
        rc = cli.main([])
        assert rc == 0
        conn.close()
