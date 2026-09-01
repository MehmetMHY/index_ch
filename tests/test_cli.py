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
        monkeypatch.setattr(cli, "warm", lambda: None)
        monkeypatch.setattr(
            cli, "ensure_fts", lambda c: (_ for _ in ()).throw(KeyboardInterrupt())
        )
        rc = cli.main([])
        assert rc == 0
        conn.close()


class TestPromptInterrupt:
    """Ctrl+C at the prompt re-prompts; Ctrl+D exits."""

    def _setup_cli(self, monkeypatch):
        import retrieve.cli as cli
        import sqlite3

        conn = sqlite3.connect(":memory:")
        monkeypatch.setattr(cli, "get_connection", lambda: conn)
        monkeypatch.setattr(cli, "backfill_message_epochs", lambda c: None)
        monkeypatch.setattr(cli, "backfill_archived", lambda c: None)
        monkeypatch.setattr(cli, "stop_startup_spinner", lambda: None)
        monkeypatch.setattr(cli, "_drain_stdin", lambda: None)
        monkeypatch.setattr(cli, "warm_connections", lambda: None)
        monkeypatch.setattr(cli, "warm", lambda: None)
        monkeypatch.setattr(cli, "ensure_fts", lambda c: None)
        import numpy as np

        monkeypatch.setattr(
            cli,
            "load_vectors",
            lambda c: (np.array([], dtype=int), np.array([]).reshape(0, 1), {}),
        )
        return cli

    def test_ctrl_c_reprompts_not_exit(self, monkeypatch):
        cli = self._setup_cli(monkeypatch)
        # first input raises KeyboardInterrupt (re-prompt), second raises
        # EOFError (exit) so the loop terminates and we can assert the return
        inputs = iter([KeyboardInterrupt(), EOFError()])

        def fake_input(prompt):
            raise next(inputs)

        monkeypatch.setattr("builtins.input", fake_input)
        with patch("builtins.print"):
            rc = cli.main([""])
        assert rc == 0  # exited via EOF on the second call, not via crash

    def test_ctrl_d_exits_immediately(self, monkeypatch):
        cli = self._setup_cli(monkeypatch)

        def fake_input(prompt):
            raise EOFError()

        monkeypatch.setattr("builtins.input", fake_input)
        with patch("builtins.print"):
            rc = cli.main([""])
        assert rc == 0

    def test_ctrl_c_during_search_reprompts(self, monkeypatch):
        """Ctrl+C during search/rerank must re-prompt, not crash."""
        cli = self._setup_cli(monkeypatch)
        # first call: a real query that triggers search -> KeyboardInterrupt
        # second call: EOFError to exit the loop
        inputs = iter(["stock picks", EOFError()])

        def fake_input(prompt):
            val = next(inputs)
            if isinstance(val, str):
                return val
            raise val

        def fake_search(*a, **kw):
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", fake_input)
        monkeypatch.setattr(cli, "search", fake_search)
        monkeypatch.setattr(
            cli, "Spinner", lambda *a, **kw: __import__("contextlib").nullcontext()
        )
        with patch("builtins.print"):
            rc = cli.main([""])
        assert rc == 0  # exited via EOF on second input, not via crash

    def test_ctrl_c_during_command_reprompts(self, monkeypatch):
        """Ctrl+C during any command handler must re-prompt, not crash."""
        cli = self._setup_cli(monkeypatch)
        inputs = iter(["/ls", EOFError()])

        def fake_input(prompt):
            val = next(inputs)
            if isinstance(val, str):
                return val
            raise val

        def fake_ls(*a, **kw):
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", fake_input)
        monkeypatch.setattr(cli, "handle_ls", fake_ls)
        with patch("builtins.print"):
            rc = cli.main([""])
        assert rc == 0

    def test_run_false_exits_repl(self, monkeypatch):
        cli = self._setup_cli(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda prompt: "/run 1")
        monkeypatch.setattr(cli, "handle_run", lambda session, args: False)
        with patch("builtins.print"):
            rc = cli.main([""])
        assert rc == 0

    def test_dump_false_exits_repl(self, monkeypatch):
        cli = self._setup_cli(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda prompt: "/dump 1")
        monkeypatch.setattr(cli, "handle_dump", lambda session, args: False)
        with patch("builtins.print"):
            rc = cli.main([""])
        assert rc == 0

    def test_ls_false_exits_repl(self, monkeypatch):
        cli = self._setup_cli(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda prompt: "/ls")
        monkeypatch.setattr(cli, "handle_ls", lambda conn, arch, tf, in_search: False)
        with patch("builtins.print"):
            rc = cli.main([""])
        assert rc == 0


class TestStartupCommands:
    def test_startup_ls_calls_handle_ls_without_prompt(self, monkeypatch):
        import retrieve.cli as cli
        import sqlite3

        conn = sqlite3.connect(":memory:")
        monkeypatch.setattr(cli, "get_connection", lambda: conn)
        monkeypatch.setattr(cli, "backfill_message_epochs", lambda c: None)
        monkeypatch.setattr(cli, "backfill_archived", lambda c: None)
        monkeypatch.setattr(cli, "stop_startup_spinner", lambda: None)
        monkeypatch.setattr(cli, "_drain_stdin", lambda: None)

        with patch("retrieve.cli.handle_ls") as mock_handle_ls:
            rc = cli.main(["ls"])
            assert rc == 0
            mock_handle_ls.assert_called_once_with(conn, False, None, in_search=False)
