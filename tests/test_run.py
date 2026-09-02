"""Tests for the new run.py: unified REPL entry point.

Tests the lightweight functions (prompt_update_cache, format_ts,
open_history_picker, require_ch_dirs) and the main loop's exit paths.
No fzf or child scripts are spawned: every subprocess call is mocked.
"""

import importlib
import os
import sys
import time
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture
def run_module(monkeypatch):
    """Import run.py fresh (it lives at the repo root, not under src/, so it
    is not on pytest's pythonpath). Does NOT stub require_ch_dirs so tests
    can verify it; tests that call main() stub it themselves."""
    sys.path.insert(0, "")
    if "run" in sys.modules:
        del sys.modules["run"]
    import run

    importlib.reload(run)
    yield run
    if "run" in sys.modules:
        del sys.modules["run"]


def _fzf_pick(stdout, returncode=0):
    """Return a MagicMock mimicking a fzf subprocess.run result."""
    proc = MagicMock()
    proc.stdout = stdout
    proc.returncode = returncode
    return proc


class TestFormatTs:
    """format_ts: unified MM/DD/YYYY HH:MMZ timestamp formatting."""

    def test_valid_epoch(self, run_module):
        # 1700000000 = Nov 14, 2023 22:13:20 UTC
        assert run_module.format_ts(1700000000) == "11/14/2023 22:13Z"

    def test_none_epoch(self, run_module):
        assert run_module.format_ts(None) == ""

    def test_zero_epoch(self, run_module):
        assert run_module.format_ts(0) == ""

    def test_negative_one_is_valid_epoch(self, run_module):
        # -1 = Dec 31, 1969 23:59:59 UTC, a valid timestamp
        assert run_module.format_ts(-1) == "12/31/1969 23:59Z"


class TestFilenameEpoch:
    """_filename_epoch: extract epoch from ch_session_<epoch>.json."""

    def test_valid_filename(self, run_module):
        assert (
            run_module._filename_epoch("/tmp/ch_session_1740001234.json") == 1740001234
        )

    def test_no_digits(self, run_module):
        assert run_module._filename_epoch("no_epoch_here.json") is None

    def test_empty(self, run_module):
        assert run_module._filename_epoch("") is None


class TestPromptUpdateCache:
    """prompt_update_cache: fzf yes/no picker. Returns True for yes."""

    def test_yes_returns_true(self, run_module):
        proc = _fzf_pick("yes\n")
        with patch("shutil.which", lambda b: "/usr/local/bin/fzf"), patch(
            "subprocess.run", return_value=proc
        ):
            assert run_module.prompt_update_cache() is True

    def test_no_returns_false(self, run_module):
        proc = _fzf_pick("no\n")
        with patch("shutil.which", lambda b: "/usr/local/bin/fzf"), patch(
            "subprocess.run", return_value=proc
        ):
            assert run_module.prompt_update_cache() is False

    def test_cancel_returns_false(self, run_module):
        proc = _fzf_pick("", returncode=130)
        with patch("shutil.which", lambda b: "/usr/local/bin/fzf"), patch(
            "subprocess.run", return_value=proc
        ):
            assert run_module.prompt_update_cache() is False

    def test_missing_fzf_returns_false(self, run_module):
        with patch("shutil.which", lambda b: None):
            assert run_module.prompt_update_cache() is False

    def test_no_is_first_in_list(self, run_module):
        """no should be the default (first in the list) so bare Enter skips."""
        proc = _fzf_pick("no\n")
        with patch("shutil.which", lambda b: "/usr/local/bin/fzf"), patch(
            "subprocess.run", return_value=proc
        ) as mock_run:
            run_module.prompt_update_cache()
        sent_input = mock_run.call_args.kwargs["input"]
        assert sent_input.splitlines()[0] == "no"


class TestRunCacheUpdate:
    """run_cache_update: shells out to build.py + process.py."""

    def test_runs_both_scripts(self, run_module, monkeypatch):
        calls = []
        monkeypatch.setattr(run_module, "PY_CALL", "/fake/python3")
        monkeypatch.setattr(run_module.os, "system", lambda cmd: calls.append(cmd) or 0)
        run_module.run_cache_update()
        assert len(calls) == 2
        assert "build.py" in calls[0]
        assert "process.py" in calls[1]

    def test_exits_on_failure(self, run_module, monkeypatch):
        calls = {"n": 0}

        def fake_system(cmd):
            calls["n"] += 1
            return 1 if calls["n"] == 1 else 0

        monkeypatch.setattr(run_module, "PY_CALL", "/fake/python3")
        monkeypatch.setattr(run_module.os, "system", fake_system)
        with pytest.raises(SystemExit):
            run_module.run_cache_update()


class TestOpenHistoryPicker:
    """open_history_picker: fzf picker of past queries."""

    def test_empty_history_prints_message(self, run_module):
        run_module.SEARCH_HISTORY.clear()
        with patch("shutil.which", lambda b: "/usr/local/bin/fzf"):
            result = run_module.open_history_picker()
        assert result is None

    def test_missing_fzf_returns_none(self, run_module):
        run_module.SEARCH_HISTORY.clear()
        run_module.SEARCH_HISTORY.append(("test query", "01/01/2025 00:00Z"))
        with patch("shutil.which", lambda b: None):
            result = run_module.open_history_picker()
        assert result is None

    def test_picks_query(self, run_module):
        run_module.SEARCH_HISTORY.clear()
        run_module.SEARCH_HISTORY.append(("jwt auth", "01/01/2025 12:00Z"))
        run_module.SEARCH_HISTORY.append(("sqlite fts5", "01/02/2025 13:00Z"))
        proc = _fzf_pick(" 2. 01/02/2025 13:00Z  sqlite fts5\n")
        with patch("shutil.which", lambda b: "/usr/local/bin/fzf"), patch(
            "subprocess.run", return_value=proc
        ):
            result = run_module.open_history_picker()
        assert result == "sqlite fts5"

    def test_cancel_returns_none(self, run_module):
        run_module.SEARCH_HISTORY.clear()
        run_module.SEARCH_HISTORY.append(("test query", "01/01/2025 00:00Z"))
        proc = _fzf_pick("", returncode=130)
        with patch("shutil.which", lambda b: "/usr/local/bin/fzf"), patch(
            "subprocess.run", return_value=proc
        ):
            result = run_module.open_history_picker()
        assert result is None


class TestRequireChDirs:
    """require_ch_dirs: exits if ~/.ch/ or ~/.ch/tmp/ don't exist."""

    def test_missing_ch_dir_exits(self, run_module, monkeypatch):
        monkeypatch.setattr(run_module, "CH_DIR", "/nonexistent/.ch")
        monkeypatch.setattr(run_module, "CHATS_SOURCE_DIR", "/nonexistent/.ch/tmp")
        # both dirs don't exist -> exits on first check
        monkeypatch.setattr(os.path, "isdir", lambda p: False)
        with pytest.raises(SystemExit):
            run_module.require_ch_dirs()

    def test_missing_chats_dir_exits(self, run_module, monkeypatch):
        existing = set()

        def fake_isdir(p):
            return p in existing

        monkeypatch.setattr(run_module, "CH_DIR", "/fake/.ch")
        monkeypatch.setattr(run_module, "CHATS_SOURCE_DIR", "/fake/.ch/tmp")
        existing.add("/fake/.ch")  # CH_DIR exists
        # CHATS_SOURCE_DIR does not exist
        monkeypatch.setattr(os.path, "isdir", fake_isdir)
        with pytest.raises(SystemExit):
            run_module.require_ch_dirs()


class TestOpenExplorer:
    """open_explorer: parses fzf --expect output and dispatches actions."""

    def test_enter_key_dispatches_action_open_in_ch(self, run_module, monkeypatch):
        # fzf with --expect outputs an empty string on line 1 when Enter is pressed
        proc = _fzf_pick("\n1\t0001  01/01/2025 00:00Z  ch_session_1.json\n")
        called = []
        monkeypatch.setattr(
            run_module,
            "action_open_in_ch",
            lambda cids, conn, meta, multi: called.append((cids, multi)) or True,
        )
        with patch("shutil.which", lambda b: "/usr/local/bin/fzf"), patch(
            "subprocess.run", return_value=proc
        ):
            meta = {1: {"file_path": "/tmp/ch_session_1.json"}}
            run_module.open_explorer([1], MagicMock(), meta)
        assert called == [([1], False)]

    def test_ctrl_v_dispatches_action_view_editor(self, run_module, monkeypatch):
        proc = _fzf_pick("ctrl-v\n1\t0001  01/01/2025 00:00Z  ch_session_1.json\n")
        called = []
        monkeypatch.setattr(
            run_module, "action_view_editor", lambda cid, conn, meta: called.append(cid)
        )
        with patch("shutil.which", lambda b: "/usr/local/bin/fzf"), patch(
            "subprocess.run", return_value=proc
        ):
            meta = {1: {"file_path": "/tmp/ch_session_1.json"}}
            run_module.open_explorer([1], MagicMock(), meta)
        assert called == [1]

    def test_ctrl_y_dispatches_action_copy_filenames(self, run_module, monkeypatch):
        proc = _fzf_pick("ctrl-y\n1\t0001  01/01/2025 00:00Z  ch_session_1.json\n")
        called = []
        monkeypatch.setattr(
            run_module, "action_copy_filenames", lambda cids, meta: called.append(cids)
        )
        with patch("shutil.which", lambda b: "/usr/local/bin/fzf"), patch(
            "subprocess.run", return_value=proc
        ):
            meta = {1: {"file_path": "/tmp/ch_session_1.json"}}
            run_module.open_explorer([1], MagicMock(), meta)
        assert called == [[1]]

    def test_ctrl_s_dispatches_action_save_downloads(self, run_module, monkeypatch):
        proc = _fzf_pick("ctrl-s\n1\t0001  01/01/2025 00:00Z  ch_session_1.json\n")
        called = []
        monkeypatch.setattr(
            run_module,
            "action_save_downloads",
            lambda cids, conn, meta: called.append(cids),
        )
        with patch("shutil.which", lambda b: "/usr/local/bin/fzf"), patch(
            "subprocess.run", return_value=proc
        ):
            meta = {1: {"file_path": "/tmp/ch_session_1.json"}}
            run_module.open_explorer([1], MagicMock(), meta)
        assert called == [[1]]


class TestMainLoop:
    """main() must terminate cleanly on /exit and EOFError."""

    def test_exit_command_terminates(self, run_module, monkeypatch):
        inputs = iter(["/exit"])

        def fake_input(prompt):
            return next(inputs)

        monkeypatch.setattr(run_module, "require_ch_dirs", lambda: None)
        monkeypatch.setattr(run_module, "prompt_update_cache", lambda: False)
        monkeypatch.setattr(run_module, "print_banner", lambda: None)
        # stub the heavy imports inside main()
        fake_conn = MagicMock()
        fake_session = MagicMock()
        monkeypatch.setattr(run_module, "_drain_stdin", lambda: None)

        # We need to stub the heavy import section of main()
        # Since main() does imports inside, we mock the import system
        # by patching the builtins
        with patch("builtins.input", fake_input):
            # main() will try to import heavy modules - we need to prevent that
            # by patching the import machinery
            result = self._run_main_with_mocks(
                run_module, monkeypatch, fake_conn, fake_session
            )
        assert result == 0

    def _run_main_with_mocks(self, run_module, monkeypatch, fake_conn, fake_session):
        """Helper: run main() with mocked heavy imports."""
        # The heavy imports inside main() are:
        # from retrieve.spinner import ...
        # from build import ...
        # from config import ...
        # from pricing import ...
        # from retrieve.cache import ...
        # from retrieve.search import ...
        # from retrieve.state import ...
        # from retrieve.cmds.ls import ...
        # We mock them by patching __import__ to return mocks for these modules
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name.startswith("retrieve") or name in (
                "build",
                "config",
                "pricing",
                "explorer_preview",
            ):
                mod = MagicMock()
                # Set up specific attributes that main() uses
                if name == "retrieve.spinner":
                    mod.start_startup_spinner = lambda *a: None
                    mod.stop_startup_spinner = lambda *a: None
                    mod.Spinner = MagicMock()
                if name == "build":
                    mod.get_connection = lambda: fake_conn
                    mod.backfill_message_epochs = lambda c: None
                    mod.backfill_archived = lambda c: None
                if name == "config":
                    mod.TOP_K = 5
                    mod.NUM_EXPANSIONS = 3
                    mod.DB_PATH = "/fake/db"
                    mod.PREVIEW_BATCH = 500
                if name == "pricing":
                    mod.warm = lambda: None
                if name == "retrieve.cache":
                    mod.ensure_fts = lambda c: None
                    mod.load_vectors = lambda c: ([], MagicMock(), {})
                if name == "retrieve.search":
                    mod.search = lambda s, q: ([], {})
                    mod.warm_connections = lambda: None
                if name == "retrieve.state":
                    mod.Session = MagicMock(return_value=fake_session)
                if name == "retrieve.cmds.ls":
                    mod.list_chats_by_recency = lambda *a: []
                return mod
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        return run_module.main()


class TestActionSaveDownloads:
    """action_save_downloads: save single chat or merged dump to ~/Downloads."""

    def test_save_single_active_chat(self, run_module, monkeypatch, tmp_path):
        # set up fake meta and conn
        chat_path = tmp_path / "ch_session_1740001234.json"
        chat_path.write_text('{"messages":[]}')

        meta = {
            1: {
                "file_path": str(chat_path),
                "archived": False,
            }
        }
        conn = MagicMock()

        downloads_dir = tmp_path / "Downloads"
        monkeypatch.setattr(
            run_module.os.path,
            "expanduser",
            lambda p: str(downloads_dir) if p == "~/Downloads" else p,
        )
        # mock the import inside the function
        from retrieve.cmds.dump import unique_path as real_unique_path

        monkeypatch.setattr(
            "os.path.expanduser",
            lambda p: str(downloads_dir) if p == "~/Downloads" else p,
        )

        result = run_module.action_save_downloads([1], conn, meta)
        saved_files = list(downloads_dir.glob("*.json"))
        assert len(saved_files) == 1
