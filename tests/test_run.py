"""Tests for run.py: pick_action resolution and the main loop's exit paths.

The bug being guarded against: `Exit Session` matched no branch in the main
while-loop, so it silently re-ran pick_action forever instead of quitting.

No fzf or child scripts are spawned: every subprocess call is mocked.
"""

import importlib
import sys
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture
def run_module(monkeypatch):
    """Import run.py fresh (it lives at the repo root, not under src/, so it
    is not on pytest's pythonpath) with require_ch_dirs stubbed so it does not
    sys.exit on a test environment lacking ~/.ch."""
    sys.path.insert(0, "")
    if "run" in sys.modules:
        del sys.modules["run"]
    import run

    importlib.reload(run)
    monkeypatch.setattr(run, "require_ch_dirs", lambda: None)
    yield run
    if "run" in sys.modules:
        del sys.modules["run"]


def _fzf_pick(stdout, returncode=0):
    """Return a MagicMock mimicking a fzf subprocess.run result."""
    proc = MagicMock()
    proc.stdout = stdout
    proc.returncode = returncode
    return proc


class TestPickAction:
    def test_exit_session_label_maps_to_exit(self, run_module):
        proc = _fzf_pick("Exit Session\n")
        with patch("shutil.which", lambda b: "/usr/local/bin/fzf"), patch(
            "subprocess.run", return_value=proc
        ):
            assert run_module.pick_action() == "exit"

    def test_cancel_maps_to_exit(self, run_module):
        # Esc / Ctrl-C -> non-zero return code
        proc = _fzf_pick("", returncode=130)
        with patch("shutil.which", lambda b: "/usr/local/bin/fzf"), patch(
            "subprocess.run", return_value=proc
        ):
            assert run_module.pick_action() == "exit"

    def test_empty_stdout_maps_to_exit(self, run_module):
        proc = _fzf_pick("", returncode=0)
        with patch("shutil.which", lambda b: "/usr/local/bin/fzf"), patch(
            "subprocess.run", return_value=proc
        ):
            assert run_module.pick_action() == "exit"

    def test_browse_chats_maps_to_ls(self, run_module):
        proc = _fzf_pick("Browse Chats\n")
        with patch("shutil.which", lambda b: "/usr/local/bin/fzf"), patch(
            "subprocess.run", return_value=proc
        ):
            assert run_module.pick_action() == "ls"

    def test_smart_search_maps_to_retrieve(self, run_module):
        proc = _fzf_pick("Smart Search\n")
        with patch("shutil.which", lambda b: "/usr/local/bin/fzf"), patch(
            "subprocess.run", return_value=proc
        ):
            assert run_module.pick_action() == "retrieve"

    def test_update_cache_maps_to_update(self, run_module):
        proc = _fzf_pick("Update Cache\n")
        with patch("shutil.which", lambda b: "/usr/local/bin/fzf"), patch(
            "subprocess.run", return_value=proc
        ):
            assert run_module.pick_action() == "update"

    def test_missing_fzf_returns_update_retrieve(self, run_module):
        with patch("shutil.which", lambda b: None):
            assert run_module.pick_action() == "update_retrieve"


class TestMainLoop:
    """main() must terminate on every action; the exit branch in particular
    used to fall through and re-loop forever (the bug)."""

    def test_exit_action_terminates_loop(self, run_module):
        """Exit Session must return from main(), not call pick_action again."""
        calls = {"n": 0}

        def fake_pick():
            calls["n"] += 1
            return "exit"

        with patch.object(run_module, "pick_action", fake_pick):
            run_module.main()
        assert calls["n"] == 1, "exit must break the loop, not re-prompt"

    def test_retrieve_action_terminates_loop(self, run_module):
        calls = {"n": 0}

        def fake_pick():
            calls["n"] += 1
            return "retrieve"

        with patch.object(run_module, "pick_action", fake_pick), patch.object(
            run_module, "run_scripts"
        ) as mock_run:
            run_module.main()
        assert calls["n"] == 1
        mock_run.assert_called_once_with([run_module.RETRIEVE_MODULE])

    def test_ls_action_terminates_loop(self, run_module):
        calls = {"n": 0}

        def fake_pick():
            calls["n"] += 1
            return "ls"

        proc = MagicMock()
        proc.returncode = 0
        with patch.object(run_module, "pick_action", fake_pick), patch(
            "subprocess.run", return_value=proc
        ):
            run_module.main()
        assert calls["n"] == 1

    def test_update_action_no_return_exits(self, run_module):
        calls = {"n": 0}

        def fake_pick():
            calls["n"] += 1
            return "update"

        with patch.object(run_module, "pick_action", fake_pick), patch.object(
            run_module, "confirm_return_to_menu", return_value=False
        ), patch.object(run_module, "run_scripts"):
            run_module.main()
        assert calls["n"] == 1

    def test_update_action_return_loops_then_exits(self, run_module):
        """When the user opts to return to the menu after Update Cache, main()
        loops back. The second pick must then exit so the loop terminates."""
        seq = ["update", "exit"]
        calls = {"n": 0}

        def fake_pick():
            i = calls["n"]
            calls["n"] += 1
            return seq[i]

        with patch.object(run_module, "pick_action", fake_pick), patch.object(
            run_module, "confirm_return_to_menu", return_value=True
        ), patch.object(run_module, "run_scripts"):
            run_module.main()
        assert calls["n"] == 2  # update, then exit

    def test_update_retrieve_action_terminates_loop(self, run_module):
        calls = {"n": 0}

        def fake_pick():
            calls["n"] += 1
            return "update_retrieve"

        with patch.object(run_module, "pick_action", fake_pick), patch.object(
            run_module, "run_scripts"
        ):
            run_module.main()
        assert calls["n"] == 1
