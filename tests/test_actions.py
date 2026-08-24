"""Tests for retrieve/actions.py: fetch_raw, build_content, view_chat,
copy_to_clipboard, copy_chat, run_chat."""

import json
import os
import pytest
from unittest.mock import patch, MagicMock, mock_open

from retrieve.actions import (
    fetch_raw,
    build_content,
    view_chat,
    copy_to_clipboard,
    copy_chat,
    run_chat,
)


class TestFetchRaw:
    def test_found(self, db_with_chats):
        raw = fetch_raw(db_with_chats, 1)
        assert raw is not None
        assert "messages" in raw

    def test_not_found(self, db_with_chats):
        assert fetch_raw(db_with_chats, 99999) is None


class TestBuildContent:
    def test_basic(self, db_with_chats, sample_meta):
        content = build_content(db_with_chats, 1, sample_meta)
        assert "ch_session_1700000000.json" in content
        assert "A greeting chat" in content
        assert "=" * 70 in content  # divider
        assert "USER:" in content  # raw transcript

    def test_missing_summary(self, db_with_chats):
        meta = {1: {"file_path": "/tmp/ch_session_1700000000.json", "summary": None}}
        content = build_content(db_with_chats, 1, meta)
        assert "(no summary)" in content

    def test_raw_unavailable(self, db_conn):
        # no raw in DB for this id
        meta = {1: {"file_path": "test.json", "summary": "sum"}}
        content = build_content(db_conn, 1, meta)
        assert "(raw content unavailable)" in content


class TestViewChat:
    def test_temp_file_written_and_removed(
        self, db_with_chats, sample_meta, tmp_path, monkeypatch
    ):
        monkeypatch.setattr("retrieve.actions.TMP_DIR", str(tmp_path))
        with patch("retrieve.actions.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            view_chat(db_with_chats, 1, sample_meta)
            # after view_chat returns, temp file should be gone
            tmp_files = list(tmp_path.glob("chat_*.md"))
            assert len(tmp_files) == 0  # cleaned up

    def test_cleanup_on_editor_exception(
        self, db_with_chats, sample_meta, tmp_path, monkeypatch
    ):
        monkeypatch.setattr("retrieve.actions.TMP_DIR", str(tmp_path))
        with patch(
            "retrieve.actions.subprocess.run", side_effect=Exception("editor crashed")
        ):
            with pytest.raises(Exception):
                view_chat(db_with_chats, 1, sample_meta)
            # temp file should still be cleaned up
            tmp_files = list(tmp_path.glob("chat_*.md"))
            assert len(tmp_files) == 0

    def test_editor_with_args(self, db_with_chats, sample_meta, tmp_path, monkeypatch):
        monkeypatch.setattr("retrieve.actions.TMP_DIR", str(tmp_path))
        monkeypatch.setenv("EDITOR", "code -w")
        with patch("retrieve.actions.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            view_chat(db_with_chats, 1, sample_meta)
            args = mock_run.call_args[0][0]
            assert args[0] == "code"
            assert args[1] == "-w"


class TestCopyToClipboard:
    def test_darwin(self):
        with patch("retrieve.actions.platform.system", return_value="Darwin"):
            with patch("retrieve.actions.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                assert copy_to_clipboard("text") is True
                assert mock_run.call_args[0][0] == ["pbcopy"]

    def test_linux_wl_copy(self):
        with patch("retrieve.actions.platform.system", return_value="Linux"):
            with patch(
                "retrieve.actions.shutil.which", return_value="/usr/bin/wl-copy"
            ):
                with patch("retrieve.actions.subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(returncode=0)
                    assert copy_to_clipboard("text") is True
                    assert mock_run.call_args[0][0] == ["wl-copy"]

    def test_linux_xclip(self):
        with patch("retrieve.actions.platform.system", return_value="Linux"):
            with patch("retrieve.actions.shutil.which") as mock_which:
                mock_which.side_effect = lambda x: {
                    "wl-copy": None,
                    "xclip": "/usr/bin/xclip",
                }.get(x)
                with patch("retrieve.actions.subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(returncode=0)
                    assert copy_to_clipboard("text") is True
                    assert mock_run.call_args[0][0] == [
                        "xclip",
                        "-selection",
                        "clipboard",
                    ]

    def test_linux_no_tool(self):
        with patch("retrieve.actions.platform.system", return_value="Linux"):
            with patch("retrieve.actions.shutil.which", return_value=None):
                assert copy_to_clipboard("text") is False

    def test_subprocess_failure(self):
        with patch("retrieve.actions.platform.system", return_value="Darwin"):
            with patch("retrieve.actions.subprocess.run") as mock_run:
                import subprocess as sp

                mock_run.side_effect = sp.CalledProcessError(1, ["pbcopy"])
                assert copy_to_clipboard("text") is False

    def test_tool_not_found(self):
        with patch("retrieve.actions.platform.system", return_value="Darwin"):
            with patch("retrieve.actions.subprocess.run") as mock_run:
                mock_run.side_effect = FileNotFoundError()
                assert copy_to_clipboard("text") is False

    def test_unknown_platform(self):
        with patch("retrieve.actions.platform.system", return_value="UnknownOS"):
            assert copy_to_clipboard("text") is False


class TestCopyChat:
    def test_success(self, sample_meta, capsys):
        with patch("retrieve.actions.copy_to_clipboard", return_value=True):
            copy_chat(1, sample_meta)
            out = capsys.readouterr().out
            assert "Copied" in out
            assert "ch_session_1700000000.json" in out

    def test_archived_note(self, capsys):
        meta = {3: {"file_path": "/tmp/ch_session_1700001000.json", "archived": True}}
        with patch("retrieve.actions.copy_to_clipboard", return_value=True):
            copy_chat(3, meta)
            out = capsys.readouterr().out
            assert "archived" in out

    def test_failure_silent(self, sample_meta, capsys):
        with patch("retrieve.actions.copy_to_clipboard", return_value=False):
            copy_chat(1, sample_meta)
            out = capsys.readouterr().out
            assert "Copied" not in out


class TestRunChat:
    def test_missing_ch(self, sample_meta, capsys):
        with patch("retrieve.actions.shutil.which", return_value=None):
            run_chat(1, sample_meta)
            out = capsys.readouterr().out
            assert "not found" in out
            assert "github.com" in out

    def test_archived_warning(self, capsys):
        meta = {3: {"file_path": "/tmp/ch_session_1700001000.json", "archived": True}}
        with patch("retrieve.actions.shutil.which", return_value="/usr/bin/ch"):
            with patch("retrieve.actions.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                run_chat(3, meta)
                out = capsys.readouterr().out
                assert "archived" in out
                assert "may fail" in out

    def test_non_zero_exit(self, sample_meta, capsys):
        with patch("retrieve.actions.shutil.which", return_value="/usr/bin/ch"):
            with patch("retrieve.actions.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1)
                run_chat(1, sample_meta)
                out = capsys.readouterr().out
                assert "status 1" in out

    def test_always_prints_prompt(self, sample_meta, capsys):
        with patch("retrieve.actions.shutil.which", return_value="/usr/bin/ch"):
            with patch("retrieve.actions.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                run_chat(1, sample_meta)
                out = capsys.readouterr().out
                assert "Type a query or /help" in out
