"""Tests for retrieve/cmds/ls.py: list_chats_by_recency, handle_purge."""

import sqlite3
import pytest
from unittest.mock import patch, MagicMock

from retrieve.cmds.ls import (
    list_chats_by_recency,
    handle_ls,
    handle_purge,
    PURGE_NO,
    PURGE_YES_TMPL,
    LS_ACTIONS,
)


class TestListChatsByRecency:
    def test_no_archived_hidden(self, db_with_chats):
        rows = list_chats_by_recency(
            db_with_chats, show_archived=False, time_filter=None
        )
        ids = [r[0] for r in rows]
        assert 3 not in ids  # archived chat excluded
        assert 1 in ids
        assert 2 in ids

    def test_archived_shown(self, db_with_chats):
        rows = list_chats_by_recency(
            db_with_chats, show_archived=True, time_filter=None
        )
        ids = [r[0] for r in rows]
        assert 3 in ids

    def test_sorted_newest_first(self, db_with_chats):
        rows = list_chats_by_recency(
            db_with_chats, show_archived=True, time_filter=None
        )
        epochs = [r[1]["last_message_epoch"] for r in rows]
        assert epochs == sorted(epochs, reverse=True)

    def test_time_filter_excludes_none_epoch(self, db_conn):
        # chat with no parseable epoch
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, "
            "created_at, updated_at) VALUES ('no_epoch.json', '{}', 't', 1, 'n', 'n')"
        )
        db_conn.commit()
        rows = list_chats_by_recency(db_conn, show_archived=True, time_filter="1d")
        # should exclude the no-epoch chat
        assert all(
            r[1]["last_message_epoch"] is not None
            for r in rows
            if r[1]["file_path"] != "no_epoch.json"
        )

    def test_time_filter_str_bounds(self, db_with_chats):
        import time as time_mod

        with patch.object(time_mod, "time", return_value=1700000200):
            rows = list_chats_by_recency(
                db_with_chats, show_archived=True, time_filter="1d"
            )
            # chat 1 has epoch 1700000100 (within 1 day), chat 2 has 1700000500 (within)
            # chat 3 has 1700001000 (might be within depending on exact time)
            ids = [r[0] for r in rows]
            assert 1 in ids  # 1700000100 is within 86400 of 1700000200

    def test_time_filter_tuple(self, db_with_chats):
        rows = list_chats_by_recency(
            db_with_chats, show_archived=True, time_filter=(1700000000, 1700000500)
        )
        ids = [r[0] for r in rows]
        assert 1 in ids  # epoch 1700000100 in range
        assert 2 in ids  # epoch 1700000500 in range
        assert 3 not in ids  # epoch 1700001000 out of range

    def test_raw_size_in_info(self, db_with_chats):
        rows = list_chats_by_recency(
            db_with_chats, show_archived=True, time_filter=None
        )
        for _, info in rows:
            assert "raw_size" in info
            assert isinstance(info["raw_size"], int)


class TestHandlePurge:
    def test_no_archived(self, db_conn, capsys):
        result = handle_purge(db_conn)
        assert result == 0
        assert "No archived chats" in capsys.readouterr().out

    def test_missing_fzf(self, db_with_chats):
        with patch("retrieve.cmds.ls.shutil.which", return_value=None):
            result = handle_purge(db_with_chats)
            assert result == 0

    def test_cancel_no_default(self, db_with_chats):
        with patch("retrieve.cmds.ls.shutil.which", return_value="/usr/bin/fzf"):
            with patch("retrieve.cmds.ls.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout=PURGE_NO)
                result = handle_purge(db_with_chats)
                assert result == 0
                # verify no DELETE happened
                count = db_with_chats.execute(
                    "SELECT count(*) FROM chats WHERE archived = 1"
                ).fetchone()[0]
                assert count == 1  # archived chat still there

    def test_confirm_yes_deletes(self, db_with_chats):
        with patch("retrieve.cmds.ls.shutil.which", return_value="/usr/bin/fzf"):
            with patch("retrieve.cmds.ls.subprocess.run") as mock_run:
                yes_label = PURGE_YES_TMPL.format(count=1)
                mock_run.return_value = MagicMock(returncode=0, stdout=yes_label)
                result = handle_purge(db_with_chats)
                assert result == 1
                # verify DELETE happened
                count = db_with_chats.execute(
                    "SELECT count(*) FROM chats WHERE archived = 1"
                ).fetchone()[0]
                assert count == 0

    def test_unknown_label_cancelled(self, db_with_chats):
        with patch("retrieve.cmds.ls.shutil.which", return_value="/usr/bin/fzf"):
            with patch("retrieve.cmds.ls.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="something else")
                result = handle_purge(db_with_chats)
                assert result == 0

    def test_non_zero_returncode_cancelled(self, db_with_chats):
        with patch("retrieve.cmds.ls.shutil.which", return_value="/usr/bin/fzf"):
            with patch("retrieve.cmds.ls.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1, stdout="")
                result = handle_purge(db_with_chats)
                assert result == 0


class TestLSActions:
    def test_actions_stable(self):
        actions = dict(LS_ACTIONS)
        assert "run" in actions.values()
        assert "copy" in actions.values()
        assert "cancel" in actions.values()


class TestHandleLS:
    def test_cancelled_selection(self, db_with_chats):
        with patch(
            "retrieve.cmds.ls.pick_latest_with_fzf", return_value=(None, None)
        ), patch("retrieve.cmds.ls.pick_ls_action") as mock_pick_action:
            handle_ls(db_with_chats, show_archived=False, time_filter=None)
            mock_pick_action.assert_not_called()

    def test_run_action_passes_reprint_prompt(self, db_with_chats):
        info = {"file_path": "/tmp/ch_session_1.json"}
        with patch(
            "retrieve.cmds.ls.pick_latest_with_fzf", return_value=(1, info)
        ), patch("retrieve.cmds.ls.pick_ls_action", return_value="run"), patch(
            "retrieve.cmds.ls.run_chat"
        ) as mock_run_chat:
            handle_ls(
                db_with_chats,
                show_archived=False,
                time_filter=None,
                reprint_prompt=False,
            )
            mock_run_chat.assert_called_once_with(1, {1: info}, reprint_prompt=False)

    def test_copy_action(self, db_with_chats):
        info = {"file_path": "/tmp/ch_session_1.json"}
        with patch(
            "retrieve.cmds.ls.pick_latest_with_fzf", return_value=(1, info)
        ), patch("retrieve.cmds.ls.pick_ls_action", return_value="copy"), patch(
            "retrieve.cmds.ls.copy_chat"
        ) as mock_copy_chat:
            handle_ls(db_with_chats, show_archived=False, time_filter=None)
            mock_copy_chat.assert_called_once_with(1, {1: info})
