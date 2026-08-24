"""Tests for preview.py: format_timestamp, filename_epoch, chat_epoch,
format_messages_limited, preview_chat_with_conn, compute_and_save_preview."""

import json
import os
import sqlite3
import pytest

from preview import (
    format_timestamp,
    filename_epoch,
    chat_epoch,
    format_messages_limited,
    preview_chat_with_conn,
    compute_and_save_preview,
)


class TestFormatTimestamp:
    def test_none(self):
        assert format_timestamp(None) == "no date"

    def test_zero(self):
        assert format_timestamp(0) == "no date"

    def test_valid(self):
        ts = format_timestamp(1700000000)
        assert "UTC" in ts
        assert "2023" in ts

    def test_overflow(self):
        assert format_timestamp(10**18) == "no date"

    def test_negative(self):
        assert format_timestamp(-99999999999) == "no date"


class TestFilenameEpoch:
    def test_typical(self):
        assert filename_epoch("ch_session_1700000000.json") == 1700000000

    def test_no_digits(self):
        assert filename_epoch("no_numbers.json") is None

    def test_multiple_runs(self):
        assert filename_epoch("ch_session_1700000000_v2.json") == 1700000000

    def test_with_path(self):
        assert filename_epoch("/some/dir/ch_session_1700000000.json") == 1700000000


class TestChatEpoch:
    def test_last_message_present(self):
        assert chat_epoch("f.json", 1700000500) == 1700000500

    def test_falls_back(self):
        assert chat_epoch("ch_session_1700000000.json", None) == 1700000000

    def test_zero_falls_back(self):
        assert chat_epoch("ch_session_1700000000.json", 0) == 1700000000

    def test_both_none(self):
        assert chat_epoch("no_epoch.json", None) is None


class TestFormatMessagesLimited:
    def test_empty(self):
        assert format_messages_limited([]) == ""

    def test_under_limit(self):
        msgs = [{"user": "hi", "bot": "hello"}]
        result = format_messages_limited(msgs, limit=1000)
        assert "USER: hi" in result

    def test_stops_at_limit(self):
        msgs = [{"user": "a" * 100, "bot": "b" * 100} for _ in range(50)]
        result = format_messages_limited(msgs, limit=500)
        assert len(result) >= 500  # the message that crosses is fully appended
        assert len(result) < 100 * 4 * 50  # but much less than all 50 messages

    def test_skip_noise_true(self):
        msgs = [
            {"user": "File: test.py", "bot": "contents"},
            {"user": "real", "bot": "answer"},
        ]
        result = format_messages_limited(msgs, skip_noise=True, limit=1000)
        assert "File: test.py" not in result
        assert "real" in result

    def test_skip_noise_false_default(self):
        msgs = [
            {"user": "File: test.py", "bot": "contents"},
            {"user": "real", "bot": "answer"},
        ]
        result = format_messages_limited(msgs, limit=1000)
        assert "File: test.py" in result

    def test_none_user_bot(self):
        msgs = [{"user": None, "bot": None}]
        result = format_messages_limited(msgs, limit=1000)
        assert "USER: " in result

    def test_missing_keys(self):
        msgs = [{}]
        result = format_messages_limited(msgs, limit=1000)
        assert isinstance(result, str)


class TestPreviewChatWithConn:
    def test_invalid_id(self, db_conn):
        assert preview_chat_with_conn(db_conn, "not-a-number") == "Invalid chat id."

    def test_chat_not_found(self, db_conn):
        assert preview_chat_with_conn(db_conn, 99999) == "Chat not found."

    def test_valid_chat(self, db_with_chats):
        result = preview_chat_with_conn(db_with_chats, 1)
        assert "ch_session_1700000000.json" in result
        assert "active" in result
        assert "TL;DR:" in result
        assert "Full Summary:" in result

    def test_archived_chat(self, db_with_chats):
        result = preview_chat_with_conn(db_with_chats, 3)
        assert "archived" in result

    def test_turns_singular(self, db_conn):
        raw = json.dumps({"messages": [{"user": "q", "bot": "a"}]})
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, "
            "created_at, updated_at) VALUES ('/t.json', ?, 't', 1, 'n', 'n')",
            (raw,),
        )
        db_conn.commit()
        result = preview_chat_with_conn(db_conn, 1)
        assert "1 turn" in result
        assert "1 turns" not in result

    def test_turns_plural(self, db_conn):
        raw = json.dumps(
            {"messages": [{"user": "q1", "bot": "a1"}, {"user": "q2", "bot": "a2"}]}
        )
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, "
            "created_at, updated_at) VALUES ('/t.json', ?, 't', 1, 'n', 'n')",
            (raw,),
        )
        db_conn.commit()
        result = preview_chat_with_conn(db_conn, 1)
        assert "2 turns" in result

    def test_empty_raw(self, db_conn):
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, "
            "created_at, updated_at) VALUES ('/t.json', '', 't', 1, 'n', 'n')"
        )
        db_conn.commit()
        result = preview_chat_with_conn(db_conn, 1)
        assert "(raw content unavailable)" in result

    def test_invalid_json_raw(self, db_conn):
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, "
            "created_at, updated_at) VALUES ('/t.json', 'not json', 't', 1, 'n', 'n')"
        )
        db_conn.commit()
        result = preview_chat_with_conn(db_conn, 1)
        assert "(raw content unavailable)" in result

    def test_missing_summary(self, db_conn):
        raw = json.dumps({"messages": [{"user": "q", "bot": "a"}]})
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, "
            "created_at, updated_at) VALUES ('/t.json', ?, 't', 1, 'n', 'n')",
            (raw,),
        )
        db_conn.commit()
        result = preview_chat_with_conn(db_conn, 1)
        assert "(no summary yet)" in result
        assert "(no short summary yet)" in result


class TestComputeAndSavePreview:
    def test_writes_file(self, db_with_chats, tmp_path):
        db_path = str(tmp_path / "test.db")
        # dump db_with_chats to a file
        db_with_chats.commit()
        # use in-memory conn's data by saving to file db
        file_conn = sqlite3.connect(db_path)
        file_conn.execute("""
            CREATE TABLE chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT, raw TEXT, cleaned TEXT, token_estimate INTEGER,
                last_message_epoch INTEGER, content_hash TEXT,
                created_at TEXT, updated_at TEXT,
                archived INTEGER DEFAULT 0, summary TEXT,
                short_summary TEXT, embedding TEXT, error TEXT
            )
        """)
        for row in db_with_chats.execute("SELECT * FROM chats").fetchall():
            placeholders = ",".join("?" * len(row))
            file_conn.execute(f"INSERT INTO chats VALUES ({placeholders})", row)
        file_conn.commit()
        file_conn.close()

        compute_and_save_preview((1, str(tmp_path), db_path))
        out_path = tmp_path / "ls_preview_1.txt"
        assert out_path.exists()
        content = out_path.read_text()
        assert "ch_session" in content

    def test_idempotent_skip(self, db_with_chats, tmp_path):
        db_path = str(tmp_path / "test.db")
        file_conn = sqlite3.connect(db_path)
        file_conn.execute("""
            CREATE TABLE chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT, raw TEXT, cleaned TEXT, token_estimate INTEGER,
                last_message_epoch INTEGER, content_hash TEXT,
                created_at TEXT, updated_at TEXT,
                archived INTEGER DEFAULT 0, summary TEXT,
                short_summary TEXT, embedding TEXT, error TEXT
            )
        """)
        for row in db_with_chats.execute("SELECT * FROM chats").fetchall():
            placeholders = ",".join("?" * len(row))
            file_conn.execute(f"INSERT INTO chats VALUES ({placeholders})", row)
        file_conn.commit()
        file_conn.close()

        # create the output file first
        out_path = tmp_path / "ls_preview_1.txt"
        out_path.write_text("existing content")
        compute_and_save_preview((1, str(tmp_path), db_path))
        # should not have overwritten
        assert out_path.read_text() == "existing content"

    def test_atomic_rename(self, db_with_chats, tmp_path):
        db_path = str(tmp_path / "test.db")
        file_conn = sqlite3.connect(db_path)
        file_conn.execute("""
            CREATE TABLE chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT, raw TEXT, cleaned TEXT, token_estimate INTEGER,
                last_message_epoch INTEGER, content_hash TEXT,
                created_at TEXT, updated_at TEXT,
                archived INTEGER DEFAULT 0, summary TEXT,
                short_summary TEXT, embedding TEXT, error TEXT
            )
        """)
        for row in db_with_chats.execute("SELECT * FROM chats").fetchall():
            placeholders = ",".join("?" * len(row))
            file_conn.execute(f"INSERT INTO chats VALUES ({placeholders})", row)
        file_conn.commit()
        file_conn.close()

        compute_and_save_preview((1, str(tmp_path), db_path))
        # no .tmp file should remain
        assert not (tmp_path / "ls_preview_1.txt.tmp").exists()
