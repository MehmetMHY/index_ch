"""Tests for explorer_preview.py: format_ts, filename_epoch,
render_explorer_preview_with_conn, compute_and_save_preview."""

import json
import os
import sqlite3

from explorer_preview import (
    format_ts,
    filename_epoch,
    render_explorer_preview_with_conn,
    render_explorer_preview,
    compute_and_save_preview,
)


class TestFormatTs:
    def test_valid_epoch(self):
        assert format_ts(1700000000) == "11/14/2023 22:13Z"

    def test_none(self):
        assert format_ts(None) == ""

    def test_zero(self):
        assert format_ts(0) == ""


class TestFilenameEpoch:
    def test_typical(self):
        assert filename_epoch("ch_session_1700000000.json") == 1700000000

    def test_no_digits(self):
        assert filename_epoch("no_numbers.json") is None

    def test_empty(self):
        assert filename_epoch("") is None

    def test_with_path(self):
        assert filename_epoch("/some/dir/ch_session_1700000000.json") == 1700000000


class TestRenderExplorerPreview:
    def test_invalid_id(self, db_conn):
        assert "Invalid" in render_explorer_preview_with_conn(db_conn, "not-a-number")

    def test_chat_not_found(self, db_conn):
        assert render_explorer_preview_with_conn(db_conn, 99999) == "Chat not found."

    def test_valid_chat(self, db_with_chats):
        result = render_explorer_preview_with_conn(db_with_chats, 1)
        assert "ch_session_1700000000.json" in result
        assert "enter" in result  # legend
        assert "ctrl-v" in result
        assert "Turn 1" in result
        assert "User:" in result
        assert "Assistant:" in result

    def test_archived_chat(self, db_with_chats):
        result = render_explorer_preview_with_conn(db_with_chats, 3)
        assert "archived" in result

    def test_active_chat(self, db_with_chats):
        result = render_explorer_preview_with_conn(db_with_chats, 1)
        assert "active" in result

    def test_turns_singular(self, db_conn):
        raw = json.dumps({"messages": [{"user": "q", "bot": "a"}], "model": "m"})
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, "
            "created_at, updated_at) VALUES ('/t.json', ?, 't', 1, 'n', 'n')",
            (raw,),
        )
        db_conn.commit()
        result = render_explorer_preview_with_conn(db_conn, 1)
        assert "1 turn" in result
        assert "1 turns" not in result

    def test_turns_plural(self, db_conn):
        raw = json.dumps(
            {
                "messages": [{"user": "q1", "bot": "a1"}, {"user": "q2", "bot": "a2"}],
                "model": "m",
            }
        )
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, "
            "created_at, updated_at) VALUES ('/t.json', ?, 't', 1, 'n', 'n')",
            (raw,),
        )
        db_conn.commit()
        result = render_explorer_preview_with_conn(db_conn, 1)
        assert "2 turns" in result

    def test_model_displayed(self, db_conn):
        raw = json.dumps({"messages": [{"user": "q", "bot": "a"}], "model": "gpt-4o"})
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, "
            "created_at, updated_at) VALUES ('/t.json', ?, 't', 1, 'n', 'n')",
            (raw,),
        )
        db_conn.commit()
        result = render_explorer_preview_with_conn(db_conn, 1)
        assert "gpt-4o" in result

    def test_short_summary_as_title(self, db_conn):
        raw = json.dumps({"messages": [{"user": "q", "bot": "a"}], "model": "m"})
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, "
            "created_at, updated_at, short_summary) "
            "VALUES ('/t.json', ?, 't', 1, 'n', 'n', 'My custom title')",
            (raw,),
        )
        db_conn.commit()
        result = render_explorer_preview_with_conn(db_conn, 1)
        assert "My custom title" in result

    def test_filename_fallback_title(self, db_conn):
        raw = json.dumps({"messages": [{"user": "q", "bot": "a"}], "model": "m"})
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, "
            "created_at, updated_at) "
            "VALUES ('/tmp/ch_session_1700000000.json', ?, 't', 1, 'n', 'n')",
            (raw,),
        )
        db_conn.commit()
        result = render_explorer_preview_with_conn(db_conn, 1)
        assert "ch_session_1700000000.json" in result

    def test_empty_raw(self, db_conn):
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, "
            "created_at, updated_at) VALUES ('/t.json', '', 't', 1, 'n', 'n')"
        )
        db_conn.commit()
        result = render_explorer_preview_with_conn(db_conn, 1)
        assert "unknown" in result  # model falls back

    def test_code_block_detection(self, db_conn):
        raw = json.dumps(
            {
                "messages": [
                    {"user": "show code", "bot": "```python\nprint('hi')\n```"}
                ],
                "model": "m",
            }
        )
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, "
            "created_at, updated_at) VALUES ('/t.json', ?, 't', 1, 'n', 'n')",
            (raw,),
        )
        db_conn.commit()
        result = render_explorer_preview_with_conn(db_conn, 1)
        assert "```python" in result

    def test_summary_displayed(self, db_conn):
        raw = json.dumps({"messages": [{"user": "q", "bot": "a"}], "model": "m"})
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, "
            "created_at, updated_at, summary) "
            "VALUES ('/t.json', ?, 't', 1, 'n', 'n', 'This is the full summary.')",
            (raw,),
        )
        db_conn.commit()
        result = render_explorer_preview_with_conn(db_conn, 1)
        assert "This is the full summary." in result

    def test_legend_present(self, db_conn):
        raw = json.dumps({"messages": [{"user": "q", "bot": "a"}], "model": "m"})
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, "
            "created_at, updated_at) VALUES ('/t.json', ?, 't', 1, 'n', 'n')",
            (raw,),
        )
        db_conn.commit()
        result = render_explorer_preview_with_conn(db_conn, 1)
        assert "ctrl-y" in result
        assert "ctrl-s" in result
        assert "tab" in result
        assert "alt-j/k" in result
        assert "esc" in result


class TestComputeAndSavePreview:
    def test_writes_file(self, db_with_chats, tmp_path):
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
        out_path = tmp_path / "preview_1.txt"
        assert out_path.exists()
        content = out_path.read_text()
        assert "ch_session" in content
        assert "enter" in content  # legend

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

        out_path = tmp_path / "preview_1.txt"
        out_path.write_text("existing content")
        compute_and_save_preview((1, str(tmp_path), db_path))
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
        assert not (tmp_path / "preview_1.txt.tmp").exists()


class TestRenderStandalone:
    def test_render_explorer_preview(self, db_with_chats, tmp_path):
        """render_explorer_preview (standalone) opens its own connection."""
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

        result = render_explorer_preview(1, db_path)
        assert "ch_session" in result
        assert "enter" in result
