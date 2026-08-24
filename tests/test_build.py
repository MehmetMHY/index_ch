"""Tests for build.py: estimate_tokens, is_auto_entry, format_messages,
message_epoch, load_and_clean, and DB functions."""

import json
import hashlib
import sqlite3
import pytest

from build import (
    estimate_tokens,
    is_auto_entry,
    format_messages,
    message_epoch,
    load_and_clean,
    insert_entries,
    update_entries,
    backfill_message_epochs,
    backfill_archived,
    stored_hashes,
)


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_none(self):
        assert estimate_tokens(None) == 0

    def test_simple_word(self):
        result = estimate_tokens("hello")
        assert result > 0
        assert isinstance(result, int)

    def test_long_text(self):
        text = "word " * 100
        result = estimate_tokens(text)
        assert result > 0

    def test_unicode(self):
        result = estimate_tokens("héllo wörld")
        assert result > 0

    def test_punctuation(self):
        result = estimate_tokens("hello!!! world???")
        assert result > 0


class TestIsAutoEntry:
    def test_empty(self):
        assert is_auto_entry("") is False

    def test_none(self):
        assert is_auto_entry(None) is False

    def test_file_prefix(self):
        assert is_auto_entry("File: some/path.py") is True

    def test_file_prefix_no_space(self):
        assert is_auto_entry("File:some/path.py") is False

    def test_code_dump(self):
        assert is_auto_entry("some text === Code Dump === more") is True

    def test_file_marker(self):
        assert is_auto_entry("=== FILE: test.py ===") is True

    def test_command_output(self):
        assert (
            is_auto_entry(
                "The user executed the following command and here is the output:"
            )
            is True
        )

    def test_code_dump_starts_case_insensitive(self):
        assert is_auto_entry("[CODE-DUMP STARTS]") is True
        assert is_auto_entry("[code-dump starts]") is True

    def test_normal_text(self):
        assert is_auto_entry("What is the best way to do X?") is False


class TestFormatMessages:
    def test_empty(self):
        assert format_messages([]) == ""

    def test_basic(self):
        msgs = [{"user": "hi", "bot": "hello"}]
        result = format_messages(msgs)
        assert "USER: hi" in result
        assert "BOT: hello" in result

    def test_skip_noise_true(self):
        msgs = [
            {"user": "File: test.py", "bot": "contents here"},
            {"user": "real question", "bot": "real answer"},
        ]
        result = format_messages(msgs, skip_noise=True)
        assert "real question" in result
        assert "File: test.py" not in result

    def test_skip_noise_false(self):
        msgs = [
            {"user": "File: test.py", "bot": "contents here"},
            {"user": "real question", "bot": "real answer"},
        ]
        result = format_messages(msgs, skip_noise=False)
        assert "File: test.py" in result
        assert "real question" in result

    def test_noise_in_bot_drops_turn(self):
        msgs = [{"user": "real question", "bot": "=== Code Dump === stuff"}]
        result = format_messages(msgs, skip_noise=True)
        assert result == ""

    def test_none_user_bot(self):
        msgs = [{"user": None, "bot": None}]
        result = format_messages(msgs)
        assert "USER: " in result
        assert "BOT: " in result

    def test_multiple_messages(self):
        msgs = [
            {"user": "q1", "bot": "a1"},
            {"user": "q2", "bot": "a2"},
        ]
        result = format_messages(msgs)
        assert result.count("USER:") == 2
        assert result.count("BOT:") == 2


class TestMessageEpoch:
    def test_empty_messages(self):
        assert message_epoch({}) is None

    def test_no_messages_key(self):
        assert message_epoch({"other": "data"}) is None

    def test_none_messages(self):
        assert message_epoch({"messages": None}) is None

    def test_valid_epoch(self):
        raw = {"messages": [{"time": 1700000000}]}
        assert message_epoch(raw) == 1700000000

    def test_last_message_epoch(self):
        raw = {"messages": [{"time": 1700000000}, {"time": 1700000100}]}
        assert message_epoch(raw) == 1700000100

    def test_trailing_no_time_falls_back(self):
        raw = {
            "messages": [{"time": 1700000500}, {"time": 1700000600}, {"no_time": True}]
        }
        assert message_epoch(raw) == 1700000600

    def test_string_time_skipped(self):
        raw = {"messages": [{"time": "not-a-number"}]}
        assert message_epoch(raw) is None

    def test_zero_time_skipped(self):
        raw = {"messages": [{"time": 0}]}
        assert message_epoch(raw) is None

    def test_negative_time_skipped(self):
        raw = {"messages": [{"time": -100}]}
        assert message_epoch(raw) is None

    def test_float_time_truncated(self):
        raw = {"messages": [{"time": 1700000000.9}]}
        assert message_epoch(raw) == 1700000000


class TestLoadAndClean:
    def test_valid_file(self, tmp_path):
        raw = {"messages": [{"user": "hi", "bot": "hello"}]}
        f = tmp_path / "test.json"
        f.write_bytes(json.dumps(raw).encode())
        result = load_and_clean(str(f))
        assert result is not None
        assert result["raw"] == raw
        assert "USER: hi" in result["cleaned"]
        assert result["content_hash"] == hashlib.sha256(f.read_bytes()).hexdigest()

    def test_invalid_json(self, tmp_path, capsys):
        f = tmp_path / "bad.json"
        f.write_text("not json")
        result = load_and_clean(str(f))
        assert result is None
        assert "skipping" in capsys.readouterr().out

    def test_missing_file(self, tmp_path, capsys):
        result = load_and_clean(str(tmp_path / "nonexistent.json"))
        assert result is None

    def test_missing_messages_key(self, tmp_path, capsys):
        f = tmp_path / "no_messages.json"
        f.write_text(json.dumps({"other": "data"}))
        result = load_and_clean(str(f))
        assert result is None

    def test_hash_matches_bytes(self, tmp_path):
        raw = {"messages": [{"user": "x", "bot": "y"}]}
        data = json.dumps(raw).encode()
        f = tmp_path / "test.json"
        f.write_bytes(data)
        result = load_and_clean(str(f))
        assert result["content_hash"] == hashlib.sha256(data).hexdigest()


class TestInsertEntries:
    def test_insert(self, db_conn):
        raw = {"messages": [{"user": "hi", "bot": "hello", "time": 1700000000}]}
        result = {"raw": raw, "cleaned": "USER: hi\nBOT: hello\n"}
        insert_entries(db_conn, [("/tmp/test.json", result, "hash123")])
        row = db_conn.execute("SELECT file_path, raw, cleaned FROM chats").fetchone()
        assert row[0] == "/tmp/test.json"
        assert json.loads(row[1]) == raw
        assert "USER: hi" in row[2]

    def test_insert_empty(self, db_conn):
        insert_entries(db_conn, [])
        count = db_conn.execute("SELECT count(*) FROM chats").fetchone()[0]
        assert count == 0


class TestUpdateEntries:
    def test_update_clears_summary(self, db_conn):
        raw = {"messages": [{"user": "hi", "bot": "hello", "time": 1700000000}]}
        result = {"raw": raw, "cleaned": "new cleaned text"}
        # first insert
        insert_entries(db_conn, [("/tmp/test.json", result, "hash1")])
        # set summary
        db_conn.execute(
            "UPDATE chats SET summary = 'old summary' WHERE file_path = '/tmp/test.json'"
        )
        db_conn.commit()
        # update with new content
        result2 = {
            "raw": {"messages": [{"user": "new", "bot": "data", "time": 1700000100}]},
            "cleaned": "USER: new\nBOT: data\n",
        }
        update_entries(db_conn, [("/tmp/test.json", result2, "hash2")])
        row = db_conn.execute(
            "SELECT summary, cleaned FROM chats WHERE file_path = '/tmp/test.json'"
        ).fetchone()
        assert row[0] is None
        assert "USER: new" in row[1]

    def test_update_empty(self, db_conn):
        update_entries(db_conn, [])
        # should not raise


class TestBackfillMessageEpochs:
    def test_idempotent(self, db_conn):
        backfill_message_epochs(db_conn)
        backfill_message_epochs(db_conn)  # second call should be no-op
        # verify column exists
        cols = {row[1] for row in db_conn.execute("PRAGMA table_info(chats)")}
        assert "last_message_epoch" in cols

    def test_populates_from_raw(self, db_conn):
        raw = json.dumps({"messages": [{"time": 1700000500}]})
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("/tmp/test.json", raw, "text", 5, "now", "now"),
        )
        db_conn.commit()
        # remove the column and re-add via backfill
        # (can't ALTER TABLE DROP COLUMN in sqlite easily, so test on a fresh DB without the column)
        # Instead test that existing rows get populated when the column is first added
        # The fixture already has the column, so let's test a fresh conn
        conn2 = sqlite3.connect(":memory:")
        conn2.execute("""
            CREATE TABLE chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                raw TEXT NOT NULL,
                cleaned TEXT NOT NULL,
                token_estimate INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn2.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("/tmp/test.json", raw, "text", 5, "now", "now"),
        )
        conn2.commit()
        backfill_message_epochs(conn2)
        epoch = conn2.execute("SELECT last_message_epoch FROM chats").fetchone()[0]
        assert epoch == 1700000500


class TestBackfillArchived:
    def test_idempotent(self, db_conn):
        backfill_archived(db_conn)
        backfill_archived(db_conn)
        cols = {row[1] for row in db_conn.execute("PRAGMA table_info(chats)")}
        assert "archived" in cols

    def test_defaults_to_zero(self, db_conn):
        # The fixture already has the column, test a fresh conn
        conn2 = sqlite3.connect(":memory:")
        conn2.execute("""
            CREATE TABLE chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                raw TEXT NOT NULL,
                cleaned TEXT NOT NULL,
                token_estimate INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn2.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, created_at, updated_at) "
            "VALUES ('/tmp/t.json', '{}', 't', 1, 'n', 'n')"
        )
        conn2.commit()
        backfill_archived(conn2)
        archived = conn2.execute("SELECT archived FROM chats").fetchone()[0]
        assert archived == 0


class TestStoredHashes:
    def test_returns_dict(self, db_conn):
        insert_entries(
            db_conn,
            [
                ("/tmp/a.json", {"raw": {"messages": []}, "cleaned": ""}, "hash_a"),
            ],
        )
        hashes = stored_hashes(db_conn)
        assert "/tmp/a.json" in hashes
        assert hashes["/tmp/a.json"] == "hash_a"
