"""Tests for process.py: fmt_duration, migrate, pending_rows, save, mark_error,
summarize (map-reduce boundary), process_row."""

import json
import pytest
from unittest.mock import MagicMock, patch

import process
from process import fmt_duration, migrate, pending_rows, save, mark_error, process_row
from config import MAX_INPUT_CHARS


class TestFmtDuration:
    def test_zero(self):
        assert fmt_duration(0) == "0s"

    def test_seconds(self):
        assert fmt_duration(59) == "59s"

    def test_minutes(self):
        assert fmt_duration(60) == "1m00s"

    def test_minutes_seconds(self):
        assert fmt_duration(125) == "2m05s"

    def test_hours(self):
        assert fmt_duration(3600) == "1h00m00s"

    def test_hours_minutes_seconds(self):
        assert fmt_duration(3661) == "1h01m01s"

    def test_float_truncated(self):
        assert fmt_duration(59.9) == "59s"

    def test_large(self):
        assert fmt_duration(7384) == "2h03m04s"


class TestMigrate:
    def test_adds_all_columns(self, db_conn):
        # fixture already has columns, so test on fresh conn without them
        import sqlite3

        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                raw TEXT NOT NULL,
                cleaned TEXT NOT NULL,
                token_estimate INTEGER NOT NULL,
                last_message_epoch INTEGER,
                content_hash TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                archived INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.commit()
        migrate(conn)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(chats)")}
        assert "summary" in cols
        assert "short_summary" in cols
        assert "embedding" in cols
        assert "error" in cols

    def test_idempotent(self, db_conn):
        migrate(db_conn)
        migrate(db_conn)  # should not raise
        cols = {row[1] for row in db_conn.execute("PRAGMA table_info(chats)")}
        assert "summary" in cols


class TestPendingRows:
    def test_returns_rows_needing_work(self, db_conn):
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, created_at, updated_at) "
            "VALUES ('/a.json', '{}', 'some text', 5, 'now', 'now')"
        )
        db_conn.commit()
        rows = pending_rows(db_conn)
        assert len(rows) == 1

    def test_excludes_empty_cleaned(self, db_conn):
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, created_at, updated_at) "
            "VALUES ('/a.json', '{}', '   ', 5, 'now', 'now')"
        )
        db_conn.commit()
        rows = pending_rows(db_conn)
        assert len(rows) == 0

    def test_excludes_error_by_default(self, db_conn):
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, created_at, updated_at, error) "
            "VALUES ('/a.json', '{}', 'text', 5, 'now', 'now', 'some error')"
        )
        db_conn.commit()
        assert len(pending_rows(db_conn)) == 0
        assert len(pending_rows(db_conn, include_errors=True)) == 1

    def test_excludes_fully_processed(self, db_conn):
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, created_at, updated_at, "
            "summary, short_summary, embedding) "
            "VALUES ('/a.json', '{}', 'text', 5, 'now', 'now', 's', 'ss', '[0.1]')"
        )
        db_conn.commit()
        assert len(pending_rows(db_conn)) == 0


class TestSave:
    def test_save_with_embedding(self, db_conn):
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, created_at, updated_at) "
            "VALUES ('/a.json', '{}', 'text', 5, 'now', 'now')"
        )
        db_conn.commit()
        result = {
            "id": 1,
            "summary": "sum",
            "short_summary": "short",
            "embedding_json": "[0.1, 0.2]",
            "summary_in": 10,
            "summary_out": 5,
            "short_in": 3,
            "short_out": 2,
            "embed_in": 8,
        }
        save(db_conn, result)
        row = db_conn.execute(
            "SELECT summary, short_summary, embedding, error FROM chats WHERE id = 1"
        ).fetchone()
        assert row[0] == "sum"
        assert row[1] == "short"
        assert row[2] == "[0.1, 0.2]"
        assert row[3] is None

    def test_save_without_embedding_preserves_existing(self, db_conn):
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, created_at, updated_at, embedding) "
            "VALUES ('/a.json', '{}', 'text', 5, 'now', 'now', '[0.9]')"
        )
        db_conn.commit()
        result = {
            "id": 1,
            "summary": "sum",
            "short_summary": "short",
            "embedding_json": None,
            "summary_in": 10,
            "summary_out": 5,
            "short_in": 3,
            "short_out": 2,
            "embed_in": 0,
        }
        save(db_conn, result)
        row = db_conn.execute(
            "SELECT summary, short_summary, embedding FROM chats WHERE id = 1"
        ).fetchone()
        assert row[0] == "sum"
        assert row[1] == "short"
        assert row[2] == "[0.9]"  # original embedding preserved


class TestMarkError:
    def test_stores_error(self, db_conn):
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, created_at, updated_at) "
            "VALUES ('/a.json', '{}', 'text', 5, 'now', 'now')"
        )
        db_conn.commit()
        mark_error(db_conn, 1, ValueError("test error"))
        row = db_conn.execute("SELECT error FROM chats WHERE id = 1").fetchone()
        assert "test error" in row[0]


class TestSummarize:
    def test_single_call_under_limit(self):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = " summary "
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 5
        with patch.object(process, "client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_resp
            result, in_tok, out_tok = process.summarize("short text")
            assert result == "summary"
            assert in_tok == 10
            assert out_tok == 5
            assert mock_client.chat.completions.create.call_count == 1

    def test_map_reduce_for_long_text(self):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = " partial "
        mock_resp.usage.prompt_tokens = 5
        mock_resp.usage.completion_tokens = 2
        with patch.object(process, "client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_resp
            long_text = "x" * (MAX_INPUT_CHARS + 1)
            result, in_tok, out_tok = process.summarize(long_text)
            # should have called: 2 chunks + 1 combine = 3
            assert mock_client.chat.completions.create.call_count == 3

    def test_boundary_exact_limit(self):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "summary"
        mock_resp.usage.prompt_tokens = 5
        mock_resp.usage.completion_tokens = 2
        with patch.object(process, "client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_resp
            text = "x" * MAX_INPUT_CHARS
            process.summarize(text)
            assert mock_client.chat.completions.create.call_count == 1


class TestProcessRow:
    def test_all_missing(self):
        row = (1, "text", None, None, False)
        with patch.object(
            process, "summarize", return_value=("sum", 10, 5)
        ) as mock_sum, patch.object(
            process, "shorten", return_value=("short", 3, 2)
        ) as mock_short, patch.object(
            process, "embed", return_value=([0.1, 0.2], MagicMock(prompt_tokens=8))
        ) as mock_embed:
            result = process_row(row)
            assert result["summary"] == "sum"
            assert result["short_summary"] == "short"
            assert result["embedding_json"] is not None
            assert mock_sum.called
            assert mock_short.called
            assert mock_embed.called

    def test_summary_present_only_short_and_embed(self):
        row = (1, "text", "existing summary", None, False)
        with patch.object(process, "summarize") as mock_sum, patch.object(
            process, "shorten", return_value=("short", 3, 2)
        ) as mock_short, patch.object(
            process, "embed", return_value=([0.1, 0.2], MagicMock(prompt_tokens=8))
        ) as mock_embed:
            result = process_row(row)
            assert result["summary"] == "existing summary"
            assert mock_sum.called is False
            assert mock_short.called
            assert mock_embed.called

    def test_has_embedding_no_embed(self):
        row = (1, "text", None, None, True)
        with patch.object(
            process, "summarize", return_value=("sum", 10, 5)
        ), patch.object(process, "shorten", return_value=("short", 3, 2)), patch.object(
            process, "embed"
        ) as mock_embed:
            result = process_row(row)
            assert result["embedding_json"] is None
            assert mock_embed.called is False
