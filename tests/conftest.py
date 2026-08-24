"""Shared test fixtures and import-time environment setup.

config.py raises SystemExit if ~/.ch/ and ~/.ch/tmp/ don't exist, and
search.py / process.py construct OpenAI clients at import time. This conftest
runs before any test module imports those, so we:
  1. Point HOME at a temp dir and create ~/.ch/tmp/ inside it.
  2. Set a dummy GROQ_API_KEY so search.py's client construction doesn't fail.
"""

import os
import sys
import tempfile

# --- import-time environment (must run before any src module is imported) ---

_tmp_home = tempfile.mkdtemp(prefix="index_ch_test_home_")
os.environ["HOME"] = _tmp_home
os.environ["GROQ_API_KEY"] = "test-key-not-real"

_ch_dir = os.path.join(_tmp_home, ".ch")
_chats_dir = os.path.join(_ch_dir, "tmp")
_cache_dir = os.path.join(_ch_dir, "index")
_tmp_cache = os.path.join(_cache_dir, "tmp")
os.makedirs(_chats_dir, exist_ok=True)
os.makedirs(_tmp_cache, exist_ok=True)

# --- fixtures ---

import sqlite3
import json
import numpy as np
import pytest

from build import get_connection, backfill_message_epochs, backfill_archived
from process import migrate
from retrieve.state import Session


def make_chat(
    raw=None,
    messages=None,
    file_path=None,
    epoch=None,
    summary=None,
    short_summary=None,
    embedding=None,
    error=None,
    archived=0,
):
    """Build a raw chat dict and/or DB row dict for test fixtures."""
    if raw is None:
        raw = {}
        if messages is not None:
            raw["messages"] = messages
        if epoch is not None:
            raw["timestamp"] = epoch
        raw.setdefault("platform", "test")
        raw.setdefault("model", "test-model")
        raw.setdefault("base_url", "http://localhost")
    return raw


@pytest.fixture
def db_conn():
    """In-memory sqlite with the full chats schema + all migrated columns."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE NOT NULL,
            raw TEXT NOT NULL,
            cleaned TEXT NOT NULL,
            token_estimate INTEGER NOT NULL,
            last_message_epoch INTEGER,
            content_hash TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    # add migrated columns that are NOT already in the CREATE TABLE
    conn.execute("ALTER TABLE chats ADD COLUMN archived INTEGER NOT NULL DEFAULT 0")
    conn.execute("ALTER TABLE chats ADD COLUMN summary TEXT")
    conn.execute("ALTER TABLE chats ADD COLUMN short_summary TEXT")
    conn.execute("ALTER TABLE chats ADD COLUMN embedding TEXT")
    conn.execute("ALTER TABLE chats ADD COLUMN error TEXT")
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def db_with_chats(db_conn):
    """DB with a few sample chats inserted, for tests that need data."""
    now = "2025-07-27T12:00:00+00:00"
    chats = [
        {
            "file_path": "/tmp/ch_session_1700000000.json",
            "raw": json.dumps(
                {
                    "messages": [
                        {"user": "hello", "bot": "hi there", "time": 1700000000},
                        {"user": "bye", "bot": "goodbye", "time": 1700000100},
                    ],
                    "platform": "test",
                    "model": "m",
                    "base_url": "u",
                    "timestamp": 1700000000,
                }
            ),
            "cleaned": "USER: hello\nBOT: hi there\nUSER: bye\nBOT: goodbye\n",
            "token_estimate": 10,
            "last_message_epoch": 1700000100,
            "summary": "A greeting chat.",
            "short_summary": "Saying hello and goodbye.",
            "embedding": json.dumps([0.1, 0.2, 0.3]),
            "archived": 0,
        },
        {
            "file_path": "/tmp/ch_session_1700000500.json",
            "raw": json.dumps(
                {
                    "messages": [
                        {"user": "question", "bot": "answer", "time": 1700000500},
                    ],
                    "platform": "test",
                    "model": "m",
                    "base_url": "u",
                    "timestamp": 1700000500,
                }
            ),
            "cleaned": "USER: question\nBOT: answer\n",
            "token_estimate": 5,
            "last_message_epoch": 1700000500,
            "summary": "A Q&A chat.",
            "short_summary": "Asking a question.",
            "embedding": json.dumps([0.4, 0.5, 0.6]),
            "archived": 0,
        },
        {
            "file_path": "/tmp/ch_session_1700001000.json",
            "raw": json.dumps(
                {
                    "messages": [
                        {"user": "old chat", "bot": "yes", "time": 1700001000},
                    ],
                    "platform": "test",
                    "model": "m",
                    "base_url": "u",
                    "timestamp": 1700001000,
                }
            ),
            "cleaned": "USER: old chat\nBOT: yes\n",
            "token_estimate": 5,
            "last_message_epoch": 1700001000,
            "summary": "An archived chat.",
            "short_summary": "Old archived session.",
            "embedding": json.dumps([0.7, 0.8, 0.9]),
            "archived": 1,
        },
    ]
    for c in chats:
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, "
            "last_message_epoch, content_hash, created_at, updated_at, "
            "archived, summary, short_summary, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                c["file_path"],
                c["raw"],
                c["cleaned"],
                c["token_estimate"],
                c["last_message_epoch"],
                "abc123",
                now,
                now,
                c["archived"],
                c["summary"],
                c["short_summary"],
                c["embedding"],
            ),
        )
    db_conn.commit()
    return db_conn


@pytest.fixture
def sample_meta():
    """A meta dict like load_vectors produces, keyed by chat id."""
    return {
        1: {
            "file_path": "/tmp/ch_session_1700000000.json",
            "summary": "A greeting chat.",
            "short_summary": "Saying hello and goodbye.",
            "last_message_epoch": 1700000100,
            "archived": False,
        },
        2: {
            "file_path": "/tmp/ch_session_1700000500.json",
            "summary": "A Q&A chat.",
            "short_summary": "Asking a question.",
            "last_message_epoch": 1700000500,
            "archived": False,
        },
        3: {
            "file_path": "/tmp/ch_session_1700001000.json",
            "summary": "An archived chat.",
            "short_summary": "Old archived session.",
            "last_message_epoch": 1700001000,
            "archived": True,
        },
    }


@pytest.fixture
def sample_ids():
    """Numpy array of chat ids matching sample_meta."""
    return np.array([1, 2, 3])


@pytest.fixture
def sample_mat():
    """Normalized embeddings matrix for 3 chats with 3-dim vectors."""
    mat = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ],
        dtype=np.float32,
    )
    norms = np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12, None)
    return mat / norms


@pytest.fixture
def sample_session(sample_meta, sample_ids, sample_mat):
    """A Session with mock conn and loaded vectors."""
    return Session(
        conn=MockConn(),
        ids=sample_ids,
        mat=sample_mat,
        meta=sample_meta,
    )


class MockConn:
    """Minimal mock sqlite connection for search() tests that don't touch DB."""

    def execute(self, *args, **kwargs):
        return []

    def commit(self):
        pass
