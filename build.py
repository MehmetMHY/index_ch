from multiprocessing import Pool
import time
import json
import os
import re
import sqlite3
from datetime import datetime, timezone

from config import DB_PATH, CHATS_SOURCE_DIR


def file_paths(dir_path):
    return [
        os.path.join(dir_path, name)
        for name in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, name))
    ]


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def estimate_tokens(text):
    if not text:
        return 0

    # count words and punctuation-like chunks
    chunks = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    # average English word is often around 1.3 tokens
    word_based_estimate = len(chunks) * 1.3

    # character-based estimate, common rough rule
    char_based_estimate = len(text) / 4

    # blend both estimates for a more stable guess
    return round((word_based_estimate + char_based_estimate) / 2)


def is_auto_entry(content):
    if not content:
        return False

    if content.startswith("File: "):
        return True

    if "=== Code Dump ===" in content or "=== FILE:" in content:
        return True

    if "The user executed the following command and here is the output:" in content:
        return True

    if "[code-dump starts]" in content.lower():
        return True

    return False


def format_messages(messages, skip_noise=True):
    """Join USER/BOT turns into plain text.

    skip_noise=True drops auto-generated noise (code dumps, file pastes, command
    output) per is_auto_entry; False keeps everything, for the raw chat view.
    """
    text = ""
    for msg in messages:
        user = msg["user"] or ""
        bot = msg["bot"] or ""
        if skip_noise and (is_auto_entry(user) or is_auto_entry(bot)):
            continue
        text = text + "USER: " + user + "\n" + "BOT: " + bot + "\n"
    return text


def message_epoch(raw):
    """Epoch of the chat's last message, or None if unavailable.

    This is the real "last active" time, unlike the filename epoch which is
    frozen when a session is first saved and goes stale for resumed/continued
    chats. Scans from the end so a trailing message missing a time still yields
    the most recent valid one.
    """
    for msg in reversed(raw.get("messages") or []):
        t = msg.get("time")
        if isinstance(t, (int, float)) and t > 0:
            return int(t)
    return None


def load_and_clean(file_path):
    raw = load_json(file_path)
    return {"raw": raw, "cleaned": format_messages(raw["messages"], skip_noise=True)}


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE NOT NULL,
            raw TEXT NOT NULL,
            cleaned TEXT NOT NULL,
            token_estimate INTEGER NOT NULL,
            last_message_epoch INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """)
    return conn


def backfill_message_epochs(conn):
    """Add and populate last_message_epoch from the raw JSON already in the DB.

    Idempotent and cheap: it runs the one-time backfill only when the column is
    first added, then returns immediately on every later call. Makes no API
    calls and reads no source files, so it never triggers re-processing.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(chats)")}
    if "last_message_epoch" in cols:
        return
    conn.execute("ALTER TABLE chats ADD COLUMN last_message_epoch INTEGER")
    updates = []
    for row_id, raw_json in conn.execute("SELECT id, raw FROM chats"):
        try:
            epoch = message_epoch(json.loads(raw_json))
        except (TypeError, ValueError):
            epoch = None
        if epoch is not None:
            updates.append((epoch, row_id))
    conn.executemany("UPDATE chats SET last_message_epoch = ? WHERE id = ?", updates)
    conn.commit()
    print(f"Backfilled last_message_epoch for {len(updates)} chats.")


def existing_file_paths(conn):
    rows = conn.execute("SELECT file_path FROM chats").fetchall()
    return {row[0] for row in rows}


def insert_entries(conn, entries):
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        """
        INSERT INTO chats (file_path, raw, cleaned, token_estimate, last_message_epoch, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                file_path,
                json.dumps(result["raw"]),
                result["cleaned"],
                estimate_tokens(result["cleaned"]),
                message_epoch(result["raw"]),
                now,
                now,
            )
            for file_path, result in entries
        ],
    )
    conn.commit()


if __name__ == "__main__":
    start_time = time.time()

    json_paths = [f for f in file_paths(CHATS_SOURCE_DIR) if f.endswith(".json")]

    conn = get_connection()
    backfill_message_epochs(conn)
    known_paths = existing_file_paths(conn)
    new_paths = [p for p in json_paths if p not in known_paths]

    if new_paths:
        with Pool() as pool:
            results = pool.map(load_and_clean, new_paths)
        insert_entries(conn, zip(new_paths, results))

    conn.close()

    runtime = time.time() - start_time

    print(f"Added {len(new_paths)} new chats. Runtime: {runtime:.2f} seconds")
