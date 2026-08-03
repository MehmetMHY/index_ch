from multiprocessing import Pool
import time
import json
import os
import re
import hashlib
import sqlite3
from datetime import datetime, timezone

from config import DB_PATH, CHATS_SOURCE_DIR


def file_hash(path):
    """SHA-256 of a file's raw bytes, used to detect when a chat file changed
    (e.g. a session resumed after ingest gains new messages)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_paths(dir_path):
    return [
        os.path.join(dir_path, name)
        for name in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, name))
    ]


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
    """Read, hash, and clean one chat file from a single read, so the stored hash
    always matches the stored content. Returns {raw, cleaned, content_hash}, or
    None if the file can't be read or parsed (e.g. caught mid-write by Ch, or
    deleted) so one bad file is skipped instead of crashing the whole build."""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        raw = json.loads(data)
        cleaned = format_messages(raw["messages"], skip_noise=True)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"skipping {os.path.basename(file_path)}: {exc}")
        return None
    return {
        "raw": raw,
        "cleaned": cleaned,
        "content_hash": hashlib.sha256(data).hexdigest(),
    }


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
            content_hash TEXT,
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


def backfill_content_hashes(conn):
    """Add content_hash and populate it from the current on-disk files.

    Idempotent: the one-time backfill runs only when the column is first added,
    then returns immediately. It blesses the current state (no re-processing):
    existing chats are recorded as-is, so only files that change *after* this
    are re-ingested. Makes no API calls.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(chats)")}
    if "content_hash" in cols:
        return
    conn.execute("ALTER TABLE chats ADD COLUMN content_hash TEXT")
    updates = []
    for row_id, file_path in conn.execute("SELECT id, file_path FROM chats").fetchall():
        try:
            updates.append((file_hash(file_path), row_id))
        except OSError:
            pass  # source file gone; leave NULL so a later build re-checks it
    conn.executemany("UPDATE chats SET content_hash = ? WHERE id = ?", updates)
    conn.commit()
    print(f"Backfilled content_hash for {len(updates)} chats.")


def backfill_archived(conn):
    """Add the archived flag (0 = source file present, 1 = gone). Idempotent:
    runs only when the column is first added. Existing rows default to 0
    (present), so paid summaries/embeddings are never touched and no disk scan
    or API call happens here. build.py's main loop sets archived = 1 for rows
    whose source file vanished from ~/.ch/tmp/; deleted chats stay searchable
    (their summary/embedding/raw are cached in the DB) but are flagged so
    retrieve.py can hide them by default and warn on /run, /dump, /copy.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(chats)")}
    if "archived" in cols:
        return
    conn.execute("ALTER TABLE chats ADD COLUMN archived INTEGER NOT NULL DEFAULT 0")
    conn.commit()
    print("Added archived column.")


def stored_hashes(conn):
    """Map of file_path -> content_hash for every chat already in the DB."""
    return dict(conn.execute("SELECT file_path, content_hash FROM chats"))


def insert_entries(conn, entries):
    """Insert new chats. entries: iterable of (file_path, result, content_hash)."""
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        """
        INSERT INTO chats (file_path, raw, cleaned, token_estimate, last_message_epoch, content_hash, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                file_path,
                json.dumps(result["raw"]),
                result["cleaned"],
                estimate_tokens(result["cleaned"]),
                message_epoch(result["raw"]),
                content_hash,
                now,
                now,
            )
            for file_path, result, content_hash in entries
        ],
    )
    conn.commit()


def update_entries(conn, entries):
    """Re-ingest changed chats in place. entries: (file_path, result, hash).

    Clears summary/embedding/error (when those columns exist) so process.py
    re-summarizes and re-embeds the chat, and bumps updated_at so retrieve.py's
    FTS and embeddings caches rebuild.
    """
    entries = list(entries)
    if not entries:
        return
    now = datetime.now(timezone.utc).isoformat()
    cols = {row[1] for row in conn.execute("PRAGMA table_info(chats)")}
    clear = "".join(
        f", {c} = NULL" for c in ("summary", "embedding", "error") if c in cols
    )
    if "archived" in cols:
        clear += ", archived = 0"
    conn.executemany(
        f"""
        UPDATE chats
        SET raw = ?, cleaned = ?, token_estimate = ?, last_message_epoch = ?,
            content_hash = ?, updated_at = ?{clear}
        WHERE file_path = ?
        """,
        [
            (
                json.dumps(result["raw"]),
                result["cleaned"],
                estimate_tokens(result["cleaned"]),
                message_epoch(result["raw"]),
                content_hash,
                now,
                file_path,
            )
            for file_path, result, content_hash in entries
        ],
    )
    conn.commit()


if __name__ == "__main__":
    start_time = time.time()

    json_paths = [f for f in file_paths(CHATS_SOURCE_DIR) if f.endswith(".json")]

    conn = get_connection()
    backfill_message_epochs(conn)
    backfill_content_hashes(conn)
    backfill_archived(conn)

    # hash every file on disk and diff against what the DB has stored: unknown
    # paths are new, known paths whose hash moved are resumed/edited chats. skip
    # any file that vanishes mid-scan; a later build picks it up
    disk_hashes = {}
    for p in json_paths:
        try:
            disk_hashes[p] = file_hash(p)
        except OSError as exc:
            print(f"skipping {os.path.basename(p)}: {exc}")

    stored = stored_hashes(conn)
    archived_flags = dict(conn.execute("SELECT file_path, archived FROM chats"))
    new_paths = [p for p in disk_hashes if p not in stored]
    changed_paths = [
        p for p in disk_hashes if p in stored and stored[p] != disk_hashes[p]
    ]

    # a chat whose source file vanished from ~/.ch/tmp/ is flagged archived but
    # never deleted (its summary/embedding/raw are cached and paid for). only
    # flip rows that are not already archived, so a steady-state build does not
    # bump updated_at and needlessly invalidate retrieve.py's caches. a file that
    # reappears (archived=1 but back on disk) is un-archived; if its hash also
    # changed it is re-ingested via changed_paths (which clears archived too).
    now = datetime.now(timezone.utc).isoformat()
    newly_missing = [
        p for p in stored if p not in disk_hashes and archived_flags.get(p) != 1
    ]
    reappeared = [p for p in disk_hashes if archived_flags.get(p) == 1]
    if newly_missing:
        conn.executemany(
            "UPDATE chats SET archived = 1, updated_at = ? WHERE file_path = ?",
            [(now, p) for p in newly_missing],
        )
        conn.commit()
    if reappeared:
        conn.executemany(
            "UPDATE chats SET archived = 0, updated_at = ? WHERE file_path = ?",
            [(now, p) for p in reappeared],
        )
        conn.commit()

    # load only new/changed files; load_and_clean returns None for any it could
    # not read or parse (e.g. caught mid-write), so those are skipped and left for
    # a later build. the stored hash comes from load_and_clean's own read, so it
    # always matches the content actually written.
    to_load = new_paths + changed_paths
    results = {}
    if to_load:
        with Pool() as pool:
            loaded = pool.map(load_and_clean, to_load)
        results = {p: r for p, r in zip(to_load, loaded) if r is not None}
        insert_entries(
            conn,
            [
                (p, results[p], results[p]["content_hash"])
                for p in new_paths
                if p in results
            ],
        )
        update_entries(
            conn,
            [
                (p, results[p], results[p]["content_hash"])
                for p in changed_paths
                if p in results
            ],
        )

    conn.close()

    runtime = time.time() - start_time
    added = sum(1 for p in new_paths if p in results)
    updated = sum(1 for p in changed_paths if p in results)
    skipped = len(to_load) - len(results)
    summary = f"Added {added} new, updated {updated} changed chats"
    if newly_missing:
        summary += f", {len(newly_missing)} archived (source file gone, kept)"
    if reappeared:
        summary += f", {len(reappeared)} un-archived (source file returned)"
    if skipped:
        summary += f" ({skipped} skipped: unreadable or mid-write)"
    print(f"{summary}. Runtime: {runtime:.2f} seconds")
