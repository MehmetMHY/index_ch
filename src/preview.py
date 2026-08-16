import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

from build import format_messages, is_auto_entry
from config import DB_PATH, PREVIEW_LIMIT


def format_timestamp(epoch):
    if not epoch:
        return "no date"
    try:
        dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    except (ValueError, OSError, OverflowError):
        return "no date"
    return dt.strftime("%b %d, %Y %H:%M UTC")


def filename_epoch(name):
    base = os.path.basename(name)
    digits = "".join(ch if ch.isdigit() else " " for ch in base).split()
    return int(digits[0]) if digits else None


def chat_epoch(file_path, last_message_epoch):
    return last_message_epoch or filename_epoch(file_path)


def format_messages_limited(messages, skip_noise=False, limit=PREVIEW_LIMIT):
    """Like format_messages, but stops once the text exceeds `limit` chars.
    For long chats this formats only the first ~20-30 messages instead of all
    of them, so preview computation stays fast regardless of chat size."""
    text = ""
    for msg in messages:
        user = msg.get("user") or ""
        bot = msg.get("bot") or ""
        if skip_noise and (is_auto_entry(user) or is_auto_entry(bot)):
            continue
        text = text + "USER: " + user + "\n" + "BOT: " + bot + "\n"
        if len(text) >= limit:
            break
    return text


def preview_chat_with_conn(conn, cid):
    """Render a chat's preview text using an existing SQLite connection. Used
    by the /ls precompute workers and background fill thread so they don't open
    a new connection per chat."""
    try:
        cid = int(cid)
    except (TypeError, ValueError):
        return "Invalid chat id."

    row = conn.execute(
        "SELECT file_path, summary, short_summary, raw, last_message_epoch, archived "
        "FROM chats WHERE id = ?",
        (cid,),
    ).fetchone()

    if row is None:
        return "Chat not found."

    file_path, summary, short_summary, raw, last_message_epoch, archived = row
    name = os.path.basename(file_path)
    ts = format_timestamp(chat_epoch(file_path, last_message_epoch))
    status = "archived (source file gone)" if archived else "active"

    try:
        data = json.loads(raw) if raw else None
        messages = data.get("messages", []) if data else []
    except (json.JSONDecodeError, AttributeError, TypeError):
        messages = []
    turns = len(messages)
    turns_line = f"{turns} turn{'s' if turns != 1 else ''}"

    parts = [
        name,
        ts,
        status,
        turns_line,
        "",
        "TL;DR:",
        "=" * 6,
        (short_summary or "(no short summary yet)").strip(),
        "",
        "Full Summary:",
        "=" * 13,
        (summary or "(no summary yet)").strip(),
        "",
        "Full Conversation:",
        "=" * 18,
        "",
    ]

    transcript = (
        format_messages_limited(messages, skip_noise=False, limit=PREVIEW_LIMIT)
        if messages
        else ""
    )

    if transcript:
        if len(transcript) > PREVIEW_LIMIT:
            transcript = transcript[:PREVIEW_LIMIT].rstrip() + "\n..."
        parts.append(transcript)
    else:
        parts.append("(raw content unavailable)")
    return "\n".join(parts)


def preview_chat(cid):
    """Standalone entry point: open a connection, compute, close. Used by the
    fzf fallback when no cached preview file exists yet."""
    conn = sqlite3.connect(DB_PATH)
    try:
        return preview_chat_with_conn(conn, cid)
    finally:
        conn.close()


def compute_and_save_preview(args):
    """Pool worker: compute one preview and write it atomically to
    tmp_dir/ls_preview_<id>.txt. Lives in preview.py (not retrieve.py) so the
    multiprocessing spawn only imports build+config, not numpy/openai/httpx."""
    cid, tmp_dir, db_path = args
    out_path = os.path.join(tmp_dir, f"ls_preview_{cid}.txt")
    if os.path.exists(out_path):
        return
    conn = sqlite3.connect(db_path)
    try:
        text = preview_chat_with_conn(conn, cid)
    except Exception:
        text = "Preview error."
    finally:
        conn.close()
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        f.write(text)
    os.replace(tmp_path, out_path)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg and arg.upper() == "HELP":
        print(
            "/ls keyboard shortcuts\n"
            "  Enter       pick a chat, then choose an action\n"
            "  Alt-j/k     scroll preview down/up\n"
            "  Alt-d/u     page preview down/up\n"
            "  Esc/Ctrl-C  exit\n"
            "  Type to fuzzy-filter the list"
        )
    else:
        print(preview_chat(arg))
