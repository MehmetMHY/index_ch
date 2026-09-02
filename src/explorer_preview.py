"""Lightweight preview renderer for the unified explorer.

Imports only stdlib + config (for DB_PATH), so multiprocessing.Pool
workers spawn fast (~50ms) without the numpy/openai/httpx/pydantic
import chain. This mirrors the design of preview.py.
"""

import json
import os
import re
import sqlite3
import sys
import textwrap
from datetime import datetime, timezone

from config import DB_PATH

# ANSI colors (matching retrieve.color for consistency)
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
BLUE = "\033[34m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"
WHITE = "\033[97m"

PREVIEW_WIDTH = 72
CODE_WIDTH = 82


def format_ts(epoch):
    """Format an epoch as MM/DD/YYYY HH:MMZ (unified for all TUI timestamps)."""
    if not epoch:
        return ""
    try:
        dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    except (ValueError, OSError, OverflowError):
        return ""
    return dt.strftime("%m/%d/%Y %H:%MZ")


def filename_epoch(file_path):
    """Extract epoch from a ch_session_<epoch>.json filename."""
    match = re.search(r"(\d{9,})", os.path.basename(file_path or ""))
    return int(match.group(1)) if match else None


def render_explorer_preview_with_conn(conn, cid):
    """Render the full preview pane: legend, metadata, transcript.

    Uses an existing SQLite connection (for synchronous pre-rendering
    and the background fill thread).
    """
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

    file_path, summary, short_summary, raw, epoch, archived = row
    name = os.path.basename(file_path)
    ts = format_ts(epoch)
    title = " ".join((short_summary or "").split()) or name
    status = "archived" if archived else "active"

    try:
        data = json.loads(raw) if raw else None
        messages = data.get("messages", []) if data else []
        model = data.get("model", "unknown") if data else "unknown"
    except (json.JSONDecodeError, AttributeError, TypeError):
        messages = []
        model = "unknown"

    turns = len(messages)
    turn_label = f"{turns} turn" if turns == 1 else f"{turns} turns"

    legend = (
        f"{CYAN}enter{RESET}   open in ch (single or merged)\n"
        f"{GREEN}ctrl-v{RESET}  view in editor ($EDITOR)\n"
        f"{YELLOW}ctrl-y{RESET}  copy filename\n"
        f"{BLUE}ctrl-s{RESET}  save to ~/Downloads\n"
        f"{MAGENTA}tab{RESET}     toggle select (multi-chat)\n"
        f"{MAGENTA}alt-j/k{RESET} scroll preview\n"
        f"{MAGENTA}esc{RESET}     back to search"
    )

    lines = [legend, ""]
    lines.append(f"{BOLD}{title}{RESET}")
    lines.append("")
    lines.append(f"{DIM}Date:{RESET}   {ts}")
    lines.append(f"{DIM}Model:{RESET}  {model}")
    lines.append(f"{DIM}Turns:{RESET}  {turn_label}")
    lines.append(f"{DIM}File:{RESET}   {name}")
    lines.append(f"{DIM}Status:{RESET} {status}")
    lines.append("")

    summary_text = (summary or "").strip()
    if summary_text:
        lines.append(f"{DIM}Summary:{RESET}")
        for sl in summary_text.splitlines() or [""]:
            if not sl:
                lines.append("")
                continue
            lines.extend(
                textwrap.wrap(
                    sl,
                    width=PREVIEW_WIDTH,
                    initial_indent="",
                    subsequent_indent="",
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
        lines.append("")

    def wrap_block(text, indent="  "):
        wrapped = []
        for raw_line in text.splitlines() or [""]:
            if not raw_line:
                wrapped.append("")
                continue
            wrapped.extend(
                textwrap.wrap(
                    raw_line,
                    width=PREVIEW_WIDTH,
                    initial_indent=indent,
                    subsequent_indent=indent,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
        return wrapped

    def clip_code(line):
        return line if len(line) <= CODE_WIDTH else line[: CODE_WIDTH - 1] + "..."

    for idx, msg in enumerate(messages, 1):
        lines.append(f"{BOLD}{BLUE}-- Turn {idx} --{RESET}")
        user_text = msg.get("user") or ""
        bot_text = msg.get("bot") or ""
        lines.append(f"{BOLD}{WHITE}User:{RESET}")
        lines.extend(wrap_block(user_text))
        lines.append("")
        lines.append(f"{BOLD}{GREEN}Assistant:{RESET}")
        in_code = False
        for bot_line in bot_text.splitlines():
            if bot_line.startswith("```"):
                in_code = not in_code
                lines.append(f"  {DIM}{bot_line}{RESET}")
            elif in_code:
                lines.append(f"  {DIM}{clip_code(bot_line)}{RESET}")
            elif bot_line.startswith("`") and bot_line.endswith("`"):
                lines.append(f"  {CYAN}{clip_code(bot_line)}{RESET}")
            else:
                lines.extend(wrap_block(bot_line))
        lines.append("")

    return "\n".join(lines)


def render_explorer_preview(cid, db_path=DB_PATH):
    """Standalone entry point: open a connection, compute, close."""
    conn = sqlite3.connect(db_path)
    try:
        return render_explorer_preview_with_conn(conn, cid)
    finally:
        conn.close()


def compute_and_save_preview(args):
    """Pool worker: compute one preview and write it atomically to
    tmp_dir/preview_<id>.txt. Lives here (not in retrieve) so the
    multiprocessing spawn only imports config, not numpy/openai/httpx."""
    cid, tmp_dir, db_path = args
    out_path = os.path.join(tmp_dir, f"preview_{cid}.txt")
    if os.path.exists(out_path):
        return
    conn = sqlite3.connect(db_path)
    try:
        text = render_explorer_preview_with_conn(conn, cid)
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
    if arg:
        print(render_explorer_preview(arg))
