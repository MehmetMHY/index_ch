#!/usr/bin/env python3
"""Unified entry point for index_ch.

Prompts to update the cache, then drops into a REPL where:
- Empty ENTER browses all chats (newest to oldest)
- A typed query runs hybrid vector + keyword search with LLM reranking
- /history re-runs a past query
- /exit quits

Both browse and search open a unified fzf split-view explorer with a
side-by-side transcript preview and direct key actions (open in Ch,
view in editor, copy filename, save to Downloads, multi-select merge).
"""

import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from multiprocessing import Pool

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

CH_DIR = os.path.join(os.path.expanduser("~"), ".ch")
CHATS_SOURCE_DIR = os.path.join(CH_DIR, "tmp")

BUILD_SCRIPT = os.path.join(SRC_DIR, "build.py")
PROCESS_SCRIPT = os.path.join(SRC_DIR, "process.py")
EXPLORER_PREVIEW_SCRIPT = os.path.join(SRC_DIR, "explorer_preview.py")

PY_CALL = os.path.join(ROOT_DIR, "env/bin/python3")
if not os.path.isfile(PY_CALL):
    PY_CALL = "python3"

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

SEARCH_HISTORY = []


def require_ch_dirs():
    """Exit if ~/.ch/ or ~/.ch/tmp/ don't exist."""
    if not os.path.isdir(CH_DIR):
        print(
            "error: ~/.ch/ does not exist. Install Ch and configure local mode first: "
            "https://github.com/MehmetMHY/ch",
            file=sys.stderr,
        )
        sys.exit(1)
    if not os.path.isdir(CHATS_SOURCE_DIR):
        print(
            "error: ~/.ch/tmp/ does not exist. Install Ch and configure local mode first: "
            "https://github.com/MehmetMHY/ch",
            file=sys.stderr,
        )
        sys.exit(1)


def format_ts(epoch):
    """Format an epoch as MM/DD/YYYY HH:MMZ (unified for all TUI timestamps)."""
    if not epoch:
        return ""
    try:
        dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    except (ValueError, OSError, OverflowError):
        return ""
    return dt.strftime("%m/%d/%Y %H:%MZ")


def _filename_epoch(file_path):
    """Extract epoch from a ch_session_<epoch>.json filename."""
    match = re.search(r"(\d{9,})", os.path.basename(file_path or ""))
    return int(match.group(1)) if match else None


def _drain_stdin():
    """Discard keystrokes typed during the startup spinner / slow imports."""
    if not sys.stdin.isatty():
        return
    try:
        import termios

        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except (ImportError, OSError, ValueError):
        pass


def prompt_update_cache():
    """Ask whether to update the cache before entering the REPL.
    fzf picker with no (default) / yes. Returns True for yes."""
    if shutil.which("fzf") is None:
        return False
    proc = subprocess.run(
        [
            "fzf",
            "--prompt=update cache? > ",
            "--cycle",
            "--layout=reverse",
            "--height=~5",
            "--no-info",
            "--no-separator",
            "--bind=esc:abort,ctrl-c:abort,ctrl-d:abort",
        ],
        input="\n".join(["no", "yes"]),
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and proc.stdout.strip() == "yes"


def run_cache_update():
    """Run build.py then process.py as subprocesses."""
    for script, label in [
        (BUILD_SCRIPT, "Scanning Ch exports..."),
        (PROCESS_SCRIPT, "Processing pending chats..."),
    ]:
        print(f"{DIM}{label}{RESET}", flush=True)
        status = os.system(f"{PY_CALL} {script}")
        if status != 0:
            print(f"error: {os.path.basename(script)} failed with status {status}")
            sys.exit(1)


def print_banner():
    print(
        f"{BLUE}Type to search {BOLD}or{RESET}{BLUE} {BOLD}ENTER{RESET}{BLUE} to browse{RESET}"
    )
    print(
        f"{YELLOW}/history{RESET}, {YELLOW}/help{RESET}, {BOLD}{MAGENTA}or{RESET} {YELLOW}/exit{RESET}"
    )


def open_history_picker():
    """Open fzf with past search queries. Returns the selected query or None."""
    if not SEARCH_HISTORY:
        print(f"{DIM}No search history yet.{RESET}")
        return None
    if shutil.which("fzf") is None:
        print(f"{RED}fzf not found on PATH.{RESET}")
        return None

    lines = []
    for i, (query, ts) in enumerate(reversed(SEARCH_HISTORY), 1):
        lines.append(f"{i:>2}. {DIM}{ts}{RESET}  {query}")

    proc = subprocess.run(
        [
            "fzf",
            "--ansi",
            "--prompt=> ",
            "--cycle",
            "--layout=reverse",
            "--no-sort",
            "--height=40%",
            "--bind=esc:abort,ctrl-c:abort,ctrl-d:abort",
        ],
        input="\n".join(lines),
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0 or not proc.stdout.strip():
        return None

    selected = proc.stdout.strip()
    parts = selected.split("  ", 1)
    return parts[-1].strip() if len(parts) > 1 else selected


def _fill_remaining_previews(cids, tmp_dir, db_path, stop_event):
    """Daemon thread: sequentially compute previews for chats the Pool didn't
    get to. Low-priority background fill so browsing the top of the list stays
    instant while the long tail is prepared."""
    from explorer_preview import render_explorer_preview_with_conn

    conn = sqlite3.connect(db_path)
    try:
        for cid in cids:
            if stop_event.is_set():
                break
            out_path = os.path.join(tmp_dir, f"preview_{cid}.txt")
            if os.path.exists(out_path):
                continue
            try:
                text = render_explorer_preview_with_conn(conn, cid)
            except Exception:
                text = "Preview error."
            tmp_path = out_path + ".tmp"
            with open(tmp_path, "w") as f:
                f.write(text)
            os.replace(tmp_path, out_path)
    finally:
        conn.close()


def open_explorer(cids, conn, meta, is_search=False):
    """Launch the unified fzf split-view explorer.

    cids: list of chat ids to show (ranked for search, recency-sorted for browse).
    Returns None (always returns to the prompt).
    """
    if shutil.which("fzf") is None:
        print(f"{RED}fzf not found on PATH - install it to use this tool.{RESET}")
        return

    if not cids:
        print(f"{DIM}No chats to show.{RESET}")
        return

    from config import DB_PATH, PREVIEW_BATCH
    from explorer_preview import (
        render_explorer_preview_with_conn,
        compute_and_save_preview,
    )
    from retrieve.spinner import Spinner

    temp_dir = tempfile.mkdtemp(prefix="index_ch_explorer_")

    # Build fzf input lines
    lines = []
    for rank, cid in enumerate(cids, 1):
        info = meta.get(cid, {})
        epoch = info.get("last_message_epoch") or _filename_epoch(
            info.get("file_path", "")
        )
        ts = format_ts(epoch)
        name = os.path.basename(info.get("file_path", f"chat_{cid}.json"))
        lines.append(f"{cid}\t{rank:04d}  {ts}  {name}")

    # Pre-render previews: few results synchronously, many in parallel + background
    stop_event = None
    fill_thread = None

    if len(cids) <= 50:
        for cid in cids:
            try:
                text = render_explorer_preview_with_conn(conn, cid)
            except Exception:
                text = "Preview error."
            with open(os.path.join(temp_dir, f"preview_{cid}.txt"), "w") as f:
                f.write(text)
    else:
        batch_cids = cids[:PREVIEW_BATCH]
        pool_args = [(cid, temp_dir, DB_PATH) for cid in batch_cids]
        stop_event = threading.Event()
        fill_cids = cids[PREVIEW_BATCH:]
        fill_thread = threading.Thread(
            target=_fill_remaining_previews,
            args=(fill_cids, temp_dir, DB_PATH, stop_event),
            daemon=True,
        )
        try:
            with Spinner(f"precomputing {len(batch_cids)} previews"):
                with Pool(processes=min(len(batch_cids), os.cpu_count() or 4)) as pool:
                    pool.map(compute_and_save_preview, pool_args, chunksize=1)
            fill_thread.start()
        except KeyboardInterrupt:
            stop_event.set()
            shutil.rmtree(temp_dir, ignore_errors=True)
            return

    # fzf preview: try cached file first, fall back to live explorer_preview.py
    import shlex

    preview_script = os.path.join(temp_dir, "preview.sh")
    with open(preview_script, "w") as f:
        f.write("#!/bin/sh\n")
        f.write(f'cat "{temp_dir}/preview_$1.txt" 2>/dev/null')
        f.write(
            f" || {shlex.quote(PY_CALL)} {shlex.quote(EXPLORER_PREVIEW_SCRIPT)} $1\n"
        )
    os.chmod(preview_script, 0o755)

    prompt = "results> " if is_search else "browse> "

    cmd = [
        "fzf",
        "--ansi",
        f"--prompt={prompt}",
        "--cycle",
        "--layout=reverse",
        "--no-separator",
        "--delimiter=\t",
        "--with-nth=2",
        "--nth=2",
        "--no-sort",
        "--multi",
        "--info=inline-right",
        "--height=96%",
        f"--preview={preview_script} {{1}}",
        "--preview-window=right:62%:wrap:border-left",
        "--bind=alt-j:preview-down,alt-k:preview-up,alt-d:preview-page-down,alt-u:preview-page-up",
        "--bind=esc:abort,ctrl-c:abort,ctrl-d:abort",
        "--expect=ctrl-v,ctrl-y,ctrl-s",
    ]

    proc = None
    try:
        proc = subprocess.run(
            cmd, input="\n".join(lines), capture_output=True, text=True
        )
    finally:
        if stop_event:
            stop_event.set()
        if fill_thread and fill_thread.is_alive():
            fill_thread.join(timeout=2.0)
        shutil.rmtree(temp_dir, ignore_errors=True)

    if proc is None or proc.returncode != 0 or not proc.stdout:
        return True

    output = proc.stdout.splitlines()
    if not output:
        return True

    key_pressed = output[0].strip().lower()
    selected_lines = [line for line in output[1:] if line.strip()]

    selected_cids = []
    for line in selected_lines:
        if "\t" in line:
            cid = int(line.split("\t", 1)[0])
            selected_cids.append(cid)

    if not selected_cids:
        return True

    is_multi = len(selected_cids) > 1

    if key_pressed == "ctrl-v":
        action_view_editor(selected_cids[0], conn, meta)
        return True
    elif key_pressed == "ctrl-y":
        action_copy_filenames(selected_cids, meta)
        return True
    elif key_pressed == "ctrl-s":
        action_save_downloads(selected_cids, conn, meta)
        return True
    else:
        return action_open_in_ch(selected_cids, conn, meta, is_multi)


def action_open_in_ch(cids, conn, meta, is_multi):
    """Enter: open single chat in ch, or merge multiple and open the merge."""
    from retrieve.pickers import confirm_return
    from retrieve.actions import run_chat
    from retrieve.cmds.dump import build_dump, load_dump_in_ch

    if is_multi:
        built = build_dump(cids, meta)
        if built is None:
            return True
        merged, skipped = built
        filename = (
            f"index_ch_dump_{len(merged['source_files'])}_{int(time.time())}.json"
        )
        return load_dump_in_ch(merged, filename, keep=False, skipped=skipped)
    else:
        cid = cids[0]
        if shutil.which("ch") is None:
            print(f"{RED}ch not found on PATH - https://github.com/MehmetMHY/ch{RESET}")
            return True
        ret = confirm_return("return to search? > ")
        if ret is None:
            return True
        run_chat(cid, meta, reprint_prompt=ret)
        return ret


def action_view_editor(cid, conn, meta):
    """Ctrl-V: open the full transcript in $EDITOR, delete temp file on exit."""
    from retrieve.actions import view_chat

    view_chat(conn, cid, meta)


def action_copy_filenames(cids, meta):
    """Ctrl-Y: copy filename(s) to clipboard."""
    from retrieve.actions import copy_to_clipboard

    if len(cids) == 1:
        name = os.path.basename(meta[cids[0]]["file_path"])
    else:
        name = "\n".join(os.path.basename(meta[c]["file_path"]) for c in cids)
    if copy_to_clipboard(name):
        label = "filename" if len(cids) == 1 else f"{len(cids)} filenames"
        print(f"{YELLOW}Copied {label} to clipboard.{RESET}")
    else:
        print(f"{YELLOW}{name}{RESET}")


def action_save_downloads(cids, conn, meta):
    """Ctrl-S: save single chat file or merged dump to ~/Downloads."""
    from retrieve.cmds.dump import (
        build_dump,
        save_dump_to_downloads,
        unique_path,
    )

    is_multi = len(cids) > 1
    out_dir = os.path.expanduser("~/Downloads")
    os.makedirs(out_dir, exist_ok=True)

    if is_multi:
        built = build_dump(cids, meta)
        if built is None:
            return
        merged, skipped = built
        filename = (
            f"index_ch_dump_{len(merged['source_files'])}_{int(time.time())}.json"
        )
        save_dump_to_downloads(merged, filename, skipped)
    else:
        cid = cids[0]
        info = meta[cid]
        path = info["file_path"]
        name = os.path.basename(path)
        if info.get("archived"):
            row = conn.execute("SELECT raw FROM chats WHERE id = ?", (cid,)).fetchone()
            if row:
                out_path = unique_path(os.path.join(out_dir, name))
                with open(out_path, "w") as f:
                    f.write(row[0])
                print(f"{BLUE}Saved {name} to {out_path}{RESET}")
            else:
                print(f"{RED}Could not read archived chat {name}.{RESET}")
        else:
            if os.path.exists(path):
                out_path = unique_path(os.path.join(out_dir, name))
                shutil.copy2(path, out_path)
                print(f"{BLUE}Saved {name} to {out_path}{RESET}")
            else:
                print(f"{RED}Source file not found: {path}{RESET}")


def main():
    """Entry point: update cache prompt, then unified search/browse REPL."""
    require_ch_dirs()

    if prompt_update_cache():
        run_cache_update()

    # Start spinner before heavy imports (numpy, openai, httpx, pydantic)
    from retrieve.spinner import start_startup_spinner, stop_startup_spinner, Spinner

    start_startup_spinner("Starting...")

    from build import get_connection, backfill_message_epochs, backfill_archived
    from config import NUM_EXPANSIONS
    from pricing import warm
    from retrieve.cache import ensure_fts, load_vectors
    from retrieve.search import search, warm_connections
    from retrieve.state import Session
    from retrieve.cmds.ls import list_chats_by_recency

    # Warm API connections in background while DB setup runs
    threading.Thread(target=warm_connections, daemon=True).start()

    conn = get_connection()
    try:
        backfill_message_epochs(conn)
        backfill_archived(conn)
        stop_startup_spinner()
        _drain_stdin()
        ensure_fts(conn)
        ids, mat, meta = load_vectors(conn)
        warm()
    except KeyboardInterrupt:
        print()
        conn.close()
        return 0

    session = Session(
        conn=conn,
        ids=ids,
        mat=mat,
        meta=meta,
        do_rerank=True,
        do_expand=NUM_EXPANSIONS > 0,
        show_archived=False,
        time_filter=None,
        result_len=9999,
    )

    _drain_stdin()
    print_banner()

    while True:
        try:
            query = input(f"{BOLD}{BLUE}> {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not query:
            # Browse all chats newest-to-oldest
            rows = list_chats_by_recency(
                conn, session.show_archived, session.time_filter
            )
            browse_cids = [cid for cid, _ in rows]
            browse_meta = dict(meta)
            for cid, info in rows:
                if cid not in browse_meta:
                    browse_meta[cid] = info
            if not open_explorer(browse_cids, conn, browse_meta, is_search=False):
                break
            continue

        if query.lower() in ("/exit", "/quit", ":q", "exit", "quit"):
            break

        if query.lower() in ("/help", "/h"):
            print_banner()
            continue

        if query.lower() in ("/history", "/hist"):
            picked = open_history_picker()
            if picked:
                ts = time.strftime("%m/%d/%Y %H:%MZ")
                SEARCH_HISTORY.append((picked, ts))
                try:
                    with Spinner("searching"):
                        results, usage = search(session, picked)
                except KeyboardInterrupt:
                    print()
                    continue
                result_cids = [cid for cid, _ in results]
                if not result_cids:
                    print(f"{DIM}No matches.{RESET}")
                    continue
                if not open_explorer(result_cids, conn, meta, is_search=True):
                    break
            continue

        # Search
        ts = time.strftime("%m/%d/%Y %H:%MZ")
        SEARCH_HISTORY.append((query, ts))
        try:
            with Spinner("searching"):
                results, usage = search(session, query)
        except KeyboardInterrupt:
            print()
            continue
        result_cids = [cid for cid, _ in results]
        if not result_cids:
            print(f"{DIM}No matches.{RESET}")
            continue
        if not open_explorer(result_cids, conn, meta, is_search=True):
            break

    conn.close()
    return 0


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
    sys.exit(0)
