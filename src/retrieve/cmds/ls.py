import os
import shlex
import shutil
import sqlite3
import threading
import subprocess
from multiprocessing import Pool

from config import TMP_DIR, DB_PATH, PREVIEW_BATCH
from preview import preview_chat_with_conn, compute_and_save_preview

from ..spinner import Spinner
from ..display import chat_epoch, format_list_timestamp, chat_preview
from ..actions import run_chat, copy_chat
from ..state import Session
from .. import color


def list_chats_by_recency(conn, show_archived, time_filter):
    """Return chat info dicts sorted newest->oldest by chat_epoch, filtered by
    the active show_archived and time_filter settings. Chats with no epoch sort
    last. Pulls every row from the DB (not just embedded ones), so unprocessed
    chats appear too."""
    from ..search import range_bounds

    lo, hi = range_bounds(time_filter) if time_filter else (None, None)
    rows = conn.execute(
        "SELECT id, file_path, summary, short_summary, last_message_epoch, archived, "
        "LENGTH(raw) AS raw_size FROM chats"
    ).fetchall()
    out = []
    for r in rows:
        cid, file_path, summary, short_summary, epoch, archived, raw_size = r
        if not show_archived and archived:
            continue
        info = {
            "file_path": file_path,
            "summary": summary,
            "short_summary": short_summary,
            "last_message_epoch": epoch,
            "archived": bool(archived),
            "raw_size": raw_size or 0,
        }
        e = chat_epoch(info)
        if lo is not None and (e is None or e < lo or (hi is not None and e > hi)):
            continue
        out.append((cid, info, e))
    out.sort(key=lambda x: (x[2] is None, -(x[2] or 0)))
    return [(cid, info) for cid, info, _ in out]


LS_HELP_TEXT = """\
/ls keyboard shortcuts
  Enter       pick a chat, then choose an action
  Alt-j/k     scroll preview down/up
  Alt-d/u     page preview down/up
  Esc/Ctrl-C  exit
  Type to fuzzy-filter the list
"""

# actions offered after picking a chat from /ls
LS_ACTIONS = [
    ("Open in ch (ch -f)", "run"),
    ("Copy filename to clipboard", "copy"),
    ("Exit/Cancel", "cancel"),
]


def pick_ls_action():
    """fzf-pick what to do with a chat chosen via /ls. Returns one of 'run',
    'copy', or 'cancel' (also 'cancel' if fzf is missing)."""
    if shutil.which("fzf") is None:
        print(color.red("fzf not found on PATH - cannot pick an action."))
        return "cancel"
    label_to_action = {label: action for label, action in LS_ACTIONS}
    proc = subprocess.run(
        ["fzf", "--prompt=action> ", "--cycle", "--layout=reverse"],
        input="\n".join(label for label, _ in LS_ACTIONS),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return "cancel"
    return label_to_action.get(proc.stdout.strip(), "cancel")


def _fill_remaining_previews(cids, tmp_dir, db_path, stop_event):
    """Daemon thread: sequentially compute previews for chats the Pool didn't
    get to. Low-priority background fill so browsing the top of the list stays
    instant while the long tail is prepared."""
    conn = sqlite3.connect(db_path)
    try:
        for cid in cids:
            if stop_event.is_set():
                break
            out_path = os.path.join(tmp_dir, f"ls_preview_{cid}.txt")
            if os.path.exists(out_path):
                continue
            try:
                text = preview_chat_with_conn(conn, cid)
            except Exception:
                text = "Preview error."
            tmp_path = out_path + ".tmp"
            with open(tmp_path, "w") as f:
                f.write(text)
            os.replace(tmp_path, out_path)
    finally:
        conn.close()


def _cleanup_previews(tmp_dir):
    """Delete all ls_preview_*.txt files from tmp_dir."""
    for name in os.listdir(tmp_dir):
        if name.startswith("ls_preview_") and name.endswith(".txt"):
            try:
                os.remove(os.path.join(tmp_dir, name))
            except OSError:
                pass


def pick_latest_with_fzf(rows):
    """fzf-pick one chat from a full newest->oldest list. Each line is
    '[UTC timestamp] short summary', or '[UTC timestamp] filename' when the chat
    has not been processed yet. The chat id is a hidden first tab-delimited
    field. Returns (cid, info), or (None, None) if fzf is missing or the user
    cancelled."""
    if not rows:
        print(color.yellow("No chats to list."))
        return None, None
    if shutil.which("fzf") is None:
        print(color.red("fzf not found on PATH - install it to use /ls."))
        return None, None

    import sys

    lines = []
    for cid, info in rows:
        ts = format_list_timestamp(chat_epoch(info))
        ts_tag = f"[{ts}]" if ts else "[no date]"
        short = " ".join((info.get("short_summary") or "").split())
        label = short or os.path.basename(info["file_path"])
        lines.append(f"{cid}\t{ts_tag} {label}")

    info_map = dict(rows)
    all_cids = [cid for cid, _ in rows]

    # precompute the most recent N previews in parallel before fzf opens so the
    # top of the list is instant to browse; the rest are filled in the background.
    # sort the batch largest-raw-first so big chats (slow json.loads) start early
    # and are absorbed into the parallel work instead of straggling at the tail;
    # chunksize=1 lets each worker grab one chat at a time dynamically.
    batch_cids = all_cids[:PREVIEW_BATCH]
    batch_by_size = sorted(
        batch_cids, key=lambda c: info_map[c].get("raw_size", 0), reverse=True
    )
    pool_args = [(cid, TMP_DIR, DB_PATH) for cid in batch_by_size]
    stop_event = threading.Event()
    fill_cids = all_cids[PREVIEW_BATCH:]
    fill_thread = threading.Thread(
        target=_fill_remaining_previews,
        args=(fill_cids, TMP_DIR, DB_PATH, stop_event),
        daemon=True,
    )

    # fzf preview: try the cached file first (instant), fall back to live preview.py
    preview_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "preview.py"
    )
    preview_cmd = (
        f"cat {shlex.quote(TMP_DIR)}/ls_preview_{{1}}.txt 2>/dev/null"
        f" || {shlex.quote(sys.executable)} {shlex.quote(preview_script)} {{1}}"
    )

    proc = None
    try:
        with Spinner(f"precomputing {len(batch_cids)} previews"):
            with Pool(processes=min(len(batch_cids), os.cpu_count() or 4)) as pool:
                pool.map(compute_and_save_preview, pool_args, chunksize=1)
        fill_thread.start()
        proc = subprocess.run(
            [
                "fzf",
                "--prompt=> ",
                "--cycle",
                "--layout=reverse",
                "--no-separator",
                "--delimiter=\t",
                "--with-nth=2",
                "--nth=2",
                "--no-sort",
                "--bind=alt-j:preview-down,alt-k:preview-up,alt-d:preview-page-down,alt-u:preview-page-up",
                "--preview",
                preview_cmd,
                "--preview-window=right:60%:wrap:border-left",
            ],
            input="\n".join(lines),
            capture_output=True,
            text=True,
        )
    finally:
        stop_event.set()
        if fill_thread.is_alive():
            fill_thread.join(timeout=2.0)
        _cleanup_previews(TMP_DIR)

    if proc is None or proc.returncode != 0 or not proc.stdout.strip():
        return None, None
    cid = int(proc.stdout.strip().split("\t", 1)[0])
    return cid, info_map[cid]


def handle_ls(conn, show_archived, time_filter, reprint_prompt=True):
    """List all chats newest->oldest in fzf (short summary per line). Picking
    one opens a second fzf menu: open in ch, copy filename, or cancel."""
    rows = list_chats_by_recency(conn, show_archived, time_filter)
    cid, info = pick_latest_with_fzf(rows)
    if cid is None:
        return

    action = pick_ls_action()
    if action == "run":
        run_chat(cid, {cid: info}, reprint_prompt=reprint_prompt)
    elif action == "copy":
        copy_chat(cid, {cid: info})


# /purge: permanently delete all archived chats (source file gone). Destructive:
# drops paid summaries/embeddings, so it is gated behind an fzf confirmation.
# "No" is listed first and is the default on a bare Enter. The choice labels
# carry the count so the user sees the blast radius before confirming.
PURGE_NO = "No (keep archived chats)"
PURGE_YES_TMPL = "Yes, delete {count} archived chat(s)"


def handle_purge(conn):
    """Delete every chat flagged archived = 1 after an fzf confirmation. Returns
    the number deleted (0 if cancelled, declined, or none to delete). The caller
    must reload its in-memory ids/mat/meta and rebuild FTS afterward, since the
    row set and embeddings cache signature have changed."""
    count = conn.execute("SELECT count(*) FROM chats WHERE archived = 1").fetchone()[0]
    if count == 0:
        print(color.yellow("No archived chats to remove."))
        return 0
    if shutil.which("fzf") is None:
        print(color.red("fzf not found on PATH - install it to confirm /purge."))
        return 0

    yes_label = PURGE_YES_TMPL.format(count=count)
    choices = [PURGE_NO, yes_label]
    print(
        color.red(
            f"This permanently deletes {count} archived chat(s) and their cached summaries/embeddings."
        )
    )
    proc = subprocess.run(
        ["fzf", "--prompt=delete all archived? > ", "--cycle"],
        input="\n".join(choices),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or proc.stdout.strip() != yes_label:
        print(color.yellow("Purge cancelled."))
        return 0

    cur = conn.execute("DELETE FROM chats WHERE archived = 1")
    conn.commit()
    print(color.green(f"Deleted {cur.rowcount} archived chat(s)."))
    return cur.rowcount
