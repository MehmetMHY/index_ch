import os
import json
import shutil
import time
import subprocess

from config import TMP_DIR

from ..display import chat_epoch
from ..pickers import resolve_picks
from ..state import Session


def build_dump(cids, meta):
    """Merge the picked chats into a single ch-resumable dict. Chats are ordered
    oldest to newest (by chat_epoch); each chat's own messages stay together and
    in their original order, so the merged log reads as one continuous
    conversation rather than an interleave. Each message is tagged with
    source_file (its original ch_*.json name). Root platform/model/base_url are
    taken from the newest chat, since `ch -f` uses those root fields (not
    per-message ones) to restore the session it resumes into - dropping them
    entirely breaks `ch -f` with a "platform not found" error.

    Returns (merged, skipped) or None if every selected file failed to read.
    """
    cids = sorted(cids, key=lambda cid: chat_epoch(meta[cid]) or 0)

    messages = []
    source_files = []
    newest_raw = None
    skipped = []
    for cid in cids:
        path = meta[cid]["file_path"]
        name = os.path.basename(path)
        if meta[cid].get("archived"):
            skipped.append((name, "archived (source file gone)"))
            continue
        try:
            with open(path, "rb") as f:
                buf = f.read()
            raw = json.loads(buf)
            for msg in raw["messages"]:
                messages.append({**msg, "source_file": name})
            source_files.append(name)
            newest_raw = raw  # cids is oldest->newest, so the last one wins
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            skipped.append((name, exc))

    if not messages:
        print("Nothing dumped - all selected files failed to read.")
        report_skipped(skipped)
        return None

    merged = {
        "timestamp": newest_raw.get("timestamp"),
        "platform": newest_raw.get("platform"),
        "model": newest_raw.get("model"),
        "base_url": newest_raw.get("base_url"),
        "source_files": source_files,
        "messages": messages,
    }
    return merged, skipped


def report_skipped(skipped):
    if skipped:
        print(f"Skipped {len(skipped)} unreadable file(s):")
        for name, exc in skipped:
            print(f"  {name}: {exc}")


def unique_path(path):
    """Return path, or path with _<n> before the extension if it already exists,
    so a save never silently overwrites an existing dump."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while os.path.exists(f"{base}_{i}{ext}"):
        i += 1
    return f"{base}_{i}{ext}"


# what to do with a merged dump, in menu order
DUMP_ACTIONS = [
    ("Save to $HOME/Downloads/", "downloads"),
    ("Load into Ch (Temporary)", "load"),
    ("Load in Ch & save to Downloads", "load_keep"),
    ("Exit/Cancel", "cancel"),
]


def pick_dump_action():
    """fzf-pick what to do with the merged dump. Returns one of 'downloads',
    'load', 'load_keep', or 'cancel' (also 'cancel' if fzf is missing)."""
    if shutil.which("fzf") is None:
        print("fzf not found on PATH - install it to use /dump.")
        return "cancel"

    label_to_action = {label: action for label, action in DUMP_ACTIONS}
    proc = subprocess.run(
        ["fzf", "--prompt=dump> ", "--cycle"],
        input="\n".join(label for label, _ in DUMP_ACTIONS),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return "cancel"
    return label_to_action.get(proc.stdout.strip(), "cancel")


def save_dump_to_downloads(merged, filename, skipped):
    out_dir = os.path.expanduser("~/Downloads")
    os.makedirs(out_dir, exist_ok=True)
    out_path = unique_path(os.path.join(out_dir, filename))
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    n = len(merged["source_files"])
    print(f"Saved {n} chat(s) ({len(merged['messages'])} messages) to {out_path}.")
    report_skipped(skipped)


def load_dump_in_ch(merged, filename, keep, skipped):
    """Write the merged dump to cache/tmp, resume it in ch, then delete it once
    ch exits - or, when keep is True, move it to ~/Downloads instead. The
    try/finally guarantees the temp file is never orphaned, even on Ctrl-C."""
    if shutil.which("ch") is None:
        print("ch not found on PATH - https://github.com/MehmetMHY/ch")
        return

    tmp_path = os.path.join(TMP_DIR, filename)
    with open(tmp_path, "w") as f:
        json.dump(merged, f, indent=2)

    n = len(merged["source_files"])
    print(f"Opening merged dump of {n} chat(s) in ch...")
    result = None
    try:
        result = subprocess.run(["ch", "-f", tmp_path])
    finally:
        if not os.path.exists(tmp_path):
            pass  # ch moved/consumed it (unexpected) - nothing to clean up
        elif keep:
            out_dir = os.path.expanduser("~/Downloads")
            os.makedirs(out_dir, exist_ok=True)
            out_path = unique_path(os.path.join(out_dir, filename))
            shutil.move(tmp_path, out_path)
            print(f"Saved merged dump to {out_path}.")
        else:
            os.remove(tmp_path)
    if result is not None and result.returncode != 0:
        print(f"ch exited with status {result.returncode}.")
    print("Type a query or /help")
    report_skipped(skipped)


def handle_dump(session: Session, args):
    """Pick one or more results (by index list or fzf multi), merge them into a
    single ch-resumable log, then fzf-pick a destination: save to ~/Downloads,
    resume it in ch (temp file, deleted on exit), resume it and keep a copy in
    ~/Downloads, or cancel."""
    cids = resolve_picks(args, session.last_results, session.meta, "/dump")
    if not cids:
        return

    built = build_dump(cids, session.meta)
    if built is None:
        return
    merged, skipped = built

    action = pick_dump_action()
    if action == "cancel":
        return

    filename = f"index_ch_dump_{len(merged['source_files'])}_{int(time.time())}.json"
    if action == "downloads":
        save_dump_to_downloads(merged, filename, skipped)
    else:  # "load" or "load_keep"
        load_dump_in_ch(merged, filename, action == "load_keep", skipped)
