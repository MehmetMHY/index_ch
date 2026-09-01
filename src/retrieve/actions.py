import os
import time
import re
import json
import shlex
import shutil
import platform
import tempfile
import subprocess

from config import TMP_DIR
from build import format_messages

from .display import chat_preview
from . import color


def fetch_raw(conn, cid):
    row = conn.execute("SELECT raw FROM chats WHERE id = ?", (cid,)).fetchone()
    return json.loads(row[0]) if row else None


def build_content(conn, cid, meta):
    """Render a chat as summary + divider + raw (unfiltered) transcript text."""
    info = meta[cid]
    name = os.path.basename(info["file_path"])
    summary = (info["summary"] or "(no summary)").strip()

    raw = fetch_raw(conn, cid)
    raw_text = (
        format_messages(raw["messages"], skip_noise=False)
        if raw
        else "(raw content unavailable)"
    )

    divider = "\n" + "=" * 70 + "\n"
    return f"# {name}\n\n{summary}\n{divider}\n{raw_text}"


def view_chat(conn, cid, meta):
    """Write summary + raw chat text to a temp file, open it in an editor, then
    delete the temp file once the editor exits."""
    content = build_content(conn, cid, meta)
    fd, path = tempfile.mkstemp(prefix="chat_", suffix=".md", dir=TMP_DIR)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        editor_cmd = shlex.split(os.environ.get("EDITOR", "vim"))
        subprocess.run(editor_cmd + [path])
    finally:
        os.remove(path)


def copy_to_clipboard(text):
    """Copy text to the system clipboard. Returns True on success."""
    system = platform.system()
    if system == "Darwin":
        cmd = ["pbcopy"]
    elif system == "Windows":
        cmd = ["clip"]
    elif system == "Linux":
        if shutil.which("wl-copy"):
            cmd = ["wl-copy"]
        elif shutil.which("xclip"):
            cmd = ["xclip", "-selection", "clipboard"]
        elif shutil.which("xsel"):
            cmd = ["xsel", "--clipboard", "--input"]
        else:
            print(
                color.red(
                    "No clipboard tool found - install xclip, xsel, or wl-clipboard."
                )
            )
            return False
    else:
        print(color.red(f"Clipboard copy isn't supported on {system}."))
        return False

    try:
        subprocess.run(cmd, input=text, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(color.red(f"Clipboard copy failed: {exc}"))
        return False
    return True


def copy_chat(cid, meta):
    name = os.path.basename(meta[cid]["file_path"])
    if copy_to_clipboard(name):
        note = " (archived - source file gone)" if meta[cid].get("archived") else ""
        print(color.green(f"Copied {name} to clipboard{note}."))


def run_chat(cid, meta, reprint_prompt=True):
    """Hand the terminal over to `ch -f <name>` to resume the session in Ch."""
    if shutil.which("ch") is None:
        print(color.red("ch not found on PATH - https://github.com/MehmetMHY/ch"))
        return
    name = os.path.basename(meta[cid]["file_path"])
    if meta[cid].get("archived"):
        print(
            color.yellow(
                f"Note: {name} is archived (source file gone); ch -f may fail."
            )
        )
    print(color.cyan(f"Opening {name} in ch..."))
    result = subprocess.run(["ch", "-f", name])
    if result.returncode != 0:
        print(color.red(f"ch exited with status {result.returncode}."))
    if reprint_prompt:
        print(color.blue("Type a query or /help"))
