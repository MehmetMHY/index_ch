import os
import re
import shutil
import subprocess

from .display import chat_epoch, format_list_timestamp, format_timestamp, chat_preview
from . import color

RETURN_YES = "yes"
RETURN_NO = "no"


def confirm_return(prompt="return to search? > "):
    """Ask whether to return to search/chats after running ch. Returns True for
    Yes (launch ch, then return), False for No (launch ch, then exit session),
    or None if cancelled (Esc / Ctrl-C, do not launch ch). No is the default
    (first in the list) so a bare Enter launches ch and exits immediately upon
    return. When fzf is missing, defaults to True."""
    if shutil.which("fzf") is None:
        return True
    proc = subprocess.run(
        ["fzf", f"--prompt={prompt}", "--cycle"],
        input="\n".join([RETURN_NO, RETURN_YES]),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return proc.stdout.strip() == RETURN_YES


def pick_with_fzf(last_results, meta, hint):
    """Fuzzy-pick one of the last results with fzf. Returns a chat id, or None
    if fzf is missing or the user cancelled (Esc / Ctrl-C / no match)."""
    if shutil.which("fzf") is None:
        print(
            color.red(f"fzf not found on PATH - install it, or use '{hint} <number>'.")
        )
        return None

    lines = []
    for i, (cid, grade) in enumerate(last_results, 1):
        info = meta[cid]
        name = os.path.basename(info["file_path"])
        ts = format_list_timestamp(chat_epoch(info))
        ts_tag = f" ({ts})" if ts else ""
        lines.append(f"[{i}] {name}{ts_tag} {chat_preview(info, 80)}")

    proc = subprocess.run(
        ["fzf", "--prompt=select> ", "--cycle"],
        input="\n".join(lines),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return None  # cancelled or no match

    idx = int(re.match(r"\[(\d+)\]", proc.stdout).group(1))
    return last_results[idx - 1][0]


def pick_many_with_fzf(last_results, meta, hint):
    """Fuzzy-pick one or more of the last results with fzf (multi-select).
    Returns a list of chat ids (empty if fzf is missing or the user cancelled)."""
    if shutil.which("fzf") is None:
        print(
            color.red(
                f"fzf not found on PATH - install it, or use '{hint} <number>...'."
            )
        )
        return []

    lines = []
    for i, (cid, grade) in enumerate(last_results, 1):
        info = meta[cid]
        name = os.path.basename(info["file_path"])
        ts = format_timestamp(chat_epoch(info))
        ts_tag = f" ({ts})" if ts else ""
        lines.append(f"[{i}] {name}{ts_tag} {chat_preview(info, 80)}")

    proc = subprocess.run(
        ["fzf", "-m", "--prompt=select> ", "--cycle"],
        input="\n".join(lines),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return []

    cids = []
    for line in proc.stdout.splitlines():
        m = re.match(r"\[(\d+)\]", line)
        if m:
            cids.append(last_results[int(m.group(1)) - 1][0])
    return cids


def resolve_pick(args, last_results, meta, hint):
    """Shared arg parsing for /view and /copy: an explicit index, or fzf."""
    if not last_results:
        print(color.yellow("No results yet - run a search first."))
        return None

    if args:
        try:
            idx = int(args[0])
        except ValueError:
            print(color.yellow(f"Usage: {hint} [1-{len(last_results)}]"))
            return None
        if not (1 <= idx <= len(last_results)):
            print(color.yellow(f"Choose a number between 1 and {len(last_results)}."))
            return None
        return last_results[idx - 1][0]

    return pick_with_fzf(last_results, meta, hint)


def resolve_picks(args, last_results, meta, hint):
    """Arg parsing for /dump: one or more explicit indices, or fzf multi-select.
    Returns a de-duplicated list of chat ids (order preserved), or [] on any
    usage error or cancel."""
    if not last_results:
        print(color.yellow("No results yet - run a search first."))
        return []

    if args:
        cids = []
        for a in args:
            try:
                idx = int(a)
            except ValueError:
                print(color.yellow(f"Usage: {hint} [1-{len(last_results)}]..."))
                return []
            if not (1 <= idx <= len(last_results)):
                print(
                    color.yellow(f"Choose numbers between 1 and {len(last_results)}.")
                )
                return []
            cids.append(last_results[idx - 1][0])
        seen, out = set(), []
        for cid in cids:
            if cid not in seen:
                seen.add(cid)
                out.append(cid)
        return out

    return pick_many_with_fzf(last_results, meta, hint)
