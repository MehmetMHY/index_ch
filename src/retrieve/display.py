import os
import re
import time
from datetime import datetime, timezone

from config import (
    PREVIEW_CHARS,
    EMBEDDING_MODEL,
    QUERY_EXPANSION_MODEL,
    RERANK_MODEL,
)
from pricing import estimate_cost

from .state import Session


def preview(summary, limit=PREVIEW_CHARS):
    """First real paragraph of a markdown summary, trimmed to ~limit chars.

    Skips leading headers / horizontal rules, joins the first paragraph, strips
    bold/code markers (but keeps underscores, which appear in searchable tokens),
    and cuts on a word boundary with an ellipsis.
    """
    text = (summary or "").strip()
    if not text:
        return ""

    para, started = [], False
    for line in text.splitlines():
        s = line.strip()
        if not started:
            # skip blank lines, markdown headers, and horizontal rules
            if not s or s.startswith("#") or set(s) <= set("-=*_ "):
                continue
            started = True
            para.append(s)
        elif s:
            para.append(s)
        else:
            break  # blank line ends the first paragraph

    paragraph = " ".join(para) if para else " ".join(text.split())
    paragraph = paragraph.replace("**", "").replace("`", "").strip()

    if len(paragraph) <= limit:
        return paragraph
    return paragraph[:limit].rsplit(" ", 1)[0].rstrip() + "..."


# words that read as unfinished when a truncated blurb ends on them
DANGLING_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "with",
    "of",
    "in",
    "to",
    "for",
    "on",
    "at",
    "by",
    "from",
    "as",
    "that",
    "which",
    "while",
    "plus",
    "into",
    "via",
    "is",
    "was",
    "were",
    "are",
    "be",
    "its",
    "their",
    "his",
    "her",
    "then",
}


def truncate(text, limit):
    """Trim text to limit chars, ending on a sentence boundary when there is a
    reasonable one, else on a word boundary with trailing connector words
    dropped. Always ends with an ellipsis, so a blurb never stops mid-phrase
    like 'one final best next bet with a...'.
    """
    if len(text) <= limit:
        return text
    window = text[:limit]
    # prefer the last sentence end, but only if it keeps most of the window;
    # otherwise a blurb opening with a short sentence would be cut to nothing
    cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if cut >= limit * 0.6:
        return window[:cut].rstrip(" .!?") + "..."

    words = window.split(" ")[:-1]  # drop the word the limit cut in half
    while words and words[-1].strip(",;:()").lower() in DANGLING_WORDS:
        words.pop()
    return " ".join(words).rstrip(" ,;:(") + "..."


def chat_preview(info, limit=PREVIEW_CHARS):
    """Blurb shown for a result: the stored 1-2 sentence short_summary when it
    exists, else the first paragraph of the long summary. The fallback keeps
    output sane for rows process.py has not backfilled yet."""
    short = " ".join((info.get("short_summary") or "").split())
    if not short:
        return preview(info["summary"], limit)
    return truncate(short, limit)


def filename_epoch(name):
    """Return the epoch embedded in a ch_session_<epoch>.json filename, or None."""
    match = re.search(r"(\d{9,})", os.path.basename(name))
    return int(match.group(1)) if match else None


def chat_epoch(info):
    """Best "last active" epoch for a chat: the last message's time (stored in
    last_message_epoch), falling back to the filename epoch when that is missing
    (e.g. an empty chat). The last-message time is what makes resumed/continued
    sessions sort and filter by when they were actually last used."""
    return info.get("last_message_epoch") or filename_epoch(info["file_path"])


def format_timestamp(epoch):
    """Format an epoch as a 24-hour UTC timestamp, e.g. 'Jul 27, 2025 14:45 UTC';
    empty string if the epoch is missing or invalid."""
    if not epoch:
        return ""
    try:
        dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    except (ValueError, OSError, OverflowError):
        return ""
    return dt.strftime("%b %d, %Y %H:%M UTC")


def format_list_timestamp(epoch):
    """Timestamp format for /ls: 'MM/DD/YY.HH:MMZ' using 24-hour UTC time."""
    if not epoch:
        return ""
    try:
        dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    except (ValueError, OSError, OverflowError):
        return ""
    return dt.strftime("%m/%d/%y\u2022%H:%MZ")


# /time filter labels and descriptions (used by cli.py prompt and format_help)
TIME_LABELS = {
    "1d": "past 1 day",
    "3d": "past 3 days",
    "1w": "past 1 week",
    "1m": "past 1 month",
    "1y": "past 1 year",
}


def time_filter_label(tf):
    """Short tag for the prompt indicator: a key, 'custom', or None."""
    if not tf:
        return None
    return "custom" if isinstance(tf, tuple) else tf


def time_filter_desc(tf):
    """Human-readable description for the 'filter set' confirmation."""
    if not tf:
        return "all time"
    if isinstance(tf, tuple):
        fmt = lambda e: datetime.fromtimestamp(e, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
        return f"{fmt(tf[0])} to {fmt(tf[1])}"
    return TIME_LABELS[tf]


def print_results(results, meta, elapsed, usage):
    for i, (cid, grade) in enumerate(results, 1):
        info = meta[cid]
        name = os.path.basename(info["file_path"])
        ts = format_timestamp(chat_epoch(info))
        ts_tag = f" \u00b7 {ts}" if ts else ""
        arch_tag = " \u00b7 archived" if info.get("archived") else ""
        tag = f" \u00b7 relevance {grade}/3" if grade is not None else ""
        print(f"{i}. {name}{ts_tag}{arch_tag}{tag}")
        print(f"   {chat_preview(info)}\n")
    if not results:
        print("No matches.\n")

    costs = [
        estimate_cost(EMBEDDING_MODEL, usage["embed_in"]),
        estimate_cost(QUERY_EXPANSION_MODEL, usage["expand_in"], usage["expand_out"]),
        estimate_cost(RERANK_MODEL, usage["rerank_in"], usage["rerank_out"]),
    ]
    cost = sum(costs) if all(c is not None for c in costs) else None
    cost_str = "?" if cost is None else f"{cost:.6f}"
    print(f"[{len(results)} results in {elapsed:.2f}s | ~${cost_str}]")


HELP_TEXT = """\033[4mStatus\033[0m
rerank: {rerank}
expansion: {expand}
archived: {archived}
time: {time_filter}
results: {result_len}
\033[4mOptions\033[0m
<query>        search your chats
/view, /v      fuzzy-pick a result, open it in $EDITOR
/view <n>      open result n directly
/copy, /c      pick a result and copy it to clipboard
/copy <n>      copy result n directly
/run, /r       fuzzy-pick a result, resume it in ch (ch -f <file>)
/run <n>       resume result n directly
/dump, /d      pick result(s) and merge them into one file
/dump <n> ...  dump result n (and more) directly
/ls            browse all chats newest->oldest in fzf, pick one to open/copy
/time, /t      pick a time window to scope searches to
/time <win>    set it directly: 1d, 3d, 1w, 1m, 1y, all, or custom
/len, /l       show the current result count
/len <n>       set how many results to show (1-25)
:fast          toggle the LLM reranker on/off
:expand        toggle LLM query expansion on/off
:archived      toggle showing archived chats (source file gone) on/off
/purge         permanently delete all archived chats (fzf-confirm)
/help, /h      show this list
quit, exit, :q exit"""


def format_help(session: Session) -> str:
    return HELP_TEXT.format(
        rerank="on" if session.do_rerank else "off",
        expand="on" if session.do_expand else "off",
        archived="shown" if session.show_archived else "hidden",
        time_filter=time_filter_desc(session.time_filter),
        result_len=session.result_len,
    )
