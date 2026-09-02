#!/usr/bin/env python3
"""Unified Search & Browse Explorer - MVP Demo

A self-contained prototype with hardcoded mock data demonstrating the unified
index_ch experience: browse chats, smart search, side-by-side transcript preview,
multi-select merge, and direct key actions. No database, no API calls, no
cascading sub-menus.

Usage:
    python3 demo_split_search.py
"""

import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import time

# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
BLUE = "\033[34m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
GRAY = "\033[90m"
WHITE = "\033[97m"
RED = "\033[31m"

PREVIEW_WIDTH = 72
CODE_WIDTH = 82

SEARCH_HISTORY = []

MOCK_CHATS = {
    1: {
        "filename": "ch_session_1740001234.json",
        "title": "Fix JWT token expiration & refresh race condition",
        "short_summary": "Resolved 401 token refresh loops by adding an in-flight refresh promise mutex in Axios interceptor.",
        "epoch": 1740001234,
        "date_str": "02/19/2025 18:47Z",
        "model": "gpt-4o",
        "messages": [
            {
                "user": "Our frontend is getting caught in an infinite refresh loop when multiple API requests 401 simultaneously.",
                "bot": "The issue happens because multiple parallel requests each trigger their own `/refresh` call simultaneously when the access token expires.\n\nTo fix this, maintain a single shared `isRefreshing` promise mutex across all interceptors:\n\n```typescript\nlet refreshPromise: Promise<string> | null = null;\n\napi.interceptors.response.use(\n  res => res,\n  async err => {\n    if (err.response?.status === 401) {\n      if (!refreshPromise) {\n        refreshPromise = authService.refreshToken().finally(() => {\n          refreshPromise = null;\n        });\n      }\n      const newToken = await refreshPromise;\n      err.config.headers['Authorization'] = `Bearer ${newToken}`;\n      return api(err.config);\n    }\n    return Promise.reject(err);\n  }\n);\n```",
            },
            {
                "user": "What happens if the refresh token itself has expired?",
                "bot": "If the refresh token itself 401s, clear the stored tokens, reject the `refreshPromise`, and redirect the user to `/login`.\n\nMake sure to reject any pending queued requests so they don't hang indefinitely.",
            },
        ],
    },
    2: {
        "filename": "ch_session_1740114500.json",
        "title": "Implement SQLite FTS5 Hybrid Search with Reciprocal Rank Fusion",
        "short_summary": "Combined BM25 keyword rankings with dense vector cosine similarity using RRF formula (1 / (60 + rank)).",
        "epoch": 1740114500,
        "date_str": "02/21/2025 02:15Z",
        "model": "gpt-4o-mini",
        "messages": [
            {
                "user": "How do we combine FTS5 BM25 search rankings with vector cosine similarity scores fairly?",
                "bot": "Reciprocal Rank Fusion (RRF) is the best standard approach because raw BM25 scores and vector dot products are on completely different scales.\n\nFormula:\n`RRF_score(d) = sum( 1.0 / (k + rank_i(d)) )` where `k = 60` is the standard smoothing constant.\n\n```python\ndef reciprocal_rank_fusion(vector_ranked_ids, keyword_ranked_ids, k=60):\n    scores = {}\n    for rank, doc_id in enumerate(vector_ranked_ids, start=1):\n        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))\n    for rank, doc_id in enumerate(keyword_ranked_ids, start=1):\n        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))\n    return sorted(scores.items(), key=lambda x: x[1], reverse=True)\n```",
            },
            {
                "user": "Does FTS5 support trigram tokenizers for substring matching?",
                "bot": "Yes! SQLite FTS5 has a built-in `trigram` tokenizer (`tokenize='trigram'`) available starting in SQLite 3.34.0. It allows efficient `LIKE '%substring%'` equivalent queries.",
            },
        ],
    },
    3: {
        "filename": "ch_session_1740228000.json",
        "title": "Optimize React render performance & useMemo dependencies",
        "short_summary": "Profiled unnecessary child re-renders using React DevTools; wrapped data transformations in useMemo and callbacks in useCallback.",
        "epoch": 1740228000,
        "date_str": "02/22/2025 09:40Z",
        "model": "claude-3-7-sonnet",
        "messages": [
            {
                "user": "My heavy TableComponent is re-rendering on every keystroke in an unrelated search input field.",
                "bot": "The parent component is recreating objects/callbacks on every render, causing the memoized Table to fail shallow equality checks.\n\n1. Wrap `TableComponent` in `React.memo`\n2. Memoize sorting/filtering with `useMemo`\n3. Memoize row click handlers with `useCallback`",
            },
            {
                "user": "Is it worth memoizing primitive props like numbers or strings?",
                "bot": "No. Primitives compare by value in JavaScript (`1 === 1` is always true). Memoization has overhead and should only be used for object references, arrays, and functions.",
            },
        ],
    },
    4: {
        "filename": "ch_session_1740331000.json",
        "title": "Debug Docker compose bridge network DNS resolution issue",
        "short_summary": "Fixed container communication failures by defining a shared user-defined bridge network in docker-compose.yml.",
        "epoch": 1740331000,
        "date_str": "02/23/2025 14:20Z",
        "model": "gpt-4o",
        "messages": [
            {
                "user": "Container 'api' cannot connect to 'redis:6379' inside docker compose. Error: `Name or service not known`.",
                "bot": "In Docker Compose, service discovery via container name only works when containers share a user-defined network.\n\nCheck your `docker-compose.yml`:\n```yaml\nservices:\n  api:\n    build: .\n    networks:\n      - backend\n  redis:\n    image: redis:7-alpine\n    networks:\n      - backend\n\nnetworks:\n  backend:\n    driver: bridge\n```",
            }
        ],
    },
    5: {
        "filename": "ch_session_1740449000.json",
        "title": "Design REST API schema for multi-tenant organization permissions",
        "short_summary": "Architected RBAC schema with Organization -> Role -> Permission mapping and JWT claims.",
        "epoch": 1740449000,
        "date_str": "02/24/2025 23:10Z",
        "model": "gpt-4o",
        "messages": [
            {
                "user": "How should we structure multi-tenant RBAC permissions in PostgreSQL?",
                "bot": "Use a 5-table schema:\n1. `organizations`\n2. `users`\n3. `roles` (scoped to `org_id` or system default)\n4. `permissions` (e.g. `billing:read`, `project:write`)\n5. `memberships` (links user, org, and role)",
            }
        ],
    },
    6: {
        "filename": "ch_session_1740550000.json",
        "title": "Set up GitHub Actions CI with Go test and lint pipeline",
        "short_summary": "Configured go test, gofmt-check, gosec, and gitleaks in a single workflow with caching.",
        "epoch": 1740550000,
        "date_str": "02/25/2025 11:30Z",
        "model": "gpt-4o",
        "messages": [
            {
                "user": "How do I set up a Go CI pipeline with test, lint, and security checks?",
                "bot": "Use a single workflow with parallel jobs:\n\n```yaml\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n      - uses: actions/setup-go@v5\n        with:\n          go-version: '1.26'\n      - run: go test -count=1 ./...\n  lint:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n      - run: gofmt -l . | tee /dev/stderr | ! read\n```",
            }
        ],
    },
    7: {
        "filename": "ch_session_1740660000.json",
        "title": "Python multiprocessing Pool vs ThreadPool for I/O-bound tasks",
        "short_summary": "ThreadPool is better for API calls (no pickle overhead), Pool for CPU-bound work.",
        "epoch": 1740660000,
        "date_str": "02/26/2025 18:00Z",
        "model": "gpt-4o-mini",
        "messages": [
            {
                "user": "Should I use multiprocessing.Pool or ThreadPool for concurrent API calls?",
                "bot": "ThreadPool is the right choice for I/O-bound work like API calls. Processes (Pool) are for CPU-bound work where the GIL is the bottleneck.\n\nThreadPool avoids pickle overhead, shares memory (so large objects like connection pools are shared), and has lower spawn latency. The GIL does not matter for I/O-bound work because the thread releases it during blocking syscalls.",
            },
            {
                "user": "What about the max workers count?",
                "bot": "For I/O-bound work, a good rule of thumb is `min(32, os.cpu_count() + 4)`. For pure API calls with high latency, you can go higher (64-128) since most of the time is spent waiting for the network, not executing Python.",
            },
        ],
    },
    8: {
        "filename": "ch_session_1740770000.json",
        "title": "Implement fzf custom keybindings with --expect and --bind",
        "short_summary": "Used --expect for action keys and --bind for scroll/abort, parsing stdout line-by-line.",
        "epoch": 1740770000,
        "date_str": "02/27/2025 08:15Z",
        "model": "gpt-4o",
        "messages": [
            {
                "user": "How do I make fzf trigger different actions based on which key the user presses?",
                "bot": "Use the `--expect` flag. Pass a comma-separated list of keys:\n```\nfzf --expect=ctrl-v,ctrl-y,ctrl-s\n```\n\nWhen the user presses one of those keys, fzf outputs the key name on the first line of stdout, followed by the selected item(s) on subsequent lines. If the user presses Enter (the default), the first line is empty.",
            }
        ],
    },
}


def render_preview(cid):
    """Render the full preview pane: multi-line legend, metadata, transcript."""
    chat = MOCK_CHATS.get(cid)
    if not chat:
        return "Chat not found."

    turns = len(chat["messages"])
    turn_label = f"{turns} turn" if turns == 1 else f"{turns} turns"

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
    lines.append(f"{BOLD}{chat['title']}{RESET}")
    lines.append("")
    lines.append(f"{DIM}Date:{RESET}  {chat['date_str']}")
    lines.append(f"{DIM}Model:{RESET} {chat['model']}")
    lines.append(f"{DIM}Turns:{RESET} {turn_label}")
    lines.append(f"{DIM}File:{RESET}  {chat['filename']}")
    lines.append("")
    lines.extend(wrap_block(f"Summary: {chat['short_summary']}", indent=""))
    lines.append("")

    for idx, msg in enumerate(chat["messages"], 1):
        lines.append(f"{BOLD}{BLUE}-- Turn {idx} --{RESET}")
        lines.append(f"{BOLD}{WHITE}User:{RESET}")
        lines.extend(wrap_block(msg["user"]))
        lines.append("")
        lines.append(f"{BOLD}{GREEN}Assistant:{RESET}")
        in_code = False
        for bot_line in msg["bot"].splitlines():
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


def rank_chats(query):
    """Keyword-based ranking for the demo. Returns list of cids best-to-worst."""
    terms = [t.lower() for t in query.split() if t]
    scored = []
    for cid, chat in MOCK_CHATS.items():
        text = (
            f"{chat['title']} {chat['short_summary']} "
            f"{json.dumps(chat['messages'])}"
        ).lower()
        match_count = sum(1 for term in terms if term in text)
        score = (match_count * 10) + (10 - cid)
        scored.append((cid, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in scored]


def browse_chats():
    """Return all chat ids sorted newest-to-oldest (browse mode)."""
    return sorted(MOCK_CHATS.keys(), key=lambda c: MOCK_CHATS[c]["epoch"], reverse=True)


def merge_chats(cids):
    """Merge multiple chats into a single ch-resumable JSON structure.
    Oldest to newest, each message tagged with source_file."""
    sorted_cids = sorted(cids, key=lambda c: MOCK_CHATS[c]["epoch"])
    messages = []
    source_files = []
    for cid in sorted_cids:
        chat = MOCK_CHATS[cid]
        fname = chat["filename"]
        source_files.append(fname)
        for msg in chat["messages"]:
            entry = dict(msg)
            entry["source_file"] = fname
            messages.append(entry)
    newest = MOCK_CHATS[sorted_cids[-1]]
    return {
        "platform": "openai",
        "model": newest["model"],
        "timestamp": newest["epoch"],
        "source_files": source_files,
        "messages": messages,
    }


def unique_path(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while os.path.exists(f"{base}_{i}{ext}"):
        i += 1
    return f"{base}_{i}{ext}"


def copy_to_clipboard(text):
    if shutil.which("pbcopy"):
        subprocess.run(["pbcopy"], input=text, text=True)
        return True
    return False


def open_explorer(cids, is_search=False):
    """Launch the unified fzf split-view explorer.

    cids: list of chat ids to show (ranked for search, recency-sorted for browse).
    is_search: whether this was triggered by a search query (affects prompt text).
    Returns None (always returns to the prompt)."""
    if shutil.which("fzf") is None:
        print(f"{RED}fzf not found on PATH - install it to use this tool.{RESET}")
        return

    if not cids:
        print(f"{DIM}No chats to show.{RESET}")
        return

    temp_dir = tempfile.mkdtemp(prefix="index_ch_explorer_")
    preview_script = os.path.join(temp_dir, "preview.sh")

    with open(preview_script, "w") as f:
        f.write("#!/bin/sh\n")
        f.write(f'cat "{temp_dir}/preview_$1.txt" 2>/dev/null || echo "No preview"\n')
    os.chmod(preview_script, 0o755)

    lines = []
    for cid in cids:
        chat = MOCK_CHATS[cid]
        preview_text = render_preview(cid)
        with open(os.path.join(temp_dir, f"preview_{cid}.txt"), "w") as pf:
            pf.write(preview_text)
        lines.append(f"{cid}\t{chat['date_str']}  {chat['filename']}")

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

    try:
        proc = subprocess.run(
            cmd, input="\n".join(lines), capture_output=True, text=True
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    if proc.returncode != 0 or not proc.stdout.strip():
        return

    output = proc.stdout.strip().splitlines()
    if len(output) < 2:
        return

    key_pressed = output[0]
    selected_lines = output[1:]

    selected_cids = []
    for line in selected_lines:
        if "\t" in line:
            cid = int(line.split("\t", 1)[0])
            selected_cids.append(cid)

    if not selected_cids:
        return

    is_multi = len(selected_cids) > 1

    if key_pressed == "ctrl-v":
        action_view_editor(selected_cids[0])
    elif key_pressed == "ctrl-y":
        action_copy_filenames(selected_cids)
    elif key_pressed == "ctrl-s":
        action_save_downloads(selected_cids, is_multi)
    else:
        action_open_in_ch(selected_cids, is_multi)


def action_open_in_ch(cids, is_multi):
    """Enter: open single chat in ch, or merge multiple and open the merge."""
    if is_multi:
        merged = merge_chats(cids)
        names = [
            MOCK_CHATS[c]["filename"]
            for c in sorted(cids, key=lambda c: MOCK_CHATS[c]["epoch"])
        ]
        print(f"{CYAN}[simulated]{RESET} Merged {len(cids)} chats into one session:")
        for n in names:
            print(f"  {DIM}{n}{RESET}")
        print(
            f"  {DIM}-> ch -f <temp_path> ({len(merged['messages'])} messages, oldest to newest){RESET}"
        )
    else:
        chat = MOCK_CHATS[cids[0]]
        print(f"{CYAN}[simulated]{RESET} ch -f {chat['filename']}")


def action_view_editor(cid):
    """Ctrl-V: open the full transcript in $EDITOR, delete temp file on exit."""
    chat = MOCK_CHATS[cid]
    print(f"{GREEN}Opening {chat['filename']} in $EDITOR...{RESET}")
    fd, path = tempfile.mkstemp(prefix="chat_", suffix=".md")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(render_preview(cid))
        editor = os.environ.get("EDITOR", "vim")
        subprocess.run([editor, path])
    finally:
        if os.path.exists(path):
            os.remove(path)


def action_copy_filenames(cids):
    """Ctrl-Y: copy filename(s) to clipboard."""
    if len(cids) == 1:
        name = MOCK_CHATS[cids[0]]["filename"]
    else:
        name = "\n".join(MOCK_CHATS[c]["filename"] for c in cids)
    if copy_to_clipboard(name):
        label = "filename" if len(cids) == 1 else f"{len(cids)} filenames"
        print(f"{YELLOW}Copied {label} to clipboard.{RESET}")
    else:
        print(f"{YELLOW}{name}{RESET}")


def action_save_downloads(cids, is_multi):
    """Ctrl-S: save single chat file or merged dump to ~/Downloads."""
    out_dir = os.path.expanduser("~/Downloads")
    os.makedirs(out_dir, exist_ok=True)

    if is_multi:
        merged = merge_chats(cids)
        filename = f"index_ch_dump_{len(cids)}_{int(time.time())}.json"
        out_path = unique_path(os.path.join(out_dir, filename))
        with open(out_path, "w") as f:
            json.dump(merged, f, indent=4)
        print(
            f"{BLUE}Saved {len(cids)} merged chats "
            f"({len(merged['messages'])} messages) to {out_path}{RESET}"
        )
    else:
        chat = MOCK_CHATS[cids[0]]
        filename = chat["filename"]
        out_path = unique_path(os.path.join(out_dir, filename))
        demo_content = json.dumps(
            {
                "platform": "openai",
                "model": chat["model"],
                "messages": chat["messages"],
            },
            indent=4,
        )
        with open(out_path, "w") as f:
            f.write(demo_content)
        print(f"{BLUE}Saved {filename} to {out_path}{RESET}")


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


def print_banner():
    print(
        f"{BLUE}Type to search {BOLD}or{RESET}{BLUE} {BOLD}ENTER{RESET}{BLUE} to browse{RESET}"
    )
    print(
        f"{YELLOW}/history{RESET}, {YELLOW}/help{RESET}, {BOLD}{MAGENTA}or{RESET} {YELLOW}/exit{RESET}"
    )


def prompt_update_cache():
    """Ask the user whether to update the cache before entering the REPL.
    Uses fzf with No (default) / Yes. Returns True for Yes, False otherwise."""
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


def simulate_update():
    """Pretend to run build.py + process.py."""
    print(f"{DIM}Scanning Ch exports...{RESET}")
    time.sleep(0.3)
    print(f"{DIM}Processing pending chats...{RESET}")
    time.sleep(0.3)


def main():
    if prompt_update_cache():
        simulate_update()
    print_banner()
    while True:
        try:
            query = input(f"{BOLD}{BLUE}> {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not query:
            cids = browse_chats()
            open_explorer(cids, is_search=False)
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
                cids = rank_chats(picked)
                open_explorer(cids, is_search=True)
            continue

        ts = time.strftime("%m/%d/%Y %H:%MZ")
        SEARCH_HISTORY.append((query, ts))
        cids = rank_chats(query)
        open_explorer(cids, is_search=True)


if __name__ == "__main__":
    main()
