<div align="left">
  <img src="./docs/assets/icon.png" width="175">
</div>

<br>

# index_ch

## About

A small add-on for [Ch](https://github.com/MehmetMHY/ch) that indexes your saved Ch chat history and lets you search it by meaning, not just keywords.

It reads the chat JSON files that Ch stores under `~/.ch/tmp/`, cleans them, summarizes and embeds each one with OpenAI, and stores everything in a local SQLite database. You can then run a hybrid search (vector + keyword) with LLM reranking to find the chats most relevant to a question.

## Why

Your best AI answers are easy to lose. When chats pile up, exact keyword search is not enough. You remember the idea, the bug, or the decision, but not the wording. `index_ch` retrieves the right conversation anyway.

## Features

- **semantic search**: find conversations by intent and meaning, not just matching words
- **hybrid retrieval**: meaning-based search plus keyword matching for better recall
- **concise summaries**: each chat becomes a readable preview you can scan quickly
- **local cache**: your index and generated metadata are stored on your machine
- **chat browser**: browse recent sessions and jump back in when needed
- **terminal native**: a CLI with a fuzzy-picking TUI, built for keyboard-driven use

## How it works

The project is three scripts, run in order:

1. `build.py` reads the chat JSON files, strips out auto-generated noise (code dumps, file pastes, command output), and stores the cleaned text in `chats.db`. It adds new files and re-ingests any whose contents changed (detected by a content hash), so a resumed chat picks up its new messages. Unchanged files are skipped, so re-running is cheap. A chat whose source file is deleted from `~/.ch/tmp/` is never removed (you paid for its summary and embedding): it is flagged `archived` and kept searchable.

2. `process.py` summarizes each chat with `gpt-5.4-nano`, condenses that summary into a 1-2 sentence blurb (also `gpt-5.4-nano`) for the search results, and embeds the summary with `text-embedding-3-small`, saving all three back to the database. Each step is skipped when its column is already filled, so it is resumable and re-running never redoes work you already paid for.

3. `retrieve.py` is an interactive search prompt. It rewrites your query into a few alternative phrasings (query expansion, to widen recall), embeds the original plus the variants in a single batched call with `text-embedding-3-small`, runs vector search and full-text keyword search on each, fuses all the results, reranks the top candidates, and shows the top 5 matches, each with a UTC timestamp from the chat's last message. The two LLM steps run on Groq for speed (`openai/gpt-oss-20b` for expansion, `openai/gpt-oss-120b` for reranking); embeddings stay on OpenAI.

The Python scripts live in `src/`, with `run.py` at the repo root as a convenience entrypoint. The database and a small embeddings cache are stored in `~/.ch/index/`, next to Ch's own local data.

## Setup

Requires Python 3.11+, an [OpenAI API key](https://openai.com/api/) (embeddings and `process.py`), a [Groq API key](https://console.groq.com/keys) (`retrieve.py`'s rerank and query expansion), [fzf](https://github.com/junegunn/fzf) (used by `retrieve.py`'s `/view`, `/copy`, `/run`, `/dump`, `/time`, and `/ls` commands), and [Ch](https://github.com/MehmetMHY/ch) itself on PATH (used by `/run` and `/ls` to resume a session).

Create a virtual environment and install dependencies:

```bash
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
export OPENAI_API_KEY="your-openai-key-here"
export GROQ_API_KEY="your-groq-key-here"
```

## Usage

Build and process the database, then search:

```bash
python3 src/build.py
python3 src/process.py
python3 src/retrieve.py
```

Or use the convenience entrypoint:

```bash
python3 run.py
```

This fzf-picks what to run: `Browse Chats` (launches a one-shot `/ls` fzf browser of chats newest->oldest with a right-side preview), `Smart Search`, `Update Cache` (`build.py` + `process.py`), or `Exit Session`. It uses the `env/` virtual environment if it exists, otherwise falls back to `python3`, and stops on the first failure. If `fzf` isn't installed, it skips the menu and runs the full update + retrieve pipeline.

You can also run each step individually.

Build and process the database:

```bash
python3 src/build.py
python3 src/process.py
```

Then search:

```bash
python3 src/retrieve.py
```

At the `>` prompt:

- Type a question to get the top 5 matching chats with short summaries.
- Type `/view` or `/v` to fuzzy-pick one of the latest results with fzf and open its summary plus full raw transcript in `$EDITOR` (falls back to `vim`). Add a number to skip the picker, e.g. `/v 2`.
- Type `/copy` or `/c` to fuzzy-pick one of the latest results and copy its chat filename (`ch_session_<epoch>.json`) to the clipboard. Add a number to skip the picker, e.g. `/c 2`.
- Type `/run` or `/r` to fuzzy-pick one of the latest results and resume it in [Ch](https://github.com/MehmetMHY/ch) (`ch -f <file>`). Add a number to skip the picker, e.g. `/r 2`. Requires `ch` on PATH.
- Type `/dump` or `/d` to fuzzy-pick (multi-select) one or more of the latest results and merge their messages into a single ch-resumable chat log. Chats are ordered oldest to newest, each chat's messages stay together in order, and every message is tagged with which original file it came from. Pass numbers to skip the picker, e.g. `/dump 1 3 5`. After selecting, a second fzf menu asks what to do with the merged log: `Save to $HOME/Downloads/` saves it to `~/Downloads/index_ch_dump_<chat_count>_<epoch>.json`; `Load into Ch (Temporary)` resumes it in [Ch](https://github.com/MehmetMHY/ch) from a temp file that is deleted when you exit; `Load in Ch & save to Downloads` resumes it and then moves the file to `~/Downloads`; `Exit/Cancel` does nothing. Unreadable files are skipped with a warning.
- Type `/ls` to list all chats newest to oldest in fzf, one line per chat (`[08/10/26•01:55Z] short summary`) with a right-side preview. The 500 most recent previews are precomputed in parallel before fzf opens so the top is instant to browse, and the rest fill in the background. Press Enter to pick a chat, then choose: open it in [Ch](https://github.com/MehmetMHY/ch) (`ch -f <file>`), copy its filename to the clipboard, or cancel. Respects the active `:archived` toggle and `/time` filter. Requires `fzf` and `ch` on PATH.
- Type `/time` or `/t` to fzf-pick a time window: a rolling window (past 1 day, 3 days, week, month, year), all time, or `Custom` to open an interactive UTC calendar and pick an exact start/end range. Set it directly with `/time 1d`, `/time 3d`, `/time 1w`, `/time 1m`, `/time 1y`, `/time custom` (opens the calendar), or `/time all` to clear it. The filter persists across queries (shown in the prompt as `[1w]>` or `[custom]>`) and scopes every search to chats within that window, based on the time of each chat's last message.
- Type `/len` or `/l` to show the current result count, or `/len <n>` (e.g. `/len 10`) to set how many results are shown per search, from 1 to 25 (default 5). Anything else prints a usage message and leaves the count unchanged.
- Type `:fast` to toggle the LLM reranker off for quicker, keyword-and-vector-only results.
- Type `:expand` to toggle LLM query expansion off (skips the query-rewrite step; slightly faster and cheaper, but narrower recall).
- Type `:archived` to toggle showing chats whose source file is gone from `~/.ch/tmp/` (kept and flagged `archived` by `build.py`). They are hidden by default; toggle this on to search and view chats you paid to embed even after Ch drops the source file. `/run` and `/dump` warn or skip them since the file is gone, but `/view` still works (the transcript is cached in the DB).
- Type `/purge` to permanently delete all archived chats at once (fzf-confirm; the only path that drops paid rows). Useful once you are sure you no longer want the dead-source chats around.
- Type `/help` or `/h` to list all commands.
- Type `quit` to exit.

## Website

A small project website lives in `./docs/` and gives a lighter, more polished introduction to `index_ch`. It is a good place to start if you want the high-level idea, the main features, and a quick overview of how the project works without reading the full CLI docs first.

Run it locally with the included dev server (opens your browser, `Ctrl+C` to stop):

```bash
python3 docs/run.py
```

`docs/run.py` is a standalone HTTP server (no dependencies) that serves `docs/index.html` and its `docs/assets/`. It is unrelated to the root `run.py`, which is the project's CLI entrypoint.

## Notes

- The path to the Ch chats (`~/.ch/tmp/`) is fixed and never modified.
- `run.py` and the `src/` scripts require both `~/.ch/` and `~/.ch/tmp/` to already exist. Run Ch first if either directory is missing.
- Each result shows a UTC timestamp (e.g. `Jul 27, 2025 09:45 UTC`) taken from the chat's last message, so resumed sessions sort and filter by when they were actually last used. It falls back to the epoch in the filename (`ch_session_<epoch>.json`) for chats with no messages.
- `process.py` runs many requests in parallel. Set the worker count with `WORKERS=128 python3 src/process.py`.
- If a chat fails to process, the error is recorded in the database and skipped on later runs. Retry those with `RETRY_ERRORS=1 python3 src/process.py`.
- Chats larger than the model input limit are summarized with a map-reduce pass (summarize each chunk, then summarize the summaries).

## License

[MIT](./LICENSE) - Copyright (c) 2026 Mehmet Yilmaz.
