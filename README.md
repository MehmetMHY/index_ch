# index_ch

_July 19, 2026_

## About

A small add-on for [Ch](https://github.com/MehmetMHY/ch) that indexes your saved Ch chat history and lets you search it by meaning, not just keywords.

It reads the chat JSON files that Ch stores under `~/.ch/tmp/`, cleans them, summarizes and embeds each one with OpenAI, and stores everything in a local SQLite database. You can then run a hybrid search (vector + keyword) with LLM reranking to find the chats most relevant to a question.

## How it works

The project is three scripts, run in order:

1. `build.py` reads the chat JSON files, strips out auto-generated noise (code dumps, file pastes, command output), and stores the cleaned text in `chats.db`. It adds new files and re-ingests any whose contents changed (detected by a content hash), so a resumed chat picks up its new messages. Unchanged files are skipped, so re-running is cheap.

2. `process.py` summarizes each chat with `gpt-5.4-nano` and embeds the summary with `text-embedding-3-small`, saving both back to the database. It only processes chats that are not done yet, so it is resumable.

3. `retrieve.py` is an interactive search prompt. It rewrites your query into a few alternative phrasings (query expansion, to widen recall), embeds the original plus the variants in a single batched call with `text-embedding-3-small`, runs vector search and full-text keyword search on each, fuses all the results, reranks the top candidates, and shows the top 5 matches, each with a UTC timestamp from the chat's last message. The two LLM steps (expansion and rerank) run on Groq (`openai/gpt-oss-20b`) for speed; embeddings stay on OpenAI.

The database and a small embeddings cache are stored in a `cache/` directory next to the scripts, created automatically, so you can run them from any directory.

## Setup

Requires Python 3.11+, an [OpenAI API key](https://openai.com/api/) (embeddings and `process.py`), a [Groq API key](https://console.groq.com/keys) (`retrieve.py`'s rerank and query expansion), [fzf](https://github.com/junegunn/fzf) (used by `retrieve.py`'s `/view`, `/copy`, `/run`, and `/time` commands to pick a result), and [Ch](https://github.com/MehmetMHY/ch) itself on PATH (used by `/run` to resume a session).

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
python3 build.py
python3 process.py
python3 retrieve.py
```

Or run all three in order with a single command:

```bash
python3 run.py
```

This runs `build.py`, `process.py`, and `retrieve.py` in sequence, using the `env/` virtual environment if it exists, otherwise falling back to `python3`. It stops on the first failure.

You can also run each step individually.

Build and process the database:

```bash
python3 build.py
python3 process.py
```

Then search:

```bash
python3 retrieve.py
```

At the `query>` prompt:

- Type a question to get the top 5 matching chats with short summaries.
- Type `/view` or `/v` to fuzzy-pick one of the latest results with fzf and open its summary plus full raw transcript in `$EDITOR` (falls back to `vim`). Add a number to skip the picker, e.g. `/v 2`.
- Type `/copy` or `/c` to fuzzy-pick one of the latest results and copy its chat filename (`ch_session_<epoch>.json`) to the clipboard. Add a number to skip the picker, e.g. `/c 2`.
- Type `/run` or `/r` to fuzzy-pick one of the latest results and resume it in [Ch](https://github.com/MehmetMHY/ch) (`ch -f <file>`). Add a number to skip the picker, e.g. `/r 2`. Requires `ch` on PATH.
- Type `/time` or `/t` to fzf-pick a time window: a rolling window (past 1 day, 3 days, week, month, year), all time, or `Custom` to open an interactive UTC calendar and pick an exact start/end range. Set it directly with `/time 1d`, `/time 3d`, `/time 1w`, `/time 1m`, `/time 1y`, `/time custom` (opens the calendar), or `/time all` to clear it. The filter persists across queries (shown in the prompt as `query [1w]>` or `query [custom]>`) and scopes every search to chats within that window, based on the time of each chat's last message.
- Type `:fast` to toggle the LLM reranker off for quicker, keyword-and-vector-only results.
- Type `:expand` to toggle LLM query expansion off (skips the query-rewrite step; slightly faster and cheaper, but narrower recall).
- Type `/help` or `/h` to list all commands.
- Type `quit` to exit.

## Notes

- The path to the Ch chats (`~/.ch/tmp/`) is fixed and never modified.
- Each result shows a UTC timestamp (e.g. `Jul 27, 2025 09:45 UTC`) taken from the chat's last message, so resumed sessions sort and filter by when they were actually last used. It falls back to the epoch in the filename (`ch_session_<epoch>.json`) for chats with no messages.
- `process.py` runs many requests in parallel. Set the worker count with `WORKERS=128 python3 process.py`.
- If a chat fails to process, the error is recorded in the database and skipped on later runs. Retry those with `RETRY_ERRORS=1 python3 process.py`.
- Chats larger than the model input limit are summarized with a map-reduce pass (summarize each chunk, then summarize the summaries).
