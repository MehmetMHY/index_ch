# index_ch

_July 19, 2026_

## About

A small add-on for [Ch](https://github.com/MehmetMHY/ch) that indexes your saved Ch chat history and lets you search it by meaning, not just keywords.

It reads the chat JSON files that Ch stores under `~/.ch/tmp/`, cleans them, summarizes and embeds each one with OpenAI, and stores everything in a local SQLite database. You can then run a hybrid search (vector + keyword) with LLM reranking to find the chats most relevant to a question.

## How it works

The project is three scripts, run in order:

1. `build.py` reads the chat JSON files, strips out auto-generated noise (code dumps, file pastes, command output), and stores the cleaned text in `chats.db`. It only adds new files, so re-running is cheap.

2. `process.py` summarizes each chat with `gpt-5.4-nano` and embeds the summary with `text-embedding-3-small`, saving both back to the database. It only processes chats that are not done yet, so it is resumable.

3. `retrieve.py` is an interactive search prompt. It embeds your query, runs vector search and full-text keyword search, fuses the results, reranks the top candidates with `gpt-5.6-luna`, and shows the top 5 matches, each with a UTC timestamp parsed from the chat's filename.

The database and a small embeddings cache are stored in a `cache/` directory next to the scripts, created automatically, so you can run them from any directory.

## Setup

Requires Python 3.11+, an [OpenAI API key](https://openai.com/api/), [fzf](https://github.com/junegunn/fzf) (used by `retrieve.py`'s `/view`, `/copy`, and `/run` commands to pick a result), and [Ch](https://github.com/MehmetMHY/ch) itself on PATH (used by `/run` to resume a session).

Create a virtual environment and install dependencies:

```bash
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
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
- Type `:fast` to toggle the LLM reranker off for quicker, keyword-and-vector-only results.
- Type `/help` or `/h` to list all commands.
- Type `quit` to exit.

## Notes

- The path to the Ch chats (`~/.ch/tmp/`) is fixed and never modified.
- Each result shows a UTC timestamp (e.g. `Jul 27, 2025 09:45 UTC`) parsed from the epoch embedded in the chat's filename (`ch_session_<epoch>.json`).
- `process.py` runs many requests in parallel. Set the worker count with `WORKERS=128 python3 process.py`.
- If a chat fails to process, the error is recorded in the database and skipped on later runs. Retry those with `RETRY_ERRORS=1 python3 process.py`.
- Chats larger than the model input limit are summarized with a map-reduce pass (summarize each chunk, then summarize the summaries).
