# AGENTS.md

Guidance for AI agents working on this project.

## What this is

A small add-on for [Ch](https://github.com/MehmetMHY/ch) that indexes a user's
saved Ch chat history and makes it searchable by meaning. It reads Ch's chat
JSON files, cleans them, summarizes and embeds each one with OpenAI, stores
everything in a local SQLite database, and offers a hybrid (vector + keyword)
search with LLM reranking.

## Architecture

Three scripts, run in order, plus a shared config:

- `config.py` is the single source of truth for paths, models, pricing, and all
  tunables. Change models or prices here, nowhere else.
- `build.py` reads chat JSON from `~/.ch/tmp/`, strips auto-generated noise
  (code dumps, file pastes, command output), and stores cleaned text in the
  database. Incremental: only adds files not already present.
- `process.py` summarizes each chat (`gpt-5.4-nano`) and embeds the summary
  (`text-embedding-3-small`), saving both back. Resumable: only touches rows
  missing a summary or embedding.
- `retrieve.py` is an interactive prompt. Per query it optionally expands the
  query into a few variants (`gpt-5.4-nano`, structured output, reasoning off),
  embeds the original plus variants in one batched call, runs vector search and
  FTS5 keyword search per query, fuses every ranking with Reciprocal Rank
  Fusion, reranks the top candidates with `gpt-5.6-luna` (listwise, structured
  outputs) against the ORIGINAL query, and prints the top 5, each with a UTC
  timestamp parsed from the epoch in the chat's filename. `:fast` toggles the
  reranker, `:expand` toggles query expansion, and `/time`/`/t` sets a persistent
  time filter (rolling `1d`/`3d`/`1w`/`1m`/`1y`, `all` to clear, or `custom` which
  opens the `input_time.py` curses calendar for an absolute start/end range;
  fzf picker or a direct token) that scopes every subsequent search.
  It also has `/view`/`/v`, `/copy`/`/c`, and `/run`/`/r` (fzf-pick one of the
  last results, or pass a number to skip the picker) and `/help`/`/h`. `/view` writes the summary plus the full raw (unfiltered)
  transcript to a temp file under `cache/tmp/`, opens it in `$EDITOR` (falls
  back to `vim`), and deletes the file the moment the editor exits — any
  in-editor edits/saves are never persisted anywhere. `/copy` copies just the
  chat's filename (`ch_session_<epoch>.json`) to the clipboard (`pbcopy`/`clip`
  /`wl-copy`/`xclip`/`xsel` depending on OS). `/run` shells out to
  `ch -f <file>` (Ch's own `-f`/`--fetch` flag accepts a bare filename) to
  resume that session inside Ch, handing over the terminal until Ch exits.
  Requires `fzf` on PATH for the picker (degrades to a message telling the
  user to pass a number if it's missing) and `ch` on PATH for `/run`.

`build.py` owns `get_connection` and the base table; `process.py` and
`retrieve.py` import it. `retrieve.py` owns its own FTS5 index and embeddings
cache and rebuilds them automatically when the data changes.

`input_time.py` is a standalone curses UTC calendar range picker
(`pick_time_range() -> (start, end) | None`) that `retrieve.py` imports for the
`/time custom` absolute range. It has no project dependencies of its own.

`run.py` is a convenience wrapper that runs all three scripts in order. It uses
`env/bin/python3` when a virtual environment exists, otherwise falls back to
`python3`. It exits non-zero on the first script failure.

## Data and storage

- All generated data lives in `cache/` (the database, the `.npz` embeddings
  cache, SQLite journal/WAL sidecars, and `cache/tmp/` scratch files for
  `retrieve.py`'s `/view`). It is created automatically by `config.py` and is
  gitignored. Never commit it.
- The database is derived data. `build.py` rebuilds the cleaned text; re-running
  `process.py` re-fills summaries/embeddings but costs money (see below). Deleting
  `cache/chats.db` means a full rebuild and re-processing.
- The chat source path `~/.ch/tmp/` is owned by Ch and is read-only for us. Do
  not modify it or write to it.

## Setup and commands

`retrieve.py`'s `/view`, `/copy`, `/run`, and `/time` pickers need `fzf` on
PATH (not a pip package; install separately, e.g. `brew install fzf`). Both
fzf calls pass `--cycle` so the list wraps top-to-bottom and back.

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt   # httpx, numpy, openai, pydantic
export OPENAI_API_KEY="..."        # required by process.py and retrieve.py
python3 build.py                   # ingest new chats
python3 process.py                 # summarize + embed (calls OpenAI, costs money)
python3 retrieve.py                # interactive search
```

Or run all three in order with `python3 run.py`.

Useful env vars for `process.py`:

- `WORKERS=128 python3 process.py` sets parallel worker count (default 64).
- `RETRY_ERRORS=1 python3 process.py` re-attempts chats parked in the `error`
  column.

There is no test suite. Verify changes by compiling (`python3 -m py_compile`)
and by testing on a copy of the database, not the real one, when a change could
mutate or corrupt data.

## Cost awareness

`process.py` and `retrieve.py` make paid OpenAI calls. A full `process.py`
backfill of ~5600 chats costs roughly $6-7; each `retrieve.py` query costs a
fraction of a cent. Do not trigger a full re-process or bulk API calls without a
clear reason. When testing API-touching code, prefer a single call or a small
sample, and do not run the full pipeline unprompted.

## Conventions

- Keep comments short and purposeful. Prefer a brief section label or a one-line
  note explaining _why_, not line-by-line narration. `config.py` in particular
  is intentionally terse.
- No em dashes in the README.
- Pricing is keyed by model in `config.PRICING` as `(input, output)` per 1M
  tokens, with a `estimate_cost` helper. Keep price and model together.
- Paths are always derived from `__file__` via `config.py` so scripts work from
  any directory. Do not hardcode absolute paths or reintroduce per-script path
  constants.
- Match the existing style of the file you are editing (naming, spacing, idiom).

## Gotchas

- The chat cleaner (`is_auto_entry` in `build.py`) drops several noise markers,
  including Ch's `[code-dump starts]`. If cleaning changes, the stored `cleaned`
  text is stale until the database is rebuilt.
- The reranker treats document text as untrusted (summaries can contain
  instructions). Keep the prompt-injection guards in `RERANK_SYSTEM`, and keep
  the defensive validation that drops hallucinated IDs and appends dropped
  candidates in hybrid order.
- Query expansion (`expand_query`) is a recall aid, not correctness-critical:
  it degrades to `([], 0, 0)` on any failure so search falls back to the
  original query alone. Keep that graceful fallback. The reranker is always
  fed the ORIGINAL query, never a variant, so user intent stays intact.
  `embed_queries` batches the original query and all variants into one
  embeddings request (order restored via `.index`), so expansion adds a single
  nano LLM call but no extra embedding round trips.
- `retrieve.py`'s FTS index and embeddings cache invalidate on a signature of
  row count plus latest `updated_at`. If you change how rows are updated, make
  sure that signature still changes so the caches rebuild.
- `process.py` shuts its thread pool down manually (not via `with`) so Ctrl-C
  exits promptly. Keep that pattern if you touch the concurrency code.
- `retrieve.py`'s `format_timestamp` and the `/time` filter both derive a chat's
  time from the epoch in its filename (`chat_epoch`, regex `ch_session_<epoch>`),
  never from the DB `created_at` (which is ingest time). If Ch's filename format
  changes, timestamps vanish from output and time-filtered chats are excluded
  (no parseable epoch) rather than erroring.
- The `/time` filter is applied at the SEARCH level, not by trimming the final
  results: `allowed_in_range` masks the embeddings matrix so the vector leg
  returns the true top-`POOL` in-range, and `fts_search` fetches a wider window
  then filters to the in-range set. This keeps narrow ranges (e.g. `1d`) from
  coming up empty because their best chats were not in the global top-`POOL`.
- The `time_filter` value has three shapes: `None` (all time), a `TIME_RANGES`
  key (rolling window ending now, recomputed live each query; `1m`/`1y` are
  approximate 30d/365d), or a `(start_epoch, end_epoch)` tuple (absolute custom
  range from the calendar picker). `range_bounds` normalizes the latter two to
  `(lo, hi)`; keep the tuple-vs-str discriminator if you touch this.
- `build.py`'s `format_messages(messages, skip_noise=...)` is shared: `skip_noise=True`
  is the `cleaned` text `load_and_clean` stores, `skip_noise=False` is what
  `retrieve.py`'s `/view` shows as the raw transcript. Keep noise-filtering
  logic in this one function rather than duplicating it in `retrieve.py`.
