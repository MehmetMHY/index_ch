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
  tunables. Change models or prices here, nowhere else. Two providers: OpenAI
  for embeddings and `process.py` (summaries), Groq for `retrieve.py`'s rerank
  and query expansion (reached via its OpenAI-compatible endpoint,
  `GROQ_BASE_URL`). Embeddings must stay on OpenAI: the stored vectors are
  `text-embedding-3-small`, and the query has to embed in the same space, so
  the embedding model/provider cannot change without a full re-embed. Groq has
  no embedding model anyway. Any replacement rerank/expansion model must support
  `json_schema` structured outputs (`chat.completions.parse`): Groq's
  `qwen/qwen3.6-27b` was evaluated and rejected because it 400s on `json_schema`,
  and as a reasoning model it burned ~5x the output tokens at ~13x the cost.
- `build.py` reads chat JSON from `~/.ch/tmp/`, strips auto-generated noise
  (code dumps, file pastes, command output), and stores cleaned text in the
  database. Incremental via a per-file `content_hash` (SHA-256 of the bytes):
  new files are inserted, files whose hash changed are re-ingested in place
  (clearing summary/embedding so process.py redoes just them), unchanged files
  are skipped.
- `process.py` summarizes each chat (`gpt-5.4-nano`), condenses that summary
  into the 1-2 sentence `short_summary` shown in search results (same model,
  fed the summary not the raw chat, so backfilling it is ~10x cheaper), and
  embeds the summary (`text-embedding-3-small`). Resumable: `pending_rows`
  selects rows missing any of the three, and `process_row` skips each step whose
  column is already filled, so adding a column never re-summarizes or re-embeds
  what is already paid for. `save` only writes the embedding when that run
  computed one, so a short-summary-only pass cannot clobber existing vectors.
- `retrieve.py` is an interactive prompt. Per query it optionally expands the
  query into a few variants (Groq `openai/gpt-oss-20b`, structured output),
  embeds the original plus variants in one batched call (OpenAI
  `text-embedding-3-small`), runs vector search and FTS5 keyword search per
  query, fuses every ranking with Reciprocal Rank Fusion, reranks the top
  candidates with Groq `openai/gpt-oss-120b` (listwise, structured outputs)
  against the ORIGINAL query, and prints the top 5, each with a UTC timestamp
  from the chat's last message (`chat_epoch`). Within an equal rerank grade,
  results are ordered most-recent-first (recency tiebreak). On startup a daemon
  thread (`warm_connections`) opens the OpenAI + Groq HTTPS connections so the
  first query does not pay cold-start latency. `:fast` toggles the
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
- Schema changes are done as additive, idempotent migrations that read only
  existing data, never via a full rebuild: `process.py`'s `migrate` adds
  summary/short_summary/embedding/error; `build.py`'s `backfill_message_epochs` adds and
  populates `last_message_epoch` from the stored `raw`, and
  `backfill_content_hashes` adds `content_hash` and blesses each row with its
  current on-disk hash. This is deliberate: the summaries/embeddings cost ~$6-7
  to regenerate, so new columns must be backfillable in place without any OpenAI
  calls (the content-hash backfill in particular must not make everything look
  "changed", which would re-embed the whole DB).
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
export OPENAI_API_KEY="..."        # embeddings (process.py + retrieve.py)
export GROQ_API_KEY="..."          # retrieve.py rerank + query expansion
python3 build.py                   # ingest new chats
python3 process.py                 # summarize + embed (calls OpenAI, costs money)
python3 retrieve.py                # interactive search
```

`retrieve.py` reaches Groq with the same `openai` SDK, just a second client
pointed at `GROQ_BASE_URL` (see `groq_client`). If `GROQ_API_KEY` is unset,
rerank and expansion fail and degrade gracefully (hybrid order / original query
only) rather than erroring, but retrieval quality drops, so treat it as
required.

Or run all three in order with `python3 run.py`.

Useful env vars for `process.py`:

- `WORKERS=128 python3 process.py` sets parallel worker count (default 64).
- `RETRY_ERRORS=1 python3 process.py` re-attempts chats parked in the `error`
  column.

There is no test suite. Verify changes by compiling (`python3 -m py_compile`)
and by testing on a copy of the database, not the real one, when a change could
mutate or corrupt data.

## Cost awareness

`process.py` makes paid OpenAI calls; `retrieve.py` makes a paid OpenAI
embedding call plus paid Groq rerank/expansion calls. A full `process.py`
backfill of ~5600 chats costs roughly $6-7; each `retrieve.py` query costs a
fraction of a cent (a few tenths, mostly the Groq rerank). Do not trigger a full
re-process or bulk API calls without a clear reason. When testing API-touching
code, prefer a single call or a small sample, and do not run the full pipeline
unprompted.

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
- A chat's displayed timestamp and `/time` filtering both use `chat_epoch(info)`
  in `retrieve.py`: the `last_message_epoch` column (the last message's own
  `time` field), falling back to the filename epoch (`filename_epoch`, regex
  `ch_session_<epoch>`) only when it is NULL. Neither uses the DB `created_at`
  (ingest time) nor file mtime. The filename epoch alone is wrong for resumed/
  continued sessions: it is frozen when the file is first saved, so a chat last
  used yesterday can look days old. `build.py` computes `last_message_epoch` at
  ingest (`message_epoch`); `backfill_message_epochs` populates it for existing
  DBs from the stored `raw` JSON (one-time, idempotent, no API calls) and is
  called at the start of both `build.py` and `retrieve.py`.
- Resumed chats stay current via the `content_hash` diff in `build.py`: when a
  session gains messages after ingest, its file hash changes, so the next
  `build.py` re-ingests it (`update_entries`) and clears summary/embedding so
  `process.py` re-embeds it. So `run.py` self-heals drift, but only on the next
  build (the DB is a snapshot as of the last run, not live). `backfill_content_hashes`
  blessed the existing rows with their current on-disk hash (one-time, no
  re-processing), so chats that had already drifted before this feature are not
  retroactively refreshed until they change again on disk.
- Because Ch may be writing a session file while `build.py` reads it,
  `load_and_clean` reads the bytes once, hashes and parses that same buffer
  (so the stored `content_hash` always matches the stored content), and returns
  `None` on any read/parse failure instead of raising. The `build.py` main loop
  skips those Nones and files that error while hashing, so one mid-write or
  deleted file cannot crash the whole build; they are re-ingested on a later
  build once fully written. Keep this fail-soft behavior.
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
- Result blurbs come from `chat_preview`, which prefers the stored
  `short_summary` and falls back to `preview()` (first paragraph of the long
  summary) when it is NULL, so output stays sane mid-backfill. `truncate` cuts
  on a sentence boundary when one falls late enough in the window, else on a
  word boundary with trailing connector words (`DANGLING_WORDS`) dropped, so a
  blurb never trails off as "... best next bet with a...". `short_summary` is
  display-only: embeddings and FTS still index the long `summary`, so changing
  it costs nothing in retrieval quality.
- `build.py`'s `format_messages(messages, skip_noise=...)` is shared: `skip_noise=True`
  is the `cleaned` text `load_and_clean` stores, `skip_noise=False` is what
  `retrieve.py`'s `/view` shows as the raw transcript. Keep noise-filtering
  logic in this one function rather than duplicating it in `retrieve.py`.
