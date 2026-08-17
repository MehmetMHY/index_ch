# AGENTS.md

Guidance for AI agents working on this project.

## What this is

A small add-on for [Ch](https://github.com/MehmetMHY/ch) that indexes a user's
saved Ch chat history and makes it searchable by meaning. It reads Ch's chat
JSON files, cleans them, summarizes and embeds each one with OpenAI, stores
everything in a local SQLite database, and offers a hybrid (vector + keyword)
search with LLM reranking.

## Architecture

All Python modules live in `src/` (`build.py`, `process.py`, `retrieve.py`,
`config.py`, `input_time.py`, `preview.py`); only `run.py` sits at the repo
root, as the entrypoint. The imports between the modules are flat (`from config import ...`,
`from build import ...`), which works because they are run as scripts (Python
puts the script's own dir on `sys.path`), not as an installed package. Do not
add an `__init__.py` or convert them to a package, or those imports break. The
generated `cache/` dir lives at `src/cache/` because `config.py` anchors it to
its own `__file__`; moving `config.py` moves the cache with it.

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
  are skipped. A chat whose source file vanished from `~/.ch/tmp/` is never
  deleted (its summary/embedding/raw are cached and were paid for): it is
  flagged `archived = 1` and bumped `updated_at` so retrieve.py's caches
  rebuild. A file that reappears (archived=1 but back on disk) is un-archived,
  and if its hash also changed it is re-ingested via the changed path (which
  clears the flag). Only rows not already archived are flipped, so a steady-
  state build does not bump `updated_at` and needlessly invalidate caches.
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
  reranker, `:expand` toggles query expansion, and `:archived` toggles whether
  chats whose source file is gone from `~/.ch/tmp/` (flagged `archived = 1` by
  `build.py`) show in search results. Archived chats are hidden by default (the
  flag is in-memory per session, like `:fast`); toggling it on surfaces them so
  you can still search/view chats you paid to embed even after Ch drops the
  source file. They stay in the FTS index and embeddings cache at all times, so
  the toggle is instant (no rebuild): `allowed_in_range` masks both the vector
  and keyword legs, alongside the `/time` filter. `/time`/`/t` sets a persistent
  time filter (rolling `1d`/`3d`/`1w`/`1m`/`1y`, `all` to clear, or `custom` which
  opens the `input_time.py` curses calendar for an absolute start/end range;
  fzf picker or a direct token) that scopes every subsequent search. `/len`/`/l`
  sets how many results are shown per search (`1`-`25`, default `5`, from
  `config.TOP_K`); with no argument it prints the current count, and anything
  non-numeric or out of range prints a usage message and leaves it unchanged.
  When raised above `RERANK_POOL`, `search` reranks `max(RERANK_POOL, top_k)`
  candidates instead of the fixed pool, so results beyond the default pool
  size still come back graded rather than falling back to hybrid order.
  It also has `/view`/`/v`, `/copy`/`/c`, `/run`/`/r`, and `/dump`/`/d` (fzf-pick
  one of the last results, or pass a number to skip the picker; `/dump` accepts
  multiple numbers like `/dump 1 3 5` and uses fzf's `-m` multi-select mode) and
  `/help`/`/h`. `/ls` lists every chat newest->oldest in fzf (one line each:
  `[MM/DD/YY•HH:MMZ] short_summary`, falling back to the filename when no
  `short_summary` is stored yet), filtered by the active `:archived` toggle and
  `/time` filter (same `chat_epoch`/`range_bounds` logic as search, applied at
  list-build time via `list_chats_by_recency`). It shows a right-side preview
  via the lightweight `preview.py` helper (no vector/API imports), and pressing
  Enter opens a second fzf menu (`pick_ls_action`, `LS_ACTIONS`) to open the
  selected chat in Ch (`ch -f <file>`) or copy its filename to the clipboard.
  Previews are precomputed: the `PREVIEW_BATCH` (500) most recent chats are
  rendered in parallel (`multiprocessing.Pool`) before fzf opens so the top of
  the list is instant to browse, and a daemon thread fills the remaining
  previews sequentially while fzf is open. Each preview is written atomically
  to `TMP_DIR/ls_preview_<id>.txt` (write to `.tmp`, rename) so fzf's `cat`
  never reads a partial file; the fzf preview command tries the cached file
  first and falls back to live `preview.py` if not yet computed. All temp files
  are cleaned up in a `try/finally` when fzf exits (even on Ctrl-C). The
  `format_messages_limited` early-exit formatter stops once the transcript
  exceeds `PREVIEW_LIMIT` chars, so long chats render as fast as short ones. The
  preview header shows the number of turns (prompt & response pairs) in that
  specific chat, not a global chat count. It queries all DB rows directly rather than the embeddings-loaded `meta`, so
  unprocessed chats appear too. It requires `fzf` and `ch` on PATH.
  `/view` writes the summary plus the full raw (unfiltered)
  transcript to a temp file under `src/cache/tmp/`, opens it in `$EDITOR` (falls
  back to `vim`), and deletes the file the moment the editor exits; any
  in-editor edits/saves are never persisted anywhere. `/copy` copies just the
  chat's filename (`ch_session_<epoch>.json`) to the clipboard (`pbcopy`/`clip`
  /`wl-copy`/`xclip`/`xsel` depending on OS). `/run` shells out to
  `ch -f <file>` (Ch's own `-f`/`--fetch` flag accepts a bare filename) to
  resume that session inside Ch, handing over the terminal until Ch exits.
  `/dump` merges the picked chats' messages into a single ch-resumable log:
  chats are ordered oldest to newest (by `chat_epoch`), each chat's own
  messages stay together and in order (no interleaving), and every message is
  tagged with `source_file` (its original `ch_*.json` name). Root `platform`/
  `model`/`base_url`/`timestamp` are taken from the newest chat and
  `source_files` lists all merged filenames in order. `ch -f` reads those root
  fields (not per-message ones) to restore the session it resumes, and dropping
  them breaks it with a "platform not found" error; extra keys (root or
  per-message) are safely ignored by `ch`'s permissive JSON unmarshal. After
  selecting, a second fzf menu (`pick_dump_action`, `DUMP_ACTIONS`) chooses the
  destination: `Save to $HOME/Downloads/` writes
  `~/Downloads/index_ch_dump_<chat_count>_<epoch>.json`; `Load into Ch
(Temporary)` writes the log to `src/cache/tmp/` and resumes it via
  `ch -f <full path>` (Ch's `-f` accepts an absolute path, not just a bare
  filename), deleting the temp file when Ch exits; `Load in Ch & save to
Downloads` does the same but moves the temp file to `~/Downloads` on exit
  instead of deleting it; `Exit/Cancel` does nothing. The temp
  write/resume/cleanup runs in a `try/finally` so the `src/cache/tmp/` file is
  never orphaned, even on Ctrl-C. `unique_path` appends `_<n>` before the
  extension if a `~/Downloads` target name already exists, so a save never
  silently overwrites. The destination menu requires `fzf` (prints an error and
  aborts the dump if missing, even when the chats were picked by number), and
  the two "Load" options require `ch` on PATH. Skips unreadable files with a
  printed warning. The chat picker itself requires `fzf` (degrades to a message
  telling the user to pass a number if it's missing); `/run` requires `ch` on
  PATH.

`build.py` owns `get_connection` and the base table; `process.py` and
`retrieve.py` import it. `retrieve.py` owns its own FTS5 index and embeddings
cache and rebuilds them automatically when the data changes.

`input_time.py` is a standalone curses UTC calendar range picker
(`pick_time_range() -> (start, end) | None`) that `retrieve.py` imports for the
`/time custom` absolute range. It has no project dependencies of its own.

`preview.py` is the lightweight fzf preview helper for `/ls`. It imports only
`build` (for `format_messages` and `is_auto_entry`) and `config` (for `DB_PATH`
and `PREVIEW_LIMIT`), never `retrieve.py` or numpy/openai/httpx/pydantic. This is
deliberate: `retrieve.py`'s `multiprocessing.Pool` targets
`compute_and_save_preview` in `preview.py`, so each spawned worker only pays the
light import cost (~50ms), not the full retrieve.py import stack (~2-3s). Its
`format_messages_limited` early-exit formatter stops once the transcript exceeds
`PREVIEW_LIMIT` chars, so long chats render as fast as short ones. The preview
header shows the number of turns (prompt & response pairs) in that specific
chat, not a global chat count. It has a standalone CLI (`python3
src/preview.py <id>`) used as the fzf fallback when a cached preview file does
not exist yet.

`run.py` (at the repo root) is a convenience wrapper around the `src/` scripts.
`pick_action` fzf-picks one of `Browse Chats` (`retrieve.py ls`, a one-shot
`/ls` startup mode that exits after the fzf list instead of loading search
vectors or warming API connections), `Smart Search` (`retrieve.py` only),
`Update Cache` (`build.py` + `process.py`), or `Exit Session` (does nothing);
cancelling the picker (Esc/Ctrl-C) also does nothing.
There is no flag-based bypass - unlike `retrieve.py`'s pickers, which degrade
to "pass a number" when `fzf` is missing, `run.py` has no non-interactive
alternative to fall back to, so it degrades by running the full pipeline
(build + process + retrieve) instead, with a printed note - this keeps headless
callers (e.g. cron) working the way bare `python3 run.py` always did, before
this picker existed. It uses `env/bin/python3` when a virtual environment
exists (`env/` stays at the root), otherwise falls back to `python3`, and
exits non-zero on the first script failure.

`docs/` is the static project website (`index.html` plus `assets/`), unrelated
to the Python pipeline. It is served by `docs/run.py`, a zero-dependency
`http.server`-based dev server that localizes a canonical `<link rel=canonical>`
origin so the page renders correctly offline, binds to `127.0.0.1` on the
first free port in `8000`-`8099`, opens the browser, and stops on `Ctrl+C` /
`Ctrl+D` / SIGTERM / SIGHUP. Do not confuse it with the root `run.py` (the CLI
entrypoint). The website text is intentionally high-level (no model names, no
schema details, no command flags); the root `README.md` and this file remain
the source of truth for the CLI. `index.html` includes an animated demo
terminal that types out a sample `retrieve.py` session when scrolled into
view (respects `prefers-reduced-motion`, replays on scroll-in past a 3-minute
cooldown); the demo box has a fixed height so it never shifts the page as it
fills, and its horizontal scroll is hidden but available via touch-swipe on
mobile and click-drag on desktop (pointer-capture, 6px threshold so clicks and
text selection are not disturbed).

## Data and storage

- All generated data lives in `src/cache/` (the database, the `.npz` embeddings
  cache, SQLite journal/WAL sidecars, `src/cache/tmp/` scratch files for
  `retrieve.py`'s `/view`, and `src/cache/tmp/ls_preview_*.txt` files for
  `/ls`'s precomputed fzf previews). It is created automatically by `config.py`
  and is gitignored (the `cache/` pattern is un-anchored, so it matches at
  `src/cache/` too). Never commit it.
- The database is derived data. `build.py` rebuilds the cleaned text; re-running
  `process.py` re-fills summaries/embeddings but costs money (see below). Deleting
  `src/cache/chats.db` means a full rebuild and re-processing.
- Schema changes are done as additive, idempotent migrations that read only
  existing data, never via a full rebuild: `process.py`'s `migrate` adds
  summary/short_summary/embedding/error; `build.py`'s `backfill_message_epochs` adds and
  populates `last_message_epoch` from the stored `raw`, and
  `backfill_content_hashes` adds `content_hash` and blesses each row with its
  current on-disk hash, and `backfill_archived` adds the `archived` flag
  (DEFAULT 0, no data scan). This is deliberate: the summaries/embeddings cost
  ~$6-7 to regenerate, so new columns must be backfillable in place without any
  OpenAI calls (the content-hash backfill in particular must not make everything
  look "changed", which would re-embed the whole DB).
- The chat source path `~/.ch/tmp/` is owned by Ch and is read-only for us. Do
  not modify it or write to it.

## Setup and commands

`retrieve.py`'s `/view`, `/copy`, `/run`, `/dump`, and `/time` pickers need
`fzf` on PATH (not a pip package; install separately, e.g. `brew install fzf`).
All fzf calls pass `--cycle` so the list wraps top-to-bottom and back.

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt   # httpx, numpy, openai, pydantic
export OPENAI_API_KEY="..."        # embeddings (process.py + retrieve.py)
export GROQ_API_KEY="..."          # retrieve.py rerank + query expansion
python3 src/build.py               # ingest new chats
python3 src/process.py             # summarize + embed (calls OpenAI, costs money)
python3 src/retrieve.py            # interactive search
```

`retrieve.py` reaches Groq with the same `openai` SDK, just a second client
pointed at `GROQ_BASE_URL` (see `groq_client`). If `GROQ_API_KEY` is unset,
rerank and expansion fail and degrade gracefully (hybrid order / original query
only) rather than erroring, but retrieval quality drops, so treat it as
required.

Or use the fzf-driven entrypoint, `python3 run.py` (see the `run.py` section
above).

Useful env vars for `process.py`:

- `WORKERS=128 python3 src/process.py` sets parallel worker count (default 64).
- `RETRY_ERRORS=1 python3 src/process.py` re-attempts chats parked in the `error`
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
- No em dashes in the README or AGENTS.md.
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
  The same search-level masking applies to the `:archived` toggle:
  `allowed_in_range` drops archived rows from both legs when hidden, so toggling
  `:archived` on never starves a query the way trimming final results would.
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
- A chat flagged `archived = 1` (source file gone from `~/.ch/tmp/`) is kept
  forever: its `summary`, `short_summary`, `embedding`, and `raw` are all
  cached in the DB, so search and `/view` keep working for it with no disk
  access. `/copy` still copies the (now-dead) filename with a note, `/run`
  warns that `ch -f` may fail before handing off, and `/dump` skips it with an
  "archived (source file gone)" message instead of a generic read error. The
  flag is additive (`backfill_archived`, DEFAULT 0, no data scan) so existing
  paid rows are untouched; only `build.py`'s disk reconciliation ever sets it,
  and only for rows not already flagged (so a steady-state build does not bump
  `updated_at` and needlessly invalidate `retrieve.py`'s caches). `/purge` is
  the only path that drops archived rows: it `DELETE`s every `archived = 1`
  chat after an fzf confirmation ("No" first so a bare Enter is safe; both
  choice labels carry the row count so the user sees the blast radius), then the caller reloads `ids/mat/meta` via `load_vectors` and
  rebuilds FTS via `ensure_fts` in place, since the row set and the embeddings
  cache signature (count:latest) both changed. This is the one destructive op
  in the project - it discards paid summaries/embeddings, so keep the
  confirmation gate.
