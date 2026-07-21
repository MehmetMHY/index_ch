import os
import json
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from openai import OpenAI

# reuse the shared db helpers from the build script; all tunables live in config
from build import get_connection
from config import (
    SUMMARY_MODEL,
    EMBEDDING_MODEL,
    DEFAULT_WORKERS,
    COMMIT_EVERY,
    PRINT_EVERY,
    MAX_INPUT_CHARS,
    estimate_cost,
)

# these calls are network bound (waiting on OpenAI), not cpu bound, so threads
# run truly in parallel here (the GIL is released during I/O) and are faster and
# lighter than multiprocessing for this workload. the real ceiling is OpenAI's
# rate limits, not local CPU. at Tier 5 there is huge headroom: the binding limit
# is text-embedding-3-small at 10,000 RPM (~166 req/s), with gpt-5.4-nano at
# 30,000 RPM — so per-request latency, not the API, is what caps throughput.
# override with e.g. WORKERS=128 python process.py; the client retries on 429.
MAX_WORKERS = int(os.environ.get("WORKERS", DEFAULT_WORKERS))

# set RETRY_ERRORS=1 to re-attempt chats that previously failed (error column set)
RETRY_ERRORS = os.environ.get("RETRY_ERRORS", "").lower() in ("1", "true", "yes")

SUMMARY_PROMPT = (
    "You are summarizing a conversation between a user and an AI assistant so it "
    "can be found later with semantic search. Write a concise summary (a few "
    "sentences) capturing the main topics, questions asked, and any conclusions "
    "or answers reached. Focus on concrete, searchable specifics. Do not add "
    "commentary or a preamble, just the summary."
)

# used by the map-reduce path for chats too big for a single request
CHUNK_PROMPT = (
    "You are summarizing ONE part of a longer conversation between a user and an "
    "AI assistant. Concisely summarize the topics, questions, and answers in this "
    "part. It is only a fragment, so do not worry about overall conclusions. Do "
    "not add commentary or a preamble, just the summary."
)

COMBINE_PROMPT = (
    "The following are summaries of consecutive parts of a single conversation "
    "between a user and an AI assistant. Combine them into one concise summary "
    "capturing the main topics, questions asked, and any conclusions reached, so "
    "it can be found later with semantic search. Do not add commentary or a "
    "preamble, just the summary."
)

# one-line blurb shown in retrieve.py's results, condensed from the long summary
# (not the raw chat) so the backfill is cheap. leads with the topic because the
# long summaries tend to open with boilerplate like "The user asked ...".
SHORT_SUMMARY_PROMPT = (
    "Condense this summary of a conversation into one or two plain sentences, "
    "at most 35 words total, stating what the conversation was about. Lead with "
    "the concrete topic and the specifics that make it recognizable. Do not "
    "start with phrases like 'The user asked' or 'This conversation'. No "
    "markdown, no quotes, no preamble, just the sentences."
)

# with high concurrency we may occasionally brush a rate limit; let the SDK
# back off and retry automatically instead of failing the chat. also size the
# underlying httpx connection pool to the worker count — its default cap of 100
# connections would otherwise serialize requests once WORKERS climbs past it.
client = OpenAI(
    max_retries=5,
    http_client=httpx.Client(
        limits=httpx.Limits(
            max_connections=MAX_WORKERS + 10,
            max_keepalive_connections=MAX_WORKERS + 10,
        )
    ),
)


def migrate(conn):
    """Add the summary, short_summary, embedding and error columns if missing."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(chats)")}
    if "summary" not in cols:
        conn.execute("ALTER TABLE chats ADD COLUMN summary TEXT")
    if "short_summary" not in cols:
        conn.execute("ALTER TABLE chats ADD COLUMN short_summary TEXT")
    if "embedding" not in cols:
        conn.execute("ALTER TABLE chats ADD COLUMN embedding TEXT")
    if "error" not in cols:
        conn.execute("ALTER TABLE chats ADD COLUMN error TEXT")
    conn.commit()


def pending_rows(conn, include_errors=False):
    """Rows that still need a summary, short summary or embedding, skipping empty
    chats.

    Returns the existing summary and short_summary alongside the cleaned text so
    process_row only redoes the missing piece: a row that just needs a short
    summary must not pay to re-summarize the whole chat again.

    By default rows that already failed (error set) are skipped so a deterministic
    failure is not retried forever; pass include_errors=True to retry them.
    """
    query = """
        SELECT id, cleaned, summary, short_summary, embedding IS NOT NULL
        FROM chats
        WHERE (summary IS NULL OR short_summary IS NULL OR embedding IS NULL)
          AND TRIM(cleaned) != ''
    """
    if not include_errors:
        query += " AND error IS NULL"
    return conn.execute(query).fetchall()


def _summarize_once(text, prompt):
    resp = client.chat.completions.create(
        model=SUMMARY_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
    )
    return (
        resp.choices[0].message.content.strip(),
        resp.usage.prompt_tokens,
        resp.usage.completion_tokens,
    )


def summarize(text):
    """Summarize a chat, using map-reduce for chats too big for one request.

    Returns (summary, total_input_tokens, total_output_tokens) with the token
    counts summed across every underlying call so cost tracking stays accurate.
    """
    if len(text) <= MAX_INPUT_CHARS:
        return _summarize_once(text, SUMMARY_PROMPT)

    # map: summarize each chunk of the oversized chat
    chunks = [
        text[i : i + MAX_INPUT_CHARS] for i in range(0, len(text), MAX_INPUT_CHARS)
    ]
    in_tok = 0
    out_tok = 0
    partials = []
    for chunk in chunks:
        partial, ci, co = _summarize_once(chunk, CHUNK_PROMPT)
        partials.append(partial)
        in_tok += ci
        out_tok += co

    # reduce: the partial summaries are small, so this always fits in one request
    summary, ci, co = _summarize_once("\n\n".join(partials), COMBINE_PROMPT)
    return summary, in_tok + ci, out_tok + co


def shorten(summary):
    """Condense a long summary into the 1-2 sentence blurb shown in results."""
    return _summarize_once(summary, SHORT_SUMMARY_PROMPT)


def embed(text):
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding, resp.usage


def process_row(row):
    """Fill in whatever a chat is missing (summary, short summary, embedding).

    Each step is skipped when the column is already populated, so backfilling a
    new column costs only that column: re-running this never re-summarizes or
    re-embeds work that is already paid for. Runs inside a worker thread.
    """
    row_id, cleaned, summary, short_summary, has_embedding = row
    summary_in = summary_out = short_in = short_out = embed_in = 0

    if summary is None:
        summary, summary_in, summary_out = summarize(cleaned)
    if short_summary is None:
        short_summary, short_in, short_out = shorten(summary)

    embedding_json = None
    if not has_embedding:
        embedding, embed_usage = embed(summary)
        embedding_json = json.dumps(embedding)
        embed_in = embed_usage.prompt_tokens

    return {
        "id": row_id,
        "summary": summary,
        "short_summary": short_summary,
        "embedding_json": embedding_json,
        "summary_in": summary_in,
        "summary_out": summary_out,
        "short_in": short_in,
        "short_out": short_out,
        "embed_in": embed_in,
    }


def save(conn, result):
    """Persist a processed row. The embedding is only written when this run
    computed one, so a short-summary-only pass leaves the existing vector alone."""
    now = datetime.now(timezone.utc).isoformat()
    if result["embedding_json"] is not None:
        conn.execute(
            "UPDATE chats SET summary = ?, short_summary = ?, embedding = ?, "
            "error = NULL, updated_at = ? WHERE id = ?",
            (
                result["summary"],
                result["short_summary"],
                result["embedding_json"],
                now,
                result["id"],
            ),
        )
    else:
        conn.execute(
            "UPDATE chats SET summary = ?, short_summary = ?, error = NULL, "
            "updated_at = ? WHERE id = ?",
            (result["summary"], result["short_summary"], now, result["id"]),
        )


def mark_error(conn, row_id, exc):
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE chats SET error = ?, updated_at = ? WHERE id = ?",
        (str(exc), now, row_id),
    )


def fmt_duration(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def print_summary(done, errors, runtime, tok):
    cost = (
        estimate_cost(SUMMARY_MODEL, tok["summary_in"], tok["summary_out"])
        + estimate_cost(SUMMARY_MODEL, tok["short_in"], tok["short_out"])
        + estimate_cost(EMBEDDING_MODEL, tok["embed_in"])
    )
    total = sum(tok.values())

    print("=" * 48)
    print(f"Processed {done} chats ({errors} errors) in {fmt_duration(runtime)}")
    print("-" * 48)
    print(f"  summary input tokens : {tok['summary_in']:>12,}")
    print(f"  summary output tokens: {tok['summary_out']:>12,}")
    print(f"  short input tokens   : {tok['short_in']:>12,}")
    print(f"  short output tokens  : {tok['short_out']:>12,}")
    print(f"  embedding tokens     : {tok['embed_in']:>12,}")
    print(f"  total tokens         : {total:>12,}")
    print(f"  estimated cost       : ${cost:>11.4f}")
    print("=" * 48)


if __name__ == "__main__":
    start_time = time.time()

    conn = get_connection()
    migrate(conn)

    rows = pending_rows(conn, include_errors=RETRY_ERRORS)
    total = len(rows)
    mode = " (retrying previously failed)" if RETRY_ERRORS else ""
    print(f"{total} chats to process with {MAX_WORKERS} workers{mode}")

    if total == 0:
        conn.close()
        print("Nothing to do.")
        raise SystemExit

    done = 0
    errors = 0
    tok = {
        "summary_in": 0,
        "summary_out": 0,
        "short_in": 0,
        "short_out": 0,
        "embed_in": 0,
    }
    interrupted = False

    # not using a `with` block on purpose: its __exit__ blocks on shutdown(wait=True)
    # which drains every queued future, making Ctrl-C hang. we shut down manually.
    pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    futures = {pool.submit(process_row, row): row[0] for row in rows}
    try:
        for future in as_completed(futures):
            row_id = futures[future]
            try:
                result = future.result()
                save(conn, result)
                done += 1
                for key in tok:
                    tok[key] += result[key]
            except Exception as exc:
                errors += 1
                mark_error(conn, row_id, exc)
                print(f"error on chat id {row_id}: {exc}")
                continue

            if done % COMMIT_EVERY == 0:
                conn.commit()

            if done % PRINT_EVERY == 0 or done == total:
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                remaining = total - done
                eta = remaining / rate if rate > 0 else 0
                pct = done / total * 100
                print(
                    f"{done}/{total} ({pct:5.1f}%) | "
                    f"{rate:4.1f} chats/s | "
                    f"elapsed {fmt_duration(elapsed)} | "
                    f"ETA {fmt_duration(eta)}"
                )
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted — cancelling pending work and saving progress...")
    finally:
        # cancel queued-but-not-started work and don't wait on in-flight calls,
        # so we exit fast. already-saved rows are flushed by the commit below;
        # any in-flight chats are simply re-done on the next run (resumable).
        pool.shutdown(wait=False, cancel_futures=True)
        conn.commit()
        conn.close()

    runtime = time.time() - start_time
    print_summary(done, errors, runtime, tok)

    if interrupted:
        print("Stopped early. Re-run to finish the remaining chats.")
