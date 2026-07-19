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
    """Add the summary, embedding and error columns if they are not there yet."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(chats)")}
    if "summary" not in cols:
        conn.execute("ALTER TABLE chats ADD COLUMN summary TEXT")
    if "embedding" not in cols:
        conn.execute("ALTER TABLE chats ADD COLUMN embedding TEXT")
    if "error" not in cols:
        conn.execute("ALTER TABLE chats ADD COLUMN error TEXT")
    conn.commit()


def pending_rows(conn, include_errors=False):
    """Rows that still need a summary or embedding, skipping empty chats.

    By default rows that already failed (error set) are skipped so a deterministic
    failure is not retried forever; pass include_errors=True to retry them.
    """
    query = """
        SELECT id, cleaned FROM chats
        WHERE (summary IS NULL OR embedding IS NULL)
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


def embed(text):
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding, resp.usage


def process_row(row):
    """Summarize then embed a single chat. Runs inside a worker thread."""
    row_id, cleaned = row
    summary, summary_in, summary_out = summarize(cleaned)
    embedding, embed_usage = embed(summary)
    return {
        "id": row_id,
        "summary": summary,
        "embedding_json": json.dumps(embedding),
        "summary_in": summary_in,
        "summary_out": summary_out,
        "embed_in": embed_usage.prompt_tokens,
    }


def save(conn, result):
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE chats SET summary = ?, embedding = ?, error = NULL, updated_at = ? "
        "WHERE id = ?",
        (result["summary"], result["embedding_json"], now, result["id"]),
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


def print_summary(done, errors, runtime, tok_summary_in, tok_summary_out, tok_embed_in):
    cost = estimate_cost(
        SUMMARY_MODEL, tok_summary_in, tok_summary_out
    ) + estimate_cost(EMBEDDING_MODEL, tok_embed_in)

    print("=" * 48)
    print(f"Processed {done} chats ({errors} errors) in {fmt_duration(runtime)}")
    print("-" * 48)
    print(f"  summary input tokens : {tok_summary_in:>12,}")
    print(f"  summary output tokens: {tok_summary_out:>12,}")
    print(f"  embedding tokens     : {tok_embed_in:>12,}")
    print(
        f"  total tokens         : "
        f"{tok_summary_in + tok_summary_out + tok_embed_in:>12,}"
    )
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
    tok_summary_in = 0
    tok_summary_out = 0
    tok_embed_in = 0
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
                tok_summary_in += result["summary_in"]
                tok_summary_out += result["summary_out"]
                tok_embed_in += result["embed_in"]
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
    print_summary(done, errors, runtime, tok_summary_in, tok_summary_out, tok_embed_in)

    if interrupted:
        print("Stopped early. Re-run to finish the remaining chats.")
