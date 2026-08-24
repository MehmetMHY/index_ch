import os
import time

import numpy as np
import httpx
from openai import OpenAI

from config import (
    EMBEDDING_MODEL,
    GROQ_BASE_URL,
    RERANK_MODEL,
    RERANK_EFFORT,
    QUERY_EXPANSION_MODEL,
    QUERY_EXPANSION_EFFORT,
    NUM_EXPANSIONS,
    TIME_RANGES,
    POOL,
    RERANK_POOL,
    TOP_K,
    RRF_K,
)

from .models import RerankResult, ExpandedQueries, RERANK_SYSTEM, QUERY_EXPANSION_SYSTEM
from .display import chat_epoch
from .state import Session
from .cache import fts_search

# OpenAI for embeddings (the stored vectors are text-embedding-3-small, so the
# query must embed in the same space). Groq for the two retrieval LLM steps
# (rerank, query expansion) via its OpenAI-compatible endpoint, for speed.
client = OpenAI(max_retries=3, http_client=httpx.Client(timeout=30))
groq_client = OpenAI(
    base_url=GROQ_BASE_URL,
    api_key=os.environ.get("GROQ_API_KEY"),
    max_retries=3,
    http_client=httpx.Client(timeout=30),
)


def warm_connections():
    """Best-effort: open the HTTPS connections to OpenAI and Groq up front so the
    first real query does not pay the TLS/handshake cold-start (~1s). Runs in a
    background thread at startup; any failure is ignored (the query path retries).
    """
    try:
        client.embeddings.create(model=EMBEDDING_MODEL, input="warmup")
    except Exception:
        pass
    try:
        groq_client.chat.completions.create(
            model=RERANK_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
    except Exception:
        pass


def embed_queries(queries):
    """Embed one or more query strings in a single request.

    Batching keeps query expansion from adding embedding round trips: the
    original query and every variant are embedded together in one call.
    Returns (list of normalized vectors in input order, input tokens used).
    """
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=queries)
    # the API may return items out of order; index restores the input order
    data = sorted(resp.data, key=lambda d: d.index)
    vecs = []
    for d in data:
        v = np.array(d.embedding, dtype=np.float32)
        v = v / np.clip(np.linalg.norm(v), 1e-12, None)
        vecs.append(v)
    return vecs, resp.usage.prompt_tokens


def vector_search(ids, mat, query_vec, n):
    """Return up to n chat ids ranked by cosine similarity."""
    sims = mat @ query_vec
    top = np.argsort(-sims)[:n]
    return [int(ids[i]) for i in top]


def rrf_fuse(rankings, k=RRF_K):
    """Merge several ranked id lists via Reciprocal Rank Fusion."""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: -scores[d])


def expand_query(query, n=NUM_EXPANSIONS):
    """Rewrite the query into n alternative phrasings to widen recall.

    Returns (variants, in_tokens, out_tokens). Degrades to ([], 0, 0) on any
    failure, so search simply falls back to the original query alone. Variants
    equal to the original (case-insensitively) or to each other are dropped.
    """
    if n <= 0:
        return [], 0, 0
    try:
        resp = groq_client.chat.completions.parse(
            model=QUERY_EXPANSION_MODEL,
            reasoning_effort=QUERY_EXPANSION_EFFORT,
            messages=[
                {"role": "system", "content": QUERY_EXPANSION_SYSTEM.format(n=n)},
                {"role": "user", "content": query},
            ],
            response_format=ExpandedQueries,
        )
        raw = resp.choices[0].message.parsed.queries
    except Exception as exc:
        print(f"(query expansion failed, using original query only: {exc})")
        return [], 0, 0

    seen = {query.strip().lower()}
    variants = []
    for q in raw:
        q = (q or "").strip()
        key = q.lower()
        if q and key not in seen:
            variants.append(q)
            seen.add(key)
    return variants[:n], resp.usage.prompt_tokens, resp.usage.completion_tokens


def rerank(query, candidate_ids, meta):
    """Listwise LLM rerank. Returns ([(id, grade)], in_tokens, out_tokens).

    Degrades to hybrid order (and 0 tokens) if the call fails.
    """
    docs = [
        {"id": str(cid), "text": (meta[cid]["summary"] or "")} for cid in candidate_ids
    ]
    formatted = "\n\n".join(
        f"<document id={d['id']!r}>\n{d['text']}\n</document>" for d in docs
    )
    try:
        resp = groq_client.chat.completions.parse(
            model=RERANK_MODEL,
            reasoning_effort=RERANK_EFFORT,
            messages=[
                {"role": "system", "content": RERANK_SYSTEM},
                {
                    "role": "user",
                    "content": f"<query>\n{query}\n</query>\n\n"
                    f"<documents>\n{formatted}\n</documents>",
                },
            ],
            response_format=RerankResult,
        )
        parsed = resp.choices[0].message.parsed
    except Exception as exc:
        print(f"(rerank failed, falling back to hybrid order: {exc})")
        return [(cid, None) for cid in candidate_ids], 0, 0

    # validate: keep only real, in-set ids, each once, in the model's order
    candidate_set = set(candidate_ids)
    ordered, seen = [], set()
    for rd in parsed.ranked_documents:
        try:
            did = int(rd.document_id)
        except (TypeError, ValueError):
            continue
        if did in candidate_set and did not in seen:
            ordered.append((did, rd.relevance))
            seen.add(did)
    # append any candidates the model dropped, in their original hybrid order
    for cid in candidate_ids:
        if cid not in seen:
            ordered.append((cid, None))
    return ordered, resp.usage.prompt_tokens, resp.usage.completion_tokens


def range_bounds(time_filter):
    """Return (lo_epoch, hi_epoch) for the active filter; hi None means no upper
    bound. A TIME_RANGES key is a rolling window ending now (computed live per
    query); a (start, end) tuple is an absolute custom range from the picker."""
    if isinstance(time_filter, tuple):
        return time_filter
    return time.time() - TIME_RANGES[time_filter], None


def allowed_in_range(ids, meta, time_filter, show_archived):
    """Return (filtered_id_array, filtered_row_mask, allowed_set) for the active
    filters, or (ids, None, None) when no filter is active.

    Archived chats (source file gone from disk) are excluded unless
    show_archived is True. When a time filter is also on, chats whose filename
    has no parseable epoch are excluded. The mask aligns with the embeddings
    matrix rows so the vector leg can be restricted to in-range chats; the set
    filters the keyword leg. Archived rows stay in the index and embeddings
    cache at all times, so toggling show_archived is instant (no rebuild).
    """
    if time_filter is None and show_archived:
        return ids, None, None
    lo, hi = range_bounds(time_filter) if time_filter else (None, None)
    mask = np.fromiter(
        (
            (show_archived or not meta[int(i)].get("archived"))
            and (
                lo is None
                or (e := chat_epoch(meta[int(i)])) is not None
                and e >= lo
                and (hi is None or e <= hi)
            )
            for i in ids
        ),
        dtype=bool,
        count=len(ids),
    )
    f_ids = ids[mask]
    return f_ids, mask, set(int(i) for i in f_ids)


def search(session: Session, query: str):
    """Run a hybrid vector + keyword search with optional query expansion and
    LLM reranking. Returns (ranked_results, usage_dict)."""
    do_expand = session.do_expand
    do_rerank = session.do_rerank
    time_filter = session.time_filter
    top_k = session.result_len
    show_archived = session.show_archived

    # optionally rewrite the query into a few variants, then embed the original
    # plus all variants in one batched call (expansion adds an LLM call but no
    # extra embedding round trips). vector + keyword search each query and fuse
    # every ranking; the reranker still judges against the user's original query.
    expand_in = expand_out = 0
    if do_expand:
        variants, expand_in, expand_out = expand_query(query)
    else:
        variants = []

    queries = [query] + variants
    vecs, embed_in = embed_queries(queries)

    # restrict both retrieval legs to the active time window (if any) and to
    # non-archived chats (unless toggled on) at the search level, so narrow
    # ranges and archived-hiding still surface their best in-range matches
    f_ids, mask, allowed = allowed_in_range(
        session.ids, session.meta, time_filter, show_archived
    )
    f_mat = session.mat if mask is None else session.mat[mask]

    rankings = []
    for text, vec in zip(queries, vecs):
        rankings.append(vector_search(f_ids, f_mat, vec, POOL))
        rankings.append(fts_search(session.conn, text, POOL, allowed))
    fused = rrf_fuse(rankings)

    rerank_in = rerank_out = 0
    if do_rerank and fused:
        # rerank at least as many candidates as the caller wants back, so a
        # /len above the default RERANK_POOL still returns graded results
        rerank_n = max(RERANK_POOL, top_k)
        ranked, rerank_in, rerank_out = rerank(query, fused[:rerank_n], session.meta)
    else:
        ranked = [(cid, None) for cid in fused]

    # tiebreak: within the same rerank grade, show the most recent chat first.
    # grade stays primary (a lower grade never outranks a higher one); ungraded
    # results (fast mode / candidates the reranker dropped) keep their hybrid
    # order. the sort is stable, so equal keys preserve their existing order.
    ranked.sort(
        key=lambda item: (
            (1, 0, 0)
            if item[1] is None
            else (0, -item[1], -(chat_epoch(session.meta[item[0]]) or 0))
        )
    )

    usage = {
        "embed_in": embed_in,
        "expand_in": expand_in,
        "expand_out": expand_out,
        "rerank_in": rerank_in,
        "rerank_out": rerank_out,
    }
    return ranked[:top_k], usage
