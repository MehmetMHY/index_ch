import json
import os
import re

import numpy as np

from config import EMBEDDINGS_CACHE_PATH
from build import get_connection  # noqa: F401 (re-exported for cli.py convenience)
from . import color


def ensure_fts(conn):
    """Create the FTS5 index if missing, and rebuild it when chats have changed.

    The change signature combines row count with the latest updated_at, so both
    newly ingested chats (build.py) and freshly filled-in summaries (process.py,
    which bumps updated_at) trigger a rebuild.
    """
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chats_fts "
        "USING fts5(summary, cleaned, content='chats', content_rowid='id')"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS fts_state "
        "(id INTEGER PRIMARY KEY CHECK (id = 1), signature TEXT)"
    )
    count, latest = conn.execute(
        "SELECT count(*), COALESCE(MAX(updated_at), '') FROM chats"
    ).fetchone()
    signature = f"{count}:{latest}"
    stored = conn.execute("SELECT signature FROM fts_state WHERE id = 1").fetchone()
    if stored is None or stored[0] != signature:
        print(color.yellow("Updating search index..."), flush=True)
        conn.execute("INSERT INTO chats_fts(chats_fts) VALUES('rebuild')")
        conn.execute(
            "INSERT INTO fts_state(id, signature) VALUES(1, ?) "
            "ON CONFLICT(id) DO UPDATE SET signature = excluded.signature",
            (signature,),
        )
        conn.commit()


def fts_search(conn, query, n, allowed=None):
    """Return up to n chat ids ranked by BM25 keyword relevance.

    When `allowed` (a set of in-range ids) is given, fetch a wider window and
    keep only matches inside it, so a time filter does not silently starve the
    keyword leg when its best global hits fall outside the range.
    """
    tokens = re.findall(r"\w+", query.lower())
    if not tokens:
        return []
    match = " OR ".join(f'"{t}"' for t in tokens)
    limit = n if allowed is None else max(n * 20, 500)
    rows = conn.execute(
        "SELECT rowid FROM chats_fts WHERE chats_fts MATCH ? "
        "ORDER BY bm25(chats_fts) LIMIT ?",
        (match, limit),
    ).fetchall()
    if allowed is not None:
        return [r[0] for r in rows if r[0] in allowed][:n]
    return [r[0] for r in rows]


def _embedding_signature(conn):
    count, latest = conn.execute(
        "SELECT count(*), COALESCE(MAX(updated_at), '') FROM chats "
        "WHERE embedding IS NOT NULL"
    ).fetchone()
    return f"{count}:{latest}"


def load_vectors(conn):
    """Return (ids, normalized matrix, metadata dict).

    Parsing 5.6k json embedding arrays takes ~3s, so the normalized matrix is
    cached to a .npz keyed by a signature (embedding count + latest updated_at).
    Later launches load it in well under a second; the cache rebuilds itself
    whenever process.py adds or changes embeddings. Metadata (file_path, summary)
    is always read fresh from the db - that part is cheap, no float parsing.
    """
    rows = conn.execute(
        "SELECT id, file_path, summary, short_summary, last_message_epoch, "
        "archived, json_array_length(json_extract(raw, '$.messages')) "
        "FROM chats WHERE embedding IS NOT NULL"
    ).fetchall()
    meta = {
        r[0]: {
            "file_path": r[1],
            "summary": r[2],
            "short_summary": r[3],
            "last_message_epoch": r[4],
            "archived": bool(r[5]),
            "message_count": r[6] if r[6] is not None else 0,
        }
        for r in rows
    }
    signature = _embedding_signature(conn)

    if os.path.exists(EMBEDDINGS_CACHE_PATH):
        try:
            cached = np.load(EMBEDDINGS_CACHE_PATH, allow_pickle=False)
            if cached["signature"].item() == signature:
                return cached["ids"], cached["mat"], meta
        except Exception:
            pass  # corrupt or old cache -> fall through and rebuild

    emb = conn.execute(
        "SELECT id, embedding FROM chats WHERE embedding IS NOT NULL"
    ).fetchall()
    ids = np.array([r[0] for r in emb])
    mat = np.array([json.loads(r[1]) for r in emb], dtype=np.float32)
    # normalize so a dot product is cosine similarity
    mat /= np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12, None)
    try:
        np.savez(EMBEDDINGS_CACHE_PATH, ids=ids, mat=mat, signature=np.array(signature))
    except Exception:
        pass  # caching is best-effort; retrieval still works without it
    return ids, mat, meta
