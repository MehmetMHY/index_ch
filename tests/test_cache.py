"""Tests for retrieve/cache.py: ensure_fts, fts_search, load_vectors."""

import json
import os
import numpy as np
import pytest

import retrieve.cache as cache_mod
from retrieve.cache import ensure_fts, fts_search, load_vectors, _embedding_signature


class TestEnsureFts:
    def test_creates_fts_table(self, db_conn):
        ensure_fts(db_conn)
        # fts table should exist
        tables = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {t[0] for t in tables}
        assert "chats_fts" in names
        assert "fts_state" in names

    def test_signature_match_no_rebuild(self, db_conn):
        ensure_fts(db_conn)
        # call again with no data change -> should not rebuild
        ensure_fts(db_conn)

    def test_signature_mismatch_triggers_rebuild(self, db_conn):
        ensure_fts(db_conn)
        # insert a row, changing the signature
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, "
            "created_at, updated_at) VALUES ('/x.json', '{}', 't', 1, 'n', 'n2')"
        )
        db_conn.commit()
        ensure_fts(db_conn)  # should rebuild without error


class TestFtsSearch:
    def test_empty_query(self, db_conn):
        ensure_fts(db_conn)
        assert fts_search(db_conn, "", 10) == []

    def test_no_tokens(self, db_conn):
        ensure_fts(db_conn)
        assert fts_search(db_conn, "!!!", 10) == []

    def test_search_with_results(self, db_conn):
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, summary, "
            "created_at, updated_at) "
            "VALUES ('/a.json', '{}', 'python testing', 5, 'about python tests', 'n', 'n')"
        )
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, summary, "
            "created_at, updated_at) "
            "VALUES ('/b.json', '{}', 'cooking pasta', 5, 'making dinner', 'n', 'n')"
        )
        db_conn.commit()
        ensure_fts(db_conn)
        results = fts_search(db_conn, "python", 10)
        assert len(results) >= 1
        assert 1 in results  # the python chat

    def test_allowed_filtering(self, db_conn):
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, summary, "
            "created_at, updated_at) "
            "VALUES ('/a.json', '{}', 'python testing', 5, 'python', 'n', 'n')"
        )
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, summary, "
            "created_at, updated_at) "
            "VALUES ('/b.json', '{}', 'python coding', 5, 'python', 'n', 'n')"
        )
        db_conn.commit()
        ensure_fts(db_conn)
        results = fts_search(db_conn, "python", 10, allowed={1})
        assert all(r in {1} for r in results)

    def test_wider_window_when_allowed(self, db_conn):
        # just verify max(n*20, 500) logic doesn't crash
        db_conn.execute(
            "INSERT INTO chats (file_path, raw, cleaned, token_estimate, summary, "
            "created_at, updated_at) "
            "VALUES ('/a.json', '{}', 'test data', 5, 'test', 'n', 'n')"
        )
        db_conn.commit()
        ensure_fts(db_conn)
        results = fts_search(db_conn, "test", 5, allowed={1})
        assert isinstance(results, list)


class TestLoadVectors:
    def test_meta_construction(self, db_with_chats, tmp_path, monkeypatch):
        monkeypatch.setattr(
            cache_mod, "EMBEDDINGS_CACHE_PATH", str(tmp_path / "cache.npz")
        )
        ids, mat, meta = load_vectors(db_with_chats)
        assert 1 in meta
        assert meta[1]["file_path"] == "/tmp/ch_session_1700000000.json"
        assert isinstance(meta[1]["archived"], bool)

    def test_cache_hit(self, db_with_chats, tmp_path, monkeypatch):
        cache_path = str(tmp_path / "cache.npz")
        monkeypatch.setattr(cache_mod, "EMBEDDINGS_CACHE_PATH", cache_path)
        # first call builds and saves cache
        ids1, mat1, meta1 = load_vectors(db_with_chats)
        assert os.path.exists(cache_path)
        # second call loads from cache
        ids2, mat2, meta2 = load_vectors(db_with_chats)
        np.testing.assert_array_equal(ids1, ids2)
        np.testing.assert_allclose(mat1, mat2)

    def test_cache_corrupt_rebuilds(self, db_with_chats, tmp_path, monkeypatch):
        cache_path = str(tmp_path / "cache.npz")
        # write garbage to cache file
        with open(cache_path, "wb") as f:
            f.write(b"not a npz file")
        monkeypatch.setattr(cache_mod, "EMBEDDINGS_CACHE_PATH", cache_path)
        ids, mat, meta = load_vectors(db_with_chats)
        assert len(ids) == 3  # 3 chats in db_with_chats

    def test_normalization(self, db_with_chats, tmp_path, monkeypatch):
        monkeypatch.setattr(
            cache_mod, "EMBEDDINGS_CACHE_PATH", str(tmp_path / "cache.npz")
        )
        ids, mat, meta = load_vectors(db_with_chats)
        norms = np.linalg.norm(mat, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_archived_bool_cast(self, db_with_chats, tmp_path, monkeypatch):
        monkeypatch.setattr(
            cache_mod, "EMBEDDINGS_CACHE_PATH", str(tmp_path / "cache.npz")
        )
        ids, mat, meta = load_vectors(db_with_chats)
        assert meta[3]["archived"] is True
        assert meta[1]["archived"] is False
